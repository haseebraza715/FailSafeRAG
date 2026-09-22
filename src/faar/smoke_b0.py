"""One-document B0 end-to-end smoke (Phase 0 checkpoint).

Uses already-prepared OHR smoke PDF/OCR/image assets only. Flows through
BenchmarkRepository → retrieval → quality gate → diagnosis/recovery routing
decision → answer generation → result serialization.

This path is explicitly marked as smoke and is never a paper result.
Default retrieval is a local lexical mock so the smoke stays reproducible
without Hugging Face downloads or paid VLM calls. Pin metadata still records
the locked stack (NV-Embed-v2, bge-reranker-v2-m3, gpt-4o-2024-11-20, …).
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api_logging import vlm_cost_rates
from .benchmarks import BenchmarkRepository, load_benchmark_repository_from_manifest
from .data import DatasetUnavailableError
from .dataset_paths import SPLIT_RELATIVE_PATH
from .experiment_profiles import apply_profile
from .metrics import exact_match, token_f1
from .results_aggregator import summarize_examples
from .settings import AppSettings
from .types import Chunk, RetrievalHit

DEFAULT_SMOKE_DOC = "academic/DUDE_157911e3080d18f4d799a122aaeb33fb"
DEFAULT_SMOKE_ROOT = Path("data/benchmark_prep/smoke")
DEFAULT_SMOKE_SPLIT = "val"
REQUIRED_SUMMARY_FIELDS = ("EM", "F1", "vlm_rate", "harm_rate", "cost_usd", "runtime_sec")
B0_PROFILE = "naive_rag"
SMOKE_LABEL = "B0 one-document smoke (not a paper result)"


class LexicalSmokeRetriever:
    """Deterministic BM25-free lexical retriever for smoke runs only.

    Scores chunks by query-term overlap so the pipeline uses real OCR text
    without loading NV-Embed-v2 or bge-reranker-v2-m3.
    """

    def __init__(self, chunks: list[Chunk], settings: Any = None, *, cache_dir: Any = None) -> None:
        self.chunks = list(chunks)
        self.settings = settings

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = top_k or getattr(self.settings, "top_k", 5) or 5
        query_terms = set(re.findall(r"[a-z0-9%$]+", query.lower()))
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            chunk_terms = set(re.findall(r"[a-z0-9%$]+", chunk.text.lower()))
            overlap = len(query_terms & chunk_terms)
            score = float(overlap) + (0.05 if chunk.text.strip() else 0.0)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for rank, (score, chunk) in enumerate(scored[: max(1, min(k, len(scored)))]):
            rerank = min(0.99, 0.55 + score / 20.0) if score > 0 else 0.15
            if rank > 0:
                rerank = max(0.05, rerank - 0.1 * rank)
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    bm25_score=float(score),
                    dense_score=float(score),
                    fused_score=float(score),
                    reranker_score=round(rerank, 6),
                )
            )
        return hits


def smoke_root_path(project_root: Path, smoke_root: Path | str | None = None) -> Path:
    root = project_root.expanduser().resolve()
    if smoke_root is None:
        return (root / DEFAULT_SMOKE_ROOT).resolve()
    path = Path(smoke_root)
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve()


def _page_asset_paths(smoke_root: Path, doc_name: str, page_id: int) -> tuple[Path, Path]:
    ocr = smoke_root / "ocr" / f"{doc_name}_page_{page_id}.txt"
    image = smoke_root / "images" / f"{doc_name}_page_{page_id}.png"
    return ocr, image


def _inventory_page_ids(smoke_root: Path, doc_name: str) -> list[int]:
    prepare_result = smoke_root / f"prepare_result_{doc_name.replace('/', '__')}.json"
    if prepare_result.is_file():
        payload = json.loads(prepare_result.read_text(encoding="utf-8"))
        page_ids = payload.get("page_ids")
        if isinstance(page_ids, list) and page_ids:
            return sorted({int(page_id) for page_id in page_ids})
    provenance = smoke_root / "provenance" / f"{doc_name}.json"
    if provenance.is_file():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        page_ids = payload.get("page_ids")
        if isinstance(page_ids, list) and page_ids:
            return sorted({int(page_id) for page_id in page_ids})
    ocr_dir = smoke_root / "ocr"
    found: set[int] = set()
    if ocr_dir.is_dir():
        for path in ocr_dir.rglob("*.txt"):
            match = re.match(rf"{re.escape(doc_name)}_page_(\d+)$", path.relative_to(ocr_dir).with_suffix("").as_posix())
            if match:
                found.add(int(match.group(1)))
    if not found:
        raise DatasetUnavailableError(
            f"No prepared page inventory found under {smoke_root} for smoke doc {doc_name!r}."
        )
    return sorted(found)


def build_ohr_one_doc_smoke_manifest(
    project_root: Path,
    *,
    doc_name: str = DEFAULT_SMOKE_DOC,
    split: str = DEFAULT_SMOKE_SPLIT,
    smoke_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build a one-document manifest from prepared smoke assets and locked split/qas.

    Does not fabricate QA rows or page counts. Reads real config/datasets/ohr_split.json + qas_v2.json
    and requires prepared OCR/image files for every inventoried page of the smoke doc.
    """
    project_root = project_root.expanduser().resolve()
    smoke_root = smoke_root_path(project_root, smoke_root)
    split_path = project_root / SPLIT_RELATIVE_PATH
    qas_path = project_root / "OHR-Bench/data/qas_v2.json"
    if not split_path.is_file():
        raise DatasetUnavailableError(f"Missing immutable split file: {split_path}")
    if not qas_path.is_file():
        raise DatasetUnavailableError(f"Missing OHR QA source: {qas_path}")

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if split not in split_payload.get("splits", {}):
        raise ValueError(f"Unknown OHR-Bench split {split!r}.")
    selected_ids = set(split_payload["splits"][split])
    source_rows = json.loads(qas_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in source_rows:
        example_id = str(row["ID"])
        if example_id not in selected_ids:
            continue
        if str(row.get("doc_name", "")) != doc_name:
            continue
        evidence_pages = [int(page) for page in _listify(row.get("evidence_page_no"))]
        records.append(
            {
                "example_id": example_id,
                "doc_name": doc_name,
                "question": row["questions"],
                "correct_answer": row["answers"],
                "page_ids": evidence_pages,
                "corpus_ids": [],
                "metadata": {
                    "doc_type": row.get("doc_type", ""),
                    "evidence_source": row.get("evidence_source", ""),
                    "evidence_page_ids": evidence_pages,
                    "smoke": True,
                    "smoke_doc": doc_name,
                },
            }
        )
    if not records:
        raise DatasetUnavailableError(
            f"No {split} examples in config/datasets/ohr_split.json/qas_v2.json for smoke doc {doc_name!r}."
        )

    page_ids = _inventory_page_ids(smoke_root, doc_name)
    inventory = {doc_name: page_ids}
    corpus_pages: list[dict[str, Any]] = []
    missing: list[str] = []
    for page_id in page_ids:
        ocr_path, image_path = _page_asset_paths(smoke_root, doc_name, page_id)
        if not ocr_path.is_file():
            missing.append(f"{doc_name}:p{page_id}: OCR text ({ocr_path})")
        if not image_path.is_file():
            missing.append(f"{doc_name}:p{page_id}: page image ({image_path})")
        corpus_pages.append(
            {
                "corpus_id": f"{doc_name}:p{page_id}",
                "doc_name": doc_name,
                "page_id": page_id,
                "ocr_text_path": ocr_path.relative_to(project_root).as_posix(),
                "image_path": image_path.relative_to(project_root).as_posix(),
            }
        )
    if missing:
        preview = "\n".join(missing[:10])
        raise DatasetUnavailableError(
            "Prepared one-document smoke assets are incomplete.\n"
            f"Missing (first 10):\n{preview}"
        )

    for record in records:
        evidence = {int(page_id) for page_id in record["page_ids"]}
        if not evidence.issubset(set(page_ids)):
            raise DatasetUnavailableError(
                f"Evidence pages for {record['example_id']} are outside the smoke document inventory."
            )
        record["corpus_ids"] = [f"{doc_name}:p{page_id}" for page_id in page_ids]

    return {
        "dataset": "ohrbench",
        "split": split,
        "smoke": True,
        "paper_result": False,
        "smoke_doc": doc_name,
        "records": records,
        "corpus_pages": corpus_pages,
        "document_inventory": inventory,
    }


def write_smoke_manifest(
    project_root: Path,
    *,
    doc_name: str = DEFAULT_SMOKE_DOC,
    split: str = DEFAULT_SMOKE_SPLIT,
    smoke_root: Path | str | None = None,
    destination: Path | None = None,
) -> Path:
    project_root = project_root.expanduser().resolve()
    smoke_root = smoke_root_path(project_root, smoke_root)
    payload = build_ohr_one_doc_smoke_manifest(
        project_root, doc_name=doc_name, split=split, smoke_root=smoke_root
    )
    dest = destination or (smoke_root / "b0_one_doc_manifest.json")
    if not dest.is_absolute():
        dest = project_root / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def load_one_doc_smoke_repository(
    project_root: Path,
    *,
    doc_name: str = DEFAULT_SMOKE_DOC,
    split: str = DEFAULT_SMOKE_SPLIT,
    smoke_root: Path | str | None = None,
    manifest_path: Path | None = None,
) -> tuple[BenchmarkRepository, Path]:
    project_root = project_root.expanduser().resolve()
    if manifest_path is None:
        manifest_path = write_smoke_manifest(
            project_root, doc_name=doc_name, split=split, smoke_root=smoke_root
        )
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        if not manifest_path.is_file():
            raise DatasetUnavailableError(f"Smoke manifest not found: {manifest_path}")
    repo = load_benchmark_repository_from_manifest(
        project_root,
        manifest_path,
        dataset="ohrbench",
        split=split,
    )
    return repo, manifest_path


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _empty_usage() -> dict[str, int | float]:
    return {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def _read_vlm_usage(path: Path) -> dict[str, int | float]:
    if not path.exists():
        return _empty_usage()
    usage = _empty_usage()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "started":
            usage["api_requests"] = int(usage["api_requests"]) + 1
        usage["prompt_tokens"] = int(usage["prompt_tokens"]) + int(record.get("prompt_tokens", 0))
        usage["completion_tokens"] = int(usage["completion_tokens"]) + int(record.get("completion_tokens", 0))
        usage["cost_usd"] = float(usage["cost_usd"]) + float(record.get("cost_usd", 0.0))
    usage["cost_usd"] = round(float(usage["cost_usd"]), 6)
    return usage


def probe_diagnosis_recovery_routing(
    settings: AppSettings,
    repo: BenchmarkRepository,
    *,
    example_id: str,
) -> dict[str, Any]:
    """Exercise diagnosis/recovery routing on the same prepared assets without VLM calls.

    Forces a failing gate and keeps VLM disabled so structural recovery cannot
    spend money. Expects a non-pass failure_type and a recovery policy action.
    Not a paper result.
    """
    from .graph import build_graph

    probe_settings = settings.model_copy(deep=True)
    probe_settings = apply_profile(probe_settings, "faar_full")
    probe_settings.experiment.force_direct_answer = False
    probe_settings.experiment.force_recovery = False
    probe_settings.experiment.force_vlm = False
    probe_settings.experiment.disable_vlm = True
    probe_settings.recovery.enable_vlm = False
    # Strict threshold so the top lexical mock score (capped at 0.99) fails the gate.
    probe_settings.gate.quality_threshold = 1.0
    graph = build_graph(probe_settings, repo=repo)
    result = graph.invoke({"example_id": example_id})
    return {
        "example_id": example_id,
        "gate": result.get("gate", {}),
        "failure_type": result.get("failure_type"),
        "policy_action": result.get("policy_action"),
        "action_outcome": result.get("action_outcome", {}),
        "answer_present": bool(str(result.get("answer", "")).strip()),
        "paper_result": False,
        "smoke_probe": True,
    }


def run_one_document_b0_smoke(
    settings: AppSettings,
    *,
    out_path: Path,
    doc_name: str = DEFAULT_SMOKE_DOC,
    split: str = DEFAULT_SMOKE_SPLIT,
    smoke_root: Path | str | None = None,
    manifest_path: Path | None = None,
    mock_retrieval: bool = True,
    probe_routing: bool = True,
) -> dict[str, Any]:
    """Run clearly marked one-document B0 smoke and serialize required metrics."""
    from .experiment_runner import run_profile

    project_root = settings.project_root.expanduser().resolve()
    settings = apply_profile(settings, B0_PROFILE)
    # B0 never uses a paid VLM; keep API surface closed for the smoke.
    settings.recovery.enable_vlm = False
    settings.recovery.api_enabled = False

    repo, used_manifest = load_one_doc_smoke_repository(
        project_root,
        doc_name=doc_name,
        split=split,
        smoke_root=smoke_root,
        manifest_path=manifest_path,
    )
    example_ids = repo.list_example_ids()
    if not example_ids:
        raise DatasetUnavailableError("Smoke repository has no examples.")

    chunks = repo.get_corpus_chunks(settings.retrieval)
    if not chunks:
        raise DatasetUnavailableError("Smoke repository produced no corpus chunks.")
    ocr_bytes = sum(len(chunk.text) for chunk in chunks)
    if ocr_bytes == 0:
        raise DatasetUnavailableError("Smoke OCR text is empty; prepared assets look invalid.")

    if mock_retrieval:
        import faar.graph as graph_module

        original = graph_module.HybridRetriever
        graph_module.HybridRetriever = LexicalSmokeRetriever  # type: ignore[assignment]
        try:
            started = time.perf_counter()
            usage_path = project_root / "logs/vlm_calls.jsonl"
            start_usage = _read_vlm_usage(usage_path)
            rows = run_profile(
                settings,
                profile_name=B0_PROFILE,
                output_dir=out_path.parent / f"{out_path.stem}_rows",
                dataset="ohrbench",
                split=split,
                repo=repo,
            )
            end_usage = _read_vlm_usage(usage_path)
            runtime_sec = round(time.perf_counter() - started, 4)
        finally:
            graph_module.HybridRetriever = original  # type: ignore[assignment]
    else:
        started = time.perf_counter()
        usage_path = project_root / "logs/vlm_calls.jsonl"
        start_usage = _read_vlm_usage(usage_path)
        rows = run_profile(
            settings,
            profile_name=B0_PROFILE,
            output_dir=out_path.parent / f"{out_path.stem}_rows",
            dataset="ohrbench",
            split=split,
            repo=repo,
        )
        end_usage = _read_vlm_usage(usage_path)
        runtime_sec = round(time.perf_counter() - started, 4)

    summary_core = summarize_examples(rows)
    summary = {
        "EM": float(summary_core["em"]),
        "F1": float(summary_core["f1"]),
        "vlm_rate": float(summary_core["vlm_rate"]),
        "harm_rate": 0.0,  # B0 is the baseline; no matching baseline harm comparison.
        "api_requests": int(end_usage["api_requests"]) - int(start_usage["api_requests"]),
        "prompt_tokens": int(end_usage["prompt_tokens"]) - int(start_usage["prompt_tokens"]),
        "completion_tokens": int(end_usage["completion_tokens"]) - int(start_usage["completion_tokens"]),
        "cost_usd": round(float(end_usage["cost_usd"]) - float(start_usage["cost_usd"]), 6),
        "runtime_sec": runtime_sec,
    }

    routing_probe: dict[str, Any] | None = None
    if probe_routing:
        import faar.graph as graph_module

        original = graph_module.HybridRetriever
        graph_module.HybridRetriever = LexicalSmokeRetriever  # type: ignore[assignment]
        try:
            routing_probe = probe_diagnosis_recovery_routing(
                settings, repo, example_id=example_ids[0]
            )
        finally:
            graph_module.HybridRetriever = original  # type: ignore[assignment]

    first_row = rows[0] if rows else {}
    payload: dict[str, Any] = {
        "label": SMOKE_LABEL,
        "profile": B0_PROFILE,
        "baseline_id": "B0",
        "smoke": True,
        "paper_result": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_spec": {
            "dataset": "ohrbench",
            "split": split,
            "smoke_doc": doc_name,
            "manifest_path": used_manifest.relative_to(project_root).as_posix()
            if used_manifest.is_relative_to(project_root)
            else str(used_manifest),
            "mock_retrieval": mock_retrieval,
            "gate": "off",
            "recovery": "off",
            "embedding_model": settings.retrieval.embedding_model,
            "reranker": settings.retrieval.reranker,
            "ocr_engine": settings.recovery.ocr_engine,
            "vlm_backend": settings.recovery.vlm_backend,
            "vlm_model": settings.vlm_request_model(),
            "vlm_cost_rates": vlm_cost_rates(settings.recovery.vlm_backend),
            "model_provenance": settings.model_provenance(),
            "manifest_sha256": getattr(repo, "manifest_sha256", None),
            "note": (
                "One-document Phase 0 smoke after OHR asset preparation. "
                "Not a paper-run result. Do not aggregate into Phase 1–7 tables."
            ),
        },
        "pipeline_stages": {
            "benchmark_repository": True,
            "retrieval": True,
            "quality_gate": bool(first_row.get("gate") is not None),
            "diagnosis_recovery_routing": {
                "b0_decision": "answer_direct",
                "gate_recorded": bool(first_row.get("gate")),
                "probe": routing_probe,
            },
            "answer_generation": bool(str(first_row.get("predicted_answer", "")).strip() or rows),
            "result_serialization": True,
            "corpus_chunk_count": len(chunks),
            "example_count": len(rows),
        },
        "summary": summary,
        "rows": rows,
    }

    if rows:
        payload["summary"]["EM"] = round(
            sum(float((row.get("metrics") or {}).get("em", exact_match(row.get("predicted_answer", ""), row.get("gold_answer", "")))) for row in rows)
            / len(rows),
            4,
        )
        payload["summary"]["F1"] = round(
            sum(float((row.get("metrics") or {}).get("f1", token_f1(row.get("predicted_answer", ""), row.get("gold_answer", "")))) for row in rows)
            / len(rows),
            4,
        )

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
