from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .answering import answer_from_hits
from .api_logging import is_valid_api_usage, vlm_cost_rates, zero_api_usage
from .benchmarks import load_benchmark_repository
from .data import Phase0Repository
from .experiment_profiles import apply_profile
from .graph import build_graph, random_recovery_type
from .metrics import exact_match, ndcg_at_k, recall_at_k, token_f1
from .operations import ProgressReporter, check_termination
from .resource_limits import enforce_memory_budget, is_fatal_resource_error
from .run_io import atomic_write_json, run_fingerprint, safe_checkpoint_stem, select_shard
from .settings import AppSettings


def _effect_label(actual: float, counterfactual: float) -> str:
    if actual > counterfactual:
        return "improved"
    if actual < counterfactual:
        return "worsened"
    return "equal"


def run_profile(
    settings: AppSettings,
    profile_name: str,
    max_examples: int | None = None,
    output_dir: Path | None = None,
    example_ids: list[str] | None = None,
    selection: dict[str, Any] | None = None,
    dataset: str | None = None,
    split: str | None = None,
    repo: Any | None = None,
    *,
    resume: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    settings = apply_profile(settings, profile_name)
    use_repo_kwarg = repo is not None or bool(dataset and split)
    if repo is None:
        if dataset and split:
            repo = load_benchmark_repository(settings.project_root, dataset, split)
        else:
            repo = Phase0Repository(settings)
    fingerprint = run_fingerprint(
        settings,
        profile=profile_name,
        dataset=dataset,
        split=split,
        manifest_sha256=getattr(repo, "manifest_sha256", None),
    )
    selected_ids = list(example_ids) if example_ids is not None else repo.list_example_ids()
    if max_examples is not None and example_ids is None:
        selected_ids = selected_ids[: max(0, max_examples)]
    if num_shards is not None:
        selected_ids = select_shard(selected_ids, shard_index, num_shards)
    base_output = output_dir or (settings.project_root / "logs/phase3" / profile_name)
    base_output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows_by_id: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    graph = None
    if resume:
        for example_id in selected_ids:
            cached = _load_checkpoint(_checkpoint_path(base_output, example_id))
            if cached is not None and _is_valid_checkpoint(cached, example_id, profile_name, fingerprint, settings):
                rows_by_id[example_id] = cached
            else:
                pending.append(example_id)
        if pending:
            graph = build_graph(settings, repo=repo) if use_repo_kwarg else build_graph(settings)
    else:
        pending = list(selected_ids)
        if pending or (example_ids is not None and num_shards is None):
            graph = build_graph(settings, repo=repo) if use_repo_kwarg else build_graph(settings)
    if resume and rows_by_id:
        print(
            f"[faar] resume: reused {len(rows_by_id)}/{len(selected_ids)} examples from checkpoints",
            flush=True,
        )
    reporter = ProgressReporter(f"profile:{profile_name}", len(pending))
    failures: list[tuple[str, str]] = []
    processed = 0
    for example_id in pending:
        check_termination()
        enforce_memory_budget(f"profile:{profile_name} before example {example_id}")
        try:
            result = graph.invoke({"example_id": example_id})
            result_hits = result.get("corrected_hits") or result.get("retrieved_hits", [])
            hit_texts = [hit.chunk.text for hit in result_hits]
            hit_image_paths = list(dict.fromkeys(hit.chunk.image_path for hit in result_hits if hit.chunk.image_path))
            gold = result["example"].correct_answer
            prediction = result.get("answer", "")
            visual_result = result.get("visual_result") or {}
            if visual_result.get("status") == "failed":
                raise RuntimeError(
                    f"example {example_id!r}: paid VLM call failed with reason "
                    f"{visual_result.get('reason', 'vlm_call_failed')!r}; "
                    "refusing to score or checkpoint a failed row"
                )
            api_usage = visual_result.get("api_usage")
            if api_usage is None:
                if visual_result:
                    raise ValueError(f"example {example_id!r} returned no API usage")
                api_usage = zero_api_usage()
            elif not is_valid_api_usage(api_usage):
                raise ValueError(f"example {example_id!r} returned invalid API usage")
            question = result.get("question", "")
            retrieved_hits = result.get("retrieved_hits", [])
            counterfactual_answer = answer_from_hits(question, retrieved_hits).get("answer", "")
            em = exact_match(prediction, gold)
            f1 = token_f1(prediction, gold)
            counterfactual_em = exact_match(counterfactual_answer, gold)
            counterfactual_f1 = token_f1(counterfactual_answer, gold)
            row = {
                "profile": profile_name,
                "example_id": example_id,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": prediction,
                "failure_type": result.get("failure_type", "pass"),
                "gate": result.get("gate", {}),
                "policy_action": result.get("policy_action", "answer_direct"),
                "action_outcome": result.get("action_outcome", {}),
                "request_model": visual_result.get("request_model"),
                "response_model": visual_result.get("response_model"),
                "completed_at_utc": visual_result.get("completed_at_utc"),
                "cost_rates": visual_result.get("cost_rates") or vlm_cost_rates(settings.recovery.vlm_backend),
                "api_usage": api_usage,
                "metrics": {
                    "ndcg@5": ndcg_at_k(hit_texts, gold, k=5),
                    "recall@5": recall_at_k(hit_texts, gold, k=5),
                    "em": em,
                    "f1": f1,
                },
                "recovery_metrics": {
                    "counterfactual_answer": counterfactual_answer,
                    "recovery_changed_answer": counterfactual_answer != prediction,
                    "em": {
                        "actual": round(em, 4),
                        "counterfactual": round(counterfactual_em, 4),
                        "delta": round(em - counterfactual_em, 4),
                        "effect": _effect_label(em, counterfactual_em),
                    },
                    "f1": {
                        "actual": round(f1, 4),
                        "counterfactual": round(counterfactual_f1, 4),
                        "delta": round(f1 - counterfactual_f1, 4),
                        "effect": _effect_label(f1, counterfactual_f1),
                    },
                },
                "top_hit_texts": hit_texts[:5],
                "top_reranker_score": (result.get("gate") or {}).get("top_reranker_score", 0.0),
                "source_assets": {
                    "ocr_text_path": str(getattr(result["example"], "ocr_text_path", "")),
                    "image_paths": hit_image_paths,
                },
                "run_metadata": {
                    "profile": profile_name,
                    "run_id": run_id,
                    "run_fingerprint": fingerprint,
                    "manifest_sha256": getattr(repo, "manifest_sha256", None),
                    "api_enabled": settings.recovery.api_enabled,
                    "vlm_backend": settings.recovery.vlm_backend,
                    "openai_model": settings.recovery.openai_model,
                    "vlm_model": settings.vlm_request_model(),
                    "enable_byt5": settings.recovery.enable_byt5,
                    "byt5_model": settings.recovery.byt5_model,
                    "seed": seed,
                    "cost_rates": vlm_cost_rates(settings.recovery.vlm_backend),
                    "evaluation_size": len(selected_ids),
                    "selection": selection or {"max_examples": max_examples},
                    "dataset": dataset or "phase0",
                    "split": split or "development",
                },
            }
            if result.get("recovery_type") is not None:
                row["recovery_type"] = result["recovery_type"]
        except Exception as exc:
            if is_fatal_resource_error(exc):
                raise
            reason = f"{type(exc).__name__}: {exc}"
            failed_row = {
                "profile": profile_name,
                "example_id": example_id,
                "action_outcome": {"action": "failed", "status": "failed", "reason": reason},
                "api_usage": zero_api_usage(),
                "error": reason,
                "run_metadata": {
                    "profile": profile_name,
                    "run_id": run_id,
                    "run_fingerprint": fingerprint,
                    "manifest_sha256": getattr(repo, "manifest_sha256", None),
                    "evaluation_size": len(selected_ids),
                    "selection": selection or {"max_examples": max_examples},
                    "dataset": dataset or "phase0",
                    "split": split or "development",
                    "seed": seed,
                },
            }
            atomic_write_json(_checkpoint_path(base_output, example_id), failed_row)
            failures.append((example_id, reason))
            print(f"[faar] example {example_id!r} FAILED and was not scored: {reason}", flush=True)
            processed += 1
            reporter.update(processed)
            continue
        rows_by_id[example_id] = row
        atomic_write_json(_checkpoint_path(base_output, example_id), row)
        enforce_memory_budget(f"profile:{profile_name} after example {example_id}")
        processed += 1
        reporter.update(processed)
    if failures:
        ids = ", ".join(example_id for example_id, _ in failures)
        raise RuntimeError(
            f"{len(failures)} example(s) failed and were not scored: {ids}. "
            "Completed examples remain checkpointed; fix the cause and resume."
        )
    reporter.finish()
    return [rows_by_id[example_id] for example_id in selected_ids if example_id in rows_by_id]


def _checkpoint_path(base_output: Path, example_id: str) -> Path:
    return base_output / f"{safe_checkpoint_stem(example_id)}.json"


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) else None


def _is_valid_checkpoint(
    row: dict[str, Any],
    example_id: str,
    profile_name: str,
    fingerprint: str,
    settings: AppSettings,
) -> bool:
    if not (
        row.get("example_id") == example_id
        and row.get("profile") == profile_name
        and (row.get("run_metadata") or {}).get("run_fingerprint") == fingerprint
        and is_valid_api_usage(row.get("api_usage"))
    ):
        return False
    if (row.get("action_outcome") or {}).get("status") == "failed":
        return False
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for key in ("ndcg@5", "recall@5", "em", "f1"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
    if settings.experiment.random_recovery and row.get("failure_type") == "random":
        expected = random_recovery_type(settings.experiment.random_seed, example_id)
        if row.get("recovery_type") != expected:
            return False
    return True
