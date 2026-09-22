"""Phase 0 one-document B0 end-to-end smoke contracts.

Exercises prepared-asset → BenchmarkRepository → retrieval → gate → routing →
answer → serialization without paid APIs or full-model downloads.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from faar.data import DatasetUnavailableError
from faar.settings import AppSettings
from faar.smoke_b0 import (
    DEFAULT_SMOKE_DOC,
    REQUIRED_SUMMARY_FIELDS,
    build_ohr_one_doc_smoke_manifest,
    load_one_doc_smoke_repository,
    run_one_document_b0_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_SMOKE_ROOT = ROOT / "data/benchmark_prep/smoke"
REAL_SMOKE_OCR = (
    REAL_SMOKE_ROOT
    / "ocr"
    / "academic"
    / "DUDE_157911e3080d18f4d799a122aaeb33fb_page_0.txt"
)
REAL_SMOKE_IMAGE = (
    REAL_SMOKE_ROOT
    / "images"
    / "academic"
    / "DUDE_157911e3080d18f4d799a122aaeb33fb_page_0.png"
)


def _write_locked_models(project: Path) -> None:
    (project / "config").mkdir(parents=True, exist_ok=True)
    src = ROOT / "config/model_revisions.json"
    if src.is_file():
        shutil.copy(src, project / "config/model_revisions.json")
    else:
        (project / "config/model_revisions.json").write_text(
            json.dumps(
                {
                    "models": {
                        "embedding": {"repository": "nvidia/NV-Embed-v2", "revision": "a" * 40},
                        "reranker": {"repository": "BAAI/bge-reranker-v2-m3", "revision": "b" * 40},
                        "got_ocr": {
                            "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                            "revision": "c" * 40,
                        },
                        "byt5": {"repository": "google/byt5-small", "revision": "d" * 40},
                        "colpali": {
                            "repository": "vidore/colpali-v1.2-hf",
                            "revision": "e" * 40,
                        },
                        "visrag": {"repository": "openbmb/VisRAG-Ret", "revision": "f" * 40},
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _seed_mini_ohr_fixture(project: Path, *, doc_name: str = DEFAULT_SMOKE_DOC) -> Path:
    """Seed a one-doc smoke tree with real split/qas when available, else minimal fixtures."""
    _write_locked_models(project)
    smoke_root = project / "data/benchmark_prep/smoke"
    ocr_path = smoke_root / "ocr" / f"{doc_name}_page_0.txt"
    image_path = smoke_root / "images" / f"{doc_name}_page_0.png"
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    if REAL_SMOKE_OCR.is_file() and REAL_SMOKE_IMAGE.is_file():
        ocr_path.write_text(REAL_SMOKE_OCR.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        shutil.copy(REAL_SMOKE_IMAGE, image_path)
    else:
        ocr_path.write_text(
            "The Water Quality of Surface Water and Groundwater in Qala-i-Naw city, Afghanistan.\n"
            "Copper (Cu) concentrations in well water samples from Kabul city meet the Japanese 2015 standard.\n"
            "Yes, Cu concentration range is compliant.\n",
            encoding="utf-8",
        )
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nsmoke")

    (smoke_root / f"prepare_result_{doc_name.replace('/', '__')}.json").write_text(
        json.dumps({"doc_name": doc_name, "page_ids": [0]}) + "\n",
        encoding="utf-8",
    )

    example_id = "82f09e85-85a0-459c-88fc-0479b095cb6f"
    qas_src = ROOT / "OHR-Bench/data/qas_v2.json"
    split_src = ROOT / "config/datasets/ohr_split.json"
    (project / "OHR-Bench/data").mkdir(parents=True, exist_ok=True)
    (project / "config/datasets").mkdir(parents=True, exist_ok=True)
    if qas_src.is_file() and split_src.is_file():
        # Use real locked split + the real QA row for the smoke doc (no fabrication).
        split_payload = json.loads(split_src.read_text(encoding="utf-8"))
        qas_rows = json.loads(qas_src.read_text(encoding="utf-8"))
        row = next(r for r in qas_rows if str(r.get("doc_name")) == doc_name and str(r["ID"]) == example_id)
        (project / "config/datasets/ohr_split.json").write_text(
            json.dumps({"splits": {"train": [], "val": [example_id], "test": []}}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Keep only the smoke example so the fixture stays one-document.
        (project / "OHR-Bench/data/qas_v2.json").write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
        # Preserve original split membership assertion in a sidecar (not used by code).
        (project / "data/benchmark_prep/smoke/source_split_membership.json").write_text(
            json.dumps(
                {
                    "example_id": example_id,
                    "in_locked_val": example_id in set(split_payload["splits"]["val"]),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (project / "config/datasets/ohr_split.json").write_text(
            json.dumps({"splits": {"train": [], "val": [example_id], "test": []}}) + "\n",
            encoding="utf-8",
        )
        (project / "OHR-Bench/data/qas_v2.json").write_text(
            json.dumps(
                [
                    {
                        "ID": example_id,
                        "doc_name": doc_name,
                        "questions": "Is Cu concentration range compliant with Japanese 2015 standard?",
                        "answers": "Yes",
                        "evidence_page_no": 0,
                        "doc_type": "academic",
                    }
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return smoke_root


def test_build_manifest_from_prepared_smoke_assets(tmp_path: Path) -> None:
    smoke_root = _seed_mini_ohr_fixture(tmp_path)
    manifest = build_ohr_one_doc_smoke_manifest(tmp_path, smoke_root=smoke_root)

    assert manifest["smoke"] is True
    assert manifest["paper_result"] is False
    assert manifest["smoke_doc"] == DEFAULT_SMOKE_DOC
    assert len(manifest["records"]) == 1
    assert manifest["document_inventory"][DEFAULT_SMOKE_DOC] == [0]
    assert manifest["records"][0]["corpus_ids"] == [f"{DEFAULT_SMOKE_DOC}:p0"]
    assert Path(tmp_path / manifest["corpus_pages"][0]["ocr_text_path"]).is_file()
    assert Path(tmp_path / manifest["corpus_pages"][0]["image_path"]).is_file()

    repo, _ = load_one_doc_smoke_repository(tmp_path, smoke_root=smoke_root)
    chunks = repo.get_corpus_chunks(AppSettings(project_root=tmp_path).retrieval)
    assert len(chunks) >= 1
    assert any("water" in chunk.text.lower() or "cu" in chunk.text.lower() or chunk.text.strip() for chunk in chunks)


def test_b0_one_doc_smoke_end_to_end_pipeline(tmp_path: Path) -> None:
    smoke_root = _seed_mini_ohr_fixture(tmp_path)
    settings = AppSettings(project_root=tmp_path)
    out_path = tmp_path / "results/smoke/b0_one_doc.json"

    payload = run_one_document_b0_smoke(
        settings,
        out_path=out_path,
        smoke_root=smoke_root,
        mock_retrieval=True,
        probe_routing=True,
    )

    assert out_path.is_file()
    assert payload["smoke"] is True
    assert payload["paper_result"] is False
    assert payload["baseline_id"] == "B0"
    assert payload["profile"] == "naive_rag"
    for field in REQUIRED_SUMMARY_FIELDS:
        assert field in payload["summary"], field
        assert isinstance(payload["summary"][field], (int, float))

    stages = payload["pipeline_stages"]
    assert stages["benchmark_repository"] is True
    assert stages["retrieval"] is True
    assert stages["quality_gate"] is True
    assert stages["answer_generation"] is True
    assert stages["result_serialization"] is True
    assert stages["example_count"] == 1
    assert stages["corpus_chunk_count"] >= 1

    row = payload["rows"][0]
    assert "gate" in row
    assert row["action_outcome"]["action"] == "answer_direct"
    assert str(row["predicted_answer"]).strip() != ""
    assert row["source_assets"]["image_paths"] or row["top_hit_texts"]

    probe = stages["diagnosis_recovery_routing"]["probe"]
    assert probe is not None
    assert probe["smoke_probe"] is True
    assert probe["paper_result"] is False
    assert probe["failure_type"] in {"semantic", "word_level", "structural"}
    assert probe["policy_action"] in {"correct_text", "retry_retrieval", "invoke_vlm", "answer_direct"}
    assert probe["answer_present"] is True

    # B0 smoke must not issue paid VLM traffic.
    assert payload["summary"]["api_requests"] == 0
    assert payload["summary"]["cost_usd"] == 0.0
    assert payload["summary"]["vlm_rate"] == 0.0


def test_manifest_rejects_missing_prepared_assets(tmp_path: Path) -> None:
    smoke_root = _seed_mini_ohr_fixture(tmp_path)
    (smoke_root / "ocr" / f"{DEFAULT_SMOKE_DOC}_page_0.txt").unlink()
    with pytest.raises(DatasetUnavailableError, match="incomplete|OCR"):
        build_ohr_one_doc_smoke_manifest(tmp_path, smoke_root=smoke_root)


@pytest.mark.skipif(
    not (REAL_SMOKE_OCR.is_file() and REAL_SMOKE_IMAGE.is_file()),
    reason="Prepared one-document OHR smoke assets are not present on this machine.",
)
def test_real_prepared_smoke_assets_cli(tmp_path: Path) -> None:
    out_path = tmp_path / "b0_one_doc.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/smoke/run_b0_one_doc_smoke.py"),
        "--project-root",
        str(ROOT),
        "--smoke-root",
        str(REAL_SMOKE_ROOT),
        "--out",
        str(out_path),
    ]
    child_env = os.environ.copy()
    for key in ("KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_BLOCKTIME", "OMP_WAIT_POLICY"):
        child_env.pop(key, None)
    child_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "KMP_BLOCKTIME": "0",
            "OMP_WAIT_POLICY": "PASSIVE",
            "KMP_INIT_AT_FORK": "TRUE",
        }
    )
    stdout_path = tmp_path / "smoke_stdout.txt"
    stderr_path = tmp_path / "smoke_stderr.txt"
    stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        child_pid = os.posix_spawn(
            sys.executable,
            cmd,
            child_env,
            file_actions=[
                (os.POSIX_SPAWN_DUP2, stdout_fd, 1),
                (os.POSIX_SPAWN_DUP2, stderr_fd, 2),
            ],
        )
        _, status = os.waitpid(child_pid, 0)
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)
    returncode = os.waitstatus_to_exitcode(status)
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    assert returncode == 0, stderr + stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["smoke"] is True
    assert payload["paper_result"] is False
    for field in REQUIRED_SUMMARY_FIELDS:
        assert field in payload["summary"]
    assert payload["summary"]["api_requests"] == 0
    assert payload["pipeline_stages"]["corpus_chunk_count"] >= 1
    # Real smoke uses the locked val example for the prepared document.
    assert payload["rows"][0]["example_id"] == "82f09e85-85a0-459c-88fc-0479b095cb6f"
