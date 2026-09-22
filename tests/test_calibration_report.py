from __future__ import annotations

import inspect
import json
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPORT_PATH = Path(__file__).resolve().parents[1] / "cluster/calibration_report.py"
REPORT_SPEC = spec_from_file_location("calibration_report", REPORT_PATH)
assert REPORT_SPEC and REPORT_SPEC.loader
report = module_from_spec(REPORT_SPEC)
sys.modules["calibration_report"] = report
REPORT_SPEC.loader.exec_module(report)

PAGES = 108
DOCLING_SEC = 20.0
RENDER_SEC = 25.0
OCR_PER_PAGE = 1.5


def build_checkpoint(tmp_path: Path, *, pages: int = PAGES, resumed: bool = False) -> Path:
    checkpoint = {
        "dataset": "ohrbench",
        "split": "val",
        "smoke_doc": "manual/User_Manual_1500S_Classic_EN",
        "resumed": resumed,
        "completed": {
            "manual/User_Manual_1500S_Classic_EN": {
                "page_ids": list(range(pages)),
                "pdf_sha256": "a" * 64,
                "outputs": {
                    "pdf": "out/pdfs/manual/User_Manual_1500S_Classic_EN.pdf",
                    "docling_audit": "out/docling/manual/User_Manual_1500S_Classic_EN.docling.md",
                    "provenance": "out/provenance/manual/User_Manual_1500S_Classic_EN.json",
                },
                "metrics": {
                    "device": "cpu",
                    "docling_runtime_sec": DOCLING_SEC,
                    "render_runtime_sec": RENDER_SEC,
                    "got_ocr_runtime_sec_total": OCR_PER_PAGE * pages,
                    "per_page_got_ocr_runtime_sec": {str(i): OCR_PER_PAGE for i in range(pages)},
                    "peak_rss_bytes": 2**30,
                    "cuda_peak_allocated_bytes": 8 * 2**30,
                    "cuda_peak_reserved_bytes": 10 * 2**30,
                    "storage": {
                        "png_bytes_total": pages * 1_000_000,
                        "ocr_bytes_total": pages * 5_000,
                        "per_page_bytes": {str(i): {"png": 1_000_000, "ocr": 5_000} for i in range(pages)},
                    },
                },
            }
        },
        "failed": {},
        "calibration": {
            "runtime_sec_total_wall": DOCLING_SEC + RENDER_SEC + OCR_PER_PAGE * pages + 5.0,
            "resumed": resumed,
            "cache_bytes_before": 1_000_000_000,
            "cache_bytes_after": 1_500_000_000,
            "cache_path": "/tmp/hf-cache",
        },
    }
    path = tmp_path / "prepare_checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    return path


def build_project(tmp_path: Path) -> Path:
    project = tmp_path
    (project / "OHR-Bench/data").mkdir(parents=True, exist_ok=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(json.dumps([{"ID": "e1", "doc_name": "manual/x"}]))
    (project / "config/datasets").mkdir(parents=True, exist_ok=True)
    (project / "config/datasets/ohr_split.json").write_text(json.dumps({"splits": {"val": ["e1"]}}))
    return project


def make_summary(tmp_path: Path, resumed: bool = False) -> dict:
    project = build_project(tmp_path)
    return report.build_summary(
        checkpoint_path=build_checkpoint(tmp_path, resumed=resumed),
        project_root=project,
        preflight_path=None,
    )


def test_build_summary_schema_complete(tmp_path: Path) -> None:
    summary = make_summary(tmp_path)
    for key in (
        "schema_version",
        "kind",
        "created_at_utc",
        "commit_sha",
        "configuration",
        "gpu",
        "vram_peak",
        "process",
        "runtime",
        "documents",
        "pages",
        "storage",
        "cache",
        "throughput",
        "failures",
        "resumed",
        "checksums",
    ):
        assert key in summary, key
    assert summary["kind"] == "calibration-summary"
    assert summary["pages"]["attempted"] == PAGES
    assert summary["pages"]["completed"] == PAGES
    assert summary["documents"]["completed"] == 1
    assert summary["runtime"]["docling_sec"] == DOCLING_SEC
    assert summary["runtime"]["ocr_sec"] == OCR_PER_PAGE * PAGES
    assert summary["vram_peak"]["allocated_bytes"] == 8 * 2**30
    assert summary["process"]["peak_rss_bytes"] == 2**30
    assert summary["cache"]["growth_bytes"] == 500_000_000
    assert summary["throughput"]["ocr_page_runtime_sec"]["mean"] == OCR_PER_PAGE
    assert summary["throughput"]["ocr_page_runtime_sec"]["p95"] == OCR_PER_PAGE
    assert summary["checksums"]["split_sha256"]


def test_build_summary_reflects_resume_and_failures(tmp_path: Path) -> None:
    summary = make_summary(tmp_path, resumed=True)
    assert summary["resumed"] is True
    assert summary["failures"]["count"] == 0
    checkpoint = json.loads(build_checkpoint(tmp_path).read_text())
    checkpoint["failed"] = {"manual/x": {"error": "boom"}}
    checkpoint["resumed"] = False
    (tmp_path / "prepare_checkpoint.json").write_text(json.dumps(checkpoint))
    project = build_project(tmp_path)
    summary = report.build_summary(checkpoint_path=tmp_path / "prepare_checkpoint.json", project_root=project)
    assert summary["failures"]["count"] == 1
    assert summary["resumed"] is False


def test_build_summary_with_preflight_gpu(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    preflight = {
        "torch": {
            "available": True,
            "cuda_available": True,
            "devices": [
                {
                    "name": "NVIDIA A100",
                    "total_memory_bytes": 80 * 1024**3,
                    "free_memory_bytes": 60 * 1024**3,
                }
            ],
        }
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(preflight))
    summary = report.build_summary(
        checkpoint_path=build_checkpoint(tmp_path),
        project_root=project,
        preflight_path=path,
    )
    assert summary["gpu"]["model"] == "NVIDIA A100"
    assert summary["gpu"]["total_memory_bytes"] == 80 * 1024**3


def test_per_page_storage_keys_never_collide_across_documents(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    checkpoint = json.loads(build_checkpoint(tmp_path).read_text())
    second_doc = {
        "page_ids": list(range(2)),
        "pdf_sha256": "b" * 64,
        "outputs": {},
        "metrics": {
            "docling_runtime_sec": 5.0,
            "render_runtime_sec": 3.0,
            "got_ocr_runtime_sec_total": 2.0,
            "per_page_got_ocr_runtime_sec": {"0": 1.0, "1": 1.0},
            "peak_rss_bytes": 2**30,
            "storage": {
                "png_bytes_total": 2 * 100,
                "ocr_bytes_total": 2 * 10,
                "per_page_bytes": {"0": {"png": 100, "ocr": 10}, "1": {"png": 100, "ocr": 10}},
            },
        },
    }
    checkpoint["completed"]["paper/second_doc"] = second_doc
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(checkpoint))
    summary = report.build_summary(checkpoint_path=path, project_root=project)
    assert len(summary["storage"]["per_page_bytes"]) == PAGES + 2
    assert "manual/User_Manual_1500S_Classic_EN:p0" in summary["storage"]["per_page_bytes"]
    assert "paper/second_doc:p0" in summary["storage"]["per_page_bytes"]
    assert summary["pages"]["attempted"] == PAGES + 2
    assert summary["documents"]["completed"] == 2


def test_build_summary_includes_carried_forward_attempt_timings(tmp_path: Path) -> None:
    """A resumed calibration keeps prior-attempt OCR timings per page."""
    project = build_project(tmp_path)
    checkpoint = json.loads(build_checkpoint(tmp_path).read_text())
    entry = checkpoint["completed"]["manual/User_Manual_1500S_Classic_EN"]
    per_page = {str(i): OCR_PER_PAGE for i in range(PAGES)}
    per_page["0"] = 0.4
    per_page["1"] = 0.6
    entry["metrics"]["per_page_got_ocr_runtime_sec"] = per_page
    entry["metrics"]["got_ocr_runtime_sec_total"] = round(0.4 + 0.6 + (PAGES - 2) * OCR_PER_PAGE, 4)
    entry["metrics"]["docling_runtime_sec"] = DOCLING_SEC
    checkpoint["resumed"] = True
    path = tmp_path / "resumed.json"
    path.write_text(json.dumps(checkpoint))
    summary = report.build_summary(checkpoint_path=path, project_root=project)
    assert summary["resumed"] is True
    distribution = summary["throughput"]["ocr_page_runtime_sec"]
    assert distribution["n"] == PAGES
    assert distribution["min"] == 0.4 and distribution["max"] == OCR_PER_PAGE
    assert summary["runtime"]["ocr_sec"] == pytest.approx(0.4 + 0.6 + (PAGES - 2) * OCR_PER_PAGE, abs=1e-3)


def test_build_rejects_empy_or_malformed_checkpoint(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    with pytest.raises(SystemExit, match="unreadable JSON"):
        report.build_summary(checkpoint_path=tmp_path / "missing.json", project_root=project)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"completed": {}}))
    with pytest.raises(SystemExit, match="no completed document metrics"):
        report.build_summary(checkpoint_path=bad, project_root=project)
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(SystemExit, match="unreadable JSON"):
        report.build_summary(checkpoint_path=broken, project_root=project)


def test_projection_keeps_components_separate(tmp_path: Path) -> None:
    summary = make_summary(tmp_path)
    projection = report.project_preparation(
        summary,
        headroom_fraction=0.25,
        walltime_hours_per_shard=24.0,
        val_documents=10,
        val_pages=100,
        test_documents=10,
        test_pages=100,
    )
    components = projection["projections"]["val"]["components"]
    assert set(components) == {"docling_sec", "render_sec", "ocr_sec"}
    assert set(projection["rates"]) == {
        "docling_sec_per_document",
        "render_sec_per_page",
        "ocr_sec_per_page",
        "storage_bytes_per_page",
        "pages_per_document_observed",
    }
    assert projection["rates"]["docling_sec_per_document"] == DOCLING_SEC
    assert projection["rates"]["ocr_sec_per_page"]["mean"] == OCR_PER_PAGE
    assert components["docling_sec"] == 10 * DOCLING_SEC
    assert components["ocr_sec"]["mean"] == 100 * OCR_PER_PAGE
    total = projection["projections"]["val"]["total_sec"]["mean"]
    assert total == pytest.approx(10 * DOCLING_SEC + 100 * (RENDER_SEC / PAGES) + 100 * OCR_PER_PAGE, abs=0.1)
    with_headroom = projection["projections"]["val"]["total_with_headroom_sec"]
    assert with_headroom == pytest.approx(total * 1.25, abs=0.1)


def test_projection_default_inventory_and_sharding(tmp_path: Path) -> None:
    summary = make_summary(tmp_path)
    projection = report.project_preparation(
        summary,
        headroom_fraction=0.25,
        walltime_hours_per_shard=24.0,
        val_documents=549,
        val_pages=7037,
        test_documents=567,
        test_pages=6849,
    )
    val = projection["projections"]["val"]
    test = projection["projections"]["test"]
    assert val["documents"] == 549 and val["pages"] == 7037
    assert test["documents"] == 567 and test["pages"] == 6849
    assert projection["sharding"]["suggested_shards_val"] == max(
        1, math.ceil(val["total_with_headroom_sec"] / (24 * 3600))
    )
    assert any("multi-document pilot" in warning for warning in projection["warnings"])
    assert any("single-document calibration" in warning for warning in projection["warnings"])
    assert any("must not be collapsed" in warning for warning in projection["warnings"])


def test_projection_storage_scales_with_headroom(tmp_path: Path) -> None:
    summary = make_summary(tmp_path)
    projection = report.project_preparation(
        summary,
        headroom_fraction=0.5,
        walltime_hours_per_shard=24.0,
        val_documents=1,
        val_pages=100,
        test_documents=1,
        test_pages=100,
    )
    storage = projection["projections"]["val"]["storage_bytes"]
    per_page = (1_000_000 + 5_000)
    assert storage["raw"] == 100 * per_page
    assert storage["with_headroom"] == int(100 * per_page * 1.5)


def test_cli_build_and_project_roundtrip(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    checkpoint = build_checkpoint(tmp_path)
    summary_out = tmp_path / "summary.json"
    projection_out = tmp_path / "projection.json"
    assert report.main(["build", "--checkpoint", str(checkpoint), "--project-root", str(project), "--out", str(summary_out)]) == 0
    assert report.main(
        ["project", "--summary", str(summary_out), "--out", str(projection_out), "--val-pages", "100", "--val-documents", "10", "--test-pages", "100", "--test-documents", "10"]
    ) == 0
    summary = json.loads(summary_out.read_text())
    projection = json.loads(projection_out.read_text())
    assert summary["kind"] == "calibration-summary"
    assert projection["kind"] == "preparation-projection"
    assert not list(tmp_path.glob(".summary.json.*.tmp"))


def test_cli_project_rejects_non_summary(tmp_path: Path) -> None:
    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({"kind": "other"}))
    with pytest.raises(SystemExit, match="not a calibration-summary"):
        report.main(["project", "--summary", str(junk)])


def test_build_summary_exposes_incomplete_timing(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    checkpoint = json.loads(build_checkpoint(tmp_path).read_text())
    checkpoint["calibration"]["timing_complete"] = False
    checkpoint["calibration"]["runtime_sec_total_wall"] = None
    checkpoint["calibration"]["measured_wall_sec"] = 12.0
    checkpoint["attempt_records"] = [
        {"id": 1, "state": "completed", "elapsed_sec": 4.0},
        {"id": 2, "state": "unclean", "elapsed_sec": None},
    ]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(checkpoint))
    summary = report.build_summary(checkpoint_path=path, project_root=project)
    assert summary["runtime"]["timing_complete"] is False
    assert summary["runtime"]["total_wall_sec"] is None
    assert summary["runtime"]["measured_wall_sec"] == 12.0
    with pytest.raises(SystemExit, match="scripts/data/prepare_benchmark_assets.py --scheduler-elapsed-sec"):
        report.project_preparation(
            summary,
            headroom_fraction=0.25,
            walltime_hours_per_shard=24.0,
            val_documents=10,
            val_pages=100,
            test_documents=10,
            test_pages=100,
        )


def test_project_cli_rejects_incomplete_timing(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    checkpoint = json.loads(build_checkpoint(tmp_path).read_text())
    checkpoint["calibration"]["timing_complete"] = False
    checkpoint["calibration"]["runtime_sec_total_wall"] = None
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(checkpoint))
    summary = report.build_summary(checkpoint_path=path, project_root=project)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match="scripts/data/prepare_benchmark_assets.py --scheduler-elapsed-sec"):
        report.main(["project", "--summary", str(summary_path)])


def test_project_parser_rejects_scheduler_elapsed_sec(capsys: pytest.CaptureFixture[str]) -> None:
    assert "scheduler_elapsed_sec" not in inspect.signature(report.project_preparation).parameters
    with pytest.raises(SystemExit) as exc:
        report.main(["project", "--summary", "missing.json", "--scheduler-elapsed-sec", "12"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
    assert "--scheduler-elapsed-sec" in err


def test_complete_summary_projects_successfully(tmp_path: Path) -> None:
    summary = make_summary(tmp_path)
    assert summary["runtime"]["timing_complete"] is not False
    projection = report.project_preparation(
        summary,
        headroom_fraction=0.25,
        walltime_hours_per_shard=24.0,
        val_documents=10,
        val_pages=100,
        test_documents=10,
        test_pages=100,
    )
    assert projection["kind"] == "preparation-projection"
    assert projection["timing"]["complete"] is True
    assert "scheduler_elapsed_sec" not in projection["timing"]


def test_projection_cannot_claim_complete_while_summary_is_incomplete(tmp_path: Path) -> None:
    summary = make_summary(tmp_path)
    summary["runtime"]["timing_complete"] = False
    with pytest.raises(SystemExit, match="timing is incomplete"):
        report.project_preparation(
            summary,
            headroom_fraction=0.25,
            walltime_hours_per_shard=24.0,
            val_documents=10,
            val_pages=100,
            test_documents=10,
            test_pages=100,
        )
