from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_benchmark_assets import load_checkpoint, save_checkpoint
from prepare_benchmark_assets import page_image_name as cli_page_image_name

from faar.asset_preparation import (
    execute_document_preparation,
    page_image_name,
    page_ocr_name,
    plan_document_work,
)


def test_deterministic_output_naming() -> None:
    assert page_image_name("academic/demo", 3) == "academic/demo_page_3.png"
    assert page_ocr_name("academic/demo", 3) == "academic/demo_page_3.txt"
    assert cli_page_image_name("academic/demo", 3) == "academic/demo_page_3.png"


def test_smoke_plan_is_resumable_and_portable(tmp_path: Path) -> None:
    pdf = tmp_path / "prepared/pdfs/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    out_root = tmp_path / "prepared"
    work = plan_document_work(
        project_root=tmp_path,
        doc_rel="academic/demo",
        pdf_path=pdf,
        page_ids=[0, 1],
        out_root=out_root,
        got_ocr={"repository": "stepfun-ai/GOT-OCR-2.0-hf", "revision": "a" * 40},
    )

    assert work["pdf"] == "prepared/pdfs/academic/demo.pdf"
    assert work["stages"]["render_pages"]["outputs"] == [
        "prepared/images/academic/demo_page_0.png",
        "prepared/images/academic/demo_page_1.png",
    ]
    assert work["stages"]["got_ocr"]["outputs"] == [
        "prepared/ocr/academic/demo_page_0.txt",
        "prepared/ocr/academic/demo_page_1.txt",
    ]
    assert work["checkpoint"]["render_pages"] is False


def test_execute_one_document_with_injected_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path
    (project / "config").mkdir()
    (project / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    },
                    "docling": {
                        "repository": "docling-project/docling-models",
                        "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
                    },
                }
            }
        )
    )
    pdf_src = project / "pdfs/academic/demo.pdf"
    pdf_src.parent.mkdir(parents=True)
    # Minimal valid-enough PDF bytes are not required because we mock render/OCR/Docling,
    # but filesystem source must exist for copy.
    pdf_src.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")

    out_root = project / "prepared"
    revisions_seen: list[str | None] = []

    def fake_docling(pdf_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# docling audit\n", encoding="utf-8")
        return output_path

    def fake_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        revisions_seen.append(revision)
        return f"OCR for {image_path.name}"

    # Monkeypatch render via writing PNGs ourselves by patching render_pdf_pages.
    import faar.asset_preparation as prep

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_id]))
        return {"pdf_page_count": len(page_ids), "render_runtime_sec": 0.01, "render_scale": scale}

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)

    result = execute_document_preparation(
        project_root=project,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=out_root,
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )

    assert result.got_ocr_revision == "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
    assert revisions_seen == ["d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"]
    assert (out_root / "images/academic/demo_page_0.png").stat().st_size > 0
    assert (out_root / "ocr/academic/demo_page_0.txt").read_text().startswith("OCR for")
    assert not Path(result.outputs["images"][0]).is_absolute()
    provenance = json.loads((out_root / "provenance/academic/demo.json").read_text())
    assert provenance["got_ocr_revision"] == result.got_ocr_revision
    assert "Users" not in json.dumps(provenance)


def test_execute_document_checks_memory_at_every_expensive_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faar.asset_preparation as prep

    (tmp_path / "config").mkdir()
    (tmp_path / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    },
                    "docling": {
                        "repository": "docling-project/docling-models",
                        "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "pdfs/academic/demo.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    stages: list[str] = []

    def fake_docling(_pdf: Path, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("audit", encoding="utf-8")
        return output

    def fake_render(_pdf: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id in page_ids:
            image_paths[page_id].parent.mkdir(parents=True, exist_ok=True)
            image_paths[page_id].write_bytes(b"png")
        return {"pdf_page_count": 1, "render_runtime_sec": 0.0, "render_scale": scale}

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    monkeypatch.setattr(prep, "enforce_memory_budget", stages.append)

    execute_document_preparation(
        project_root=tmp_path,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=tmp_path / "prepared",
        pdf_root=tmp_path / "pdfs",
        extract_got_ocr_fn=lambda *_args, **_kwargs: "text",
        export_docling_fn=fake_docling,
    )

    assert stages == [
        "asset preparation before PDF extraction",
        "asset preparation after PDF extraction",
        "asset preparation before Docling audit",
        "asset preparation after Docling audit",
        "asset preparation before page rendering",
        "asset preparation after page rendering",
        "asset preparation before GOT-OCR page 0",
        "asset preparation after GOT-OCR page 0",
    ]


def test_render_checks_memory_and_termination_per_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faar.asset_preparation as prep

    events: list[str] = []

    class FakeImage:
        def save(self, path: Path, format: str) -> None:
            path.write_bytes(b"png")

        def close(self) -> None:
            events.append("image.close")

    class FakeBitmap:
        def to_pil(self) -> FakeImage:
            return FakeImage()

        def close(self) -> None:
            events.append("bitmap.close")

    class FakePage:
        def render(self, scale: float) -> FakeBitmap:
            return FakeBitmap()

        def close(self) -> None:
            events.append("page.close")

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            pass

        def __len__(self) -> int:
            return 2

        def __getitem__(self, _page_id: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            events.append("document.close")

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=FakeDocument))
    monkeypatch.setattr(prep, "check_termination", lambda: events.append("termination"))
    monkeypatch.setattr(prep, "enforce_memory_budget", events.append)
    outputs = {page_id: tmp_path / f"page-{page_id}.png" for page_id in (0, 1)}

    prep.render_pdf_pages(tmp_path / "input.pdf", [0, 1], outputs)

    assert events.count("termination") == 4
    assert events.count("image.close") == 2
    assert events.count("bitmap.close") == 2
    assert events.count("page.close") == 2
    assert events[-1] == "document.close"
    for page_id in (0, 1):
        assert f"asset preparation before rendering page {page_id}" in events
        assert f"asset preparation after rendering page {page_id}" in events


def test_checkpoint_writes_atomically_and_corruption_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    checkpoint = tmp_path / "checkpoint.json"
    writes: list[tuple[Path, str]] = []
    monkeypatch.setattr(cli, "atomic_write_text", lambda path, text: writes.append((path, text)))
    save_checkpoint(checkpoint, {"completed": {}})
    assert writes[0][0] == checkpoint
    assert json.loads(writes[0][1])["updated_at_utc"]

    checkpoint.write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable JSON"):
        load_checkpoint(checkpoint)


def test_cli_smoke_writes_plan_without_execution(tmp_path: Path) -> None:
    inventory = tmp_path / "OHR-Bench/data/retrieval_base/gt/academic"
    inventory.mkdir(parents=True)
    (inventory / "demo.json").write_text(json.dumps([{"text": "x", "page_idx": 0}]))
    pdf = tmp_path / "pdfs/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    },
                    "docling": {
                        "repository": "docling-project/docling-models",
                        "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
                    },
                }
            }
        )
    )
    out_root = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts/data/prepare_benchmark_assets.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "ohrbench",
            "--pdf-root",
            str(tmp_path / "pdfs"),
            "--document-inventory",
            str(tmp_path / "OHR-Bench/data/retrieval_base/gt"),
            "--out-root",
            str(out_root),
            "--smoke-doc",
            "academic/demo",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "plan_only"
    assert payload["page_ids"] == [0]
    assert (out_root / "prepare_checkpoint.json").is_file()


def test_calibration_first_run_preemption_persists_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    project = tmp_path
    (project / "config").mkdir()
    (project / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    },
                    "docling": {
                        "repository": "docling-project/docling-models",
                        "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
                    },
                }
            }
        )
    )
    (project / "OHR-Bench/data").mkdir(parents=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(json.dumps([{"ID": "e1", "doc_name": "academic/demo"}]))
    (project / "config/datasets").mkdir(parents=True, exist_ok=True)
    (project / "config/datasets/ohr_split.json").write_text(json.dumps({"splits": {"val": ["e1"]}}))
    gt = project / "OHR-Bench/data/retrieval_base/gt/academic"
    gt.mkdir(parents=True)
    (gt / "demo.json").write_text(json.dumps([{"text": "x", "page_idx": 0}]))
    pdf = project / "pdfs/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    out_root = project / "out"
    checkpoint_path = out_root / "prepare_checkpoint.json"
    args = [
        "--dataset",
        "ohrbench",
        "--pdf-root",
        str(project / "pdfs"),
        "--document-inventory",
        str(project / "OHR-Bench/data/retrieval_base/gt"),
        "--out-root",
        str(out_root),
        "--smoke-doc",
        "academic/demo",
        "--execute",
    ]
    monkeypatch.chdir(project)

    def interrupted_execute(**kwargs):
        raise SystemExit(143)

    monkeypatch.setattr(cli, "execute_document_preparation", interrupted_execute)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(args)
    assert excinfo.value.code == 143
    assert checkpoint_path.is_file()
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["attempts"] == 1
    assert checkpoint["cache_bytes_before"] is not None
    assert checkpoint["resumed"] is False
    records = checkpoint["attempt_records"]
    assert len(records) == 1
    assert records[0]["state"] == "interrupted"
    assert records[0]["exit_code"] == 143
    assert records[0]["elapsed_sec"] > 0
    assert records[0]["elapsed_source"] == "monotonic"

    def successful_execute(**kwargs):
        pdf_out = out_root / "pdfs/academic/demo.pdf"
        pdf_out.parent.mkdir(parents=True, exist_ok=True)
        pdf_out.write_bytes(b"%PDF")
        return SimpleNamespace(
            doc_name="academic/demo",
            page_ids=[0],
            pdf_path=pdf_out,
            pdf_sha256="a" * 64,
            got_ocr_repository="stepfun-ai/GOT-OCR-2.0-hf",
            got_ocr_revision="d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
            device="cpu",
            metrics={
                "docling_runtime_sec": 1.0,
                "render_runtime_sec": 1.0,
                "got_ocr_runtime_sec_total": 1.0,
                "per_page_got_ocr_runtime_sec": {"0": 1.0},
                "peak_rss_bytes": 1,
                "storage": {"png_bytes_total": 1, "ocr_bytes_total": 1, "per_page_bytes": {}},
            },
            outputs={"pdf": "out/pdfs/academic/demo.pdf"},
        )

    monkeypatch.setattr(cli, "execute_document_preparation", successful_execute)
    cli.main(args)
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["attempts"] == 2
    assert checkpoint["resumed"] is True
    calibration = checkpoint["calibration"]
    assert calibration["attempts"] == 2
    assert calibration["attempts_elapsed_sec"] > 0
    assert calibration["runtime_sec_total_wall"] == calibration["attempts_elapsed_sec"]
    assert calibration["last_attempt_sec"] > 0
    assert calibration["cache_bytes_before"] == checkpoint["cache_bytes_before"]
    assert calibration["timing_complete"] is True
    assert [record["state"] for record in checkpoint["attempt_records"]] == ["interrupted", "completed"]
    assert all(record["elapsed_sec"] > 0 for record in checkpoint["attempt_records"])


def test_plan_imports_do_not_load_inference_runtime() -> None:
    """Plan-only imports must not pull Torch, Transformers, or ColPali."""
    repo_root = Path(__file__).resolve().parents[1]
    probe = r"""
import sys
from pathlib import Path
root = Path(%r)
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts" / "data"))
from faar.asset_preparation import load_locked_got_ocr, plan_document_work, page_image_name
import prepare_benchmark_assets
banned = {"torch", "transformers", "colpali_engine"}
loaded = sorted({name.split(".")[0] for name in sys.modules if name.split(".")[0] in banned})
assert not loaded, loaded
assert page_image_name("doc", 0).endswith("_page_0.png")
print("PLAN_IMPORTS_CLEAN")
""" % (str(repo_root),)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "PLAN_IMPORTS_CLEAN" in completed.stdout


LOCK = {
    "models": {
        "got_ocr": {
            "repository": "stepfun-ai/GOT-OCR-2.0-hf",
            "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
        },
        "docling": {
            "repository": "docling-project/docling-models",
            "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
        },
    }
}


def _calibration_project(tmp_path: Path, docs: tuple[str, ...] = ("academic/demo",)) -> Path:
    project = tmp_path
    (project / "config/datasets").mkdir(parents=True)
    (project / "config/model_revisions.json").write_text(json.dumps(LOCK), encoding="utf-8")
    (project / "OHR-Bench/data").mkdir(parents=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(
        json.dumps([{"ID": f"e{i}", "doc_name": doc} for i, doc in enumerate(docs)])
    )
    (project / "config/datasets/ohr_split.json").write_text(json.dumps({"splits": {"val": [f"e{i}" for i in range(len(docs))]}}))
    for doc in docs:
        gt = project / "OHR-Bench/data/retrieval_base/gt" / Path(doc).parent
        gt.mkdir(parents=True, exist_ok=True)
        (gt / f"{Path(doc).name}.json").write_text(json.dumps([{"text": "x", "page_idx": 0}]))
        pdf = project / "pdfs" / f"{doc}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"%PDF-{doc}".encode())
    return project


def _cli_args(project: Path, doc: str, *, execute: bool = False, extra: list[str] | None = None) -> list[str]:
    args = [
        "--project-root",
        str(project),
        "--dataset",
        "ohrbench",
        "--pdf-root",
        str(project / "pdfs"),
        "--document-inventory",
        str(project / "OHR-Bench/data/retrieval_base/gt"),
        "--out-root",
        str(project / "out"),
        "--smoke-doc",
        doc,
        "--checkpoint",
        str(project / "out/prepare_checkpoint.json"),
    ]
    if execute:
        args.append("--execute")
    if extra:
        args.extend(extra)
    return args


def _fake_result(project: Path, doc: str) -> SimpleNamespace:
    pdf_out = project / f"out/pdfs/{doc}.pdf"
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.write_bytes(b"%PDF")
    return SimpleNamespace(
        doc_name=doc,
        page_ids=[0],
        pdf_path=pdf_out,
        pdf_sha256="a" * 64,
        got_ocr_repository="stepfun-ai/GOT-OCR-2.0-hf",
        got_ocr_revision="d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
        device="cpu",
        metrics={
            "docling_runtime_sec": 1.0,
            "render_runtime_sec": 1.0,
            "got_ocr_runtime_sec_total": 1.0,
            "per_page_got_ocr_runtime_sec": {"0": 1.0},
            "peak_rss_bytes": 1,
            "storage": {"png_bytes_total": 1, "ocr_bytes_total": 1, "per_page_bytes": {}},
        },
        outputs={"pdf": f"out/pdfs/{doc}.pdf"},
    )


def test_checkpoint_rejects_different_smoke_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path, ("academic/demo", "manual/other"))
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    checkpoint_path = project / "out/prepare_checkpoint.json"
    before = checkpoint_path.read_bytes()
    with pytest.raises(SystemExit, match="identity mismatch"):
        cli.main(_cli_args(project, "manual/other", execute=True))
    assert checkpoint_path.read_bytes() == before


def test_checkpoint_rejects_lock_and_page_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path)
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    checkpoint_path = project / "out/prepare_checkpoint.json"
    before = checkpoint_path.read_bytes()
    with pytest.raises(SystemExit, match="identity mismatch"):
        cli.main(_cli_args(project, "academic/demo", execute=True, extra=["--page-ids", "0,1"]))
    assert checkpoint_path.read_bytes() == before
    (project / "pdfs/academic/demo.pdf").write_bytes(b"%PDF-changed")
    with pytest.raises(SystemExit, match="identity mismatch"):
        cli.main(_cli_args(project, "academic/demo", execute=True))
    assert checkpoint_path.read_bytes() == before
    (project / "pdfs/academic/demo.pdf").write_bytes(b"%PDF-academic/demo")
    lock = json.loads((project / "config/model_revisions.json").read_text())
    lock["models"]["got_ocr"]["revision"] = "c" * 40
    (project / "config/model_revisions.json").write_text(json.dumps(lock))
    with pytest.raises(SystemExit, match="identity mismatch"):
        cli.main(_cli_args(project, "academic/demo", execute=True))
    assert checkpoint_path.read_bytes() == before
    lock["models"]["got_ocr"]["revision"] = "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
    lock["models"]["docling"]["revision"] = "b" * 40
    (project / "config/model_revisions.json").write_text(json.dumps(lock))
    with pytest.raises(SystemExit, match="identity mismatch"):
        cli.main(_cli_args(project, "academic/demo", execute=True))
    assert checkpoint_path.read_bytes() == before


def test_identical_resume_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path)
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    checkpoint = json.loads((project / "out/prepare_checkpoint.json").read_text())
    assert checkpoint["attempts"] == 2
    assert checkpoint["timing_complete"] is True


def test_unclean_running_attempt_is_detected_and_projection_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path)
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    checkpoint_path = project / "out/prepare_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["attempt_records"].append(
        {
            "id": 99,
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "state": "running",
            "elapsed_sec": None,
            "elapsed_source": None,
        }
    )
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    cli.main(_cli_args(project, "academic/demo", execute=True))
    checkpoint = json.loads(checkpoint_path.read_text())
    unclean = [record for record in checkpoint["attempt_records"] if record["state"] == "unclean"]
    assert unclean
    assert unclean[0]["elapsed_sec"] is None
    assert checkpoint["timing_complete"] is False
    assert checkpoint["calibration"]["runtime_sec_total_wall"] is None

    report_path = Path(__file__).resolve().parents[1] / "cluster/calibration_report.py"
    spec = spec_from_file_location("calibration_report_unclean", report_path)
    assert spec and spec.loader
    report = module_from_spec(spec)
    spec.loader.exec_module(report)
    summary = report.build_summary(checkpoint_path=checkpoint_path, project_root=project)
    assert summary["runtime"]["timing_complete"] is False
    assert summary["runtime"]["total_wall_sec"] is None
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
    cli.main(_cli_args(project, "academic/demo", extra=["--scheduler-elapsed-sec", "12.5"]))
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["timing_complete"] is True
    assert any(record.get("elapsed_source") == "scheduler" for record in checkpoint["attempt_records"])
    assert checkpoint["calibration"]["runtime_sec_total_wall"] == pytest.approx(
        checkpoint["calibration"]["measured_wall_sec"]
    )
    rebuilt = report.build_summary(checkpoint_path=checkpoint_path, project_root=project)
    assert rebuilt["runtime"]["timing_complete"] is True
    projection = report.project_preparation(
        rebuilt,
        headroom_fraction=0.25,
        walltime_hours_per_shard=24.0,
        val_documents=10,
        val_pages=100,
        test_documents=10,
        test_pages=100,
    )
    assert projection["timing"]["complete"] is True
    assert "scheduler_elapsed_sec" not in projection["timing"]


def test_scheduler_elapsed_corrects_one_unclean_attempt_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path)
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    checkpoint_path = project / "out/prepare_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["attempt_records"].extend(
        [
            {
                "id": 98,
                "state": "unclean",
                "elapsed_sec": None,
                "elapsed_source": None,
            },
            {
                "id": 99,
                "state": "unclean",
                "elapsed_sec": None,
                "elapsed_source": None,
            },
        ]
    )
    checkpoint["timing_complete"] = False
    checkpoint["calibration"]["timing_complete"] = False
    checkpoint["calibration"]["runtime_sec_total_wall"] = None
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    cli.main(_cli_args(project, "academic/demo", extra=["--scheduler-elapsed-sec", "8.0"]))
    after_one = json.loads(checkpoint_path.read_text())
    unresolved = [
        record
        for record in after_one["attempt_records"]
        if record.get("state") == "unclean" and not isinstance(record.get("elapsed_sec"), (int, float))
    ]
    assert len(unresolved) == 1
    assert after_one["timing_complete"] is False
    cli.main(_cli_args(project, "academic/demo", extra=["--scheduler-elapsed-sec", "9.0"]))
    after_two = json.loads(checkpoint_path.read_text())
    assert after_two["timing_complete"] is True
    scheduler_records = [
        record for record in after_two["attempt_records"] if record.get("elapsed_source") == "scheduler"
    ]
    assert {record["elapsed_sec"] for record in scheduler_records} == {8.0, 9.0}


def test_scheduler_elapsed_fails_when_no_unresolved_unclean_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path)
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    cli.main(_cli_args(project, "academic/demo", execute=True))
    with pytest.raises(SystemExit, match="no unclean attempt is missing elapsed time"):
        cli.main(_cli_args(project, "academic/demo", extra=["--scheduler-elapsed-sec", "12.5"]))


def test_fast_attempt_stores_positive_raw_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_benchmark_assets as cli

    project = _calibration_project(tmp_path)
    monkeypatch.setattr(cli, "execute_document_preparation", lambda **kwargs: _fake_result(project, kwargs["doc_rel"]))
    for _ in range(25):
        if (project / "out/prepare_checkpoint.json").exists():
            (project / "out/prepare_checkpoint.json").unlink()
        cli.main(_cli_args(project, "academic/demo", execute=True))
        checkpoint = json.loads((project / "out/prepare_checkpoint.json").read_text())
        elapsed = checkpoint["attempt_records"][0]["elapsed_sec"]
        assert elapsed > 0
        assert elapsed == checkpoint["calibration"]["last_attempt_sec"]


def test_relative_inventory_from_outside_working_directory(tmp_path: Path) -> None:
    project = _calibration_project(tmp_path)
    outsider = tmp_path / "cwd"
    outsider.mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts/data/prepare_benchmark_assets.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(project),
            "--dataset",
            "ohrbench",
            "--pdf-root",
            "pdfs",
            "--document-inventory",
            "OHR-Bench/data/retrieval_base/gt",
            "--out-root",
            "out",
            "--smoke-doc",
            "academic/demo",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=outsider,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json.loads(completed.stdout)["mode"] == "plan_only"


def test_docling_lock_change_regenerates_audit_and_records_new_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faar.asset_preparation as prep

    project = _calibration_project(tmp_path)
    calls: list[str] = []

    def fake_docling(pdf_path: Path, output_path: Path, repository: str = "", revision: str = "", **kwargs) -> Path:
        calls.append(revision)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"# audit {revision}\n", encoding="utf-8")
        return output_path

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_id]))
        return {"pdf_page_count": len(page_ids), "render_runtime_sec": 0.01, "render_scale": scale}

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    monkeypatch.setitem(
        sys.modules,
        "pypdfium2",
        __import__("types").SimpleNamespace(
            PdfDocument=type(
                "FakePageCountDocument",
                (),
                {
                    "__init__": lambda self, _path: None,
                    "__len__": lambda self: 1,
                    "close": lambda self: None,
                },
            )
        ),
    )
    execute_document_preparation(
        project_root=project,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=project / "prepared",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=lambda *_args, **_kwargs: "text",
        export_docling_fn=fake_docling,
    )
    assert calls == ["2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4"]
    lock = json.loads((project / "config/model_revisions.json").read_text())
    lock["models"]["docling"]["revision"] = "b" * 40
    (project / "config/model_revisions.json").write_text(json.dumps(lock), encoding="utf-8")
    execute_document_preparation(
        project_root=project,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=project / "prepared",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=lambda *_args, **_kwargs: "text",
        export_docling_fn=fake_docling,
    )
    assert calls == ["2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4", "b" * 40]
    provenance = json.loads((project / "prepared/provenance/academic/demo.json").read_text())
    assert provenance["docling_audit"]["revision"] == "b" * 40
    assert provenance["docling_models"]["revision"] == "b" * 40


def test_docling_provenance_survives_termination_after_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faar.asset_preparation as prep

    project = _calibration_project(tmp_path)
    provenance_path = project / "prepared/provenance/academic/demo.json"
    docling_calls = {"count": 0}

    def fake_docling(pdf_path: Path, output_path: Path, **kwargs) -> Path:
        docling_calls["count"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# audit\n", encoding="utf-8")
        return output_path

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
        return {"pdf_page_count": 1, "render_runtime_sec": 0.0, "render_scale": scale}

    def terminate_after_docling_publish() -> None:
        if provenance_path.is_file():
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            if payload.get("docling_audit", {}).get("revision"):
                raise SystemExit(143)

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    monkeypatch.setattr(prep, "check_termination", terminate_after_docling_publish)
    with pytest.raises(SystemExit) as excinfo:
        execute_document_preparation(
            project_root=project,
            doc_rel="academic/demo",
            page_ids=[0],
            out_root=project / "prepared",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=lambda *_args, **_kwargs: "text",
            export_docling_fn=fake_docling,
        )
    assert excinfo.value.code == 143
    assert (project / "prepared/docling/academic/demo.docling.md").is_file()
    payload = json.loads(provenance_path.read_text())
    assert payload["docling_audit"]["revision"] == "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4"
    monkeypatch.setattr(prep, "check_termination", lambda: None)
    execute_document_preparation(
        project_root=project,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=project / "prepared",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=lambda *_args, **_kwargs: "text",
        export_docling_fn=fake_docling,
    )
    assert docling_calls["count"] == 1
