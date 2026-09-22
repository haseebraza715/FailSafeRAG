from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PREPARE_PATH = Path(__file__).resolve().parents[1] / "cluster/prepare_assets.py"
PREPARE_SPEC = spec_from_file_location("prepare_assets", PREPARE_PATH)
assert PREPARE_SPEC and PREPARE_SPEC.loader
prepare = module_from_spec(PREPARE_SPEC)
sys.modules["prepare_assets"] = prepare
PREPARE_SPEC.loader.exec_module(prepare)

DOCS = ["academic/doc_a", "academic/doc_b", "manual/doc_c"]
PAGES = [0, 1]

MODEL_LOCK = {
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


def write_lock(project: Path) -> None:
    config = project / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "model_revisions.json").write_text(json.dumps(MODEL_LOCK), encoding="utf-8")
    (config / "split_checksums.json").write_text(
        json.dumps(
            {
                "split_sha256": prepare.sha256_file(project / "config/datasets/ohr_split.json"),
                "qas_v2_sha256": prepare.sha256_file(project / "OHR-Bench/data/qas_v2.json"),
            }
        ),
        encoding="utf-8",
    )


def build_project(tmp_path: Path, docs: list[str] = DOCS) -> Path:
    project = tmp_path
    (project / "OHR-Bench/data").mkdir(parents=True, exist_ok=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(
        json.dumps([{"ID": f"e{i}", "doc_name": doc} for i, doc in enumerate(docs)])
    )
    (project / "config/datasets").mkdir(parents=True, exist_ok=True)
    (project / "config/datasets/ohr_split.json").write_text(
        json.dumps(
            {
                "splits": {
                    "val": [f"e{i}" for i in range(len(docs))],
                    "test": [f"e{i}" for i in range(len(docs))],
                }
            }
        )
    )
    gt = project / "OHR-Bench/data/retrieval_base/gt"
    for doc in docs:
        inventory = gt / f"{doc}.json"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(json.dumps([{"text": "x", "page_idx": page} for page in PAGES]))
        pdf = project / "pdfs" / f"{doc}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"%PDF-1.4\n{doc}\n".encode())
    write_lock(project)
    return project


def install_fakes(monkeypatch: pytest.MonkeyPatch, ocr_calls: list[str] | None = None) -> None:
    import faar.asset_preparation as prep

    class FakePageCountDocument:
        def __init__(self, _path: str) -> None:
            self.page_count = len(PAGES)

        def __len__(self) -> int:
            return self.page_count

        def close(self) -> None:
            pass

    def fake_docling(pdf_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# audit\n", encoding="utf-8")
        return output_path

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_id]))
        return {"pdf_page_count": len(page_ids), "render_runtime_sec": 0.01, "render_scale": scale}

    def fake_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        if ocr_calls is not None:
            ocr_calls.append(image_path.name)
        return f"OCR for {image_path.name}"

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    monkeypatch.setitem(
        sys.modules,
        "pypdfium2",
        __import__("types").SimpleNamespace(PdfDocument=FakePageCountDocument),
    )
    return fake_ocr, fake_docling


def run_shard(project: Path, *args, monkeypatch=None, ocr_calls=None, **kwargs):
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    return prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
        **kwargs,
    )


def test_deterministic_sharding_and_full_coverage() -> None:
    docs = sorted({f"{category}/{name}" for category in ("academic", "manual", "paper") for name in ("a", "b")})
    shards = [prepare.select_shard(docs, index, 4) for index in range(4)]
    assert all(not set(shard) & set(other) for shard in shards for other in shards if shard is not other)
    assert sorted({doc for shard in shards for doc in shard}) == sorted(docs)
    first = prepare.select_shard(docs, 1, 4)
    for _ in range(3):
        assert prepare.select_shard(docs, 1, 4) == first


def test_select_shard_rejects_invalid_metadata() -> None:
    docs = sorted({f"{category}/{name}" for category in ("academic", "manual", "paper") for name in ("a", "b")})
    for bad_index in (-1, 4, 7):
        with pytest.raises(ValueError, match="out of range"):
            prepare.select_shard(docs, bad_index, 4)
    with pytest.raises(ValueError, match="num_shards"):
        prepare.select_shard(docs, 0, 0)
    with pytest.raises(ValueError, match="num_shards"):
        prepare.select_shard(docs, 0, True)
    with pytest.raises(ValueError, match="shard_index"):
        prepare.select_shard(docs, False, 4)
    with pytest.raises(ValueError, match="duplicates"):
        prepare.select_shard([doc for doc in DOCS] + [DOCS[0]], 0, 1)


def test_bounds_apply_deterministically() -> None:
    page_ids_by_doc = {doc: PAGES for doc in DOCS}
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_documents=2) == (DOCS[:2], True)
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_documents=10) == (DOCS, False)
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_pages=3) == (DOCS[:1], True)
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_pages=6) == (DOCS, False)
    assert prepare.bound_documents(DOCS, page_ids_by_doc) == (DOCS, False)


def test_run_shard_prepares_all_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls)
    assert summary["failed"] == []
    assert summary["pages_completed"] == len(DOCS) * len(PAGES)
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert set(checkpoint["completed"]) == set(DOCS)
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_completed"] == DOCS
    assert manifest["documents_failed"] == {}
    assert manifest["split"] == "val"
    assert len(ocr_calls) == len(DOCS) * len(PAGES)
    for doc in DOCS:
        provenance = json.loads((project / "out/provenance" / f"{doc}.json").read_text())
        assert provenance["pdf_sha256"] == manifest.get("pdf_sha256") or provenance["pages"]


def test_rerun_is_idempotent_and_skips_valid_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls)
    first = json.loads((project / "out/checkpoint.json").read_text())
    ocr_calls.clear()
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == DOCS
    assert ocr_calls == []
    second = json.loads((project / "out/checkpoint.json").read_text())
    assert first["completed"] == second["completed"]
    assert first["updated_at_utc"] != second["updated_at_utc"]


def test_rerun_without_resume_refuses_existing_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit, match="pass --resume"):
        run_shard(project, monkeypatch=monkeypatch)


def test_resume_after_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    calls = {"count": 0}
    original_check = prepare.check_termination

    def interrupting_check() -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise SystemExit(143)

    monkeypatch.setattr(prepare, "check_termination", interrupting_check)
    with pytest.raises(SystemExit) as excinfo:
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=fake_ocr,
            export_docling_fn=fake_docling,
        )
    assert excinfo.value.code == 143
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert set(checkpoint["completed"]) == {DOCS[0]}
    assert not (project / "out/shard_manifest.json").is_file()
    monkeypatch.setattr(prepare, "check_termination", original_check)
    ocr_calls.clear()
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["failed"] == []
    assert summary["skipped"] == [DOCS[0]]
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["resumed"] is True
    assert len(ocr_calls) == len(DOCS[1:]) * len(PAGES)


def test_corrupt_checkpoint_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    (project / "out").mkdir(parents=True)
    (project / "out/checkpoint.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable JSON"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)
    (project / "out/checkpoint.json").write_text(json.dumps({"completed": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid 'completed' section"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)
    (project / "out/checkpoint.json").write_text(
        json.dumps({"completed": {}, "split": "test"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Refusing to resume"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)


def test_recoverable_document_error_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)

    def failing_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        if "doc_b" in str(image_path):
            raise ValueError("simulated recoverable OCR failure")
        return "text"

    fake_docling = install_fakes(monkeypatch)[1]
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=failing_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary["failed"] == ["academic/doc_b"]
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert set(checkpoint["completed"]) == {"academic/doc_a", "manual/doc_c"}
    assert "simulated recoverable OCR failure" in checkpoint["failed"]["academic/doc_b"]["error"]
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_completed"] == ["academic/doc_a", "manual/doc_c"]
    assert manifest["documents_failed"]["academic/doc_b"]["error"].startswith("simulated")


def test_fatal_memory_error_aborts_with_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_MAX_RSS_GB", "0.000001")
    fake_ocr, fake_docling = install_fakes(monkeypatch)
    with pytest.raises(MemoryError):
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=fake_ocr,
            export_docling_fn=fake_docling,
        )
    assert not (project / "out/shard_manifest.json").is_file()
    monkeypatch.delenv("FAAR_MAX_RSS_GB")
    monkeypatch.setenv("FAAR_MAX_RSS_GB", "0.000001")
    with pytest.raises(MemoryError):
        run_shard(project, monkeypatch=monkeypatch)


def test_empty_shard_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    fake_ocr, fake_docling = install_fakes(monkeypatch)
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=[],
        page_ids_by_doc={},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        shard_index=2,
        num_shards=3,
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary["failed"] == []
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_assigned"] == []
    assert manifest["documents_completed"] == []
    assert manifest["shard_index"] == 2 and manifest["num_shards"] == 3


def test_merge_shard_manifests_validates_coverage(tmp_path: Path) -> None:
    documents = [["academic/doc_a"], ["academic/doc_b"], ["manual/doc_c"]]
    paths: list[Path] = []
    for index, docs in enumerate(documents):
        payload = {
            "schema_version": 1,
            "kind": "ohr-asset-shard-manifest",
            "dataset": "ohrbench",
            "split": "val",
            "shard_index": index,
            "num_shards": 3,
            "created_at_utc": "t",
            "finished_at_utc": "t",
            "resumed": False,
            "bounds": {"max_documents": None, "max_pages": None, "bounds_hit": False},
            "documents_assigned": docs,
            "documents_completed": docs,
            "documents_failed": {},
            "retried_documents": [],
            "pages_assigned": 2,
            "pages_completed": 2,
            "runtime_sec": 1.0,
            "checkpoint": "c",
            "out_root": "o",
        }
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    merged = prepare.merge_shard_manifests(paths, expected_documents=DOCS)
    assert merged["documents"] == DOCS
    assert merged["pages_completed"] == 6
    with pytest.raises(SystemExit, match="missing shard indices"):
        prepare.merge_shard_manifests(paths[:2], expected_documents=DOCS)
    with pytest.raises(SystemExit, match="declared by"):
        prepare.merge_shard_manifests([paths[0], paths[1], paths[2], paths[2]], expected_documents=DOCS)


def test_merge_rejects_uncompleted_documents(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "ohr-asset-shard-manifest",
        "dataset": "ohrbench",
        "split": "val",
        "shard_index": 0,
        "num_shards": 1,
        "created_at_utc": "t",
        "finished_at_utc": "t",
        "resumed": False,
        "bounds": {"max_documents": None, "max_pages": None, "bounds_hit": False},
        "documents_assigned": DOCS,
        "documents_completed": DOCS[:1],
        "documents_failed": {"manual/doc_c": {"error": "x"}},
        "retried_documents": [],
        "pages_assigned": 6,
        "pages_completed": 2,
        "runtime_sec": 1.0,
        "checkpoint": "c",
        "out_root": "o",
    }
    path = tmp_path / "shard0.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="uncompleted documents"):
        prepare.merge_shard_manifests([path], expected_documents=DOCS)


def test_path_traversal_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        prepare._safe_output_path("../escape", Path("/tmp/root"))
    with pytest.raises(ValueError, match="non-empty"):
        prepare._safe_output_path("", Path("/tmp/root"))


def test_corrupt_checkpoint_outputs_recovered_by_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    evil_entry = dict(json.loads((project / "out/checkpoint.json").read_text()))
    evil_entry["completed"]["academic/doc_a"]["outputs"]["images"][0] = "../escaped.png"
    (project / "out/checkpoint.json").write_text(json.dumps(evil_entry))
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["failed"] == []
    assert ocr_calls == []
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_completed"] == DOCS
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert checkpoint["completed"]["academic/doc_a"]["outputs"]["images"][0] != "../escaped.png"


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    out_root = project / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--split",
            "val",
            "--shard-index",
            "1",
            "--num-shards",
            "2",
            "--out-root",
            str(out_root),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["documents_assigned"] == ["manual/doc_c"]
    assert payload["shard_label"] == "shard2of2"
    assert not out_root.exists()


def test_cli_dry_run_applies_bounds(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--max-documents",
            "1",
            "--max-pages",
            "2",
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["documents_assigned"] == DOCS[:1]
    assert payload["bounds"]["bounds_hit"] is True


def test_cli_rejects_train_split_and_bad_shards(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    for extra in (["--split", "train"], ["--num-shards", "0"], ["--shard-index", "5", "--num-shards", "2"]):
        completed = subprocess.run(
            [
                sys.executable,
                str(PREPARE_PATH),
                "--project-root",
                str(project),
                "--out-root",
                str(project / "out"),
                "--dry-run",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=project,
        )
        assert completed.returncode != 0, extra


def test_cli_rejects_missing_split_lock(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    (project / "config/split_checksums.json").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode != 0
    assert "split checksum lock" in completed.stderr


def test_page_level_ocr_resume_after_mid_document_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    fake_docling = install_fakes(monkeypatch)[1]
    ocr_calls = {"count": 0}

    def interrupting_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        ocr_calls["count"] += 1
        if ocr_calls["count"] >= 2:
            raise SystemExit(143)
        return "text"

    with pytest.raises(SystemExit) as excinfo:
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=interrupting_ocr,
            export_docling_fn=fake_docling,
        )
    assert excinfo.value.code == 143
    provenance = json.loads((project / "out/provenance/academic/doc_a.json").read_text())
    assert "ocr_sha256" in provenance["pages"]["0"]
    assert "ocr_sha256" not in provenance["pages"]["1"]
    assert not (project / "out/ocr/academic/doc_a_page_1.txt").is_file()
    calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=calls, resume=True)
    assert summary["failed"] == []
    assert calls.count("doc_a_page_0.png") == 0
    assert calls.count("doc_a_page_1.png") == 1
    assert len(calls) == 5
    assert (project / "out/ocr/academic/doc_a_page_1.txt").read_text().startswith("OCR for")


def test_sigterm_saves_checkpoint_and_exits_143(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    fake_docling = install_fakes(monkeypatch)[1]
    import signal as signal_module

    sent = {"done": False}

    def sigterm_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        if not sent["done"]:
            sent["done"] = True
            signal_module.raise_signal(signal_module.SIGTERM)
        return "text"

    prepare.install_graceful_termination_handler()
    previous_term = signal_module.getsignal(signal_module.SIGTERM)
    previous_int = signal_module.getsignal(signal_module.SIGINT)
    try:
        with pytest.raises(SystemExit) as excinfo:
            prepare.run_shard(
                project_root=project,
                split="val",
                doc_names=DOCS,
                page_ids_by_doc={doc: PAGES for doc in DOCS},
                out_root=project / "out",
                checkpoint_path=project / "out/checkpoint.json",
                manifest_path=project / "out/shard_manifest.json",
                pdf_root=project / "pdfs",
                extract_got_ocr_fn=sigterm_ocr,
                export_docling_fn=fake_docling,
            )
    finally:
        signal_module.signal(signal_module.SIGTERM, previous_term)
        signal_module.signal(signal_module.SIGINT, previous_int)
        import faar.operations as ops

        ops._TERMINATION_SIGNAL = None
    assert excinfo.value.code == 143
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert "academic/doc_a" not in checkpoint["completed"]
    assert not (project / "out/shard_manifest.json").is_file()


def test_end_to_end_prepare_interrupt_resume_finish_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise prepare -> checkpoint -> interrupt -> resume -> finish shard
    -> merge -> validate provenance for a complete two-shard split."""
    project = build_project(tmp_path)
    shard0_docs = prepare.select_shard(DOCS, 0, 2)
    shard1_docs = prepare.select_shard(DOCS, 1, 2)
    assert shard0_docs == DOCS[:2] and shard1_docs == DOCS[2:]
    calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, calls)
    original_check = prepare.check_termination
    interrupts = {"count": 0}

    def interrupting_check() -> None:
        interrupts["count"] += 1
        if interrupts["count"] >= 2:
            raise SystemExit(143)

    monkeypatch.setattr(prepare, "check_termination", interrupting_check)
    with pytest.raises(SystemExit) as excinfo:
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=shard0_docs,
            page_ids_by_doc={doc: PAGES for doc in shard0_docs},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint_shard1of2.json",
            manifest_path=project / "out/shard_manifest_shard1of2.json",
            shard_index=0,
            num_shards=2,
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=fake_ocr,
            export_docling_fn=fake_docling,
        )
    assert excinfo.value.code == 143
    checkpoint = json.loads((project / "out/checkpoint_shard1of2.json").read_text())
    assert set(checkpoint["completed"]) == {DOCS[0]}
    assert not (project / "out/shard_manifest_shard1of2.json").is_file()

    monkeypatch.setattr(prepare, "check_termination", original_check)
    calls.clear()
    summary0 = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=shard0_docs,
        page_ids_by_doc={doc: PAGES for doc in shard0_docs},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint_shard1of2.json",
        manifest_path=project / "out/shard_manifest_shard1of2.json",
        shard_index=0,
        num_shards=2,
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
        resume=True,
    )
    assert summary0["failed"] == []
    assert summary0["skipped"] == [DOCS[0]]

    summary1 = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=shard1_docs,
        page_ids_by_doc={doc: PAGES for doc in shard1_docs},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint_shard2of2.json",
        manifest_path=project / "out/shard_manifest_shard2of2.json",
        shard_index=1,
        num_shards=2,
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary1["failed"] == []

    merged = prepare.merge_shard_manifests(
        [project / "out/shard_manifest_shard1of2.json", project / "out/shard_manifest_shard2of2.json"],
        expected_documents=DOCS,
    )
    assert merged["documents"] == DOCS
    assert merged["pages_completed"] == len(DOCS) * len(PAGES)

    for doc in DOCS:
        provenance = json.loads((project / "out/provenance" / f"{doc}.json").read_text())
        assert provenance["doc_name"] == doc
        assert provenance["pdf_sha256"]
        assert provenance["got_ocr_revision"] == "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
        assert provenance["docling_models"] == {
            "repository": "docling-project/docling-models",
            "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
        }
        for page_id in PAGES:
            page = provenance["pages"][str(page_id)]
            assert page["ocr_sha256"]
            assert page["png_sha256"]
            assert (project / page["ocr_text_path"]).is_file()
            assert (project / page["image_path"]).is_file()
    assert not list((project / "out").glob("*.tmp"))
    assert not list((project / "out").glob(".*.tmp"))


def test_input_change_never_silently_reuses_stale_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    before = json.loads((project / "out/provenance/academic/doc_a.json").read_text())
    (project / "pdfs/academic/doc_a.pdf").write_bytes(b"%PDF-1.4\nCHANGED\n")
    calls: list[str] = []
    run_shard(project, monkeypatch=monkeypatch, ocr_calls=calls, resume=True)
    after = json.loads((project / "out/provenance/academic/doc_a.json").read_text())
    assert before["pdf_sha256"] != after["pdf_sha256"]
    assert len(calls) == len(PAGES)
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert checkpoint["completed"]["academic/doc_a"]["pdf_sha256"] == after["pdf_sha256"]


def test_cuda_oom_is_fatal_and_saves_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)

    def oom_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

    fake_docling = install_fakes(monkeypatch)[1]
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=oom_ocr,
            export_docling_fn=fake_docling,
        )
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert "academic/doc_a" not in checkpoint["completed"]
    assert not (project / "out/shard_manifest.json").is_file()


def test_cli_fails_closed_on_missing_inventory(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    (project / "OHR-Bench/data/retrieval_base/gt/academic/doc_a.json").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode != 0
    assert "No complete inventory" in completed.stderr


def test_stale_page_inventory_is_not_silently_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    inventory = project / "OHR-Bench/data/retrieval_base/gt/academic/doc_a.json"
    inventory.write_text(json.dumps([{"text": "x", "page_idx": 0}]))
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={**{doc: PAGES for doc in DOCS}, "academic/doc_a": [0]},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
        resume=True,
    )
    assert "academic/doc_a" not in summary["skipped"]
    assert "academic/doc_a" in summary["failed"]
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert "academic/doc_a" not in checkpoint["completed"]


def test_merge_rejects_out_of_range_shard_indices(tmp_path: Path) -> None:
    documents = [["academic/doc_a"], ["academic/doc_b"], ["manual/doc_c"]]
    paths: list[Path] = []
    for index, docs in enumerate(documents):
        payload = {
            "schema_version": 1,
            "kind": "ohr-asset-shard-manifest",
            "dataset": "ohrbench",
            "split": "val",
            "shard_index": index,
            "num_shards": 2,
            "created_at_utc": "t",
            "finished_at_utc": "t",
            "resumed": False,
            "bounds": {"max_documents": None, "max_pages": None, "bounds_hit": False},
            "documents_assigned": docs,
            "documents_completed": docs,
            "documents_failed": {},
            "retried_documents": [],
            "pages_assigned": 2,
            "pages_completed": 2,
            "runtime_sec": 1.0,
            "checkpoint": "c",
            "out_root": "o",
        }
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    with pytest.raises(SystemExit, match="expected an index in"):
        prepare.merge_shard_manifests(paths, expected_documents=DOCS)


def test_timing_from_earlier_attempts_is_preserved_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    fake_docling = install_fakes(monkeypatch)[1]
    ocr_calls = {"count": 0}

    def interrupt_after_first_page(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        ocr_calls["count"] += 1
        if ocr_calls["count"] >= 2:
            raise SystemExit(143)
        return "text"

    with pytest.raises(SystemExit) as excinfo:
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=interrupt_after_first_page,
            export_docling_fn=fake_docling,
        )
    assert excinfo.value.code == 143
    calls: list[str] = []
    fake_ocr2, fake_docling2 = install_fakes(monkeypatch, calls)
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr2,
        export_docling_fn=fake_docling2,
        resume=True,
    )
    assert summary["failed"] == []
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    per_page = checkpoint["completed"]["academic/doc_a"]["metrics"]["per_page_got_ocr_runtime_sec"]
    assert set(per_page) == {"0", "1"}
    total = checkpoint["completed"]["academic/doc_a"]["metrics"]["got_ocr_runtime_sec_total"]
    assert total == pytest.approx(sum(per_page.values()), abs=1e-6)


def test_cli_honors_env_dataset_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    pdf_root = tmp_path / "alt-pdfs"
    pdf_root.mkdir()
    for doc in DOCS:
        pdf_root.mkdir(parents=True, exist_ok=True)
        source = pdf_root / f"{doc}.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((project / "pdfs" / f"{doc}.pdf").read_bytes())
    monkeypatch.setenv("FAAR_PDF_ROOT", str(pdf_root))
    monkeypatch.setenv("FAAR_DOCUMENT_INVENTORY", str(project / "OHR-Bench/data/retrieval_base/gt"))
    monkeypatch.chdir(project)
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    monkeypatch.setattr(prepare, "export_docling_markdown", fake_docling)
    monkeypatch.setattr("faar.ocr.extract_got_ocr", fake_ocr)
    code = prepare.main(
        [
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
        ]
    )
    assert code == 0
    assert len(ocr_calls) == len(DOCS) * len(PAGES)
    checkpoint = json.loads((project / "out/checkpoint_shard1of1.json").read_text())
    assert set(checkpoint["completed"]) == set(DOCS)


def test_cli_rejects_missing_env_pdf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_PDF_ROOT", str(tmp_path / "does-not-exist"))
    monkeypatch.chdir(project)
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode != 0
    assert "FAAR_PDF_ROOT" in completed.stderr


def test_paths_with_spaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path / "my project with spaces")
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out space",
        checkpoint_path=project / "out space/checkpoint.json",
        manifest_path=project / "out space/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary["failed"] == []
    assert len(ocr_calls) == len(DOCS) * len(PAGES)


def test_split_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0
    with (project / "config/datasets/ohr_split.json").open("a") as handle:
        handle.write(" ")
    tampered = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert tampered.returncode != 0
    assert "integrity check failed" in tampered.stderr


def _rewrite_lock(
    project: Path,
    *,
    got_revision: str | None = None,
    got_repository: str | None = None,
    docling_revision: str | None = None,
) -> None:
    payload = json.loads((project / "config/model_revisions.json").read_text(encoding="utf-8"))
    if got_revision is not None:
        payload["models"]["got_ocr"]["revision"] = got_revision
    if got_repository is not None:
        payload["models"]["got_ocr"]["repository"] = got_repository
    if docling_revision is not None:
        payload["models"]["docling"]["revision"] = docling_revision
    (project / "config/model_revisions.json").write_text(json.dumps(payload), encoding="utf-8")


def test_got_ocr_lock_change_does_not_skip_completed_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    _rewrite_lock(project, got_revision="c" * 40)
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == []
    assert ocr_calls
    provenance = json.loads((project / "out/provenance/academic/doc_a.json").read_text())
    assert provenance["got_ocr_revision"] == "c" * 40


def test_got_ocr_repository_change_reruns_ocr_and_records_new_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    new_repository = "example/new-got-ocr-repository"
    revision = MODEL_LOCK["models"]["got_ocr"]["revision"]
    _rewrite_lock(project, got_repository=new_repository)
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == []
    assert ocr_calls
    for doc in DOCS:
        for page_id in PAGES:
            assert f"{Path(doc).name}_page_{page_id}.png" in ocr_calls
        provenance = json.loads((project / f"out/provenance/{doc}.json").read_text())
        assert provenance["got_ocr_repository"] == new_repository
        assert provenance["got_ocr_revision"] == revision
        for page_id in PAGES:
            page = provenance["pages"][str(page_id)]
            assert page["got_ocr_repository"] == new_repository
            assert page["got_ocr_revision"] == revision
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    locked_got = prepare.load_locked_got_ocr(project)
    locked_docling = prepare.load_locked_docling(project)
    for doc in DOCS:
        source_sha = prepare._source_pdf_sha256(
            project_root=project,
            doc_rel=doc,
            pdf_root=project / "pdfs",
            pdf_zip=None,
            inventory_dir=project / "OHR-Bench/data/retrieval_base/gt",
        )
        assert prepare._completed_doc_valid(
            project,
            doc,
            checkpoint["completed"][doc],
            page_ids=PAGES,
            source_sha256=source_sha,
            got_ocr=locked_got,
            docling=locked_docling,
        )
    ocr_calls.clear()
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == DOCS
    assert ocr_calls == []


def test_missing_page_got_ocr_repository_forces_ocr_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    for doc in DOCS:
        path = project / f"out/provenance/{doc}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for page in payload["pages"].values():
            page.pop("got_ocr_repository", None)
        path.write_text(json.dumps(payload), encoding="utf-8")
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == []
    assert ocr_calls
    provenance = json.loads((project / "out/provenance/academic/doc_a.json").read_text())
    assert provenance["got_ocr_repository"] == MODEL_LOCK["models"]["got_ocr"]["repository"]
    assert provenance["pages"]["0"]["got_ocr_repository"] == MODEL_LOCK["models"]["got_ocr"]["repository"]


def test_docling_lock_change_does_not_skip_completed_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    _rewrite_lock(project, docling_revision="b" * 40)
    docling_calls: list[str] = []
    fake_ocr, _unused = install_fakes(monkeypatch)

    def counting_docling(pdf_path: Path, output_path: Path, **kwargs) -> Path:
        docling_calls.append(str(kwargs.get("revision")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# audit B\n", encoding="utf-8")
        return output_path

    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=counting_docling,
        resume=True,
    )
    assert summary["skipped"] == []
    assert docling_calls
    assert set(docling_calls) == {"b" * 40}
    provenance = json.loads((project / "out/provenance/academic/doc_a.json").read_text())
    assert provenance["docling_models"]["revision"] == "b" * 40
    assert provenance["docling_audit"]["revision"] == "b" * 40


def test_missing_source_pdf_fails_closed_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    for doc in DOCS:
        (project / "pdfs" / f"{doc}.pdf").unlink()
    with pytest.raises(SystemExit, match="Cannot verify source PDF"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)


def test_modified_image_or_ocr_invalidates_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    image = project / "out/images/academic/doc_a_page_0.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    import faar.asset_preparation as prep

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_id]))
        return {"pdf_page_count": len(PAGES), "render_runtime_sec": 0.01, "render_scale": scale}

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
        resume=True,
    )
    assert "academic/doc_a" not in summary["skipped"]
    assert any("doc_a_page_0" in name for name in ocr_calls)

    ocr_calls.clear()
    run_shard(project, monkeypatch=monkeypatch, resume=True)
    text = project / "out/ocr/academic/doc_b_page_1.txt"
    text.write_text(text.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert "academic/doc_b" not in summary["skipped"]
    assert any("doc_b_page_1" in name for name in ocr_calls)


def test_matching_completed_document_is_still_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls)
    ocr_calls.clear()
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == DOCS
    assert ocr_calls == []


def test_cli_relative_dataset_paths_resolve_against_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    outsider = tmp_path / "elsewhere"
    outsider.mkdir()
    monkeypatch.chdir(outsider)
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            "out",
            "--pdf-root",
            "pdfs",
            "--document-inventory",
            "OHR-Bench/data/retrieval_base/gt",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=outsider,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "dry_run"
