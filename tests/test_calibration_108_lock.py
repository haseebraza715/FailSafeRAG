from __future__ import annotations

import json
from pathlib import Path

import prepare_benchmark_assets as cli
import pytest

CALIBRATION_DOC = cli.CALIBRATION_108_DOC
EXPECTED_PAGES = list(range(108))


def _write_inventory(project: Path, doc: str, page_ids: list[int]) -> None:
    gt_dir = project / "OHR-Bench/data/retrieval_base/gt" / Path(doc).parent
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{Path(doc).name}.json").write_text(
        json.dumps([{"text": "x", "page_idx": page_id} for page_id in page_ids]),
        encoding="utf-8",
    )


def _project(tmp_path: Path, *, doc: str = CALIBRATION_DOC, page_ids: list[int] | None = None) -> Path:
    page_ids = EXPECTED_PAGES if page_ids is None else page_ids
    project = tmp_path
    (project / "config/datasets").mkdir(parents=True)
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
        ),
        encoding="utf-8",
    )
    (project / "OHR-Bench/data").mkdir(parents=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(json.dumps([{"ID": "e1", "doc_name": doc}]))
    (project / "config/datasets/ohr_split.json").write_text(json.dumps({"splits": {"val": ["e1"]}}))
    _write_inventory(project, doc, page_ids)
    pdf = project / "pdfs" / f"{doc}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF")
    return project


def _args(project: Path, doc: str, extra: list[str] | None = None, execute: bool = False) -> list[str]:
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
        "--require-calibration-108",
    ]
    if execute:
        args.append("--execute")
    if extra:
        args.extend(extra)
    return args


def _assert_no_completed_result(project: Path) -> None:
    out = project / "out"
    assert not list(out.glob("prepare_result_*.json"))
    checkpoint = out / "prepare_checkpoint.json"
    if checkpoint.is_file():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert not payload.get("completed")


def test_validate_calibration_108_accepts_exact_pages() -> None:
    cli.validate_calibration_108(smoke_doc=CALIBRATION_DOC, page_ids=EXPECTED_PAGES)


@pytest.mark.parametrize(
    "page_ids, match",
    [
        (list(range(107)), "page count is 107"),
        (list(range(109)), "page count is 109"),
        ([], "page inventory is missing"),
        (list(range(108)) + [0], "duplicate page indices"),
        ([page for page in range(108) if page != 5] + [108], "missing page indices"),
        (list(range(0, 108, 2)), "not contiguous"),
        (list(range(1, 109)), "unexpected page indices"),
    ],
)
def test_validate_calibration_108_rejects_bad_page_sets(page_ids: list[int], match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        cli.validate_calibration_108(smoke_doc=CALIBRATION_DOC, page_ids=page_ids)


def test_validate_calibration_108_rejects_other_document() -> None:
    with pytest.raises(SystemExit, match="academic/demo"):
        cli.validate_calibration_108(smoke_doc="academic/demo", page_ids=EXPECTED_PAGES)


def test_require_calibration_108_accepts_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    monkeypatch.setattr(
        cli,
        "execute_document_preparation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not execute during plan")),
    )
    assert cli.main(_args(project, CALIBRATION_DOC)) is None
    payload = json.loads((project / "out/prepare_plan_manual__User_Manual_1500S_Classic_EN.json").read_text())
    assert payload["page_ids"] == EXPECTED_PAGES
    assert not list((project / "out").glob("prepare_result_*.json"))


def test_unlocked_prep_still_accepts_other_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, doc="academic/demo", page_ids=[0])
    monkeypatch.setattr(
        cli,
        "execute_document_preparation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not execute during plan")),
    )
    args = [token for token in _args(project, "academic/demo") if token != "--require-calibration-108"]
    assert cli.main(args) is None


def test_require_calibration_108_rejects_other_document_before_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, doc="academic/demo", page_ids=[0])
    called = {"execute": False}

    def fake_execute(**kwargs):
        called["execute"] = True
        raise AssertionError("must not process")

    monkeypatch.setattr(cli, "execute_document_preparation", fake_execute)
    monkeypatch.setattr(cli, "hash_source_pdf", lambda **kwargs: (_ for _ in ()).throw(AssertionError("too late")))
    with pytest.raises(SystemExit, match="Bounded 108-page calibration lock failed"):
        cli.main(_args(project, "academic/demo", execute=True))
    assert called["execute"] is False
    _assert_no_completed_result(project)


@pytest.mark.parametrize(
    "page_ids, extra, match",
    [
        (list(range(107)), None, "page count is 107"),
        (list(range(109)), None, "page count is 109"),
        (list(range(108)), ["--page-ids", ",,,"], "page inventory is missing"),
        (list(range(108)), ["--page-ids", ",".join(["0"] + [str(i) for i in range(107)])], "duplicate"),
        (
            [page for page in range(108) if page != 50] + [200],
            None,
            "missing page indices",
        ),
        (list(range(0, 108, 2)), None, "not contiguous"),
        (list(range(1, 109)), None, "unexpected page indices"),
    ],
)
def test_require_calibration_108_rejects_bad_inventories_before_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    page_ids: list[int],
    extra: list[str] | None,
    match: str,
) -> None:
    project = _project(tmp_path, page_ids=page_ids)
    called = {"execute": False, "hash": False}

    def fake_execute(**kwargs):
        called["execute"] = True
        raise AssertionError("must not process")

    def fake_hash(**kwargs):
        called["hash"] = True
        raise AssertionError("must not hash")

    monkeypatch.setattr(cli, "execute_document_preparation", fake_execute)
    monkeypatch.setattr(cli, "hash_source_pdf", fake_hash)
    with pytest.raises(SystemExit, match=match):
        cli.main(_args(project, CALIBRATION_DOC, extra=extra, execute=True))
    assert called == {"execute": False, "hash": False}
    _assert_no_completed_result(project)
