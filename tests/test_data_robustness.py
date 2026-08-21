from pathlib import Path

import pytest

from faar.data import Phase0Repository
from faar.settings import AppSettings


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
        "ex2,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text(
        '{"results":[{"example_id":"ex1","ocr_text_path":"'
        + str(tmp_path / "artifacts/phase0/ocr_text/ex1.txt")
        + '"}]}'
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text("===== PAGE 0 =====\ncertified engineers")
    (tmp_path / "artifacts/phase0/ocr_text/ex2.txt").write_text("")


def test_get_example_missing_ocr_artifact_raises_clear_error(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "artifacts/phase0/ocr_text/ex2.txt").unlink()
    settings = AppSettings(project_root=tmp_path)
    repo = Phase0Repository(settings)
    with pytest.raises(FileNotFoundError, match="ex2"):
        repo.get_example("ex2")


def test_get_example_empty_ocr_artifact_raises_clear_error(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    settings = AppSettings(project_root=tmp_path)
    repo = Phase0Repository(settings)
    with pytest.raises(ValueError, match="ex2"):
        repo.get_example("ex2")


def test_get_example_rejects_summary_path_escaping_project_root(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leaked.txt").write_text("secret")
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text(
        '{"results":[{"example_id":"ex1","ocr_text_path":"../../outside/leaked.txt"}]}'
    )
    repo = Phase0Repository(AppSettings(project_root=tmp_path))
    with pytest.raises(ValueError, match=r"must not contain '\.\.'"):
        repo.get_example("ex1")


def test_get_example_unknown_id_raises_clear_error(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    settings = AppSettings(project_root=tmp_path)
    repo = Phase0Repository(settings)
    with pytest.raises(FileNotFoundError, match="unknown-id"):
        repo.get_example("unknown-id")
