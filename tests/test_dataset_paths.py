from __future__ import annotations

import os
from pathlib import Path

import pytest

from faar.dataset_paths import (
    DatasetPathError,
    load_project_dotenv,
    resolve_against_project_root,
    resolve_dataset_paths,
)


def test_relative_paths_resolve_against_project_root_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    inventory = project / "OHR-Bench/data/retrieval_base/gt"
    inventory.mkdir(parents=True)
    pdf_root = project / "pdfs"
    pdf_root.mkdir()
    zip_path = project / "data/ohr_bench_raw/pdfs.zip"
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(b"PK")
    outsider = tmp_path / "cwd"
    outsider.mkdir()
    monkeypatch.chdir(outsider)
    resolved = resolve_dataset_paths(
        project_root=project,
        inventory="OHR-Bench/data/retrieval_base/gt",
        pdf_root="pdfs",
        pdf_zip="data/ohr_bench_raw/pdfs.zip",
    )
    assert resolved.inventory_dir == inventory.resolve()
    assert resolved.pdf_root == pdf_root.resolve()
    assert resolved.pdf_zip == zip_path.resolve()
    assert resolve_against_project_root("pdfs", project) == pdf_root.resolve()


def test_inventory_file_is_rejected(tmp_path: Path) -> None:
    project = tmp_path
    inventory = project / "OHR-Bench/data/retrieval_base/gt"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("not a directory")
    with pytest.raises(DatasetPathError, match="document inventory is not a directory"):
        resolve_dataset_paths(project_root=project, inventory=inventory)


def test_explicit_pdf_zip_is_validated_even_when_pdf_root_is_set(tmp_path: Path) -> None:
    project = tmp_path
    inventory = project / "OHR-Bench/data/retrieval_base/gt"
    inventory.mkdir(parents=True)
    pdf_root = project / "pdfs"
    pdf_root.mkdir()
    missing_zip = project / "missing.zip"
    with pytest.raises(DatasetPathError, match="FAAR_PDF_ZIP / --pdf-zip is not a file"):
        resolve_dataset_paths(
            project_root=project,
            inventory=inventory,
            pdf_root=pdf_root,
            pdf_zip=missing_zip,
        )


def test_load_project_dotenv_does_not_override_exported_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("FAAR_SCRATCH=/from-dotenv\nFAAR_PDF_ZIP=/from-dotenv.zip\n")
    monkeypatch.delenv("FAAR_SCRATCH", raising=False)
    monkeypatch.setenv("FAAR_PDF_ZIP", "/already-exported.zip")
    outsider = tmp_path / "cwd"
    outsider.mkdir()
    monkeypatch.chdir(outsider)
    assert load_project_dotenv(project) is True
    assert os.environ["FAAR_SCRATCH"] == "/from-dotenv"
    assert os.environ["FAAR_PDF_ZIP"] == "/already-exported.zip"


def test_load_project_dotenv_missing_file_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAAR_SCRATCH", raising=False)
    assert load_project_dotenv(tmp_path) is False
    assert os.getenv("FAAR_SCRATCH") is None


def test_pdf_root_must_be_a_directory(tmp_path: Path) -> None:
    project = tmp_path
    inventory = project / "gt"
    inventory.mkdir()
    pdf_root = project / "pdfs"
    pdf_root.write_text("file")
    with pytest.raises(DatasetPathError, match="FAAR_PDF_ROOT / --pdf-root is not a directory"):
        resolve_dataset_paths(project_root=project, inventory=inventory, pdf_root=pdf_root)
