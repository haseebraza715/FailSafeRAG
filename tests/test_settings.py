from pathlib import Path

import pytest

from faar.settings import AppSettings


def test_validate_runtime_paths_passes_with_existing_paths(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text("example_id\nx\n")
    settings = AppSettings(project_root=tmp_path)
    settings.validate_runtime_paths()


def test_validate_runtime_paths_raises_for_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        settings.validate_runtime_paths()
