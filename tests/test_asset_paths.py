from __future__ import annotations

from pathlib import Path

import pytest

from faar.asset_paths import AssetPathError, resolve_project_asset, to_relative_project_path


def test_to_relative_project_path_is_portable(tmp_path: Path) -> None:
    asset = tmp_path / "data" / "page.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"png")

    relative = to_relative_project_path(asset, tmp_path)

    assert relative == "data/page.png"
    assert not relative.startswith("/")
    assert "Users" not in relative


def test_resolve_project_asset_loads_relative_paths(tmp_path: Path) -> None:
    asset = tmp_path / "ocr" / "doc_page_1.txt"
    asset.parent.mkdir(parents=True)
    asset.write_text("OCR")

    resolved = resolve_project_asset("ocr/doc_page_1.txt", tmp_path)

    assert resolved == asset.resolve()


def test_resolve_project_asset_rejects_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("nope")

    with pytest.raises(AssetPathError, match="must not contain '\\.\\.'|escapes project root"):
        resolve_project_asset("../secret.txt", tmp_path)
