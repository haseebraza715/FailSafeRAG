from __future__ import annotations

import json
from pathlib import Path

import pytest

from faar import settings as settings_module
from faar.settings import AppSettings


def test_openai_requires_dated_snapshot(tmp_path) -> None:
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.openai_model = "gpt-4o"

    with pytest.raises(ValueError, match="gpt-4o-2024-11-20"):
        settings.validate_openai_snapshot()


def test_hugging_face_revisions_must_be_immutable_commit_shas(tmp_path) -> None:
    settings = AppSettings(project_root=tmp_path)
    locked = AppSettings(project_root=tmp_path)
    settings.retrieval.embedding_revision = None
    settings.retrieval.reranker_revision = None
    settings.recovery.got_ocr_revision = None
    settings.recovery.byt5_revision = None
    settings.retrieval.colpali_revision = None

    with pytest.raises(ValueError, match="embedding"):
        settings.validate_model_revisions(include_visual="colpali")

    settings.retrieval.embedding_revision = locked.retrieval.embedding_revision
    settings.retrieval.reranker_revision = locked.retrieval.reranker_revision
    settings.retrieval.colpali_revision = locked.retrieval.colpali_revision
    settings.recovery.got_ocr_revision = locked.recovery.got_ocr_revision
    settings.recovery.byt5_revision = locked.recovery.byt5_revision
    settings.validate_model_revisions(include_visual="colpali")

    provenance = settings.model_provenance()
    assert provenance["embedding"]["backend"] == "sentence-transformers"
    assert provenance["embedding"]["revision"] == locked.retrieval.embedding_revision
    assert provenance["recovery"] == {
        "enable_byt5": True,
        "byt5_correction": settings.recovery.correction.model_dump(),
    }
    assert provenance["vlm"]["backend"] == "openai"
    assert provenance["vlm"]["provider"] == "openai"
    assert provenance["vlm"]["model"] == "gpt-4o-2024-11-20"

    settings.retrieval.embedding_revision = "a" * 40
    with pytest.raises(ValueError, match="model_revisions.json"):
        settings.validate_model_revisions(include_visual="colpali")


def test_model_revision_lock_is_available_as_a_package_resource() -> None:
    package_path = Path(settings_module.__file__).with_name("model_revisions.json")
    repository_path = Path(__file__).resolve().parents[1] / "config/model_revisions.json"

    assert package_path.is_file()
    assert json.loads(package_path.read_text()) == json.loads(repository_path.read_text())


@pytest.mark.parametrize("backend", ["claude-sonnet-4-5", "anthropic", "claude"])
def test_vlm_provenance_encodes_anthropic_aliases(tmp_path, backend: str) -> None:
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = backend
    settings.recovery.anthropic_model = "claude-sonnet-4-5"

    vlm = settings.model_provenance()["vlm"]

    assert vlm["backend"] == backend
    assert vlm["provider"] == "anthropic"
    assert vlm["model"] == "claude-sonnet-4-5"


def test_vlm_provenance_tracks_selected_request_model(tmp_path) -> None:
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    assert settings.model_provenance()["vlm"]["model"] == settings.recovery.openai_model
    assert settings.vlm_request_model() == settings.recovery.openai_model

    settings.recovery.vlm_backend = "anthropic"
    assert settings.model_provenance()["vlm"]["model"] == settings.recovery.anthropic_model
    assert settings.vlm_request_model() == settings.recovery.anthropic_model


def test_embedding_backend_changes_published_model_provenance(tmp_path) -> None:
    settings = AppSettings(project_root=tmp_path)
    neural = settings.model_provenance()
    settings.retrieval.embedding_backend = "local-hash-v1"
    local = settings.model_provenance()

    assert neural["embedding"]["repository"] == local["embedding"]["repository"]
    assert neural["embedding"]["backend"] == "sentence-transformers"
    assert local["embedding"]["backend"] == "local-hash-v1"
    assert neural != local
