from __future__ import annotations

from pathlib import Path

from faar.experiment_profiles import apply_profile
from faar.experiment_runner import run_profile
from faar.settings import AppSettings


def test_apply_profile_resets_profile_controlled_state(tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)

    apply_profile(settings, "faar_no_vlm")
    assert settings.recovery.enable_vlm is False

    apply_profile(settings, "faar_full")
    assert settings.recovery.enable_vlm is True
    assert settings.experiment.wordlevel_fallback is None
    assert settings.recovery.wordlevel_fallback is None

    apply_profile(settings, "faar_symspell")
    assert settings.recovery.enable_vlm is True
    assert settings.experiment.wordlevel_fallback == "symspell"
    assert settings.recovery.wordlevel_fallback == "symspell"

    apply_profile(settings, "faar_always_vlm")
    assert settings.recovery.enable_vlm is True
    assert settings.experiment.wordlevel_fallback is None
    assert settings.recovery.wordlevel_fallback is None


def test_sequential_run_profiles_reset_shared_settings(monkeypatch, tmp_path: Path) -> None:
    observed: list[tuple[str, bool, str | None, str | None]] = []

    def fake_build_graph(settings: AppSettings, *, repo: object) -> object:
        observed.append(
            (
                settings.experiment.profile_name,
                settings.recovery.enable_vlm,
                settings.experiment.wordlevel_fallback,
                settings.recovery.wordlevel_fallback,
            )
        )
        return object()

    monkeypatch.setattr("faar.experiment_runner.build_graph", fake_build_graph)
    settings = AppSettings(project_root=tmp_path)
    for profile_name in ("faar_no_vlm", "faar_full", "faar_symspell", "faar_always_vlm"):
        run_profile(settings, profile_name=profile_name, example_ids=[], repo=object())

    assert observed == [
        ("faar_no_vlm", False, None, None),
        ("faar_full", True, None, None),
        ("faar_symspell", True, "symspell", "symspell"),
        ("faar_always_vlm", True, None, None),
    ]
