from __future__ import annotations

from dataclasses import dataclass

from .settings import AppSettings


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    disable_diagnosis: bool = False
    disable_backtracking: bool = False
    disable_vlm: bool = False
    force_direct_answer: bool = False
    force_recovery: bool = False
    force_vlm: bool = False
    random_recovery: bool = False
    wordlevel_fallback: str | None = None


PROFILES: dict[str, ExperimentProfile] = {
    "naive_rag": ExperimentProfile(
        name="naive_rag",
        disable_diagnosis=True,
        disable_backtracking=True,
        disable_vlm=True,
        force_direct_answer=True,
    ),
    "faar_full": ExperimentProfile(name="faar_full"),
    "faar_always_vlm": ExperimentProfile(name="faar_always_vlm", force_vlm=True),
    "faar_no_backtrack": ExperimentProfile(name="faar_no_backtrack", disable_backtracking=True),
    "faar_no_vlm": ExperimentProfile(name="faar_no_vlm", disable_vlm=True),
    "faar_no_diagnosis": ExperimentProfile(
        name="faar_no_diagnosis",
        disable_diagnosis=True,
        random_recovery=True,
    ),
    "faar_no_gate": ExperimentProfile(name="faar_no_gate", force_recovery=True),
    "faar_symspell": ExperimentProfile(name="faar_symspell", wordlevel_fallback="symspell"),
}


def get_profile(name: str) -> ExperimentProfile:
    if name not in PROFILES:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unsupported profile '{name}'. Supported profiles: {supported}")
    return PROFILES[name]


def apply_profile(settings: AppSettings, profile_name: str) -> AppSettings:
    profile = get_profile(profile_name)
    settings.experiment.profile_name = profile.name
    settings.experiment.disable_diagnosis = profile.disable_diagnosis
    settings.experiment.disable_backtracking = profile.disable_backtracking
    settings.experiment.disable_vlm = profile.disable_vlm
    settings.experiment.force_direct_answer = profile.force_direct_answer
    settings.experiment.force_recovery = profile.force_recovery
    settings.experiment.force_vlm = profile.force_vlm
    settings.experiment.random_recovery = profile.random_recovery
    settings.experiment.wordlevel_fallback = profile.wordlevel_fallback
    settings.recovery.wordlevel_fallback = profile.wordlevel_fallback
    settings.recovery.enable_vlm = not profile.disable_vlm
    return settings
