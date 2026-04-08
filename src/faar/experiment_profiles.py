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


PROFILES: dict[str, ExperimentProfile] = {
    "naive_rag": ExperimentProfile(
        name="naive_rag",
        disable_diagnosis=True,
        disable_backtracking=True,
        disable_vlm=True,
        force_direct_answer=True,
    ),
    "faar_full": ExperimentProfile(name="faar_full"),
    "faar_no_backtrack": ExperimentProfile(name="faar_no_backtrack", disable_backtracking=True),
    "faar_no_vlm": ExperimentProfile(name="faar_no_vlm", disable_vlm=True),
    "faar_no_diagnosis": ExperimentProfile(
        name="faar_no_diagnosis",
        disable_diagnosis=True,
        disable_backtracking=True,
        disable_vlm=True,
        force_direct_answer=True,
    ),
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
    if settings.experiment.disable_vlm:
        settings.recovery.enable_vlm = False
    return settings

