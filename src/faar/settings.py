from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class RetrievalSettings(BaseModel):
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    top_k: int = 5
    semantic_backtrack_top_k: int = 8
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class GateSettings(BaseModel):
    quality_threshold: float = 0.52
    structural_threshold: int = 1
    weird_char_threshold: float = 0.10
    lexical_floor: float = 0.10
    dense_floor: float = 0.20


class RecoverySettings(BaseModel):
    byt5_model: str = "google/byt5-small"
    enable_backtracking: bool = True
    vlm_backend: str = "mock"
    openai_model: str = "gpt-4o"
    enable_vlm: bool = True
    api_enabled: bool = True
    request_timeout_seconds: int = 60


class ExperimentSettings(BaseModel):
    profile_name: str = "faar_full"
    disable_diagnosis: bool = False
    disable_backtracking: bool = False
    disable_vlm: bool = False
    force_direct_answer: bool = False


class AppSettings(BaseModel):
    project_root: Path = Field(default_factory=lambda: _default_project_root())
    phase0_manifest: Path | None = None
    phase0_summary: Path | None = None
    phase0_manual_labels: Path | None = None
    phase0_ocr_dir: Path | None = None
    logs_dir: Path | None = None
    artifacts_dir: Path | None = None
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)

    def model_post_init(self, __context: object) -> None:
        self.project_root = self.project_root.resolve()
        self.phase0_manifest = (self.phase0_manifest or self.project_root / "data/phase0/sample_manifest.csv").resolve()
        self.phase0_summary = (self.phase0_summary or self.project_root / "data/phase0/phase0_asset_summary.json").resolve()
        self.phase0_manual_labels = (self.phase0_manual_labels or self.project_root / "data/phase0/manual_labels.csv").resolve()
        self.phase0_ocr_dir = (self.phase0_ocr_dir or self.project_root / "artifacts/phase0/ocr_text").resolve()
        self.logs_dir = (self.logs_dir or self.project_root / "logs/phase1").resolve()
        self.artifacts_dir = (self.artifacts_dir or self.project_root / "artifacts/phase1").resolve()

    def validate_runtime_paths(self) -> None:
        required = {
            "project_root": self.project_root,
            "phase0_manifest": self.phase0_manifest,
            "phase0_ocr_dir": self.phase0_ocr_dir,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if path is None or not path.exists()]
        if missing:
            details = "\n".join(missing)
            raise FileNotFoundError(
                "FAAR settings validation failed.\n"
                "Set FAAR_PROJECT_ROOT or pass --project-root to point at the repository root.\n"
                f"Missing paths:\n{details}"
            )


def _default_project_root() -> Path:
    env_root = os.getenv("FAAR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.cwd()
