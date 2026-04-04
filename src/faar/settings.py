from __future__ import annotations

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


class AppSettings(BaseModel):
    project_root: Path = Field(default_factory=lambda: Path("/Users/x/Downloads/Thesis-Paper/Code"))
    phase0_manifest: Path | None = None
    phase0_summary: Path | None = None
    phase0_ocr_dir: Path | None = None
    logs_dir: Path | None = None
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)

    def model_post_init(self, __context: object) -> None:
        self.phase0_manifest = self.phase0_manifest or self.project_root / "data/phase0/sample_manifest.csv"
        self.phase0_summary = self.phase0_summary or self.project_root / "data/phase0/phase0_asset_summary.json"
        self.phase0_ocr_dir = self.phase0_ocr_dir or self.project_root / "artifacts/phase0/ocr_text"
        self.logs_dir = self.logs_dir or self.project_root / "logs/phase1"
