from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Local credentials live in an ignored .env file; explicit shell exports win.
load_dotenv(override=False)


def _load_locked_model_config() -> dict[str, dict[str, str]]:
    paths = (
        Path(__file__).resolve().parents[2] / "config/model_revisions.json",
        Path(__file__).with_name("model_revisions.json"),
    )
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload.get("models", {})
        return models if isinstance(models, dict) else {}
    return {}


LOCKED_MODELS = _load_locked_model_config()
MODEL_REPOSITORY_ALIASES = {
    "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}
ANTHROPIC_VLM_BACKENDS = {"claude-sonnet-4-5", "anthropic", "claude"}


def _locked_model_value(role: str, key: str, fallback: str | None = None) -> str | None:
    model = LOCKED_MODELS.get(role, {})
    value = model.get(key) if isinstance(model, dict) else None
    return str(value) if value else fallback


def _canonical_model_repository(repository: str) -> str:
    return MODEL_REPOSITORY_ALIASES.get(repository, repository)


def resolve_retrieval_models(
    *, embed: str | None = None, reranker: str | None = None
) -> tuple[str, str]:
    """Resolve effective embedding/reranker model names exactly as scripts/experiments/run.py will.

    Applies CLI override, then environment (EMBED_MODEL/EMBED_MODEL_REPO,
    RERANKER/RERANKER_MODEL_REPO), then committed model-revision locks, then
    defaults. Pure configuration resolution: no model weights, CUDA contexts,
    downloads, or API keys are touched.
    """
    retrieval = RetrievalSettings()
    embedding_model = embed or retrieval.embedding_model
    reranker_model = reranker or retrieval.reranker
    return embedding_model, reranker_model


def effective_retrieval_provenance(
    *, embed: str | None = None, reranker: str | None = None
) -> dict[str, dict[str, str | None]]:
    """Gate-relevant model provenance the next B0 will record (embedding + reranker).

    Mirrors scripts/experiments/run.py's resolution so gate-lock validation can compare the lock
    against the exact provenance a fresh run will produce, without importing
    torch or initializing any external client.
    """
    retrieval = RetrievalSettings()
    embedding_model = embed or retrieval.embedding_model
    reranker_model = reranker or retrieval.reranker
    return {
        "embedding": {
            "repository": _canonical_model_repository(embedding_model),
            "revision": retrieval.embedding_revision,
        },
        "reranker": {
            "repository": _canonical_model_repository(reranker_model),
            "revision": retrieval.reranker_revision,
        },
    }


def _positive_int_env(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; received {raw!r}.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value}.")
    return value


class RetrievalSettings(BaseModel):
    chunk_size_words: int = Field(default=180, gt=0)
    chunk_overlap_words: int = Field(default=40, ge=0)
    top_k: int = Field(default=5, ge=1)
    semantic_backtrack_top_k: int = Field(default=8, ge=1)
    embedding_backend: str = "sentence-transformers"
    embedding_batch_size: int = Field(default=64, gt=0, le=1024)
    max_chunks: int = Field(default=10_000, gt=0)
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL",
            os.getenv("EMBED_MODEL_REPO", _locked_model_value("embedding", "repository", "NV-Embed-v2")),
        )
    )
    embedding_revision: str | None = Field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL_REVISION",
            _locked_model_value("embedding", "revision"),
        )
    )
    reranker: str = Field(
        default_factory=lambda: os.getenv(
            "RERANKER",
            os.getenv("RERANKER_MODEL_REPO", _locked_model_value("reranker", "repository", "bge-reranker-v2-m3")),
        )
    )
    reranker_revision: str | None = Field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_REVISION",
            _locked_model_value("reranker", "revision"),
        )
    )
    visual_rag_base: str = Field(default_factory=lambda: os.getenv("VISUAL_RAG_BASE", "colpali"))
    colpali_model: str = Field(
        default_factory=lambda: os.getenv(
            "COLPALI_MODEL",
            os.getenv(
                "COLPALI_MODEL_REPO",
                _locked_model_value("colpali", "repository", "vidore/colpali-v1.2-hf"),
            ),
        )
    )
    colpali_revision: str | None = Field(
        default_factory=lambda: os.getenv(
            "COLPALI_MODEL_REVISION",
            _locked_model_value("colpali", "revision"),
        )
    )
    visrag_model: str = Field(
        default_factory=lambda: os.getenv(
            "VISRAG_MODEL",
            os.getenv("VISRAG_MODEL_REPO", _locked_model_value("visrag", "repository", "openbmb/VisRAG-Ret")),
        )
    )
    visrag_revision: str | None = Field(
        default_factory=lambda: os.getenv(
            "VISRAG_MODEL_REVISION",
            _locked_model_value("visrag", "revision"),
        )
    )
    visual_batch_size: int = Field(
        default_factory=lambda: _positive_int_env("FAAR_VISUAL_BATCH_SIZE", 1)
    )
    embed_batch_size: int = Field(
        default_factory=lambda: _positive_int_env("FAAR_EMBED_BATCH_SIZE", 2)
    )
    visual_score_batch_size: int = Field(
        default_factory=lambda: _positive_int_env("FAAR_VISUAL_SCORE_BATCH_SIZE", 8)
    )

    @model_validator(mode="after")
    def _validate_chunk_geometry(self) -> RetrievalSettings:
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError(
                f"chunk_overlap_words ({self.chunk_overlap_words}) must be smaller than "
                f"chunk_size_words ({self.chunk_size_words})"
            )
        return self


class GateSettings(BaseModel):
    # This value is overwritten by config/gate_threshold.json after Phase 2.
    quality_threshold: float = Field(
        default_factory=lambda: float(os.getenv("FAAR_GATE_THRESHOLD", "0.5")),
        ge=0.0,
        le=1.0,
    )
    structural_threshold: int = Field(default=1, ge=0)
    weird_char_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    lexical_floor: float = Field(default=0.10, ge=0.0, le=1.0)
    dense_floor: float = Field(default=0.20, ge=0.0, le=1.0)


class CorrectionSettings(BaseModel):
    """ByT5 correction-gate parameters. Defaults match the Phase 3 batch-run behavior."""

    min_weird_char_ratio: float = Field(default=0.08, ge=0.0)
    min_length_ratio: float = Field(default=0.6, ge=0.0)
    max_length_ratio: float = Field(default=1.4, ge=0.0)
    min_token_overlap: float = Field(default=0.5, ge=0.0, le=1.0)
    max_noise_increase: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _validate_length_ratio_bounds(self) -> CorrectionSettings:
        if self.min_length_ratio > self.max_length_ratio:
            raise ValueError(
                f"min_length_ratio ({self.min_length_ratio}) must not exceed "
                f"max_length_ratio ({self.max_length_ratio})"
            )
        return self


class RecoverySettings(BaseModel):
    byt5_model: str = Field(
        default_factory=lambda: os.getenv(
            "BYT5_MODEL_REPO",
            _locked_model_value("byt5", "repository", "google/byt5-small"),
        )
    )
    byt5_revision: str | None = Field(
        default_factory=lambda: os.getenv("BYT5_MODEL_REVISION", _locked_model_value("byt5", "revision"))
    )
    correction: CorrectionSettings = Field(default_factory=CorrectionSettings)
    enable_byt5: bool = True
    vlm_backend: str = Field(default_factory=lambda: os.getenv("VLM_BACKEND", "openai"))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20"))
    anthropic_model: str = "claude-sonnet-4-5"
    ocr_engine: str = Field(default_factory=lambda: os.getenv("OCR_ENGINE", "got-ocr-2"))
    got_ocr_model: str = Field(
        default_factory=lambda: os.getenv(
            "GOT_OCR_MODEL",
            os.getenv(
                "GOT_OCR_MODEL_REPO",
                _locked_model_value("got_ocr", "repository", "stepfun-ai/GOT-OCR-2.0-hf"),
            ),
        )
    )
    got_ocr_revision: str | None = Field(
        default_factory=lambda: os.getenv(
            "GOT_OCR_MODEL_REVISION",
            _locked_model_value("got_ocr", "revision"),
        )
    )
    enable_vlm: bool = True
    api_enabled: bool = True
    request_timeout_seconds: int = Field(
        default_factory=lambda: _positive_int_env("FAAR_VLM_TIMEOUT_SECONDS", 60)
    )
    log_vlm_calls: bool = Field(default_factory=lambda: os.getenv("LOG_VLM_CALLS", "true").lower() == "true")
    wordlevel_fallback: str | None = None


class ExperimentSettings(BaseModel):
    profile_name: str = "faar_full"
    disable_diagnosis: bool = False
    disable_backtracking: bool = False
    disable_vlm: bool = False
    force_direct_answer: bool = False
    force_recovery: bool = False
    force_vlm: bool = False
    random_recovery: bool = False
    random_seed: int = 42
    wordlevel_fallback: str | None = None


class AppSettings(BaseModel):
    project_root: Path = Field(default_factory=lambda: _default_project_root())
    phase0_manifest: Path | None = None
    phase0_summary: Path | None = None
    phase0_manual_labels: Path | None = None
    phase0_ocr_dir: Path | None = None
    logs_dir: Path | None = None
    artifacts_dir: Path | None = None
    gate_threshold_path: Path | None = None
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
        self.gate_threshold_path = (self.gate_threshold_path or self.project_root / "config/gate_threshold.json").resolve()

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

    def validate_openai_snapshot(self) -> None:
        if self.recovery.vlm_backend == "openai" and self.recovery.openai_model != "gpt-4o-2024-11-20":
            raise ValueError(
                "AAAI runs require the pinned OpenAI snapshot gpt-4o-2024-11-20; "
                f"received {self.recovery.openai_model!r}."
            )

    def validate_model_revisions(self, *, include_visual: str | None = None) -> None:
        models = {
            "embedding": (self.retrieval.embedding_model, self.retrieval.embedding_revision),
            "reranker": (self.retrieval.reranker, self.retrieval.reranker_revision),
            "got_ocr": (self.recovery.got_ocr_model, self.recovery.got_ocr_revision),
            "byt5": (self.recovery.byt5_model, self.recovery.byt5_revision),
        }
        if include_visual == "colpali":
            models["colpali"] = (self.retrieval.colpali_model, self.retrieval.colpali_revision)
        elif include_visual == "visrag":
            models["visrag"] = (self.retrieval.visrag_model, self.retrieval.visrag_revision)
        invalid = [
            name
            for name, (_, revision) in models.items()
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision)
        ]
        if invalid:
            raise ValueError(
                "Paper runs require immutable 40-character Hugging Face commit revisions for: "
                + ", ".join(invalid)
                + ". Resolve and export the corresponding *_MODEL_REVISION values before inference."
            )
        mismatched = []
        for name, (repository, revision) in models.items():
            locked_repository = _locked_model_value(name, "repository")
            locked_revision = _locked_model_value(name, "revision")
            if (
                locked_repository is None
                or _canonical_model_repository(repository) != locked_repository
                or revision != locked_revision
            ):
                mismatched.append(name)
        if mismatched:
            raise ValueError(
                "Model repository/revision pairs differ from config/model_revisions.json for: "
                + ", ".join(mismatched)
                + ". Resolve the intended commits and update the lock file before inference."
            )

    def vlm_provider(self) -> str:
        if self.recovery.vlm_backend == "openai":
            return "openai"
        if self.recovery.vlm_backend in ANTHROPIC_VLM_BACKENDS:
            return "anthropic"
        return self.recovery.vlm_backend

    def vlm_request_model(self) -> str | None:
        if self.recovery.vlm_backend == "openai":
            return self.recovery.openai_model
        if self.recovery.vlm_backend in ANTHROPIC_VLM_BACKENDS:
            return self.recovery.anthropic_model
        return None

    def model_provenance(self) -> dict[str, dict[str, object]]:
        return {
            "embedding": {
                "backend": self.retrieval.embedding_backend,
                "repository": _canonical_model_repository(self.retrieval.embedding_model),
                "revision": self.retrieval.embedding_revision,
            },
            "reranker": {
                "repository": _canonical_model_repository(self.retrieval.reranker),
                "revision": self.retrieval.reranker_revision,
            },
            "got_ocr": {"repository": self.recovery.got_ocr_model, "revision": self.recovery.got_ocr_revision},
            "byt5": {"repository": self.recovery.byt5_model, "revision": self.recovery.byt5_revision},
            "colpali": {"repository": self.retrieval.colpali_model, "revision": self.retrieval.colpali_revision},
            "visrag": {"repository": self.retrieval.visrag_model, "revision": self.retrieval.visrag_revision},
            "vlm": {
                "backend": self.recovery.vlm_backend,
                "provider": self.vlm_provider(),
                "model": self.vlm_request_model(),
            },
            "recovery": {
                "enable_byt5": self.recovery.enable_byt5,
                "byt5_correction": self.recovery.correction.model_dump(),
            },
        }


def _default_project_root() -> Path:
    env_root = os.getenv("FAAR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.cwd()
