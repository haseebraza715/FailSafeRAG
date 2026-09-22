import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIRS = (
    ROOT / "scripts/experiments",
    ROOT / "scripts/data",
    ROOT / "scripts/annotation",
    ROOT / "scripts/smoke",
    ROOT / "scripts/release",
)
for import_path in (SRC, *SCRIPT_DIRS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


@pytest.fixture(autouse=True)
def isolate_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from faar import resource_limits

    monkeypatch.setattr(resource_limits, "_GPU_MEMORY_FRACTION_APPLIED", False)
    for name in (
        "EMBED_MODEL",
        "EMBED_MODEL_REPO",
        "EMBED_MODEL_REVISION",
        "RERANKER",
        "RERANKER_MODEL_REPO",
        "RERANKER_MODEL_REVISION",
        "GOT_OCR_MODEL",
        "GOT_OCR_MODEL_REPO",
        "GOT_OCR_MODEL_REVISION",
        "COLPALI_MODEL",
        "COLPALI_MODEL_REPO",
        "COLPALI_MODEL_REVISION",
        "VISRAG_MODEL",
        "VISRAG_MODEL_REPO",
        "VISRAG_MODEL_REVISION",
        "BYT5_MODEL_REPO",
        "BYT5_MODEL_REVISION",
        "OPENAI_MODEL",
        "OPENAI_INPUT_USD_PER_MTOK",
        "OPENAI_OUTPUT_USD_PER_MTOK",
        "VLM_BACKEND",
        "FAAR_EMBED_BATCH_SIZE",
        "FAAR_VISUAL_SCORE_BATCH_SIZE",
        "FAAR_MAX_RSS_GB",
        "FAAR_GPU_BUDGET_GB",
        "FAAR_MIN_GPU_FREE_GB",
        "FAAR_MAX_GPU_MEMORY_FRACTION",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def isolate_run_benchmark_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    run = sys.modules.get("run")
    if run is None:
        return

    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
