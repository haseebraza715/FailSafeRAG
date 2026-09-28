import os
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

# macOS only. The faiss-cpu and torch wheels each bundle their own libomp.dylib.
# When one process runs a FAISS search and later a multi-threaded torch operation,
# the process segfaults or hangs (tests/test_bounded_memory_batches.py crashed in
# the full suite after any test that used FAISS). With both runtimes limited to one
# thread the crash does not occur. Linux wheels share one libgomp, so nothing is
# set there. The limit is set through each library's API and not through
# OMP_NUM_THREADS, because isolate_model_configuration deletes that variable before
# each test and libomp reads it lazily at its first parallel region. With the
# pinned torch 2.6.0 the thread limit alone is enough. Newer torch wheels (2.14
# was tried) abort at the first parallel region with "OMP: Error #15" unless
# KMP_DUPLICATE_LIB_OK is also set, so it is set here as well.
MACOS_OPENMP_WORKAROUND = sys.platform == "darwin"
if MACOS_OPENMP_WORKAROUND:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import faiss
        import torch

        faiss.omp_set_num_threads(1)
        torch.set_num_threads(1)
    except ImportError:
        MACOS_OPENMP_WORKAROUND = False


def pytest_report_header(config: pytest.Config) -> list[str]:
    lines: list[str] = []
    if MACOS_OPENMP_WORKAROUND:
        lines.append(
            "macOS OpenMP workaround active: faiss and torch limited to 1 thread, "
            "KMP_DUPLICATE_LIB_OK set (their wheels each bundle libomp)"
        )
    return lines


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
