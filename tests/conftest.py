import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
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

# The offline suite must not download models. Set these before any test imports
# huggingface_hub or transformers, which read them at import time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# macOS only. The faiss-cpu and torch wheels each bundle their own libomp.dylib, so
# one process holds two OpenMP runtimes. With the pinned versions (faiss-cpu
# 1.9.0.post1, torch 2.6.0), importing faiss before torch is enough: the first torch
# operation that runs in parallel (a tensor of more than 32768 elements) then
# segfaults or hangs. In the other order, the first FAISS search on many queries
# does. Limiting both libraries to one thread avoids it in either order, because
# neither runtime then starts a thread pool. Limiting only one library fails in one
# of the two orders. scripts/diagnostics/openmp_check.py reproduces the matrix.
#
# KMP_DUPLICATE_LIB_OK is not set. With these versions it changes nothing. A newer
# torch (2.14 was tried) aborts with "OMP: Error #15" without it, so rerun the
# diagnostic after any torch or faiss-cpu change. Nothing is set on Linux, which
# this was not tested on.
#
# The limit goes through each library's API, not OMP_NUM_THREADS: the
# isolate_model_configuration fixture deletes that variable before each test.
MACOS_OPENMP_WORKAROUND = sys.platform == "darwin"
if MACOS_OPENMP_WORKAROUND:
    try:
        import faiss
        import torch

        faiss.omp_set_num_threads(1)
        torch.set_num_threads(1)
    except ImportError:
        MACOS_OPENMP_WORKAROUND = False


def pytest_report_header(config: pytest.Config) -> list[str]:
    lines = ["faar offline tests: HF_HUB_OFFLINE=1, non-loopback sockets blocked, real logs/ guarded"]
    if MACOS_OPENMP_WORKAROUND:
        lines.append(
            "macOS OpenMP workaround active: faiss and torch limited to 1 thread "
            "(their wheels each bundle libomp; see scripts/diagnostics/openmp_check.py)"
        )
    return lines


def _logs_snapshot() -> dict[str, tuple[int, int]] | None:
    if not LOGS_DIR.is_dir():
        return None
    return {
        str(path.relative_to(LOGS_DIR)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(LOGS_DIR.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def guard_real_logs_directory():
    """Fail the run if any test creates, edits or removes a file under the repository's logs/."""
    before = _logs_snapshot()
    yield
    after = _logs_snapshot()
    if before == after:
        return
    before_files = before or {}
    after_files = after or {}
    changed = sorted(
        name
        for name in before_files.keys() | after_files.keys()
        if before_files.get(name) != after_files.get(name)
    )
    pytest.fail(f"tests changed the real logs/ directory: {changed[:10]}", pytrace=False)


@pytest.fixture(autouse=True)
def redirect_api_call_logs(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Send ApiCallLogger output that targets the repository's logs/ to a temporary directory.

    Tests that pass their own project root keep writing under that root.
    """
    from faar import api_logging

    redirect_root = tmp_path_factory.mktemp("redirected-logs")
    original_init = api_logging.ApiCallLogger.__init__

    def init(self, log_path, enabled=True):
        log_path = Path(log_path)
        try:
            relative = log_path.resolve().relative_to(LOGS_DIR.resolve())
        except ValueError:
            pass
        else:
            log_path = redirect_root / relative
        original_init(self, log_path, enabled)

    monkeypatch.setattr(api_logging.ApiCallLogger, "__init__", init)


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without provider credentials or tracking configuration.

    faar.settings loads a local .env into os.environ when it is first imported, so
    import it before clearing.
    """
    import faar.settings  # noqa: F401

    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse in-process connections to anything except loopback and Unix sockets."""
    loopback = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(host: object) -> None:
        raise OSError(f"external network access is blocked in offline tests: {host!r}")

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and str(host) not in loopback:
            refuse(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self, address):
        if isinstance(address, tuple) and str(address[0]) not in loopback:
            refuse(address[0])
        return real_connect(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and str(address[0]) not in loopback:
            refuse(address[0])
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


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
