from __future__ import annotations

import os
import signal
import time
from typing import Any

_TERMINATION_SIGNAL: int | None = None


def install_graceful_termination_handler() -> None:
    """Arrange for SIGTERM/SIGINT to stop work at the next example boundary."""
    global _TERMINATION_SIGNAL
    _TERMINATION_SIGNAL = None

    def _handler(signum: int, frame: Any) -> None:
        global _TERMINATION_SIGNAL
        if _TERMINATION_SIGNAL is None:
            _TERMINATION_SIGNAL = signum

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def check_termination() -> None:
    """Raise SystemExit(128+signum) once a graceful termination was requested."""
    signum = _TERMINATION_SIGNAL
    if signum is not None:
        raise SystemExit(128 + signum)


def _gpu_allocated_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        return None


def _int_env(name: str, fallback: int) -> int:
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


def _float_env(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number; received {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number; received {value}.")
    return value


class ProgressReporter:
    def __init__(
        self,
        stage: str,
        total: int,
        *,
        interval_seconds: float | None = None,
        interval_items: int | None = None,
    ) -> None:
        self.stage = stage
        self.total = max(total, 0)
        self.interval_seconds = (
            _float_env("FAAR_LOG_INTERVAL_SECONDS", 60.0) if interval_seconds is None else interval_seconds
        )
        self.interval_items = (
            _int_env("FAAR_LOG_INTERVAL_ITEMS", 25) if interval_items is None else interval_items
        )
        self.started = time.monotonic()
        self._last_time = self.started
        self._last_item = 0
        self.processed = 0

    def update(self, processed: int, *, skipped: int = 0) -> None:
        self.processed = max(self.processed, processed)
        now = time.monotonic()
        if (
            self.processed < self.total
            and now - self._last_time < self.interval_seconds
            and self.processed - self._last_item < self.interval_items
        ):
            return
        self._last_time = now
        self._last_item = self.processed
        elapsed = now - self.started
        rate = self.processed / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.processed) / rate if rate > 0 else float("nan")
        gpu = _gpu_allocated_gb()
        gpu_text = f" gpu_alloc={gpu:.2f}GiB" if gpu is not None else ""
        print(
            f"[faar] stage={self.stage} progress={self.processed}/{self.total} "
            f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s{gpu_text} skipped={skipped}",
            flush=True,
        )

    def finish(self, *, skipped: int = 0) -> None:
        self.update(self.total, skipped=skipped)
        print(
            f"[faar] stage={self.stage} done processed={self.total} "
            f"total_elapsed={time.monotonic() - self.started:.1f}s",
            flush=True,
        )


def startup_report(*, settings: Any = None) -> None:
    """Print the effective compute configuration once at process start."""
    device = "cpu"
    gpus = 0
    try:
        import torch

        gpus = torch.cuda.device_count()
        if gpus:
            device = f"cuda:{torch.cuda.current_device()}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    batch = ""
    if settings is not None:
        retrieval = getattr(settings, "retrieval", None)
        if retrieval is not None:
            batch = (
                f" embed_batch={retrieval.embed_batch_size} "
                f"visual_batch={retrieval.visual_batch_size} "
                f"score_batch={retrieval.visual_score_batch_size}"
            )
    print(
        f"[faar] device={device} cuda_gpus_visible={gpus}{f' (CUDA_VISIBLE_DEVICES={visible!r})' if visible else ''}"
        f"{batch}"
        f" omp_threads={os.getenv('OMP_NUM_THREADS', 'default')}"
        f" mkl_threads={os.getenv('MKL_NUM_THREADS', 'default')}"
        f" max_rss_gb={os.getenv('FAAR_MAX_RSS_GB', 'unset')}"
        f" gpu_fraction={os.getenv('FAAR_MAX_GPU_MEMORY_FRACTION', 'unset')}",
        flush=True,
    )
