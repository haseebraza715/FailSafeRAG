from __future__ import annotations

import math
import os
import sys
from typing import Any

try:
    import resource
except ImportError:
    resource = None


def _optional_gb(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of GiB; received {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite positive number of GiB; received {value}.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of GiB; received {value}.")
    return value


def _optional_fraction(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive fraction in (0, 1]; received {raw!r}.") from exc
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be a positive fraction in (0, 1]; received {value}.")
    return value


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value <= 0:
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


def _rss_bytes_from_proc_status(handle: Any) -> int | None:
    for line in handle:
        if line.startswith("VmRSS:"):
            try:
                value = int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                return None
            return value if value > 0 else None
    return None


def current_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            rss = _rss_bytes_from_proc_status(handle)
    except OSError:
        rss = None
    if rss is not None:
        return rss
    return _peak_rss_bytes()


def _import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def is_fatal_resource_error(exc: BaseException, torch_module: Any | None = None) -> bool:
    """True when exc is process-wide resource exhaustion that must abort the run.

    MemoryError and torch CUDA OutOfMemoryError are fatal; RuntimeError
    messages that unambiguously name CUDA out-of-memory are also fatal. Other
    exceptions (bad data, exhausted API calls) remain per-example recoverable
    failures. torch is imported lazily only when needed so CPU-only paths do
    not pay an eager import cost.
    """
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if "out of memory" not in message:
            return False
        if "cuda" in message:
            return True
        if torch_module is None:
            torch_module = _import_torch()
        if torch_module is not None:
            oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
            if oom_type is not None and isinstance(exc, oom_type):
                return True
    return False


def torch_device(torch_module: Any | None = None) -> Any:
    if torch_module is None:
        torch_module = _import_torch()
        if torch_module is None:
            raise RuntimeError("torch is required to select a compute device")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda:0")
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def select_dtype(device: Any, torch_module: Any | None = None) -> Any:
    if torch_module is None:
        torch_module = _import_torch()
        if torch_module is None:
            raise RuntimeError("torch is required to select a compute dtype")
    if device.type == "cuda":
        is_bf16 = getattr(torch_module.cuda, "is_bf16_supported", None)
        if callable(is_bf16) and is_bf16():
            return torch_module.bfloat16
        return torch_module.float16
    if device.type == "mps":
        return torch_module.float16
    return torch_module.float32


def release_cuda_cache(torch_module: Any | None = None) -> None:
    active_exception = sys.exception() is not None
    if torch_module is None:
        torch_module = _import_torch()
    if torch_module is None:
        return
    if not torch_module.cuda.is_available():
        return
    try:
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()
    except RuntimeError:
        if not active_exception:
            raise


_GPU_MEMORY_FRACTION_APPLIED = False


def enforce_gpu_memory_fraction(torch_module: Any | None = None) -> None:
    global _GPU_MEMORY_FRACTION_APPLIED
    fraction = _optional_fraction("FAAR_MAX_GPU_MEMORY_FRACTION")
    if fraction is None:
        return
    if torch_module is None:
        torch_module = _import_torch()
        if torch_module is None:
            return
    if not torch_module.cuda.is_available():
        return
    if _GPU_MEMORY_FRACTION_APPLIED:
        return
    try:
        torch_module.cuda.set_per_process_memory_fraction(fraction, 0)
    except AttributeError as exc:
        raise RuntimeError(
            "torch.cuda.set_per_process_memory_fraction is unavailable; "
            "cannot enforce FAAR_MAX_GPU_MEMORY_FRACTION."
        ) from exc
    _GPU_MEMORY_FRACTION_APPLIED = True


def enforce_memory_budget(stage: str, torch_module: Any | None = None) -> None:
    """Fail before the next inference step when an optional budget is exceeded.

    The cluster scheduler remains the hard CPU-RAM limit. These checks provide a
    portable, fail-fast guard inside the process once the cluster-specific
    values are supplied through the environment.
    """

    max_rss_gb = _optional_gb("FAAR_MAX_RSS_GB")
    if max_rss_gb is not None:
        rss = current_rss_bytes()
        if rss is not None and rss > max_rss_gb * 1024**3:
            raise MemoryError(
                f"FAAR_MAX_RSS_GB exceeded before {stage}: "
                f"current RSS is {rss / 1024**3:.2f} GiB, limit is {max_rss_gb:.2f} GiB."
            )

    min_gpu_free_gb = _optional_gb("FAAR_MIN_GPU_FREE_GB")
    if min_gpu_free_gb is None:
        return
    if torch_module is None:
        torch_module = _import_torch()
        if torch_module is None:
            return
    if not torch_module.cuda.is_available():
        return
    release_cuda_cache(torch_module)
    free_bytes, _ = torch_module.cuda.mem_get_info()
    if free_bytes < min_gpu_free_gb * 1024**3:
        raise MemoryError(
            f"FAAR_MIN_GPU_FREE_GB not available before {stage}: "
            f"free GPU memory is {free_bytes / 1024**3:.2f} GiB, "
            f"required reserve is {min_gpu_free_gb:.2f} GiB."
        )
