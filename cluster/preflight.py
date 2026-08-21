#!/usr/bin/env python3
"""Collect cluster hardware and quota facts without loading FAAR models."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.api_logging import ANTHROPIC_VLM_BACKENDS
from faar.dataset_paths import (
    SPLIT_RELATIVE_PATH,
    DatasetPathError,
    env_raw,
    load_project_dotenv,
    resolve_dataset_paths,
)

try:
    import resource
except ImportError:
    resource = None

_SECRET_PATTERNS = tuple(
    re.compile(pattern) for pattern in (
        r"(sk-[A-Za-z0-9_-]{8,})",
        r"(AKIA[0-9A-Z]{16})",
        r"(AIza[0-9A-Za-z_-]{20,})",
        r"(ghp_[0-9A-Za-z]{20,})",
        r"(github_pat_[0-9A-Za-z_]{20,})",
        r"(gh[pousr]_[0-9A-Za-z]{20,})",
        r"(glpat-[0-9A-Za-z_-]{15,})",
        r"(hf_[0-9A-Za-z]{20,})",
        r"(xox[baprs]-[0-9A-Za-z-]{10,})",
        r"([\w-]*API[_-]KEY[=:\s]+[^\s,}]+)",
        r"([\w-]*SECRET[=:\s]+[^\s,}]+)",
    )
)


def _run(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _bytes_from_text(raw: str) -> int | None:
    raw = raw.strip()
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _parse_memory_bytes(raw: str | None, default_multiplier: int = 1024**2) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() in {"max", "unlimited", "0"}:
        return None
    match = re.fullmatch(r"(\d+)\s*([KMGTP])?i?[bB]?", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    suffix = (match.group(2) or "").upper()
    multiplier = default_multiplier if not suffix else {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[suffix]
    return value * multiplier


def _parse_walltime_seconds(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if "-" in text:
        days_part, clock = text.split("-", 1)
        try:
            days = int(days_part)
        except ValueError:
            return None
    else:
        days = 0
        clock = text
    parts = clock.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        try:
            return int(text) * 60
        except ValueError:
            return None
    if len(numbers) == 3:
        return days * 86400 + numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return days * 86400 + numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0] * 60
    return None


def _parse_gpu_count(raw: str | None, *, id_list: bool = False) -> int | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None

    def token_count(token: str, id_is_one: bool) -> int:
        parts = token.split(":")
        if token.isdigit():
            return 1 if id_is_one else int(token)
        if parts[-1].isdigit():
            return int(parts[-1])
        return 0

    if len(tokens) == 1:
        return token_count(tokens[0], id_is_one=id_list) or None
    total = sum(token_count(token, id_is_one=True) for token in tokens)
    return total or None


def _cgroup_memory() -> dict[str, int | None]:
    v2 = {
        "current": Path("/sys/fs/cgroup/memory.current"),
        "peak": Path("/sys/fs/cgroup/memory.peak"),
        "limit": Path("/sys/fs/cgroup/memory.max"),
    }
    v1 = {
        "current": Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "peak": Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
        "limit": Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    }
    source = "none"
    payload: dict[str, int | None] = {"current_bytes": None, "peak_bytes": None, "limit_bytes": None}
    for mapping, version in ((v2, "v2"), (v1, "v1")):
        if any(path.is_file() for path in mapping.values()):
            source = version
            for key, path in mapping.items():
                if path.is_file():
                    try:
                        payload[f"{key}_bytes"] = _bytes_from_text(path.read_text(encoding="utf-8"))
                    except OSError:
                        pass
            break
    payload["controller"] = source
    return payload


def _memory_bytes() -> dict[str, int | None]:
    total: int | None = None
    if hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    cgroup = _cgroup_memory()
    cgroup_limit = cgroup.get("limit_bytes")
    if total and cgroup_limit and cgroup_limit > total:
        cgroup_limit = None
    return {
        "physical_bytes": total,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup.get("current_bytes"),
        "cgroup_peak_bytes": cgroup.get("peak_bytes"),
        "cgroup_controller": cgroup.get("controller"),
    }


def _ulimit() -> dict[str, Any]:
    payload: dict[str, Any] = {"available": resource is not None}
    if resource is not None:
        for name, constant in (
            ("RLIMIT_AS", resource.RLIMIT_AS),
            ("RLIMIT_DATA", resource.RLIMIT_DATA),
            ("RLIMIT_RSS", resource.RLIMIT_RSS),
            ("RLIMIT_NOFILE", resource.RLIMIT_NOFILE),
            ("RLIMIT_NPROC", resource.RLIMIT_NPROC),
            ("RLIMIT_STACK", resource.RLIMIT_STACK),
        ):
            try:
                soft, hard = resource.getrlimit(constant)
                payload[name] = {"soft": soft, "hard": hard}
            except (ValueError, OSError):
                continue
    return payload


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.total - usage.free, "free_bytes": usage.free}


def _writable_free(path: Path) -> dict[str, Any]:
    try:
        exists = path.exists()
    except OSError:
        exists = False
    writable = exists and os.access(path, os.W_OK)
    free_bytes: int | None = None
    total_bytes: int | None = None
    if exists:
        try:
            usage = shutil.disk_usage(path)
            free_bytes, total_bytes = usage.free, usage.total
        except OSError:
            pass
    return {
        "path": str(path),
        "exists": exists,
        "writable": writable,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def _locations_from_env(names: tuple[str, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        path = Path(raw).expanduser()
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        results.append({"env": name, **_writable_free(path)})
    return results


def _torch_info(cuda: bool = True) -> dict[str, Any]:
    if not cuda:
        return {
            "available": False,
            "cuda_available": False,
            "devices": [],
            "skipped": "CUDA checks disabled by --no-cuda",
        }
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"available": False, "error": f"torch import failed: {exc}"}

    payload: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "devices": [],
    }
    if torch.cuda.is_available():
        driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            payload["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "free_memory_bytes": free_bytes,
                    "driver_version": driver.get("stdout") if driver.get("available") else None,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "device_count": torch.cuda.device_count(),
                }
            )
    return payload


def _scheduler() -> dict[str, Any]:
    is_slurm = os.getenv("SLURM_JOB_ID") is not None
    is_pbs = os.getenv("PBS_JOBID") is not None
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    return {
        "scheduler_detected": "slurm" if is_slurm else "pbs" if is_pbs else None,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "slurm_partition": os.getenv("SLURM_JOB_PARTITION") or os.getenv("SLURM_PARTITION"),
        "slurm_qos": os.getenv("SLURM_JOB_QOS"),
        "slurm_num_nodes": _int_env("SLURM_JOB_NUM_NODES"),
        "slurm_cpus_on_node": _int_env("SLURM_CPUS_ON_NODE"),
        "slurm_cpus_per_task": _int_env("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_node_bytes": _parse_memory_bytes(os.getenv("SLURM_MEM_PER_NODE")),
        "slurm_mem_per_cpu": os.getenv("SLURM_MEM_PER_CPU"),
        "slurm_mem_per_cpu_bytes": _parse_memory_bytes(os.getenv("SLURM_MEM_PER_CPU")),
        "slurm_time_limit_seconds": _parse_walltime_seconds(os.getenv("SLURM_TIME_LIMIT")),
        "slurm_gpus_requested": (
            _parse_gpu_count(slurm_job_gpus, id_list=True)
            if slurm_job_gpus is not None
            else _parse_gpu_count(os.getenv("SLURM_GPUS"))
        ),
        "slurm_gpus_on_node": _parse_gpu_count(os.getenv("SLURM_GPUS_ON_NODE")),
        "pbs_job_id": os.getenv("PBS_JOBID"),
        "pbs_job_name": os.getenv("PBS_JOBNAME"),
        "pbs_queue": os.getenv("PBS_O_QUEUE") or os.getenv("PBS_QUEUE"),
        "pbs_num_nodes": _int_env("PBS_NUM_NODES"),
        "pbs_num_ppn": _int_env("PBS_NUM_PPN"),
        "pbs_ncpus": _int_env("PBS_NCPUS"),
        "pbs_mem": _pbs_mem_raw(),
        "pbs_mem_bytes": _parse_memory_bytes(_pbs_mem_raw()),
        "pbs_gpus_requested": _parse_gpu_count(os.getenv("PBS_GPUS")),
        "pbs_gpus_on_node": _pbs_gpufile_count(),
        "pbs_walltime_seconds": _parse_walltime_seconds(os.getenv("PBS_WALLTIME")),
    }


def _pbs_mem_raw() -> str | None:
    return next((os.getenv(name) for name in ("PBS_RESC_MEM", "PBS_MEM", "PBS_MEMORY") if os.getenv(name)), None)


def _pbs_gpufile_count() -> int | None:
    raw = os.getenv("PBS_GPUFILE")
    if not raw:
        return None
    gpufile = Path(raw).expanduser()
    try:
        if gpufile.is_file():
            return sum(1 for line in gpufile.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        pass
    return _parse_gpu_count(raw)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        masked = value
        for pattern in _SECRET_PATTERNS:
            masked = pattern.sub("<redacted>", masked)
        return masked
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def render(payload: dict[str, Any]) -> str:
    return json.dumps(redact(payload), indent=2, sort_keys=True)


def collect(path: Path, *, cuda: bool = True) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": _memory_bytes(),
        "disk": {"path": str(path), **_disk(path)},
        "scheduler": _scheduler(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": _torch_info(cuda=cuda),
        "nvidia_smi": (
            _run(["nvidia-smi", "-L"])
            if cuda and shutil.which("nvidia-smi")
            else {"available": False}
        ),
        "ulimit": _ulimit(),
        "scratch": _locations_from_env(("TMPDIR", "TMP", "TEMP", "SCRATCH", "SCRATCH_DIR", "LOCAL_SCRATCH", "TMPFS")),
        "cache": _locations_from_env(("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")),
        "relevant_environment": {
            key: os.getenv(key)
            for key in (
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "FAAR_VISUAL_BATCH_SIZE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "FAAR_GPU_BUDGET_GB",
                "FAAR_MAX_RSS_GB",
                "FAAR_MIN_GPU_FREE_GB",
                "FAAR_MAX_GPU_MEMORY_FRACTION",
            )
            if os.getenv(key) is not None
        },
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


GIB = 1024**3
REQUIRED_DEPENDENCIES = ("docling", "transformers", "pypdfium2", "PIL", "huggingface_hub")
DISK_WARNING_GIB = 20.0
RELEVANT_ENV_KEYS = (
    "FAAR_GPU_BUDGET_GB",
    "FAAR_MIN_GPU_FREE_GB",
    "FAAR_MAX_RSS_GB",
    "FAAR_MAX_GPU_MEMORY_FRACTION",
    "FAAR_OUT_ROOT",
    "FAAR_SCRATCH",
    "FAAR_PDF_ROOT",
    "FAAR_PDF_ZIP",
    "FAAR_DOCUMENT_INVENTORY",
    "FAAR_CACHE_ROOT",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VLM_BACKEND",
    "CUDA_VISIBLE_DEVICES",
)
RELEVANT_ENV_KEYS_PRESENCE_ONLY = ("HF_TOKEN",)


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _positive_gb_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _hf_repo_reachable(repo: str, revision: str) -> tuple[bool, str]:
    """HEAD the model file on the HF CDN; never downloads weights and never
    incurs model-generation charges. Attaches HF_TOKEN as a bearer header when
    set so gated repositories are verified with the operator's credentials;
    the token itself is never included in the returned detail."""
    url = f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
    headers: dict[str, str] = {}
    token = os.getenv("HF_TOKEN")
    if token and token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status < 400, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return exc.code < 400, f"HTTP {exc.code}"
    except OSError:
        return False, "network error"


def _model_lock(project_root: Path) -> list[dict[str, str]]:
    lock_path = project_root / "config/model_revisions.json"
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        return []
    return [
        {"role": str(role), "repository": str(entry.get("repository", "")), "revision": str(entry.get("revision", ""))}
        for role, entry in models.items()
        if isinstance(entry, dict) and entry.get("repository")
    ]


def _writable_check(path: Path) -> tuple[bool, str | None]:
    try:
        if path.is_dir():
            writable = os.access(path, os.W_OK)
        else:
            writable = os.access(path.parent, os.W_OK)
    except OSError as exc:
        return False, str(exc)
    return writable, None


def run_checks(
    payload: dict[str, Any],
    *,
    project_root: Path,
    path: Path | None = None,
    cuda: bool = True,
) -> dict[str, Any]:
    """Evaluate the collected payload against handoff requirements.

    Returns a checks report with per-check status. Exit code rules:
    required checks that fail are blocking (exit 1); warning-severity
    failures and warnings exit 2; a fully passing report exits 0.
    """
    checks: list[dict[str, Any]] = []

    def record(name: str, severity: str, status: str, detail: Any = None, **extra: Any) -> None:
        entry: dict[str, Any] = {"name": name, "severity": severity, "status": status}
        if detail is not None:
            entry["detail"] = detail
        entry.update(extra)
        checks.append(entry)

    major, minor = sys.version_info[:2]
    record(
        "python_version",
        "required",
        "pass" if (major, minor) >= (3, 12) else "fail",
        detail=sys.version.split()[0],
        measurement=f"{major}.{minor}",
    )

    missing_deps = [name for name in REQUIRED_DEPENDENCIES if not _find_spec(name)]
    if cuda and not _find_spec("torch"):
        missing_deps.append("torch")
    record(
        "dependencies",
        "required",
        "pass" if not missing_deps else "fail",
        detail="missing: " + ", ".join(missing_deps) if missing_deps else "all present",
        measurement={"missing": missing_deps},
    )

    torch_info = payload.get("torch") or {}
    devices = torch_info.get("devices") or []
    if not cuda:
        record("cuda_visibility", "required", "skip", detail="CUDA checks disabled by --no-cuda")
    elif torch_info.get("available") is False or not torch_info.get("cuda_available") or not devices:
        record(
            "cuda_visibility",
            "required",
            "fail",
            detail=torch_info.get("error") or "torch reports no usable CUDA devices",
        )
    else:
        record(
            "cuda_visibility",
            "required",
            "pass",
            detail=f"{len(devices)} device(s) visible",
            measurement=[
                {"index": d.get("index"), "name": d.get("name"), "total_memory_gib": round((d.get("total_memory_bytes") or 0) / GIB, 2)}
                for d in devices
            ],
        )
        scheduler = payload.get("scheduler") or {}
        requested = scheduler.get("slurm_gpus_requested") or scheduler.get("pbs_gpus_requested")
        if requested is not None and requested != len(devices):
            record(
                "gpu_allocation_match",
                "required",
                "fail",
                detail=f"scheduler requested {requested} GPU(s); {len(devices)} are visible",
                measurement={"requested": requested, "visible": len(devices)},
            )
        else:
            record(
                "gpu_allocation_match",
                "required",
                "pass",
                detail=f"visible={len(devices)} requested={requested}",
                measurement={"requested": requested, "visible": len(devices)},
            )
        total_gib = (devices[0].get("total_memory_bytes") or 0) / GIB
        budget_raw = os.getenv("FAAR_GPU_BUDGET_GB")
        fraction_raw = os.getenv("FAAR_MAX_GPU_MEMORY_FRACTION")
        budget = _positive_gb_env("FAAR_GPU_BUDGET_GB")
        fraction = _positive_gb_env("FAAR_MAX_GPU_MEMORY_FRACTION")
        if budget_raw and budget_raw.strip() and fraction_raw and fraction_raw.strip():
            record(
                "gpu_budget_consistency",
                "required",
                "fail",
                detail="FAAR_GPU_BUDGET_GB and FAAR_MAX_GPU_MEMORY_FRACTION are both set",
            )
        else:
            if fraction is not None and fraction > 1.0:
                record(
                    "gpu_budget_consistency",
                    "required",
                    "fail",
                    detail="FAAR_MAX_GPU_MEMORY_FRACTION must be in (0, 1]",
                )
            elif budget is not None and budget > total_gib:
                record(
                    "gpu_budget_consistency",
                    "required",
                    "fail",
                    detail=f"FAAR_GPU_BUDGET_GB={budget:.2f} exceeds visible VRAM {total_gib:.2f} GiB",
                )
            elif budget is None and fraction is None:
                record(
                    "gpu_budget_consistency",
                    "required",
                    "warn",
                    detail="no GPU process budget set; GPU launches fail closed until FAAR_GPU_BUDGET_GB is chosen",
                )
            else:
                record(
                    "gpu_budget_consistency",
                    "required",
                    "pass",
                    detail=f"budget={budget or fraction:.2f} GiB within {total_gib:.2f} GiB",
                    measurement={"budget_gib": budget if budget is not None else (fraction or 0.0) * total_gib},
                )
        reserve_raw = _positive_gb_env("FAAR_MIN_GPU_FREE_GB")
        reserve = reserve_raw if reserve_raw is not None else round(total_gib * 0.2, 2)
        free_gib = (devices[0].get("free_memory_bytes") or 0) / GIB
        budget_gib = budget if budget is not None else ((fraction or 0.0) * total_gib if fraction is not None else 0.0)
        required_gib = budget_gib + reserve
        if free_gib >= required_gib:
            record(
                "gpu_free_vram",
                "warning",
                "pass",
                detail=f"free {free_gib:.2f} GiB >= budget+reserve {required_gib:.2f} GiB",
                measurement={"free_gib": round(free_gib, 2), "reserve_gib": reserve, "required_gib": round(required_gib, 2)},
            )
        else:
            record(
                "gpu_free_vram",
                "warning",
                "warn",
                detail=f"free {free_gib:.2f} GiB below budget+reserve {required_gib:.2f} GiB; co-tenant state may change",
                measurement={"free_gib": round(free_gib, 2), "reserve_gib": reserve, "required_gib": round(required_gib, 2)},
            )

    memory = payload.get("memory") or {}
    scheduler = payload.get("scheduler") or {}
    ram_limit = memory.get("cgroup_limit_bytes") or scheduler.get("slurm_mem_per_node_bytes") or scheduler.get("pbs_mem_bytes")
    per_cpu = scheduler.get("slurm_mem_per_cpu_bytes")
    cpus = scheduler.get("slurm_cpus_per_task") or scheduler.get("slurm_cpus_on_node") or scheduler.get("pbs_ncpus")
    if per_cpu and cpus:
        per_cpu_total = per_cpu * int(cpus)
        ram_limit = per_cpu_total if not ram_limit else min(ram_limit, per_cpu_total)
    if ram_limit:
        record(
            "ram_limit",
            "required",
            "pass",
            detail=f"{ram_limit / GIB:.1f} GiB available to the process",
            measurement={"limit_bytes": ram_limit},
        )
    else:
        record(
            "ram_limit",
            "required",
            "warn",
            detail="no scheduler/cgroup RAM limit detected; launcher will fall back to 50% of physical memory",
        )

    omp = os.getenv("OMP_NUM_THREADS")
    mkl = os.getenv("MKL_NUM_THREADS")
    if omp and mkl:
        record("thread_limits", "warning", "pass", detail=f"OMP_NUM_THREADS={omp} MKL_NUM_THREADS={mkl}")
    else:
        record(
            "thread_limits",
            "warning",
            "warn",
            detail="OMP_NUM_THREADS/MKL_NUM_THREADS not set; launcher derives them at launch",
        )

    dataset_checks: list[tuple[str, Path]] = [
        (SPLIT_RELATIVE_PATH.as_posix(), project_root / SPLIT_RELATIVE_PATH),
        ("qas_v2.json", project_root / "OHR-Bench/data/qas_v2.json"),
    ]
    missing_paths = [label for label, candidate in dataset_checks if not candidate.is_file()]
    try:
        resolve_dataset_paths(
            project_root=project_root,
            inventory=env_raw("FAAR_DOCUMENT_INVENTORY"),
            pdf_root=env_raw("FAAR_PDF_ROOT"),
            pdf_zip=env_raw("FAAR_PDF_ZIP"),
            require_pdf_source=True,
        )
    except DatasetPathError as exc:
        missing_paths.append(str(exc))
    if missing_paths:
        record("dataset_paths", "required", "fail", detail="missing: " + ", ".join(missing_paths))
    else:
        record("dataset_paths", "required", "pass", detail="all dataset paths present")

    lock_path = project_root / "config/split_checksums.json"
    if not lock_path.is_file():
        record("locked_split_checksums", "required", "fail", detail=f"missing lock: {lock_path}")
    else:
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record("locked_split_checksums", "required", "fail", detail=f"unreadable lock: {lock_path}")
        else:
            mismatches = []
            measurements = {}
            for relative, key in (
                (SPLIT_RELATIVE_PATH.as_posix(), "split_sha256"),
                ("OHR-Bench/data/qas_v2.json", "qas_v2_sha256"),
            ):
                expected = lock.get(key) if isinstance(lock, dict) else None
                source = project_root / relative
                if not source.is_file() or not isinstance(expected, str):
                    mismatches.append(relative)
                    continue
                actual = hashlib.sha256(source.read_bytes()).hexdigest()
                measurements[relative] = actual
                if actual != expected:
                    mismatches.append(relative)
            if mismatches:
                record("locked_split_checksums", "required", "fail", detail="checksum mismatch: " + ", ".join(mismatches), measurement=measurements)
            else:
                record("locked_split_checksums", "required", "pass", detail="split and qas_v2 match the committed lock", measurement=measurements)

    cache_paths: list[tuple[str, Path]] = []
    for env_name in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        raw = os.getenv(env_name)
        if raw and raw.strip():
            cache_paths.append((env_name, Path(raw).expanduser()))
    if not cache_paths:
        cache_paths.append(("default", Path.home() / ".cache/huggingface"))
    cache_failures = []
    cache_warnings = []
    for label, cache_path in cache_paths:
        writable, error = _writable_check(cache_path)
        if cache_path.exists() and not writable:
            cache_failures.append(f"{label}:{cache_path}")
        elif not cache_path.exists():
            cache_warnings.append(f"{label}:{cache_path} (created lazily)")
    if cache_failures:
        record("model_cache_paths", "required", "fail", detail="not writable: " + ", ".join(cache_failures))
    elif cache_warnings:
        record("model_cache_paths", "required", "warn", detail="; ".join(cache_warnings))
    else:
        record("model_cache_paths", "required", "pass", detail="cache paths writable")

    output_paths: list[tuple[str, Path]] = []
    for env_name in ("FAAR_OUT_ROOT", "FAAR_SCRATCH", "TMPDIR", "TMP", "SCRATCH", "SCRATCH_DIR", "LOCAL_SCRATCH"):
        raw = os.getenv(env_name)
        if raw and raw.strip():
            output_paths.append((env_name, Path(raw).expanduser()))
    if not output_paths:
        output_paths.append(("default results/", project_root / "results"))
    output_failures = []
    output_warnings = []
    for label, output_path in output_paths:
        writable, error = _writable_check(output_path)
        if output_path.exists() and not writable:
            output_failures.append(f"{label}:{output_path}")
        elif not output_path.exists():
            output_warnings.append(f"{label}:{output_path} (created by jobs)")
    if output_failures:
        record("output_scratch_paths", "required", "fail", detail="not writable: " + ", ".join(output_failures))
    elif output_warnings:
        record("output_scratch_paths", "required", "warn", detail="; ".join(output_warnings))
    else:
        record("output_scratch_paths", "required", "pass", detail="output/scratch paths writable")

    disk_target = path or project_root
    try:
        free_gib = shutil.disk_usage(disk_target).free / GIB
    except OSError:
        free_gib = None
    if free_gib is not None and free_gib < DISK_WARNING_GIB:
        record(
            "disk_space",
            "warning",
            "warn",
            detail=f"only {free_gib:.1f} GiB free on {disk_target}",
            measurement={"free_gib": round(free_gib, 1)},
        )
    else:
        record(
            "disk_space",
            "warning",
            "pass",
            detail=f"{free_gib:.1f} GiB free on {disk_target}" if free_gib is not None else "unmeasurable",
        )

    hf_statuses = []
    if os.getenv("FAAR_PREFLIGHT_OFFLINE"):
        record("hf_model_access", "warning", "skip", detail="network checks disabled by FAAR_PREFLIGHT_OFFLINE")
    else:
        for entry in _model_lock(project_root):
            reachable, detail = _hf_repo_reachable(entry["repository"], entry["revision"])
            hf_statuses.append({**entry, "reachable": reachable, "detail": detail})
            if reachable:
                record("hf_model_access", "warning", "pass", detail=f"{entry['repository']} reachable")
            else:
                record(
                    "hf_model_access",
                    "warning",
                    "warn",
                    detail=f"{entry['repository']}: {detail} (gated models need HF_TOKEN; network may be blocked)",
                )
        if not hf_statuses:
            record("hf_model_access", "warning", "warn", detail="no model lock found; cannot verify HF access")

    vlm_backend = os.getenv("VLM_BACKEND", "openai")
    if vlm_backend in ANTHROPIC_VLM_BACKENDS:
        required_key = "ANTHROPIC_API_KEY"
    else:
        required_key = "OPENAI_API_KEY"
    key_present = bool(os.getenv(required_key, "").strip())
    if key_present:
        record(
            "api_keys_by_stage",
            "warning",
            "pass",
            detail="required VLM credential present (value never shown)",
            measurement={"required_for": "B1/B2/B4 paid VLM stages", "present": True},
        )
    else:
        record(
            "api_keys_by_stage",
            "warning",
            "warn",
            detail="required VLM credential absent; only needed for paid VLM stages (B1/B2/B4), never for B0 or asset preparation",
            measurement={"required_for": "B1/B2/B4 paid VLM stages", "present": False},
        )

    executables: list[tuple[str, bool]] = [("nvidia-smi", cuda)]
    for executable, required in executables + [("sbatch", False), ("srun", False), ("qsub", False), ("pdftotext", False)]:
        available = shutil.which(executable) is not None
        if required:
            if not available:
                record("external_executables", "required", "fail", detail=f"{executable} not found")
            else:
                record("external_executables", "required", "pass", detail=f"{executable} available")
        elif not available:
            record("external_executables", "warning", "warn", detail=f"{executable} not found (optional on this host)")
        else:
            record("external_executables", "warning", "pass", detail=f"{executable} available")

    blocking_failed = any(
        check["status"] == "fail" and check["severity"] == "required" for check in checks
    )
    if blocking_failed:
        exit_code = 1
    elif any(check["status"] in {"fail", "warn"} for check in checks):
        exit_code = 2
    else:
        exit_code = 0

    relevant = {key: os.getenv(key) for key in RELEVANT_ENV_KEYS if os.getenv(key) is not None}
    presence = {key: os.getenv(key) is not None for key in RELEVANT_ENV_KEYS_PRESENCE_ONLY}
    return {
        "schema_version": 3,
        "mode": "check",
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "exit_code": exit_code,
        "summary": {
            "passed": sum(check["status"] == "pass" for check in checks),
            "warnings": sum(check["status"] in {"warn", "fail"} and check["severity"] != "required" for check in checks)
            + sum(check["status"] == "warn" and check["severity"] == "required" for check in checks),
            "failed": sum(check["status"] == "fail" and check["severity"] == "required" for check in checks),
            "skipped": sum(check["status"] == "skip" for check in checks),
        },
        "checks": checks,
        "environment": relevant,
        "secret_presence": presence,
        "facts": payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path.cwd(), help="Filesystem path whose quota should be measured")
    parser.add_argument("--project-root", type=Path, default=None, help="Repository root for dataset and lock checks (default: --path)")
    parser.add_argument("--out", type=Path, help="Optional JSON output path, written atomically (raw facts without --check; the checks report with --check)")
    parser.add_argument("--check", action="store_true", help="Run handoff checks with exit codes and a machine-readable report")
    parser.add_argument("--no-cuda", action="store_true", help="Skip all CUDA/GPU checks (safe on login nodes)")
    parser.add_argument("--dry-run", action="store_true", help="Print the report without writing any file")
    args = parser.parse_args(argv)
    cuda = not args.no_cuda
    measured_path = args.path.expanduser().resolve()
    project_root = (args.project_root or args.path).expanduser().resolve()
    load_project_dotenv(project_root)
    payload = collect(measured_path, cuda=cuda)
    if not args.check:
        rendered = render(payload)
        print(rendered)
        if args.out and not args.dry_run:
            _atomic_write_text(args.out, rendered + "\n")
        return 0
    report = run_checks(payload, project_root=project_root, path=measured_path, cuda=cuda)
    rendered = render(report)
    print(rendered)
    if args.out and not args.dry_run:
        _atomic_write_text(args.out, rendered + "\n")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
