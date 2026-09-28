#!/usr/bin/env python3
"""Check whether faiss and torch can share one process without an OpenMP conflict.

On macOS the faiss-cpu and torch wheels each bundle their own libomp.dylib. This
script records the installed versions, finds the OpenMP runtimes each package
ships, and runs a small matrix of fresh subprocesses. Each subprocess does one
FAISS search and a few multi-threaded torch operations, in both import orders,
under different thread and environment settings. It prints one JSON report.

The workloads are tiny (20000 x 64 float vectors, 768 x 768 matmuls). The script
downloads nothing, calls no model and touches no repository data. On Linux it
runs the same matrix; the wheels there normally share one libgomp, so the default
case is expected to pass.

Usage:
    python scripts/diagnostics/openmp_check.py [--trials 5] [--timeout 20] [--indent 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
from importlib import metadata
from typing import Any

# Distributions whose wheels may bundle an OpenMP runtime.
DISTRIBUTIONS = ("faiss-cpu", "torch", "scikit-learn", "numpy", "scipy")
OPENMP_FILE = re.compile(r"(^|/)lib(g?omp|iomp5?)[-.\w]*\.(dylib|so)([.\d]*)$")
OPENMP_MODULES = ("faiss", "torch", "sklearn")
ORDERS = ("faiss,torch", "torch,faiss")

# Variables removed from every child so the case controls them exactly.
CONTROLLED_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_DUPLICATE_LIB_OK")

# name -> (environment overrides, libraries the worker limits to one thread through their APIs)
CASES: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "default": ({}, ()),
    "kmp_only": ({"KMP_DUPLICATE_LIB_OK": "TRUE"}, ()),
    "api_limit_only": ({}, ("faiss", "torch")),
    "omp_env_only": ({"OMP_NUM_THREADS": "1"}, ()),
    "kmp_and_api_limit": ({"KMP_DUPLICATE_LIB_OK": "TRUE"}, ("faiss", "torch")),
    "kmp_and_omp_env": ({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}, ()),
    "api_limit_faiss_only": ({}, ("faiss",)),
    "api_limit_torch_only": ({}, ("torch",)),
}

SIGNAL_NAMES = {int(sig): sig.name for sig in signal.Signals}


# ---------------------------------------------------------------- parsing helpers


def is_openmp_library(path: str) -> bool:
    """Return True when the path names an OpenMP runtime such as libomp, libgomp or libiomp5."""
    return OPENMP_FILE.search(path.replace("\\", "/")) is not None


def filter_openmp_images(paths: list[str]) -> list[str]:
    """Keep the loaded images that are OpenMP runtimes, in load order, without duplicates."""
    seen: list[str] = []
    for path in paths:
        if is_openmp_library(path) and path not in seen:
            seen.append(path)
    return seen


def shorten(path: str) -> str:
    """Drop everything up to site-packages so reports do not depend on the checkout location."""
    marker = "site-packages/"
    return path.split(marker, 1)[1] if marker in path else path


def classify_outcome(returncode: int | None, timed_out: bool, stdout: str, stderr: str) -> dict[str, Any]:
    """Turn a finished child process into a short outcome label.

    Labels: ok, timeout, omp_error_15, signal:<NAME> and exit:<N>. A child counts
    as ok only when it exits 0 and prints the DONE marker.
    """
    if timed_out:
        return {"outcome": "timeout", "returncode": None, "signal": None}
    assert returncode is not None
    sig = SIGNAL_NAMES.get(-returncode) if returncode < 0 else None
    if "OMP: Error #15" in stderr:
        label = "omp_error_15"
    elif sig:
        label = f"signal:{sig}"
    elif returncode == 0 and "DONE" in stdout.split():
        label = "ok"
    else:
        label = f"exit:{returncode}"
    return {"outcome": label, "returncode": returncode, "signal": sig}


def stderr_summary(stderr: str) -> str:
    """Return the most telling stderr line: the libomp error if there is one, else the last line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("OMP: Error"):
            return line[:200]
    return lines[-1][:200] if lines else ""


def summarize_trials(outcomes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in outcomes:
        counts[label] = counts.get(label, 0) + 1
    return counts


def verdict(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    good = counts.get("ok", 0)
    if total == 0:
        return "not_run"
    if good == total:
        return "pass"
    if good == 0:
        return "fail"
    return "flaky"


# ---------------------------------------------------------------- environment facts


def loaded_images() -> list[str]:
    """List the shared libraries mapped into this process, in load order where the OS gives one."""
    if sys.platform == "darwin":
        import ctypes

        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        return [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    if sys.platform.startswith("linux"):
        paths: list[str] = []
        with open("/proc/self/maps", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].strip().startswith("/") and parts[5].strip() not in paths:
                    paths.append(parts[5].strip())
        return paths
    return []


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_name(path: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(["otool", "-D", path], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in result.stdout.splitlines()[1:] if line.strip()]
    return lines[0] if lines else None


def package_facts() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    bundled: list[dict[str, Any]] = []
    for name in DISTRIBUTIONS:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = dist.version
        for file in dist.files or []:
            if not is_openmp_library(str(file)):
                continue
            located = str(dist.locate_file(file))
            if not os.path.isfile(located):
                continue
            bundled.append(
                {
                    "distribution": name,
                    "file": str(file),
                    "sha256": sha256(located),
                    "bytes": os.path.getsize(located),
                    "install_name": install_name(located),
                }
            )
    return {"versions": versions, "bundled_openmp": bundled}


def platform_facts() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------- child processes


def worker(order: str, limited: set[str]) -> int:
    """Run the FAISS and torch workloads in the given order inside this process."""
    import numpy as np

    def run_faiss() -> None:
        import faiss

        if "faiss" in limited:
            faiss.omp_set_num_threads(1)
        rng = np.random.default_rng(0)
        vectors = rng.random((20000, 64), dtype=np.float32)
        queries = rng.random((500, 64), dtype=np.float32)
        index = faiss.IndexFlatL2(64)
        index.add(vectors)
        index.search(queries, 5)
        faiss.Kmeans(64, 16, niter=3, seed=1).train(vectors)
        print("faiss ok", flush=True)

    def run_torch() -> None:
        import torch

        if "torch" in limited:
            torch.set_num_threads(1)
        left = torch.randn(768, 768)
        right = torch.randn(768, 768)
        for _ in range(3):
            (left @ right).relu().sum()
        torch.nn.functional.softmax(torch.randn(256, 4096), dim=-1).sum()
        print("torch ok", flush=True)

    steps = {"faiss": run_faiss, "torch": run_torch}
    for step in order.split(","):
        steps[step]()
    print("DONE", flush=True)
    return 0


def import_order_probe(order: str) -> int:
    """Print the OpenMP runtimes mapped after each import, as JSON lines."""
    for module in order.split(","):
        __import__(module)
        print(json.dumps({"after": module, "openmp": filter_openmp_images(loaded_images())}), flush=True)
    print("DONE", flush=True)
    return 0


def child_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in CONTROLLED_ENV}
    env.update(overrides)
    return env


def run_child(args: list[str], env: dict[str, str], timeout: float) -> dict[str, Any]:
    """Run one child in its own session so a hang can be killed with everything it spawned."""
    process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    result = classify_outcome(process.returncode, timed_out, stdout, stderr)
    result["stderr_tail"] = stderr_summary(stderr)
    result["_stdout"] = stdout
    return result


def probe_import(order: str, timeout: float) -> dict[str, Any]:
    result = run_child(["--imports", order], child_env({}), timeout)
    stdout = result.pop("_stdout")
    steps = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    return {
        "order": order,
        "outcome": result["outcome"],
        "returncode": result["returncode"],
        "loaded_openmp_after_each_import": [
            {"after": step["after"], "openmp": [shorten(path) for path in step["openmp"]]} for step in steps
        ],
        "stderr_tail": result["stderr_tail"],
    }


def run_matrix(trials: int, timeout: float, orders: tuple[str, ...], cases: list[str]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        overrides, limited = CASES[case]
        for order in orders:
            args = ["--worker", order] + (["--limit", ",".join(limited)] if limited else [])
            outcomes: list[dict[str, Any]] = []
            for _ in range(trials):
                result = run_child(args, child_env(overrides), timeout)
                result.pop("_stdout")
                outcomes.append(result)
            counts = summarize_trials([item["outcome"] for item in outcomes])
            rows.append(
                {
                    "case": case,
                    "order": order,
                    "env": overrides,
                    "api_thread_limit": list(limited),
                    "trials": trials,
                    "counts": counts,
                    "verdict": verdict(counts),
                    "returncodes": [item["returncode"] for item in outcomes],
                    "signals": sorted({item["signal"] for item in outcomes if item["signal"]}),
                    "stderr_tail": next((item["stderr_tail"] for item in outcomes if item["stderr_tail"]), ""),
                }
            )
    return rows


def build_report(trials: int, timeout: float, cases: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"platform": platform_facts()}
    report.update(package_facts())
    versions = report["versions"]
    missing = [name for name in ("faiss-cpu", "torch") if versions.get(name) is None]
    if missing:
        report["matrix"] = "not applicable: " + ", ".join(missing) + " not installed"
        return report
    report["imports"] = [probe_import(order, timeout) for order in ORDERS]
    report["matrix"] = run_matrix(trials, timeout, ORDERS, cases)
    default_rows = [row for row in report["matrix"] if row["case"] == "default"]
    report["summary"] = {
        "default_case_passes_in_every_order": all(row["verdict"] == "pass" for row in default_rows),
        "cases_passing_in_every_order": [
            case
            for case in cases
            if all(row["verdict"] == "pass" for row in report["matrix"] if row["case"] == case)
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=5, help="fresh subprocesses per case and order (default 5)")
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds before a trial counts as hung")
    parser.add_argument("--cases", default=",".join(CASES), help="comma-separated subset of: " + ", ".join(CASES))
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--worker", metavar="ORDER", help=argparse.SUPPRESS)
    parser.add_argument("--limit", default="", help=argparse.SUPPRESS)
    parser.add_argument("--imports", metavar="ORDER", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, {name for name in args.limit.split(",") if name})
    if args.imports:
        return import_order_probe(args.imports)
    cases = [name for name in args.cases.split(",") if name]
    unknown = [name for name in cases if name not in CASES]
    if unknown:
        parser.error("unknown case: " + ", ".join(unknown))
    print(json.dumps(build_report(args.trials, args.timeout, cases), indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
