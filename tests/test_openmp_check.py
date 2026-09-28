"""Failure list for the parsing and report logic of scripts/diagnostics/openmp_check.py.

The script decides whether an OpenMP conflict happened from a child process's exit
status. A wrong label would hide or invent a crash, so these cases cover the ways
the label or the library filter can be wrong:

1. A negative return code is read as a plain exit code instead of a signal.
2. A hung child (no return code) is counted as a pass.
3. A child that exits 0 without printing its DONE marker is counted as a pass.
4. The libomp "Error #15" abort is reported as a generic signal.
5. The library filter matches names that only contain "omp" (libcompression,
   libcompiler_rt) or misses the mangled libgomp name of Linux wheels.
6. A verdict hides a partial failure (flaky) or reports a pass for zero trials.
7. Missing faiss or torch crashes the report instead of saying "not applicable".

The script is loaded by path: scripts/diagnostics is not on the test import path.
"""

from __future__ import annotations

import importlib.util
import signal
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnostics/openmp_check.py"
spec = importlib.util.spec_from_file_location("openmp_check", SCRIPT)
assert spec is not None and spec.loader is not None
openmp_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openmp_check)


@pytest.mark.parametrize(
    ("returncode", "label", "sig"),
    [
        (-int(signal.SIGSEGV), "signal:SIGSEGV", "SIGSEGV"),
        (-int(signal.SIGABRT), "signal:SIGABRT", "SIGABRT"),
        (3, "exit:3", None),
        (-250, "exit:-250", None),
    ],
)
def test_nonzero_status_becomes_signal_or_exit_label(returncode: int, label: str, sig: str | None) -> None:
    result = openmp_check.classify_outcome(returncode, False, "faiss ok\n", "")
    assert result["outcome"] == label
    assert result["signal"] == sig


def test_timeout_is_never_ok() -> None:
    result = openmp_check.classify_outcome(None, True, "faiss ok\nDONE\n", "")
    assert result["outcome"] == "timeout"
    assert result["returncode"] is None


def test_exit_zero_without_done_marker_is_not_ok() -> None:
    assert openmp_check.classify_outcome(0, False, "faiss ok\n", "")["outcome"] == "exit:0"
    assert openmp_check.classify_outcome(0, False, "faiss ok\ntorch ok\nDONE\n", "")["outcome"] == "ok"


def test_libomp_error_15_beats_the_signal_label() -> None:
    stderr = "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.\n"
    result = openmp_check.classify_outcome(-int(signal.SIGABRT), False, "", stderr)
    assert result["outcome"] == "omp_error_15"
    assert result["signal"] == "SIGABRT"


@pytest.mark.parametrize(
    "path",
    [
        "/env/lib/python3.12/site-packages/torch/lib/libomp.dylib",
        "/env/lib/python3.12/site-packages/faiss/.dylibs/libomp.dylib",
        "/env/lib/python3.12/site-packages/faiss_cpu.libs/libgomp-a34b3233.so.1.0.0",
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        "/opt/intel/lib/libiomp5.dylib",
    ],
)
def test_openmp_runtimes_are_recognised(path: str) -> None:
    assert openmp_check.is_openmp_library(path)


@pytest.mark.parametrize(
    "path",
    [
        "/usr/lib/libcompression.dylib",
        "/usr/lib/system/libcompiler_rt.dylib",
        "/env/lib/python3.12/site-packages/pyarrow/libarrow_compute.2500.dylib",
        "/env/lib/python3.12/site-packages/numpy/.dylibs/libgfortran.5.dylib",
    ],
)
def test_names_that_only_contain_omp_are_not_openmp_runtimes(path: str) -> None:
    assert not openmp_check.is_openmp_library(path)


def test_filter_keeps_load_order_and_drops_duplicates() -> None:
    paths = ["/a/libomp.dylib", "/usr/lib/libSystem.B.dylib", "/b/libomp.dylib", "/a/libomp.dylib"]
    assert openmp_check.filter_openmp_images(paths) == ["/a/libomp.dylib", "/b/libomp.dylib"]


def test_shorten_strips_the_environment_prefix() -> None:
    assert openmp_check.shorten("/x/y/site-packages/torch/lib/libomp.dylib") == "torch/lib/libomp.dylib"
    assert openmp_check.shorten("/usr/lib/libgomp.so.1") == "/usr/lib/libgomp.so.1"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"ok": 5}, "pass"),
        ({"timeout": 5}, "fail"),
        ({"ok": 3, "signal:SIGSEGV": 2}, "flaky"),
        ({}, "not_run"),
    ],
)
def test_verdict_reports_partial_failure(counts: dict[str, int], expected: str) -> None:
    assert openmp_check.verdict(counts) == expected


def test_summarize_trials_counts_labels() -> None:
    assert openmp_check.summarize_trials(["ok", "timeout", "ok"]) == {"ok": 2, "timeout": 1}


def test_report_says_not_applicable_when_faiss_or_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = {"versions": {"faiss-cpu": None, "torch": "2.6.0"}, "bundled_openmp": []}
    monkeypatch.setattr(openmp_check, "package_facts", lambda: facts)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("no child process may run when a package is missing")

    monkeypatch.setattr(openmp_check, "run_child", fail)
    report = openmp_check.build_report(trials=1, timeout=1.0, cases=["default"])
    assert report["matrix"] == "not applicable: faiss-cpu not installed"
    assert "summary" not in report


def test_stderr_summary_prefers_the_libomp_error_line() -> None:
    stderr = "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.\nOMP: Hint This means\n"
    assert openmp_check.stderr_summary(stderr).startswith("OMP: Error #15")
    assert openmp_check.stderr_summary("warning\nlast line\n") == "last line"
    assert openmp_check.stderr_summary("") == ""
