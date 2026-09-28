"""Checks that tests/conftest.py keeps the suite away from credentials, network and real logs."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from faar.api_logging import ApiCallLogger, new_record

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"


def test_provider_credentials_are_removed_before_each_test() -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "HF_TOKEN", "WANDB_PROJECT"):
        assert name not in os.environ


def test_logger_aimed_at_real_logs_directory_writes_to_a_temporary_path() -> None:
    real_path = LOGS_DIR / "vlm_calls.jsonl"
    existed = real_path.exists()
    size = real_path.stat().st_size if existed else None

    logger = ApiCallLogger(real_path)
    logger.log(new_record(provider="mock", model="m", operation="isolation_check", status="started"))

    assert LOGS_DIR not in logger.log_path.parents
    assert logger.log_path.name == "vlm_calls.jsonl"
    assert "isolation_check" in logger.log_path.read_text(encoding="utf-8")
    assert real_path.exists() == existed
    assert (real_path.stat().st_size if existed else None) == size


def test_logger_with_its_own_path_is_left_alone(tmp_path: Path) -> None:
    own_path = tmp_path / "logs/vlm_calls.jsonl"
    logger = ApiCallLogger(own_path)
    assert logger.log_path == own_path


def test_external_hosts_are_refused_and_loopback_still_works() -> None:
    with pytest.raises(OSError, match="blocked in offline tests"):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(OSError, match="blocked in offline tests"):
        socket.socket().connect(("93.184.216.34", 443))

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass


def test_session_guard_fails_the_run_when_a_test_writes_to_logs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "logs").mkdir()
    (project / "logs/existing.jsonl").write_text("{}\n", encoding="utf-8")
    shutil.copy(ROOT / "tests/conftest.py", project / "tests/conftest.py")
    (project / "tests/test_writes_logs.py").write_text(
        "from pathlib import Path\n"
        "def test_writes_real_log():\n"
        "    (Path(__file__).resolve().parents[1] / 'logs/leak.jsonl').write_text('x')\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "tests changed the real logs/ directory: ['leak.jsonl']" in output, output
