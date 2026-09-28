"""Checks for scripts/experiments/registry.py.

Failure modes covered, written before the script:
1. The committed registry stops validating or a recorded output checksum drifts.
2. A line is not valid JSON.
3. A required key is omitted instead of recorded as null.
4. kind or status is outside the documented enums.
5. Top-level status disagrees with the latest attempt.
6. A later snapshot of a run changes its identity (it should be a new run).
7. A resume is recorded after an identity file changed on disk.
8. A resume keeps the run_id and adds a numbered attempt.
9. A tracked output is missing; a missing untracked output is only a warning.
"""

from __future__ import annotations

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = spec_from_file_location("registry", ROOT / "scripts/experiments/registry.py")
registry = module_from_spec(_spec)
_spec.loader.exec_module(registry)


def _record(tmp_path: Path, **overrides) -> dict:
    config = tmp_path / "config.json"
    config.write_text('{"seed": 1}\n')
    output = tmp_path / "out.json"
    output.write_text('{"ok": true}\n')
    record = {
        "schema_version": 1, "run_id": "test-run", "recorded_at": "2026-09-28T00:00:00Z",
        "kind": "development_pilot", "status": "completed", "purpose": "fixture",
        "dataset": {"name": "fixture", "split": None, "pilot_id": None},
        "identity": {"files": {"config.json": hashlib.sha256(config.read_bytes()).hexdigest()}, "settings": {}},
        "command": "python run.py", "environment": None, "seed": 1, "models": None,
        "attempts": [{"attempt": 1, "status": "completed", "code_commit": None, "dirty": None,
                      "started_at": None, "ended_at": None, "note": None}],
        "outputs": [{"path": "out.json", "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "in_git": True, "backup": "git"}],
        "checkpoint": None, "related_runs": [], "limitations": [], "evidence": [],
    }
    record.update(overrides)
    return record


def _write(path: Path, *records: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_committed_registry_is_valid() -> None:
    errors, _ = registry.check(ROOT, ROOT / "experiments/registry.jsonl")
    assert errors == []


def test_invalid_json_line_is_rejected(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    reg.write_text("{not json\n")
    with pytest.raises(registry.RegistryError):
        registry.check(tmp_path, reg)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"kind": "benchmark"}, "kind"),
        ({"status": "done"}, "status"),
        ({"status": "failed"}, "disagrees"),
    ],
)
def test_enum_and_status_problems(tmp_path: Path, overrides: dict, fragment: str) -> None:
    errors, _ = registry.check(tmp_path, _write(tmp_path / "r.jsonl", _record(tmp_path, **overrides)))
    assert any(fragment in e for e in errors)


def test_omitted_key_is_rejected(tmp_path: Path) -> None:
    record = _record(tmp_path)
    del record["seed"]
    errors, _ = registry.check(tmp_path, _write(tmp_path / "r.jsonl", record))
    assert any("missing key 'seed'" in e for e in errors)


def test_identity_change_under_same_run_id_is_rejected(tmp_path: Path) -> None:
    first = _record(tmp_path)
    second = _record(tmp_path, seed=2)
    errors, _ = registry.check(tmp_path, _write(tmp_path / "r.jsonl", first, second))
    assert any("changes identity" in e for e in errors)


def test_output_checksum_drift_is_detected(tmp_path: Path) -> None:
    reg = _write(tmp_path / "r.jsonl", _record(tmp_path))
    (tmp_path / "out.json").write_text('{"ok": false}\n')
    errors, _ = registry.check(tmp_path, reg)
    assert any("no longer matches" in e for e in errors)


def test_missing_outputs_error_only_when_tracked(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record["outputs"].append({"path": "bulk/pages", "sha256": None, "in_git": False, "backup": "none"})
    reg = _write(tmp_path / "r.jsonl", record)
    errors, warnings = registry.check(tmp_path, reg)
    assert errors == [] and any("bulk/pages" in w for w in warnings)
    (tmp_path / "out.json").unlink()
    errors, _ = registry.check(tmp_path, reg)
    assert any("out.json is not present" in e for e in errors)


def test_resume_keeps_identity_and_numbers_attempts(tmp_path: Path) -> None:
    reg = _write(tmp_path / "r.jsonl", _record(tmp_path, status="interrupted", attempts=[
        {"attempt": 1, "status": "interrupted", "code_commit": None, "dirty": None, "started_at": None, "ended_at": None, "note": None}]))
    registry.cmd_attempt(tmp_path, reg, "test-run", "running", None, "resume after preemption")
    registry.cmd_finish(tmp_path, reg, "test-run", "completed", None, None, ["out.json"], "none")
    current = registry.latest(registry.load(reg))["test-run"]
    assert [a["attempt"] for a in current["attempts"]] == [1, 2]
    assert current["status"] == current["attempts"][-1]["status"] == "completed"
    assert len(registry.load(reg)) == 3  # earlier snapshots are kept
    errors, _ = registry.check(tmp_path, reg)
    assert errors == []


def test_resume_refused_when_identity_file_changed(tmp_path: Path) -> None:
    reg = _write(tmp_path / "r.jsonl", _record(tmp_path, status="interrupted", attempts=[
        {"attempt": 1, "status": "interrupted", "code_commit": None, "dirty": None, "started_at": None, "ended_at": None, "note": None}]))
    (tmp_path / "config.json").write_text('{"seed": 2}\n')
    with pytest.raises(registry.RegistryError) as exc:
        registry.cmd_attempt(tmp_path, reg, "test-run", "running", None, None)
    assert exc.value.code == registry.EXIT_IDENTITY
    assert len(registry.load(reg)) == 1
