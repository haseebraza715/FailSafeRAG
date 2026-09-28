"""Validate and append run records in experiments/registry.jsonl.

The registry is append-only JSON Lines. Each line is a complete snapshot of one
run record; the last line for a run_id is its current state and earlier lines
are its history. The format is documented in experiments/README.md.

    check                      validate every line; verify recorded output checksums
    add RECORD.json            append a new run (run_id must be unused)
    attempt RUN_ID --status S  append a new attempt (resume) after checking identity files are unchanged
    finish RUN_ID --status S   close the latest attempt and optionally hash outputs

Exit codes: 0 ok, 1 usage or I/O error, 2 invalid registry or record, 3 identity changed.
The script never runs experiments and never edits result payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KINDS = {"data_preparation", "engineering_check", "development_pilot", "scientific_evaluation"}
STATUSES = {"planned", "running", "interrupted", "failed", "completed", "abandoned"}
REQUIRED = (
    "schema_version", "run_id", "recorded_at", "kind", "status", "purpose", "dataset", "identity",
    "command", "environment", "seed", "models", "attempts", "outputs", "checkpoint", "related_runs",
    "limitations", "evidence",
)
ATTEMPT_REQUIRED = ("attempt", "status", "code_commit", "dirty", "started_at", "ended_at", "note")
OUTPUT_REQUIRED = ("path", "sha256", "in_git", "backup")
# Changing any of these means a different experiment, so it needs a new run_id.
IDENTITY_KEYS = ("kind", "dataset", "identity", "command", "seed", "models")
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
EXIT_INVALID, EXIT_IDENTITY = 2, 3


class RegistryError(Exception):
    def __init__(self, message: str, code: int = EXIT_INVALID) -> None:
        super().__init__(message)
        self.code = code


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def validate(record: Any, where: str) -> list[str]:
    """Return schema problems for one record; an empty list means valid."""
    if not isinstance(record, dict):
        return [f"{where}: record is not a JSON object"]
    problems = [f"{where}: missing key {key!r} (record unknown values as null)" for key in REQUIRED if key not in record]
    if problems:
        return problems
    if record["schema_version"] != SCHEMA_VERSION:
        problems.append(f"{where}: unsupported schema_version {record['schema_version']!r}")
    if not isinstance(record["run_id"], str) or not RUN_ID.match(record["run_id"]):
        problems.append(f"{where}: malformed run_id {record['run_id']!r}")
    if record["kind"] not in KINDS:
        problems.append(f"{where}: kind {record['kind']!r} not in {sorted(KINDS)}")
    if record["status"] not in STATUSES:
        problems.append(f"{where}: status {record['status']!r} not in {sorted(STATUSES)}")
    if not isinstance(record["recorded_at"], str) or not TIMESTAMP.match(record["recorded_at"]):
        problems.append(f"{where}: recorded_at must be an ISO 8601 timestamp")
    if not isinstance(record["identity"], dict):
        problems.append(f"{where}: identity must be an object")
    else:
        for path, digest in record["identity"].get("files", {}).items():
            if digest is not None and not SHA256.match(str(digest)):
                problems.append(f"{where}: identity file {path} has a malformed sha256")
    attempts = record["attempts"]
    if not isinstance(attempts, list):
        return problems + [f"{where}: attempts must be a list"]
    if record["status"] != "planned" and not attempts:
        problems.append(f"{where}: status {record['status']} needs at least one attempt")
    for i, attempt in enumerate(attempts, 1):
        missing = [key for key in ATTEMPT_REQUIRED if key not in attempt]
        if missing:
            problems.append(f"{where}: attempt {i} missing {missing}")
            continue
        if attempt["attempt"] != i:
            problems.append(f"{where}: attempts must be numbered 1..n in order")
        if attempt["status"] not in STATUSES - {"planned"}:
            problems.append(f"{where}: attempt {i} status {attempt['status']!r} is invalid")
        for key in ("started_at", "ended_at"):
            if attempt[key] is not None and not TIMESTAMP.match(str(attempt[key])):
                problems.append(f"{where}: attempt {i} {key} must be ISO 8601 or null")
    if attempts and all(key in attempts[-1] for key in ATTEMPT_REQUIRED) and attempts[-1]["status"] != record["status"]:
        problems.append(f"{where}: status {record['status']} disagrees with the latest attempt ({attempts[-1]['status']})")
    if not isinstance(record["outputs"], list):
        return problems + [f"{where}: outputs must be a list"]
    for output in record["outputs"]:
        missing = [key for key in OUTPUT_REQUIRED if key not in output]
        if missing:
            problems.append(f"{where}: output {output.get('path')} missing {missing}")
        elif output["sha256"] is not None and not SHA256.match(str(output["sha256"])):
            problems.append(f"{where}: output {output['path']} has a malformed sha256")
    return problems


def load(registry: Path) -> list[dict[str, Any]]:
    records = []
    for n, line in enumerate(registry.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{registry.name}:{n}: invalid JSON ({exc.msg})") from exc
    return records


def latest(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for record in records:
        current[record["run_id"]] = record
    return current


def check(root: Path, registry: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the whole registry."""
    records = load(registry)
    errors: list[str] = []
    first: dict[str, dict[str, Any]] = {}
    for n, record in enumerate(records, 1):
        where = f"{registry.name}:{n}"
        errors += validate(record, where)
        if not isinstance(record, dict) or "run_id" not in record:
            continue
        if record["run_id"] in first:
            changed = [key for key in IDENTITY_KEYS if record.get(key) != first[record["run_id"]].get(key)]
            if changed:
                errors.append(f"{where}: run {record['run_id']} changes identity fields {changed}; use a new run_id")
        else:
            first[record["run_id"]] = record
    warnings: list[str] = []
    if errors:
        return errors, warnings
    for run_id, record in latest(records).items():
        for output in record["outputs"]:
            path = root / output["path"]
            if not path.exists():
                (errors if output["in_git"] else warnings).append(f"{run_id}: output {output['path']} is not present")
            elif output["sha256"] and path.is_file() and sha256_file(path) != output["sha256"]:
                errors.append(f"{run_id}: output {output['path']} no longer matches its recorded sha256")
    return errors, warnings


def append(registry: Path, record: dict[str, Any]) -> None:
    problems = validate(record, record.get("run_id", "record") if isinstance(record, dict) else "record")
    if problems:
        raise RegistryError("; ".join(problems))
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def identity_drift(root: Path, record: dict[str, Any]) -> list[str]:
    drift = []
    for rel, recorded in record["identity"].get("files", {}).items():
        path = root / rel
        actual = sha256_file(path) if path.is_file() else None
        if recorded is not None and actual != recorded:
            drift.append(rel)
    return drift


def cmd_add(root: Path, registry: Path, record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if registry.exists() and record.get("run_id") in latest(load(registry)):
        raise RegistryError(f"run_id {record.get('run_id')} already exists; use attempt/finish or a new run_id")
    commit, dirty = git_state(root)
    for attempt in record.get("attempts", []):
        if attempt.get("code_commit") == "auto":
            attempt["code_commit"], attempt["dirty"] = commit, dirty
    record["recorded_at"] = now()
    append(registry, record)
    return record


def cmd_attempt(root: Path, registry: Path, run_id: str, status: str, started_at: str | None, note: str | None) -> dict[str, Any]:
    current = latest(load(registry))
    if run_id not in current:
        raise RegistryError(f"unknown run_id {run_id}")
    record = json.loads(json.dumps(current[run_id]))
    drift = identity_drift(root, record)
    if drift:
        raise RegistryError(f"identity files changed since {run_id} was recorded: {drift}; register a new run", EXIT_IDENTITY)
    commit, dirty = git_state(root)
    record["attempts"].append({
        "attempt": len(record["attempts"]) + 1, "status": status, "code_commit": commit, "dirty": dirty,
        "started_at": started_at or now(), "ended_at": None, "note": note,
    })
    record["status"], record["recorded_at"] = status, now()
    append(registry, record)
    return record


def cmd_finish(root: Path, registry: Path, run_id: str, status: str, ended_at: str | None, note: str | None,
               outputs: list[str], backup: str) -> dict[str, Any]:
    current = latest(load(registry))
    if run_id not in current:
        raise RegistryError(f"unknown run_id {run_id}")
    record = json.loads(json.dumps(current[run_id]))
    if not record["attempts"]:
        raise RegistryError(f"{run_id} has no attempt to finish; use attempt first")
    last = record["attempts"][-1]
    last["status"], last["ended_at"] = status, ended_at or now()
    if note:
        last["note"] = note if not last["note"] else f"{last['note']} | {note}"
    known = {output["path"]: output for output in record["outputs"]}
    tracked = set(subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True).stdout.splitlines())
    for rel in outputs:
        path = root / rel
        if not path.is_file():
            raise RegistryError(f"output {rel} is not a file")
        known[rel] = {"path": rel, "sha256": sha256_file(path), "in_git": rel in tracked, "backup": "git" if rel in tracked else backup}
    record["outputs"] = list(known.values())
    record["status"], record["recorded_at"] = status, now()
    append(registry, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.jsonl"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    add = sub.add_parser("add")
    add.add_argument("record", type=Path)
    attempt = sub.add_parser("attempt")
    attempt.add_argument("run_id")
    attempt.add_argument("--status", choices=sorted(STATUSES - {"planned"}), default="running")
    attempt.add_argument("--started-at")
    attempt.add_argument("--note")
    finish = sub.add_parser("finish")
    finish.add_argument("run_id")
    finish.add_argument("--status", choices=sorted(STATUSES - {"planned", "running"}), required=True)
    finish.add_argument("--ended-at")
    finish.add_argument("--note")
    finish.add_argument("--output", action="append", default=[], help="Project-relative output file to hash (repeatable).")
    finish.add_argument("--backup", default="none", help="Backup location for untracked outputs (default: none).")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    registry = args.registry if args.registry.is_absolute() else root / args.registry
    try:
        if args.command == "check":
            errors, warnings = check(root, registry)
            for line in warnings:
                print(f"warning: {line}")
            for line in errors:
                print(f"error: {line}", file=sys.stderr)
            runs = latest(load(registry))
            print(json.dumps({"runs": len(runs), "errors": len(errors), "warnings": len(warnings),
                              "by_status": {s: sum(r["status"] == s for r in runs.values()) for s in sorted(STATUSES)}}, sort_keys=True))
            return EXIT_INVALID if errors else 0
        if args.command == "add":
            record = cmd_add(root, registry, args.record)
        elif args.command == "attempt":
            record = cmd_attempt(root, registry, args.run_id, args.status, args.started_at, args.note)
        else:
            record = cmd_finish(root, registry, args.run_id, args.status, args.ended_at, args.note, args.output, args.backup)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": record["run_id"], "status": record["status"], "attempts": len(record["attempts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
