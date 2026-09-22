from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from statistics import mean

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.gate_tuning import require_paper_gate_threshold
from faar.results_aggregator import summarize_api_usage
from faar.settings import effective_retrieval_provenance

STAGE_ORDER = ("b0", "b1", "b2", "b3", "b4")

STAGE_SPEC = {
    "b0": ("naive_rag", "Text-only RAG"),
    "b1": ("faar_always_vlm", "Always-VLM"),
    "b2": ("faar_no_diagnosis", "Random recovery"),
    "b3": ("colpali", "colpali"),
    "b4": ("visrag", "visrag"),
}

STAGE_ARGS = {
    "b0": ["--gate", "off", "--recovery", "off"],
    "b1": ["--gate", "off", "--recovery", "always_vlm"],
    "b2": ["--gate", "on", "--recovery", "random_type"],
    "b3": ["--mode", "colpali"],
    "b4": ["--mode", "visrag"],
}

RUN_SPEC_MATCH_KEYS = (
    "dataset",
    "split",
    "seed",
    "max_examples",
    "shard_index",
    "num_shards",
    "embedding_model",
    "reranker",
    "ocr_engine",
    "model_provenance",
    "manifest_sha256",
)

SUMMARY_KEYS = (
    "EM",
    "F1",
    "vlm_rate",
    "harm_rate",
    "api_requests",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "runtime_sec",
)
API_COUNT_KEYS = {"api_requests", "prompt_tokens", "completion_tokens"}

_SIGNALS = (signal.SIGTERM, signal.SIGINT)

_CHILD: subprocess.Popen[str] | None = None
_pending_signal: int | None = None
_forwarded_signal: int | None = None


def _forward(signum: int, frame: object) -> None:
    global _pending_signal, _forwarded_signal
    _pending_signal = signum
    _forwarded_signal = signum
    child = _CHILD
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass


def _install_signal_handlers() -> dict[int, object]:
    previous = {}
    for signum in _SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _forward)
    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_baselines")
    parser.add_argument("--dataset", default="ohrbench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--vlm")
    parser.add_argument("--embed")
    parser.add_argument("--reranker")
    parser.add_argument("--ocr")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--faar-python")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default="b4",
        help="Stop after this stage. Only 'b0' skips the B2 gate-lock prerequisite, "
        "so validation B0 can be run before gate tuning.",
    )
    return parser.parse_args(argv)


def project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root).expanduser().resolve()


def results_dir(root: Path, dataset: str, split: str) -> Path:
    return root / "results" / "baselines" / dataset / split


def output_path(root: Path, dataset: str, split: str, stage: str) -> Path:
    return results_dir(root, dataset, split) / f"{stage}.json"


def common_run_args(args: argparse.Namespace, out: Path) -> list[str]:
    run_args = [
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--seed",
        str(args.seed),
        "--out",
        str(out),
    ]
    if args.max_examples is not None:
        run_args += ["--max-examples", str(args.max_examples)]
    for flag, value in (
        ("--vlm", args.vlm),
        ("--embed", args.embed),
        ("--reranker", args.reranker),
        ("--ocr", args.ocr),
    ):
        if value:
            run_args += [flag, value]
    return run_args


def stage_run_args(
    stage: str,
    args: argparse.Namespace,
    out: Path,
    baseline: Path | None,
) -> list[str]:
    run_args = common_run_args(args, out)
    run_args += STAGE_ARGS[stage]
    if args.resume:
        run_args.append("--resume")
    if baseline is not None:
        run_args += ["--baseline", str(baseline)]
    return run_args


def launcher_command(args: argparse.Namespace, run_args: list[str]) -> list[str]:
    root = project_root(args)
    command = [
        args.faar_python or sys.executable,
        str(root / "cluster" / "launcher.py"),
        "--project-root",
        str(root),
    ]
    if args.cpu_only:
        command.append("--cpu-only")
    if args.faar_python:
        command += ["--faar-python", args.faar_python]
    command += run_args
    return command


def validate_gate_lock(
    root: Path,
    *,
    dataset: str | None = None,
    model_provenance: dict | None = None,
) -> dict:
    path = root / "config" / "gate_threshold.json"
    if not path.is_file():
        raise SystemExit(
            f"B2 requires a locked gate threshold at {path}; run scripts/experiments/tune_gate.py on validation results first."
        )
    try:
        return require_paper_gate_threshold(
            path,
            dataset=dataset,
            model_provenance=model_provenance,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"B2 gate lock is invalid at {path}: {exc}") from exc


def _example_ids(payload: dict, path: Path) -> list[str]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} must contain a nonempty rows list.")
    ids: list[str] = []
    for row in rows:
        value = str(row.get("example_id") or "").strip() if isinstance(row, dict) else ""
        if not value:
            raise SystemExit(f"{path} contains a row with a missing example_id.")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path} contains duplicate example_id values.")
    return ids


def _validate_summary(payload: dict, path: Path) -> None:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit(f"{path} has no summary object.")
    for key in SUMMARY_KEYS:
        value = summary.get(key)
        if value is None:
            raise SystemExit(f"{path} summary.{key} is missing or null.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{path} summary.{key} is not numeric.") from exc
        if not math.isfinite(number):
            raise SystemExit(f"{path} summary.{key} is not finite.")
        if key in API_COUNT_KEYS and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise SystemExit(f"{path} summary.{key} must be a nonnegative integer.")
        if key in {"EM", "F1", "vlm_rate", "harm_rate"} and not 0.0 <= number <= 1.0:
            raise SystemExit(f"{path} summary.{key} is outside [0, 1].")
        if key in {"cost_usd", "runtime_sec"} and number < 0.0:
            raise SystemExit(f"{path} summary.{key} is negative.")


def _row_score(row: dict, key: str, path: Path) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or metrics.get(key) is None:
        raise SystemExit(f"{path} row {row.get('example_id')!r} has no metrics.{key}.")
    try:
        score = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path} row {row.get('example_id')!r} has invalid metrics.{key}.") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SystemExit(f"{path} row {row.get('example_id')!r} has metrics.{key} outside [0, 1].")
    return score


def _validate_summary_against_rows(payload: dict, path: Path, baseline_payload: dict | None) -> None:
    rows = payload["rows"]
    em_values = [_row_score(row, "em", path) for row in rows]
    f1_values = [_row_score(row, "f1", path) for row in rows]
    vlm_values = [
        1.0
        if (row.get("action_outcome") or {}).get("action") == "invoke_vlm"
        or (row.get("visual_result") or {}).get("status") == "succeeded"
        else 0.0
        for row in rows
    ]
    harm_values = [0.0 for _ in rows]
    if baseline_payload is not None:
        baseline_by_id = {
            str(row["example_id"]).strip(): row for row in baseline_payload["rows"]
        }
        harm_values = [
            1.0
            if _row_score(row, "f1", path)
            < _row_score(
                baseline_by_id[str(row["example_id"]).strip()],
                "f1",
                path.with_name("b0.json"),
            )
            else 0.0
            for row in rows
        ]
    derived = {
        "EM": round(mean(em_values), 4),
        "F1": round(mean(f1_values), 4),
        "vlm_rate": round(mean(vlm_values), 4),
        "harm_rate": round(mean(harm_values), 4),
    }
    try:
        derived.update(summarize_api_usage(rows))
    except ValueError as exc:
        raise SystemExit(f"{path} contains invalid API accounting: {exc}.") from exc
    for key, expected in derived.items():
        precision = 6 if key == "cost_usd" else 4
        observed = round(float(payload["summary"][key]), precision)
        expected = round(float(expected), precision)
        if observed != expected:
            raise SystemExit(
                f"{path} summary.{key} is {observed}; rows recompute to {expected}."
            )


def _reject_failed_rows(payload: dict, path: Path) -> None:
    failed_ids = [
        str(row.get("example_id") or "<unknown>")
        for row in payload["rows"]
        if isinstance(row, dict)
        and (
            (row.get("action_outcome") or {}).get("status") == "failed"
            or bool(row.get("error"))
        )
    ]
    if failed_ids:
        raise SystemExit(
            f"{path} contains failed rows that cannot be scored or published: {failed_ids}."
        )


def validate_output(
    path: Path,
    stage: str,
    args: argparse.Namespace,
    baseline_payload: dict | None,
) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing stage output: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Stage output is unreadable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Stage output must be a JSON object: {path}")
    if not isinstance(payload.get("rows"), list):
        raise SystemExit(f"{path} has no rows list.")
    _reject_failed_rows(payload, path)
    expected_profile, expected_label = STAGE_SPEC[stage]
    if payload.get("profile") != expected_profile:
        raise SystemExit(
            f"{path} profile is {payload.get('profile')!r}; expected {expected_profile!r}."
        )
    if payload.get("label") != expected_label:
        raise SystemExit(
            f"{path} label is {payload.get('label')!r}; expected {expected_label!r}."
        )
    _validate_summary(payload, path)
    run_spec = payload.get("run_spec")
    if not isinstance(run_spec, dict):
        raise SystemExit(f"{path} has no run_spec provenance.")
    for key, expected in (
        ("dataset", args.dataset),
        ("split", args.split),
        ("seed", args.seed),
        ("max_examples", args.max_examples),
    ):
        if key not in run_spec or run_spec[key] != expected:
            raise SystemExit(
                f"{path} run_spec.{key} is {run_spec.get(key)!r}; expected {expected!r}."
            )
    for key in RUN_SPEC_MATCH_KEYS:
        if key not in run_spec:
            if key == "manifest_sha256":
                continue
            raise SystemExit(f"{path} is missing run_spec.{key} provenance.")
    for key in ("embedding_model", "reranker", "ocr_engine"):
        if not isinstance(run_spec[key], str) or not run_spec[key].strip():
            raise SystemExit(f"{path} run_spec.{key} must be nonempty.")
    if not isinstance(run_spec["model_provenance"], dict) or not run_spec["model_provenance"]:
        raise SystemExit(f"{path} run_spec.model_provenance must be a nonempty object.")
    for key, expected in (
        ("embedding_model", args.embed),
        ("reranker", args.reranker),
        ("ocr_engine", args.ocr),
        ("vlm_backend", args.vlm),
    ):
        if expected is not None and run_spec.get(key) != expected:
            raise SystemExit(
                f"{path} run_spec.{key} is {run_spec.get(key)!r}; expected {expected!r}."
            )
    if run_spec["shard_index"] is not None or run_spec["num_shards"] is not None:
        raise SystemExit(f"{path} records sharded output in the unsharded baseline workflow.")
    ids = _example_ids(payload, path)
    if stage != "b0":
        if baseline_payload is None:
            raise SystemExit(f"{path} cannot be validated without the B0 result.")
        baseline_spec = baseline_payload["run_spec"]
        for key in RUN_SPEC_MATCH_KEYS:
            if key == "manifest_sha256" and (key not in run_spec or key not in baseline_spec):
                continue
            if run_spec[key] != baseline_spec[key]:
                raise SystemExit(f"{path} run_spec.{key} does not match B0.")
        baseline_ids = _example_ids(baseline_payload, path.with_name("b0.json"))
        baseline_set = set(baseline_ids)
        ids_set = set(ids)
        missing = [item for item in baseline_ids if item not in ids_set]
        extra = [item for item in ids if item not in baseline_set]
        if missing or extra:
            raise SystemExit(
                f"{path} rows do not exactly match B0 "
                f"(missing {len(missing)}, extra {len(extra)})."
            )
    _validate_summary_against_rows(payload, path, baseline_payload)
    return payload


def _validate_output_scope(value: str, label: str) -> None:
    if not value.strip() or value in {".", ".."} or Path(value).name != value:
        raise SystemExit(f"{label} must be a single nonempty path segment: {value!r}.")


def run_child(argv: list[str], cwd: Path) -> int:
    global _CHILD
    if _pending_signal is not None:
        return 128 + _pending_signal
    previous = _install_signal_handlers()
    try:
        if _pending_signal is not None:
            return 128 + _pending_signal
        _CHILD = subprocess.Popen(argv, cwd=str(cwd), env=os.environ.copy())
        if _pending_signal is not None:
            try:
                _CHILD.send_signal(_pending_signal)
            except ProcessLookupError:
                pass
        child_code = _CHILD.wait()
        if _forwarded_signal is not None:
            return 128 + _forwarded_signal
        return child_code
    finally:
        _CHILD = None
        _restore_signal_handlers(previous)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root(args)
    if not root.is_dir():
        raise SystemExit(f"project root is not a directory: {root}")
    _validate_output_scope(args.dataset, "dataset")
    _validate_output_scope(args.split, "split")
    if args.max_examples is not None and args.max_examples < 1:
        raise SystemExit("--max-examples must be at least 1.")
    if not args.resume:
        for stage in STAGE_ORDER[: STAGE_ORDER.index(args.stop_after) + 1]:
            out = output_path(root, args.dataset, args.split, stage)
            if out.exists():
                raise SystemExit(f"Refusing to run: target output already exists: {out}")
    if args.dry_run:
        b0_path = output_path(root, args.dataset, args.split, "b0")
        for stage in STAGE_ORDER[: STAGE_ORDER.index(args.stop_after) + 1]:
            out = output_path(root, args.dataset, args.split, stage)
            baseline = b0_path if stage != "b0" else None
            command = launcher_command(args, stage_run_args(stage, args, out, baseline))
            print(f"[run-baselines] {stage}: {shlex.join(command)}", flush=True)
        return 0
    b0_path = output_path(root, args.dataset, args.split, "b0")
    baseline_payload: dict | None = None
    if args.resume and b0_path.is_file():
        # Resume must validate existing B0 before its provenance is trusted for
        # the gate-lock check; a tampered B0 fails here before any stage runs.
        baseline_payload = validate_output(b0_path, "b0", args, None)
    # Validate the B2 gate lock before launching any stage so a missing or
    # invalid lock never spends API/GPU resources on B0 or paid B1. Without an
    # existing B0, resolve the exact provenance the fresh run would record so
    # a lock tuned for a different embedding/reranker configuration also
    # fails here instead of after paid B1.
    gate_lock = None
    if args.stop_after != "b0":
        baseline_mp = (
            baseline_payload["run_spec"].get("model_provenance")
            if baseline_payload is not None
            else effective_retrieval_provenance(embed=args.embed, reranker=args.reranker)
        )
        gate_lock = validate_gate_lock(
            root,
            dataset=args.dataset,
            model_provenance=baseline_mp,
        )
        print(
            f"[run-baselines] gate lock validated for B2 (threshold {gate_lock['threshold']})",
            flush=True,
        )
    previous = _install_signal_handlers()
    try:
        for stage in STAGE_ORDER[: STAGE_ORDER.index(args.stop_after) + 1]:
            if _pending_signal is not None:
                return 128 + _pending_signal
            gate_lock = None
            if stage == "b2":
                baseline_mp = (
                    baseline_payload["run_spec"].get("model_provenance")
                    if baseline_payload is not None
                    else None
                )
                gate_lock = validate_gate_lock(
                    root,
                    dataset=args.dataset,
                    model_provenance=baseline_mp,
                )
            out = output_path(root, args.dataset, args.split, stage)
            baseline = b0_path if stage != "b0" else None
            if out.exists():
                if not args.resume:
                    raise SystemExit(f"Refusing to run: target output already exists: {out}")
                payload = validate_output(out, stage, args, baseline_payload)
                print(f"[run-baselines] reuse {stage}: {out}", flush=True)
            else:
                command = launcher_command(args, stage_run_args(stage, args, out, baseline))
                print(f"[run-baselines] launch {stage}: {out}", flush=True)
                try:
                    code = run_child(command, root)
                except OSError as exc:
                    raise SystemExit(f"failed to launch {stage}: {exc}") from exc
                if _pending_signal is not None:
                    return 128 + _pending_signal
                if code != 0:
                    raise SystemExit(f"stage {stage} exited with code {code}; stopping.")
                payload = validate_output(out, stage, args, baseline_payload)
                print(f"[run-baselines] done {stage}: {out}", flush=True)
            if gate_lock is not None and payload["run_spec"].get("gate_threshold") != float(gate_lock["threshold"]):
                raise SystemExit(f"{out} run_spec.gate_threshold does not match the locked B2 threshold.")
            if stage == "b0":
                baseline_payload = payload
        if _pending_signal is not None:
            return 128 + _pending_signal
        completed = STAGE_ORDER[: STAGE_ORDER.index(args.stop_after) + 1]
        print(
            "[run-baselines] stages complete: " + " ".join(stage.upper() for stage in completed),
            flush=True,
        )
        return 0
    finally:
        _restore_signal_handlers(previous)


if __name__ == "__main__":
    raise SystemExit(main())
