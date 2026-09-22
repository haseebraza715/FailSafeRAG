from __future__ import annotations

import argparse
import json
import signal
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from faar.gate_tuning import _gate_relevant_source_digest

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
RUN_SPEC = spec_from_file_location("cluster_run_baselines", CLUSTER / "run_baselines.py")
assert RUN_SPEC and RUN_SPEC.loader
run_baselines = module_from_spec(RUN_SPEC)
sys.modules["cluster_run_baselines"] = run_baselines
RUN_SPEC.loader.exec_module(run_baselines)

DATASET = "ohrbench"
SPLIT = "test"
SEED = 42
IDS = ["id-a", "id-b"]


def _effective_provenance() -> dict:
    """Gate-relevant provenance the runner resolves for a fresh run (no overrides)."""
    return run_baselines.effective_retrieval_provenance()


def _gate_lock(root: Path) -> None:
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    source = root / "results/baselines/ohrbench/val/b0.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source_payload = {
        "profile": "naive_rag",
        "run_spec": {
            "profile": "naive_rag",
            "dataset": DATASET,
            "split": "val",
            "model_provenance": _effective_provenance(),
        },
        "rows": [],
    }
    source.write_text(json.dumps(source_payload) + "\n", encoding="utf-8")
    (config / "gate_threshold.json").write_text(
        json.dumps(
            {
                "source_split": "val",
                "source_path": str(source),
                "source_digest": _gate_relevant_source_digest(source_payload),
                "dataset": DATASET,
                "split": "val",
                "model_provenance": _effective_provenance(),
                "gate_source_metric": "em",
                "signal": "BGE-reranker-v2-m3 top-1 score",
                "threshold": 0.5,
                "precision": 0.8,
                "recall": 0.75,
                "f1": 0.77,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _rows(ids: list[str]) -> list[dict]:
    return [
        {
            "example_id": item,
            "metrics": {"em": 1.0, "f1": 1.0},
            "api_usage": {
                "api_requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            },
        }
        for item in ids
    ]


def _payload(stage: str, args, ids: list[str]) -> dict:
    profile, label = run_baselines.STAGE_SPEC[stage]
    embedding = args.embed or "nvidia/NV-Embed-v2"
    reranker = args.reranker or "BAAI/bge-reranker-v2-m3"
    ocr = args.ocr or "got-ocr-2"
    return {
        "label": label,
        "profile": profile,
        "summary": {
            "EM": 1.0,
            "F1": 1.0,
            "vlm_rate": 0.0,
            "harm_rate": 0.0,
            "api_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "runtime_sec": 1.0,
        },
        "run_spec": {
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "max_examples": args.max_examples,
            "shard_index": None,
            "num_shards": None,
            "embedding_model": embedding,
            "reranker": reranker,
            "ocr_engine": ocr,
            "vlm_backend": args.vlm or "openai",
            "gate_threshold": 0.5 if stage == "b2" else None,
            "manifest_sha256": "a" * 64,
            "model_provenance": _effective_provenance(),
        },
        "rows": _rows(ids),
    }


def _write_output(root: Path, stage: str, args, ids: list[str]) -> Path:
    path = run_baselines.output_path(root, DATASET, SPLIT, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_payload(stage, args, ids)) + "\n", encoding="utf-8")
    return path


def _args(root: Path, **kwargs) -> argparse.Namespace:
    argv = ["--project-root", str(root)]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return run_baselines.parse_args(argv)


def _never(command, cwd):
    raise AssertionError("must not launch or overwrite")


def _stage_from_command(command: list[str]) -> str:
    out = Path(command[command.index("--out") + 1])
    return out.stem


def _successful_run_child(root: Path, args):
    calls = []

    def fake(command, cwd):
        calls.append(command)
        _write_output(root, _stage_from_command(command), args, IDS)
        return 0

    return fake, calls


def test_baselines_run_in_exact_order_with_b2_included(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, dataset=DATASET, split=SPLIT, seed=SEED)
    fake, calls = _successful_run_child(tmp_path, args)
    monkeypatch.setattr(run_baselines, "run_child", fake)
    result = run_baselines.main(["--project-root", str(tmp_path), "--dataset", DATASET, "--split", SPLIT, "--seed", str(SEED)])
    assert result == 0
    stages = [_stage_from_command(command) for command in calls]
    assert stages == ["b0", "b1", "b2", "b3", "b4"]
    for command in calls:
        assert command[0] == sys.executable
        assert command[1] == str(tmp_path / "cluster" / "launcher.py")
        assert "--project-root" in command
    b2_command = calls[2]
    assert b2_command[b2_command.index("--gate") + 1] == "on"
    assert b2_command[b2_command.index("--recovery") + 1] == "random_type"
    assert "--baseline" in b2_command


def test_stage_commands_constructed_correctly(tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(
        tmp_path,
        dataset=DATASET,
        split=SPLIT,
        seed=7,
        max_examples=5,
        vlm="openai",
        embed="NV-Embed-v2",
        reranker="bge-reranker-v2-m3",
        ocr="got-ocr-2",
    )
    b0 = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b0")
    b0_run_args = run_baselines.stage_run_args("b0", args, b0, None)
    assert b0_run_args[b0_run_args.index("--gate") + 1] == "off"
    assert b0_run_args[b0_run_args.index("--recovery") + 1] == "off"
    assert "--baseline" not in b0_run_args
    assert b0_run_args[b0_run_args.index("--dataset") + 1] == DATASET
    assert b0_run_args[b0_run_args.index("--split") + 1] == SPLIT
    assert b0_run_args[b0_run_args.index("--seed") + 1] == "7"
    assert b0_run_args[b0_run_args.index("--max-examples") + 1] == "5"
    for flag, value in (
        ("--vlm", "openai"),
        ("--embed", "NV-Embed-v2"),
        ("--reranker", "bge-reranker-v2-m3"),
        ("--ocr", "got-ocr-2"),
    ):
        assert b0_run_args[b0_run_args.index(flag) + 1] == value
    for stage, flag, value in (
        ("b1", "--recovery", "always_vlm"),
        ("b2", "--gate", "on"),
        ("b3", "--mode", "colpali"),
        ("b4", "--mode", "visrag"),
    ):
        run_args = run_baselines.stage_run_args(stage, args, b0, b0)
        assert run_args[run_args.index(flag) + 1] == value
        assert run_args[run_args.index("--baseline") + 1] == str(b0)
    launcher = run_baselines.launcher_command(args, b0_run_args)
    assert launcher[:2] == [sys.executable, str(tmp_path / "cluster" / "launcher.py")]
    assert "--cpu-only" not in launcher
    assert "--faar-python" not in launcher
    launcher2 = run_baselines.launcher_command(
        _args(tmp_path, cpu_only=True, faar_python="/opt/python3"),
        b0_run_args,
    )
    assert launcher2[0] == "/opt/python3"
    assert "--cpu-only" in launcher2
    assert launcher2[launcher2.index("--faar-python") + 1] == "/opt/python3"

    resume_args = _args(tmp_path, resume=True)
    for stage in ("b0", "b1", "b2"):
        assert "--resume" in run_baselines.stage_run_args(stage, resume_args, b0, None)
    for stage in ("b3", "b4"):
        assert "--resume" in run_baselines.stage_run_args(stage, resume_args, b0, b0)
    assert "--resume" not in run_baselines.stage_run_args("b0", _args(tmp_path), b0, None)


def test_stops_immediately_on_stage_failure(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path)
    calls = []

    def fake(command, cwd):
        calls.append(command)
        stage = _stage_from_command(command)
        if stage == "b0":
            _write_output(tmp_path, stage, args, IDS)
            return 0
        return 1

    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.main(["--project-root", str(tmp_path)])
    assert "stage b1 exited with code 1" in str(excinfo.value)
    assert [_stage_from_command(command) for command in calls] == ["b0", "b1"]


def test_cancellation_stops_future_stages(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path)
    monkeypatch.setattr(run_baselines, "_pending_signal", None)
    calls = []

    def fake(command, cwd):
        calls.append(command)
        stage = _stage_from_command(command)
        _write_output(tmp_path, stage, args, IDS)
        run_baselines._pending_signal = signal.SIGTERM
        return 0

    monkeypatch.setattr(run_baselines, "run_child", fake)
    result = run_baselines.main(["--project-root", str(tmp_path)])
    assert result == 128 + signal.SIGTERM
    assert [_stage_from_command(command) for command in calls] == ["b0"]


def test_resume_reuses_valid_outputs_without_launching(monkeypatch, tmp_path: Path, capsys) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    for stage in run_baselines.STAGE_ORDER:
        _write_output(tmp_path, stage, args, IDS)
    monkeypatch.setattr(run_baselines, "run_child", _never)
    result = run_baselines.main(["--project-root", str(tmp_path), "--resume"])
    assert result == 0
    out = capsys.readouterr().out
    for stage in run_baselines.STAGE_ORDER:
        assert f"reuse {stage}" in out


def test_resume_rejects_tampered_provenance(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    _write_output(tmp_path, "b0", args, IDS)
    payload = _payload("b1", args, IDS)
    payload["run_spec"]["seed"] = 7
    b1 = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b1")
    b1.parent.mkdir(parents=True, exist_ok=True)
    b1.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])
    assert "run_spec.seed" in str(excinfo.value)
    assert b1.read_text(encoding="utf-8") == json.dumps(payload) + "\n"


def test_resume_rejects_model_provenance_mismatch(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    _write_output(tmp_path, "b0", args, IDS)
    payload = _payload("b1", args, IDS)
    payload["run_spec"]["model_provenance"]["embedding"]["revision"] = "tampered"
    b1 = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b1")
    b1.parent.mkdir(parents=True, exist_ok=True)
    b1.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit, match="model_provenance"):
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])


def test_resume_rejects_tampered_profile(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    _write_output(tmp_path, "b0", args, IDS)
    payload = _payload("b1", args, IDS)
    payload["profile"] = "faar_full"
    b1 = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b1")
    b1.parent.mkdir(parents=True, exist_ok=True)
    b1.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])
    assert "profile" in str(excinfo.value)


def test_resume_rejects_missing_row_coverage(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    _write_output(tmp_path, "b0", args, IDS)
    _write_output(tmp_path, "b1", args, ["id-a"])
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])
    assert "do not exactly match B0" in str(excinfo.value)


def test_non_resume_refuses_any_existing_output(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path)
    _write_output(tmp_path, "b2", args, IDS)
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.main(["--project-root", str(tmp_path)])
    assert "already exists" in str(excinfo.value)


def test_gate_lock_required_for_b2(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.validate_gate_lock(tmp_path)
    assert "B2" in str(excinfo.value)
    assert "gate_threshold.json" in str(excinfo.value)
    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "gate_threshold.json").write_text(
        json.dumps({"source_split": "test", "threshold": 0.5, "precision": 0.9, "recall": 0.9})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.validate_gate_lock(tmp_path)
    assert "B2" in str(excinfo.value)
    _gate_lock(tmp_path)
    run_baselines.validate_gate_lock(tmp_path)


def test_missing_gate_lock_launches_zero_stages(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path)
    fake, calls = _successful_run_child(tmp_path, args)
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="B2 requires"):
        run_baselines.main(["--project-root", str(tmp_path)])
    assert calls == []


def test_invalid_gate_lock_launches_zero_stages(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "gate_threshold.json").write_text(
        json.dumps({"source_split": "test", "threshold": "not-a-number"}) + "\n",
        encoding="utf-8",
    )
    args = _args(tmp_path)
    fake, calls = _successful_run_child(tmp_path, args)
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="B2 gate lock is invalid"):
        run_baselines.main(["--project-root", str(tmp_path)])
    assert calls == []


def test_validation_b0_only_workflow_runs_without_gate_lock(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path, stop_after="b0")
    fake, calls = _successful_run_child(tmp_path, args)
    monkeypatch.setattr(run_baselines, "run_child", fake)
    result = run_baselines.main(
        ["--project-root", str(tmp_path), "--dataset", DATASET, "--split", SPLIT, "--seed", str(SEED), "--stop-after", "b0"]
    )
    assert result == 0
    assert [_stage_from_command(command) for command in calls] == ["b0"]
    b0_command = calls[0]
    assert b0_command[b0_command.index("--gate") + 1] == "off"
    assert b0_command[b0_command.index("--recovery") + 1] == "off"


def test_resume_validates_b0_before_gate_lock_and_launches_zero_stages(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, resume=True)
    _write_output(tmp_path, "b0", args, IDS)
    payload = _payload("b0", args, IDS)
    payload["run_spec"]["seed"] = 7
    b0 = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b0")
    b0.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit) as excinfo:
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])
    assert "run_spec.seed" in str(excinfo.value)


def test_resume_rejects_missing_gate_lock_before_launching_missing_stages(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, resume=True)
    _write_output(tmp_path, "b0", args, IDS)
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit, match="B2 requires"):
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])


def test_b2_remains_mandatory_without_lock_even_when_stopping_after_b1(
    monkeypatch, tmp_path: Path
) -> None:
    fake, calls = _successful_run_child(tmp_path, _args(tmp_path))
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="B2 requires"):
        run_baselines.main(["--project-root", str(tmp_path), "--stop-after", "b1"])
    assert calls == []


def test_fresh_run_mismatched_embedding_launches_zero_stages(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    fake, calls = _successful_run_child(tmp_path, _args(tmp_path))
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="retrieval model provenance does not match"):
        run_baselines.main(["--project-root", str(tmp_path), "--embed", "other/model"])
    assert calls == []


def test_fresh_run_mismatched_reranker_launches_zero_stages(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    fake, calls = _successful_run_child(tmp_path, _args(tmp_path))
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="retrieval model provenance does not match"):
        run_baselines.main(["--project-root", str(tmp_path), "--reranker", "other/reranker"])
    assert calls == []


def test_env_override_provenance_mismatch_launches_zero_stages(
    monkeypatch, tmp_path: Path
) -> None:
    _gate_lock(tmp_path)
    monkeypatch.setenv("EMBED_MODEL", "other/model")
    fake, calls = _successful_run_child(tmp_path, _args(tmp_path))
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="retrieval model provenance does not match"):
        run_baselines.main(["--project-root", str(tmp_path)])
    assert calls == []


def test_revision_mismatch_launches_zero_stages(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    monkeypatch.setenv("EMBED_MODEL_REVISION", "0" * 40)
    fake, calls = _successful_run_child(tmp_path, _args(tmp_path))
    monkeypatch.setattr(run_baselines, "run_child", fake)
    with pytest.raises(SystemExit, match="retrieval model provenance does not match"):
        run_baselines.main(["--project-root", str(tmp_path)])
    assert calls == []


def test_compatible_explicit_cli_models_launch_full_sequence(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(
        tmp_path,
        embed="nvidia/NV-Embed-v2",
        reranker="BAAI/bge-reranker-v2-m3",
    )
    fake, calls = _successful_run_child(tmp_path, args)
    monkeypatch.setattr(run_baselines, "run_child", fake)
    result = run_baselines.main(
        [
            "--project-root",
            str(tmp_path),
            "--embed",
            "nvidia/NV-Embed-v2",
            "--reranker",
            "BAAI/bge-reranker-v2-m3",
        ]
    )
    assert result == 0
    assert [_stage_from_command(command) for command in calls] == ["b0", "b1", "b2", "b3", "b4"]


def test_gate_provenance_resolution_is_side_effect_free(monkeypatch, tmp_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("torch must not be imported while resolving gate provenance")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    provenance = run_baselines.effective_retrieval_provenance()
    assert set(provenance) == {"embedding", "reranker"}
    assert provenance["embedding"]["repository"]
    assert provenance["embedding"]["revision"]
    assert provenance["reranker"]["repository"]
    assert provenance["reranker"]["revision"]


def test_resume_rejects_null_summary_metric(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    payload = _payload("b0", args, IDS)
    payload["summary"]["F1"] = None
    path = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit, match="summary.F1"):
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])


def test_resume_rejects_summary_that_disagrees_with_rows(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    payload = _payload("b0", args, IDS)
    payload["summary"]["F1"] = 0.25
    path = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)
    with pytest.raises(SystemExit, match="rows recompute"):
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])


def test_resume_rejects_failed_row_even_when_it_has_zero_metrics(monkeypatch, tmp_path: Path) -> None:
    _gate_lock(tmp_path)
    args = _args(tmp_path, resume=True)
    payload = _payload("b0", args, IDS)
    payload["rows"][0]["metrics"] = {"em": 0.0, "f1": 0.0}
    payload["rows"][0]["action_outcome"] = {"action": "failed", "status": "failed"}
    payload["rows"][0]["error"] = "FileNotFoundError: missing OCR text"
    payload["summary"]["EM"] = 0.5
    payload["summary"]["F1"] = 0.5
    path = run_baselines.output_path(tmp_path, DATASET, SPLIT, "b0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_baselines, "run_child", _never)

    with pytest.raises(SystemExit, match="failed rows"):
        run_baselines.main(["--project-root", str(tmp_path), "--resume"])


def test_dry_run_prints_commands_in_order(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(run_baselines, "run_child", _never)
    result = run_baselines.main(["--project-root", str(tmp_path), "--dry-run"])
    assert result == 0
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[run-baselines]")
    ]
    stages = [line.split()[1] for line in lines]
    assert stages == ["b0:", "b1:", "b2:", "b3:", "b4:"]
    b0_line = lines[0]
    assert "--gate off" in b0_line
    assert "--recovery off" in b0_line
    assert "--baseline" not in b0_line
    assert "--recovery always_vlm" in lines[1]
    assert "--gate on" in lines[2]
    assert "--mode colpali" in lines[3]
    assert "--mode visrag" in lines[4]
    for line, stage in zip(lines, run_baselines.STAGE_ORDER):
        assert "--out" in line
        assert f"{stage}.json" in line
