from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import run
from evaluate import evaluate_results

from faar.settings import AppSettings


def _row(example_id: str, f1: float, *, action: str = "answer_direct") -> dict[str, Any]:
    return {
        "example_id": example_id,
        "predicted_answer": "prediction",
        "gold_answer": "gold",
        "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": f1},
        "action_outcome": {"action": action},
        "api_usage": {
            "api_requests": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost_usd": 0.0001,
        },
    }


def _write_b0(tmp_path: Path, settings: AppSettings, *example_ids: str) -> Path:
    run_spec = {
        "dataset": "ohrbench",
        "split": "test",
        "seed": 42,
        "max_examples": None,
        "shard_index": None,
        "num_shards": None,
        "embedding_model": settings.retrieval.embedding_model,
        "reranker": settings.retrieval.reranker,
        "ocr_engine": settings.recovery.ocr_engine,
        "model_provenance": settings.model_provenance(),
    }
    rows = [
        {
            "example_id": example_id,
            "predicted_answer": "gold",
            "gold_answer": "gold",
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 0.75},
            "action_outcome": {"action": "answer_direct"},
        }
        for example_id in example_ids
    ]
    path = tmp_path / "b0.json"
    path.write_text(
        json.dumps(
            {
                "label": "B0",
                "profile": "naive_rag",
                "run_spec": run_spec,
                "summary": {"EM": 1.0, "F1": 0.75, "vlm_rate": 0.0, "harm_rate": 0.0},
                "rows": rows,
            }
        )
    )
    return path


def _isolate_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppSettings:
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(run, "_settings_from_args", lambda args: settings)
    monkeypatch.setattr(AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(run, "_validate_baseline", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_require_gate_threshold", lambda settings, profile, run_spec: None)
    return settings


def _capture_atomic_writes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    real_write = run.atomic_write_text
    writes: list[str] = []

    def spy(path: Path, text: str) -> None:
        writes.append(str(path))
        real_write(path, text)

    monkeypatch.setattr(run, "atomic_write_text", spy)
    return writes


def _interrupt(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("simulated cluster cancellation")


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--gate", "off", "--recovery", "always_vlm"],
        ["--gate", "on", "--recovery", "random_type"],
    ],
    ids=["b1", "b2"],
)
def test_text_stage_publishes_once_with_computed_harm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_args: list[str],
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / "stage.json"
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: [_row("ex1", 0.25)])
    writes = _capture_atomic_writes(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", *cli_args, "--baseline", str(baseline), "--out", str(out)],
    )

    run.main()

    assert writes == [str(out)]
    payload = json.loads(out.read_text())
    assert payload["summary"]["harm_rate"] == 1.0


@pytest.mark.parametrize("mode", ["colpali", "visrag"], ids=["b3", "b4"])
def test_visual_stage_publishes_once_with_computed_harm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / f"{mode}.json"
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run,
        "run_visual_baseline",
        lambda *args, **kwargs: [_row("ex1", 0.25, action="invoke_vlm")],
    )
    writes = _capture_atomic_writes(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", mode, "--baseline", str(baseline), "--out", str(out)],
    )

    run.main()

    assert writes == [str(out)]
    payload = json.loads(out.read_text())
    assert payload["summary"]["harm_rate"] == 1.0


@pytest.mark.parametrize("mode", ["colpali", "visrag"], ids=["b3", "b4"])
def test_visual_stage_dict_return_fails_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / f"{mode}.json"
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run,
        "run_visual_baseline",
        lambda *args, **kwargs: {"summary": {}, "rows": [_row("ex1", 0.25, action="invoke_vlm")]},
    )
    writes = _capture_atomic_writes(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", mode, "--baseline", str(baseline), "--out", str(out)],
    )

    with pytest.raises(SystemExit, match="expected list\\[dict\\] per-example rows"):
        run.main()

    assert writes == []
    assert not out.exists()


@pytest.mark.parametrize("mode", ["colpali", "visrag"], ids=["b3", "b4"])
def test_visual_stage_harm_failure_leaves_no_public_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / f"{mode}.json"
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run,
        "run_visual_baseline",
        lambda *args, **kwargs: [_row("ex1", 0.25, action="invoke_vlm")],
    )
    monkeypatch.setattr(run, "_baseline_harm_rate", _interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", mode, "--baseline", str(baseline), "--out", str(out)],
    )

    with pytest.raises(RuntimeError, match="simulated cluster cancellation"):
        run.main()

    assert not out.exists()


def test_text_stage_harm_failure_leaves_no_public_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / "b1.json"
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: [_row("ex1", 0.25)])
    monkeypatch.setattr(run, "_baseline_harm_rate", _interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--gate",
            "off",
            "--recovery",
            "always_vlm",
            "--baseline",
            str(baseline),
            "--out",
            str(out),
        ],
    )

    with pytest.raises(RuntimeError, match="simulated cluster cancellation"):
        run.main()

    assert not out.exists()


def test_text_stage_baseline_coverage_failure_leaves_no_public_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex2")
    out = tmp_path / "b1.json"
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: [_row("ex1", 0.25)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--gate",
            "off",
            "--recovery",
            "always_vlm",
            "--baseline",
            str(baseline),
            "--out",
            str(out),
        ],
    )

    with pytest.raises(SystemExit, match="does not cover every result example ID"):
        run.main()

    assert not out.exists()


def test_text_stage_interrupted_before_harm_leaves_no_public_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / "b1.json"
    monkeypatch.setattr(run, "run_profile", _interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--gate",
            "off",
            "--recovery",
            "always_vlm",
            "--baseline",
            str(baseline),
            "--out",
            str(out),
        ],
    )

    with pytest.raises(RuntimeError, match="simulated cluster cancellation"):
        run.main()

    assert not out.exists()


def test_visual_stage_interrupted_before_harm_leaves_no_public_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _write_b0(tmp_path, settings, "ex1")
    out = tmp_path / "colpali.json"
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(run, "run_visual_baseline", _interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "colpali", "--baseline", str(baseline), "--out", str(out)],
    )

    with pytest.raises(RuntimeError, match="simulated cluster cancellation"):
        run.main()

    assert not out.exists()


def test_profile_helper_publish_false_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: [_row("ex1", 0.25)])
    out = tmp_path / "b1.json"

    payload = run._run_profile_to_result(
        settings,
        "faar_always_vlm",
        out,
        "B1",
        None,
        {"dataset": "ohrbench", "split": "test", "seed": 42},
        "ohrbench",
        "test",
        publish=False,
    )

    assert not out.exists()
    assert payload["summary"]["harm_rate"] is None


def test_b0_publishes_once_with_zero_harm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    out = tmp_path / "b0.json"
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: [_row("ex1", 0.5)])
    writes = _capture_atomic_writes(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--gate", "off", "--recovery", "off", "--out", str(out)],
    )

    run.main()

    assert writes == [str(out)]
    payload = json.loads(out.read_text())
    assert payload["profile"] == "naive_rag"
    assert payload["summary"]["harm_rate"] == 0.0


def test_baseline_harm_rate_matches_evaluate_results(tmp_path: Path) -> None:
    rows = [
        {
            "example_id": "ex1",
            "predicted_answer": "a",
            "gold_answer": "b",
            "metrics": {"em": 0.0, "f1": 0.2},
            "action_outcome": {"action": "answer_direct"},
        },
        {
            "example_id": "ex2",
            "predicted_answer": "b",
            "gold_answer": "b",
            "metrics": {"em": 1.0, "f1": 1.0},
            "action_outcome": {"action": "invoke_vlm"},
        },
        {
            "example_id": "ex3",
            "predicted_answer": "c",
            "gold_answer": "c",
            "metrics": {"em": 1.0, "f1": 0.8},
            "action_outcome": {"action": "answer_direct"},
        },
    ]
    baseline_rows = [
        {
            "example_id": "ex1",
            "predicted_answer": "a",
            "gold_answer": "b",
            "metrics": {"em": 0.0, "f1": 0.6},
        },
        {
            "example_id": "ex2",
            "predicted_answer": "b",
            "gold_answer": "b",
            "metrics": {"em": 1.0, "f1": 0.9},
        },
        {
            "example_id": "ex3",
            "predicted_answer": "c",
            "gold_answer": "c",
            "metrics": {"em": 1.0, "f1": 0.5},
        },
    ]
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"rows": rows}))
    baseline = tmp_path / "b0.json"
    baseline.write_text(json.dumps({"rows": baseline_rows}))

    expected = evaluate_results(results, baseline_path=baseline)["harm_rate"]

    assert run._baseline_harm_rate(rows, baseline_rows) == expected
