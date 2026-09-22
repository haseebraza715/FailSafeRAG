"""Contracts for the AAAI-facing experiment runner.

These tests intentionally describe the required runner behavior without loading
models or calling external services.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import run

from faar.gate_tuning import _gate_relevant_source_digest
from faar.settings import AppSettings


def _row(example_id: str, f1: float) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "predicted_answer": "prediction",
        "gold_answer": "gold",
        "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": f1},
        "action_outcome": {"action": "answer_direct"},
    }


def _result(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "summary": {
            "EM": 0.0,
            "F1": 0.0,
            "vlm_rate": 0.0,
            "harm_rate": 0.0,
            "cost_usd": 0.0,
            "runtime_sec": 0.0,
        },
        "rows": rows,
    }


def _b0_result(
    settings: AppSettings,
    rows: list[dict[str, Any]],
    **run_spec_overrides: Any,
) -> dict[str, Any]:
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
    run_spec.update(run_spec_overrides)
    payload = _result("B0", rows)
    payload["profile"] = "naive_rag"
    payload["run_spec"] = run_spec
    return payload


def _locked_gate_threshold(tmp_path: Path, **overrides: Any) -> None:
    threshold_path = tmp_path / "config/gate_threshold.json"
    threshold_path.parent.mkdir(parents=True)
    source = tmp_path / "results/baselines/ohrbench/val/b0.json"
    source.parent.mkdir(parents=True)
    source_payload = {
        "profile": "naive_rag",
        "run_spec": {
            "profile": "naive_rag",
            "dataset": "ohrbench",
            "split": "val",
            "model_provenance": {
                "embedding": {"repository": "nvidia/NV-Embed-v2", "revision": "e" * 40},
                "reranker": {"repository": "BAAI/bge-reranker-v2-m3", "revision": "r" * 40},
            },
        },
        "rows": [],
    }
    source.write_text(json.dumps(source_payload))
    payload = {
        "source_split": "val",
        "source_path": str(source),
        "source_digest": _gate_relevant_source_digest(source_payload),
        "dataset": "ohrbench",
        "split": "val",
        "model_provenance": {
            "embedding": {"repository": "nvidia/NV-Embed-v2", "revision": "e" * 40},
            "reranker": {"repository": "BAAI/bge-reranker-v2-m3", "revision": "r" * 40},
        },
        "gate_source_metric": "em",
        "signal": "BGE-reranker-v2-m3 top-1 score",
        "threshold": 0.4,
        "precision": 0.8,
        "recall": 0.75,
        "f1": 0.77,
    }
    payload.update(overrides)
    threshold_path.write_text(json.dumps(payload))


def _isolate_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppSettings:
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(run, "_settings_from_args", lambda args: settings)
    monkeypatch.setattr(AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    return settings


def test_b1_maps_to_a_dedicated_always_vlm_profile() -> None:
    baseline_id, profile, label = run.BASELINE_MAP[("off", "always_vlm")]

    assert baseline_id == "B1"
    assert profile == "faar_always_vlm"
    assert label == "Always-VLM"


def test_run_profile_for_args_rejects_unsupported_default_combination() -> None:
    args = run.argparse.Namespace(
        mode=None,
        ablate=None,
        gate=None,
        recovery=None,
    )
    with pytest.raises(SystemExit, match="Unsupported baseline combination: gate=None, recovery=None"):
        run._run_profile_for_args(args)


@pytest.mark.parametrize("mode", ["colpali", "visrag"])
def test_visual_modes_dispatch_to_visual_baseline_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline_path = tmp_path / "matching-baseline.json"
    baseline_path.write_text(json.dumps(_b0_result(settings, [_row("ex1", 0.75)])))
    calls: list[dict[str, Any]] = []

    def fake_visual_baseline(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"args": args, "kwargs": kwargs})
        return [
            {
                **_row("ex1", 0.25),
                "api_usage": {
                    "api_requests": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cost_usd": 0.0001,
                },
            }
        ]

    monkeypatch.setattr(run, "run_visual_baseline", fake_visual_baseline, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", mode, "--baseline", str(baseline_path), "--out", str(tmp_path / f"{mode}.json")],
    )

    run.main()

    assert len(calls) == 1
    call = calls[0]
    assert mode in call["args"] or call["kwargs"].get("mode") == mode


@pytest.mark.parametrize(
    "mode_args, expected_profile",
    [
        (["--mode", "faar"], "faar_full"),
        (["--ablate", "no_gate"], "faar_no_gate"),
        (["--ablate", "no_diagnosis"], "faar_no_diagnosis"),
        (["--ablate", "no_wordlevel_llm"], "faar_symspell"),
        (["--ablate", "no_semantic_retry"], "faar_no_backtrack"),
    ],
)
def test_faar_and_ablations_compute_harm_against_supplied_matching_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode_args: list[str],
    expected_profile: str,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline_path = tmp_path / "matching-baseline.json"
    baseline_path.write_text(json.dumps(_b0_result(settings, [_row("ex1", 0.75)])))
    out_path = tmp_path / f"{expected_profile}.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert profile == expected_profile
        payload = _result(profile, [_row("ex1", 0.25)])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            *mode_args,
            "--baseline",
            str(baseline_path),
            "--out",
            str(out_path),
        ],
    )

    run.main()

    payload = json.loads(out_path.read_text())
    assert payload["summary"]["harm_rate"] == 1.0


def test_saved_run_provenance_records_openai_cost_rates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    output_path = tmp_path / "b0.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        label: str,
        max_examples: int | None,
        run_spec: dict[str, Any],
        dataset: str,
        split: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _result(profile, [])
        payload["run_spec"] = run_spec
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--gate", "off", "--recovery", "off", "--out", str(output_path)],
    )

    run.main()

    saved = json.loads(output_path.read_text())
    rates = saved["run_spec"]["vlm_cost_rates"]
    assert rates["input_usd_per_million_tokens"] == 2.5
    assert rates["output_usd_per_million_tokens"] == 10.0


def test_gate_dependent_profile_fails_before_computation_without_locked_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = AppSettings(project_root=tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: calls.append(args) or [])
    with pytest.raises(SystemExit, match="locked gate threshold"):
        run._run_profile_to_result(
            settings,
            "faar_full",
            tmp_path / "out.json",
            "label",
            None,
            {"dataset": "ohrbench", "split": "test", "seed": 42},
            "ohrbench",
            "test",
        )
    assert calls == []


def test_gate_bypass_profiles_do_not_require_locked_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = AppSettings(project_root=tmp_path)

    def fake_run_profile(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "example_id": "ex1",
                "metrics": {"ndcg@5": 0.5, "recall@5": 0.5, "em": 0.0, "f1": 0.5},
                "action_outcome": {"action": "answer_direct"},
                "api_usage": {
                    "api_requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                },
            }
        ]

    monkeypatch.setattr(run, "run_profile", fake_run_profile)
    payload = run._run_profile_to_result(
        settings,
        "naive_rag",
        tmp_path / "out.json",
        "B0",
        None,
        {"dataset": "ohrbench", "split": "test", "seed": 42},
        "ohrbench",
        "test",
    )
    assert payload["profile"] == "naive_rag"


def test_gate_dependent_profile_rejects_sub_bar_locked_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _locked_gate_threshold(tmp_path, precision=0.6)
    settings = AppSettings(project_root=tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: calls.append(args) or [])
    with pytest.raises(SystemExit, match="paper bar"):
        run._run_profile_to_result(
            settings,
            "faar_no_diagnosis",
            tmp_path / "out.json",
            "B2",
            None,
            {"dataset": "ohrbench", "split": "test", "seed": 42},
            "ohrbench",
            "test",
        )
    assert calls == []


def test_gate_dependent_result_records_locked_threshold_and_null_harm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _locked_gate_threshold(tmp_path)
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(
        run,
        "run_profile",
        lambda *args, **kwargs: [
            {
                "example_id": "ex1",
                "metrics": {"ndcg@5": 0.5, "recall@5": 0.5, "em": 0.0, "f1": 0.5},
                "action_outcome": {"action": "answer_direct"},
                "api_usage": {
                    "api_requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                },
            }
        ],
    )
    run_spec = {"dataset": "ohrbench", "split": "test", "seed": 42}
    payload = run._run_profile_to_result(
        settings,
        "faar_full",
        tmp_path / "out.json",
        "FAAR",
        None,
        run_spec,
        "ohrbench",
        "test",
    )
    assert payload["run_spec"]["gate_threshold"] == 0.4
    assert payload["summary"]["harm_rate"] is None


def test_empty_b0_shard_serializes_zero_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(run, "run_profile", lambda *args, **kwargs: [])
    payload = run._run_profile_to_result(
        settings,
        "naive_rag",
        tmp_path / "b0.json",
        "B0",
        None,
        {},
        "ohrbench",
        "test",
    )
    assert payload["summary"]["vlm_rate"] == 0.0
    assert payload["summary"]["harm_rate"] == 0.0


def test_faar_mode_requires_explicit_baseline_before_computation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(
        run,
        "_run_profile_to_result",
        lambda *args, **kwargs: calls.append(args) or {"summary": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "faar", "--out", str(tmp_path / "faar.json")],
    )
    with pytest.raises(SystemExit, match="Missing required --baseline"):
        run.main()
    assert calls == []


def test_b2_requires_explicit_baseline_before_computation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(
        run,
        "_run_profile_to_result",
        lambda *args, **kwargs: calls.append(args) or {"summary": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--gate",
            "on",
            "--recovery",
            "random_type",
            "--out",
            str(tmp_path / "b2.json"),
        ],
    )
    with pytest.raises(SystemExit, match="Missing required --baseline"):
        run.main()
    assert calls == []


def test_b2_requires_paid_vlm_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    seen: list[str] = []
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: seen.append(backend))
    baseline_path = tmp_path / "matching-baseline.json"
    baseline_path.write_text(json.dumps(_b0_result(settings, [_row("ex1", 0.75)])))
    out_path = tmp_path / "b2.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        label: str,
        max_examples: int | None,
        run_spec: dict[str, Any],
        dataset: str,
        split: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _result(profile, [_row("ex1", 0.25)])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--gate",
            "on",
            "--recovery",
            "random_type",
            "--baseline",
            str(baseline_path),
            "--out",
            str(out_path),
        ],
    )
    run.main()
    assert seen == ["openai"]


def test_baseline_seed_provenance_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(
        json.dumps(
            _b0_result(settings, [_row("ex1", 0.75)], seed=7)
        )
    )
    out_path = tmp_path / "faar.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        label: str,
        max_examples: int | None,
        run_spec: dict[str, Any],
        dataset: str,
        split: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _result(profile, [_row("ex1", 0.25)])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--mode",
            "faar",
            "--baseline",
            str(baseline_path),
            "--out",
            str(out_path),
        ],
    )
    with pytest.raises(SystemExit, match="Baseline mismatch: expected seed=42"):
        run.main()


def test_baseline_missing_provenance_fails_before_computation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline = _b0_result(settings, [_row("ex1", 0.75)])
    del baseline["run_spec"]["model_provenance"]
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(json.dumps(baseline))
    calls: list[Any] = []
    monkeypatch.setattr(run, "_run_profile_to_result", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "faar", "--baseline", str(baseline_path), "--out", str(tmp_path / "faar.json")],
    )
    with pytest.raises(SystemExit, match="missing run_spec.model_provenance"):
        run.main()
    assert calls == []


def test_b0_rejects_baseline_argument(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--gate",
            "off",
            "--recovery",
            "off",
            "--baseline",
            str(baseline_path),
            "--out",
            str(tmp_path / "new-b0.json"),
        ],
    )
    with pytest.raises(SystemExit, match="does not accept --baseline"):
        run.main()


def test_baseline_missing_example_id_coverage_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _isolate_cli(monkeypatch, tmp_path)
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(json.dumps(_b0_result(settings, [_row("ex1", 0.75)])))
    out_path = tmp_path / "faar.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        label: str,
        max_examples: int | None,
        run_spec: dict[str, Any],
        dataset: str,
        split: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _result(profile, [_row("ex1", 0.25), _row("ex2", 0.5)])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--mode",
            "faar",
            "--baseline",
            str(baseline_path),
            "--out",
            str(out_path),
        ],
    )
    with pytest.raises(SystemExit, match="does not cover every result example ID"):
        run.main()
