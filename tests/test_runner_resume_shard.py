from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import run

from faar.experiment_runner import run_profile
from faar.settings import AppSettings


class FakeGraph:
    def __init__(self) -> None:
        self.invoked: list[str] = []

    def invoke(self, state: dict[str, str]) -> dict[str, Any]:
        self.invoked.append(state["example_id"])
        return {
            "question": "What is required?",
            "example": type("Ex", (), {"correct_answer": "certified engineers"})(),
            "answer": "certified engineers",
            "failure_type": "semantic",
            "policy_action": "retry_retrieval",
            "action_outcome": {"action": "retry_retrieval", "status": "succeeded"},
            "visual_result": {
                "request_model": "gpt-4o-2024-11-20",
                "response_model": "gpt-4o-2024-11-20",
                "completed_at_utc": "2026-07-23T12:00:00+00:00",
                "api_usage": {
                    "api_requests": 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cost_usd": 0.00045,
                },
            },
            "retrieved_hits": [],
            "corrected_hits": [],
        }


def _prepare_phase0(tmp_path: Path, count: int) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    ids = [f"ex{i}" for i in range(1, count + 1)]
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        + "".join(f"{e},manual/doc,What is required?,certified engineers,0\n" for e in ids)
    )
    for e in ids:
        (tmp_path / "artifacts/phase0/ocr_text" / f"{e}.txt").write_text("===== PAGE 0 =====\ncertified engineers")


def _patch_graph(monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    graph = FakeGraph()
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings, **kwargs: graph)
    return graph


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(project_root=tmp_path)


def test_resume_reuses_matching_checkpoint_without_recomputing(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)

    first = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])
    assert graph.invoked == ["ex1", "ex2"]

    graph.invoked.clear()
    second = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)

    assert graph.invoked == []
    assert [row["example_id"] for row in second] == ["ex1", "ex2"]
    assert first[0] == second[0]
    assert second[0]["run_metadata"]["run_fingerprint"]


def test_resume_includes_cached_rows_in_returned_rows(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)
    assert len(rows) == 2
    assert all(row["predicted_answer"] == "certified engineers" for row in rows)


def test_resume_recomputes_when_fingerprint_changes(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])

    graph.invoked.clear()
    settings.retrieval.embedding_model = "other/embedder"
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)

    assert graph.invoked == ["ex1", "ex2"]
    assert len(rows) == 2


def test_resume_recomputes_checkpoints_from_different_code(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])

    monkeypatch.setattr("faar.run_io._measurement_code_digest", lambda: "0" * 64)
    graph.invoked.clear()
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)

    assert graph.invoked == ["ex1", "ex2"]
    assert len(rows) == 2


def test_resume_recomputes_corrupt_checkpoint_only(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])

    (tmp_path / "logs/phase3/faar_full/ex1.json").write_text("{not-json")
    graph.invoked.clear()
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)

    assert graph.invoked == ["ex1"]
    assert len(rows) == 2


def test_graph_not_built_when_all_requested_rows_are_cache_hits(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    calls: list[dict[str, Any]] = []
    graph = FakeGraph()

    def spy_build_graph(settings: AppSettings, **kwargs: Any) -> FakeGraph:
        calls.append({"settings": settings, "kwargs": kwargs})
        return graph

    monkeypatch.setattr("faar.experiment_runner.build_graph", spy_build_graph)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])
    assert len(calls) == 1

    calls.clear()
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)
    assert calls == []
    assert len(rows) == 2


def test_resume_does_not_skip_without_flag(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 1)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1"])

    graph.invoked.clear()
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1"])
    assert graph.invoked == ["ex1"]
    assert len(rows) == 1


def test_shards_partition_selection_without_output_collision(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 4)
    _patch_graph(monkeypatch)
    settings = _settings(tmp_path)

    first = run_profile(
        settings,
        profile_name="faar_full",
        example_ids=["ex1", "ex2", "ex3", "ex4"],
        shard_index=0,
        num_shards=2,
    )
    second = run_profile(
        settings,
        profile_name="faar_full",
        example_ids=["ex1", "ex2", "ex3", "ex4"],
        shard_index=1,
        num_shards=2,
    )

    first_ids = [row["example_id"] for row in first]
    second_ids = [row["example_id"] for row in second]
    assert first_ids == ["ex1", "ex2"]
    assert second_ids == ["ex3", "ex4"]
    assert sorted(first_ids + second_ids) == ["ex1", "ex2", "ex3", "ex4"]
    assert set(first_ids).isdisjoint(second_ids)

    rows_dir = tmp_path / "logs/phase3/faar_full"
    assert sorted(path.stem for path in rows_dir.glob("*.json")) == ["ex1", "ex2", "ex3", "ex4"]


def _isolate_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(run, "_settings_from_args", lambda args: settings)
    monkeypatch.setattr(AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(run, "_validate_baseline", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_apply_baseline_harm", lambda payload, *args, **kwargs: payload)


def _write_baseline(tmp_path: Path) -> Path:
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(
        json.dumps(
            {
                "run_spec": {"dataset": "ohrbench", "split": "test", "seed": 42},
                "summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0},
                "rows": [{"example_id": "ex1"}],
            }
        )
    )
    return baseline_path


def _capture_profile_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_profile_run(settings, profile, out, label, max_examples, run_spec, dataset, split, **kwargs):
        captured["out"] = out
        captured["kwargs"] = kwargs
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}}))
        return {"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}}

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    return captured


def test_cli_resume_flag_reaches_text_profile_path(monkeypatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    captured = _capture_profile_run(monkeypatch)
    baseline_path = _write_baseline(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "faar", "--resume", "--baseline", str(baseline_path), "--out", str(tmp_path / "faar.json")],
    )
    run.main()
    assert captured["kwargs"] == {"resume": True, "publish": False}


def test_cli_shard_flags_suffix_output_and_flow_to_runner(monkeypatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    captured = _capture_profile_run(monkeypatch)
    baseline_path = _write_baseline(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "faar", "--shard-index", "0", "--num-shards", "2", "--baseline", str(baseline_path), "--out", str(tmp_path / "faar.json")],
    )
    run.main()
    assert captured["kwargs"] == {"shard_index": 0, "num_shards": 2, "publish": False}
    assert captured["out"].name == "faar_shard1of2.json"


def test_cli_defaults_unchanged_without_flags(monkeypatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    captured = _capture_profile_run(monkeypatch)
    baseline_path = _write_baseline(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "faar", "--baseline", str(baseline_path), "--out", str(tmp_path / "faar.json")],
    )
    run.main()
    assert captured["kwargs"] == {"publish": False}
    assert captured["out"].name == "faar.json"


def test_cli_visual_path_rejects_shard_flags(monkeypatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    visual_calls: list[dict[str, Any]] = []

    def fake_visual_baseline(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        visual_calls.append({"args": args, "kwargs": kwargs})
        return []

    monkeypatch.setattr(run, "run_visual_baseline", fake_visual_baseline, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "colpali", "--shard-index", "0", "--num-shards", "2", "--out", str(tmp_path / "vis.json")],
    )
    with pytest.raises(SystemExit, match="not supported for visual modes"):
        run.main()
    assert visual_calls == []


def test_cli_visual_resume_reaches_baseline(monkeypatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    visual_calls: list[dict[str, Any]] = []

    def fake_visual_baseline(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        visual_calls.append({"args": args, "kwargs": kwargs})
        return []

    monkeypatch.setattr(run, "run_visual_baseline", fake_visual_baseline, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "colpali", "--resume", "--out", str(tmp_path / "vis.json")],
    )
    run.main()
    assert len(visual_calls) == 1
    assert visual_calls[0]["kwargs"]["resume"] is True
    assert visual_calls[0]["kwargs"]["max_examples"] is None
    assert visual_calls[0]["args"][2] == "colpali"


def test_empty_shard_does_not_build_graph(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 1)
    calls: list[object] = []
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda *args, **kwargs: calls.append(object()))

    rows = run_profile(
        _settings(tmp_path),
        profile_name="faar_full",
        shard_index=2,
        num_shards=3,
    )

    assert rows == []
    assert calls == []


def test_cli_partial_shard_flags_fail_closed(monkeypatch, tmp_path: Path) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    captured = _capture_profile_run(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--mode", "faar", "--shard-index", "0", "--out", str(tmp_path / "faar.json")],
    )
    with pytest.raises(SystemExit, match="--shard-index requires --num-shards"):
        run.main()
    assert captured == {}


def test_text_row_stores_api_usage(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 1)
    _patch_graph(monkeypatch)
    settings = _settings(tmp_path)

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1"])

    assert rows[0]["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": 0.00045,
    }


def test_resume_recomputes_checkpoint_with_malformed_api_usage(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])

    checkpoint = tmp_path / "logs/phase3/faar_full/ex1.json"
    row = json.loads(checkpoint.read_text())
    row["api_usage"]["api_requests"] = -1
    checkpoint.write_text(json.dumps(row))
    graph.invoked.clear()

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)

    assert graph.invoked == ["ex1"]
    assert len(rows) == 2
    assert rows[0]["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": 0.00045,
    }


def test_resume_recomputes_checkpoint_without_api_usage(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    graph = _patch_graph(monkeypatch)
    settings = _settings(tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])

    checkpoint = tmp_path / "logs/phase3/faar_full/ex1.json"
    row = json.loads(checkpoint.read_text())
    del row["api_usage"]
    checkpoint.write_text(json.dumps(row))
    graph.invoked.clear()

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True)

    assert graph.invoked == ["ex1"]
    assert len(rows) == 2


class FakeBenchmarkRepo:
    dataset = "ohrbench"
    split = "test"
    manifest_sha256 = "f" * 64

    def list_example_ids(self) -> list[str]:
        return ["ex1", "ex2", "ex3", "ex4"]


def _prepare_run_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeGraph:
    monkeypatch.setattr(
        "faar.experiment_runner.load_benchmark_repository",
        lambda project_root, dataset, split: FakeBenchmarkRepo(),
    )
    graph = FakeGraph()
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings, **kwargs: graph)
    monkeypatch.setattr(run, "_require_gate_threshold", lambda settings, profile, run_spec: None)
    return graph


def _text_run_spec() -> dict[str, Any]:
    return {"dataset": "ohrbench", "split": "test"}


def test_profile_result_forwards_nondefault_seed_to_every_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_run_environment(monkeypatch, tmp_path)
    settings = _settings(tmp_path)
    settings.experiment.random_seed = 17
    run_spec = _text_run_spec()
    run_spec["seed"] = 17

    payload = run._run_profile_to_result(
        settings,
        "faar_full",
        tmp_path / "faar.json",
        "FAAR",
        2,
        run_spec,
        "ohrbench",
        "test",
    )

    assert {row["run_metadata"]["seed"] for row in payload["rows"]} == {17}


def test_cli_profile_resume_reports_totals_for_cached_and_new_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph = _prepare_run_environment(monkeypatch, tmp_path)
    settings = _settings(tmp_path)
    out = tmp_path / "faar.json"

    first = run._run_profile_to_result(
        settings, "faar_full", out, "FAAR", 2, _text_run_spec(), "ohrbench", "test"
    )
    assert first["summary"]["api_requests"] == 2
    assert first["summary"]["prompt_tokens"] == 200
    assert first["summary"]["completion_tokens"] == 40
    assert first["summary"]["cost_usd"] == 0.0009

    graph.invoked.clear()
    partial = run._run_profile_to_result(
        settings, "faar_full", out, "FAAR", 4, _text_run_spec(), "ohrbench", "test", resume=True
    )

    assert graph.invoked == ["ex3", "ex4"]
    assert partial["summary"]["api_requests"] == 4
    assert partial["summary"]["prompt_tokens"] == 400
    assert partial["summary"]["completion_tokens"] == 80
    assert partial["summary"]["cost_usd"] == 0.0018


def test_cli_profile_all_cache_resume_keeps_original_totals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph = _prepare_run_environment(monkeypatch, tmp_path)
    settings = _settings(tmp_path)
    out = tmp_path / "faar.json"

    first = run._run_profile_to_result(
        settings, "faar_full", out, "FAAR", 2, _text_run_spec(), "ohrbench", "test"
    )
    graph.invoked.clear()
    resumed = run._run_profile_to_result(
        settings, "faar_full", out, "FAAR", 2, _text_run_spec(), "ohrbench", "test", resume=True
    )

    assert graph.invoked == []
    assert resumed["summary"]["api_requests"] == first["summary"]["api_requests"]
    assert resumed["summary"]["prompt_tokens"] == first["summary"]["prompt_tokens"]
    assert resumed["summary"]["completion_tokens"] == first["summary"]["completion_tokens"]
    assert resumed["summary"]["cost_usd"] == first["summary"]["cost_usd"]


class FakeVisualRepo:
    dataset = "ohrbench"
    split = "test"
    manifest_sha256 = "f" * 64

    def list_example_ids(self) -> list[str]:
        return ["q1", "q2", "q3", "q4"]

    def get_example(self, example_id: str):
        return SimpleNamespace(
            example_id=example_id,
            doc_name="doc",
            page_ids=[],
            question=f"Question {example_id}?",
            correct_answer=f"answer-{example_id}",
        )

    def corpus_image_page_map(self) -> dict[Path, tuple[str, int]]:
        return {}


class FakeVisualRetriever:
    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        return []


class FakeVisualFallback:
    def __init__(self, calls: list[str], settings: AppSettings) -> None:
        self.calls = calls
        self.settings = settings

    def answer(self, question: str, image_paths: list[Path], fallback_context: str):
        self.calls.append("answer")
        example_id = question.split()[1].rstrip("?")
        return {
            "status": "succeeded",
            "answer": f"answer-{example_id}",
            "request_model": "gpt-4o-2024-11-20",
            "response_model": "gpt-4o-2024-11-20",
            "completed_at_utc": "2026-07-23T12:00:00+00:00",
            "cost_rates": {
                "provider": "openai",
                "currency": "USD",
                "input_usd_per_million_tokens": 2.5,
                "output_usd_per_million_tokens": 10.0,
            },
            "api_usage": {
                "api_requests": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost_usd": 0.00045,
            },
        }


def test_cli_visual_resume_reports_totals_for_cached_and_new_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fallback_calls: list[str] = []

    def make_retriever(*args: Any, **kwargs: Any) -> FakeVisualRetriever:
        return FakeVisualRetriever()

    def make_fallback(settings: AppSettings) -> FakeVisualFallback:
        return FakeVisualFallback(fallback_calls, settings)

    monkeypatch.setattr("faar.visual_baselines.build_visual_retriever", make_retriever)
    monkeypatch.setattr("faar.visual_baselines.VisualFallback", make_fallback)
    monkeypatch.setattr(
        run,
        "load_benchmark_repository",
        lambda project_root, dataset, split: FakeVisualRepo(),
    )
    settings = _settings(tmp_path)
    out = tmp_path / "colpali.json"

    first = run._run_visual_baseline_to_result(
        settings, "colpali", out, 2, _text_run_spec(), "ohrbench", "test"
    )
    assert first["summary"]["api_requests"] == 2
    assert first["summary"]["prompt_tokens"] == 200
    assert first["summary"]["completion_tokens"] == 40
    assert first["summary"]["cost_usd"] == 0.0009

    partial = run._run_visual_baseline_to_result(
        settings, "colpali", out, 4, _text_run_spec(), "ohrbench", "test", resume=True
    )

    assert fallback_calls == ["answer"] * 4
    assert partial["summary"]["api_requests"] == 4
    assert partial["summary"]["prompt_tokens"] == 400
    assert partial["summary"]["completion_tokens"] == 80
    assert partial["summary"]["cost_usd"] == 0.0018
