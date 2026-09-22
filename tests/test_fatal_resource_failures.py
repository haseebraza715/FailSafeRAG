from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from faar import experiment_runner, resource_limits, visual_baselines
from faar.settings import AppSettings


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
        "ex2,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text("===== PAGE 0 =====\ncertified engineers")
    (tmp_path / "artifacts/phase0/ocr_text/ex2.txt").write_text("===== PAGE 0 =====\ncertified engineers")
    (tmp_path / "data/phase0/manual_labels.csv").write_text(
        "example_id,question,correct_answer,ocr_output_snippet,failure_type,notes\n"
        "ex1,What is required?,certified engineers,snippet,no_issue,stable\n"
        "ex2,What is required?,certified engineers,snippet,text_corruption,noisy\n"
    )


def _success_result(example_id: str) -> dict:
    return {
        "question": f"What is required for {example_id}?",
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
                "api_requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            },
        },
        "retrieved_hits": [],
        "corrected_hits": [],
    }


class _FailingGraph:
    def __init__(self, failure: Exception, fail_at: str) -> None:
        self.failure = failure
        self.fail_at = fail_at
        self.invocations = 0

    def invoke(self, state):
        self.invocations += 1
        example_id = state["example_id"]
        if example_id == self.fail_at:
            raise self.failure
        return _success_result(example_id)


def _install_graph(monkeypatch: pytest.MonkeyPatch, graph) -> None:
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings, **kwargs: graph)


def _text_checkpoint(tmp_path: Path, example_id: str) -> Path:
    return tmp_path / "logs/phase3/faar_full" / f"{example_id}.json"


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("process memory exhausted"),
        torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    ],
)
def test_text_runner_aborts_on_fatal_resource_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception
) -> None:
    _prepare_phase0(tmp_path)
    graph = _FailingGraph(failure, fail_at="ex2")
    _install_graph(monkeypatch, graph)
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(type(failure), match="memory|CUDA"):
        experiment_runner.run_profile(
            settings, profile_name="faar_full", example_ids=["ex1", "ex2"]
        )
    assert graph.invocations == 2
    assert _text_checkpoint(tmp_path, "ex1").is_file()
    assert not _text_checkpoint(tmp_path, "ex2").exists()


def test_text_runner_isolates_recoverable_value_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    graph = _FailingGraph(ValueError("bad row data"), fail_at="ex2")
    _install_graph(monkeypatch, graph)
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(RuntimeError, match="1 example\\(s\\) failed"):
        experiment_runner.run_profile(
            settings, profile_name="faar_full", example_ids=["ex1", "ex2"]
        )
    assert graph.invocations == 2
    failed = json.loads(_text_checkpoint(tmp_path, "ex2").read_text())
    assert failed["action_outcome"]["status"] == "failed"
    assert "ValueError" in failed["error"]
    assert json.loads(_text_checkpoint(tmp_path, "ex1").read_text())["metrics"]["em"] == 1.0


def test_text_runner_resume_preserves_completed_checkpoints_after_fatal_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_phase0(tmp_path)
    first = _FailingGraph(MemoryError("boom"), fail_at="ex2")
    _install_graph(monkeypatch, first)
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(MemoryError):
        experiment_runner.run_profile(
            settings, profile_name="faar_full", example_ids=["ex1", "ex2"]
        )
    second = _FailingGraph(MemoryError("boom"), fail_at="ex2")
    _install_graph(monkeypatch, second)
    with pytest.raises(MemoryError):
        experiment_runner.run_profile(
            settings, profile_name="faar_full", example_ids=["ex1", "ex2"], resume=True
        )
    assert second.invocations == 1
    assert json.loads(_text_checkpoint(tmp_path, "ex1").read_text())["metrics"]["em"] == 1.0
    assert not _text_checkpoint(tmp_path, "ex2").exists()


def test_text_runner_checks_memory_budget_at_every_example_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_phase0(tmp_path)
    _install_graph(monkeypatch, _FailingGraph(ValueError("x"), fail_at="never"))
    settings = AppSettings(project_root=tmp_path)
    budget_calls: list[str] = []
    monkeypatch.setattr(
        experiment_runner,
        "enforce_memory_budget",
        lambda stage, torch_module=None: budget_calls.append(stage),
    )
    experiment_runner.run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])
    assert budget_calls == [
        "profile:faar_full before example ex1",
        "profile:faar_full after example ex1",
        "profile:faar_full before example ex2",
        "profile:faar_full after example ex2",
    ]


def test_text_runner_memory_budget_failure_aborts_before_example(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_phase0(tmp_path)
    graph = _FailingGraph(ValueError("x"), fail_at="never")
    _install_graph(monkeypatch, graph)
    settings = AppSettings(project_root=tmp_path)
    budget_calls: list[str] = []

    def budget(stage, torch_module=None):
        budget_calls.append(stage)
        if stage.endswith("before example ex2"):
            raise MemoryError("FAAR_MAX_RSS_GB exceeded")

    monkeypatch.setattr(experiment_runner, "enforce_memory_budget", budget)
    with pytest.raises(MemoryError, match="exceeded"):
        experiment_runner.run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])
    assert budget_calls == [
        "profile:faar_full before example ex1",
        "profile:faar_full after example ex1",
        "profile:faar_full before example ex2",
    ]
    assert graph.invocations == 1


class _VisualRepo:
    dataset = "ohrbench"
    split = "test"
    manifest_sha256 = "f" * 64

    def __init__(self, example_ids: list[str] | None = None) -> None:
        self._ids = list(example_ids) if example_ids is not None else ["q1", "q2"]

    def list_example_ids(self) -> list[str]:
        return list(self._ids)

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


class _EmptyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        return []


class _VisualFallback:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def answer(self, question: str, image_paths: list[Path], fallback_context: str):
        if self.failure is not None and question.startswith("Question q2"):
            raise self.failure
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


def _install_visual_fakes(
    monkeypatch: pytest.MonkeyPatch, fallback: _VisualFallback | None = None
) -> None:
    monkeypatch.setattr(
        visual_baselines,
        "build_visual_retriever",
        lambda *args, **kwargs: _EmptyRetriever(),
    )
    monkeypatch.setattr(
        visual_baselines,
        "VisualFallback",
        lambda settings: fallback if fallback is not None else _VisualFallback(),
    )


def _visual_checkpoint(tmp_path: Path, mode: str, example_id: str) -> Path:
    return tmp_path / "logs/phase3" / mode / f"{example_id}.json"


def test_visual_runner_aborts_on_fatal_resource_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_visual_fakes(monkeypatch, _VisualFallback(failure=MemoryError("boom")))
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(MemoryError, match="boom"):
        visual_baselines.run_visual_baseline(settings, _VisualRepo(["q1", "q2"]), "colpali")
    assert _visual_checkpoint(tmp_path, "colpali", "q1").is_file()
    assert not _visual_checkpoint(tmp_path, "colpali", "q2").exists()


def test_visual_runner_isolates_recoverable_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_visual_fakes(monkeypatch, _VisualFallback(failure=ValueError("bad row")))
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(RuntimeError, match="1 example\\(s\\) failed"):
        visual_baselines.run_visual_baseline(settings, _VisualRepo(["q1", "q2"]), "colpali")
    failed = json.loads(_visual_checkpoint(tmp_path, "colpali", "q2").read_text())
    assert failed["action_outcome"]["status"] == "failed"
    assert "ValueError" in failed["error"]
    assert json.loads(_visual_checkpoint(tmp_path, "colpali", "q1").read_text())["metrics"]["em"] == 1.0


def test_visual_runner_checks_memory_budget_at_every_example_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_visual_fakes(monkeypatch)
    settings = AppSettings(project_root=tmp_path)
    budget_calls: list[str] = []
    monkeypatch.setattr(
        visual_baselines,
        "enforce_memory_budget",
        lambda stage, torch_module=None: budget_calls.append(stage),
    )
    visual_baselines.run_visual_baseline(settings, _VisualRepo(["q1", "q2"]), "visrag")
    assert budget_calls == [
        "profile:visrag before example q1",
        "profile:visrag after example q1",
        "profile:visrag before example q2",
        "profile:visrag after example q2",
    ]


def test_fatal_classifier_detects_cuda_oom_variants() -> None:
    assert resource_limits.is_fatal_resource_error(MemoryError("x"))
    assert resource_limits.is_fatal_resource_error(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )
    assert resource_limits.is_fatal_resource_error(
        torch.cuda.OutOfMemoryError("CUDA out of memory")
    )
    assert resource_limits.is_fatal_resource_error(
        torch.cuda.OutOfMemoryError("cuda allocator out of memory")
    )
    assert not resource_limits.is_fatal_resource_error(ValueError("bad data"))
    assert not resource_limits.is_fatal_resource_error(KeyError("x"))
    assert not resource_limits.is_fatal_resource_error(RuntimeError("rate limit exhausted"))
    assert not resource_limits.is_fatal_resource_error(
        RuntimeError("out of memory while processing a single request")
    )


def test_fatal_classifier_avoids_eager_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import():
        raise AssertionError("torch must not be imported for ordinary exceptions")

    monkeypatch.setattr(resource_limits, "_import_torch", fail_import)
    assert not resource_limits.is_fatal_resource_error(ValueError("x"))
    assert not resource_limits.is_fatal_resource_error(RuntimeError("bad data"))
    assert resource_limits.is_fatal_resource_error(MemoryError("x"))
    assert resource_limits.is_fatal_resource_error(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )
