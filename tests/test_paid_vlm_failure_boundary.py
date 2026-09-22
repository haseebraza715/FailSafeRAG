from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from faar.experiment_runner import run_profile
from faar.recovery import VisualFallback
from faar.settings import AppSettings
from faar.visual_baselines import run_visual_baseline

FAILED_VLM_RESULT = {
    "backend": "openai",
    "status": "failed",
    "reason": "openai_error:RuntimeError",
    "answer": "",
    "request_model": "gpt-4o-2024-11-20",
    "response_model": None,
    "completed_at_utc": "2026-08-09T00:00:00+00:00",
    "cost_rates": {
        "provider": "openai",
        "currency": "USD",
        "input_usd_per_million_tokens": 2.5,
        "output_usd_per_million_tokens": 10.0,
    },
    "api_usage": {
        "api_requests": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    },
}

SKIPPED_VLM_RESULT = {
    "backend": "openai",
    "status": "skipped",
    "reason": "api_disabled",
    "answer": "",
    "used_images": [],
    "api_usage": {
        "api_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    },
}

SUCCEEDED_VLM_RESULT = {
    "backend": "openai",
    "status": "succeeded",
    "reason": "visual_fallback_answer_generated",
    "answer": "certified engineers",
    "request_model": "gpt-4o-2024-11-20",
    "response_model": "gpt-4o-2024-11-20",
    "completed_at_utc": "2026-08-09T00:00:00+00:00",
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


class FakeGraph:
    def __init__(self, visual_result: dict[str, Any]) -> None:
        self.visual_result = visual_result

    def invoke(self, state: dict[str, str]) -> dict[str, Any]:
        return {
            "question": "What is required?",
            "example": type("Ex", (), {"correct_answer": "certified engineers"})(),
            "answer": self.visual_result.get("answer", ""),
            "failure_type": "structural",
            "policy_action": "invoke_vlm",
            "action_outcome": {
                "action": "invoke_vlm",
                "status": self.visual_result.get("status", "succeeded"),
                "reason": self.visual_result.get("reason", "vlm_action_executed"),
            },
            "visual_result": self.visual_result,
            "retrieved_hits": [],
            "corrected_hits": [],
        }


def _text_checkpoint(tmp_path: Path, example_id: str) -> Path:
    return tmp_path / "logs/phase3/faar_full" / f"{example_id}.json"


def test_text_runner_failed_vlm_records_failed_checkpoint_and_raises(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph(FAILED_VLM_RESULT))
    settings = AppSettings(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="failed and were not scored"):
        run_profile(settings, profile_name="faar_full", example_ids=["ex2"])

    checkpoint = json.loads(_text_checkpoint(tmp_path, "ex2").read_text(encoding="utf-8"))
    assert checkpoint["action_outcome"]["status"] == "failed"
    assert "refusing to score or checkpoint a failed row" in checkpoint["error"]


def test_text_runner_failed_vlm_checkpoint_is_not_reused_on_resume(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph(FAILED_VLM_RESULT))
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(RuntimeError, match="failed and were not scored"):
        run_profile(settings, profile_name="faar_full", example_ids=["ex2"])

    with pytest.raises(RuntimeError, match="failed and were not scored"):
        run_profile(settings, profile_name="faar_full", example_ids=["ex2"], resume=True)


def test_text_runner_isolates_one_failed_example_from_others(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    calls: list[str] = []

    def flaky_graph(settings):
        def invoke(state):
            calls.append(state["example_id"])
            if state["example_id"] == "ex1":
                return FakeGraph(FAILED_VLM_RESULT).invoke(state)
            return FakeGraph(SUCCEEDED_VLM_RESULT).invoke(state)

        return type("G", (), {"invoke": staticmethod(invoke)})()

    monkeypatch.setattr("faar.experiment_runner.build_graph", flaky_graph)
    settings = AppSettings(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="1 example\\(s\\) failed"):
        run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"])

    assert calls == ["ex1", "ex2"]
    succeeded = json.loads(_text_checkpoint(tmp_path, "ex2").read_text(encoding="utf-8"))
    assert succeeded["action_outcome"]["status"] == "succeeded"
    failed = json.loads(_text_checkpoint(tmp_path, "ex1").read_text(encoding="utf-8"))
    assert failed["action_outcome"]["status"] == "failed"


def test_text_runner_successful_vlm_still_checkpoints(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph(SUCCEEDED_VLM_RESULT))
    settings = AppSettings(project_root=tmp_path)

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex2"])

    assert len(rows) == 1
    assert rows[0]["metrics"]["em"] == 1.0
    checkpoint = json.loads(_text_checkpoint(tmp_path, "ex2").read_text(encoding="utf-8"))
    assert checkpoint["action_outcome"]["status"] == "succeeded"
    assert checkpoint["api_usage"]["api_requests"] == 1


def test_text_runner_skipped_vlm_still_checkpoints(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph(SKIPPED_VLM_RESULT))
    settings = AppSettings(project_root=tmp_path)

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex2"])

    assert len(rows) == 1
    checkpoint = json.loads(_text_checkpoint(tmp_path, "ex2").read_text(encoding="utf-8"))
    assert checkpoint["action_outcome"]["status"] == "skipped"
    assert checkpoint["api_usage"] == {
        "api_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }


def test_text_resume_recomputes_legacy_failed_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph(SUCCEEDED_VLM_RESULT))
    settings = AppSettings(project_root=tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex2"])

    checkpoint_path = _text_checkpoint(tmp_path, "ex2")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["action_outcome"]["status"] = "failed"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex2"], resume=True)

    assert rows[0]["action_outcome"]["status"] == "succeeded"


class FakeRepo:
    dataset = "ohrbench"
    split = "test"
    manifest_sha256 = "f" * 64

    def __init__(self, example_ids: list[str] | None = None) -> None:
        self._ids = list(example_ids) if example_ids is not None else ["q1"]

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


class FakeRetriever:
    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        return []


class _FakeFallback:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def answer(self, question: str, image_paths: list[Path], fallback_context: str) -> dict[str, Any]:
        return self.result


def _patch_visual(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> None:
    def make_retriever(*args: Any, **kwargs: Any) -> FakeRetriever:
        return FakeRetriever()

    def make_fallback(settings: AppSettings) -> _FakeFallback:
        return _FakeFallback(result)

    monkeypatch.setattr("faar.visual_baselines.build_visual_retriever", make_retriever)
    monkeypatch.setattr("faar.visual_baselines.VisualFallback", make_fallback)


def _visual_success(example_id: str) -> dict[str, Any]:
    return {
        "backend": "openai",
        "status": "succeeded",
        "reason": "visual_fallback_answer_generated",
        "answer": f"answer-{example_id}",
        "request_model": "gpt-4o-2024-11-20",
        "response_model": "gpt-4o-2024-11-20",
        "completed_at_utc": "2026-08-09T00:00:00+00:00",
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


def _visual_checkpoint(tmp_path: Path, example_id: str) -> Path:
    return tmp_path / "logs/phase3/colpali" / f"{example_id}.json"


def test_visual_runner_failed_vlm_records_failed_checkpoint_and_raises(monkeypatch, tmp_path: Path) -> None:
    _patch_visual(monkeypatch, FAILED_VLM_RESULT)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1"])

    with pytest.raises(RuntimeError, match="failed and were not scored"):
        run_visual_baseline(settings, repo, "colpali")

    checkpoint = json.loads(_visual_checkpoint(tmp_path, "q1").read_text(encoding="utf-8"))
    assert checkpoint["action_outcome"]["status"] == "failed"
    assert "refusing to score or checkpoint a failed row" in checkpoint["error"]


def test_visual_runner_successful_vlm_still_checkpoints(monkeypatch, tmp_path: Path) -> None:
    _patch_visual(monkeypatch, _visual_success("q1"))
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1"])

    rows = run_visual_baseline(settings, repo, "colpali")

    assert rows[0]["metrics"]["em"] == 1.0
    checkpoint = json.loads(_visual_checkpoint(tmp_path, "q1").read_text(encoding="utf-8"))
    assert checkpoint["action_outcome"]["status"] == "succeeded"
    assert checkpoint["api_usage"]["api_requests"] == 1


def test_visual_runner_skipped_vlm_still_checkpoints(monkeypatch, tmp_path: Path) -> None:
    _patch_visual(monkeypatch, SKIPPED_VLM_RESULT)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1"])

    rows = run_visual_baseline(settings, repo, "colpali")

    assert len(rows) == 1
    checkpoint = json.loads(_visual_checkpoint(tmp_path, "q1").read_text(encoding="utf-8"))
    assert checkpoint["action_outcome"]["status"] == "skipped"
    assert checkpoint["api_usage"] == {
        "api_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }


def test_visual_resume_recomputes_legacy_failed_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _patch_visual(monkeypatch, _visual_success("q1"))
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1"])
    run_visual_baseline(settings, repo, "colpali")

    checkpoint_path = _visual_checkpoint(tmp_path, "q1")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["action_outcome"]["status"] = "failed"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert rows[0]["action_outcome"]["status"] == "succeeded"


def test_openai_client_constructed_with_zero_retries(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    captured: dict[str, Any] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                model="gpt-4o-2024-11-20",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setattr("faar.recovery.OpenAI", fake_openai)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "succeeded"
    assert captured == {"max_retries": 0}


def test_anthropic_client_constructed_with_zero_retries(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-placeholder")
    captured: dict[str, Any] = {}

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                model="claude-sonnet-4-5",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                content=[SimpleNamespace(type="text", text="answer")],
            )

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(messages=FakeMessages())

    monkeypatch.setattr("anthropic.Anthropic", fake_anthropic)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"

    result = VisualFallback(settings).answer("Question?", [image_path], "ctx")

    assert result["status"] == "succeeded"
    assert captured == {"api_key": "test-only-placeholder", "max_retries": 0}
