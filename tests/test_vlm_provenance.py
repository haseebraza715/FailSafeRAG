from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import run as run_module

from faar.api_logging import anthropic_cost_rates, openai_cost_rates, vlm_cost_rates
from faar.experiment_runner import run_profile
from faar.recovery import VisualFallback
from faar.settings import AppSettings


def test_anthropic_cost_rates_normalize_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_INPUT_USD_PER_MTOK", raising=False)
    monkeypatch.delenv("ANTHROPIC_OUTPUT_USD_PER_MTOK", raising=False)

    rates = anthropic_cost_rates()

    assert rates == {
        "provider": "anthropic",
        "currency": "USD",
        "input_usd_per_million_tokens": 3.0,
        "output_usd_per_million_tokens": 15.0,
    }


def test_anthropic_cost_rates_use_same_env_as_cost_estimation(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_INPUT_USD_PER_MTOK", "2.0")
    monkeypatch.setenv("ANTHROPIC_OUTPUT_USD_PER_MTOK", "12.0")

    from faar.api_logging import estimate_anthropic_cost_usd

    rates = anthropic_cost_rates()
    assert rates["input_usd_per_million_tokens"] == 2.0
    assert rates["output_usd_per_million_tokens"] == 12.0
    assert estimate_anthropic_cost_usd("claude-sonnet-4-5", 1_000_000, 1_000_000) == 14.0


def test_vlm_cost_rates_selects_by_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_INPUT_USD_PER_MTOK", "2.50")
    monkeypatch.setenv("OPENAI_OUTPUT_USD_PER_MTOK", "10.00")
    monkeypatch.setenv("ANTHROPIC_INPUT_USD_PER_MTOK", "3.0")
    monkeypatch.setenv("ANTHROPIC_OUTPUT_USD_PER_MTOK", "15.0")

    assert vlm_cost_rates("openai") == openai_cost_rates()
    assert vlm_cost_rates("claude-sonnet-4-5") == anthropic_cost_rates()
    assert vlm_cost_rates("anthropic") == anthropic_cost_rates()
    assert vlm_cost_rates("claude") == anthropic_cost_rates()
    assert vlm_cost_rates("mock") is None


def test_anthropic_provider_result_persists_cost_rates(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("ANTHROPIC_INPUT_USD_PER_MTOK", "3.0")
    monkeypatch.setenv("ANTHROPIC_OUTPUT_USD_PER_MTOK", "15.0")

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                model="claude-sonnet-4-5",
                usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                content=[SimpleNamespace(type="text", text="answer")],
            )

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: SimpleNamespace(messages=FakeMessages()))
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "ctx")

    assert result["status"] == "succeeded"
    assert result["cost_rates"] == anthropic_cost_rates()
    records = [json.loads(line) for line in (tmp_path / "logs/vlm_calls.jsonl").read_text().splitlines()]
    assert records[-1]["metadata"]["cost_rates"] == anthropic_cost_rates()
    assert records[-1]["metadata"]["request_model"] == "claude-sonnet-4-5"


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


class _FakeGraph:
    def __init__(self, visual_result: dict) -> None:
        self.visual_result = visual_result

    def invoke(self, state: dict) -> dict:
        return {
            "question": "What is required?",
            "example": type("Ex", (), {"correct_answer": "certified engineers"})(),
            "answer": "certified engineers",
            "failure_type": "semantic",
            "policy_action": "retry_retrieval",
            "action_outcome": {"action": "retry_retrieval", "status": "succeeded"},
            "visual_result": self.visual_result,
            "retrieved_hits": [],
            "corrected_hits": [],
        }


def test_text_rows_persist_anthropic_cost_rates_instead_of_none(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    visual_result = {
        "request_model": "claude-sonnet-4-5",
        "response_model": "claude-sonnet-4-5",
        "completed_at_utc": "2026-07-23T12:00:00+00:00",
        "api_usage": {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
    }
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: _FakeGraph(visual_result))
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"

    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex2"])

    assert rows[0]["cost_rates"] == anthropic_cost_rates()
    assert rows[0]["run_metadata"]["cost_rates"] == anthropic_cost_rates()
    assert rows[0]["run_metadata"]["vlm_backend"] == "claude-sonnet-4-5"
    assert rows[0]["run_metadata"]["vlm_model"] == "claude-sonnet-4-5"
    saved = json.loads((tmp_path / "logs/phase3/faar_full/ex2.json").read_text(encoding="utf-8"))
    assert saved["cost_rates"] == anthropic_cost_rates()
    assert saved["run_metadata"]["cost_rates"] == anthropic_cost_rates()


def test_run_spec_persists_anthropic_cost_rates(monkeypatch, tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"
    settings.recovery.anthropic_model = "claude-sonnet-4-5"
    monkeypatch.setattr(run_module, "_settings_from_args", lambda args: settings)
    monkeypatch.setattr(AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(run_module, "load_benchmark_repository", lambda *args, **kwargs: object())
    output_path = tmp_path / "b0.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        label: str,
        max_examples: int | None,
        run_spec: dict,
        dataset: str,
        split: str,
        **kwargs: object,
    ) -> dict:
        payload = {
            "label": label,
            "profile": profile,
            "run_spec": run_spec,
            "rows": [],
            "summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0, "cost_usd": 0.0},
        }
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run_module, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--gate", "off", "--recovery", "off", "--out", str(output_path)],
    )

    run_module.main()

    saved = json.loads(output_path.read_text())
    assert saved["run_spec"]["vlm_backend"] == "claude-sonnet-4-5"
    assert saved["run_spec"]["vlm_model"] == "claude-sonnet-4-5"
    assert saved["run_spec"]["vlm_cost_rates"] == anthropic_cost_rates()
