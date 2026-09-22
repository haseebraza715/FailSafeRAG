"""API accounting contracts for one logical request with lifecycle events."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from faar.api_logging import (
    ApiCallLogger,
    estimate_anthropic_cost_usd,
    estimate_openai_cost_usd,
    new_record,
    openai_cost_rates,
)
from faar.recovery import VisualFallback
from faar.settings import AppSettings

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def test_openai_standard_cost_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_INPUT_USD_PER_MTOK", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_USD_PER_MTOK", raising=False)

    rates = openai_cost_rates()

    assert rates["input_usd_per_million_tokens"] == 2.5
    assert rates["output_usd_per_million_tokens"] == 10.0
    assert estimate_openai_cost_usd(1_000_000, 1_000_000) == 12.5


def test_openai_cost_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_INPUT_USD_PER_MTOK", "1.25")
    monkeypatch.setenv("OPENAI_OUTPUT_USD_PER_MTOK", "4.50")

    rates = openai_cost_rates()

    assert rates["input_usd_per_million_tokens"] == 1.25
    assert rates["output_usd_per_million_tokens"] == 4.5
    assert estimate_openai_cost_usd(1_000_000, 1_000_000) == 5.75


def test_started_and_succeeded_events_count_as_one_api_request(tmp_path: Path) -> None:
    log_path = tmp_path / "vlm_calls.jsonl"
    logger = ApiCallLogger(log_path)

    logger.log(
        new_record(
            provider="anthropic",
            model="claude-sonnet-4-5",
            operation="visual_fallback",
            status="started",
            metadata={"example_id": "ex1"},
        )
    )
    logger.log(
        new_record(
            provider="anthropic",
            model="claude-sonnet-4-5",
            operation="visual_fallback",
            status="succeeded",
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.01,
            metadata={"example_id": "ex1"},
        )
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["status"] for record in records] == ["started", "succeeded"]
    assert logger.call_count == 1
    assert logger.total_cost_usd == 0.01


def test_openai_request_is_logged_before_execution_and_usage_after(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG)
    log_path = tmp_path / "logs/vlm_calls.jsonl"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    class FakeCompletions:
        def create(self, **kwargs):
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            assert records[-1]["status"] == "started"
            assert kwargs["model"] == "gpt-4o-2024-11-20"
            return SimpleNamespace(
                model="gpt-4o-2024-11-20",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert result["status"] == "succeeded"
    assert [record["status"] for record in records] == ["started", "succeeded"]
    assert records[0]["metadata"]["request_id"] == records[1]["metadata"]["request_id"]
    assert records[1]["prompt_tokens"] == 100
    assert records[1]["completion_tokens"] == 20
    assert records[1]["cost_usd"] > 0
    assert records[1]["metadata"]["request_model"] == "gpt-4o-2024-11-20"
    assert records[1]["metadata"]["response_model"] == "gpt-4o-2024-11-20"
    assert records[1]["metadata"]["cost_rates"]["input_usd_per_million_tokens"] == 2.5
    assert records[1]["metadata"]["cost_rates"]["output_usd_per_million_tokens"] == 10.0
    assert result["response_model"] == "gpt-4o-2024-11-20"
    assert result["completed_at_utc"]


def test_openai_success_carries_normalized_api_usage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG)
    log_path = tmp_path / "logs/vlm_calls.jsonl"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.delenv("OPENAI_INPUT_USD_PER_MTOK", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_USD_PER_MTOK", raising=False)

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                model="gpt-4o-2024-11-20",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert result["status"] == "succeeded"
    assert result["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": records[-1]["cost_usd"],
    }
    assert records[-1]["prompt_tokens"] == 100
    assert records[-1]["completion_tokens"] == 20
    assert records[-1]["cost_usd"] == estimate_openai_cost_usd(100, 20, openai_cost_rates())


def test_openai_failed_request_counts_once_and_carries_api_usage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("provider down")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "failed"
    assert result["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }

    records = [json.loads(line) for line in (tmp_path / "logs/vlm_calls.jsonl").read_text().splitlines()]
    assert [record["status"] for record in records] == ["started", "failed"]
    assert records[0]["metadata"]["request_id"] == records[-1]["metadata"]["request_id"]
    assert records[-1]["metadata"]["error"] == "RuntimeError"


def test_skipped_fallback_paths_count_zero_api_usage(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    settings = AppSettings(project_root=tmp_path)

    settings.recovery.vlm_backend = "mock"
    mock = VisualFallback(settings).answer("Question?", [page], "ctx")
    assert mock["api_usage"] == {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = False
    disabled = VisualFallback(settings).answer("Question?", [page], "ctx")
    assert disabled["status"] == "skipped"
    assert disabled["api_usage"] == {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = True
    no_images = VisualFallback(settings).answer("Question?", [], "ctx")
    assert no_images["status"] == "skipped"
    assert no_images["api_usage"] == {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_anthropic_success_carries_normalized_api_usage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("anthropic")
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG)
    log_path = tmp_path / "logs/vlm_calls.jsonl"
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

    fake_client = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "ctx")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert result["status"] == "succeeded"
    assert result["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": records[-1]["cost_usd"],
    }
    assert records[-1]["cost_usd"] == round(
        estimate_anthropic_cost_usd("claude-sonnet-4-5", 100, 20), 8
    )


@pytest.mark.parametrize("backend", ["claude-sonnet-4-5", "anthropic", "claude"])
def test_disabled_anthropic_alias_paths_skip_without_credentials_or_client(
    monkeypatch,
    tmp_path: Path,
    backend: str,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def raiser(*args, **kwargs):
        raise AssertionError("anthropic dispatch must not occur when api_enabled=False")

    monkeypatch.setattr(VisualFallback, "_answer_with_anthropic", raiser)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = backend
    settings.recovery.api_enabled = False

    result = VisualFallback(settings).answer("Question?", [image_path], "ctx")

    assert result["status"] == "skipped"
    assert result["reason"] == "api_disabled"
    assert result["backend"] == "anthropic"
    assert result["answer"] == ""
    assert result["used_images"] == []
    assert result["api_usage"] == {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_disabled_openai_path_skips_without_credentials_or_client(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def raiser(*args, **kwargs):
        raise AssertionError("openai dispatch must not occur when api_enabled=False")

    monkeypatch.setattr(VisualFallback, "_answer_with_openai", raiser)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = False

    result = VisualFallback(settings).answer("Question?", [image_path], "ctx")

    assert result["status"] == "skipped"
    assert result["reason"] == "api_disabled"
    assert result["backend"] == "openai"
    assert result["answer"] == ""
    assert result["used_images"] == []
    assert result["api_usage"] == {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
