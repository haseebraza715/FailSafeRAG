from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from faar.recovery import VisualFallback
from faar.settings import AppSettings


def _error(name: str, status: int | None = None) -> Exception:
    exc = type(name, (Exception,), {})()
    if status is not None:
        exc.status = status
    return exc


def _ok_openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-4o-2024-11-20",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
    )


def _records(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "logs/vlm_calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _image(tmp_path: Path) -> Path:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
    return image_path


def test_openai_transient_429_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _error("RateLimitError", status=429)
            return _ok_openai_response()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "succeeded"
    assert calls == 2
    assert result["api_usage"]["api_requests"] == 2
    assert [record["status"] for record in _records(tmp_path)] == ["started", "started", "succeeded"]


def test_openai_transient_by_status_alone_retries(monkeypatch, tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _error("ProviderBusy", status=503)
            return _ok_openai_response()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "succeeded"
    assert calls == 2
    assert result["api_usage"]["api_requests"] == 2


def test_openai_persistent_500_fails_after_max_attempts(monkeypatch, tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise _error("InternalServerError", status=500)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "failed"
    assert calls == 3
    assert result["api_usage"]["api_requests"] == 3
    assert [record["status"] for record in _records(tmp_path)] == ["started", "started", "started", "failed"]


def test_openai_non_transient_error_does_not_retry(monkeypatch, tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise _error("AuthenticationError")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "failed"
    assert calls == 1
    assert result["api_usage"]["api_requests"] == 1
    assert [record["status"] for record in _records(tmp_path)] == ["started", "failed"]


def test_openai_plain_value_error_does_not_retry(monkeypatch, tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise ValueError("bad request")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "failed"
    assert calls == 1
    assert result["api_usage"]["api_requests"] == 1


def test_openai_max_retries_env_limits_attempts(monkeypatch, tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    monkeypatch.setenv("FAAR_VLM_MAX_RETRIES", "1")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise _error("InternalServerError", status=500)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "failed"
    assert calls == 1
    assert result["api_usage"]["api_requests"] == 1
    assert [record["status"] for record in _records(tmp_path)] == ["started", "failed"]


def test_anthropic_transient_429_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("anthropic")
    image_path = _image(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-placeholder")
    monkeypatch.setattr("faar.recovery.time.sleep", lambda seconds: None)

    calls = 0

    class FakeMessages:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _error("RateLimitError", status=429)
            return SimpleNamespace(
                model="claude-sonnet-4-5",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                content=[SimpleNamespace(type="text", text="answer")],
            )

    fake_client = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "ctx")

    assert result["status"] == "succeeded"
    assert calls == 2
    assert result["api_usage"]["api_requests"] == 2
    assert [record["status"] for record in _records(tmp_path)] == ["started", "started", "succeeded"]
