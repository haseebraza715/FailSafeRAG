from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from faar.recovery import VisualFallback, _media_type_for_image
from faar.settings import AppSettings

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"
JPEG = b"\xff\xd8\xff\xe0" + b"payload"
GIF = b"GIF89a" + b"payload"
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"payload"
PDF = b"%PDF-1.7" + b"payload"
UNKNOWN = b"not an image at all"


def test_png_bytes_map_to_image_png() -> None:
    assert _media_type_for_image(PNG) == "image/png"


def test_jpeg_bytes_map_to_image_jpeg() -> None:
    assert _media_type_for_image(JPEG) == "image/jpeg"


def test_gif_bytes_map_to_image_gif() -> None:
    assert _media_type_for_image(GIF) == "image/gif"


def test_webp_bytes_map_to_image_webp() -> None:
    assert _media_type_for_image(WEBP) == "image/webp"


def test_pdf_bytes_return_none() -> None:
    assert _media_type_for_image(PDF) is None


def test_unknown_bytes_return_none() -> None:
    assert _media_type_for_image(UNKNOWN) is None


def test_openai_payload_uses_detected_media_type(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(JPEG)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(
                model="gpt-4o-2024-11-20",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = False

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    assert result["status"] == "succeeded"
    image_parts = [part for part in captured["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image_parts[0]["image_url"]["url"].split(",", 1)[1] == base64.b64encode(JPEG).decode("utf-8")


def test_openai_partial_skip_succeeds_and_records_skipped_images(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(JPEG)
    unsupported_path = tmp_path / "page.pdf"
    unsupported_path.write_bytes(PDF)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(
                model="gpt-4o-2024-11-20",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path, unsupported_path], "")

    assert result["status"] == "succeeded"
    image_parts = [part for part in captured["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result["used_images"] == [str(image_path)]
    assert result["skipped_images"] == {"count": 1, "paths": [str(unsupported_path)]}

    records = [json.loads(line) for line in (tmp_path / "logs/vlm_calls.jsonl").read_text().splitlines()]
    assert records[0]["metadata"]["skipped_images"] == {"count": 1, "paths": [str(unsupported_path)]}


def test_openai_all_images_skipped_raises(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.pdf"
    image_path.write_bytes(PDF)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    class FakeCompletions:
        def create(self, **kwargs):
            raise AssertionError("must not reach the provider when all images are skipped")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda **kwargs: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"

    with pytest.raises(RuntimeError, match="application/pdf") as excinfo:
        VisualFallback(settings).answer("Question?", [image_path], "")

    assert str(image_path) in str(excinfo.value)


def test_anthropic_all_images_skipped_raises(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("anthropic")
    image_path = tmp_path / "page.unknown"
    image_path.write_bytes(UNKNOWN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-placeholder")

    class FakeMessages:
        def create(self, **kwargs):
            raise AssertionError("must not reach the provider when all images are skipped")

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: SimpleNamespace(messages=FakeMessages()))
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "claude-sonnet-4-5"

    with pytest.raises(RuntimeError, match="unknown") as excinfo:
        VisualFallback(settings).answer("Question?", [image_path], "ctx")

    assert str(image_path) in str(excinfo.value)
