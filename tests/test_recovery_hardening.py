from pathlib import Path

import pytest

from faar.recovery import (
    ByT5Corrector,
    VisualFallback,
    _should_accept_correction,
    semantic_backtrack,
)
from faar.settings import AppSettings
from faar.types import Chunk, RetrievalHit


def test_propose_correction_skips_kana_and_hangul_sources() -> None:
    corrector = ByT5Corrector("google/byt5-small")
    assert corrector.propose_correction("こんにちは世界")["reason"] == "non_latin_source"
    assert corrector.propose_correction("안녕하세요")["reason"] == "non_latin_source"


def test_propose_correction_skips_formula_like_source() -> None:
    corrector = ByT5Corrector("google/byt5-small")
    proposal = corrector.propose_correction("x = \\alpha + \\beta")
    assert proposal["applied"] is False
    assert proposal["reason"] == "formula_like_source"


def test_propose_correction_handles_empty_input() -> None:
    corrector = ByT5Corrector("google/byt5-small")
    proposal = corrector.propose_correction("   ")
    assert proposal["applied"] is False
    assert proposal["reason"] == "empty_input"


def test_propose_correction_rejects_empty_candidate(monkeypatch) -> None:
    corrector = ByT5Corrector("google/byt5-small")
    monkeypatch.setattr(ByT5Corrector, "_generate_correction", lambda self, text, max_new_tokens=128: "")
    proposal = corrector.propose_correction("inv0ice t0tal $91 due")
    assert proposal["applied"] is False
    assert proposal["reason"] == "empty_correction"


def test_propose_correction_rejects_length_shift(monkeypatch) -> None:
    corrector = ByT5Corrector("google/byt5-small")
    monkeypatch.setattr(
        ByT5Corrector,
        "_generate_correction",
        lambda self, text, max_new_tokens=128: text + " appended padding" * 40,
    )
    proposal = corrector.propose_correction("invoice t0tal $91 due on receipt")
    assert proposal["applied"] is False
    assert proposal["reason"] == "length_shift_too_large"


def test_propose_correction_degrades_on_inference_runtime_error(monkeypatch) -> None:
    corrector = ByT5Corrector("google/byt5-small")

    def _boom(self, text, max_new_tokens=128) -> str:
        raise RuntimeError("transformer generation failed")

    monkeypatch.setattr(ByT5Corrector, "_generate_correction", _boom)
    proposal = corrector.propose_correction("invoice t0tal $98 due on receipt")
    assert proposal["applied"] is False
    assert proposal["reason"] == "byt5_inference_failed"
    assert proposal["text"] == "invoice t0tal $98 due on receipt"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        MemoryError("host RAM exhausted"),
    ],
    ids=["cuda-oom", "ram-oom"],
)
def test_propose_correction_escalates_resource_exhaustion(monkeypatch, error: Exception) -> None:
    corrector = ByT5Corrector("google/byt5-small")

    def _boom(self, text, max_new_tokens=128) -> str:
        raise error

    monkeypatch.setattr(ByT5Corrector, "_generate_correction", _boom)
    with pytest.raises(type(error)):
        corrector.propose_correction("invoice t0tal $98 due on receipt")


def test_accept_correction_rejects_low_token_preservation() -> None:
    reason = _should_accept_correction("apple banana cherry", "orange grape pear")[1]
    assert reason == "low_token_preservation"


def test_accept_correction_rejects_noise_increase() -> None:
    reason = _should_accept_correction("valid 100", "vali¢ 100")[1]
    assert reason == "noise_not_reduced"


def test_accept_correction_accepts_clean_candidate() -> None:
    accepted, reason = _should_accept_correction("invoice t0tal $91", "invoice total $91")
    assert accepted is True
    assert reason == "accepted"


def test_accept_correction_rejects_no_change() -> None:
    accepted, reason = _should_accept_correction("plain sentence", "plain sentence")
    assert accepted is False
    assert reason == "no_change"


def _hit(text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(chunk_id="c1", example_id="e1", doc_name="doc", page_id=0, text=text),
        bm25_score=0.5,
        dense_score=0.5,
        fused_score=0.5,
    )


def test_semantic_backtrack_without_hits_returns_query() -> None:
    assert semantic_backtrack("what is the answer?", []) == "what is the answer?"


def test_semantic_backtrack_anchors_top_hit_words() -> None:
    query = "what is the answer?"
    long_first = " ".join(f"anchor{i}" for i in range(30))
    hits = [_hit(long_first), _hit("second chunk text")]
    rebuilt = semantic_backtrack(query, hits)
    assert rebuilt.startswith(query)
    assert "anchor23" in rebuilt
    assert "anchor24" not in rebuilt
    assert "second chunk" not in rebuilt


def test_visual_fallback_mock_backend_is_noop() -> None:
    settings = AppSettings()
    settings.recovery.vlm_backend = "mock"
    settings.recovery.enable_vlm = True
    result = VisualFallback(settings).answer("Q?", [Path("/tmp/img.png")], "context")
    assert result["status"] == "skipped"
    assert result["reason"] == "mock_backend_noop"
    assert result["used_images"] == ["/tmp/img.png"]


def test_visual_fallback_openai_skips_when_api_disabled() -> None:
    settings = AppSettings()
    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = False
    result = VisualFallback(settings).answer("Q?", [Path("/tmp/img.png")], "context")
    assert result["status"] == "skipped"
    assert result["reason"] == "api_disabled"


def test_visual_fallback_openai_skips_without_images() -> None:
    settings = AppSettings()
    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = True
    result = VisualFallback(settings).answer("Q?", [], "context")
    assert result["status"] == "skipped"
    assert result["reason"] == "no_images_provided"


def test_visual_fallback_openai_reports_image_read_error(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = True
    result = VisualFallback(settings).answer("Q?", [tmp_path / "missing.png"], "context")
    assert result["status"] == "failed"
    assert result["reason"].startswith("image_read_error:")


def test_visual_fallback_openai_reports_client_error(tmp_path: Path, monkeypatch) -> None:
    # The key is a placeholder that only satisfies the presence check. The patched
    # OpenAI class replaces the real client, so no request leaves the process.
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-key-for-mocked-client")
    create_calls: list[dict] = []
    constructed: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            raise TimeoutError("connection timed out")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.chat = FakeChat()

    monkeypatch.setattr("faar.recovery.OpenAI", FakeClient)
    monkeypatch.setattr("faar.recovery._sleep_before_retry", lambda *args, **kwargs: None)
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.api_enabled = True
    result = VisualFallback(settings).answer("Q?", [image], "context")
    assert result["status"] == "failed"
    assert result["reason"] == "openai_error:TimeoutError"
    # The fake client was built and called, so the failure came from the mock.
    assert constructed == [{"max_retries": 0}]
    assert create_calls
    assert (tmp_path / "logs/vlm_calls.jsonl").is_file()
