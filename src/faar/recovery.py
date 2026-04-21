from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .settings import AppSettings
from .types import RetrievalHit


@lru_cache(maxsize=1)
def _load_byt5(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


class ByT5Corrector:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def correct(self, text: str, max_new_tokens: int = 128) -> str:
        return self.propose_correction(text, max_new_tokens=max_new_tokens)["text"]

    def propose_correction(self, text: str, max_new_tokens: int = 128) -> dict[str, str | bool]:
        if not text.strip():
            return {"text": text, "candidate": text, "applied": False, "reason": "empty_input"}
        should_attempt, skip_reason = _should_attempt_correction(text)
        if not should_attempt:
            return {"text": text, "candidate": text, "applied": False, "reason": skip_reason}
        corrected = self._generate_correction(text, max_new_tokens=max_new_tokens)
        accepted, reason = _should_accept_correction(text, corrected)
        return {
            "text": corrected if accepted else text,
            "candidate": corrected,
            "applied": accepted,
            "reason": reason,
        }

    def _generate_correction(self, text: str, max_new_tokens: int = 128) -> str:
        # Bound inference input to keep Phase 3 batch runs tractable.
        clipped = text[:512]
        tokenizer, model = _load_byt5(self.model_name)
        prompt = f"correct ocr noise: {clipped}"
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True)
        output = model.generate(**tokens, max_new_tokens=min(max_new_tokens, 64))
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()


class VisualFallback:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def answer(self, question: str, image_paths: list[Path], fallback_context: str) -> dict:
        if not self.settings.recovery.api_enabled and self.settings.recovery.vlm_backend == "openai":
            return {
                "backend": "openai",
                "status": "skipped",
                "reason": "api_disabled",
                "answer": "",
                "used_images": [],
            }
        if self.settings.recovery.vlm_backend == "openai":
            return self._answer_with_openai(question, image_paths)
        return {
            "backend": "mock",
            "status": "skipped",
            "reason": "vlm_backend_not_configured",
            "answer": "",
            "used_images": [str(path) for path in image_paths],
            "fallback_context": fallback_context[:500],
        }

    def _answer_with_openai(self, question: str, image_paths: list[Path]) -> dict:
        if not image_paths:
            return {
                "backend": "openai",
                "status": "skipped",
                "reason": "no_images_provided",
                "answer": "",
                "used_images": [],
            }
        client = OpenAI()
        content: list[dict] = [{"type": "text", "text": f"Answer the question using only the page image.\n\nQuestion: {question}"}]
        for path in image_paths:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        try:
            response = client.chat.completions.create(
                model=self.settings.recovery.openai_model,
                messages=[{"role": "user", "content": content}],
                timeout=self.settings.recovery.request_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - external runtime dependent
            return {
                "backend": "openai",
                "status": "failed",
                "reason": f"openai_error:{type(exc).__name__}",
                "answer": "",
                "used_images": [str(path) for path in image_paths],
            }
        return {
            "backend": "openai",
            "status": "succeeded",
            "reason": "visual_fallback_answer_generated",
            "answer": response.choices[0].message.content or "",
            "used_images": [str(path) for path in image_paths],
        }


def semantic_backtrack(query: str, hits: list[RetrievalHit]) -> str:
    context = " ".join(hit.chunk.text for hit in hits)
    if not context:
        return query
    anchor_words = context.split()[:24]
    return f"{query} {' '.join(anchor_words)}"


def _should_attempt_correction(text: str) -> tuple[bool, str]:
    if _contains_cjk(text):
        return False, "non_latin_source"
    if _looks_formula_like(text):
        return False, "formula_like_source"
    if _weird_char_ratio(text) < 0.08 and not _has_ocr_like_token_noise(text):
        return False, "source_not_noisy_enough"
    return True, "eligible"


def _should_accept_correction(source: str, candidate: str) -> tuple[bool, str]:
    if not candidate.strip():
        return False, "empty_correction"
    if _normalize_whitespace(source) == _normalize_whitespace(candidate):
        return False, "no_change"
    if _numeric_signature(source) != _numeric_signature(candidate):
        return False, "numeric_signature_changed"
    if not 0.6 <= _length_ratio(source, candidate) <= 1.4:
        return False, "length_shift_too_large"
    if _informative_token_overlap(source, candidate) < 0.5:
        return False, "low_token_preservation"
    if _weird_char_ratio(candidate) > _weird_char_ratio(source) + 0.01:
        return False, "noise_not_reduced"
    return True, "accepted"


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _looks_formula_like(text: str) -> bool:
    formula_markers = ("\\", "{", "}", "^", "_", "operatorname", "Delta", "alpha", "beta")
    return any(marker in text for marker in formula_markers)


def _weird_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird = sum(1 for char in text if not (char.isalnum() or char.isspace() or char in ".,;:!?$%()-/'\"&"))
    return weird / max(len(text), 1)


def _has_ocr_like_token_noise(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]+\d+[A-Za-z]*|\d+[A-Za-z]{2,}|[A-Za-z]{2,}\d+[A-Za-z]*", text))


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _numeric_signature(text: str) -> list[str]:
    return re.findall(r"(?:US\s*)?\$?\d[\d,]*(?:\.\d+)?%?", text)


def _informative_token_overlap(source: str, candidate: str) -> float:
    source_tokens = _informative_tokens(source)
    candidate_tokens = _informative_tokens(candidate)
    if not source_tokens:
        return 1.0
    overlap = len(source_tokens & candidate_tokens)
    return overlap / len(source_tokens)


def _informative_tokens(text: str) -> set[str]:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9$%]+", text)}
    return {token for token in tokens if len(token) >= 3 or any(char.isdigit() for char in token)}


def _length_ratio(source: str, candidate: str) -> float:
    if not source:
        return 1.0
    return len(candidate) / len(source)
