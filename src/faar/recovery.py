from __future__ import annotations

import base64
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
        if not text.strip():
            return text
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
