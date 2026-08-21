from __future__ import annotations

import base64
import os
import random
import re
import time
from datetime import UTC, datetime
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from .api_logging import (
    anthropic_cost_rates,
    estimate_anthropic_cost_usd,
    estimate_openai_cost_usd,
    make_api_usage,
    make_vlm_logger,
    new_record,
    openai_cost_rates,
    vlm_cost_rates,
    zero_api_usage,
)
from .resource_limits import (
    enforce_gpu_memory_fraction,
    enforce_memory_budget,
    is_fatal_resource_error,
    select_dtype,
    torch_device,
)
from .settings import AppSettings, CorrectionSettings
from .textnoise import CORRECTION_ALLOWED_PUNCTUATION, char_noise_ratio
from .types import RetrievalHit


def _media_type_for_image(data: bytes) -> str | None:
    header = data[:16]
    if header.startswith(b"\x89PNG"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _build_image_payloads(
    image_paths: list[Path],
    *,
    openai_format: bool,
) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
    payloads: list[dict] = []
    used_images: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in image_paths:
        raw = path.read_bytes()
        media_type = _media_type_for_image(raw)
        if media_type is None:
            label = "application/pdf" if raw.startswith(b"%PDF") else "unknown"
            skipped.append((str(path), label))
            continue
        encoded = base64.b64encode(raw).decode("utf-8")
        used_images.append(str(path))
        if openai_format:
            payloads.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        else:
            payloads.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": encoded},
                }
            )
    return payloads, used_images, skipped


def _skipped_images_metadata(skipped: list[tuple[str, str]]) -> dict[str, int | list[str]]:
    return {"count": len(skipped), "paths": [path for path, _ in skipped]}


def _raise_all_images_skipped(skipped: list[tuple[str, str]]) -> None:
    first_path, first_type = skipped[0]
    raise RuntimeError(
        f"All {len(skipped)} image(s) were skipped for the visual fallback request because "
        f"their media type is not supported; first skipped image {first_path!r} has detected type {first_type!r}."
    )


_RETRY_RANDOM = random.Random()
_TRANSIENT_VLM_ERROR_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "APIStatusError",
}
_TRANSIENT_VLM_STATUS_CODES = {429, 500, 502, 503, 504}


def _positive_int_env(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; received {raw!r}.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value}.")
    return value


def _positive_float_env(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number; received {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number; received {value}.")
    return value


def _is_transient_vlm_error(exc: Exception) -> bool:
    if exc.__class__.__name__ in _TRANSIENT_VLM_ERROR_NAMES:
        return True
    return getattr(exc, "status", None) in _TRANSIENT_VLM_STATUS_CODES


def _sleep_before_retry(backoff_seconds: float, attempts: int) -> None:
    time.sleep(backoff_seconds * (2**attempts) + _RETRY_RANDOM.uniform(0, 0.25))


@lru_cache(maxsize=1)
def _load_byt5(model_name: str, revision: str | None):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ByT5 word-level correction needs the ml extra: pip install 'faar[ml]'."
        ) from exc
    torch = import_module("torch")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    device = torch_device(torch)
    dtype = select_dtype(device, torch)
    enforce_gpu_memory_fraction(torch)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    enforce_memory_budget("ByT5 model load", torch)
    return tokenizer, model


class ByT5Corrector:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        correction_settings: CorrectionSettings | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.correction_settings = correction_settings or CorrectionSettings()

    def correct(self, text: str, max_new_tokens: int = 128) -> str:
        return self.propose_correction(text, max_new_tokens=max_new_tokens)["text"]

    def propose_correction(self, text: str, max_new_tokens: int = 128) -> dict[str, str | bool]:
        if not text.strip():
            return {"text": text, "candidate": text, "applied": False, "reason": "empty_input"}
        should_attempt, skip_reason = _should_attempt_correction(text, self.correction_settings)
        if not should_attempt:
            return {"text": text, "candidate": text, "applied": False, "reason": skip_reason}
        try:
            corrected = self._generate_correction(text, max_new_tokens=max_new_tokens)
        except ImportError:
            return {"text": text, "candidate": text, "applied": False, "reason": "byt5_ml_extra_missing"}
        except OSError:
            return {"text": text, "candidate": text, "applied": False, "reason": "byt5_model_unavailable"}
        except RuntimeError as exc:
            if is_fatal_resource_error(exc):
                raise
            return {"text": text, "candidate": text, "applied": False, "reason": "byt5_inference_failed"}
        accepted, reason = _should_accept_correction(text, corrected, self.correction_settings)
        return {
            "text": corrected if accepted else text,
            "candidate": corrected,
            "applied": accepted,
            "reason": reason,
        }

    def _generate_correction(self, text: str, max_new_tokens: int = 128) -> str:
        # Bound inference input to keep Phase 3 batch runs tractable.
        clipped = text[:512]
        tokenizer, model = _load_byt5(self.model_name, self.revision)
        prompt = f"correct ocr noise: {clipped}"
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True)
        output = model.generate(**tokens, max_new_tokens=min(max_new_tokens, 64))
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()


class VisualFallback:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.logger = make_vlm_logger(settings.project_root, enabled=settings.recovery.log_vlm_calls)

    _PAID_BACKENDS = {"openai", "claude-sonnet-4-5", "anthropic", "claude"}

    def answer(self, question: str, image_paths: list[Path], fallback_context: str) -> dict:
        if not self.settings.recovery.api_enabled and self.settings.recovery.vlm_backend in self._PAID_BACKENDS:
            return {
                "backend": "openai" if self.settings.recovery.vlm_backend == "openai" else "anthropic",
                "status": "skipped",
                "reason": "api_disabled",
                "answer": "",
                "used_images": [],
                "cost_rates": vlm_cost_rates(self.settings.recovery.vlm_backend),
                "api_usage": zero_api_usage(),
            }
        if self.settings.recovery.vlm_backend == "openai":
            return self._answer_with_openai(question, image_paths)
        if self.settings.recovery.vlm_backend in {"claude-sonnet-4-5", "anthropic", "claude"}:
            return self._answer_with_anthropic(question, image_paths, fallback_context)
        if self.settings.recovery.vlm_backend == "mock":
            return {
                "backend": "mock",
                "status": "skipped",
                "reason": "mock_backend_noop",
                "answer": "",
                "used_images": [str(path) for path in image_paths],
                "fallback_context": fallback_context[:500],
                "api_usage": zero_api_usage(),
            }
        return {
            "backend": "mock",
            "status": "skipped",
            "reason": "vlm_backend_not_configured",
            "answer": "",
            "used_images": [str(path) for path in image_paths],
            "fallback_context": fallback_context[:500],
            "api_usage": zero_api_usage(),
        }

    def _answer_with_openai(self, question: str, image_paths: list[Path]) -> dict:
        if not image_paths:
            return {
                "backend": "openai",
                "status": "skipped",
                "reason": "no_images_provided",
                "answer": "",
                "used_images": [],
                "api_usage": zero_api_usage(),
            }
        try:
            payloads, used_images, skipped = _build_image_payloads(image_paths, openai_format=True)
        except OSError as exc:
            return {
                "backend": "openai",
                "status": "failed",
                "reason": f"image_read_error:{type(exc).__name__}",
                "answer": "",
                "used_images": [str(path) for path in image_paths],
                "api_usage": make_api_usage(api_requests=0),
            }
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for VLM_BACKEND=openai.")
        client = OpenAI(max_retries=0)
        request_id = str(uuid4())
        skipped_images = _skipped_images_metadata(skipped)
        if not used_images:
            _raise_all_images_skipped(skipped)
        content: list[dict] = [{"type": "text", "text": f"Answer the question using only the page image.\n\nQuestion: {question}"}]
        content.extend(payloads)
        max_attempts = _positive_int_env("FAAR_VLM_MAX_RETRIES", 3)
        backoff_seconds = _positive_float_env("FAAR_VLM_RETRY_BACKOFF_SECONDS", 2.0)
        attempts = 0
        last_error: Exception | None = None
        response = None
        while attempts < max_attempts:
            attempts += 1
            self.logger.log(
                new_record(
                    provider="openai",
                    model=self.settings.recovery.openai_model,
                    operation="visual_fallback",
                    status="started",
                    metadata={
                        "request_id": request_id,
                        "image_count": len(used_images),
                        "skipped_images": skipped_images,
                    },
                )
            )
            try:
                response = client.chat.completions.create(
                    model=self.settings.recovery.openai_model,
                    messages=[{"role": "user", "content": content}],
                    timeout=self.settings.recovery.request_timeout_seconds,
                )
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - external runtime dependent
                last_error = exc
                if not _is_transient_vlm_error(exc) or attempts >= max_attempts:
                    break
                _sleep_before_retry(backoff_seconds, attempts)
        if last_error is not None:
            completed_at_utc = datetime.now(UTC).isoformat()
            cost_rates = openai_cost_rates()
            self.logger.log(
                new_record(
                    provider="openai",
                    model=self.settings.recovery.openai_model,
                    operation="visual_fallback",
                    status="failed",
                    metadata={
                        "request_id": request_id,
                        "error": type(last_error).__name__,
                        "request_model": self.settings.recovery.openai_model,
                        "response_model": None,
                        "completed_at_utc": completed_at_utc,
                        "cost_rates": cost_rates,
                        "skipped_images": skipped_images,
                    },
                )
            )
            return {
                "backend": "openai",
                "status": "failed",
                "reason": f"openai_error:{type(last_error).__name__}",
                "answer": "",
                "used_images": used_images,
                "skipped_images": skipped_images,
                "request_model": self.settings.recovery.openai_model,
                "response_model": None,
                "completed_at_utc": completed_at_utc,
                "cost_rates": cost_rates,
                "api_usage": make_api_usage(api_requests=attempts),
            }
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        response_model = getattr(response, "model", None) or self.settings.recovery.openai_model
        completed_at_utc = datetime.now(UTC).isoformat()
        cost_rates = openai_cost_rates()
        cost_usd = estimate_openai_cost_usd(prompt_tokens, completion_tokens, cost_rates)
        self.logger.log(
            new_record(
                provider="openai",
                model=self.settings.recovery.openai_model,
                operation="visual_fallback",
                status="succeeded",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                metadata={
                    "request_id": request_id,
                    "image_count": len(used_images),
                    "skipped_images": skipped_images,
                    "request_model": self.settings.recovery.openai_model,
                    "response_model": response_model,
                    "completed_at_utc": completed_at_utc,
                    "cost_rates": cost_rates,
                },
            )
        )
        return {
            "backend": "openai",
            "status": "succeeded",
            "reason": "visual_fallback_answer_generated",
            "answer": response.choices[0].message.content or "",
            "used_images": used_images,
            "skipped_images": skipped_images,
            "request_model": self.settings.recovery.openai_model,
            "response_model": response_model,
            "completed_at_utc": completed_at_utc,
            "cost_rates": cost_rates,
            "api_usage": make_api_usage(
                api_requests=attempts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            ),
        }

    def _answer_with_anthropic(self, question: str, image_paths: list[Path], fallback_context: str) -> dict:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for VLM_BACKEND=claude-sonnet-4-5.")
        try:  # pragma: no cover - optional external runtime
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional external runtime
            raise RuntimeError("The `anthropic` package is required for VLM_BACKEND=claude-sonnet-4-5.") from exc

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=0)
        request_id = str(uuid4())
        payloads, used_images, skipped = _build_image_payloads(image_paths, openai_format=False)
        skipped_images = _skipped_images_metadata(skipped)
        if not used_images:
            _raise_all_images_skipped(skipped)
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Answer the question using the page image when available and the OCR context only as supporting evidence.\n\n"
                    f"Question: {question}\n\nOCR context:\n{fallback_context[:4000]}"
                ),
            }
        ]
        content.extend(payloads)
        max_attempts = _positive_int_env("FAAR_VLM_MAX_RETRIES", 3)
        backoff_seconds = _positive_float_env("FAAR_VLM_RETRY_BACKOFF_SECONDS", 2.0)
        attempts = 0
        last_error: Exception | None = None
        response = None
        while attempts < max_attempts:
            attempts += 1
            self.logger.log(
                new_record(
                    provider="anthropic",
                    model=self.settings.recovery.anthropic_model,
                    operation="visual_fallback",
                    status="started",
                    metadata={
                        "request_id": request_id,
                        "image_count": len(used_images),
                        "skipped_images": skipped_images,
                    },
                )
            )
            try:
                response = client.messages.create(
                    model=self.settings.recovery.anthropic_model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": content}],
                    timeout=self.settings.recovery.request_timeout_seconds,
                )
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - external runtime dependent
                last_error = exc
                if not _is_transient_vlm_error(exc) or attempts >= max_attempts:
                    break
                _sleep_before_retry(backoff_seconds, attempts)
        if last_error is not None:
            completed_at_utc = datetime.now(UTC).isoformat()
            cost_rates = anthropic_cost_rates()
            self.logger.log(
                new_record(
                    provider="anthropic",
                    model=self.settings.recovery.anthropic_model,
                    operation="visual_fallback",
                    status="failed",
                    metadata={
                        "request_id": request_id,
                        "error": type(last_error).__name__,
                        "request_model": self.settings.recovery.anthropic_model,
                        "response_model": None,
                        "completed_at_utc": completed_at_utc,
                        "cost_rates": cost_rates,
                        "skipped_images": skipped_images,
                    },
                )
            )
            return {
                "backend": "anthropic",
                "status": "failed",
                "reason": f"anthropic_error:{type(last_error).__name__}",
                "answer": "",
                "used_images": used_images,
                "skipped_images": skipped_images,
                "request_model": self.settings.recovery.anthropic_model,
                "response_model": None,
                "completed_at_utc": completed_at_utc,
                "cost_rates": cost_rates,
                "api_usage": make_api_usage(api_requests=attempts),
            }
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        text_parts = [getattr(block, "text", "") for block in getattr(response, "content", []) if getattr(block, "type", "") == "text"]
        response_model = getattr(response, "model", None) or self.settings.recovery.anthropic_model
        completed_at_utc = datetime.now(UTC).isoformat()
        cost_rates = anthropic_cost_rates()
        cost_usd = estimate_anthropic_cost_usd(self.settings.recovery.anthropic_model, prompt_tokens, completion_tokens)
        self.logger.log(
            new_record(
                provider="anthropic",
                model=self.settings.recovery.anthropic_model,
                operation="visual_fallback",
                status="succeeded",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                metadata={
                    "request_id": request_id,
                    "image_count": len(used_images),
                    "skipped_images": skipped_images,
                    "request_model": self.settings.recovery.anthropic_model,
                    "response_model": response_model,
                    "completed_at_utc": completed_at_utc,
                    "cost_rates": cost_rates,
                },
            )
        )
        return {
            "backend": "anthropic",
            "status": "succeeded",
            "reason": "visual_fallback_answer_generated",
            "answer": "\n".join(part for part in text_parts if part).strip(),
            "used_images": used_images,
            "skipped_images": skipped_images,
            "request_model": self.settings.recovery.anthropic_model,
            "response_model": response_model,
            "completed_at_utc": completed_at_utc,
            "cost_rates": cost_rates,
            "api_usage": make_api_usage(
                api_requests=attempts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            ),
        }


def semantic_backtrack(query: str, hits: list[RetrievalHit]) -> str:
    context = " ".join(hit.chunk.text for hit in hits)
    if not context:
        return query
    anchor_words = context.split()[:24]
    return f"{query} {' '.join(anchor_words)}"


def _should_attempt_correction(
    text: str,
    settings: CorrectionSettings | None = None,
) -> tuple[bool, str]:
    thresholds = settings or CorrectionSettings()
    if _contains_cjk(text):
        return False, "non_latin_source"
    if _looks_formula_like(text):
        return False, "formula_like_source"
    if _weird_char_ratio(text) < thresholds.min_weird_char_ratio and not _has_ocr_like_token_noise(text):
        return False, "source_not_noisy_enough"
    return True, "eligible"


def _should_accept_correction(
    source: str,
    candidate: str,
    settings: CorrectionSettings | None = None,
) -> tuple[bool, str]:
    thresholds = settings or CorrectionSettings()
    if not candidate.strip():
        return False, "empty_correction"
    if _normalize_whitespace(source) == _normalize_whitespace(candidate):
        return False, "no_change"
    if _numeric_signature(source) != _numeric_signature(candidate):
        return False, "numeric_signature_changed"
    if not thresholds.min_length_ratio <= _length_ratio(source, candidate) <= thresholds.max_length_ratio:
        return False, "length_shift_too_large"
    if _informative_token_overlap(source, candidate) < thresholds.min_token_overlap:
        return False, "low_token_preservation"
    if _weird_char_ratio(candidate) > _weird_char_ratio(source) + thresholds.max_noise_increase:
        return False, "noise_not_reduced"
    return True, "accepted"


def _contains_cjk(text: str) -> bool:
    # Covers kana (3040-30ff), CJK unified ideographs (3400-9fff) and hangul
    # (ac00-d7af) so non-Latin scripts never reach the Latin-tuned corrector.
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))


def _looks_formula_like(text: str) -> bool:
    formula_markers = ("\\", "{", "}", "^", "_", "operatorname", "Delta", "alpha", "beta")
    return any(marker in text for marker in formula_markers)


def _weird_char_ratio(text: str) -> float:
    # Shares the counting mechanism with the quality gate but keeps its own
    # punctuation whitelist; see faar/textnoise.py for why the sets differ.
    return char_noise_ratio(text, CORRECTION_ALLOWED_PUNCTUATION)


def _has_ocr_like_token_noise(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]+\d+[A-Za-z]*|\d+[A-Za-z]{2,}|[A-Za-z]{2,}\d+[A-Za-z]*", text))


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


_NUMERIC_SIGNATURE_RE = re.compile(r"(?<![A-Za-z])(?:US\s*)?\$?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z])")


def _numeric_signature(text: str) -> list[str]:
    # Digits embedded inside words (e.g. "t0tal") are OCR noise, not numeric
    # facts: guard with letter boundaries so correcting them is not treated as
    # numeric drift.
    return _NUMERIC_SIGNATURE_RE.findall(text)


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
