from __future__ import annotations

import re
from statistics import mean

from .settings import GateSettings
from .types import RetrievalHit


def weird_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird_chars = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,:%$()-/=+"))
    char_ratio = weird_chars / max(len(text), 1)
    tokens = re.findall(r"\S+", text)
    weird_tokens = 0
    for token in tokens:
        core = token.strip(".,;:()[]{}")
        if not core:
            continue
        mixed_digit_letter = bool(re.search(r"(?=.*[A-Za-z])(?=.*\d)", core))
        punct_inside = bool(re.search(r"[A-Za-z0-9][,;:/][A-Za-z0-9]", core))
        symbol_dense = (sum(not ch.isalnum() for ch in core) / max(len(core), 1)) > 0.2
        ocr_glued_token = bool(re.search(r"[a-z]{3,}[A-Z][A-Za-z]*", core))
        if mixed_digit_letter or punct_inside or symbol_dense or ocr_glued_token:
            weird_tokens += 1
    token_ratio = weird_tokens / max(len(tokens), 1)
    return max(char_ratio, token_ratio)


def layout_signals(text: str) -> dict[str, bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    lengths = [len(line) for line in lines] or [0]
    return {
        "excess_whitespace": len(re.findall(r" {3,}", text)) > 5,
        "bare_numbers": len(re.findall(r"\n\d+[\s\d,\.]+\n", text)) > 2,
        "length_variance": max(lengths) / (min(lengths) + 1) > 12 if lengths else False,
        "pipe_heavy": text.count("|") > 8,
        "fraction_heavy": len(re.findall(r"\b\d+/\d+\b", text)) >= 3,
    }


def quality_gate(hits: list[RetrievalHit], settings: GateSettings) -> dict:
    if not hits:
        return {
            "quality_score": 0.0,
            "pass_gate": False,
            "reasons": ["no_retrieval_hits"],
            "layout_signal_count": 0,
            "corruption_score": 1.0,
        }

    top = hits[0]
    corruption_score = weird_char_ratio(top.chunk.text)
    layout = layout_signals(top.chunk.text)
    layout_count = sum(layout.values())
    lexical = top.bm25_score
    dense = top.dense_score
    quality_score = (0.40 * dense) + (0.35 * lexical) + (0.25 * (1.0 - min(corruption_score, 1.0)))
    reasons: list[str] = []
    if quality_score < settings.quality_threshold:
        reasons.append("low_quality_score")
    if layout_count >= settings.structural_threshold:
        reasons.append("layout_alert")
    if corruption_score > settings.weird_char_threshold:
        reasons.append("word_noise_alert")
    if lexical < settings.lexical_floor:
        reasons.append("low_lexical_score")
    if dense < settings.dense_floor:
        reasons.append("low_dense_score")

    return {
        "quality_score": round(quality_score, 4),
        "pass_gate": not reasons,
        "reasons": reasons,
        "layout_signal_count": layout_count,
        "layout_signals": layout,
        "corruption_score": round(corruption_score, 4),
        "avg_hit_score": round(mean(hit.fused_score for hit in hits), 4),
    }


def diagnose_failure(hits: list[RetrievalHit], gate_result: dict, settings: GateSettings) -> str:
    if not hits:
        return "semantic"
    if gate_result["layout_signal_count"] >= settings.structural_threshold:
        return "structural"
    if gate_result["corruption_score"] > settings.weird_char_threshold:
        return "word_level"
    return "semantic"
