from __future__ import annotations

import re

from .types import RetrievalHit

TOKEN_RE = re.compile(r"[a-z0-9%$]+")
MONTH_PATTERN = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
DATE_PATTERNS = [
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}}|\s+of\s+each\s+year|\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:days?|months?|years?)\b", re.IGNORECASE),
    re.compile(r"\b[a-z-]+\s*\(\d+\)\s*days\b", re.IGNORECASE),
]
RANGE_PATTERN = re.compile(r"(?:US\s*)?\$?\d[\d,]*(?:\.\d+)?\s*(?:to|-|–)\s*(?:US\s*)?\$?\d[\d,]*(?:\.\d+)?")
PERCENT_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%")
CURRENCY_PATTERN = re.compile(r"(?:US\s*)?\$[\d,]+(?:\.\d+)?")
NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s?(?:mm|cm|m|ns|ms|s|%|years?|days?|months?|hours?|eur|usd|rm))?",
    re.IGNORECASE,
)
YES_NO_PREFIXES = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "should ", "has ", "have ", "had ", "will ", "would ")
YES_CUES = (" is ", " are ", " was ", " were ", " mandatory", " required", " must ", " shall ", " yes")
NO_CUES = (" not ", " no ", " never ", " prohibited", " forbidden", " optional", " not required", " cannot ", " can't ")
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "why",
}


def answer_from_hits(question: str, hits: list[RetrievalHit]) -> dict:
    question_terms = {token for token in TOKEN_RE.findall(question.lower()) if token not in STOPWORDS}
    best_text = ""
    best_score = -1
    best_mode = "extractive_overlap"
    best_source_chunk = ""
    for hit in hits:
        candidates = _segment_candidates(hit.chunk.text)
        for candidate in candidates:
            candidate_terms = set(TOKEN_RE.findall(candidate.lower()))
            score = len(question_terms & candidate_terms) + _candidate_bonus(question, candidate)
            score += int(round(hit.fused_score * 4))
            extracted, mode = _extract_answer(question, candidate)
            if extracted and extracted != candidate.strip():
                score += 2
            if len(candidate.split()) <= 12:
                score += 1
            if score > best_score:
                best_score = score
                best_text = extracted or candidate.strip()
                best_mode = mode
                best_source_chunk = hit.chunk.chunk_id
    return {
        "answer": best_text or hits[0].chunk.text[:240] if hits else "",
        "answer_mode": best_mode,
        "support_score": best_score,
        "source_chunk_id": best_source_chunk,
    }


def _candidate_bonus(question: str, candidate: str) -> int:
    bonus = 0
    question_lower = question.lower()
    if any(term in question_lower for term in ["condition", "equation", "formula"]):
        if "=" in candidate or "+" in candidate:
            bonus += 4
    if any(term in question_lower for term in ["percentage", "amount", "total", "difference", "width"]):
        if re.search(r"\d", candidate):
            bonus += 2
    return bonus


def _segment_candidates(text: str) -> list[str]:
    segments: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        segments.append(stripped)
        if len(stripped.split()) > 20:
            for sentence in re.split(r"(?<=[\.\?\!;])\s+", stripped):
                sentence = sentence.strip()
                if sentence and sentence != stripped:
                    segments.append(sentence)
    return segments or [text.strip()]


def _extract_answer(question: str, candidate: str) -> tuple[str, str]:
    cleaned_candidate = candidate.strip().strip(":;,.")
    if not cleaned_candidate:
        return "", "extractive_overlap"
    question_lower = question.lower().strip()
    if question_lower.startswith(YES_NO_PREFIXES):
        verdict = _extract_yes_no_answer(cleaned_candidate)
        if verdict:
            return verdict, "yes_no_inference"
    if _expects_date_like_answer(question_lower):
        extracted = _first_match(cleaned_candidate, DATE_PATTERNS)
        if extracted:
            return extracted, "date_span"
    if _expects_range_answer(question_lower):
        extracted = _first_match(cleaned_candidate, [RANGE_PATTERN])
        if extracted:
            return extracted, "range_span"
    if _expects_numeric_answer(question_lower):
        extracted = _extract_numeric_span(question_lower, cleaned_candidate)
        if extracted:
            return extracted, "numeric_span"
    return cleaned_candidate, "extractive_overlap"


def _extract_yes_no_answer(candidate: str) -> str:
    lower = f" {candidate.lower()} "
    if any(cue in lower for cue in NO_CUES):
        return "No"
    if any(cue in lower for cue in YES_CUES):
        return "Yes"
    return ""


def _expects_numeric_answer(question: str) -> bool:
    triggers = (
        "how many",
        "how much",
        "percentage",
        "amount",
        "total",
        "difference",
        "width",
        "maximum",
        "minimum",
        "fall time",
        "copay",
        "price",
    )
    return any(trigger in question for trigger in triggers)


def _expects_range_answer(question: str) -> bool:
    return "range" in question or "ranging" in question


def _expects_date_like_answer(question: str) -> bool:
    return "deadline" in question or question.startswith("when ") or "month" in question or "time frame" in question


def _extract_numeric_span(question: str, candidate: str) -> str:
    patterns: list[re.Pattern[str]] = []
    if "percentage" in question:
        patterns.append(PERCENT_PATTERN)
    if any(term in question for term in ["amount", "total", "copay", "price"]):
        patterns.extend([CURRENCY_PATTERN, NUMBER_WITH_UNIT_PATTERN])
    if any(term in question for term in ["width", "fall time", "maximum", "minimum", "difference", "how many", "how much"]):
        patterns.append(NUMBER_WITH_UNIT_PATTERN)
    patterns.extend([RANGE_PATTERN, PERCENT_PATTERN, CURRENCY_PATTERN, NUMBER_WITH_UNIT_PATTERN])
    for pattern in patterns:
        match = pattern.search(candidate)
        if match:
            return match.group(0).strip()
    return ""


def _first_match(candidate: str, patterns: list[re.Pattern[str]]) -> str:
    for pattern in patterns:
        match = pattern.search(candidate)
        if match:
            return match.group(0).strip()
    return ""
