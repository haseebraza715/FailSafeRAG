from __future__ import annotations

import re

from .types import RetrievalHit


def answer_from_hits(question: str, hits: list[RetrievalHit]) -> dict:
    question_terms = set(re.findall(r"[a-z0-9%$]+", question.lower()))
    best_text = ""
    best_score = -1
    for hit in hits:
        candidates = re.split(r"(?<=[\.\?\!])\s+|\n", hit.chunk.text)
        for candidate in candidates:
            candidate_terms = set(re.findall(r"[a-z0-9%$]+", candidate.lower()))
            score = len(question_terms & candidate_terms) + _candidate_bonus(question, candidate)
            if score > best_score:
                best_score = score
                best_text = candidate.strip()
    return {
        "answer": best_text or hits[0].chunk.text[:240] if hits else "",
        "answer_mode": "extractive_overlap",
        "support_score": best_score,
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
