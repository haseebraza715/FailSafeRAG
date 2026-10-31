"""Per-case measurements for the failure analysis. Pure functions, no file access.

Every metric here is FAAR-defined and secondary unless it says it copies an
OHR-Bench formula. The official EM and F1 come from ``faar.ohr_scoring``.

- ``levenshtein``: character edit distance (Myers/Hyyrö bit-parallel, exact).
- ``page_cer`` / ``page_ned``: MinerU against the clean reference text of one
  page, after collapsing whitespace runs. CER divides by the reference length,
  NED by the longer string (the usual OCR "normalised edit distance").
- ``page_recall_at_k``: share of gold evidence pages among the pages of the
  top-k retrieved chunks.
- ``retrieval_lcs``: OHR-Bench's retrieval metric copied from upstream
  ``lcs_score``: word-level LCS of the gold evidence context inside the joined
  retrieved text, divided by the evidence length. Upstream splits on spaces,
  so unspaced Han text becomes a few long "words"; ``retrieval_lcs_chars``
  is the same ratio over characters and is FAAR-defined.
- ``anls``: average normalised Levenshtein similarity (threshold 0.5) of one
  answer against one reference, lowercased with whitespace collapsed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from faar.ohr_scoring import normalize_answer

_WS = re.compile(r"\s+")


def collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Exact Levenshtein distance by Hyyrö's bit-vector form of Myers' algorithm."""
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq: dict[str, int] = {}
    for i, ch in enumerate(b):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for ch in a:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def page_cer(noisy: str, clean: str) -> float | None:
    clean, noisy = collapse(clean), collapse(noisy)
    if not clean:
        return None
    return levenshtein(noisy, clean) / len(clean)


def page_ned(noisy: str, clean: str) -> float | None:
    clean, noisy = collapse(clean), collapse(noisy)
    longest = max(len(clean), len(noisy))
    if longest == 0:
        return None
    return levenshtein(noisy, clean) / longest


def page_recall_at_k(retrieved_pages: Sequence[int], evidence_pages: Sequence[int], k: int) -> float:
    top = set(retrieved_pages[:k])
    return sum(1 for p in evidence_pages if p in top) / len(evidence_pages)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def retrieval_lcs(retrieved_text: str, evidence: str) -> float:
    """Upstream ``lcs_score(prediction=retrieved, ground_truth=evidence)``; 0.5 when the evidence is empty, as upstream."""
    gold = normalize_answer(evidence).split()
    got = normalize_answer(retrieved_text).split()
    if not gold:
        return 0.5
    return _lcs_len(gold, got) / len(gold)


def retrieval_lcs_chars(retrieved_text: str, evidence: str) -> float | None:
    gold = normalize_answer(evidence).replace(" ", "")
    got = normalize_answer(retrieved_text).replace(" ", "")
    if not gold:
        return None
    return _lcs_len(list(gold), list(got)) / len(gold)


def anls(prediction: str, reference: str, threshold: float = 0.5) -> float:
    pred, ref = collapse(prediction).lower(), collapse(reference).lower()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    distance = levenshtein(pred, ref) / max(len(pred), len(ref))
    return 1.0 - distance if distance < threshold else 0.0
