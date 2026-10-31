"""Gate and diagnosis for the FAAR comparison (docs/research/faar-vs-gated-preregistration.md).

Everything here reads runtime inputs only: the question, the retrieved chunks with their scores and text, the
initial answer's outcome, and the per-page OCR status of the document (study brief section 4). Nothing reads gold
answers, evidence pages, evidence contexts or clean reference text.

Gate. ``abstain-or-ocr-gap-v1`` flags a question whose initial outcome is ``no_evidence``, whose initial answer
abstained, or whose document has a page with empty or missing MinerU text. A ``GateRule`` may add one
retrieval-confidence condition, a feature below a threshold. The feature and threshold are chosen on development
data only (analysis plan, amendment 3).

Diagnosis, first match wins:

=============  ===============================================================================  ================
missing_text   no text evidence was retrieved, or a page the page policy would show has          invoke_vlm
               MinerU text that is not ``ok``
structural     the top chunk shows at least ``structural_threshold`` layout signals            invoke_vlm
word_level     the top chunk's ``weird_char_ratio`` is above ``weird_char_threshold``          correct_text
semantic       none of the above                                                               retry_retrieval
=============  ===============================================================================  ================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .quality import layout_signals, weird_char_ratio
from .text_units import tokenize
from .visual_repair import DEFAULT_BUDGET, select_pages

GATE_BASE_ID = "abstain-or-ocr-gap-v1"
DIAGNOSIS_ID = "faar-diagnosis-v1"
CONFIDENCE_FEATURES = ("top_fused_score", "fused_margin", "query_token_coverage")
ROUTE = {
    "missing_text": "invoke_vlm",
    "structural": "invoke_vlm",
    "word_level": "correct_text",
    "semantic": "retry_retrieval",
}
DEFAULT_STRUCTURAL_THRESHOLD = 1
DEFAULT_WEIRD_CHAR_THRESHOLD = 0.10


class DiagnosisError(Exception):
    """A gate or diagnosis input is invalid."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk as the diagnosis sees it: rank, page, score and the noisy text."""

    rank: int
    page_idx: int
    fused_score: float
    text: str


@dataclass(frozen=True)
class GateRule:
    """The base gate, plus at most one retrieval-confidence condition: ``feature < threshold`` flags."""

    feature: str | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if (self.feature is None) != (self.threshold is None):
            raise DiagnosisError("a confidence condition needs both a feature and a threshold")
        if self.feature is not None and self.feature not in CONFIDENCE_FEATURES:
            raise DiagnosisError(f"unknown confidence feature {self.feature!r}; known: {CONFIDENCE_FEATURES}")

    @property
    def gate_id(self) -> str:
        if self.feature is None:
            return GATE_BASE_ID
        return f"{GATE_BASE_ID}+{self.feature}<{self.threshold:g}"


def confidence_features(question: str, chunks: Sequence[RetrievedChunk], tokenizer: str) -> dict[str, float | None]:
    """Retrieval-confidence features of one question. All are ``None`` when nothing was retrieved.

    Scores are min-max normalised within the document, so the top fused score is a weak signal on its own; the
    margin and the share of distinct query tokens found in the retrieved text are recorded next to it.
    """
    if not chunks:
        return dict.fromkeys(CONFIDENCE_FEATURES)
    ranked = sorted(chunks, key=lambda c: c.rank)
    top = ranked[0].fused_score
    second = ranked[1].fused_score if len(ranked) > 1 else 0.0
    query_tokens = set(tokenize(question, tokenizer))
    found = set(tokenize(" ".join(c.text for c in ranked), tokenizer))
    coverage = len(query_tokens & found) / len(query_tokens) if query_tokens else 0.0
    return {"top_fused_score": top, "fused_margin": top - second, "query_token_coverage": coverage}


def gate(
    *,
    initial_status: str,
    abstained: bool,
    pages: Sequence[Mapping[str, Any]],
    features: Mapping[str, float | None],
    rule: GateRule,
) -> tuple[bool, list[str]]:
    """Whether the gate flags the question, and every reason that fired."""
    reasons = []
    if initial_status == "no_evidence":
        reasons.append("no_evidence")
    if abstained:
        reasons.append("abstained")
    if any(p["ocr_status"] != "ok" for p in pages):
        reasons.append("document_ocr_gap")
    if rule.feature is not None:
        value = features.get(rule.feature)
        if value is not None and value < rule.threshold:
            reasons.append(f"low_{rule.feature}")
    return bool(reasons), reasons


def diagnose(
    *,
    initial_sent: bool,
    chunks: Sequence[RetrievedChunk],
    pages: Sequence[Mapping[str, Any]],
    budget: int = DEFAULT_BUDGET,
    structural_threshold: int = DEFAULT_STRUCTURAL_THRESHOLD,
    weird_char_threshold: float = DEFAULT_WEIRD_CHAR_THRESHOLD,
) -> dict[str, Any]:
    """The diagnosis of one flagged question, its route, and the signal values behind it."""
    if initial_sent and not chunks:
        raise DiagnosisError("the initial request was sent, so it must have retrieved chunks")
    ranked = sorted(chunks, key=lambda c: c.rank)
    retrieved_pages = [c.page_idx for c in ranked]
    shown = select_pages(pages, retrieved_pages, budget)
    status = {int(p["page_idx"]): p["ocr_status"] for p in pages}
    gap_pages = [p for p in shown if status[p] != "ok"]
    top_text = ranked[0].text if ranked else ""
    layout = layout_signals(top_text)
    signals = {
        "no_text_evidence": not initial_sent,
        "shown_pages": shown,
        "shown_pages_with_ocr_gap": gap_pages,
        "layout_signals": layout,
        "layout_signal_count": sum(layout.values()),
        "weird_char_ratio": weird_char_ratio(top_text),
        "structural_threshold": structural_threshold,
        "weird_char_threshold": weird_char_threshold,
    }
    if not initial_sent or gap_pages:
        label = "missing_text"
    elif signals["layout_signal_count"] >= structural_threshold:
        label = "structural"
    elif signals["weird_char_ratio"] > weird_char_threshold:
        label = "word_level"
    else:
        label = "semantic"
    return {"diagnosis": label, "route": ROUTE[label], "signals": signals}
