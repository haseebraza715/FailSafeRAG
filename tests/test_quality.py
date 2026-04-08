from faar.quality import diagnose_failure, quality_gate
from faar.settings import GateSettings
from faar.types import Chunk, RetrievalHit


def _hit(text: str, bm25: float = 0.5, dense: float = 0.5, fused: float = 0.5) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            chunk_id="c1",
            example_id="e1",
            doc_name="doc",
            page_id=0,
            text=text,
        ),
        bm25_score=bm25,
        dense_score=dense,
        fused_score=fused,
    )


def test_quality_gate_handles_empty_hits() -> None:
    gate = quality_gate([], GateSettings())
    assert gate["pass_gate"] is False
    assert "no_retrieval_hits" in gate["reasons"]


def test_diagnose_failure_structural_priority() -> None:
    settings = GateSettings(structural_threshold=1, weird_char_threshold=0.9)
    hits = [_hit("a | b | c | d | e | f | g | h | i 1/2 2/3 3/4")]
    gate = quality_gate(hits, settings)
    failure = diagnose_failure(hits, gate, settings)
    assert failure == "structural"


def test_diagnose_failure_word_level_when_corrupt() -> None:
    settings = GateSettings(structural_threshold=10, weird_char_threshold=0.05)
    hits = [_hit("aB12 x/y z:w q1w2e3r4")]
    gate = quality_gate(hits, settings)
    failure = diagnose_failure(hits, gate, settings)
    assert failure == "word_level"
