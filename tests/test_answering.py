from faar.answering import answer_from_hits
from faar.types import Chunk, RetrievalHit


def _hit(text: str, fused: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            chunk_id="c1",
            example_id="ex1",
            doc_name="doc",
            page_id=0,
            text=text,
        ),
        bm25_score=0.5,
        dense_score=0.5,
        fused_score=fused,
    )


def test_answer_from_hits_returns_yes_for_yes_no_question() -> None:
    result = answer_from_hits(
        "Is it mandatory for the system to be installed by certified engineers?",
        [_hit("It is mandatory that the system be installed by certified engineers.")],
    )
    assert result["answer"] == "Yes"
    assert result["answer_mode"] == "yes_no_inference"


def test_answer_from_hits_extracts_percentage() -> None:
    result = answer_from_hits(
        "What percentage of transactions were subject to collateral agreements?",
        [_hit("At December 31, 2021, 88% of transactions were subject to collateral agreements.")],
    )
    assert result["answer"] == "88%"
    assert result["answer_mode"] == "numeric_span"


def test_answer_from_hits_extracts_range() -> None:
    result = answer_from_hits(
        "What is the range of premium increases per pay period?",
        [_hit("Workers would see premium increases per pay period ranging from $22 to $35.80.")],
    )
    assert result["answer"] == "$22 to $35.80"


def test_answer_from_hits_extracts_date_like_deadline() -> None:
    result = answer_from_hits(
        "What is the deadline for submitting the annual plan?",
        [_hit("The deadline for the annual plan is January 1 of each year.")],
    )
    assert result["answer"] == "January 1 of each year"
    assert result["answer_mode"] == "date_span"

