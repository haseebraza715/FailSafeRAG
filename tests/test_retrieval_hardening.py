import math

import numpy as np

from faar.retrieval import HybridRetriever, LocalHashEmbedder
from faar.settings import RetrievalSettings
from faar.types import Chunk


def _chunk(index: int, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        example_id="example",
        doc_name="doc",
        page_id=0,
        text=text or f"retrieval content chunk number {index}",
    )


def _retriever(n: int = 6, texts: list[str] | None = None) -> HybridRetriever:
    chunks = [_chunk(i, texts[i] if texts else None) for i in range(n)]
    return HybridRetriever(chunks, RetrievalSettings(embedding_backend="local-hash-v1"))


def test_top_k_zero_or_negative_returns_empty() -> None:
    retriever = _retriever()
    assert retriever.retrieve("retrieval content", top_k=0) == []
    assert retriever.retrieve("retrieval content", top_k=-3) == []


def test_top_k_none_uses_settings_default() -> None:
    retriever = _retriever(n=7)
    hits = retriever.retrieve("retrieval content", top_k=None)
    assert len(hits) == RetrievalSettings().top_k


def test_top_k_beyond_corpus_clamps_to_corpus_size() -> None:
    retriever = _retriever(n=3)
    hits = retriever.retrieve("retrieval content", top_k=100)
    assert len(hits) == 3
    for hit in hits:
        assert math.isfinite(hit.dense_score)
        assert math.isfinite(hit.bm25_score)
        assert math.isfinite(hit.fused_score)


def test_identical_chunks_tie_break_by_chunk_order_deterministically() -> None:
    texts = ["identical text", "identical text", "identical text"]
    retriever = _retriever(n=3, texts=texts)
    first = retriever.retrieve("identical text", top_k=3)
    second = retriever.retrieve("identical text", top_k=3)
    assert [hit.chunk.chunk_id for hit in first] == ["c0", "c1", "c2"]
    assert [(hit.chunk.chunk_id, hit.fused_score) for hit in first] == [
        (hit.chunk.chunk_id, hit.fused_score) for hit in second
    ]


def test_repeated_retrieval_is_byte_identical() -> None:
    retriever = _retriever(n=6)
    query = "what is the retrieval content about?"
    first = [(h.chunk.chunk_id, h.bm25_score, h.dense_score, h.fused_score) for h in retriever.retrieve(query)]
    second = [(h.chunk.chunk_id, h.bm25_score, h.dense_score, h.fused_score) for h in retriever.retrieve(query)]
    assert first == second


def test_empty_query_still_returns_finite_hits() -> None:
    retriever = _retriever(n=4)
    hits = retriever.retrieve("", top_k=3)
    assert len(hits) == 3
    for hit in hits:
        assert math.isfinite(hit.fused_score)


def test_punctuation_only_query_returns_hits() -> None:
    retriever = _retriever(n=4)
    hits = retriever.retrieve("!!! ???", top_k=2)
    assert len(hits) == 2


def test_hits_are_ranked_by_fused_score_descending() -> None:
    retriever = _retriever(n=6)
    hits = retriever.retrieve("retrieval content chunk number", top_k=6)
    scores = [hit.fused_score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({hit.chunk.chunk_id for hit in hits}) == len(hits)


def test_local_hash_embedder_edge_inputs() -> None:
    embedder = LocalHashEmbedder()
    empty_list = embedder.encode([], normalize_embeddings=True, convert_to_numpy=True)
    assert empty_list.shape == (0, LocalHashEmbedder.dimensions)
    zeros = embedder.encode([""], normalize_embeddings=True, convert_to_numpy=True)
    assert np.all(zeros == 0.0)
    assert np.isclose(np.linalg.norm(zeros[0]), 0.0)
    noisy = embedder.encode(["!!! ###"], normalize_embeddings=True, convert_to_numpy=True)
    assert np.all(noisy == 0.0)


def test_local_hash_is_normalized_to_unit_norm() -> None:
    embedder = LocalHashEmbedder()
    matrix = embedder.encode(["alpha beta", "gamma delta"], normalize_embeddings=True, convert_to_numpy=True)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_unsupported_embedding_backend_raises_clear_error() -> None:
    import pytest

    settings = RetrievalSettings(embedding_backend="bogus-backend")
    with pytest.raises(ValueError, match="Unsupported embedding backend"):
        HybridRetriever([_chunk(0)], settings)
