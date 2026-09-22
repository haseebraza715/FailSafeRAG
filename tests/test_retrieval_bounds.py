import numpy as np
import pytest

from faar.retrieval import HybridRetriever, LocalHashEmbedder
from faar.settings import RetrievalSettings
from faar.types import Chunk


def _chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        example_id="example",
        doc_name="doc",
        page_id=0,
        text=f"bounded retrieval content {index}",
    )


def test_local_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = LocalHashEmbedder()
    first = embedder.encode(["same text"], normalize_embeddings=True, convert_to_numpy=True)
    second = embedder.encode(["same text"], normalize_embeddings=True, convert_to_numpy=True)
    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_retriever_rejects_configured_chunk_overflow_before_embedding() -> None:
    settings = RetrievalSettings(max_chunks=1, embedding_backend="local-hash-v1")
    with pytest.raises(ValueError, match="exceeds configured max_chunks"):
        HybridRetriever([_chunk(1), _chunk(2)], settings)
