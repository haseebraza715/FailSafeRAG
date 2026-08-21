from __future__ import annotations

from pathlib import Path

import numpy as np

from faar.retrieval import _corpus_text_digest, _encode_corpus_embeddings
from faar.settings import RetrievalSettings
from faar.types import Chunk


def _chunks(count: int = 6) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc-p{i // 2}-c{i % 2}",
            example_id="doc",
            doc_name="doc",
            page_id=i // 2,
            text=f"shared corpus text chunk number {i} with an answer forty two",
        )
        for i in range(count)
    ]


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.last_batch_size = None

    def encode(self, texts, batch_size=None, normalize_embeddings=False, convert_to_numpy=False):
        self.calls += 1
        self.last_batch_size = batch_size
        rows = np.array(
            [[float(ord(char) % 7) / 7.0 for char in text[:16]] for text in texts], dtype=np.float32
        )
        return rows if rows.size else np.zeros((0, 16), dtype=np.float32)


def _settings(tmp_path: Path) -> RetrievalSettings:
    return RetrievalSettings()


def test_text_corpus_cache_returns_identical_embeddings(tmp_path: Path) -> None:
    chunks = _chunks()
    settings = _settings(tmp_path)
    cache_dir = tmp_path / "cache"
    embedder = _FakeEmbedder()

    first = _encode_corpus_embeddings(embedder, chunks, settings, cache_dir)
    assert embedder.calls == 1
    assert len(list(cache_dir.glob("*.npz"))) == 1
    assert len(list(cache_dir.glob("*.meta.json"))) == 1

    second = _encode_corpus_embeddings(embedder, chunks, settings, cache_dir)
    assert embedder.calls == 1
    np.testing.assert_array_equal(first, second)


def test_text_corpus_cache_recomputes_on_content_change(tmp_path: Path) -> None:
    chunks = _chunks()
    settings = _settings(tmp_path)
    cache_dir = tmp_path / "cache"
    embedder = _FakeEmbedder()

    _encode_corpus_embeddings(embedder, chunks, settings, cache_dir)
    changed = list(chunks)
    changed[0] = Chunk(
        chunk_id=chunks[0].chunk_id,
        example_id=chunks[0].example_id,
        doc_name=chunks[0].doc_name,
        page_id=chunks[0].page_id,
        text="completely different ocr text",
    )
    _encode_corpus_embeddings(embedder, changed, settings, cache_dir)
    assert embedder.calls == 2


def test_text_corpus_cache_recomputes_on_model_change(tmp_path: Path) -> None:
    chunks = _chunks()
    cache_dir = tmp_path / "cache"
    embedder = _FakeEmbedder()

    _encode_corpus_embeddings(embedder, chunks, _settings(tmp_path), cache_dir)
    settings = _settings(tmp_path)
    settings.embedding_model = "other/model"
    _encode_corpus_embeddings(embedder, chunks, settings, cache_dir)
    assert embedder.calls == 2


def test_text_corpus_cache_recomputes_on_corrupt_npz(tmp_path: Path) -> None:
    chunks = _chunks()
    settings = _settings(tmp_path)
    cache_dir = tmp_path / "cache"
    embedder = _FakeEmbedder()

    _encode_corpus_embeddings(embedder, chunks, settings, cache_dir)
    npz_path = next(cache_dir.glob("*.npz"))
    npz_path.write_bytes(b"not an npz file at all")
    _encode_corpus_embeddings(embedder, chunks, settings, cache_dir)
    assert embedder.calls == 2


def test_text_digest_is_stable_and_sensitive(tmp_path: Path) -> None:
    chunks = _chunks()
    assert _corpus_text_digest(chunks) == _corpus_text_digest(list(chunks))
    assert _corpus_text_digest(chunks) != _corpus_text_digest(chunks[:3])


