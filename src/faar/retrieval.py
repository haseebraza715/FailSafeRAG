from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from functools import lru_cache
from hashlib import blake2b
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .resource_limits import (
    enforce_gpu_memory_fraction,
    enforce_memory_budget,
    release_cuda_cache,
    select_dtype,
    torch_device,
)
from .settings import RetrievalSettings
from .types import Chunk, RetrievalHit

MODEL_ALIASES = {
    "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

CORPUS_CACHE_SCHEMA_VERSION = 1
LOCAL_HASH_BACKEND = "local-hash-v1"
SENTENCE_TRANSFORMERS_BACKEND = "sentence-transformers"


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if not math.isfinite(low) or not math.isfinite(high) or math.isclose(low, high):
        return np.ones_like(values)
    return (values - low) / (high - low)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%$]+", text.lower())


def _corpus_text_digest(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _is_cuda_oom(exc: BaseException, torch_module: Any | None = None) -> bool:
    if torch_module is not None and isinstance(exc, getattr(torch_module.cuda, "OutOfMemoryError", ())):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _encode_with_oom_retry(
    embedder: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    torch = import_module("torch")
    while True:
        try:
            return (
                embedder.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                ).astype("float32")
            )
        except Exception as exc:
            if not _is_cuda_oom(exc, torch):
                raise
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            release_cuda_cache(torch)
            print(
                f"[faar] CUDA OOM during corpus encode; halving batch size -> {batch_size}",
                flush=True,
            )


def _encode_corpus_embeddings(
    embedder: SentenceTransformer,
    chunks: list[Chunk],
    settings: RetrievalSettings,
    cache_dir: Path | None,
) -> np.ndarray:
    if cache_dir is None or not chunks:
        return _encode_with_oom_retry(
            embedder,
            [chunk.text for chunk in chunks],
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
        )
    torch = import_module("torch")
    dtype_name = str(select_dtype(torch_device(torch), torch))
    text_digest = _corpus_text_digest(chunks)
    key = hashlib.sha256(
        f"{CORPUS_CACHE_SCHEMA_VERSION}|{settings.embedding_model}|"
        f"{settings.embedding_revision}|{dtype_name}|{text_digest}".encode("utf-8")
    ).hexdigest()
    cache_dir = cache_dir.expanduser().resolve()
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.meta.json"
    if npz_path.is_file() and meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            valid = (
                stored.get("schema_version") == CORPUS_CACHE_SCHEMA_VERSION
                and stored.get("model") == settings.embedding_model
                and stored.get("revision") == settings.embedding_revision
                and stored.get("dtype") == dtype_name
                and stored.get("text_digest") == text_digest
                and stored.get("count") == len(chunks)
            )
            if valid:
                embeddings = np.load(npz_path, allow_pickle=False)["embeddings"]
                if embeddings.shape == (len(chunks), int(stored["dim"])) and embeddings.dtype == np.float32:
                    print(
                        f"[faar] corpus embeddings cache hit: {npz_path.name} ({len(chunks)} chunks)",
                        flush=True,
                    )
                    return embeddings
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile):
            pass
    embeddings = _encode_with_oom_retry(
        embedder,
        [chunk.text for chunk in chunks],
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_fd, tmp_npz_name = tempfile.mkstemp(prefix=f".{npz_path.stem}.", suffix=".tmp.npz", dir=str(cache_dir))
        os.close(npz_fd)
        meta_fd, tmp_meta_name = tempfile.mkstemp(prefix=f".{meta_path.name}.", suffix=".tmp", dir=str(cache_dir))
        os.close(meta_fd)
        tmp_npz = Path(tmp_npz_name)
        tmp_meta = Path(tmp_meta_name)
        try:
            np.savez(tmp_npz, embeddings=embeddings)
            meta = {
                "schema_version": CORPUS_CACHE_SCHEMA_VERSION,
                "model": settings.embedding_model,
                "revision": settings.embedding_revision,
                "dtype": dtype_name,
                "text_digest": text_digest,
                "count": len(chunks),
                "dim": int(embeddings.shape[1]),
            }
            tmp_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_npz, npz_path)
            os.replace(tmp_meta, meta_path)
        except BaseException:
            for leftover in (tmp_npz, tmp_meta):
                try:
                    os.unlink(leftover)
                except OSError:
                    pass
            raise
    except OSError as exc:
        print(f"[faar] could not persist corpus embedding cache: {exc}", flush=True)
    return embeddings


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str, revision: str | None) -> SentenceTransformer:
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    torch = import_module("torch")
    device = torch_device(torch)
    dtype = select_dtype(device, torch)
    enforce_gpu_memory_fraction(torch)
    return SentenceTransformer(
        resolved_name,
        revision=revision,
        trust_remote_code=True,
        device=str(device),
        model_kwargs={"torch_dtype": dtype},
    )


@lru_cache(maxsize=2)
def _load_reranker(model_name: str, revision: str | None) -> CrossEncoder:
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    torch = import_module("torch")
    device = torch_device(torch)
    dtype = select_dtype(device, torch)
    enforce_gpu_memory_fraction(torch)
    return CrossEncoder(
        resolved_name,
        revision=revision,
        trust_remote_code=True,
        device=str(device),
        automodel_args={"torch_dtype": dtype},
    )


def _to_probability(score: float) -> float:
    """Map a cross-encoder logit to the [0, 1] gate score range."""
    return float(1.0 / (1.0 + np.exp(-score)))


class _Embedder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        batch_size: int | None = None,
    ) -> np.ndarray: ...


class LocalHashEmbedder:
    """Deterministic dependency-free feature hashing for offline demos/tests."""

    dimensions = 256

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        batch_size: int | None = None,
    ) -> np.ndarray:
        del convert_to_numpy, batch_size
        matrix = np.zeros((len(sentences), self.dimensions), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            for token in _tokenize(sentence):
                digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1.0, norms)
        return matrix


def _uses_local_hash(settings: RetrievalSettings) -> bool:
    return settings.embedding_backend == LOCAL_HASH_BACKEND


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        settings: RetrievalSettings,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        if len(chunks) > settings.max_chunks:
            raise ValueError(
                f"chunk count {len(chunks)} exceeds configured max_chunks={settings.max_chunks}"
            )
        self.chunks = chunks
        self.settings = settings
        self._bm25_tokens = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)
        self._local_hash = _uses_local_hash(settings)
        if self._local_hash:
            self._embedder: _Embedder = LocalHashEmbedder()
            self._reranker = None
            corpus_embeddings = self._embedder.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=settings.embedding_batch_size,
            ).astype("float32")
        elif settings.embedding_backend == SENTENCE_TRANSFORMERS_BACKEND:
            self._embedder = _load_embedding_model(settings.embedding_model, settings.embedding_revision)
            self._reranker = _load_reranker(settings.reranker, settings.reranker_revision)
            torch = import_module("torch")
            enforce_memory_budget("text retrieval model load", torch)
            corpus_embeddings = _encode_corpus_embeddings(self._embedder, chunks, settings, cache_dir)
            enforce_memory_budget("text retrieval corpus encode", torch)
        else:
            raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")
        self._dense_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self._dense_index.add(corpus_embeddings)
        if not self._local_hash:
            del corpus_embeddings

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = self.settings.top_k if top_k is None else top_k
        if k <= 0:
            return []
        k = min(k, len(self.chunks))
        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        if self._local_hash:
            query_embedding = self._embedder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=self.settings.embedding_batch_size,
            ).astype("float32")
        else:
            query_embedding = self._embedder.encode(
                [query],
                batch_size=self.settings.embed_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
        dense_scores, dense_indices = self._dense_index.search(query_embedding, k=k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        dense_lookup = np.zeros(len(self.chunks), dtype=np.float32)
        for idx, score in zip(dense_indices, dense_scores):
            if idx >= 0:
                dense_lookup[int(idx)] = float(score)

        norm_bm25 = _normalize_scores(bm25_scores)
        norm_dense = _normalize_scores(dense_lookup)
        fused = 0.45 * norm_dense + 0.35 * norm_bm25 + 0.20 * _rrf_component(
            bm25_scores, dense_lookup, stable=self._local_hash
        )

        if self._local_hash:
            candidate_indices = np.argsort(-fused, kind="stable")[:k]
            hits: list[RetrievalHit] = []
            for idx in candidate_indices:
                hits.append(
                    RetrievalHit(
                        chunk=self.chunks[int(idx)],
                        bm25_score=float(norm_bm25[idx]),
                        dense_score=float(norm_dense[idx]),
                        fused_score=float(fused[idx]),
                    )
                )
            return hits

        candidate_indices = np.argsort(-fused, kind="stable")[:k]
        candidate_pairs = [(query, self.chunks[int(idx)].text) for idx in candidate_indices]
        reranker_logits = self._reranker.predict(candidate_pairs, show_progress_bar=False)
        reranker_scores = [_to_probability(float(score)) for score in reranker_logits]
        ranked_candidates = sorted(
            zip(candidate_indices, reranker_scores), key=lambda pair: pair[1], reverse=True
        )
        hits = []
        for idx, reranker_score in ranked_candidates:
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[int(idx)],
                    bm25_score=float(norm_bm25[idx]),
                    dense_score=float(norm_dense[idx]),
                    fused_score=float(fused[idx]),
                    reranker_score=round(reranker_score, 6),
                )
            )
        return hits


def _rrf_component(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    k: int = 60,
    *,
    stable: bool = False,
) -> np.ndarray:
    kind = "stable" if stable else None
    if kind is None:
        bm25_rank = np.argsort(np.argsort(-bm25_scores))
        dense_rank = np.argsort(np.argsort(-dense_scores))
    else:
        bm25_rank = np.argsort(np.argsort(-bm25_scores, kind="stable"), kind="stable")
        dense_rank = np.argsort(np.argsort(-dense_scores, kind="stable"), kind="stable")
    return (1.0 / (k + bm25_rank + 1)) + (1.0 / (k + dense_rank + 1))
