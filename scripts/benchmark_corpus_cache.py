#!/usr/bin/env python3
"""Benchmark the corpus-embedding cache mechanism.

Measures wall time for the text corpus encode path with and without the
persistent cache, using a deterministic embedder whose per-batch cost is
simulated (the machine running this benchmark cannot load NV-Embed-v2 into
16 GiB of RAM; the mechanism under test is identical: a cache hit skips the
entire encode loop).

Verifies bitwise-identical embeddings between a fresh encode and a cache hit.

Usage: .venv-aaai/bin/python scripts/benchmark_corpus_cache.py [--chunks N] [--batch 4] [--seconds-per-batch 0.1]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from faar.retrieval import _encode_corpus_embeddings
from faar.settings import RetrievalSettings
from faar.types import Chunk


class CostedEmbedder:
    def __init__(self, seconds_per_batch: float) -> None:
        self.seconds_per_batch = seconds_per_batch
        self.calls = 0
        self.batches = 0

    def encode(self, texts, batch_size=None, normalize_embeddings=False, convert_to_numpy=False):
        self.calls += 1
        time.sleep(self.seconds_per_batch)
        rows = np.array(
            [[float(ord(char) % 7) / 7.0 for char in text[:16]] for text in texts], dtype=np.float32
        )
        return rows if rows.size else np.zeros((0, 16), dtype=np.float32)


def chunks(count: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc-p{i // 2}-c{i % 2}",
            example_id="doc",
            doc_name="doc",
            page_id=i // 2,
            text=f"shared corpus text chunk number {i} with an answer forty two and more words",
        )
        for i in range(count)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=400)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seconds-per-batch", type=float, default=0.1)
    args = parser.parse_args()

    data = chunks(args.chunks)
    settings = RetrievalSettings()
    settings.embed_batch_size = args.batch

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        embedder = CostedEmbedder(args.seconds_per_batch)

        start = time.monotonic()
        first = _encode_corpus_embeddings(embedder, data, settings, cache_dir)
        miss_seconds = time.monotonic() - start

        start = time.monotonic()
        second = _encode_corpus_embeddings(embedder, data, settings, cache_dir)
        hit_seconds = time.monotonic() - start

        files = list(cache_dir.glob("*.npz"))
        cache_size = sum(path.stat().st_size for path in files)

    identical = np.array_equal(first, second)
    print(f"chunks={args.chunks} batch={args.batch} simulated_encode_per_batch={args.seconds_per_batch}s")
    print(f"cache miss (full encode): {miss_seconds:.3f}s embedder_calls={embedder.calls}")
    print(f"cache hit (load only):    {hit_seconds:.6f}s embedder_calls={embedder.calls} after hit")
    print(f"embeddings bitwise identical: {identical}")
    print(f"cache files written: {len(files)} total_bytes={cache_size} ({files[0].name if files else 'none'})")
    if not identical:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
