from __future__ import annotations

import re

from .settings import RetrievalSettings
from .types import Chunk, Phase0Example


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def build_chunks(example: Phase0Example, settings: RetrievalSettings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_id, page_text in example.metadata["page_texts"].items():
        words = _tokenize_words(page_text)
        if not words:
            continue
        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + settings.chunk_size_words, len(words))
            text = " ".join(words[start:end]).strip()
            chunk_id = f"{example.example_id}-p{page_id}-c{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    example_id=example.example_id,
                    doc_name=example.doc_name,
                    page_id=int(page_id),
                    text=text,
                )
            )
            if end >= len(words):
                break
            start += max(settings.chunk_size_words - settings.chunk_overlap_words, 1)
            chunk_index += 1
    return chunks
