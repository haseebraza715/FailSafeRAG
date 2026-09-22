from __future__ import annotations

import re

from .settings import RetrievalSettings
from .types import Chunk, Phase0Example


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def build_chunks(example: Phase0Example, settings: RetrievalSettings) -> list[Chunk]:
    chunks: list[Chunk] = []
    page_texts = example.metadata.get("page_texts") or {page_id: example.ocr_text for page_id in example.page_ids or [0]}
    for page_id, page_text in page_texts.items():
        image_by_page = example.metadata.get("image_by_page") or {}
        chunks.extend(
            build_page_chunks(
                example_id=example.example_id,
                doc_name=example.doc_name,
                page_id=int(page_id),
                page_text=page_text,
                settings=settings,
                image_path=image_by_page.get(str(page_id)) or image_by_page.get(int(page_id)),
            )
        )
    return chunks


def build_page_chunks(
    *,
    example_id: str,
    doc_name: str,
    page_id: int,
    page_text: str,
    settings: RetrievalSettings,
    image_path: str | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    words = _tokenize_words(page_text)
    if not words:
        return chunks
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + settings.chunk_size_words, len(words))
        text = " ".join(words[start:end]).strip()
        chunk_id = f"{example_id}-p{page_id}-c{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                example_id=example_id,
                doc_name=doc_name,
                page_id=page_id,
                text=text,
                image_path=image_path,
            )
        )
        if end >= len(words):
            break
        start += max(settings.chunk_size_words - settings.chunk_overlap_words, 1)
        chunk_index += 1
    return chunks
