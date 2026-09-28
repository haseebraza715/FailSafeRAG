from __future__ import annotations

import re

from .settings import RetrievalSettings
from .text_units import CHUNK_POLICIES, LEGACY_CHUNK_POLICY, MULTILINGUAL_CHUNK_POLICY, chunk_spans
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
    chunk_policy: str = LEGACY_CHUNK_POLICY,
) -> list[Chunk]:
    """Split one page into overlapping chunks.

    ``chunk_policy`` names the boundary rule (see ``faar.text_units``). The default,
    ``whitespace-words-v1``, is the original rule and the only one that ``faar.graph``
    and ``faar.benchmarks`` use. ``cjk-weighted-words-v1`` counts each CJK character
    as half a word, so an unspaced Chinese page still splits into bounded chunks.
    Text without a CJK character gets the same chunks under both rules.
    """
    if chunk_policy not in CHUNK_POLICIES:
        raise ValueError(f"unknown chunk policy {chunk_policy!r}; expected one of {', '.join(CHUNK_POLICIES)}")
    if chunk_policy == MULTILINGUAL_CHUNK_POLICY:
        spans = chunk_spans(page_text, settings.chunk_size_words, settings.chunk_overlap_words)
        return [
            Chunk(
                chunk_id=f"{example_id}-p{page_id}-c{chunk_index}",
                example_id=example_id,
                doc_name=doc_name,
                page_id=page_id,
                text=" ".join(page_text[start:end].split()),
                image_path=image_path,
            )
            for chunk_index, (start, end) in enumerate(spans)
        ]
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
