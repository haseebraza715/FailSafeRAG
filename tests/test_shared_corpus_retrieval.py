from __future__ import annotations

from pathlib import Path

from faar.graph import build_graph
from faar.settings import AppSettings
from faar.types import Chunk, Phase0Example, RetrievalHit


class SharedCorpusRepository:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.chunks = [
            Chunk(
                chunk_id="doc-p1-c0",
                example_id="doc:p1",
                doc_name="doc",
                page_id=1,
                text="the shared corpus contains answer forty two",
                image_path=str(tmp_path / "doc_p1.png"),
            )
        ]

    def get_corpus_chunks(self, settings):
        return self.chunks

    def get_example(self, example_id: str) -> Phase0Example:
        return Phase0Example(
            example_id=example_id,
            doc_name="",
            question=f"question for {example_id}",
            correct_answer="forty two",
            page_ids=[],
            ocr_text="",
            ocr_text_path=self.tmp_path / ".query-only",
            gt_text_path=None,
            image_paths=[],
            metadata={},
        )


class CountingRetriever:
    constructions = 0

    def __init__(self, chunks, settings, *, cache_dir: Path | None = None) -> None:
        type(self).constructions += 1
        self.hit = RetrievalHit(
            chunk=chunks[0],
            bm25_score=1.0,
            dense_score=1.0,
            fused_score=1.0,
            reranker_score=1.0,
        )

    def retrieve(self, query: str, top_k: int | None = None):
        return [self.hit]


def test_dataset_run_builds_one_shared_retriever(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "doc_p1.png").write_bytes(b"image")
    CountingRetriever.constructions = 0
    monkeypatch.setattr("faar.graph.HybridRetriever", CountingRetriever)
    settings = AppSettings(project_root=tmp_path)
    graph = build_graph(settings, repo=SharedCorpusRepository(tmp_path))

    first = graph.invoke({"example_id": "q1"})
    second = graph.invoke({"example_id": "q2"})

    assert CountingRetriever.constructions == 1
    assert first["retriever"] is second["retriever"]
    assert first["retrieved_hits"][0].chunk.example_id == "doc:p1"
    assert second["retrieved_hits"][0].chunk.example_id == "doc:p1"
