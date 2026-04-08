from pathlib import Path

from faar.graph import build_graph
from faar.settings import AppSettings
from faar.types import Chunk, RetrievalHit


class FakeRetriever:
    def __init__(self, chunks, settings) -> None:
        self.hits = [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id="fake-c1",
                    example_id="ex1",
                    doc_name="doc",
                    page_id=0,
                    text="installation requires certified engineers",
                ),
                bm25_score=0.0,
                dense_score=1.0,
                fused_score=0.5,
            )
        ]

    def retrieve(self, query: str, top_k: int | None = None):
        return self.hits


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text("===== PAGE 0 =====\ninstallation details")


def test_graph_routes_to_semantic_retry(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.graph.HybridRetriever", FakeRetriever)
    settings = AppSettings(project_root=tmp_path)
    settings.gate.quality_threshold = 0.95
    settings.gate.lexical_floor = 0.5
    settings.gate.structural_threshold = 99
    graph = build_graph(settings)
    result = graph.invoke({"example_id": "ex1"})
    assert result["failure_type"] == "semantic"
    assert result["policy_action"] == "retry_retrieval"
    assert result["action_outcome"]["action"] == "retry_retrieval"
    assert result["answer"]
