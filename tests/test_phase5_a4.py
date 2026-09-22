import json
import sys
from pathlib import Path

import run

from faar.experiment_profiles import apply_profile
from faar.graph import build_graph
from faar.settings import AppSettings
from faar.types import Chunk, RetrievalHit


class CountingRetriever:
    instances: list["CountingRetriever"] = []

    def __init__(self, chunks, settings, *, cache_dir=None) -> None:
        self.queries: list[str] = []
        self.hits = [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id="semantic-hit",
                    example_id="ex1",
                    doc_name="manual",
                    page_id=0,
                    text="installation requires certified engineers",
                ),
                bm25_score=0.0,
                dense_score=0.1,
                fused_score=0.1,
                reranker_score=0.1,
            )
        ]
        type(self).instances.append(self)

    def retrieve(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        return self.hits


def _prepare_repository(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text("===== PAGE 0 =====\ninstallation details")


def test_a4_cli_selects_no_backtrack_profile(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_profile(settings, profile, out, label, max_examples, run_spec, dataset, split, **kwargs):
        captured["profile"] = profile
        payload = {"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_run_profile)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(run, "_validate_baseline", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_apply_baseline_harm", lambda payload, *args, **kwargs: payload)
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(json.dumps({"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}, "rows": []}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--ablate", "no_semantic_retry", "--baseline", str(baseline_path), "--out", str(tmp_path / "a4.json")],
    )

    run.main()

    assert captured["profile"] == "faar_no_backtrack"


def test_a4_semantic_failure_answers_without_retry_retrieval(monkeypatch, tmp_path: Path) -> None:
    _prepare_repository(tmp_path)
    CountingRetriever.instances.clear()
    monkeypatch.setattr("faar.graph.HybridRetriever", CountingRetriever)
    settings = apply_profile(AppSettings(project_root=tmp_path), "faar_no_backtrack")
    settings.gate.quality_threshold = 0.95

    result = build_graph(settings).invoke({"example_id": "ex1"})

    assert result["failure_type"] == "semantic"
    assert result["policy_action"] == "retry_retrieval"
    assert result["action_outcome"]["action"] == "answer_direct"
    assert "semantic_retry_query" not in result
    assert CountingRetriever.instances[0].queries == ["What is required?"]
