from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from faar.experiment_profiles import apply_profile
from faar.graph import build_graph
from faar.settings import AppSettings
from faar.types import Chunk, Phase0Example, RetrievalHit


class _SingleExampleRepository:
    def get_example(self, example_id: str) -> Phase0Example:
        assert example_id == "ex1"
        return Phase0Example(
            example_id="ex1",
            doc_name="manual",
            question="What is required?",
            correct_answer="certified engineers",
            page_ids=[0],
            ocr_text="Certified engineers are required.",
            ocr_text_path=Path("/tmp/ex1.txt"),
            gt_text_path=None,
            image_paths=[],
            metadata={"page_texts": {0: "Certified engineers are required."}},
        )


class _PassingRetriever:
    def __init__(self, chunks, settings, *, cache_dir=None) -> None:
        self.hits = [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id="c1",
                    example_id="ex1",
                    doc_name="manual",
                    page_id=0,
                    text="Certified engineers are required.",
                ),
                bm25_score=1.0,
                dense_score=1.0,
                fused_score=1.0,
                reranker_score=0.99,
            )
        ]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return self.hits


def test_a1_cli_uses_the_no_gate_profile(monkeypatch, tmp_path: Path) -> None:
    """A1 must not silently run the ordinary FAAR profile."""
    runner = importlib.import_module("run")
    settings = AppSettings(project_root=tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "_settings_from_args", lambda args: settings)
    monkeypatch.setattr(AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(runner, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(runner, "_validate_baseline", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_apply_baseline_harm", lambda payload, *args, **kwargs: payload)

    def fake_run_profile_to_result(settings, profile, out, label, max_examples, run_spec, dataset, split, **kwargs):
        captured["profile"] = profile
        captured["label"] = label
        payload = {"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(runner, "_run_profile_to_result", fake_run_profile_to_result)
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(json.dumps({"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}, "rows": []}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/experiments/run.py", "--ablate", "no_gate", "--baseline", str(baseline_path), "--out", str(tmp_path / "a1.json")],
    )

    runner.main()

    assert captured == {"profile": "faar_no_gate", "label": "no_gate"}


def test_a1_bypasses_a_passing_gate_and_always_uses_recovery(monkeypatch, tmp_path: Path) -> None:
    """Even a high-confidence retrieval must not take A1's direct-answer path."""
    monkeypatch.setattr("faar.graph.HybridRetriever", _PassingRetriever)
    settings = AppSettings(project_root=tmp_path)
    settings.gate.quality_threshold = 0.5
    settings.gate.structural_threshold = 99
    apply_profile(settings, "faar_no_gate")

    result = build_graph(settings, repo=_SingleExampleRepository()).invoke({"example_id": "ex1"})

    assert result["gate"]["pass_gate"] is True
    assert result["policy_action"] != "answer_direct"
    assert result["action_outcome"]["action"] in {"correct_text", "retry_retrieval", "invoke_vlm"}
    assert result["action_outcome"]["action"] == "retry_retrieval"
