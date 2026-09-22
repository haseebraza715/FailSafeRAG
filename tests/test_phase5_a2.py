"""Contract tests for Phase 5 A2: diagnosis-free random recovery."""

from __future__ import annotations

import random
from pathlib import Path

from faar.experiment_profiles import apply_profile
from faar.graph import build_graph
from faar.settings import AppSettings
from faar.types import Chunk, RetrievalHit


class _FakeRetriever:
    def __init__(self, chunks: list[Chunk], _settings: object, *, cache_dir=None) -> None:
        self._hits = [
            RetrievalHit(
                chunk=chunk,
                bm25_score=0.0,
                dense_score=0.0,
                fused_score=0.0,
                reranker_score=0.0,
            )
            for chunk in chunks
        ]

    def retrieve(self, _query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return self._hits[:top_k]


class _FakeCorrector:
    def __init__(self, _model_name: str, _revision: str | None = None, _correction=None) -> None:
        pass

    def propose_correction(self, text: str) -> dict[str, str | bool]:
        return {"text": text, "applied": False, "reason": "test_stub"}


class _FakeVisualFallback:
    def __init__(self, _settings: AppSettings) -> None:
        pass

    def answer(self, _question: str, _image_paths: list[Path], _context: str) -> dict[str, object]:
        return {"backend": "test", "status": "succeeded", "answer": "recovered", "used_images": []}


def _prepare_examples(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
        "ex2,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text("===== PAGE 0 =====\nno matching answer")
    (tmp_path / "artifacts/phase0/ocr_text/ex2.txt").write_text("===== PAGE 0 =====\nno matching answer")


def _run_a2(monkeypatch, tmp_path: Path, seed: int) -> list[tuple[str, str, str, str]]:
    _prepare_examples(tmp_path)
    monkeypatch.setattr("faar.graph.HybridRetriever", _FakeRetriever)
    monkeypatch.setattr("faar.graph.ByT5Corrector", _FakeCorrector)
    monkeypatch.setattr("faar.graph.VisualFallback", _FakeVisualFallback)
    monkeypatch.setattr("faar.graph.semantic_backtrack", lambda question, _hits: question)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("A2 must bypass diagnose_failure")

    monkeypatch.setattr("faar.graph.diagnose_failure", fail_if_called)

    settings = apply_profile(AppSettings(project_root=tmp_path), "faar_no_diagnosis")
    settings.gate.quality_threshold = 0.5
    random.seed(seed)
    graph = build_graph(settings)
    results = [graph.invoke({"example_id": example_id}) for example_id in ("ex1", "ex2")]
    return [
        (
            result["example_id"],
            result.get("failure_type", ""),
            result.get("policy_action", ""),
            result.get("action_outcome", {}).get("action", ""),
        )
        for result in results
    ]


def test_a2_profile_bypasses_diagnosis_without_disabling_recovery() -> None:
    settings = apply_profile(AppSettings(), "faar_no_diagnosis")

    assert settings.experiment.disable_diagnosis is True
    assert settings.experiment.force_direct_answer is False
    assert settings.experiment.disable_backtracking is False
    assert settings.experiment.disable_vlm is False


def test_a2_no_diagnosis_uses_seeded_random_recovery_for_failed_examples(monkeypatch, tmp_path: Path) -> None:
    """A2 keeps the gate but replaces typed diagnosis with seeded random recovery."""
    first = _run_a2(monkeypatch, tmp_path, seed=42)
    second = _run_a2(monkeypatch, tmp_path, seed=42)

    assert first == second
    assert [example_id for example_id, _, _, _ in first] == ["ex1", "ex2"]
    assert all(failure_type == "random" for _, failure_type, _, _ in first)
    assert {action for _, _, action, _ in first} <= {"correct_text", "retry_retrieval", "invoke_vlm"}
    assert all(action == executed_action for _, _, action, executed_action in first)
