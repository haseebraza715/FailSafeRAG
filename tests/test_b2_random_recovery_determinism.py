"""Focused tests for deterministic B2 random-recovery selection.

B2 (profile ``faar_no_diagnosis``) picks a recovery type per example. The
choice must be a pure function of the locked run seed plus example_id so that
it is invariant to resume, iteration order, and sharding, and it must not
depend on Python's process-randomized ``hash()`` or global RNG state. The
chosen type is persisted in each row and validated again on resume.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

import faar.experiment_runner as runner_mod
from faar.api_logging import zero_api_usage
from faar.experiment_profiles import apply_profile
from faar.experiment_runner import run_profile
from faar.graph import (
    POLICY_ACTION_BY_TYPE,
    RANDOM_RECOVERY_TYPES,
    build_graph,
    random_recovery_type,
)
from faar.settings import AppSettings
from faar.types import Chunk, RetrievalHit

IDS = ["ex1", "ex2", "ex3", "ex4"]


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

    def answer(self, _question: str, _image_paths: list[Path], _context: str) -> dict[str, Any]:
        return {
            "backend": "test",
            "status": "succeeded",
            "answer": "recovered",
            "used_images": [],
            "api_usage": zero_api_usage(),
        }


def _prepare_phase0(tmp_path: Path, count: int) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True, exist_ok=True)
    ids = [f"ex{i}" for i in range(1, count + 1)]
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        + "".join(f"{e},manual/doc,What is required?,certified engineers,0\n" for e in ids)
    )
    for e in ids:
        (tmp_path / "artifacts/phase0/ocr_text" / f"{e}.txt").write_text("===== PAGE 0 =====\nno matching answer")


def _patch_graph_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faar.graph.HybridRetriever", _FakeRetriever)
    monkeypatch.setattr("faar.graph.ByT5Corrector", _FakeCorrector)
    monkeypatch.setattr("faar.graph.VisualFallback", _FakeVisualFallback)
    monkeypatch.setattr("faar.graph.semantic_backtrack", lambda question, _hits: question)


def _spy_build_graph(monkeypatch: pytest.MonkeyPatch, invoked: list[str]) -> None:
    real = runner_mod.build_graph

    def spy(settings: AppSettings, **kwargs: Any) -> Any:
        graph = real(settings, **kwargs)
        original_invoke = graph.invoke

        def invoke(state: dict[str, Any]) -> dict[str, Any]:
            invoked.append(state["example_id"])
            return original_invoke(state)

        graph.invoke = invoke
        return graph

    monkeypatch.setattr(runner_mod, "build_graph", spy)


def _settings(tmp_path: Path, seed: int = 42) -> AppSettings:
    settings = AppSettings(project_root=tmp_path)
    settings.experiment.random_seed = seed
    return settings


def _signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "predicted_answer": row.get("predicted_answer"),
        "failure_type": row.get("failure_type"),
        "policy_action": row.get("policy_action"),
        "recovery_type": row.get("recovery_type"),
        "action_outcome": row.get("action_outcome"),
        "metrics": row.get("metrics"),
    }


def test_random_recovery_type_is_deterministic_and_seed_plus_example_sensitive() -> None:
    allowed = set(RANDOM_RECOVERY_TYPES)

    assert random_recovery_type(42, "ex1") == random_recovery_type(42, "ex1")
    assert random_recovery_type(42, "ex1") in allowed

    examples = [f"ex{i}" for i in range(1, 30)]
    assert len({random_recovery_type(42, eid) for eid in examples}) > 1
    assert any(
        random_recovery_type(1, eid) != random_recovery_type(2, eid) for eid in examples
    )


def test_random_recovery_type_ignores_global_random_state() -> None:
    expected = random_recovery_type(42, "ex1")
    random.seed(0)
    assert random_recovery_type(42, "ex1") == expected
    random.seed(987654321)
    assert random_recovery_type(42, "ex1") == expected


def test_graph_random_recovery_is_order_and_rng_state_invariant(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 3)
    _patch_graph_dependencies(monkeypatch)
    settings = apply_profile(_settings(tmp_path, seed=42), "faar_no_diagnosis")
    graph = build_graph(settings)

    random.seed(7)
    forward = {r["example_id"]: r for r in (graph.invoke({"example_id": e}) for e in IDS[:3])}
    random.seed(31337)
    backward = {r["example_id"]: r for r in (graph.invoke({"example_id": e}) for e in reversed(IDS[:3]))}

    for eid in IDS[:3]:
        assert forward[eid]["failure_type"] == "random"
        assert forward[eid]["recovery_type"] == backward[eid]["recovery_type"]
        assert forward[eid]["policy_action"] == backward[eid]["policy_action"]
        assert forward[eid]["recovery_type"] in RANDOM_RECOVERY_TYPES
        assert forward[eid]["policy_action"] == POLICY_ACTION_BY_TYPE[forward[eid]["recovery_type"]]


def test_clean_run_matches_partial_resume(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 4)
    _patch_graph_dependencies(monkeypatch)
    invoked: list[str] = []
    _spy_build_graph(monkeypatch, invoked)
    settings = _settings(tmp_path, seed=42)

    clean = run_profile(settings, profile_name="faar_no_diagnosis", example_ids=IDS)
    assert invoked == IDS

    rows_dir = tmp_path / "logs/phase3/faar_no_diagnosis"
    (rows_dir / "ex2.json").unlink()
    (rows_dir / "ex3.json").unlink()
    invoked.clear()

    resumed = run_profile(settings, profile_name="faar_no_diagnosis", example_ids=IDS, resume=True)

    assert invoked == ["ex2", "ex3"]
    assert [_signature(row) for row in resumed] == [_signature(row) for row in clean]


def test_resume_reuses_matching_recovery_type_without_recompute(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    _patch_graph_dependencies(monkeypatch)
    invoked: list[str] = []
    _spy_build_graph(monkeypatch, invoked)
    settings = _settings(tmp_path, seed=42)

    run_profile(settings, profile_name="faar_no_diagnosis", example_ids=["ex1", "ex2"])
    invoked.clear()

    rows = run_profile(settings, profile_name="faar_no_diagnosis", example_ids=["ex1", "ex2"], resume=True)

    assert invoked == []
    for row in rows:
        assert row["recovery_type"] == random_recovery_type(settings.experiment.random_seed, row["example_id"])


def test_resume_recomputes_stale_random_recovery_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    _patch_graph_dependencies(monkeypatch)
    invoked: list[str] = []
    _spy_build_graph(monkeypatch, invoked)
    settings = _settings(tmp_path, seed=42)

    run_profile(settings, profile_name="faar_no_diagnosis", example_ids=["ex1", "ex2"])

    checkpoint = tmp_path / "logs/phase3/faar_no_diagnosis/ex1.json"
    row = json.loads(checkpoint.read_text())
    assert row["failure_type"] == "random"
    original = row["recovery_type"]
    row["recovery_type"] = next(t for t in RANDOM_RECOVERY_TYPES if t != original)
    checkpoint.write_text(json.dumps(row))
    invoked.clear()

    rows = run_profile(settings, profile_name="faar_no_diagnosis", example_ids=["ex1", "ex2"], resume=True)

    assert invoked == ["ex1"]
    by_id = {row["example_id"]: row for row in rows}
    assert by_id["ex1"]["recovery_type"] == random_recovery_type(42, "ex1")
    assert by_id["ex2"]["recovery_type"] == random_recovery_type(42, "ex2")


def test_resume_recomputes_checkpoint_without_recovery_type(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 2)
    _patch_graph_dependencies(monkeypatch)
    invoked: list[str] = []
    _spy_build_graph(monkeypatch, invoked)
    settings = _settings(tmp_path, seed=42)

    run_profile(settings, profile_name="faar_no_diagnosis", example_ids=["ex1", "ex2"])

    checkpoint = tmp_path / "logs/phase3/faar_no_diagnosis/ex1.json"
    row = json.loads(checkpoint.read_text())
    del row["recovery_type"]
    checkpoint.write_text(json.dumps(row))
    invoked.clear()

    rows = run_profile(settings, profile_name="faar_no_diagnosis", example_ids=["ex1", "ex2"], resume=True)

    assert invoked == ["ex1"]
    assert rows[0]["recovery_type"] == random_recovery_type(42, "ex1")


def test_sharded_run_matches_unsharded_per_example_choice(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 4)
    _patch_graph_dependencies(monkeypatch)

    full = run_profile(
        _settings(tmp_path, seed=42),
        profile_name="faar_no_diagnosis",
        example_ids=IDS,
        output_dir=tmp_path / "full",
    )
    shard_settings = _settings(tmp_path, seed=42)
    shard0 = run_profile(
        shard_settings,
        profile_name="faar_no_diagnosis",
        example_ids=IDS,
        output_dir=tmp_path / "shards",
        shard_index=0,
        num_shards=2,
    )
    shard1 = run_profile(
        shard_settings,
        profile_name="faar_no_diagnosis",
        example_ids=IDS,
        output_dir=tmp_path / "shards",
        shard_index=1,
        num_shards=2,
    )

    full_by_id = {row["example_id"]: row for row in full}
    sharded_by_id = {row["example_id"]: row for row in shard0 + shard1}

    assert set(full_by_id) == set(IDS)
    assert set(sharded_by_id) == set(IDS)
    assert set(row["example_id"] for row in shard0).isdisjoint(row["example_id"] for row in shard1)
    for eid in IDS:
        assert sharded_by_id[eid]["recovery_type"] == full_by_id[eid]["recovery_type"]
        assert sharded_by_id[eid]["policy_action"] == full_by_id[eid]["policy_action"]


def test_non_random_profile_rows_do_not_emit_recovery_type(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, 1)
    _patch_graph_dependencies(monkeypatch)

    rows = run_profile(
        _settings(tmp_path, seed=42),
        profile_name="faar_full",
        example_ids=["ex1"],
    )

    assert len(rows) == 1
    assert "recovery_type" not in rows[0]
    assert rows[0]["failure_type"] in set(RANDOM_RECOVERY_TYPES)
    assert rows[0]["policy_action"] == POLICY_ACTION_BY_TYPE[rows[0]["failure_type"]]
