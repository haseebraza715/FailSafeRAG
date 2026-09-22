from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from faar.results_aggregator import summarize_api_usage
from faar.run_io import safe_checkpoint_stem
from faar.settings import AppSettings
from faar.visual_baselines import run_visual_baseline


class FakeRepo:
    dataset = "ohrbench"
    split = "test"
    manifest_sha256 = "f" * 64

    def __init__(self, example_ids: list[str] | None = None) -> None:
        self._ids = list(example_ids) if example_ids is not None else ["q1", "q2", "q3"]

    def list_example_ids(self) -> list[str]:
        return list(self._ids)

    def get_example(self, example_id: str):
        return SimpleNamespace(
            example_id=example_id,
            doc_name="doc",
            page_ids=[],
            question=f"Question {example_id}?",
            correct_answer=f"answer-{example_id}",
        )

    def corpus_image_page_map(self) -> dict[Path, tuple[str, int]]:
        return {}


class FakeRetriever:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        self.calls.append("retrieve")
        return []


class FakeFallback:
    def __init__(self, calls: list[str], settings: AppSettings) -> None:
        self.calls = calls
        self.settings = settings

    def answer(self, question: str, image_paths: list[Path], fallback_context: str):
        self.calls.append("answer")
        example_id = question.split()[1].rstrip("?")
        return {
            "status": "succeeded",
            "answer": f"answer-{example_id}",
            "request_model": "gpt-4o-2024-11-20",
            "response_model": "gpt-4o-2024-11-20",
            "completed_at_utc": "2026-07-23T12:00:00+00:00",
            "cost_rates": {
                "provider": "openai",
                "currency": "USD",
                "input_usd_per_million_tokens": 2.5,
                "output_usd_per_million_tokens": 10.0,
            },
            "api_usage": {
                "api_requests": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost_usd": 0.00045,
            },
        }


def _patch_visual(
    monkeypatch: pytest.MonkeyPatch,
    retriever_calls: list[str],
    fallback_calls: list[str],
    built: list[str],
) -> None:
    def make_retriever(*args: Any, **kwargs: Any) -> FakeRetriever:
        built.append("retriever")
        return FakeRetriever(retriever_calls)

    def make_fallback(settings: AppSettings) -> FakeFallback:
        built.append("fallback")
        return FakeFallback(fallback_calls, settings)

    monkeypatch.setattr("faar.visual_baselines.build_visual_retriever", make_retriever)
    monkeypatch.setattr("faar.visual_baselines.VisualFallback", make_fallback)


def _checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs/phase3/colpali"


def test_visual_resume_reuses_matching_checkpoints_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo()

    first = run_visual_baseline(settings, repo, "colpali")
    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"] * 3
    assert fallback_calls == ["answer"] * 3

    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    second = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == []
    assert retriever_calls == []
    assert fallback_calls == []
    assert [row["example_id"] for row in second] == ["q1", "q2", "q3"]
    assert second == first
    assert all(row["run_metadata"]["run_fingerprint"] for row in second)


def test_visual_resume_skips_without_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo()

    run_visual_baseline(settings, repo, "colpali")

    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    rows = run_visual_baseline(settings, repo, "colpali")

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"] * 3
    assert fallback_calls == ["answer"] * 3
    assert len(rows) == 3


def test_visual_resume_recomputes_when_settings_fingerprint_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo()
    run_visual_baseline(settings, repo, "colpali")

    settings.retrieval.embedding_model = "other/embedder"
    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"] * 3
    assert fallback_calls == ["answer"] * 3
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3"]
    assert all(row["run_metadata"]["run_fingerprint"] for row in rows)


def test_visual_resume_recomputes_when_manifest_hash_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo()
    run_visual_baseline(settings, repo, "colpali")

    repo.manifest_sha256 = "g" * 64
    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"] * 3
    assert fallback_calls == ["answer"] * 3
    assert len(rows) == 3


def test_visual_resume_recomputes_corrupt_checkpoint_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo()
    run_visual_baseline(settings, repo, "colpali")

    (_checkpoint_dir(tmp_path) / "q2.json").write_text("{not-json", encoding="utf-8")
    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"]
    assert fallback_calls == ["answer"]
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3"]
    assert rows[1]["predicted_answer"] == "answer-q2"


def test_visual_resume_recomputes_structurally_invalid_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo()
    run_visual_baseline(settings, repo, "colpali")

    checkpoint = _checkpoint_dir(tmp_path) / "q2.json"
    row = json.loads(checkpoint.read_text(encoding="utf-8"))
    row["metrics"]["f1"] = "not-a-number"
    checkpoint.write_text(json.dumps(row), encoding="utf-8")
    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()

    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"]
    assert fallback_calls == ["answer"]
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3"]


def test_visual_resume_preserves_partial_cache_with_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1", "q2", "q3", "q4"])
    run_visual_baseline(settings, repo, "colpali", max_examples=2)

    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    rows = run_visual_baseline(settings, repo, "colpali", max_examples=4, resume=True)

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"] * 2
    assert fallback_calls == ["answer"] * 2
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3", "q4"]


def test_visual_row_contains_run_metadata_and_keeps_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1"])
    rows = run_visual_baseline(settings, repo, "colpali")

    row = rows[0]
    assert row["request_model"] == "gpt-4o-2024-11-20"
    assert row["response_model"] == "gpt-4o-2024-11-20"
    assert row["completed_at_utc"] == "2026-07-23T12:00:00+00:00"
    assert row["action_outcome"]["action"] == "invoke_vlm"
    assert row["metrics"]["em"] == 1.0
    assert row["metrics"]["f1"] == 1.0
    assert "visual_hits" in row
    assert "source_assets" in row
    metadata = row["run_metadata"]
    assert metadata["profile"] == "colpali"
    assert metadata["dataset"] == "ohrbench"
    assert metadata["split"] == "test"
    assert metadata["vlm_backend"] == settings.recovery.vlm_backend
    assert metadata["run_fingerprint"]
    assert metadata["cost_rates"]["input_usd_per_million_tokens"] == 2.5

    saved = json.loads((_checkpoint_dir(tmp_path) / "q1.json").read_text(encoding="utf-8"))
    assert saved["run_metadata"]["run_fingerprint"] == metadata["run_fingerprint"]


def test_visual_checkpoint_uses_safe_stem_and_reuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["folder/q1"])

    rows = run_visual_baseline(settings, repo, "colpali")
    assert rows[0]["example_id"] == "folder/q1"
    checkpoint = _checkpoint_dir(tmp_path) / f"{safe_checkpoint_stem('folder/q1')}.json"
    assert checkpoint.is_file()
    assert "/" not in checkpoint.stem
    assert "\\" not in checkpoint.stem

    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    resumed = run_visual_baseline(settings, repo, "colpali", resume=True)
    assert built == []
    assert [row["example_id"] for row in resumed] == ["folder/q1"]


def test_visual_row_stores_api_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1"])

    rows = run_visual_baseline(settings, repo, "colpali")

    assert rows[0]["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": 0.00045,
    }


def test_visual_resume_recomputes_checkpoint_with_malformed_api_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1", "q2", "q3"])
    run_visual_baseline(settings, repo, "colpali")

    checkpoint = _checkpoint_dir(tmp_path) / "q2.json"
    row = json.loads(checkpoint.read_text(encoding="utf-8"))
    row["api_usage"]["api_requests"] = -1
    checkpoint.write_text(json.dumps(row), encoding="utf-8")
    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()

    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == ["retriever", "fallback"]
    assert retriever_calls == ["retrieve"]
    assert fallback_calls == ["answer"]
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3"]


def test_visual_resume_recomputes_checkpoint_without_api_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1", "q2", "q3"])
    run_visual_baseline(settings, repo, "colpali")

    checkpoint = _checkpoint_dir(tmp_path) / "q2.json"
    row = json.loads(checkpoint.read_text(encoding="utf-8"))
    del row["api_usage"]
    checkpoint.write_text(json.dumps(row), encoding="utf-8")
    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()

    rows = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == ["retriever", "fallback"]
    assert fallback_calls == ["answer"]
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3"]
    assert rows[1]["api_usage"] == {
        "api_requests": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": 0.00045,
    }


def test_visual_partial_resume_totals_cached_plus_new_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retriever_calls: list[str] = []
    fallback_calls: list[str] = []
    built: list[str] = []
    _patch_visual(monkeypatch, retriever_calls, fallback_calls, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(["q1", "q2", "q3", "q4"])
    run_visual_baseline(settings, repo, "colpali", max_examples=2)

    built.clear()
    retriever_calls.clear()
    fallback_calls.clear()
    rows = run_visual_baseline(settings, repo, "colpali", max_examples=4, resume=True)

    assert fallback_calls == ["answer"] * 2
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q3", "q4"]
    totals = summarize_api_usage(rows)
    assert totals == {
        "api_requests": 4,
        "prompt_tokens": 400,
        "completion_tokens": 80,
        "cost_usd": 0.0018,
    }
