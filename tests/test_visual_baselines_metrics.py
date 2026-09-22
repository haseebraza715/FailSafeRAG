from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from faar.benchmarks import BenchmarkRepository
from faar.settings import AppSettings
from faar.visual_baselines import run_visual_baseline


class FakeFallback:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def answer(self, question: str, image_paths: list[Path], fallback_context: str):
        return {
            "status": "succeeded",
            "answer": "answer",
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


class FakeRetriever:
    def __init__(self, results: list[tuple[Path, float]]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        return self.results


class FakeRepo:
    dataset = "ohrbench"
    split = "test"
    manifest_sha256 = "f" * 64

    def __init__(self, image_map: dict[Path, tuple[str, int]], example: SimpleNamespace) -> None:
        self._image_map = image_map
        self._example = example

    def list_example_ids(self) -> list[str]:
        return [self._example.example_id]

    def get_example(self, example_id: str):
        return self._example

    def corpus_image_page_map(self) -> dict[Path, tuple[str, int]]:
        return dict(self._image_map)


def _patch_visual(
    monkeypatch: pytest.MonkeyPatch,
    retriever: FakeRetriever,
    built: list[str],
) -> None:
    def make_retriever(*args: Any, **kwargs: Any) -> FakeRetriever:
        built.append("retriever")
        return retriever

    def make_fallback(settings: AppSettings) -> FakeFallback:
        built.append("fallback")
        return FakeFallback(settings)

    monkeypatch.setattr("faar.visual_baselines.build_visual_retriever", make_retriever)
    monkeypatch.setattr("faar.visual_baselines.VisualFallback", make_fallback)


def _page_images(tmp_path: Path, count: int = 5) -> dict[Path, tuple[str, int]]:
    image_map: dict[Path, tuple[str, int]] = {}
    for page in range(1, count + 1):
        path = tmp_path / f"doc_page_{page}.png"
        path.write_bytes(b"image")
        image_map[path] = ("doc", page)
    return image_map


def _gold_example() -> SimpleNamespace:
    return SimpleNamespace(
        example_id="q1",
        doc_name="doc",
        page_ids=[2],
        question="Question q1?",
        correct_answer="answer",
    )


def test_visual_metrics_real_when_gold_retrieved_in_top5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_map = _page_images(tmp_path)
    ordered = list(image_map)
    retriever = FakeRetriever(
        [
            (ordered[0], 0.9),
            (ordered[1], 0.8),
            (ordered[2], 0.7),
            (ordered[3], 0.6),
            (ordered[4], 0.5),
        ]
    )
    built: list[str] = []
    _patch_visual(monkeypatch, retriever, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(image_map, _gold_example())

    row = run_visual_baseline(settings, repo, "colpali")[0]

    metrics = row["metrics"]
    assert metrics["recall@5"] == 1.0
    assert 0.0 < metrics["ndcg@5"] <= 1.0
    assert math.isfinite(metrics["ndcg@5"])
    assert built == ["retriever", "fallback"]


def test_visual_metrics_zero_when_gold_not_retrieved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_map = _page_images(tmp_path)
    ordered = list(image_map)
    retriever = FakeRetriever(
        [
            (ordered[2], 0.9),
            (ordered[3], 0.8),
            (ordered[4], 0.7),
        ]
    )
    built: list[str] = []
    _patch_visual(monkeypatch, retriever, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(image_map, _gold_example())

    row = run_visual_baseline(settings, repo, "colpali")[0]

    assert row["metrics"]["ndcg@5"] == 0.0
    assert row["metrics"]["recall@5"] == 0.0


def test_visual_checkpoint_validation_accepts_computed_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_map = _page_images(tmp_path)
    ordered = list(image_map)
    retriever = FakeRetriever([(ordered[1], 0.9), (ordered[0], 0.8)])
    built: list[str] = []
    _patch_visual(monkeypatch, retriever, built)
    settings = AppSettings(project_root=tmp_path)
    repo = FakeRepo(image_map, _gold_example())

    first = run_visual_baseline(settings, repo, "colpali")

    built.clear()
    resumed = run_visual_baseline(settings, repo, "colpali", resume=True)

    assert built == []
    assert resumed == first
    metrics = resumed[0]["metrics"]
    assert all(
        isinstance(metrics[name], (int, float)) and math.isfinite(float(metrics[name]))
        for name in ("ndcg@5", "recall@5", "em", "f1")
    )


def test_corpus_image_page_map_keys_match_corpus_image_paths(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    for page in range(1, 4):
        (tmp_path / "images" / f"doc_page_{page}.png").write_bytes(b"image")
    record = {
        "example_id": "e1",
        "doc_name": "doc",
        "question": "Q",
        "correct_answer": "A",
        "page_ids": [1],
        "corpus_ids": [f"doc:p{page}" for page in range(1, 4)],
    }
    corpus_pages = [
        {
            "corpus_id": f"doc:p{page}",
            "doc_name": "doc",
            "page_id": page,
            "text": f"OCR {page}",
            "image_path": f"images/doc_page_{page}.png",
        }
        for page in range(1, 4)
    ]
    repo = BenchmarkRepository(
        [record],
        corpus_pages,
        tmp_path,
        "ohrbench",
        "test",
        document_inventory={"doc": [1, 2, 3]},
    )

    page_map = repo.corpus_image_page_map()

    assert set(page_map) == set(repo.corpus_image_paths())
    assert page_map[(tmp_path / "images/doc_page_2.png").resolve()] == ("doc", 2)
