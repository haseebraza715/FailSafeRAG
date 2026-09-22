from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import run

from faar.settings import AppSettings


class FakeVisualRepository:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def corpus_image_paths(self) -> list[Path]:
        return [self.image_path]

    def corpus_image_page_map(self) -> dict[Path, tuple[str, int]]:
        return {self.image_path: ("doc", 1)}

    def list_example_ids(self) -> list[str]:
        return ["q1"]

    def get_example(self, example_id: str):
        return SimpleNamespace(
            example_id=example_id,
            doc_name="doc",
            page_ids=[1],
            question="What is shown?",
            correct_answer="answer",
        )


class FakeVisualRetriever:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def retrieve(self, query: str, top_k: int):
        return [(self.image_path, 0.9)]


class FakeVisualFallback:
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


def test_saved_visual_baseline_row_contains_vlm_provenance(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    repository = FakeVisualRepository(image_path)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: repository)
    monkeypatch.setattr(
        "faar.visual_baselines.build_visual_retriever",
        lambda *args, **kwargs: FakeVisualRetriever(image_path),
    )
    monkeypatch.setattr("faar.visual_baselines.VisualFallback", FakeVisualFallback)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    output_path = tmp_path / "visual.json"

    run._run_visual_baseline_to_result(
        settings,
        "colpali",
        output_path,
        None,
        {"dataset": "ohrbench", "split": "test"},
        "ohrbench",
        "test",
    )

    saved_row = json.loads(output_path.read_text())["rows"][0]
    assert saved_row["request_model"] == "gpt-4o-2024-11-20"
    assert saved_row["response_model"] == "gpt-4o-2024-11-20"
    assert saved_row["completed_at_utc"] == "2026-07-23T12:00:00+00:00"
    assert saved_row["cost_rates"]["input_usd_per_million_tokens"] == 2.5
    assert saved_row["cost_rates"]["output_usd_per_million_tokens"] == 10.0
    run_metadata = saved_row["run_metadata"]
    assert run_metadata["profile"] == "colpali"
    assert run_metadata["run_fingerprint"]
    assert run_metadata["vlm_backend"] == "openai"
    assert run_metadata["selection"] == {"max_examples": None}
    assert run_metadata["cost_rates"]["input_usd_per_million_tokens"] == 2.5
