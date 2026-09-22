"""Phase 5 A3 contract: replace the word-level LLM with local SymSpell."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run

from faar.graph import build_graph
from faar.settings import AppSettings
from faar.types import Chunk, Phase0Example, RetrievalHit


class _StaticRetriever:
    def __init__(self, chunks, settings, *, cache_dir=None) -> None:
        self.hits = [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id="c1",
                    example_id="ex1",
                    doc_name="invoice",
                    page_id=0,
                    text="lnvoice t0tal is $98",
                ),
                bm25_score=0.0,
                dense_score=0.0,
                fused_score=0.0,
                reranker_score=0.0,
            )
        ]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return self.hits


class _SingleExampleRepository:
    def get_example(self, example_id: str) -> Phase0Example:
        assert example_id == "ex1"
        return Phase0Example(
            example_id="ex1",
            doc_name="invoice",
            question="What is the invoice total?",
            correct_answer="$98",
            page_ids=[0],
            ocr_text="lnvoice t0tal is $98",
            ocr_text_path=Path("invoice.txt"),
            gt_text_path=None,
            image_paths=[],
            metadata={},
        )


class _ByT5MustNotLoad:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("A3 must not construct or load ByT5 when --wordlevel_fallback symspell is selected")


def test_a3_cli_configures_symspell_wordlevel_fallback(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, AppSettings] = {}

    def fake_run(settings: AppSettings, profile: str, out: Path, *args, **kwargs) -> dict:
        captured["settings"] = settings
        assert profile == "faar_symspell"
        payload = {"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run.AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(run, "_validate_baseline", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_apply_baseline_harm", lambda payload, *args, **kwargs: payload)
    monkeypatch.setattr(run, "_run_profile_to_result", fake_run)
    baseline_path = tmp_path / "b0.json"
    baseline_path.write_text(json.dumps({"summary": {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}, "rows": []}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/experiments/run.py",
            "--ablate",
            "no_wordlevel_llm",
            "--wordlevel_fallback",
            "symspell",
            "--baseline",
            str(baseline_path),
            "--out",
            str(tmp_path / "a3.json"),
        ],
    )

    run.main()

    assert captured["settings"].experiment.wordlevel_fallback == "symspell"


def test_a3_wordlevel_recovery_uses_local_symspell_without_byt5(monkeypatch, tmp_path: Path) -> None:
    symspell_calls: list[str] = []

    def fake_symspell_correct(text: str) -> str:
        symspell_calls.append(text)
        return "invoice total is $98"

    monkeypatch.setattr("faar.graph.HybridRetriever", _StaticRetriever)
    monkeypatch.setattr("faar.graph.ByT5Corrector", _ByT5MustNotLoad)
    monkeypatch.setattr("faar.graph.symspell_correct_text", fake_symspell_correct)

    settings = AppSettings(project_root=tmp_path)
    settings.experiment.wordlevel_fallback = "symspell"
    settings.gate.quality_threshold = 0.5
    settings.gate.structural_threshold = 99

    result = build_graph(settings, repo=_SingleExampleRepository()).invoke({"example_id": "ex1"})

    assert symspell_calls == ["lnvoice t0tal is $98"]
    assert result["corrected_hits"][0].chunk.text == "invoice total is $98"
    assert result["action_outcome"]["decisions"][0]["reason"] == "symspell_local_fallback"
