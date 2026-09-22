"""End-to-end demo of the full pipeline on a tiny synthetic OCR-corrupted corpus.

Deterministic and offline: the default local-hash embedding backend, the mock
VLM, and ByT5 disabled. Exercises chunking -> hybrid retrieval -> quality gate
-> diagnosis -> recovery routing -> answering -> metrics on real code paths.
"""

from pathlib import Path

from faar.experiment_runner import run_profile
from faar.graph import build_graph
from faar.settings import AppSettings


def _prepare_corpus(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "clean1,doc/alpha,What is the threshold for zephyr compliance?,42%,0\n"
        "clean2,doc/beta,Is the system required to log access?,Yes,0\n"
        "noisy3,doc/gamma,What is the zephyr compliance threshold?,42%,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/clean1.txt").write_text(
        "===== PAGE 0 =====\nZephyr compliance requires a threshold of 42% and monthly reporting."
    )
    (tmp_path / "artifacts/phase0/ocr_text/clean2.txt").write_text(
        "===== PAGE 0 =====\nAccess logging is mandatory for all systems."
    )
    (tmp_path / "artifacts/phase0/ocr_text/noisy3.txt").write_text(
        "===== PAGE 0 =====\nZ3phyr c0mpliance requires a thr3shold of 42% and m0nthly r3porting."
    )


def _settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings(project_root=tmp_path)
    settings.retrieval.embedding_backend = "local-hash-v1"
    settings.gate.quality_threshold = 0.52
    settings.recovery.vlm_backend = "mock"
    settings.recovery.enable_byt5 = False
    settings.recovery.api_enabled = False
    return settings


def test_clean_example_passes_gate_and_answers_directly(tmp_path: Path) -> None:
    _prepare_corpus(tmp_path)
    result = build_graph(_settings(tmp_path)).invoke({"example_id": "clean1"})
    assert result["gate"]["pass_gate"] is True
    assert result["gate"]["reasons"] == []
    assert result.get("policy_action", "answer_direct") == "answer_direct"
    assert "42%" in result["answer"]
    assert result.get("visual_result") is None


def test_clean_yes_no_example_answers_directly(tmp_path: Path) -> None:
    _prepare_corpus(tmp_path)
    result = build_graph(_settings(tmp_path)).invoke({"example_id": "clean2"})
    assert result["gate"]["pass_gate"] is True
    assert result["answer"] == "Yes"
    assert result["answer_meta"]["answer_mode"] == "yes_no_inference"


def test_noisy_example_routes_to_word_level_recovery(tmp_path: Path) -> None:
    _prepare_corpus(tmp_path)
    result = build_graph(_settings(tmp_path)).invoke({"example_id": "noisy3"})
    assert result["gate"]["pass_gate"] is False
    assert "word_noise_alert" in result["gate"]["reasons"]
    assert result["failure_type"] == "word_level"
    assert result["policy_action"] == "correct_text"
    assert result["action_outcome"]["action"] == "correct_text"
    assert "42%" in result["answer"]
    assert result.get("visual_result") is None


def test_end_to_end_invocation_is_deterministic(tmp_path: Path) -> None:
    _prepare_corpus(tmp_path)
    graph = build_graph(_settings(tmp_path))
    first = graph.invoke({"example_id": "noisy3"})
    second = graph.invoke({"example_id": "noisy3"})
    first.pop("retriever", None)
    second.pop("retriever", None)
    assert first == second


def test_run_profile_handles_full_corpus_end_to_end(tmp_path: Path) -> None:
    _prepare_corpus(tmp_path)
    settings = _settings(tmp_path)
    rows = run_profile(settings, profile_name="faar_full", example_ids=["clean1", "clean2", "noisy3"])
    assert len(rows) == 3
    for row in rows:
        assert set(row["metrics"]) == {"ndcg@5", "recall@5", "em", "f1"}
        assert "recovery_metrics" in row
        assert row["question"]
        assert row["gold_answer"]
    by_id = {row["example_id"]: row for row in rows}
    assert by_id["clean1"]["failure_type"] == "pass"
    assert by_id["noisy3"]["failure_type"] == "word_level"
    assert by_id["clean1"]["action_outcome"]["action"] == "answer_direct"
    assert by_id["noisy3"]["action_outcome"]["action"] == "correct_text"
    out_dir = tmp_path / "logs/phase3" / "faar_full"
    assert (out_dir / "clean1.json").exists()
    assert (out_dir / "noisy3.json").exists()
