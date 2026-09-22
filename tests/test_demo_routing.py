"""Assert the demo's routing claims end-to-end against the committed fixture.

scripts/demo.sh narrates four examples and claims:
  * bundled example (446d159e…): gate fires low_lexical_score -> semantic
    recovery -> correct answer "Yes" at 0 multimodal tokens
  * 1dc1d806…: clean direct pass
  * 8960b388…: word-level recovery
  * a9afa87a…: structural/visual recovery (mock backend, zero spend)

These tests pin that routing on the real committed data (skipped when the
phase-0 fixture is not present, e.g. in a partial checkout).
"""

from pathlib import Path

import pytest

from faar.graph import build_graph
from faar.settings import AppSettings

REPO_ROOT = Path(__file__).resolve().parents[1]

BUNDLED = "446d159e-b5c2-45dc-91cc-faaa931f3649"
CLEAN_PASS = "1dc1d806-6dda-4c9e-a391-cffa15ae14f2"
WORD_LEVEL = "8960b388-5154-469f-86f5-c3e0c3b82238"
STRUCTURAL = "a9afa87a-d181-49bf-aa09-9a46f25a3c68"

FIXTURE_PRESENT = (
    (REPO_ROOT / "data/phase0/sample_manifest.csv").exists()
    and (REPO_ROOT / "artifacts/phase0/ocr_text").is_dir()
)

pytestmark = pytest.mark.skipif(
    not FIXTURE_PRESENT,
    reason="committed phase-0 fixture not present in this checkout",
)


def _invoke(example_id: str) -> dict:
    settings = AppSettings(project_root=REPO_ROOT)
    settings.retrieval.embedding_backend = "local-hash-v1"
    settings.gate.quality_threshold = 0.52
    settings.recovery.vlm_backend = "mock"
    settings.recovery.enable_byt5 = False
    settings.recovery.api_enabled = False
    graph = build_graph(settings)
    return graph.invoke({"example_id": example_id})


def test_bundled_example_gates_low_lexical_and_recovers_semantically() -> None:
    result = _invoke(BUNDLED)
    assert result["gate"]["pass_gate"] is False
    assert "low_lexical_score" in result["gate"]["reasons"]
    assert result["failure_type"] == "semantic"
    assert result["policy_action"] == "retry_retrieval"
    assert result["action_outcome"]["action"] == "retry_retrieval"
    assert result["answer"] == "Yes"
    assert result.get("visual_result") is None


def test_bundled_example_answer_mode_is_yes_no_inference() -> None:
    result = _invoke(BUNDLED)
    assert result["answer_meta"]["answer_mode"] == "yes_no_inference"


def test_clean_pass_example_answers_directly() -> None:
    result = _invoke(CLEAN_PASS)
    assert result["gate"]["pass_gate"] is True
    assert result.get("policy_action", "answer_direct") == "answer_direct"
    assert result["answer"] == "forty-five (45) days"
    assert result.get("visual_result") is None


def test_word_noise_example_routes_to_word_level_recovery() -> None:
    result = _invoke(WORD_LEVEL)
    assert "word_noise_alert" in result["gate"]["reasons"]
    assert result["failure_type"] == "word_level"
    assert result["policy_action"] == "correct_text"
    assert result["action_outcome"]["action"] == "correct_text"
    assert result.get("visual_result") is None


def test_structural_example_routes_to_mock_vlm_with_zero_spend() -> None:
    result = _invoke(STRUCTURAL)
    assert "layout_alert" in result["gate"]["reasons"]
    assert result["failure_type"] == "structural"
    assert result["policy_action"] == "invoke_vlm"
    assert result["action_outcome"]["action"] == "invoke_vlm"
    assert result["visual_result"]["backend"] == "mock"
    assert result["visual_result"]["status"] == "skipped"
    assert result["visual_result"]["reason"] == "mock_backend_noop"
    assert result["visual_result"]["answer"] == ""
