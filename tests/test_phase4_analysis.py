import json
from pathlib import Path

from faar.phase4_analysis import (
    assess_phase4_claims,
    build_case_studies,
    build_phase4_report,
    load_phase3_rows,
    summarize_profile,
)


def _row(
    *,
    example_id: str,
    profile: str,
    failure_type: str = "pass",
    policy_action: str = "answer_direct",
    action: str = "answer_direct",
    f1: float = 0.0,
    em: float = 0.0,
) -> dict:
    return {
        "example_id": example_id,
        "profile": profile,
        "question": f"Q {example_id}",
        "gold_answer": "gold",
        "predicted_answer": "pred",
        "failure_type": failure_type,
        "policy_action": policy_action,
        "action_outcome": {"action": action, "status": "succeeded"},
        "metrics": {"ndcg@5": 0.5, "recall@5": 0.5, "em": em, "f1": f1},
    }


def test_load_phase3_rows_skips_invalid_json(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs/phase3"
    (logs_root / "faar_full").mkdir(parents=True)
    (logs_root / "faar_full/valid.json").write_text(json.dumps(_row(example_id="ex1", profile="faar_full")))
    (logs_root / "faar_full/broken.json").write_text("{bad-json")

    rows = load_phase3_rows(logs_root, profiles=["faar_full", "naive_rag"])
    assert len(rows["faar_full"]) == 1
    assert rows["naive_rag"] == []


def test_assess_phase4_claims_reports_negative_delta() -> None:
    rows_by_profile = {
        "faar_full": [
            _row(
                example_id="ex1",
                profile="faar_full",
                failure_type="structural",
                policy_action="invoke_vlm",
                action="invoke_vlm",
                f1=0.10,
            ),
            _row(
                example_id="ex2",
                profile="faar_full",
                failure_type="pass",
                policy_action="answer_direct",
                action="answer_direct",
                f1=0.00,
            ),
        ],
        "naive_rag": [
            _row(example_id="ex1", profile="naive_rag", f1=0.20),
            _row(example_id="ex2", profile="naive_rag", f1=0.20),
        ],
        "faar_no_vlm": [
            _row(example_id="ex1", profile="faar_no_vlm", f1=0.20),
            _row(example_id="ex2", profile="faar_no_vlm", f1=0.20),
        ],
    }

    claims = assess_phase4_claims(rows_by_profile)
    assert claims["q1_quality_gate_reduces_expensive_recovery"]["status"] == "supported"
    assert claims["q2_typed_recovery_outperforms_naive_fallback"]["status"] == "not_supported"
    assert claims["q3_selective_visual_fallback_trades_accuracy_for_cost"]["status"] == "not_supported"
    assert claims["q4_diagnostic_categories_match_observed_errors"]["status"] == "supported"


def test_build_case_studies_splits_improvements_and_regressions() -> None:
    rows_by_profile = {
        "faar_full": [
            _row(example_id="good", profile="faar_full", f1=1.0),
            _row(example_id="bad", profile="faar_full", f1=0.0),
            _row(example_id="tie", profile="faar_full", f1=0.5),
        ],
        "naive_rag": [
            _row(example_id="good", profile="naive_rag", f1=0.0),
            _row(example_id="bad", profile="naive_rag", f1=1.0),
            _row(example_id="tie", profile="naive_rag", f1=0.5),
        ],
    }

    case_studies = build_case_studies(rows_by_profile, top_n=5)
    assert case_studies["improvements"][0]["example_id"] == "good"
    assert case_studies["regressions"][0]["example_id"] == "bad"
    assert case_studies["ties"][0]["example_id"] == "tie"


def test_build_phase4_report_contains_required_sections() -> None:
    rows_by_profile = {
        "faar_full": [_row(example_id="ex1", profile="faar_full", f1=0.2)],
        "naive_rag": [_row(example_id="ex1", profile="naive_rag", f1=0.1)],
    }
    report = build_phase4_report(rows_by_profile)
    assert "profile_summaries" in report
    assert "claim_assessment" in report
    assert "case_studies" in report
    assert summarize_profile(rows_by_profile["faar_full"])["count"] == 1
