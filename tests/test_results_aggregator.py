import pytest

from faar.api_logging import is_valid_api_usage
from faar.results_aggregator import summarize_api_usage, summarize_by_profile, summarize_examples


def test_summarize_examples_empty() -> None:
    out = summarize_examples([])
    assert out["count"] == 0
    assert out["em"] == 0.0
    assert out["vlm_rate"] == 0.0
    assert out["recovery_outcomes"]["reviewed_count"] == 0


def test_summarize_by_profile() -> None:
    rows = [
        {
            "profile": "faar_full",
            "policy_action": "answer_direct",
            "action_outcome": {"action": "answer_direct"},
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
        },
        {
            "profile": "faar_full",
            "policy_action": "invoke_vlm",
            "action_outcome": {"action": "answer_direct"},
            "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0},
        },
        {
            "profile": "faar_full",
            "policy_action": "invoke_vlm",
            "action_outcome": {"action": "invoke_vlm"},
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 0.0, "f1": 0.0},
        },
    ]
    summary = summarize_by_profile(rows)
    assert "faar_full" in summary
    assert summary["faar_full"]["count"] == 3
    assert summary["faar_full"]["visual_fallback_rate"] == 0.3333
    assert summary["faar_full"]["vlm_rate"] == 0.3333


def test_summarize_api_usage_totals_all_rows() -> None:
    rows = [
        {"api_usage": {"api_requests": 1, "prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.00045}},
        {"api_usage": {"api_requests": 2, "prompt_tokens": 50, "completion_tokens": 10, "cost_usd": 0.0002}},
        {"api_usage": {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}},
    ]
    totals = summarize_api_usage(rows)
    assert totals == {
        "api_requests": 3,
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "cost_usd": 0.00065,
    }


@pytest.mark.parametrize("usage", [None, {}, {"api_requests": 1.5, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}])
def test_summarize_api_usage_rejects_invalid_rows(usage) -> None:
    with pytest.raises(ValueError, match="invalid api_usage"):
        summarize_api_usage([{"api_usage": usage}])


def test_api_usage_requires_integral_counts() -> None:
    assert not is_valid_api_usage(
        {"api_requests": 1.5, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    )


def test_summarize_examples_aggregates_recovery_outcomes() -> None:
    rows = [
        {
            "profile": "faar_full",
            "action_outcome": {"action": "retry_retrieval"},
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
            "recovery_metrics": {
                "recovery_changed_answer": True,
                "em": {"effect": "improved"},
                "f1": {"effect": "worsened"},
            },
        },
        {
            "profile": "faar_full",
            "action_outcome": {"action": "correct_text"},
            "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0},
            "recovery_metrics": {
                "recovery_changed_answer": False,
                "em": {"effect": "equal"},
                "f1": {"effect": "equal"},
            },
        },
    ]
    outcomes = summarize_examples(rows)["recovery_outcomes"]
    assert outcomes["reviewed_count"] == 2
    assert outcomes["changed_answer_count"] == 1
    assert outcomes["changed_answer_rate"] == 0.5
    assert outcomes["em_effect_distribution"] == {"improved": 1, "equal": 1}
    assert outcomes["f1_effect_distribution"] == {"worsened": 1, "equal": 1}


def test_summarize_examples_missing_recovery_metrics() -> None:
    rows = [
        {
            "profile": "naive_rag",
            "action_outcome": {"action": "answer_direct"},
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
        }
    ]
    outcomes = summarize_examples(rows)["recovery_outcomes"]
    assert outcomes["reviewed_count"] == 0
    assert outcomes["changed_answer_count"] == 0
    assert outcomes["em_effect_distribution"] == {}


def test_summarize_by_profile_empty_recovery_outcomes() -> None:
    summary = summarize_by_profile(
        [
            {
                "profile": "naive_rag",
                "action_outcome": {"action": "answer_direct"},
                "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
            }
        ]
    )
    assert summary["naive_rag"]["recovery_outcomes"]["reviewed_count"] == 0
    assert summary["naive_rag"]["recovery_outcomes"]["changed_answer_count"] == 0
