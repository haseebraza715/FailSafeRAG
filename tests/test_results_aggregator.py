from faar.results_aggregator import summarize_by_profile, summarize_examples


def test_summarize_examples_empty() -> None:
    out = summarize_examples([])
    assert out["count"] == 0
    assert out["em"] == 0.0


def test_summarize_by_profile() -> None:
    rows = [
        {"profile": "faar_full", "policy_action": "answer_direct", "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0}},
        {"profile": "faar_full", "policy_action": "invoke_vlm", "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0}},
    ]
    summary = summarize_by_profile(rows)
    assert "faar_full" in summary
    assert summary["faar_full"]["count"] == 2
