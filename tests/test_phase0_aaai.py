import json
from pathlib import Path

import pytest
from evaluate import evaluate_results
from split_dataset import build_split


def test_split_dataset_is_deterministic(tmp_path: Path) -> None:
    data_path = tmp_path / "qas.json"
    data_path.write_text(json.dumps([{"ID": f"ex{i}"} for i in range(20)]))
    first = build_split(data_path, seed=42)
    second = build_split(data_path, seed=42)
    assert first["splits"] == second["splits"]
    assert first["counts"] == {"train": 14, "val": 3, "test": 3}


def test_evaluate_emits_required_metrics(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "example_id": "ex1",
                        "predicted_answer": "certified engineers",
                        "gold_answer": "certified engineers",
                        "action_outcome": {"action": "answer_direct"},
                    }
                ]
            }
        )
    )
    metrics = evaluate_results(results)
    assert set(metrics) == {"EM", "F1", "vlm_rate", "harm_rate"}
    assert metrics["EM"] == 1.0
    assert metrics["F1"] == 1.0


def test_evaluate_recomputes_from_rows_when_rows_are_present(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "summary": {"EM": 0.99, "F1": 0.99, "vlm_rate": 0.99, "harm_rate": 0.99},
                "rows": [
                    {
                        "example_id": "ex1",
                        "predicted_answer": "certified engineers",
                        "gold_answer": "certified engineers",
                        "metrics": {"em": 1.0, "f1": 1.0},
                        "action_outcome": {"action": "answer_direct"},
                    },
                    {
                        "example_id": "ex2",
                        "predicted_answer": "wrong",
                        "gold_answer": "right answer here",
                        "metrics": {"em": 0.0, "f1": 0.0},
                        "action_outcome": {"action": "invoke_vlm"},
                    },
                ],
            }
        )
    )
    metrics = evaluate_results(results)
    assert metrics["EM"] == 0.5
    assert metrics["F1"] == 0.5
    assert metrics["vlm_rate"] == 0.5
    assert metrics["harm_rate"] == 0.0


def test_evaluate_rejects_incomplete_summary_without_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"summary": {"EM": 0.5, "F1": 0.5}}))
    with pytest.raises(ValueError, match="missing or null"):
        evaluate_results(results)


def test_evaluate_rejects_null_summary_metrics_without_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {"summary": {"EM": 0.5, "F1": 0.5, "vlm_rate": None, "harm_rate": 0.1}}
        )
    )
    with pytest.raises(ValueError, match="vlm_rate"):
        evaluate_results(results)


def test_evaluate_returns_complete_summary_when_no_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {"summary": {"EM": 0.4, "F1": 0.6, "vlm_rate": 0.3, "harm_rate": 0.1}}
        )
    )
    assert evaluate_results(results) == {"EM": 0.4, "F1": 0.6, "vlm_rate": 0.3, "harm_rate": 0.1}


def test_evaluate_rejects_missing_baseline_file(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "example_id": "ex1",
                        "predicted_answer": "answer",
                        "gold_answer": "answer",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="does not exist"):
        evaluate_results(results, baseline_path=tmp_path / "missing-b0.json")


def test_evaluate_rejects_baseline_without_matching_row(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    baseline = tmp_path / "b0.json"
    results.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "example_id": "ex2",
                        "predicted_answer": "answer",
                        "gold_answer": "answer",
                    }
                ]
            }
        )
    )
    baseline.write_text(json.dumps({"rows": [{"example_id": "ex1"}]}))
    with pytest.raises(ValueError, match="does not cover"):
        evaluate_results(results, baseline_path=baseline)
