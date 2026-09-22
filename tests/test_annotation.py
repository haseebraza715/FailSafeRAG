import json
from pathlib import Path

import pytest

from faar.annotation import cohens_kappa, load_label_studio_labels, sample_failures


def test_samples_only_incorrect_baseline_rows(tmp_path: Path) -> None:
    baseline = tmp_path / "b0.json"
    baseline.write_text(
        json.dumps(
            {
                "rows": [
                    {"example_id": "correct", "metrics": {"em": 1.0}},
                    {"example_id": "wrong-1", "metrics": {"em": 0.0}},
                    {"example_id": "wrong-2", "metrics": {"em": 0.0}},
                ]
            }
        )
    )
    assert [sample["example_id"] for sample in sample_failures(baseline, 2)] == ["wrong-1", "wrong-2"]


def test_sampling_requires_requested_failure_count(tmp_path: Path) -> None:
    baseline = tmp_path / "b0.json"
    baseline.write_text(json.dumps({"rows": [{"example_id": "wrong", "metrics": {"em": 0.0}}]}))
    with pytest.raises(ValueError, match="contains only 1"):
        sample_failures(baseline, 100)


def test_kappa_reads_two_independent_label_exports(tmp_path: Path) -> None:
    def export(path: Path, labels: list[str]) -> None:
        tasks = []
        for index, label in enumerate(labels):
            tasks.append(
                {
                    "data": {"example_id": f"e{index}"},
                    "annotations": [{"result": [{"value": {"choices": [label]}}]}],
                }
            )
        path.write_text(json.dumps(tasks))

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    export(first, ["semantic", "word_level", "structural", "other"])
    export(second, ["semantic", "word_level", "structural", "other"])
    result = cohens_kappa(load_label_studio_labels(first), load_label_studio_labels(second))
    assert result["cohens_kappa"] == 1.0
    assert result["passed"] is True
