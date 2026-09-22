import json
from pathlib import Path

from faar.data import Phase0Repository
from faar.experiment_runner import run_profile
from faar.settings import AppSettings


class FakeGraph:
    def invoke(self, _state):
        return {
            "question": "What is required?",
            "example": type("Ex", (), {"correct_answer": "certified engineers"})(),
            "answer": "certified engineers",
            "failure_type": "semantic",
            "policy_action": "retry_retrieval",
            "action_outcome": {"action": "retry_retrieval", "status": "succeeded"},
            "visual_result": {
                "request_model": "gpt-4o-2024-11-20",
                "response_model": "gpt-4o-2024-11-20",
                "completed_at_utc": "2026-07-23T12:00:00+00:00",
                "api_usage": {
                    "api_requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                },
            },
            "retrieved_hits": [],
            "corrected_hits": [],
        }


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
        "ex2,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text("===== PAGE 0 =====\ncertified engineers")
    (tmp_path / "artifacts/phase0/ocr_text/ex2.txt").write_text("===== PAGE 0 =====\ncertified engineers")
    (tmp_path / "data/phase0/manual_labels.csv").write_text(
        "example_id,question,correct_answer,ocr_output_snippet,failure_type,notes\n"
        "ex1,What is required?,certified engineers,snippet,no_issue,stable\n"
        "ex2,What is required?,certified engineers,snippet,text_corruption,noisy\n"
    )


def test_run_profile_writes_rows(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph())
    settings = AppSettings(project_root=tmp_path)
    rows = run_profile(
        settings,
        profile_name="faar_full",
        max_examples=1,
        example_ids=["ex2"],
        selection={"manual_failure_types": ["text_corruption"]},
    )
    assert len(rows) == 1
    assert rows[0]["profile"] == "faar_full"
    assert rows[0]["example_id"] == "ex2"
    assert rows[0]["metrics"]["em"] == 1.0
    assert rows[0]["run_metadata"]["selection"]["manual_failure_types"] == ["text_corruption"]
    saved = json.loads((tmp_path / "logs/phase3/faar_full/ex2.json").read_text())
    assert saved["request_model"] == "gpt-4o-2024-11-20"
    assert saved["response_model"] == "gpt-4o-2024-11-20"
    assert saved["completed_at_utc"] == "2026-07-23T12:00:00+00:00"
    assert saved["cost_rates"]["input_usd_per_million_tokens"] == 2.5
    assert saved["cost_rates"]["output_usd_per_million_tokens"] == 10.0
    assert saved["run_metadata"]["cost_rates"]["input_usd_per_million_tokens"] == 2.5
    assert saved["run_metadata"]["cost_rates"]["output_usd_per_million_tokens"] == 10.0


def test_run_profile_records_recovery_metrics(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FakeGraph())
    settings = AppSettings(project_root=tmp_path)
    rows = run_profile(
        settings,
        profile_name="faar_full",
        max_examples=1,
        example_ids=["ex2"],
        selection={"manual_failure_types": ["text_corruption"]},
    )
    recovery = rows[0]["recovery_metrics"]
    # No pre-recovery hits exist in the fake, so the counterfactual answer is empty
    # and the direct-answer path is the only answer producer.
    assert recovery["counterfactual_answer"] == ""
    assert recovery["recovery_changed_answer"] is True
    assert set(recovery["em"]) == {"actual", "counterfactual", "delta", "effect"}
    assert set(recovery["f1"]) == {"actual", "counterfactual", "delta", "effect"}
    assert recovery["em"]["actual"] == 1.0
    assert recovery["em"]["effect"] == "improved"


def test_phase0_repository_supports_stratified_selection(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    settings = AppSettings(project_root=tmp_path)
    repo = Phase0Repository(settings)
    selected = repo.select_example_ids(stratify_by="manual_failure_type", examples_per_stratum=1)
    assert selected == ["ex1", "ex2"]
