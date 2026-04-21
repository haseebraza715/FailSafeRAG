from pathlib import Path

from faar.data import Phase0Repository
from faar.experiment_runner import run_profile
from faar.settings import AppSettings


class FakeGraph:
    def invoke(self, state):
        example_id = state["example_id"]
        return {
            "question": "What is required?",
            "example": type("Ex", (), {"correct_answer": "certified engineers"})(),
            "answer": "certified engineers",
            "failure_type": "semantic",
            "policy_action": "retry_retrieval",
            "action_outcome": {"action": "retry_retrieval", "status": "succeeded"},
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


def test_phase0_repository_supports_stratified_selection(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    settings = AppSettings(project_root=tmp_path)
    repo = Phase0Repository(settings)
    selected = repo.select_example_ids(stratify_by="manual_failure_type", examples_per_stratum=1)
    assert selected == ["ex1", "ex2"]
