import json
from pathlib import Path

from faar.cli import run_example


class FakeGraph:
    def invoke(self, state):
        return {
            "question": state.get("question") or "Q?",
            "example": type("Ex", (), {"correct_answer": "A"})(),
            "answer": "A",
            "gate": {"pass_gate": True},
            "failure_type": "pass",
            "policy_action": "answer_direct",
            "semantic_retry_query": None,
            "visual_result": None,
            "answer_meta": {"answer_mode": "extractive_overlap"},
            "action_outcome": {"action": "answer_direct", "status": "succeeded", "reason": "passed_quality_gate"},
            "retrieved_hits": [],
        }


def test_cli_payload_contains_run_metadata(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text("example_id,doc_name,question,correct_answer,page_no\n")
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text('{"results":[]}')
    monkeypatch.setattr("faar.cli.build_graph", lambda settings: FakeGraph())

    out = tmp_path / "logs/phase1/out.json"
    run_example(
        example_id="ex1",
        question=None,
        output=out,
        project_root=tmp_path,
        vlm_backend="mock",
        seed=42,
    )
    payload = json.loads(out.read_text())
    assert "run_metadata" in payload
    assert "action_outcome" in payload
