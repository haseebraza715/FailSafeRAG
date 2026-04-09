import csv
import json
from pathlib import Path

from faar.cli import run_phase4


def _row(*, profile: str, example_id: str, f1: float, action: str = "answer_direct", failure_type: str = "pass") -> dict:
    return {
        "profile": profile,
        "example_id": example_id,
        "question": f"Q {example_id}",
        "gold_answer": "gold",
        "predicted_answer": "pred",
        "failure_type": failure_type,
        "policy_action": "answer_direct" if failure_type == "pass" else "retry_retrieval",
        "action_outcome": {"action": action, "status": "succeeded"},
        "metrics": {"ndcg@5": 0.5, "recall@5": 0.5, "em": 0.0, "f1": f1},
    }


def test_run_phase4_writes_artifacts(tmp_path: Path) -> None:
    logs = tmp_path / "logs/phase3"
    (logs / "faar_full").mkdir(parents=True)
    (logs / "naive_rag").mkdir(parents=True)

    (logs / "faar_full/ex1.json").write_text(
        json.dumps(_row(profile="faar_full", example_id="ex1", f1=0.1, action="invoke_vlm", failure_type="semantic"))
    )
    (logs / "naive_rag/ex1.json").write_text(json.dumps(_row(profile="naive_rag", example_id="ex1", f1=0.0)))

    run_phase4(project_root=tmp_path, baseline_profile="naive_rag")

    out_dir = tmp_path / "artifacts/phase4"
    assert (out_dir / "phase4_summary.json").exists()
    assert (out_dir / "claim_assessment.json").exists()
    assert (out_dir / "case_studies.json").exists()
    assert (out_dir / "profile_deltas_vs_baseline.csv").exists()

    csv_path = out_dir / "profile_deltas_vs_baseline.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["profile"] == "faar_full" for row in rows)
