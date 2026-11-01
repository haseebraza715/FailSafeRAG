"""Failure list: wrong sample identity, split, labels, overwrite, or invalid decision."""

import importlib.util
import shutil
from pathlib import Path

import pytest
from test_compare_confirmation import confirmation, frozen  # noqa: F401
from test_compare_round import change_json, jsonl, read_jsonl, sha
from test_compare_round import sample as round_sample  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/faar_test/compare_test.py"
spec = importlib.util.spec_from_file_location("compare_test", SCRIPT)
test_comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_comparison)
test_comparison.confirmation = confirmation


@pytest.fixture(params=["primary", "secondary"])
def test_sample(request):
    root, dirs, pilot, _ = request.getfixturevalue("frozen")
    kind = "multipage" if request.param == "primary" else "singlepage"
    pilot_id = f"ohr_test_{kind}_v1"
    prefix = f"2026-11-01-ohr-test-{kind}-v1-"
    target = pilot.parent / pilot_id
    shutil.move(pilot, target)
    for filename in ("selection_record.json", "runtime_manifest.json", "evaluation_manifest.json"):
        change_json(target / filename, lambda x: x.update(pilot_id=pilot_id))
    change_json(target / "selection_record.json", lambda x: x["config"]["selection"].update(question_splits=["test"]))
    runtime_sha = sha(target / "runtime_manifest.json")
    suffixes = {"baseline": "baseline-answer-v2-terra", "diagnosis": "diagnosis", "invoke_vlm": "visual-terra", "retry_retrieval": "retry-keep-top3-terra", "correct_text": "correct-symspell-terra"}
    new_dirs = {}
    for name, old in dirs.items():
        new = old.parent / (prefix + suffixes[name])
        shutil.move(old, new)
        new_dirs[name] = new
    change_json(new_dirs["baseline"] / "run_config.json", lambda x: x["identity"].update(runtime_manifest_sha256=runtime_sha))
    change_json(new_dirs["baseline"] / "run_summary.json", lambda x: x.update(pilot_id=pilot_id))
    for name in ("diagnosis", *test_comparison.confirmation.shared.REPAIRS):
        change_json(new_dirs[name] / "run_config.json", lambda x: x.update(pilot_id=pilot_id, baseline_run=new_dirs["baseline"].name, runtime_manifest_sha256=runtime_sha))
    return root, new_dirs, target, request.param, prefix


def test_comparison_samples(test_sample):
    root, _, _, sample, prefix = test_sample
    result = test_comparison.compare(root, sample)
    report = (root / "results/analysis" / (prefix + "final-test") / "report.md").read_text()
    assert f"final test policy comparison ({sample})" in report
    assert result["decision"]["label"] in report
    assert "Descriptive headroom (not a claim)" in report
    with pytest.raises(test_comparison.confirmation.shared.ComparisonError, match="output directory exists"):
        test_comparison.compare(root, sample)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_wrong_split_refused(test_sample, split):
    root, _, pilot, sample, _ = test_sample
    change_json(pilot / "selection_record.json", lambda x: x["config"]["selection"].update(question_splits=[split]))
    with pytest.raises(test_comparison.confirmation.shared.ComparisonError):
        test_comparison.compare(root, sample)


def test_invalid_comparison(test_sample):
    root, dirs, _, sample, prefix = test_sample
    repair = dirs["retry_retrieval"]
    rows = read_jsonl(repair / "predictions.jsonl")
    rows[0].update(status="execution_failed", blocked=True, answer=None, usage=None)
    jsonl(repair / "predictions.jsonl", rows)
    events = read_jsonl(repair / "attempts.jsonl")
    event = next(e for e in events if e.get("attempt_id") == "retry_retrieval-q0" and e["event"] == "saved")
    event.update(event="failed", outcome="unknown", kind="timeout")
    event.pop("measured_cost")
    jsonl(repair / "attempts.jsonl", events)
    change_json(repair / "run_summary.json", lambda x: x.update(complete=False, blocked=["q0"], measured_cost_usd=0.4, reserved_cost_usd=0.5))
    assert test_comparison.compare(root, sample)["decision"]["met"] is None
    assert "Invalid comparison under amendment 2. No decision." in (root / "results/analysis" / (prefix + "final-test") / "report.md").read_text()
