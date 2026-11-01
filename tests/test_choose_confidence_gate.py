"""Failure list for choosing a confidence gate.

T1. Non-train, confidence-gated or mismatched inputs are accepted.
T2. Scoring, strict comparison, exclusions or cap are wrong.
T3. Ties, nonpositive or undefined J choose the wrong condition.
T4. The printed threshold changes the gate, or output is overwritten.
T5. An undefined feature suppresses a valid candidate, or input provenance is incomplete.
T6. The chooser cannot consume an actual diagnosis run configuration.
"""
import csv
import hashlib
import json
import shlex
import sys

import choose_confidence_gate as choice
import diagnose_run
import pytest
import run_visual_repair as rvr
from test_text_repair import fake_replay
from test_visual_repair import PILOT, Project

from faar.diagnosis import CONFIDENCE_FEATURES, GateRule, gate


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fixture(root, values, wrong, flagged=(), null=()):
    pilot = root / "results/pilots/pilot"
    base = root / "baseline"
    diag = root / "diagnosis"
    ids = [str(i) for i in range(len(values))]
    write(pilot / "selection_record.json", {"pilot_id": "pilot", "config": {"selection": {"question_splits": ["train"]}}, "selected_question_ids": ids})
    write(pilot / "evaluation_manifest.json", {"pilot_id": "pilot", "questions": {i: {"answers": "yes"} for i in ids}})
    lines(base / "requests.jsonl", [{"question_id": i} for i in ids])
    lines(base / "predictions.jsonl", [{"question_id": i, "status": "answered", "answer": "no" if int(i) in wrong else "yes"} for i in ids])
    lines(diag / "diagnosis.jsonl", [{"question_id": i, "flagged": int(i) in flagged,
         "features": {f: None if int(i) in null else values[int(i)] for f in CONFIDENCE_FEATURES}} for i in ids])
    rvr.write_identity(diag, {"pilot_id": "pilot", "baseline_run": "baseline",
          "baseline_requests_sha256": hashlib.sha256((base / "requests.jsonl").read_bytes()).hexdigest(),
          "gate_rule": {"feature": None, "threshold": None}}, None)
    return pilot, base, diag, root / "out"


def run(root, paths):
    _, base, diag, out = paths
    return choice.choose(root, "pilot", base, diag, out)


def test_refusals_before_output_and_gold(tmp_path):
    paths = fixture(tmp_path, list(range(10)), {0, 1})
    pilot, base, diag, out = paths
    record = choice.read_json(pilot / "selection_record.json")
    record["config"]["selection"]["question_splits"] = ["val"]
    write(pilot / "selection_record.json", record)
    (pilot / "evaluation_manifest.json").unlink()
    with pytest.raises(ValueError, match="split"):
        run(tmp_path, paths)
    assert not out.exists()
    record["config"]["selection"]["question_splits"] = ["train"]
    write(pilot / "selection_record.json", record)
    config = choice.read_json(diag / "run_config.json")
    config["gate_rule"] = {"feature": "fused_margin", "threshold": 0.1}
    write(diag / "run_config.json", config)
    with pytest.raises(ValueError, match="base gate"):
        run(tmp_path, paths)
    config["gate_rule"] = {"feature": None, "threshold": None}
    config["baseline_requests_sha256"] = "wrong"
    write(diag / "run_config.json", config)
    with pytest.raises(ValueError, match="mismatch"):
        run(tmp_path, paths)
    assert not out.exists()


def test_strict_exclusions_and_round_trip(tmp_path, monkeypatch):
    paths = fixture(tmp_path, [0.1, 0.12345678901234566, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0], {0, 10, 11}, flagged={10}, null={11})
    result = run(tmp_path, paths)
    assert result["feature"] == CONFIDENCE_FEATURES[0]
    assert result["threshold"] == 0.12345678901234566
    assert result["population"] == {"N": 10, "cap": 3, "positives": 1, "negatives": 9}
    tokens = shlex.split(result["gate_arguments"])
    threshold = float(tokens[3])
    parsed = []
    def capture(**kwargs):
        parsed.append(kwargs["rule"])
        return []
    monkeypatch.setattr(diagnose_run, "diagnose_run", capture)
    monkeypatch.setattr(sys, "argv", ["diagnose_run.py", "--pilot-id", "pilot", "--baseline-run-dir", "baseline", "--out-dir", "out", *tokens])
    assert diagnose_run.main() == 0
    rule = parsed[0]
    assert rule == GateRule(tokens[1], threshold)
    rows = choice.indexed(paths[2] / "diagnosis.jsonl")
    selected = [r for r in rows.values() if not r["flagged"] and gate(initial_status="answered", abstained=False, pages=[], features=r["features"], rule=rule)[0]]
    assert len(selected) == result["flagged"] == 1
    assert rows["1"]["features"][tokens[1]] == threshold
    with pytest.raises(ValueError, match="not empty"):
        run(tmp_path, paths)


def test_cap_excludes_higher_j(tmp_path):
    paths = fixture(tmp_path, list(range(10)), {0, 1, 2, 3, 4})
    result = run(tmp_path, paths)
    assert result["threshold"] == 3.0
    with (paths[3] / "candidates.csv").open() as f:
        rows = list(csv.DictReader(f))
    higher = next(r for r in rows if r["feature"] == CONFIDENCE_FEATURES[0] and float(r["t"]) == 5)
    assert float(higher["J"]) > result["J"] and higher["within_cap"] == "False"


def test_tie_fewer_flags_then_feature(tmp_path):
    paths = fixture(tmp_path, list(range(10)), {0, 2, 4, 6, 8})
    result = run(tmp_path, paths)
    assert result["threshold"] == 1 and result["flagged"] == 1
    assert result["feature"] == CONFIDENCE_FEATURES[0]


@pytest.mark.parametrize("wrong", [set(), set(range(10))])
def test_undefined(tmp_path, wrong):
    result = run(tmp_path, fixture(tmp_path, list(range(10)), wrong))
    assert result["feature"] is None and "undefined" in result["reason"]


def test_no_positive_j(tmp_path):
    result = run(tmp_path, fixture(tmp_path, list(range(10)), {7, 8, 9}))
    assert result["feature"] is None and "above zero" in result["reason"]


def test_failed_and_no_evidence_are_wrong_even_if_answer_matches(tmp_path):
    paths = fixture(tmp_path, list(range(10)), set())
    predictions = list(choice.indexed(paths[1] / "predictions.jsonl").values())
    predictions[0]["status"] = "execution_failed"
    predictions[1]["status"] = "no_evidence"
    lines(paths[1] / "predictions.jsonl", predictions)
    result = run(tmp_path, paths)
    assert result["positives"] == 2 and result["threshold"] == 2


def test_undefined_feature_does_not_suppress_other_candidates(tmp_path):
    paths = fixture(tmp_path, list(range(10)), {0})
    rows = list(choice.indexed(paths[2] / "diagnosis.jsonl").values())
    for row in rows:
        row["features"]["query_token_coverage"] = None
    lines(paths[2] / "diagnosis.jsonl", rows)
    result = run(tmp_path, paths)
    assert result["feature"] == "top_fused_score"
    assert result["threshold"] == 1.0
    assert result["reason"] is None


def test_records_scoring_and_selection_hashes(tmp_path):
    paths = fixture(tmp_path, list(range(10)), {0})
    run(tmp_path, paths)
    saved = choice.read_json(paths[3] / "gate_choice.json")
    for path in (paths[0] / "selection_record.json", choice.ROOT / "src/faar/ohr_scoring.py"):
        assert saved["sha256"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_choose_from_actual_diagnosis_run(tmp_path, monkeypatch):
    project = Project(tmp_path)
    pilot = tmp_path / "results/pilots" / PILOT
    ids = ["q1", "q2", "q3"]
    write(pilot / "selection_record.json", {"pilot_id": PILOT, "config": {"selection": {"question_splits": ["train"]}}, "selected_question_ids": ids})
    lines(project.baseline / "predictions.jsonl", [{"question_id": qid, "status": "answered", "answer": "x", "abstained": False} for qid in ids])
    monkeypatch.setattr(diagnose_run, "replay_retrieval", lambda root, pilot_id, requests: fake_replay(project))
    diagnosis = tmp_path / "real_diagnosis"
    diagnose_run.diagnose_run(root=tmp_path, pilot_id=PILOT, baseline_run_dir=project.baseline, out_dir=diagnosis, rule=GateRule())
    assert (diagnosis / "run_config.json").is_file()
    result = choice.choose(tmp_path, PILOT, project.baseline, diagnosis, tmp_path / "choice")
    assert result["feature"] is None
    assert result["sha256"][str(diagnosis / "run_config.json")] == hashlib.sha256((diagnosis / "run_config.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("selection", [{}, {"pool": "train_exclusive"}, {"question_splits": []}, {"question_splits": ["train", "val"]}])
def test_unknown_split_refused_before_gold(tmp_path, selection):
    paths = fixture(tmp_path, list(range(10)), {0})
    change = choice.read_json(paths[0] / "selection_record.json")
    change["config"]["selection"] = selection
    write(paths[0] / "selection_record.json", change)
    (paths[0] / "evaluation_manifest.json").unlink()
    with pytest.raises(ValueError, match="split"):
        run(tmp_path, paths)
