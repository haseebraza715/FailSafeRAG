"""Failure list:

1. A non-train pilot, mismatched provenance, incomplete run or changed question set is accepted.
2. Output overwrites existing files, or input and code hashes are omitted.
3. Skips, blocked answers, oracle ties or random assignments select the wrong answer.
4. Pooled random p-values, Holm adjustment, disagreement or power use the wrong rule.
5. Unknown attempts lose their upper-bound cost, or skips incur repair cost.
6. Amendment 2 invalid comparisons are reported as valid.
7. The three output files omit the required results or route diagnostics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from faar.policies import RANDOM_SEEDS, REPAIRS, random_repair

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/faar_long_dev/compare_round.py"
spec = importlib.util.spec_from_file_location("compare_round", SCRIPT)
compare_round = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_round)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@pytest.fixture
def sample(tmp_path):
    pilot = tmp_path / "results/pilots/pilot"
    baseline, diagnosis = tmp_path / "baseline", tmp_path / "diagnosis"
    dirs = {name: tmp_path / name for name in REPAIRS}
    out = tmp_path / "out"
    questions = [{"question_id": f"q{i}", "doc_id": f"d{i // 2}", "question": f"question {i}"} for i in range(4)]
    dump(pilot / "selection_record.json", {"pilot_id": "pilot", "config": {"selection": {"question_splits": ["train"]}}, "selected_question_ids": [q["question_id"] for q in questions]})
    dump(pilot / "runtime_manifest.json", {"pilot_id": "pilot", "questions": questions, "documents": []})
    dump(pilot / "evaluation_manifest.json", {"pilot_id": "pilot", "questions": {q["question_id"]: {"answers": "yes"} for q in questions}})
    runtime_sha = sha(pilot / "runtime_manifest.json")
    req = [{"question_id": q["question_id"], "doc_id": q["doc_id"], "action": "send"} for q in questions]
    jsonl(baseline / "requests.jsonl", req)
    base_sha = sha(baseline / "requests.jsonl")
    dump(baseline / "run_config.json", {"identity": {"runtime_manifest_sha256": runtime_sha}})
    dump(baseline / "run_summary.json", {"pilot_id": "pilot", "run_state": "complete", "valid_baseline": True,
                                         "cost": {"measured": 0.4, "reserved": 0}})
    jsonl(baseline / "predictions.jsonl", [{"question_id": q["question_id"], "status": "answered", "answer": "yes" if i == 0 else "no"} for i, q in enumerate(questions)])
    events = []
    for i, q in enumerate(questions):
        qid = q["question_id"]
        events.extend([{"event": "dispatch_started", "question_id": qid, "attempt_id": qid, "cost_upper_bound": 0.2},
                       {"event": "response_saved", "question_id": qid, "attempt_id": qid, "measured_cost": 0.1,
                        "usage": {"input_tokens": 10, "output_tokens": 2}}])
    jsonl(baseline / "attempts.jsonl", events)
    identity = {"pilot_id": "pilot", "baseline_run": "baseline", "baseline_requests_sha256": base_sha, "runtime_manifest_sha256": runtime_sha}
    dump(diagnosis / "run_config.json", identity)
    diag = [{"question_id": q["question_id"], "doc_id": q["doc_id"], "flagged": i < 3, "gate_reasons": ["reason"] if i < 3 else [],
             "diagnosis": "semantic", "route": "retry_retrieval"} for i, q in enumerate(questions)]
    jsonl(diagnosis / "diagnosis.jsonl", diag)
    for repair, directory in dirs.items():
        records = [{"question_id": f"q{i}", "doc_id": f"d{i // 2}", "action": "skip" if repair == "correct_text" and i == 1 else "send",
                    "skip_reason": "nothing" if repair == "correct_text" and i == 1 else None,
                    **({"selected_pages": [0, 1]} if repair == "invoke_vlm" else {})} for i in range(3)]
        dump(directory / "run_config.json", {**identity, "run_id": repair, "repair": repair, "requests_sha256": canonical(records),
                                               "render": {"dpi": 150, "detail": "high"}})
        jsonl(directory / "requests.jsonl", records)
        predictions, ev = [], []
        for i, rec in enumerate(records):
            qid = rec["question_id"]
            if rec["action"] == "skip":
                predictions.append({"question_id": qid, "status": "skipped", "skip_reason": "nothing", "answer": "", "usage": None})
                continue
            usage = {"input_tokens": 20, "output_tokens": 3}
            predictions.append({"question_id": qid, "status": "answered", "answer": "yes", "measured_cost": 0.2, "usage": usage})
            aid = f"{repair}-{qid}"
            response_path = directory / "responses" / f"{aid}.json"
            dump(response_path, {"question_id": qid, "attempt_id": aid, "answer": "yes", "usage": usage, "measured_cost": 0.2, "latency_ms": 100})
            ev.extend([{"event": "dispatch", "question_id": qid, "attempt_id": aid, "cost_upper_bound": 0.5, "resend": False},
                       {"event": "saved", "question_id": qid, "attempt_id": aid, "measured_cost": 0.2, "response_sha256": sha(response_path)}])
        jsonl(directory / "predictions.jsonl", predictions)
        jsonl(directory / "attempts.jsonl", ev)
        dump(directory / "run_summary.json", {"complete": True, "blocked": [], "measured_cost_usd": 0.2 * len(predictions) - (0.2 if repair == "correct_text" else 0), "reserved_cost_usd": 0})
    return {"root": tmp_path, "pilot_id": "pilot", "round_number": 1, "baseline_dir": baseline,
            "diagnosis_dir": diagnosis, "repair_dirs": dirs, "out_dir": out}


def run(sample):
    return compare_round.compare(**sample)


def change_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    dump(path, value)


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def update_requests(directory, mutate):
    path = directory / "requests.jsonl"
    records = read_jsonl(path)
    mutate(records)
    jsonl(path, records)
    change_json(directory / "run_config.json", lambda config: config.update(requests_sha256=canonical(records)))


def test_outputs_policies_statistics_and_hashes(sample):
    result = run(sample)
    assert {p.name for p in sample["out_dir"].iterdir()} == {"case_scores.csv", "results.json", "report.md"}
    csv_text = (sample["out_dir"] / "case_scores.csv").read_text()
    assert "kept_initial:skip:nothing" in csv_text
    assert "q3" in csv_text
    assert result["routes"]["retry_retrieval"]["n"] == 3
    assert result["skips"]["correct_text"] == {"nothing": 1}
    assert result["policies"]["oracle"]["em"] == 0.75
    assert result["policies"]["all_correct_text"]["cost"]["per_question"]["q1"]["total"] == pytest.approx(0.1)
    assert result["policies"]["gated_visual"]["cost"]["sum"]["images"] == 6
    assert result["contrasts"]["random_pooled"]["mcnemar_em"]["p_value"] == max(result["contrasts"][f"random_{s}"]["mcnemar_em"]["p_value"] for s in RANDOM_SEEDS)
    assert result["detectable"]["disagreement_rate"] == 0
    assert result["decision"]["label"] == "development result, not a confirmation"
    assert str(SCRIPT) in result["input_and_code_sha256"]
    assert result["input_and_code_sha256"][str((sample["baseline_dir"] / "requests.jsonl").resolve())] == sha(sample["baseline_dir"] / "requests.jsonl")
    assert "descriptive, unadjusted" in (sample["out_dir"] / "report.md").read_text()
    with pytest.raises(compare_round.ComparisonError, match="not empty"):
        run(sample)


def test_oracle_tie_and_random_assignment(sample):
    reader = compare_round.Reader()
    data = compare_round.read_inputs(sample["root"], "pilot", sample["baseline_dir"], sample["diagnosis_dir"], sample["repair_dirs"], reader)
    rows = compare_round.cases(data)
    assert rows[0]["oracle:source"] == "initial"
    assert rows[1]["oracle:source"] == "repair:retry_retrieval"
    for seed in RANDOM_SEEDS:
        name = f"random_{seed}"
        choice = random_repair(seed, "q2")
        assert rows[2][f"{name}:source"] == f"repair:{choice}"


def test_holm_ordering():
    assert compare_round.holm({"a": 0.01, "b": 0.02, "c": 0.9}) == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.9})


def test_power_and_disagreement(sample):
    path = sample["repair_dirs"]["invoke_vlm"] / "predictions.jsonl"
    rows = read_jsonl(path)
    rows[2]["answer"] = "no"
    jsonl(path, rows)
    directory = sample["repair_dirs"]["invoke_vlm"]
    response = directory / "responses/invoke_vlm-q2.json"
    change_json(response, lambda x: x.update(answer="no"))
    events = read_jsonl(directory / "attempts.jsonl")
    next(e for e in events if e.get("attempt_id") == "invoke_vlm-q2" and e["event"] == "saved")["response_sha256"] = sha(response)
    jsonl(directory / "attempts.jsonl", events)
    result = run(sample)
    assert result["detectable"]["disagreement_rate"] == 0.25
    assert result["detectable"]["gain_80_percent_power"] is None
    assert compare_round.detectable_gain(1000, 0.1) == pytest.approx(0.028, abs=0.002)
    assert compare_round.detectable_gain(1000, 0) is None


@pytest.mark.parametrize("mutation, message", [
    ("split", "train"), ("diagnosis_baseline", "identity"), ("diagnosis_hash", "identity"),
    ("repair_baseline", "identity"), ("repair_hash", "identity"), ("repair_pilot", "identity"),
    ("diagnosis_set", "question set"), ("repair_set", "question set"), ("flagged_file", "question set"),
    ("baseline_incomplete", "incomplete"), ("repair_incomplete", "unfinished"),
    ("request_hash", "requests hash"), ("prediction_set", "question set"),
])
def test_refusals(sample, mutation, message):
    pilot = sample["root"] / "results/pilots/pilot"
    diag = sample["diagnosis_dir"]
    repair = sample["repair_dirs"]["retry_retrieval"]
    if mutation == "split":
        change_json(pilot / "selection_record.json", lambda x: x["config"]["selection"].update(question_splits=["val"]))
        (pilot / "evaluation_manifest.json").unlink()
    elif mutation in ("diagnosis_baseline", "diagnosis_hash", "repair_baseline", "repair_hash", "repair_pilot"):
        folder = diag if mutation.startswith("diagnosis") else repair
        key = "baseline_run" if mutation.endswith("baseline") else "pilot_id" if mutation.endswith("pilot") else "baseline_requests_sha256"
        change_json(folder / "run_config.json", lambda x: x.update({key: "wrong"}))
    elif mutation == "diagnosis_set":
        path = diag / "diagnosis.jsonl"
        jsonl(path, read_jsonl(path)[:-1])
    elif mutation == "repair_set":
        update_requests(repair, lambda rows: rows.pop())
    elif mutation == "prediction_set":
        path = repair / "predictions.jsonl"
        jsonl(path, read_jsonl(path)[:-1])
    elif mutation == "flagged_file":
        (diag / "flagged_questions.txt").write_text("q0\n")
    elif mutation == "baseline_incomplete":
        change_json(sample["baseline_dir"] / "run_summary.json", lambda x: x.update(run_state="budget_limited"))
    elif mutation == "repair_incomplete":
        path = repair / "predictions.jsonl"
        rows = read_jsonl(path)
        rows[0].update(status="execution_failed", blocked=False)
        jsonl(path, rows)
    elif mutation == "request_hash":
        change_json(repair / "run_config.json", lambda x: x.update(requests_sha256="wrong"))
    with pytest.raises(compare_round.ComparisonError, match=message):
        run(sample)
    assert not sample["out_dir"].exists()


def test_unknown_resend_cost(sample):
    repair = sample["repair_dirs"]["retry_retrieval"]
    path = repair / "attempts.jsonl"
    events = read_jsonl(path)
    events[:0] = [{"event": "dispatch", "question_id": "q1", "attempt_id": "unknown", "cost_upper_bound": 0.7, "resend": False},
                   {"event": "failed", "question_id": "q1", "attempt_id": "unknown", "outcome": "unknown", "kind": "connection_lost"}]
    next(e for e in events if e.get("attempt_id") == "retry_retrieval-q1" and e["event"] == "dispatch")["resend"] = True
    jsonl(path, events)
    change_json(repair / "run_summary.json", lambda x: x.update(reserved_cost_usd=0.7))
    result = run(sample)
    cost = result["policies"]["faar"]["cost"]["per_question"]["q1"]
    assert cost["reserved"] == 0.7
    assert cost["total"] == pytest.approx(1.0)
    assert result["execution"]["retry_retrieval"]["resends"] == 1
    assert result["execution"]["retry_retrieval"]["unknown_outcomes"] == 1


def test_blocked_scores_zero_and_invalid(sample):
    repair = sample["repair_dirs"]["retry_retrieval"]
    predictions = read_jsonl(repair / "predictions.jsonl")
    predictions[0].update(status="execution_failed", blocked=True, answer="yes", usage=None)
    jsonl(repair / "predictions.jsonl", predictions)
    path = repair / "attempts.jsonl"
    events = read_jsonl(path)
    target = next(e for e in events if e.get("attempt_id") == "retry_retrieval-q0" and e["event"] == "saved")
    target.update(event="failed", outcome="unknown", kind="timeout")
    target.pop("measured_cost")
    jsonl(path, events)
    change_json(repair / "run_summary.json", lambda x: x.update(complete=False, blocked=["q0"], measured_cost_usd=0.4, reserved_cost_usd=0.5))
    result = run(sample)
    assert result["policies"]["faar"]["em"] == 0.5
    assert result["validity"]["valid"] is False
    assert result["validity"]["run_rates_over_2_percent"]["retry_retrieval"] == pytest.approx(1 / 3)
    assert "faar - gated_visual" in result["validity"]["compared_rate_differences_over_1_point"]
    assert result["repair_damage"]["faar"]["damaged"] == 1


def test_score_failures_never_match_empty_gold():
    for status in ("execution_failed", "no_evidence"):
        assert all(value == 0 for value in compare_round.score({"status": status, "answer": ""}, "").values())


def test_cli_writes_outputs(sample, monkeypatch):
    import sys

    monkeypatch.setattr(compare_round, "ROOT", sample["root"])
    monkeypatch.setattr(sys, "argv", ["compare_round.py", "--pilot-id", "pilot", "--round", "1",
                                     "--baseline-run-dir", str(sample["baseline_dir"]), "--diagnosis-dir", str(sample["diagnosis_dir"]),
                                     "--visual-run-dir", str(sample["repair_dirs"]["invoke_vlm"]),
                                     "--retry-run-dir", str(sample["repair_dirs"]["retry_retrieval"]),
                                     "--correct-run-dir", str(sample["repair_dirs"]["correct_text"]), "--out-dir", str(sample["out_dir"])])
    assert compare_round.main() == 0
    assert (sample["out_dir"] / "report.md").is_file()


def test_pooled_random_uses_largest_distinct_p_value():
    rows = []
    for i in range(12):
        row = {"doc_id": f"d{i // 2}"}
        for metric in compare_round.METRICS:
            row[f"faar:{metric}"] = 1
            row[f"random_{RANDOM_SEEDS[0]}:{metric}"] = 0
            row[f"random_{RANDOM_SEEDS[1]}:{metric}"] = 1 if i < 6 else 0
            row[f"random_{RANDOM_SEEDS[2]}:{metric}"] = 1
        rows.append(row)
    per_seed = {name: compare_round.contrast(rows, name, "faar") for name in compare_round.RANDOM}
    pooled = compare_round.pooled_random_contrast(rows, per_seed)
    assert per_seed[compare_round.RANDOM[0]]["mcnemar_em"]["p_value"] < 0.05
    assert pooled["mcnemar_em"]["p_value"] == 1.0
    assert pooled["em"]["by_document"]["diff"] == pytest.approx(0.5)


def test_not_sent_attempt_has_no_api_cost_or_request():
    events = [{"event": "dispatch", "question_id": "q", "attempt_id": "a", "cost_upper_bound": 1},
              {"event": "failed", "question_id": "q", "attempt_id": "a", "outcome": "not_sent"}]
    row = compare_round.ledger_by_question(events, {"q": {}}, {"q": {"selected_pages": [1]}}, baseline=False)["q"]
    assert row["dispatches"] == 1
    assert row["requests"] == row["images"] == row["reserved"] == 0


def test_selection_must_cover_runtime_questions(sample):
    pilot = sample["root"] / "results/pilots/pilot"
    change_json(pilot / "selection_record.json", lambda x: x.update(selected_question_ids=[]))
    with pytest.raises(compare_round.ComparisonError, match="selection question set"):
        run(sample)
