"""Compare shared development runs without sending requests.

The pilot must record a train-only question selection. All inputs are hashed in
results.json. The oracle uses scores only after the runtime policies are assembled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT
HELPERS = CODE_ROOT / "scripts/analysis/ohr_dev_v1_failures"
sys.path.insert(0, str(HELPERS))

from measures import anls  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import beta, norm  # noqa: E402
from stats import cluster_bootstrap_diff, mcnemar_exact  # noqa: E402

from faar.ohr_scoring import exact_match_score, f1_score  # noqa: E402
from faar.policies import RANDOM_SEEDS, REPAIRS, assemble  # noqa: E402

METRICS = ("em", "f1", "anls", "nfkc_em")
RANDOM = tuple(f"random_{seed}" for seed in RANDOM_SEEDS)
POLICIES = ("no_recovery", "gated_visual", "faar", *RANDOM, "all_retry_retrieval", "all_correct_text", "oracle")


class ComparisonError(ValueError):
    """Input runs cannot support this comparison."""


class Reader:
    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def bytes(self, path: Path) -> bytes:
        data = path.read_bytes()
        self.hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path: Path) -> Any:
        return json.loads(self.bytes(path))

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.bytes(path).decode().splitlines() if line.strip()]


def indexed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {r["question_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ComparisonError(f"duplicate question in {label}")
    return result


def check_set(actual: set[str], expected: set[str], label: str) -> None:
    if actual != expected:
        raise ComparisonError(f"{label} question set differs: missing {len(expected - actual)}, extra {len(actual - expected)}")


def read_inputs(root: Path, pilot_id: str, baseline_dir: Path, diagnosis_dir: Path,
                repair_dirs: dict[str, Path], reader: Reader) -> dict[str, Any]:
    if not pilot_id or Path(pilot_id).name != pilot_id or pilot_id in (".", ".."):
        raise ComparisonError("invalid pilot id")
    pilot = root / "results/pilots" / pilot_id
    selection = reader.json(pilot / "selection_record.json")
    if selection.get("pilot_id") != pilot_id or selection.get("config", {}).get("selection", {}).get("question_splits") != ["train"]:
        raise ComparisonError("pilot selection must record train as its only question split")
    runtime = reader.json(pilot / "runtime_manifest.json")
    if runtime.get("pilot_id") != pilot_id:
        raise ComparisonError("runtime pilot mismatch")
    questions = indexed(runtime["questions"], "runtime")
    if not questions:
        raise ComparisonError("empty pilot")
    ids = set(questions)
    selected = selection.get("selected_question_ids", [])
    if len(selected) != len(ids) or set(selected) != ids:
        raise ComparisonError("pilot selection question set differs")
    baseline_config = reader.json(baseline_dir / "run_config.json")
    summary = reader.json(baseline_dir / "run_summary.json")
    if summary.get("run_state") != "complete" or summary.get("valid_baseline") is not True or summary.get("pilot_id") != pilot_id:
        raise ComparisonError("baseline is incomplete or invalid")
    runtime_sha = reader.hashes[str((pilot / "runtime_manifest.json").resolve())]
    if runtime_sha not in json.dumps(baseline_config):
        raise ComparisonError("baseline runtime manifest mismatch")
    requests = indexed(reader.jsonl(baseline_dir / "requests.jsonl"), "baseline requests")
    predictions = indexed(reader.jsonl(baseline_dir / "predictions.jsonl"), "baseline predictions")
    if summary.get("predictions_sha256") != reader.hashes[str((baseline_dir / "predictions.jsonl").resolve())]:
        raise ComparisonError("baseline predictions hash mismatch")
    check_set(set(requests), ids, "baseline requests")
    check_set(set(predictions), ids, "baseline predictions")
    for qid, request in requests.items():
        if request.get("doc_id") != questions[qid]["doc_id"] or ("question" in request and request["question"] != questions[qid]["question"]):
            raise ComparisonError(f"baseline request mismatch for {qid}")
    if any(p.get("unserved") or p.get("awaiting_reconciliation") or p.get("status") not in ("answered", "no_evidence", "execution_failed") for p in predictions.values()):
        raise ComparisonError("baseline has unfinished predictions")
    events = reader.jsonl(baseline_dir / "attempts.jsonl")
    saved_questions = set()
    for event in events:
        if event.get("event") != "response_saved":
            continue
        qid = event["question_id"]
        response_path = baseline_dir / "responses" / f"{event['attempt_id']}.json"
        response_bytes = response_path.read_bytes()
        response = json.loads(response_bytes)
        if (hashlib.sha256(response_bytes).hexdigest() != event.get("response_sha256")
                or response.get("question_id") != qid or response.get("attempt_id") != event["attempt_id"]):
            raise ComparisonError("baseline response hash or identity mismatch")
        if any(event.get(k) != response.get(k) for k in ("answer", "abstained", "output_status", "usage", "measured_cost", "returned_model", "returned_service_tier")):
            raise ComparisonError("baseline event differs from response")
        if qid in saved_questions or qid not in predictions or predictions[qid].get("status") != "answered" or predictions[qid].get("answer") != response.get("answer"):
            raise ComparisonError("baseline prediction differs from response")
        saved_questions.add(qid)
    if saved_questions != {q for q, row in predictions.items() if row["status"] == "answered"}:
        raise ComparisonError("baseline answered question has no saved response")
    baseline_hash = reader.hashes[str((baseline_dir / "requests.jsonl").resolve())]
    diag_config = reader.json(diagnosis_dir / "run_config.json")
    diagnosis = indexed(reader.jsonl(diagnosis_dir / "diagnosis.jsonl"), "diagnosis")
    check_set(set(diagnosis), ids, "diagnosis")
    for qid, d in diagnosis.items():
        if not isinstance(d.get("flagged"), bool) or d.get("doc_id") != questions[qid]["doc_id"] or d.get("route") not in REPAIRS or not isinstance(d.get("gate_reasons"), list):
            raise ComparisonError(f"invalid diagnosis for {qid}")
    flagged = {q for q, d in diagnosis.items() if d["flagged"]}
    flagged_file = diagnosis_dir / "flagged_questions.txt"
    if flagged_file.exists():
        flagged_lines = reader.bytes(flagged_file).decode().splitlines()
        listed = [line.strip() for line in flagged_lines if line.strip()]
        if len(listed) != len(set(listed)):
            raise ComparisonError("duplicate flagged question")
        check_set(set(listed), flagged, "flagged file")
    def identity(config: dict[str, Any], name: str) -> None:
        if config.get("pilot_id") != pilot_id or config.get("baseline_run") != baseline_dir.name or config.get("baseline_requests_sha256") != baseline_hash:
            raise ComparisonError(f"{name} baseline identity mismatch")
        if config.get("runtime_manifest_sha256") != runtime_sha:
            raise ComparisonError(f"{name} runtime manifest mismatch")
    identity(diag_config, "diagnosis")
    runs = {}
    for repair, directory in repair_dirs.items():
        config = reader.json(directory / "run_config.json")
        identity(config, repair)
        if config.get("repair") != repair:
            raise ComparisonError(f"wrong repair in {directory}")
        records_list = reader.jsonl(directory / "requests.jsonl")
        canonical = hashlib.sha256(json.dumps(records_list, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if config.get("requests_sha256") != canonical:
            raise ComparisonError(f"{repair} requests hash mismatch")
        records = indexed(records_list, f"{repair} requests")
        rows = indexed(reader.jsonl(directory / "predictions.jsonl"), f"{repair} predictions")
        check_set(set(records), flagged, f"{repair} requests")
        check_set(set(rows), flagged, f"{repair} predictions")
        run_summary = reader.json(directory / "run_summary.json")
        blocked = set(run_summary["blocked"])
        if not blocked <= flagged:
            raise ComparisonError(f"{repair} has unknown blocked questions")
        for qid, row in rows.items():
            if records[qid].get("doc_id") != questions[qid]["doc_id"]:
                raise ComparisonError(f"{repair} document mismatch for {qid}")
            status = row.get("status")
            if status == "execution_failed" and (qid not in blocked or not row.get("blocked")):
                raise ComparisonError(f"{repair} has unfinished question {qid}")
            if status not in ("answered", "skipped", "execution_failed"):
                raise ComparisonError(f"{repair} has invalid status")
            if (status == "skipped") != (records[qid].get("action", "send") == "skip"):
                raise ComparisonError(f"{repair} skip does not match request")
            if status == "skipped" and (not row.get("skip_reason") or row["skip_reason"] != records[qid].get("skip_reason")):
                raise ComparisonError(f"{repair} skip reason does not match request")
        if blocked != {q for q, row in rows.items() if row["status"] == "execution_failed"}:
            raise ComparisonError(f"{repair} blocked set mismatch")
        event_path = directory / "attempts.jsonl"
        repair_events = reader.jsonl(event_path) if event_path.exists() else []
        for event in repair_events:
            if event["event"] != "saved":
                continue
            response_path = directory / "responses" / f"{event['attempt_id']}.json"
            response = reader.json(response_path)
            if reader.hashes[str(response_path.resolve())] != event["response_sha256"]:
                raise ComparisonError(f"{repair} response hash mismatch")
            if response.get("question_id") != event["question_id"] or response.get("attempt_id") != event["attempt_id"]:
                raise ComparisonError(f"{repair} response identity mismatch")
            prediction = rows[event["question_id"]]
            if prediction["status"] != "answered" or prediction.get("answer") != response.get("answer") or prediction.get("measured_cost") != response.get("measured_cost") or event.get("measured_cost") != response.get("measured_cost") or prediction.get("usage") != response.get("usage"):
                raise ComparisonError(f"{repair} prediction differs from response")
            event["usage"] = response.get("usage")
            event["latency_ms"] = response.get("latency_ms")
        runs[repair] = {"config": config, "summary": run_summary, "records": records, "rows": rows, "events": repair_events}
    evaluation = reader.json(pilot / "evaluation_manifest.json")
    if evaluation.get("pilot_id") != pilot_id:
        raise ComparisonError("evaluation pilot mismatch")
    check_set(set(evaluation["questions"]), ids, "evaluation")
    for code in (Path(__file__), CODE_ROOT / "src/faar/policies.py", CODE_ROOT / "src/faar/ohr_scoring.py", HELPERS / "stats.py", HELPERS / "measures.py"):
        reader.bytes(code)
    return {"questions": questions, "evaluation": evaluation["questions"], "diagnosis": diagnosis,
            "baseline": {"config": baseline_config, "summary": summary, "requests": requests, "rows": predictions, "events": events},
            "repairs": runs, "flagged": flagged}


def score(row: dict[str, Any], gold: str) -> dict[str, float]:
    if row["status"] in ("execution_failed", "no_evidence"):
        return dict.fromkeys(METRICS, 0.0)
    answer = row.get("answer") or ""
    return {"em": exact_match_score(answer, gold), "f1": f1_score(answer, gold), "anls": anls(answer, gold),
            "nfkc_em": exact_match_score(unicodedata.normalize("NFKC", answer), unicodedata.normalize("NFKC", gold))}


def cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for qid, q in data["questions"].items():
        d = data["diagnosis"][qid]
        initial = data["baseline"]["rows"][qid]
        repair_rows = {name: run["rows"].get(qid) for name, run in data["repairs"].items()}
        policies = assemble(qid, initial, d["flagged"], d["route"], repair_rows)
        gold = data["evaluation"][qid]["answers"]
        # max is stable: the initial answer wins ties, followed by REPAIRS order.
        candidates = [(initial, "initial")]
        if d["flagged"]:
            for name in REPAIRS:
                r = repair_rows[name]
                candidates.append((initial, f"kept_initial:skip:{r['skip_reason']}") if r["status"] == "skipped" else (r, f"repair:{name}"))
        policies["oracle"] = max(candidates, key=lambda item: score(item[0], gold)["em"])
        row: dict[str, Any] = {"question_id": qid, "doc_id": q["doc_id"], "flagged": d["flagged"],
                               "gate_reasons": d["gate_reasons"], "diagnosis": d["diagnosis"], "route": d["route"]}
        for name, (answer, source) in policies.items():
            row.update({f"{name}:source": source, f"{name}:status": answer["status"], f"{name}:answer": answer.get("answer") or ""})
            row.update({f"{name}:{m}": v for m, v in score(answer, gold).items()})
        # The all-repairs diagnostic needs visual even when another route was chosen.
        for name in REPAIRS:
            repair = repair_rows[name]
            effective = initial if not d["flagged"] or repair["status"] == "skipped" else repair
            row[f"diagnostic:{name}:em"] = score(effective, gold)["em"] if d["flagged"] else None
        result.append(row)
    return result


def ledger_by_question(events: list[dict[str, Any]], rows: dict[str, dict[str, Any]],
                       records: dict[str, dict[str, Any]], *, baseline: bool) -> dict[str, dict[str, Any]]:
    result = {qid: {"measured": 0.0, "reserved": 0.0, "dispatches": 0, "requests": 0, "text_requests": 0,
                    "image_requests": 0, "images": 0, "input_tokens": 0, "output_tokens": 0,
                    "usage_records": 0, "latency_ms": 0, "latency_records": 0, "unknown_outcomes": 0, "resends": 0} for qid in rows}
    dispatch_name = "dispatch_started" if baseline else "dispatch"
    saved_name = "response_saved" if baseline else "saved"
    terminal_names = (saved_name, "attempt_failed", "outcome_unknown") if baseline else (saved_name, "failed")
    starts: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["event"] == dispatch_name:
            aid = e["attempt_id"]
            if aid in starts or e["question_id"] not in rows:
                raise ComparisonError("invalid dispatch in ledger")
            starts[aid] = e
        elif e["event"] in terminal_names:
            aid = e["attempt_id"]
            if aid not in starts or aid in terminals or e["question_id"] != starts[aid]["question_id"]:
                raise ComparisonError("invalid terminal event in ledger")
            terminals[aid] = e
        elif baseline and e["event"] == "reconciled" and e.get("resolution") == "allow_new_attempt":
            result[e["question_id"]]["resends"] += 1
    for aid, start in starts.items():
        qid = start["question_id"]
        entry = result[qid]
        end = terminals.get(aid)
        image_count = len(records[qid].get("selected_pages", [])) if not baseline else 0
        entry["dispatches"] += 1
        not_sent = end and end["event"] in ("attempt_failed", "failed") and end.get("outcome") == "not_sent"
        if not not_sent:
            entry["requests"] += 1
            entry["image_requests" if image_count else "text_requests"] += 1
            entry["images"] += image_count
        if not baseline and start.get("resend"):
            entry["resends"] += 1
        if end and end["event"] == saved_name and end.get("measured_cost") is not None:
            entry["measured"] += end["measured_cost"]
        elif not (end and end["event"] in ("attempt_failed", "failed") and end.get("outcome") == "not_sent"):
            entry["reserved"] += start["cost_upper_bound"]
        if end is None or end["event"] == "outcome_unknown" or (end["event"] == "failed" and end.get("outcome") != "not_sent"):
            entry["unknown_outcomes"] += 1
        if end and end.get("latency_ms") is not None:
            entry["latency_ms"] += end["latency_ms"]
            entry["latency_records"] += 1
        if end and end["event"] == saved_name:
            usage = end.get("usage")
            if usage:
                entry["usage_records"] += 1
                entry["input_tokens"] += usage.get("input_tokens") or 0
                entry["output_tokens"] += usage.get("output_tokens") or 0
    return result


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def interval(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    return [0.0 if not k else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]


def contrast(rows: list[dict[str, Any]], base: str, condition: str) -> dict[str, Any]:
    docs = [r["doc_id"] for r in rows]
    result: dict[str, Any] = {"base": base, "condition": condition, "n": len(rows)}
    for metric in METRICS:
        b = [r[f"{base}:{metric}"] for r in rows]
        c = [r[f"{condition}:{metric}"] for r in rows]
        result[metric] = {"base": mean(b), "condition": mean(c),
                          "by_document": cluster_bootstrap_diff(docs, b, c),
                          "by_question": cluster_bootstrap_diff(docs, b, c, by_cluster=False)}
    result["mcnemar_em"] = mcnemar_exact([r[f"{base}:em"] for r in rows], [r[f"{condition}:em"] for r in rows])
    return result


def pooled_random_contrast(rows: list[dict[str, Any]], per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pooled_rows = [{**r, **{f"random_pooled:{m}": mean([r[f"{p}:{m}"] for p in RANDOM]) for m in METRICS}} for r in rows]
    result = contrast(pooled_rows, "random_pooled", "faar")
    result["mcnemar_em"] = {"p_value": max(per_seed[p]["mcnemar_em"]["p_value"] for p in RANDOM),
                             "rule": "maximum of the three per-seed exact McNemar p-values"}
    return result


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ranked = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(ranked):
        running = max(running, min(1.0, (len(ranked) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def detectable_gain(n: int, discordant: float) -> float | None:
    if discordant == 0:
        return None
    za, zb = norm.ppf(0.975), norm.ppf(0.8)
    def needed(delta: float) -> float:
        return (za * math.sqrt(discordant) + zb * math.sqrt(discordant - delta**2)) ** 2 / delta**2
    upper = min(discordant, math.sqrt(discordant))
    if needed(upper) > n:
        return None
    return float(brentq(lambda delta: needed(delta) - n, upper * 1e-10, upper))


def analyse(data: dict[str, Any], rows: list[dict[str, Any]], pilot_id: str, round_number: int) -> dict[str, Any]:
    n = len(rows)
    flagged = [r for r in rows if r["flagged"]]
    out: dict[str, Any] = {"pilot_id": pilot_id, "round": round_number, "questions": n,
                           "documents": len({r["doc_id"] for r in rows}), "flagged": len(flagged),
                           "bootstrap": {"resamples": 10000, "seed": 20261031, "confidence": 0.95}, "familywise_alpha": 0.05,
                           "policies": {}, "contrasts": {}, "repair_damage": {}, "diagnosis_counts": {}, "routes": {}, "skips": {}}
    for policy in POLICIES:
        out["policies"][policy] = {"status_counts": dict(Counter(r[f"{policy}:status"] for r in rows)),
                                   **{m: mean([r[f"{policy}:{m}"] for r in rows]) for m in METRICS}}
        wrong = [r for r in flagged if not r["no_recovery:em"]]
        right = [r for r in flagged if r["no_recovery:em"]]
        repaired = sum(r[f"{policy}:em"] for r in wrong)
        damaged = sum(1 - r[f"{policy}:em"] for r in right)
        out["repair_damage"][policy] = {"wrong_at_first": len(wrong), "repaired": repaired,
                                         "repair_rate": repaired / len(wrong) if wrong else None, "repair_ci": interval(repaired, len(wrong)),
                                         "right_at_first": len(right), "damaged": damaged,
                                         "damage_rate": damaged / len(right) if right else None, "damage_ci": interval(damaged, len(right))}
    pairs = {"primary": ("gated_visual", "faar"), "no_recovery": ("no_recovery", "faar"),
             **{name: (name, "faar") for name in RANDOM},
             **{f"descriptive_{name}": ("no_recovery", name) for name in ("gated_visual", "all_retry_retrieval", "all_correct_text", "oracle")}}
    for name, (base, cond) in pairs.items():
        out["contrasts"][name] = contrast(rows, base, cond)
    # Pooled random values are question-level means; only the p-value is pooled conservatively.
    out["contrasts"]["random_pooled"] = pooled_random_contrast(rows, out["contrasts"])
    adjusted = holm({k: out["contrasts"][k]["mcnemar_em"]["p_value"] for k in ("primary", "random_pooled", "no_recovery")})
    for name in pairs:
        out["contrasts"][name]["label"] = "descriptive, unadjusted" if name.startswith("descriptive_") else "family member" if name in ("primary", "no_recovery") else "per-seed, unadjusted"
    out["contrasts"]["random_pooled"]["label"] = "family member"
    for name, value in adjusted.items():
        out["contrasts"][name]["holm_adjusted_p"] = value
    primary = out["contrasts"]["primary"]
    out["decision"] = {"label": "development result, not a confirmation",
                       "rule": "FAAR beat the gated visual repair only if the Holm-adjusted McNemar p is below 0.05 and the document-level interval lies above 0.",
                       "met": adjusted["primary"] < 0.05 and primary["em"]["by_document"]["ci_low"] > 0}
    out["diagnosis_counts"] = dict(Counter(r["diagnosis"] for r in flagged))
    for route in REPAIRS:
        subset = [r for r in flagged if r["route"] == route]
        out["routes"][route] = {"n": len(subset), "initial_em": mean([r["no_recovery:em"] for r in subset]),
                                 **{f"{p}_em": mean([r[f"diagnostic:{p}:em"] for r in subset]) for p in REPAIRS}}
    for repair, run in data["repairs"].items():
        out["skips"][repair] = dict(Counter(r["skip_reason"] for r in run["rows"].values() if r["status"] == "skipped"))
    discordant = primary["mcnemar_em"]["discordant"] / n
    out["detectable"] = {"disagreement_rate": discordant, "gain_80_percent_power": detectable_gain(n, discordant),
                         "formula": "n = (z(0.975)*sqrt(d) + z(0.8)*sqrt(d - delta^2))^2 / delta^2; d is disagreement rate"}
    add_costs_and_execution(out, data, rows)
    return out


def add_costs_and_execution(out: dict[str, Any], data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    baseline = data["baseline"]
    costs = {"initial": ledger_by_question(baseline["events"], baseline["rows"], baseline["requests"], baseline=True)}
    for repair, run in data["repairs"].items():
        costs[repair] = ledger_by_question(run["events"], run["rows"], run["records"], baseline=False)
    for label, run in [("initial", baseline), *data["repairs"].items()]:
        entries = costs[label]
        summary = run["summary"]
        actual = (summary["cost"]["measured"], summary["cost"]["reserved"]) if label == "initial" else (summary["measured_cost_usd"], summary["reserved_cost_usd"])
        observed = (sum(x["measured"] for x in entries.values()), sum(x["reserved"] for x in entries.values()))
        if any(abs(a - b) > 0.000002 * max(1, len(entries)) for a, b in zip(actual, observed)):
            raise ComparisonError(f"{label} ledger cost does not match summary")
        for qid, row in run["rows"].items():
            if row["status"] == "answered" and entries[qid]["dispatches"] == 0:
                raise ComparisonError(f"{label}: answered question has no dispatch")
            if label != "initial" and row["status"] == "answered" and not any(e["event"] == "saved" and e["question_id"] == qid for e in run["events"]):
                raise ComparisonError(f"{label}: answered question has no saved response")
            if label != "initial" and row["status"] == "execution_failed" and not entries[qid]["unknown_outcomes"]:
                raise ComparisonError(f"{label}: blocked question has no unknown outcome")
            if row["status"] == "skipped" and entries[qid]["dispatches"]:
                raise ComparisonError(f"{label}: skipped question has a dispatch")
    keys = tuple(next(iter(costs["initial"].values())))
    out["execution"] = {}
    blocked_sets = {"initial": {qid for qid, row in baseline["rows"].items() if row["status"] == "execution_failed" and costs["initial"][qid]["unknown_outcomes"]}}
    blocked_sets.update({name: set(run["summary"]["blocked"]) for name, run in data["repairs"].items()})
    for name, run in [("initial", baseline), *data["repairs"].items()]:
        entries = costs[name]
        blocked = len(blocked_sets[name])
        denominator = len(run["rows"])
        out["execution"][name] = {"questions": denominator, "unknown_outcomes": sum(v["unknown_outcomes"] for v in entries.values()),
                                  "questions_with_unknown_outcomes": sum(v["unknown_outcomes"] > 0 for v in entries.values()),
                                  "resends": sum(v["resends"] for v in entries.values()), "blocked": blocked,
                                  "blocked_rate": blocked / denominator if denominator else 0.0}
    for policy in POLICIES:
        per_question = {}
        for r in rows:
            qid = r["question_id"]
            value = dict(costs["initial"][qid])
            source = r[f"{policy}:source"]
            if source.startswith("repair:"):
                repair = source.split(":", 1)[1]
                for key in keys:
                    value[key] += costs[repair][qid][key]
            value["total"] = value["measured"] + value["reserved"]
            per_question[qid] = value
        totals = {k: sum(v[k] for v in per_question.values()) for k in (*keys, "total")}
        out["policies"][policy]["cost"] = {"sum": totals, "per_question_mean": {k: v / len(rows) for k, v in totals.items()},
                                             "per_question": per_question,
                                             "local_processing_time": None, "one_time_preparation_cost": None,
                                             "image_settings": data["repairs"]["invoke_vlm"]["config"].get("render")}
    visual = out["policies"]["gated_visual"]["cost"]["sum"]["total"]
    faar = out["policies"]["faar"]["cost"]["sum"]["total"]
    ratio = faar / visual if visual else None
    out["cost_rule"] = {"faar_to_gated_visual": ratio,
                        "label": None if ratio is None else "similar cost" if ratio <= 1.10 else "higher EM at higher cost"}
    run_failures = {name: e["blocked_rate"] for name, e in out["execution"].items() if e["blocked_rate"] > 0.02}
    def policy_blocked(row: dict[str, Any], policy: str) -> bool:
        source = row[f"{policy}:source"]
        run = source.split(":", 1)[1] if source.startswith("repair:") else "initial"
        return row["question_id"] in blocked_sets[run]
    policy_rates = {p: sum(policy_blocked(r, p) for r in rows) / len(rows) for p in POLICIES}
    compared = [("faar", "gated_visual"), ("faar", "no_recovery"), *(("faar", p) for p in RANDOM),
                *((p, "no_recovery") for p in ("gated_visual", "all_retry_retrieval", "all_correct_text", "oracle"))]
    differences = {f"{a} - {b}": abs(policy_rates[a] - policy_rates[b]) for a, b in compared
                   if abs(policy_rates[a] - policy_rates[b]) > 0.01 + 1e-12}
    out["validity"] = {"valid": not run_failures and not differences, "run_rates_over_2_percent": run_failures,
                       "policy_blocked_rates": policy_rates, "compared_rate_differences_over_1_point": differences}


def report(results: dict[str, Any]) -> str:
    lines = [f"# {results['pilot_id']} round {results['round']} policy comparison", "",
             f"{results['questions']} questions on {results['documents']} documents; {results['flagged']} flagged.",
             "", "## Development decision", "", results["decision"]["label"], results["decision"]["rule"],
             f"Rule met: {results['decision']['met']}. Valid comparison: {results['validity']['valid']}.", "",
             "## Policy scores and costs", "", "| Policy | EM | F1 | ANLS | NFKC EM | Measured USD | Reserved USD | Total USD | Measured / question | Reserved / question | Total / question | Requests | Text | Image requests | Images | Input tokens | Output tokens | Usage records | Latency ms | Latency records |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, p in results["policies"].items():
        c = p["cost"]["sum"]
        lines.append(f"| {name} | {p['em']:.6f} | {p['f1']:.6f} | {p['anls']:.6f} | {p['nfkc_em']:.6f} | {c['measured']:.6f} | {c['reserved']:.6f} | {c['total']:.6f} | {c['measured']/results['questions']:.6f} | {c['reserved']/results['questions']:.6f} | {c['total']/results['questions']:.6f} | {c['requests']} | {c['text_requests']} | {c['image_requests']} | {c['images']} | {c['input_tokens']} | {c['output_tokens']} | {c['usage_records']} | {c['latency_ms']} | {c['latency_records']} |")
    lines += ["", f"Image settings: {json.dumps(results['policies']['gated_visual']['cost']['image_settings'], sort_keys=True)}. Local processing time and one-time preparation cost: not recorded.", "", f"FAAR / gated visual total cost per question: {results['cost_rule']['faar_to_gated_visual']}; {results['cost_rule']['label']}.",
              "", "## Contrasts", "", "95% bootstrap intervals use 10,000 resamples and seed 20261031. Random pooled uses the mean score per question; its McNemar p-value is the maximum of the three seed p-values.",
              "", "| Contrast | EM difference | Document interval | Question interval | McNemar p | Holm p |",
              "| --- | ---: | --- | --- | ---: | ---: |"]
    for name, c in results["contrasts"].items():
        d, q = c["em"]["by_document"], c["em"]["by_question"]
        suffix = " (descriptive, unadjusted)" if name.startswith("descriptive_") else ""
        lines.append(f"| {name}{suffix} | {d['diff']:.6f} | [{d['ci_low']:.6f}, {d['ci_high']:.6f}] | [{q['ci_low']:.6f}, {q['ci_high']:.6f}] | {c['mcnemar_em']['p_value']:.6f} | {c.get('holm_adjusted_p', 'n/a')} |")
    lines += ["", "## Secondary score contrasts", "", "| Contrast | Metric | Difference | Document interval | Question interval |", "| --- | --- | ---: | --- | --- |"]
    for name, c in results["contrasts"].items():
        for metric in ("f1", "anls", "nfkc_em"):
            d, q = c[metric]["by_document"], c[metric]["by_question"]
            lines.append(f"| {name} | {metric} | {d['diff']:.6f} | [{d['ci_low']:.6f}, {d['ci_high']:.6f}] | [{q['ci_low']:.6f}, {q['ci_high']:.6f}] |")
    lines += ["", "## Flagged questions: repair and damage", "", "95% Clopper-Pearson intervals.", "",
              "| Policy | Repaired / initially wrong | Rate [interval] | Damaged / initially right | Rate [interval] |",
              "| --- | --- | --- | --- | --- |"]
    for p, x in results["repair_damage"].items():
        lines.append(f"| {p} | {x['repaired']} / {x['wrong_at_first']} | {x['repair_rate']} {x['repair_ci']} | {x['damaged']} / {x['right_at_first']} | {x['damage_rate']} {x['damage_ci']} |")
    lines += ["", "## Diagnoses and route scores", "", f"Diagnosis counts: {json.dumps(results['diagnosis_counts'], sort_keys=True)}.", "",
              "| Route | n | Initial EM | Retry EM | Correct EM | Visual EM |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for route, x in results["routes"].items():
        lines.append(f"| {route} | {x['n']} | {x['initial_em']} | {x['retry_retrieval_em']} | {x['correct_text_em']} | {x['invoke_vlm_em']} |")
    lines += ["", f"Skip reasons: {json.dumps(results['skips'], sort_keys=True)}.", "", "## Execution", "",
              "| Run | Questions | Unknown outcomes | Questions with unknown outcomes | Resends | Blocked | Blocked rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, x in results["execution"].items():
        lines.append(f"| {name} | {x['questions']} | {x['unknown_outcomes']} | {x['questions_with_unknown_outcomes']} | {x['resends']} | {x['blocked']} | {x['blocked_rate']:.6f} |")
    lines += ["", f"Rates over 2%: {results['validity']['run_rates_over_2_percent']}; compared rate differences over 1 point: {results['validity']['compared_rate_differences_over_1_point']}.",
              "", "## Disagreement and power", "", f"Disagreement rate: {results['detectable']['disagreement_rate']:.6f}. Detectable gain at 80% power: {results['detectable']['gain_80_percent_power']}.",
              results["detectable"]["formula"], ""]
    return "\n".join(lines)


def compare(root: Path, pilot_id: str, round_number: int, baseline_dir: Path, diagnosis_dir: Path,
            repair_dirs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ComparisonError("output directory is not empty")
    if round_number < 1:
        raise ComparisonError("round must be positive")
    reader = Reader()
    data = read_inputs(root, pilot_id, baseline_dir, diagnosis_dir, repair_dirs, reader)
    rows = cases(data)
    results = analyse(data, rows, pilot_id, round_number)
    results["input_and_code_sha256"] = reader.hashes
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "case_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(report(results), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    for name in ("baseline-run-dir", "diagnosis-dir", "visual-run-dir", "retry-run-dir", "correct-run-dir", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare(ROOT, args.pilot_id, args.round, args.baseline_run_dir, args.diagnosis_dir,
                {"invoke_vlm": args.visual_run_dir, "retry_retrieval": args.retry_run_dir, "correct_text": args.correct_run_dir}, args.out_dir)
    except (ComparisonError, OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"comparison refused: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
