"""Choose one confidence condition on a frozen train pilot, using no-recovery EM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

from faar.diagnosis import CONFIDENCE_FEATURES
from faar.ohr_scoring import exact_match_score

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {r["question_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question in {path}")
    return result


def choose(root, pilot_id, baseline, diagnosis, out):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output directory is not empty")
    pilot = root / "results" / "pilots" / pilot_id
    record = read_json(pilot / "selection_record.json")
    selection = record["config"]["selection"]
    if record["pilot_id"] != pilot_id or not (
        selection.get("question_splits") == ["train"]
        or (selection.get("question_splits") is None and selection.get("pool", "train_exclusive") == "train_exclusive")
    ):
        raise ValueError("pilot's recorded split is not train")
    config_path = diagnosis / "diagnosis_config.json"
    config = read_json(config_path)
    if config["gate_rule"]["feature"] is not None or config["gate_rule"]["threshold"] is not None:
        raise ValueError("diagnosis must use the base gate")
    if (config["pilot_id"] != pilot_id or config["baseline_run"] != baseline.name
            or config["baseline_requests_sha256"] != digest(baseline / "requests.jsonl")):
        raise ValueError("diagnosis baseline or pilot mismatch")
    predictions_path = baseline / "predictions.jsonl"
    diagnosis_path = diagnosis / "diagnosis.jsonl"
    evaluation_path = pilot / "evaluation_manifest.json"
    predictions = indexed(predictions_path)
    diagnoses = indexed(diagnosis_path)
    evaluation = read_json(evaluation_path)
    gold = evaluation["questions"]
    ids = set(record["selected_question_ids"])
    if evaluation["pilot_id"] != pilot_id or not ids or len(ids) != len(record["selected_question_ids"]) or ids != predictions.keys() or ids != diagnoses.keys() or ids != gold.keys():
        raise ValueError("pilot, predictions and diagnosis questions differ")
    positives = {qid: p["status"] in ("execution_failed", "no_evidence") or
                 exact_match_score(p["answer"] or "", gold[qid]["answers"]) == 0
                 for qid, p in predictions.items()}
    candidates = []
    populations = {}
    undefined = []
    for feature in CONFIDENCE_FEATURES:
        population = []
        for qid, row in diagnoses.items():
            value = row["features"][feature]
            if row["flagged"] is False and value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("non-finite feature value")
                population.append((value, positives[qid]))
        n = len(population)
        pos = sum(p for _, p in population)
        neg = n - pos
        cap = 3 * n // 10
        populations[feature] = {"N": n, "cap": cap, "positives": pos, "negatives": neg}
        if not pos or not neg:
            undefined.append(feature)
        for t in sorted({v for v, _ in population}):
            flagged = [p for v, p in population if v < t]
            tp = sum(flagged)
            fp = len(flagged) - tp
            tpr = Fraction(tp, pos) if pos else None
            fpr = Fraction(fp, neg) if neg else None
            j = tpr - fpr if tpr is not None and fpr is not None else None
            candidates.append(dict(feature=feature, t=t, flagged=len(flagged), TP=tp, FP=fp,
                                   TPR=tpr, FPR=fpr, J=j, within_cap=len(flagged) <= cap))
    eligible = [c for c in candidates if c["within_cap"] and c["J"] is not None]
    best = min(eligible, key=lambda c: (-c["J"], c["flagged"], CONFIDENCE_FEATURES.index(c["feature"]), c["t"])) if eligible and not undefined else None
    if best and best["J"] <= 0:
        best = None
    reason = ("J undefined: population has no positives or no negatives for " + ", ".join(undefined)
              if undefined else (None if best else "no candidate has J above zero"))
    def number(value):
        return float(value) if isinstance(value, Fraction) else value
    chosen = {k: number(best[k]) for k in ("feature", "t", "flagged", "TP", "FP", "TPR", "FPR", "J")} if best else None
    stats = populations[best["feature"]] if best else None
    args = f"--gate-feature {best['feature']} --gate-threshold {repr(best['t'])}" if best else ""
    result = {"rule_id": "youden-j-cap30-v1", "feature": best["feature"] if best else None,
              "threshold": best["t"] if best else None, "J": chosen["J"] if chosen else None,
              "TPR": chosen["TPR"] if chosen else None, "FPR": chosen["FPR"] if chosen else None,
              "flagged": chosen["flagged"] if chosen else None,
              "N": stats["N"] if stats else None, "cap": stats["cap"] if stats else None,
              "positives": stats["positives"] if stats else None, "negatives": stats["negatives"] if stats else None,
              "population": stats,
              "populations": populations, "reason": reason, "gate_arguments": args,
              "population_note": "N differs by feature only when null values differ; current features are null together.",
              "sha256": {str(p): digest(p) for p in (predictions_path, diagnosis_path, config_path, evaluation_path, Path(__file__), ROOT / "src/faar/diagnosis.py")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_choice.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "t", "flagged", "TP", "FP", "TPR", "FPR", "J", "within_cap"])
        writer.writeheader()
        writer.writerows({k: number(v) for k, v in row.items()} for row in candidates)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = choose(ROOT, args.pilot_id, args.baseline_run_dir, args.diagnosis_dir, args.out_dir)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps({k: result[k] for k in ("feature", "threshold", "J", "reason", "gate_arguments")}, indent=2))


if __name__ == "__main__":
    main()
