from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.metrics import exact_match, token_f1

REQUIRED_KEYS = ("EM", "F1", "vlm_rate", "harm_rate")


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if "rows" in payload:
        return list(payload["rows"])
    if "examples" in payload:
        return list(payload["examples"])
    if "summary" in payload:
        return []
    return [payload]


def evaluate_results(results_path: Path, baseline_path: Path | None = None) -> dict[str, float]:
    payload = json.loads(results_path.read_text())
    summary = payload.get("summary") if isinstance(payload, dict) else None

    rows = load_rows(results_path)
    baseline_by_id = {}
    if baseline_path is not None and not baseline_path.is_file():
        raise ValueError(f"Baseline result does not exist: {baseline_path}")
    if baseline_path is not None:
        baseline_by_id = {row.get("example_id"): row for row in load_rows(baseline_path)}

    if rows:
        em_values = []
        f1_values = []
        vlm_values = []
        harm_values = []
        for row in rows:
            row_metrics = row.get("metrics") or {}
            prediction = row.get("predicted_answer", row.get("answer", ""))
            gold = row.get("gold_answer", row.get("correct_answer", ""))
            em = float(row_metrics.get("em", exact_match(prediction, gold)))
            f1 = float(row_metrics.get("f1", token_f1(prediction, gold)))
            action = row.get("action_outcome") or {}
            visual = row.get("visual_result") or {}
            used_vlm = action.get("action") == "invoke_vlm" or visual.get("status") == "succeeded"
            baseline = baseline_by_id.get(row.get("example_id"))
            baseline_f1 = None
            if baseline:
                baseline_metrics = baseline.get("metrics") or {}
                baseline_prediction = baseline.get("predicted_answer", baseline.get("answer", ""))
                baseline_gold = baseline.get("gold_answer", baseline.get("correct_answer", ""))
                baseline_f1 = float(
                    baseline_metrics.get("f1", token_f1(baseline_prediction, baseline_gold))
                )
            em_values.append(em)
            f1_values.append(f1)
            vlm_values.append(1.0 if used_vlm else 0.0)
            if baseline_path is not None and baseline_f1 is None:
                raise ValueError(f"Baseline result does not cover example_id={row.get('example_id')!r}.")
            harm_values.append(1.0 if baseline_f1 is not None and f1 < baseline_f1 else 0.0)
        metrics = {
            "EM": mean(em_values),
            "F1": mean(f1_values),
            "vlm_rate": mean(vlm_values),
            "harm_rate": mean(harm_values),
        }
        return {key: round(float(metrics[key]), 4) for key in REQUIRED_KEYS}

    if not isinstance(summary, dict):
        raise ValueError(f"{results_path} contains no evaluable rows and no summary object.")
    if baseline_path is not None:
        raise ValueError("harm_rate cannot be recomputed against a baseline without per-example rows.")
    nullish = [key for key in REQUIRED_KEYS if key not in summary or summary.get(key) is None]
    if nullish:
        raise ValueError(
            f"{results_path} summary is missing or null for required metrics: {', '.join(nullish)}."
        )
    return {key: float(summary[key]) for key in REQUIRED_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FAAR result JSON and emit the four required AAAI metrics.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    metrics = evaluate_results(args.results, baseline_path=args.baseline)
    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
