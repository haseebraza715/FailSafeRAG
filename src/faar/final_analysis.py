from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .metrics import token_f1
from .results_aggregator import summarize_api_usage


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("summary"), dict)
        or not isinstance(payload.get("rows"), list)
    ):
        raise ValueError(f"{path} is not a complete FAAR result JSON with summary and rows.")
    return payload


_RUN_SPEC_MATCH_KEYS = (
    "dataset",
    "split",
    "seed",
    "max_examples",
    "shard_index",
    "num_shards",
    "embedding_model",
    "reranker",
    "ocr_engine",
    "model_provenance",
    "manifest_sha256",
)

_VLM_MATCH_KEYS = ("vlm_backend", "vlm_model", "vlm_cost_rates")


def _require_unique_example_ids(rows: list[dict[str, Any]], path: Path) -> None:
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a row that is not an object.")
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"{path} contains a row with a missing example_id.")
        if example_id in seen:
            raise ValueError(f"{path} contains duplicate example_id {example_id!r}.")
        if (row.get("action_outcome") or {}).get("status") == "failed" or row.get("error"):
            raise ValueError(f"{path} contains failed row {example_id!r}; failed rows cannot be analyzed.")
        seen.add(example_id)


def _validate_analysis_inputs(baseline_path: Path, result_paths: list[Path]) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("Final analysis requires at least one result JSON.")
    baseline = load_result(baseline_path)
    if baseline.get("profile") != "naive_rag":
        raise ValueError(f"Baseline must be a naive_rag profile result: {baseline_path}")
    baseline_spec = baseline.get("run_spec")
    if not isinstance(baseline_spec, dict):
        raise ValueError(f"{baseline_path} has no run_spec provenance.")
    baseline_rows = list(baseline["rows"])
    if not baseline_rows:
        raise ValueError(f"{baseline_path} contains no baseline rows.")
    _require_unique_example_ids(baseline_rows, baseline_path)
    baseline_ids = {str(row["example_id"]).strip() for row in baseline_rows}
    vlm_signatures: list[tuple[Any, ...]] = []
    for result_path in result_paths:
        result = load_result(result_path)
        run_spec = result.get("run_spec")
        if not isinstance(run_spec, dict):
            raise ValueError(f"{result_path} has no run_spec provenance.")
        for key in _VLM_MATCH_KEYS:
            if key not in run_spec:
                raise ValueError(f"{result_path} run_spec is missing {key}.")
        vlm_signatures.append(tuple(run_spec[key] for key in _VLM_MATCH_KEYS))
        for key in _RUN_SPEC_MATCH_KEYS:
            if key not in baseline_spec:
                if key == "manifest_sha256":
                    continue
                raise ValueError(f"{baseline_path} run_spec is missing {key}.")
            if key not in run_spec:
                if key == "manifest_sha256":
                    continue
                raise ValueError(f"{result_path} run_spec is missing {key}.")
            if run_spec[key] != baseline_spec[key]:
                raise ValueError(f"{result_path} run_spec.{key} does not match baseline run_spec.")
        result_rows = list(result["rows"])
        if not result_rows:
            raise ValueError(f"{result_path} contains no result rows.")
        _require_unique_example_ids(result_rows, result_path)
        result_ids = {str(row["example_id"]).strip() for row in result_rows}
        uncovered = sorted(result_ids - baseline_ids)
        if uncovered:
            raise ValueError(f"{result_path} rows not covered by baseline B0: {uncovered}")
        omitted = sorted(baseline_ids - result_ids)
        if omitted:
            raise ValueError(f"{result_path} omits baseline B0 rows: {omitted}")
    if any(signature != vlm_signatures[0] for signature in vlm_signatures):
        raise ValueError(
            "Final analysis requires matching vlm_backend, vlm_model, and vlm_cost_rates "
            "across all non-B0 results so providers/models/rates cannot be mixed."
        )
    return baseline


def bootstrap_ci(values: Iterable[float], samples: int = 10_000, seed: int = 42) -> dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {
        "mean": round(mean(values), 4),
        "lower": round(estimates[int(0.025 * (samples - 1))], 4),
        "upper": round(estimates[int(0.975 * (samples - 1))], 4),
    }


def _harm_rate(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> float:
    baseline_by_id = {str(row.get("example_id")): row for row in baseline_rows}
    uncovered = [row for row in rows if str(row.get("example_id")) not in baseline_by_id]
    if uncovered:
        ids = sorted({str(row.get("example_id")) for row in uncovered})
        raise ValueError(f"Harm analysis cannot compare rows not covered by baseline B0: {ids}")
    if not rows:
        return 0.0
    harmed = 0.0
    for row in rows:
        row_metrics = row.get("metrics") or {}
        prediction = row.get("predicted_answer", row.get("answer", ""))
        gold = row.get("gold_answer", row.get("correct_answer", ""))
        f1 = float(row_metrics.get("f1", token_f1(prediction, gold)))
        baseline = baseline_by_id[str(row.get("example_id"))]
        baseline_metrics = baseline.get("metrics") or {}
        baseline_prediction = baseline.get("predicted_answer", baseline.get("answer", ""))
        baseline_gold = baseline.get("gold_answer", baseline.get("correct_answer", ""))
        baseline_f1 = float(baseline_metrics.get("f1", token_f1(baseline_prediction, baseline_gold)))
        harmed += 1.0 if f1 < baseline_f1 else 0.0
    return round(harmed / len(rows), 4)


def _recompute_summary(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    em = mean(float((row.get("metrics") or {}).get("em", 0.0)) for row in rows)
    f1 = mean(float((row.get("metrics") or {}).get("f1", 0.0)) for row in rows)
    vlm_rate = mean(
        1.0 if (row.get("action_outcome") or {}).get("action") == "invoke_vlm" else 0.0
        for row in rows
    )
    try:
        api = summarize_api_usage(rows)
    except ValueError as exc:
        raise ValueError(f"{path} contains invalid API accounting: {exc}.") from exc
    return {
        "EM": round(em, 4),
        "F1": round(f1, 4),
        "vlm_rate": round(vlm_rate, 4),
        "harm_rate": _harm_rate(rows, baseline_rows),
        "api_requests": int(api["api_requests"]),
        "prompt_tokens": int(api["prompt_tokens"]),
        "completion_tokens": int(api["completion_tokens"]),
        "cost_usd": round(float(api["cost_usd"]), 6),
    }


def _validate_recomputed_summary(
    stored: dict[str, Any],
    recomputed: dict[str, Any],
    path: Path,
) -> None:
    for key, expected in recomputed.items():
        if key not in stored or stored.get(key) is None:
            continue
        observed = float(stored[key])
        if abs(observed - float(expected)) > 1e-6:
            raise ValueError(f"{path} summary.{key} is {observed}; rows recompute to {expected}.")


def failure_type_breakdown(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("failure_type", "other"))].append(row)
    return {
        failure_type: {
            "count": len(group),
            "EM": round(mean(float((row.get("metrics") or {}).get("em", 0.0)) for row in group), 4),
            "F1": round(mean(float((row.get("metrics") or {}).get("f1", 0.0)) for row in group), 4),
            "vlm_rate": round(
                mean(1.0 if (row.get("action_outcome") or {}).get("action") == "invoke_vlm" else 0.0 for row in group), 4
            ),
            "harm_rate": _harm_rate(group, baseline_rows),
        }
        for failure_type, group in sorted(grouped.items())
    }


def summarize_analysis(result_paths: list[Path], baseline_path: Path) -> dict[str, Any]:
    baseline = _validate_analysis_inputs(baseline_path, result_paths)
    baseline_rows = list(baseline["rows"])
    analyses: dict[str, Any] = {}
    for result_path in result_paths:
        result = load_result(result_path)
        label = str(result.get("label") or result_path.stem)
        if label in analyses:
            raise ValueError(f"Final analysis contains duplicate result label {label!r}.")
        rows = list(result["rows"])
        stored_summary = dict(result["summary"])
        recomputed = _recompute_summary(rows, baseline_rows, result_path)
        _validate_recomputed_summary(stored_summary, recomputed, result_path)
        if "runtime_sec" in stored_summary:
            recomputed["runtime_sec"] = stored_summary["runtime_sec"]
        analyses[label] = {
            "source": str(result_path),
            "summary": recomputed,
            "bootstrap_95_ci": {
                "EM": bootstrap_ci((row.get("metrics") or {}).get("em", 0.0) for row in rows),
                "F1": bootstrap_ci((row.get("metrics") or {}).get("f1", 0.0) for row in rows),
            },
            "failure_type_breakdown": failure_type_breakdown(rows, baseline_rows),
            "harm_rate": _harm_rate(rows, baseline_rows),
        }
    return {
        "baseline": str(baseline_path),
        "results": analyses,
    }


def pareto_points(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for label, result in analysis["results"].items():
        summary = result["summary"]
        points.append(
            {
                "label": label,
                "cost_usd": float(summary["cost_usd"]),
                "EM": float(summary["EM"]),
                "F1": float(summary["F1"]),
                "vlm_rate": float(summary["vlm_rate"]),
            }
        )
    return points


def pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            point
            for point in points
            if not any(
                other["cost_usd"] <= point["cost_usd"]
                and other["F1"] >= point["F1"]
                and (other["cost_usd"] < point["cost_usd"] or other["F1"] > point["F1"])
                for other in points
            )
        ],
        key=lambda point: (point["cost_usd"], -point["F1"]),
    )


def write_pareto_pdf(points: list[dict[str, Any]], path: Path) -> None:
    if not points:
        raise ValueError("No result points available for the cost-accuracy Pareto figure.")
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Plotly is required to export the Phase 6 Pareto figure.") from exc
    frontier = pareto_frontier(points)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point["cost_usd"] for point in points],
            y=[point["F1"] for point in points],
            mode="markers+text",
            text=[point["label"] for point in points],
            textposition="top center",
            marker={"size": 10, "color": "#1677ff"},
            name="Systems",
            hovertemplate="%{text}<br>Cost: $%{x:.4f}<br>F1: %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point["cost_usd"] for point in frontier],
            y=[point["F1"] for point in frontier],
            mode="lines",
            line={"color": "#d4380d", "width": 2},
            name="Pareto frontier",
        )
    )
    figure.update_layout(
        template="plotly_white",
        width=960,
        height=620,
        title="Cost-Accuracy Pareto Curve",
        xaxis_title="VLM cost (USD)",
        yaxis_title="Answer F1",
        margin={"l": 80, "r": 40, "t": 80, "b": 70},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.write_image(str(path), format="pdf")
    except ValueError as exc:  # pragma: no cover - kaleido availability varies
        raise RuntimeError("Kaleido is required for Plotly PDF export. Install the project dependencies and retry.") from exc
