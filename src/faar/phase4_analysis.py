from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .results_aggregator import summarize_examples

EXPECTED_POLICY_BY_FAILURE: dict[str, str] = {
    "pass": "answer_direct",
    "semantic": "retry_retrieval",
    "word_level": "correct_text",
    "structural": "invoke_vlm",
}


def load_phase3_rows(logs_root: Path, profiles: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    selected_profiles = list(profiles) if profiles is not None else sorted(path.name for path in logs_root.glob("*") if path.is_dir())
    rows_by_profile: dict[str, list[dict[str, Any]]] = {}
    for profile_name in selected_profiles:
        profile_dir = logs_root / profile_name
        rows: list[dict[str, Any]] = []
        if profile_dir.exists():
            for path in sorted(profile_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        rows_by_profile[profile_name] = rows
    return rows_by_profile


def summarize_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = summarize_examples(rows)
    action_distribution = _distribution((row.get("action_outcome") or {}).get("action", "unknown") for row in rows)
    failure_distribution = _distribution(row.get("failure_type", "unknown") for row in rows)
    non_pass_rate = round(
        mean(1.0 if row.get("failure_type", "pass") != "pass" else 0.0 for row in rows),
        4,
    ) if rows else 0.0
    return {
        **base,
        "non_pass_rate": non_pass_rate,
        "action_distribution": action_distribution,
        "failure_distribution": failure_distribution,
    }


def baseline_comparison(
    profile_summaries: dict[str, dict[str, Any]],
    baseline_profile: str = "naive_rag",
) -> list[dict[str, Any]]:
    baseline = profile_summaries.get(baseline_profile)
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for profile_name, summary in sorted(profile_summaries.items()):
        rows.append(
            {
                "profile": profile_name,
                "count": summary.get("count", 0),
                "em": summary.get("em", 0.0),
                "f1": summary.get("f1", 0.0),
                "ndcg@5": summary.get("ndcg@5", 0.0),
                "recall@5": summary.get("recall@5", 0.0),
                "visual_fallback_rate": summary.get("visual_fallback_rate", 0.0),
                "delta_em_vs_baseline": round(summary.get("em", 0.0) - baseline.get("em", 0.0), 4),
                "delta_f1_vs_baseline": round(summary.get("f1", 0.0) - baseline.get("f1", 0.0), 4),
                "delta_ndcg@5_vs_baseline": round(summary.get("ndcg@5", 0.0) - baseline.get("ndcg@5", 0.0), 4),
                "delta_recall@5_vs_baseline": round(summary.get("recall@5", 0.0) - baseline.get("recall@5", 0.0), 4),
            }
        )
    return rows


def assess_phase4_claims(rows_by_profile: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    profile_summaries = {profile: summarize_profile(rows) for profile, rows in rows_by_profile.items()}
    faar_full = profile_summaries.get("faar_full", {})
    naive = profile_summaries.get("naive_rag", {})
    no_vlm = profile_summaries.get("faar_no_vlm", {})
    faar_rows = rows_by_profile.get("faar_full", [])

    q1 = _assess_quality_gate_cost_control(faar_full)
    q2 = _assess_typed_recovery_vs_naive(faar_full, naive)
    q3 = _assess_selective_vlm_benefit(faar_full, no_vlm)
    q4 = _assess_diagnostic_alignment(faar_rows)

    return {
        "q1_quality_gate_reduces_expensive_recovery": q1,
        "q2_typed_recovery_outperforms_naive_fallback": q2,
        "q3_selective_visual_fallback_trades_accuracy_for_cost": q3,
        "q4_diagnostic_categories_match_observed_errors": q4,
    }


def build_case_studies(
    rows_by_profile: dict[str, list[dict[str, Any]]],
    focus_profile: str = "faar_full",
    baseline_profile: str = "naive_rag",
    top_n: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    focus_by_example = {row.get("example_id", ""): row for row in rows_by_profile.get(focus_profile, [])}
    baseline_by_example = {row.get("example_id", ""): row for row in rows_by_profile.get(baseline_profile, [])}
    shared_ids = [example_id for example_id in focus_by_example if example_id and example_id in baseline_by_example]
    comparisons: list[dict[str, Any]] = []
    for example_id in shared_ids:
        focus = focus_by_example[example_id]
        baseline = baseline_by_example[example_id]
        focus_f1 = _metric(focus, "f1")
        baseline_f1 = _metric(baseline, "f1")
        focus_em = _metric(focus, "em")
        baseline_em = _metric(baseline, "em")
        comparisons.append(
            {
                "example_id": example_id,
                "question": focus.get("question", ""),
                "gold_answer": focus.get("gold_answer", ""),
                "focus_predicted_answer": focus.get("predicted_answer", ""),
                "baseline_predicted_answer": baseline.get("predicted_answer", ""),
                "focus_failure_type": focus.get("failure_type", "pass"),
                "focus_policy_action": focus.get("policy_action", "answer_direct"),
                "focus_action_outcome": (focus.get("action_outcome") or {}).get("action", "unknown"),
                "focus_f1": round(focus_f1, 4),
                "baseline_f1": round(baseline_f1, 4),
                "f1_delta": round(focus_f1 - baseline_f1, 4),
                "focus_em": round(focus_em, 4),
                "baseline_em": round(baseline_em, 4),
                "em_delta": round(focus_em - baseline_em, 4),
            }
        )
    comparisons.sort(key=lambda item: (item["f1_delta"], item["em_delta"]), reverse=True)
    improvements = [item for item in comparisons if item["f1_delta"] > 0][: max(0, top_n)]
    regressions = sorted(
        (item for item in comparisons if item["f1_delta"] < 0),
        key=lambda item: (item["f1_delta"], item["em_delta"]),
    )[: max(0, top_n)]
    ties = [item for item in comparisons if item["f1_delta"] == 0][: max(0, top_n)]
    return {
        "improvements": improvements,
        "regressions": regressions,
        "ties": ties,
    }


def build_phase4_report(
    rows_by_profile: dict[str, list[dict[str, Any]]],
    baseline_profile: str = "naive_rag",
) -> dict[str, Any]:
    profile_summaries = {profile: summarize_profile(rows) for profile, rows in rows_by_profile.items()}
    comparison_rows = baseline_comparison(profile_summaries, baseline_profile=baseline_profile)
    claim_assessment = assess_phase4_claims(rows_by_profile)
    case_studies = build_case_studies(rows_by_profile, focus_profile="faar_full", baseline_profile=baseline_profile)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_profile": baseline_profile,
        "profile_summaries": profile_summaries,
        "baseline_comparison": comparison_rows,
        "claim_assessment": claim_assessment,
        "case_studies": case_studies,
    }


def write_phase4_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "profile",
        "count",
        "em",
        "f1",
        "ndcg@5",
        "recall@5",
        "visual_fallback_rate",
        "delta_em_vs_baseline",
        "delta_f1_vs_baseline",
        "delta_ndcg@5_vs_baseline",
        "delta_recall@5_vs_baseline",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _metric(row: dict[str, Any], metric_name: str) -> float:
    metrics = row.get("metrics") or {}
    value = metrics.get(metric_name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _distribution(values: Iterable[str]) -> dict[str, dict[str, float]]:
    counter = Counter(values)
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {
        key: {"count": count, "rate": round(count / total, 4)}
        for key, count in sorted(counter.items(), key=lambda item: item[0])
    }


def _assess_quality_gate_cost_control(faar_full_summary: dict[str, Any]) -> dict[str, Any]:
    if not faar_full_summary:
        return {
            "status": "inconclusive",
            "reason": "missing_faar_full_profile",
        }
    visual_rate = float(faar_full_summary.get("visual_fallback_rate", 0.0))
    cost_savings_vs_always_on_visual = round(1.0 - visual_rate, 4)
    return {
        "status": "supported" if visual_rate < 1.0 else "not_supported",
        "faar_full_visual_fallback_rate": round(visual_rate, 4),
        "estimated_cost_savings_vs_always_on_visual": cost_savings_vs_always_on_visual,
    }


def _assess_typed_recovery_vs_naive(faar_full_summary: dict[str, Any], naive_summary: dict[str, Any]) -> dict[str, Any]:
    if not faar_full_summary or not naive_summary:
        return {
            "status": "inconclusive",
            "reason": "missing_faar_full_or_naive_profile",
        }
    delta_f1 = round(float(faar_full_summary.get("f1", 0.0)) - float(naive_summary.get("f1", 0.0)), 4)
    delta_em = round(float(faar_full_summary.get("em", 0.0)) - float(naive_summary.get("em", 0.0)), 4)
    if delta_f1 > 0 or delta_em > 0:
        status = "supported"
    elif delta_f1 <= 0 and delta_em <= 0:
        status = "not_supported"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "delta_f1_vs_naive_rag": delta_f1,
        "delta_em_vs_naive_rag": delta_em,
    }


def _assess_selective_vlm_benefit(faar_full_summary: dict[str, Any], no_vlm_summary: dict[str, Any]) -> dict[str, Any]:
    if not faar_full_summary or not no_vlm_summary:
        return {
            "status": "inconclusive",
            "reason": "missing_faar_full_or_faar_no_vlm_profile",
        }
    delta_f1 = round(float(faar_full_summary.get("f1", 0.0)) - float(no_vlm_summary.get("f1", 0.0)), 4)
    delta_em = round(float(faar_full_summary.get("em", 0.0)) - float(no_vlm_summary.get("em", 0.0)), 4)
    visual_rate = round(float(faar_full_summary.get("visual_fallback_rate", 0.0)), 4)
    if delta_f1 > 0 or delta_em > 0:
        status = "supported"
    elif visual_rate > 0 and delta_f1 <= 0 and delta_em <= 0:
        status = "not_supported"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "delta_f1_vs_faar_no_vlm": delta_f1,
        "delta_em_vs_faar_no_vlm": delta_em,
        "faar_full_visual_fallback_rate": visual_rate,
    }


def _assess_diagnostic_alignment(faar_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not faar_rows:
        return {
            "status": "inconclusive",
            "reason": "missing_faar_full_rows",
        }
    matches = 0
    for row in faar_rows:
        failure_type = row.get("failure_type", "pass")
        expected_policy = EXPECTED_POLICY_BY_FAILURE.get(failure_type, "answer_direct")
        policy_action = row.get("policy_action", "answer_direct")
        if policy_action == expected_policy:
            matches += 1
    consistency = round(matches / len(faar_rows), 4)
    non_pass_rate = round(mean(1.0 if row.get("failure_type", "pass") != "pass" else 0.0 for row in faar_rows), 4)
    status = "supported" if consistency >= 0.95 and non_pass_rate > 0 else "not_supported"
    return {
        "status": status,
        "diagnosis_policy_consistency": consistency,
        "non_pass_rate": non_pass_rate,
        "failure_distribution": _distribution(row.get("failure_type", "unknown") for row in faar_rows),
    }
