from __future__ import annotations

from statistics import mean
from typing import Any


def summarize_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "ndcg@5": 0.0,
            "recall@5": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "visual_fallback_rate": 0.0,
        }
    ndcg = mean(row["metrics"]["ndcg@5"] for row in rows)
    recall = mean(row["metrics"]["recall@5"] for row in rows)
    em = mean(row["metrics"]["em"] for row in rows)
    f1 = mean(row["metrics"]["f1"] for row in rows)
    visual_rate = mean(1.0 if row.get("policy_action") == "invoke_vlm" else 0.0 for row in rows)
    return {
        "count": len(rows),
        "ndcg@5": round(ndcg, 4),
        "recall@5": round(recall, 4),
        "em": round(em, 4),
        "f1": round(f1, 4),
        "visual_fallback_rate": round(visual_rate, 4),
    }


def summarize_by_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["profile"], []).append(row)
    return {profile: summarize_examples(rows) for profile, rows in grouped.items()}

