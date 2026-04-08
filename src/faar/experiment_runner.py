from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import Phase0Repository
from .experiment_profiles import apply_profile
from .graph import build_graph
from .metrics import exact_match, ndcg_at_k, recall_at_k, token_f1
from .settings import AppSettings


def run_profile(
    settings: AppSettings,
    profile_name: str,
    max_examples: int | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    settings = apply_profile(settings, profile_name)
    repo = Phase0Repository(settings)
    graph = build_graph(settings)
    example_ids = repo.list_example_ids()
    if max_examples is not None:
        example_ids = example_ids[: max(0, max_examples)]
    base_output = output_dir or (settings.project_root / "logs/phase3" / profile_name)
    base_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for example_id in example_ids:
        result = graph.invoke({"example_id": example_id})
        hit_texts = [hit.chunk.text for hit in result.get("corrected_hits") or result.get("retrieved_hits", [])]
        gold = result["example"].correct_answer
        prediction = result.get("answer", "")
        row = {
            "profile": profile_name,
            "example_id": example_id,
            "question": result.get("question", ""),
            "gold_answer": gold,
            "predicted_answer": prediction,
            "failure_type": result.get("failure_type", "pass"),
            "policy_action": result.get("policy_action", "answer_direct"),
            "action_outcome": result.get("action_outcome", {}),
            "metrics": {
                "ndcg@5": ndcg_at_k(hit_texts, gold, k=5),
                "recall@5": recall_at_k(hit_texts, gold, k=5),
                "em": exact_match(prediction, gold),
                "f1": token_f1(prediction, gold),
            },
            "top_hit_texts": hit_texts[:5],
            "run_metadata": {
                "profile": profile_name,
                "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                "api_enabled": settings.recovery.api_enabled,
                "vlm_backend": settings.recovery.vlm_backend,
                "openai_model": settings.recovery.openai_model,
            },
        }
        rows.append(row)
        destination = base_output / f"{example_id}.json"
        destination.write_text(json.dumps(row, indent=2))
    return rows

