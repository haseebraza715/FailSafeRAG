"""Deterministic tripwires that committed summaries match a fresh offline run.

The committed benchmark artifacts (artifacts/phase3) are the repo's evidence
base. These tests catch a stale-artifact regression: if the pipeline code or the
fixture changes the aggregate numbers, the committed summaries no longer match a
fresh offline faar_full aggregation and these tests fail loudly.

The fresh run is kept deterministic and environment-independent: it uses the
local-hash embedding backend (default), the mock VLM backend, and ByT5 disabled
so results do not depend on whether the ml extra / model weights are installed.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_PRESENT = (
    (REPO_ROOT / "data/phase0/sample_manifest.csv").exists()
    and (REPO_ROOT / "artifacts/phase0/ocr_text").is_dir()
    and (REPO_ROOT / "artifacts/phase3/metrics_summary.json").exists()
)

pytestmark = pytest.mark.skipif(
    not FIXTURE_PRESENT,
    reason="committed phase-0 fixture or phase-3 summaries not present in this checkout",
)


def _committed_faar_full_summary() -> dict:
    path = REPO_ROOT / "artifacts/phase3/metrics_summary.json"
    payload = json.loads(path.read_text())
    return payload["profiles"]["faar_full"]


def _fresh_faar_full_summary() -> dict:
    from faar.answering import answer_from_hits
    from faar.data import Phase0Repository
    from faar.graph import build_graph
    from faar.metrics import exact_match, ndcg_at_k, recall_at_k, token_f1
    from faar.settings import AppSettings

    random.seed(42)
    np.random.seed(42)
    settings = AppSettings(project_root=REPO_ROOT)
    settings.retrieval.embedding_backend = "local-hash-v1"
    settings.gate.quality_threshold = 0.52
    settings.recovery.vlm_backend = "mock"
    settings.recovery.enable_byt5 = False
    settings.recovery.api_enabled = False
    repo = Phase0Repository(settings)
    graph = build_graph(settings)
    em = f1 = recall = ndcg = 0.0
    changed = 0
    reviewed = 0
    visual = 0
    em_effects: Counter[str] = Counter()
    f1_effects: Counter[str] = Counter()
    count = 0
    for example_id in repo.list_example_ids():
        result = graph.invoke({"example_id": example_id})
        gold = result["example"].correct_answer
        prediction = result.get("answer", "")
        question = result.get("question", "")
        hits = result.get("corrected_hits") or result.get("retrieved_hits", [])
        hit_texts = [hit.chunk.text for hit in hits]
        counterfactual = answer_from_hits(question, result.get("retrieved_hits", []))["answer"]
        actual_em = exact_match(prediction, gold)
        actual_f1 = token_f1(prediction, gold)
        counterfactual_em = exact_match(counterfactual, gold)
        counterfactual_f1 = token_f1(counterfactual, gold)
        action = (result.get("action_outcome") or {}).get("action", "answer_direct")
        if action == "invoke_vlm":
            visual += 1
        if action in {"correct_text", "retry_retrieval", "invoke_vlm"}:
            reviewed += 1
            if counterfactual != prediction:
                changed += 1
            em_effects[_effect(actual_em, counterfactual_em)] += 1
            f1_effects[_effect(actual_f1, counterfactual_f1)] += 1
        em += actual_em
        f1 += actual_f1
        recall += recall_at_k(hit_texts, gold, k=5)
        ndcg += ndcg_at_k(hit_texts, gold, k=5)
        count += 1
    return {
        "count": count,
        "ndcg@5": round(ndcg / count, 4),
        "recall@5": round(recall / count, 4),
        "em": round(em / count, 4),
        "f1": round(f1 / count, 4),
        "visual_fallback_rate": round(visual / count, 4),
        "recovery_outcomes": {
            "reviewed_count": reviewed,
            "changed_answer_count": changed,
            "changed_answer_rate": round(changed / reviewed, 4) if reviewed else 0.0,
            "em_effect_distribution": dict(em_effects),
            "f1_effect_distribution": dict(f1_effects),
        },
    }


def _effect(actual: float, counterfactual: float) -> str:
    if actual > counterfactual:
        return "improved"
    if actual < counterfactual:
        return "worsened"
    return "equal"


def test_committed_summary_matches_fresh_offline_aggregation() -> None:
    committed = _committed_faar_full_summary()
    fresh = _fresh_faar_full_summary()
    assert fresh["count"] == 40
    for metric in ("ndcg@5", "recall@5", "em", "f1", "visual_fallback_rate"):
        assert fresh[metric] == committed[metric], (
            f"committed {metric} {committed[metric]} no longer matches a fresh offline "
            f"run {fresh[metric]}; regenerate artifacts/phase3 from current code"
        )


def test_committed_recovery_outcome_matches_fresh_offline_run() -> None:
    committed = _committed_faar_full_summary()
    fresh = _fresh_faar_full_summary()
    assert fresh["recovery_outcomes"] == committed["recovery_outcomes"]
