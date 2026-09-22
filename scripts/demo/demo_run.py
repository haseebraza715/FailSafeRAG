"""Pipeline milestone demo over the synthetic corpus (offline, deterministic).

Prints the real per-example milestones from the compiled LangGraph pipeline
(build_graph) and the aggregate metrics from the experiment runner
(run_profile): retrieval scores -> quality gate -> diagnosis -> recovery
action -> answer with measured confidence. No fabricated output: every value is
read from the graph/runner results.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from faar.experiment_runner import run_profile
from faar.graph import build_graph
from faar.settings import AppSettings

random.seed(42)
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "examples/demo_corpus"
EXAMPLES = ["noisy_threshold", "clean_threshold", "clean_accesslog"]


def settings() -> AppSettings:
    s = AppSettings(project_root=CORPUS)
    s.retrieval.embedding_backend = "local-hash-v1"
    s.gate.quality_threshold = 0.52
    s.recovery.vlm_backend = "mock"
    s.recovery.enable_byt5 = False
    s.recovery.api_enabled = False
    return s


def shorten(text: str, width: int = 88) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def show_example(graph, example_id: str) -> None:
    result = graph.invoke({"example_id": example_id})
    top = result["retrieved_hits"][0]
    gate = result["gate"]
    answer_meta = result.get("answer_meta", {})
    outcome = result.get("action_outcome", {})
    failure = result.get("failure_type", "pass")
    policy = result.get("policy_action", "answer_direct")

    print()
    print(f"example   : {example_id}")
    print(f"question  : {result['question']}")
    print(f"gold      : {result['example'].correct_answer}")
    print(f"  retrieve  top hit  fused={top.fused_score:.3f}  bm25={top.bm25_score:.3f}  dense={top.dense_score:.3f}")
    print(f"            text     {shorten(top.chunk.text)}")
    verdict = "PASS" if gate["pass_gate"] else "FAIL"
    reasons = ", ".join(gate["reasons"]) or "-"
    print(
        f"  quality   {verdict}  score={gate['quality_score']:.4f}  corruption={gate['corruption_score']:.4f}  reasons=[{reasons}]"
    )
    if not gate["pass_gate"]:
        print(f"  diagnose  {failure}  ->  policy: {policy}")
        print(
            f"  recover   {outcome.get('action')}  status={outcome.get('status')}  reason={outcome.get('reason')}"
        )
    else:
        print("  recover   none — direct answer")
    mode = answer_meta.get("answer_mode", "-")
    support = answer_meta.get("support_score", "-")
    visual = result.get("visual_result")
    visual_note = (
        "skipped (mock backend)" if visual and visual.get("status") == "skipped" else "none — no multimodal spend"
    )
    print(f"  answer    {result.get('answer', '')!r}  (mode={mode}, support_score={support})")
    print(f"  visual    {visual_note}")


def main() -> None:
    print("FAAR · Failure-Aware Agentic Recovery for OCR-RAG — pipeline milestones")
    print(f"corpus    : {CORPUS.relative_to(REPO_ROOT)}")
    print("engine    : local-hash embeddings · mock VLM · seed 42 · ByT5 disabled")
    print("pipeline  : retrieve -> quality gate -> diagnose -> recover -> answer")

    graph = build_graph(settings())
    for example_id in EXAMPLES:
        show_example(graph, example_id)

    rows = run_profile(
        settings(),
        profile_name="faar_full",
        example_ids=EXAMPLES,
        seed=42,
        output_dir=REPO_ROOT / "logs/demo_corpus",
    )
    print()
    print("metrics   : per-example EM/F1 from run_profile (experiment runner)")
    for row in rows:
        m = row["metrics"]
        rec = row["recovery_metrics"]["em"]["effect"]
        print(
            f"  {row['example_id']:<18} em={m['em']:.2f}  f1={m['f1']:.3f}  ndcg@5={m['ndcg@5']:.3f}  "
            f"recall@5={m['recall@5']:.3f}  recovery_em_effect={rec}"
        )


if __name__ == "__main__":
    main()
