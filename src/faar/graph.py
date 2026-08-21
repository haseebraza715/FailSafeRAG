from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answering import answer_from_hits
from .api_logging import zero_api_usage
from .chunking import build_chunks
from .data import Phase0Repository
from .quality import diagnose_failure, quality_gate
from .recovery import ByT5Corrector, VisualFallback, semantic_backtrack
from .retrieval import HybridRetriever
from .settings import AppSettings
from .symspell import correct_text as symspell_correct_text
from .types import Chunk, RetrievalHit

RANDOM_RECOVERY_TYPES = ("semantic", "word_level", "structural")

POLICY_ACTION_BY_TYPE = {
    "word_level": "correct_text",
    "structural": "invoke_vlm",
    "semantic": "retry_retrieval",
}


def random_recovery_type(seed: int, example_id: str) -> str:
    """Deterministic per-example random recovery type for the B2 ablation.

    The choice is a pure function of the locked run seed and the example id, so
    it is invariant to resume, iteration order, and sharding. Seed and example
    id are mixed through SHA-256 instead of Python's process-randomized hash(),
    keeping the result stable across processes and interpreter runs.
    """
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    index = int.from_bytes(digest, byteorder="big") % len(RANDOM_RECOVERY_TYPES)
    return RANDOM_RECOVERY_TYPES[index]


class GraphState(TypedDict, total=False):
    example_id: str
    question: str
    example: Any
    chunks: list[Any]
    retriever: Any
    retrieved_hits: list[RetrievalHit]
    gate: dict[str, Any]
    failure_type: str
    policy_action: str
    recovery_type: str
    corrected_hits: list[RetrievalHit]
    semantic_retry_query: str
    visual_result: dict[str, Any]
    answer: str
    answer_meta: dict[str, Any]
    action_outcome: dict[str, Any]


def build_graph(settings: AppSettings, repo: Phase0Repository | Any | None = None):
    repo = repo or Phase0Repository(settings)
    wordlevel_fallback = settings.experiment.wordlevel_fallback or settings.recovery.wordlevel_fallback
    corrector = (
        None
        if wordlevel_fallback == "symspell"
        else ByT5Corrector(
            settings.recovery.byt5_model,
            settings.recovery.byt5_revision,
            settings.recovery.correction,
        )
    )
    visual_fallback = VisualFallback(settings)
    shared_chunks = repo.get_corpus_chunks(settings.retrieval) if hasattr(repo, "get_corpus_chunks") else None
    cache_dir = Path(os.getenv("FAAR_CACHE_DIR", str(settings.project_root / "cache"))) / "text_embeddings"
    shared_retriever = (
        HybridRetriever(shared_chunks, settings.retrieval, cache_dir=cache_dir)
        if shared_chunks is not None
        else None
    )

    def load_example(state: GraphState) -> GraphState:
        example = repo.get_example(state["example_id"])
        question = state.get("question") or example.question
        return {"example": example, "question": question}

    def prepare_retrieval(state: GraphState) -> GraphState:
        example = state["example"]
        chunks = shared_chunks if shared_chunks is not None else build_chunks(example, settings.retrieval)
        if not chunks:
            raise ValueError(
                f"No retrieval chunks could be built for example {example.example_id!r}: "
                "the OCR text contains no words"
            )
        retriever = (
            shared_retriever
            if shared_retriever is not None
            else HybridRetriever(chunks, settings.retrieval, cache_dir=cache_dir)
        )
        return {"chunks": chunks, "retriever": retriever}

    def retrieve(state: GraphState) -> GraphState:
        hits = state["retriever"].retrieve(state["question"], top_k=settings.retrieval.top_k)
        return {"retrieved_hits": hits}

    def gate_node(state: GraphState) -> GraphState:
        gate = quality_gate(
            state["retrieved_hits"],
            settings.gate,
            prototype_signals=settings.retrieval.embedding_backend == "local-hash-v1",
        )
        return {"gate": gate}

    def route_after_gate(state: GraphState) -> str:
        if settings.experiment.force_direct_answer:
            return "answer_direct"
        if settings.experiment.force_vlm:
            return "invoke_vlm"
        if settings.experiment.force_recovery:
            return "diagnose"
        return "answer_direct" if state["gate"]["pass_gate"] else "diagnose"

    def diagnose_node(state: GraphState) -> GraphState:
        if settings.experiment.random_recovery:
            recovery_type = random_recovery_type(settings.experiment.random_seed, state["example_id"])
            return {
                "failure_type": "random",
                "policy_action": POLICY_ACTION_BY_TYPE[recovery_type],
                "recovery_type": recovery_type,
            }
        if settings.experiment.disable_diagnosis:
            return {"failure_type": "semantic", "policy_action": "answer_direct"}
        failure_type = diagnose_failure(state["retrieved_hits"], state["gate"], settings.gate)
        policy_action = POLICY_ACTION_BY_TYPE[failure_type]
        return {"failure_type": failure_type, "policy_action": policy_action}

    def route_after_diagnosis(state: GraphState) -> str:
        if settings.experiment.force_direct_answer:
            return "answer_direct"
        if settings.experiment.disable_backtracking and state.get("policy_action") == "retry_retrieval":
            return "answer_direct"
        if settings.experiment.disable_vlm and state.get("policy_action") == "invoke_vlm":
            return "answer_direct"
        return state["policy_action"]

    def word_level_recovery(state: GraphState) -> GraphState:
        updated_hits: list[RetrievalHit] = []
        applied = 0
        decisions: list[dict[str, str | bool]] = []
        for hit in state["retrieved_hits"]:
            if wordlevel_fallback == "symspell":
                corrected_text = symspell_correct_text(hit.chunk.text)
                proposal = {
                    "text": corrected_text,
                    "applied": corrected_text != hit.chunk.text,
                    "reason": "symspell_local_fallback",
                }
            elif not settings.recovery.enable_byt5:
                proposal = {
                    "text": hit.chunk.text,
                    "candidate": hit.chunk.text,
                    "applied": False,
                    "reason": "byt5_disabled_by_profile",
                }
                corrected_text = hit.chunk.text
            else:
                assert corrector is not None
                proposal = corrector.propose_correction(hit.chunk.text)
                corrected_text = str(proposal["text"]) or hit.chunk.text
            decisions.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "applied": bool(proposal["applied"]),
                    "reason": str(proposal["reason"]),
                }
            )
            if proposal["applied"]:
                applied += 1
            updated_hits.append(
                RetrievalHit(
                    chunk=Chunk(
                        chunk_id=hit.chunk.chunk_id,
                        example_id=hit.chunk.example_id,
                        doc_name=hit.chunk.doc_name,
                        page_id=hit.chunk.page_id,
                        text=corrected_text,
                        image_path=hit.chunk.image_path,
                    ),
                    bm25_score=hit.bm25_score,
                    dense_score=hit.dense_score,
                    fused_score=hit.fused_score,
                    reranker_score=hit.reranker_score,
                )
            )
        if wordlevel_fallback == "symspell":
            outcome_reason = "symspell_local_fallback" if applied else "symspell_no_change"
        elif not settings.recovery.enable_byt5:
            outcome_reason = "byt5_disabled_by_profile"
        elif applied:
            outcome_reason = "byt5_correction_applied"
        else:
            outcome_reason = "byt5_correction_guarded"
        return {
            "corrected_hits": updated_hits,
            "action_outcome": {
                "action": "correct_text",
                "status": "succeeded" if applied else "skipped",
                "reason": outcome_reason,
                "applied_count": applied,
                "reviewed_count": len(state["retrieved_hits"]),
                "decisions": decisions,
            },
        }

    def semantic_recovery(state: GraphState) -> GraphState:
        retry_query = semantic_backtrack(state["question"], state["retrieved_hits"])
        retry_hits = state["retriever"].retrieve(retry_query, top_k=settings.retrieval.semantic_backtrack_top_k)
        return {
            "semantic_retry_query": retry_query,
            "corrected_hits": retry_hits,
            "action_outcome": {
                "action": "retry_retrieval",
                "status": "succeeded",
                "reason": "semantic_backtrack_query_executed",
            },
        }

    def structural_recovery(state: GraphState) -> GraphState:
        if not settings.recovery.enable_vlm:
            return {
                "visual_result": {
                    "backend": settings.recovery.vlm_backend,
                    "status": "skipped",
                    "reason": "vlm_disabled_by_profile",
                    "answer": "",
                    "used_images": [],
                    "api_usage": zero_api_usage(),
                },
                "action_outcome": {
                    "action": "invoke_vlm",
                    "status": "skipped",
                    "reason": "vlm_disabled_by_profile",
                },
            }
        fallback_context = "\n".join(hit.chunk.text for hit in state["retrieved_hits"])
        image_paths = list(
            dict.fromkeys(Path(hit.chunk.image_path) for hit in state["retrieved_hits"] if hit.chunk.image_path)
        )
        result = visual_fallback.answer(state["question"], image_paths, fallback_context)
        return {
            "visual_result": result,
            "action_outcome": {
                "action": "invoke_vlm",
                "status": result.get("status", "unknown"),
                "reason": result.get("reason", "vlm_action_executed"),
            },
        }

    def direct_answer(state: GraphState) -> GraphState:
        answer_meta = answer_from_hits(state["question"], state["retrieved_hits"])
        return {
            "answer": answer_meta["answer"],
            "answer_meta": answer_meta,
            "action_outcome": {
                "action": "answer_direct",
                "status": "succeeded",
                "reason": "passed_quality_gate",
            },
        }

    def recovered_answer(state: GraphState) -> GraphState:
        if state.get("visual_result") and state["visual_result"].get("status") == "succeeded":
            return {
                "answer": state["visual_result"]["answer"],
                "answer_meta": {"answer_mode": "visual_fallback", "visual_backend": state["visual_result"]["backend"]},
            }
        hits = state.get("corrected_hits") or state["retrieved_hits"]
        answer_meta = answer_from_hits(state["question"], hits)
        return {"answer": answer_meta["answer"], "answer_meta": answer_meta}

    graph = StateGraph(GraphState)
    graph.add_node("load_example", load_example)
    graph.add_node("prepare_retrieval", prepare_retrieval)
    graph.add_node("retrieve", retrieve)
    # Node id must differ from GraphState key "gate" under langgraph 0.3.x.
    graph.add_node("quality_gate", gate_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("correct_text", word_level_recovery)
    graph.add_node("retry_retrieval", semantic_recovery)
    graph.add_node("invoke_vlm", structural_recovery)
    graph.add_node("answer_direct", direct_answer)
    graph.add_node("answer_recovered", recovered_answer)

    graph.add_edge(START, "load_example")
    graph.add_edge("load_example", "prepare_retrieval")
    graph.add_edge("prepare_retrieval", "retrieve")
    graph.add_edge("retrieve", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_gate,
        {"answer_direct": "answer_direct", "diagnose": "diagnose", "invoke_vlm": "invoke_vlm"},
    )
    graph.add_conditional_edges(
        "diagnose",
        route_after_diagnosis,
        {
            "answer_direct": "answer_direct",
            "correct_text": "correct_text",
            "retry_retrieval": "retry_retrieval",
            "invoke_vlm": "invoke_vlm",
        },
    )
    graph.add_edge("correct_text", "answer_recovered")
    graph.add_edge("retry_retrieval", "answer_recovered")
    graph.add_edge("invoke_vlm", "answer_recovered")
    graph.add_edge("answer_direct", END)
    graph.add_edge("answer_recovered", END)
    return graph.compile()
