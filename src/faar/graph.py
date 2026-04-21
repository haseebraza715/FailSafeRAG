from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answering import answer_from_hits
from .chunking import build_chunks
from .data import Phase0Repository
from .quality import diagnose_failure, quality_gate
from .recovery import ByT5Corrector, VisualFallback, semantic_backtrack
from .retrieval import HybridRetriever
from .settings import AppSettings
from .types import Chunk, RetrievalHit


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
    corrected_hits: list[RetrievalHit]
    semantic_retry_query: str
    visual_result: dict[str, Any]
    answer: str
    answer_meta: dict[str, Any]
    action_outcome: dict[str, Any]


def build_graph(settings: AppSettings):
    repo = Phase0Repository(settings)
    corrector = ByT5Corrector(settings.recovery.byt5_model)
    visual_fallback = VisualFallback(settings)

    def load_example(state: GraphState) -> GraphState:
        example = repo.get_example(state["example_id"])
        question = state.get("question") or example.question
        return {"example": example, "question": question}

    def prepare_retrieval(state: GraphState) -> GraphState:
        example = state["example"]
        chunks = build_chunks(example, settings.retrieval)
        retriever = HybridRetriever(chunks, settings.retrieval)
        return {"chunks": chunks, "retriever": retriever}

    def retrieve(state: GraphState) -> GraphState:
        hits = state["retriever"].retrieve(state["question"], top_k=settings.retrieval.top_k)
        return {"retrieved_hits": hits}

    def gate_node(state: GraphState) -> GraphState:
        gate = quality_gate(state["retrieved_hits"], settings.gate)
        return {"gate": gate}

    def route_after_gate(state: GraphState) -> str:
        if settings.experiment.force_direct_answer:
            return "answer_direct"
        return "answer_direct" if state["gate"]["pass_gate"] else "diagnose"

    def diagnose_node(state: GraphState) -> GraphState:
        if settings.experiment.disable_diagnosis:
            return {"failure_type": "semantic", "policy_action": "answer_direct"}
        failure_type = diagnose_failure(state["retrieved_hits"], state["gate"], settings.gate)
        policy_action = {
            "word_level": "correct_text",
            "structural": "invoke_vlm",
            "semantic": "retry_retrieval",
        }[failure_type]
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
                    ),
                    bm25_score=hit.bm25_score,
                    dense_score=hit.dense_score,
                    fused_score=hit.fused_score,
                )
            )
        return {
            "corrected_hits": updated_hits,
            "action_outcome": {
                "action": "correct_text",
                "status": "succeeded" if applied else "skipped",
                "reason": "byt5_correction_applied" if applied else "byt5_correction_guarded",
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
                },
                "action_outcome": {
                    "action": "invoke_vlm",
                    "status": "skipped",
                    "reason": "vlm_disabled_by_profile",
                },
            }
        example = state["example"]
        fallback_context = "\n".join(hit.chunk.text for hit in state["retrieved_hits"])
        result = visual_fallback.answer(example.question, example.image_paths, fallback_context)
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
    graph.add_node("gate", gate_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("correct_text", word_level_recovery)
    graph.add_node("retry_retrieval", semantic_recovery)
    graph.add_node("invoke_vlm", structural_recovery)
    graph.add_node("answer_direct", direct_answer)
    graph.add_node("answer_recovered", recovered_answer)

    graph.add_edge(START, "load_example")
    graph.add_edge("load_example", "prepare_retrieval")
    graph.add_edge("prepare_retrieval", "retrieve")
    graph.add_edge("retrieve", "gate")
    graph.add_conditional_edges("gate", route_after_gate, {"answer_direct": "answer_direct", "diagnose": "diagnose"})
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
