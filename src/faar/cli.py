from __future__ import annotations

import json
from pathlib import Path

import typer

from .graph import build_graph
from .settings import AppSettings

app = typer.Typer(add_completion=False, help="FAAR Phase 1 prototype CLI")


@app.command()
def run_example(
    example_id: str = typer.Option(..., help="Phase 0 example id to run"),
    question: str | None = typer.Option(None, help="Override the benchmark question"),
    output: Path | None = typer.Option(None, help="Optional path for the JSON log"),
    vlm_backend: str = typer.Option("mock", help="Visual fallback backend: mock or openai"),
) -> None:
    settings = AppSettings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.recovery.vlm_backend = vlm_backend
    graph = build_graph(settings)
    result = graph.invoke({"example_id": example_id, "question": question})
    payload = {
        "example_id": example_id,
        "question": result["question"],
        "correct_answer": result["example"].correct_answer,
        "answer": result.get("answer", ""),
        "gate": result.get("gate", {}),
        "failure_type": result.get("failure_type", "pass"),
        "policy_action": result.get("policy_action", "answer_direct"),
        "semantic_retry_query": result.get("semantic_retry_query"),
        "visual_result": result.get("visual_result"),
        "answer_meta": result.get("answer_meta", {}),
        "top_hits": [hit.to_dict() for hit in result.get("corrected_hits") or result.get("retrieved_hits", [])],
    }
    destination = output or settings.logs_dir / f"{example_id}.json"
    destination.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))
    typer.echo(f"\nSaved log to {destination}")
