from __future__ import annotations

import json
import platform
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer

from .graph import build_graph
from .settings import AppSettings

app = typer.Typer(add_completion=False, help="FAAR Phase 1 prototype CLI")


@app.command()
def run_example(
    example_id: str = typer.Option(..., help="Phase 0 example id to run"),
    question: str | None = typer.Option(None, help="Override the benchmark question"),
    output: Path | None = typer.Option(None, help="Optional path for the JSON log"),
    project_root: Path | None = typer.Option(None, help="Override repository project root"),
    vlm_backend: str = typer.Option("mock", help="Visual fallback backend: mock or openai"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    settings = AppSettings(project_root=project_root) if project_root else AppSettings()
    settings.recovery.vlm_backend = vlm_backend
    settings.validate_runtime_paths()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(settings)
    result = graph.invoke({"example_id": example_id, "question": question})
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "project_root": str(settings.project_root),
        "paths": {
            "phase0_manifest": str(settings.phase0_manifest),
            "phase0_summary": str(settings.phase0_summary),
            "phase0_ocr_dir": str(settings.phase0_ocr_dir),
            "logs_dir": str(settings.logs_dir),
        },
        "model_config": {
            "embedding_model": settings.retrieval.embedding_model,
            "byt5_model": settings.recovery.byt5_model,
            "vlm_backend": settings.recovery.vlm_backend,
            "openai_model": settings.recovery.openai_model,
        },
        "thresholds": {
            "quality_threshold": settings.gate.quality_threshold,
            "structural_threshold": settings.gate.structural_threshold,
            "weird_char_threshold": settings.gate.weird_char_threshold,
            "lexical_floor": settings.gate.lexical_floor,
            "dense_floor": settings.gate.dense_floor,
        },
    }
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
        "action_outcome": result.get("action_outcome", {}),
        "run_metadata": run_metadata,
        "top_hits": [hit.to_dict() for hit in result.get("corrected_hits") or result.get("retrieved_hits", [])],
    }
    destination = output or settings.logs_dir / f"{example_id}.json"
    destination.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))
    typer.echo(f"\nSaved log to {destination}")
