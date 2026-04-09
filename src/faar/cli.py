from __future__ import annotations

import json
import platform
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer

from .experiment_profiles import PROFILES
from .experiment_runner import run_profile
from .graph import build_graph
from .phase4_analysis import build_phase4_report, load_phase3_rows, write_phase4_comparison_csv
from .results_aggregator import summarize_by_profile
from .results_export import write_json, write_metrics_csv
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


@app.command("run-benchmark")
def run_benchmark(
    profile: str = typer.Option("faar_full", help=f"Profile: {', '.join(sorted(PROFILES))}"),
    max_examples: int = typer.Option(20, help="Limit examples for this run"),
    project_root: Path | None = typer.Option(None, help="Override repository root"),
    vlm_backend: str = typer.Option("openai", help="Visual fallback backend"),
    seed: int = typer.Option(42, help="Random seed"),
    api_enabled: bool = typer.Option(True, help="Enable API-backed runtime features"),
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    settings = AppSettings(project_root=project_root) if project_root else AppSettings()
    settings.recovery.vlm_backend = vlm_backend
    settings.recovery.api_enabled = api_enabled
    settings.validate_runtime_paths()
    settings.logs_dir = (settings.project_root / "logs/phase3").resolve()
    settings.artifacts_dir = (settings.project_root / "artifacts/phase3").resolve()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows = run_profile(settings, profile_name=profile, max_examples=max_examples)
    profile_summaries = summarize_by_profile(rows)
    summary = profile_summaries.get(
        profile,
        {"count": 0, "ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0, "visual_fallback_rate": 0.0},
    )
    out_summary = settings.artifacts_dir / f"{profile}_summary.json"
    write_json(out_summary, {"profile": profile, "summary": summary, "count": len(rows)})
    typer.echo(json.dumps({"profile": profile, "summary": summary, "count": len(rows)}, indent=2))
    typer.echo(f"\nSaved profile summary to {out_summary}")


@app.command("run-benchmark-all")
def run_benchmark_all(
    max_examples: int = typer.Option(20, help="Limit examples per profile"),
    project_root: Path | None = typer.Option(None, help="Override repository root"),
    vlm_backend: str = typer.Option("openai", help="Visual fallback backend"),
    seed: int = typer.Option(42, help="Random seed"),
    api_enabled: bool = typer.Option(True, help="Enable API-backed runtime features"),
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    settings = AppSettings(project_root=project_root) if project_root else AppSettings()
    settings.recovery.vlm_backend = vlm_backend
    settings.recovery.api_enabled = api_enabled
    settings.validate_runtime_paths()
    settings.logs_dir = (settings.project_root / "logs/phase3").resolve()
    settings.artifacts_dir = (settings.project_root / "artifacts/phase3").resolve()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    full_summary: dict[str, dict] = {}
    for profile_name in sorted(PROFILES):
        profile_rows = run_profile(settings, profile_name=profile_name, max_examples=max_examples)
        all_rows.extend(profile_rows)
        profile_summaries = summarize_by_profile(profile_rows)
        per_profile = profile_summaries.get(
            profile_name,
            {"count": 0, "ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0, "visual_fallback_rate": 0.0},
        )
        full_summary[profile_name] = per_profile
        write_json(settings.artifacts_dir / f"{profile_name}_summary.json", {"profile": profile_name, "summary": per_profile})

    summary = summarize_by_profile(all_rows)
    if not summary:
        summary = full_summary
    write_json(settings.artifacts_dir / "metrics_summary.json", {"profiles": summary, "generated_at_utc": datetime.now(UTC).isoformat()})
    csv_rows = [{"profile": name, **values} for name, values in summary.items()]
    write_metrics_csv(settings.artifacts_dir / "qa_metrics.csv", csv_rows)
    write_metrics_csv(settings.artifacts_dir / "retrieval_metrics.csv", csv_rows)
    write_json(
        settings.artifacts_dir / "ablation_summary.json",
        {
            "faar_full": summary.get("faar_full", {}),
            "faar_no_backtrack": summary.get("faar_no_backtrack", {}),
            "faar_no_vlm": summary.get("faar_no_vlm", {}),
            "faar_no_diagnosis": summary.get("faar_no_diagnosis", {}),
        },
    )
    typer.echo(json.dumps({"profiles": summary}, indent=2))
    typer.echo(f"\nSaved global summary to {settings.artifacts_dir / 'metrics_summary.json'}")


@app.command("aggregate-phase3")
def aggregate_phase3(
    project_root: Path | None = typer.Option(None, help="Override repository root"),
) -> None:
    settings = AppSettings(project_root=project_root) if project_root else AppSettings()
    base = settings.project_root / "logs/phase3"
    artifacts_dir = settings.project_root / "artifacts/phase3"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for profile_name in sorted(PROFILES):
        profile_dir = base / profile_name
        if not profile_dir.exists():
            continue
        for path in sorted(profile_dir.glob("*.json")):
            all_rows.append(json.loads(path.read_text()))
    summary = summarize_by_profile(all_rows)
    write_json(artifacts_dir / "metrics_summary.json", {"profiles": summary, "generated_at_utc": datetime.now(UTC).isoformat()})
    csv_rows = [{"profile": name, **values} for name, values in summary.items()]
    write_metrics_csv(artifacts_dir / "qa_metrics.csv", csv_rows)
    write_metrics_csv(artifacts_dir / "retrieval_metrics.csv", csv_rows)
    typer.echo(json.dumps({"profiles": summary}, indent=2))


@app.command("run-phase4")
def run_phase4(
    project_root: Path | None = typer.Option(None, help="Override repository root"),
    baseline_profile: str = typer.Option("naive_rag", help="Baseline profile for delta metrics"),
) -> None:
    settings = AppSettings(project_root=project_root) if project_root else AppSettings()
    logs_root = settings.project_root / "logs/phase3"
    artifacts_dir = settings.project_root / "artifacts/phase4"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows_by_profile = load_phase3_rows(logs_root, profiles=sorted(PROFILES))
    report = build_phase4_report(rows_by_profile, baseline_profile=baseline_profile)

    write_json(artifacts_dir / "phase4_summary.json", report)
    write_json(artifacts_dir / "claim_assessment.json", report["claim_assessment"])
    write_json(artifacts_dir / "case_studies.json", report["case_studies"])
    write_phase4_comparison_csv(artifacts_dir / "profile_deltas_vs_baseline.csv", report["baseline_comparison"])

    typer.echo(
        json.dumps(
            {
                "baseline_profile": baseline_profile,
                "claim_assessment": report["claim_assessment"],
                "artifact_dir": str(artifacts_dir),
            },
            indent=2,
        )
    )
