from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import load_rows

from faar.api_logging import ANTHROPIC_VLM_BACKENDS, vlm_cost_rates
from faar.benchmarks import load_benchmark_repository
from faar.experiment_runner import run_profile
from faar.gate_tuning import require_paper_gate_threshold
from faar.metrics import token_f1
from faar.operations import install_graceful_termination_handler, startup_report
from faar.results_aggregator import summarize_api_usage, summarize_examples
from faar.run_io import atomic_write_text, select_shard, shard_label
from faar.settings import AppSettings, resolve_retrieval_models
from faar.visual_baselines import run_visual_baseline

BASELINE_MAP = {
    ("off", "off"): ("B0", "naive_rag", "Text-only RAG"),
    ("off", "always_vlm"): ("B1", "faar_always_vlm", "Always-VLM"),
    ("on", "random_type"): ("B2", "faar_no_diagnosis", "Random recovery"),
}

GATE_BYPASS_PROFILES = {"naive_rag", "faar_always_vlm", "faar_no_gate"}
BASELINE_MATCH_KEYS = (
    "dataset",
    "split",
    "seed",
    "max_examples",
    "shard_index",
    "num_shards",
    "embedding_model",
    "reranker",
    "ocr_engine",
    "model_provenance",
    "manifest_sha256",
)

ABLATION_PROFILE_MAP = {
    "no_gate": "faar_no_gate",
    "no_diagnosis": "faar_no_diagnosis",
    "no_wordlevel_llm": "faar_symspell",
    "no_semantic_retry": "faar_no_backtrack",
}


def _run_profile_for_args(args: argparse.Namespace) -> str:
    if args.mode in {"colpali", "visrag"}:
        return args.mode
    if args.mode == "faar":
        return "faar_full"
    if args.ablate:
        return ABLATION_PROFILE_MAP[args.ablate]
    key = (args.gate or "on", args.recovery or "off")
    if key not in BASELINE_MAP:
        raise SystemExit(f"Unsupported baseline combination: gate={args.gate}, recovery={args.recovery}")
    return BASELINE_MAP[key][1]


def _require_gate_threshold(settings: AppSettings, profile: str, run_spec: dict[str, Any]) -> float | None:
    if profile in GATE_BYPASS_PROFILES:
        return None
    try:
        payload = require_paper_gate_threshold(
            settings.gate_threshold_path,
            dataset=str(run_spec.get("dataset") or "") or None,
            model_provenance=run_spec.get("model_provenance") or None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    threshold = float(payload["threshold"])
    settings.gate.quality_threshold = threshold
    return threshold


def _validate_baseline(
    baseline_path: Path | None,
    *,
    label: str,
    run_spec: dict[str, Any],
) -> None:
    if baseline_path is None:
        raise SystemExit(
            f"Missing required --baseline: {label} needs a matching B0 result to define harm_rate."
        )
    if not baseline_path.is_file():
        raise SystemExit(f"Matching B0 result does not exist: {baseline_path}")
    try:
        baseline_payload = json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Matching B0 result is unreadable: {baseline_path}") from exc
    if not isinstance(baseline_payload, dict) or baseline_payload.get("profile") != "naive_rag":
        raise SystemExit(f"Matching baseline must be a B0 naive_rag result: {baseline_path}")
    baseline_spec = baseline_payload.get("run_spec")
    if not isinstance(baseline_spec, dict):
        raise SystemExit(f"Matching B0 result has no run_spec provenance: {baseline_path}")
    for key in BASELINE_MATCH_KEYS:
        if key not in baseline_spec:
            if key == "manifest_sha256":
                continue
            raise SystemExit(f"Matching B0 result is missing run_spec.{key}: {baseline_path}")
        expected = run_spec.get(key)
        observed = baseline_spec[key]
        if observed != expected:
            raise SystemExit(
                f"Baseline mismatch: expected {key}={expected!r}, but {baseline_path} records {observed!r}."
            )


def _require_key_for_paid_vlm(vlm_backend: str) -> None:
    if vlm_backend in ANTHROPIC_VLM_BACKENDS and not (
        os.getenv("ANTHROPIC_API_KEY") or ""
    ).strip():
        raise SystemExit("Missing required key: ANTHROPIC_API_KEY for VLM_BACKEND=claude-sonnet-4-5.")
    if vlm_backend == "openai" and not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise SystemExit("Missing required key: OPENAI_API_KEY for VLM_BACKEND=openai.")
    if vlm_backend not in {"openai"} | ANTHROPIC_VLM_BACKENDS:
        raise SystemExit(f"Unsupported paid VLM backend for paper runs: {vlm_backend!r}.")


def _settings_from_args(args: argparse.Namespace) -> AppSettings:
    settings = AppSettings(project_root=Path.cwd())
    settings.recovery.vlm_backend = args.vlm or os.getenv("VLM_BACKEND", settings.recovery.vlm_backend)
    if settings.recovery.vlm_backend.startswith("claude"):
        settings.recovery.anthropic_model = settings.recovery.vlm_backend
    if settings.recovery.vlm_backend == "openai":
        settings.recovery.openai_model = os.getenv("OPENAI_MODEL", settings.recovery.openai_model)
    settings.retrieval.embedding_model, settings.retrieval.reranker = resolve_retrieval_models(
        embed=args.embed,
        reranker=args.reranker,
    )
    settings.recovery.ocr_engine = args.ocr or os.getenv("OCR_ENGINE", settings.recovery.ocr_engine)
    settings.recovery.log_vlm_calls = True
    settings.experiment.random_seed = args.seed
    if args.wordlevel_fallback:
        settings.experiment.wordlevel_fallback = args.wordlevel_fallback
        settings.recovery.wordlevel_fallback = args.wordlevel_fallback
    settings.validate_openai_snapshot()
    settings.validate_model_revisions(include_visual=args.mode if args.mode in {"colpali", "visrag"} else None)
    return settings


def _run_profile_to_result(
    settings: AppSettings,
    profile: str,
    out: Path,
    label: str,
    max_examples: int | None,
    run_spec: dict[str, Any],
    dataset: str,
    split: str,
    *,
    resume: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    run_spec["gate_threshold"] = _require_gate_threshold(settings, profile, run_spec)
    start = time.perf_counter()
    rows = run_profile(
        settings,
        profile_name=profile,
        max_examples=max_examples,
        output_dir=out.parent / f"{out.stem}_rows",
        dataset=dataset,
        split=split,
        resume=resume,
        shard_index=shard_index,
        num_shards=num_shards,
        seed=int(run_spec.get("seed", settings.experiment.random_seed)),
    )
    summary = summarize_examples(rows)
    api_totals = summarize_api_usage(rows)
    payload = {
        "label": label,
        "profile": profile,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_spec": run_spec,
        "summary": {
            "EM": summary["em"],
            "F1": summary["f1"],
            "vlm_rate": summary["vlm_rate"],
            "harm_rate": 0.0 if profile == "naive_rag" else None,
            "api_requests": api_totals["api_requests"],
            "prompt_tokens": api_totals["prompt_tokens"],
            "completion_tokens": api_totals["completion_tokens"],
            "cost_usd": api_totals["cost_usd"],
            "runtime_sec": round(time.perf_counter() - start, 4),
        },
        "rows": rows,
    }
    if publish:
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, json.dumps(payload, indent=2) + "\n")
    return payload


def _baseline_harm_rate(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> float:
    if not rows:
        raise ValueError("harm_rate cannot be recomputed against a baseline without per-example rows.")
    baseline_by_id = {row.get("example_id"): row for row in baseline_rows}
    harmed = 0.0
    for row in rows:
        row_metrics = row.get("metrics") or {}
        prediction = row.get("predicted_answer", row.get("answer", ""))
        gold = row.get("gold_answer", row.get("correct_answer", ""))
        f1 = float(row_metrics.get("f1", token_f1(prediction, gold)))
        baseline = baseline_by_id.get(row.get("example_id"))
        baseline_metrics = baseline.get("metrics") or {}
        baseline_prediction = baseline.get("predicted_answer", baseline.get("answer", ""))
        baseline_gold = baseline.get("gold_answer", baseline.get("correct_answer", ""))
        baseline_f1 = float(baseline_metrics.get("f1", token_f1(baseline_prediction, baseline_gold)))
        harmed += 1.0 if f1 < baseline_f1 else 0.0
    return round(harmed / len(rows), 4)


def _apply_baseline_harm(
    payload: dict[str, Any],
    output_path: Path,
    baseline_path: Path | None,
    *,
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    if baseline_path is None:
        return payload
    _validate_baseline(
        baseline_path,
        label=str(payload.get("label") or "run"),
        run_spec=run_spec,
    )
    result_rows = payload.get("rows")
    if not isinstance(result_rows, list):
        raise SystemExit(
            f"Result payload is missing in-memory rows for harm computation "
            f"(got {type(result_rows).__name__}); refusing to re-read the output path."
        )
    result_ids = _validated_example_ids(result_rows, output_path)
    baseline_rows = load_rows(baseline_path)
    baseline_ids = _validated_example_ids(baseline_rows, baseline_path)
    missing = sorted(result_ids - baseline_ids)
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(
            f"Baseline {baseline_path} does not cover every result example ID; "
            f"missing {len(missing)} IDs: {preview}."
        )
    payload["summary"]["harm_rate"] = _baseline_harm_rate(result_rows, baseline_rows)
    atomic_write_text(output_path, json.dumps(payload, indent=2) + "\n")
    return payload


def _validated_example_ids(rows: list[dict[str, Any]], path: Path) -> set[str]:
    values = [str(row.get("example_id") or "").strip() for row in rows]
    if any(not value for value in values):
        raise SystemExit(f"Result contains a missing example_id: {path}")
    if len(values) != len(set(values)):
        raise SystemExit(f"Result contains duplicate example_id values: {path}")
    return set(values)


def _run_visual_baseline_to_result(
    settings: AppSettings,
    mode: str,
    out: Path,
    max_examples: int | None,
    run_spec: dict[str, Any],
    dataset: str,
    split: str,
    *,
    resume: bool = False,
    publish: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo = load_benchmark_repository(settings.project_root, dataset, split)
    rows = run_visual_baseline(
        settings,
        repo,
        mode,
        max_examples=max_examples,
        resume=resume,
        output_dir=out.parent / f"{out.stem}_rows",
    )
    if not isinstance(rows, list):
        raise SystemExit(
            f"run_visual_baseline({mode}) returned {type(rows).__name__}; "
            "expected list[dict] per-example rows."
        )
    summary = summarize_examples(rows)
    api_totals = summarize_api_usage(rows)
    payload = {
        "label": mode,
        "profile": mode,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_spec": run_spec,
        "summary": {
            "EM": summary["em"],
            "F1": summary["f1"],
            "vlm_rate": summary["vlm_rate"],
            "harm_rate": None,
            "api_requests": api_totals["api_requests"],
            "prompt_tokens": api_totals["prompt_tokens"],
            "completion_tokens": api_totals["completion_tokens"],
            "cost_usd": api_totals["cost_usd"],
            "runtime_sec": round(time.perf_counter() - started, 4),
        },
        "rows": rows,
    }
    if publish:
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, json.dumps(payload, indent=2) + "\n")
    return payload


def _sharded_out_path(out: Path, shard_index: int | None, num_shards: int | None) -> Path:
    if num_shards is None:
        if shard_index is not None:
            raise SystemExit("--shard-index requires --num-shards.")
        return out
    if shard_index is None:
        raise SystemExit("--num-shards requires --shard-index.")
    try:
        select_shard([], shard_index, num_shards)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return out.with_name(f"{out.stem}_{shard_label(shard_index, num_shards)}{out.suffix}")


def _text_run_kwargs(resume: bool, shard_index: int | None, num_shards: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if resume:
        kwargs["resume"] = True
    if num_shards is not None:
        kwargs["shard_index"] = shard_index
        kwargs["num_shards"] = num_shards
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="FAAR AAAI experiment runner compatibility wrapper.")
    parser.add_argument("--gate", choices=["on", "off"])
    parser.add_argument("--recovery", choices=["off", "always_vlm", "random_type"])
    parser.add_argument("--mode", choices=["faar", "colpali", "visrag"])
    parser.add_argument("--dataset", default="ohrbench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ocr")
    parser.add_argument("--embed")
    parser.add_argument("--reranker")
    parser.add_argument("--vlm")
    parser.add_argument("--ablate", choices=["no_gate", "no_diagnosis", "no_wordlevel_llm", "no_semantic_retry"])
    parser.add_argument("--wordlevel_fallback")
    parser.add_argument("--baseline", type=Path, help="Matching B0 result used to compute harm_rate.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reuse per-example checkpoints whose run fingerprint matches.")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based shard to run; requires --num-shards.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shards; requires --shard-index.")
    args = parser.parse_args()

    install_graceful_termination_handler()
    random.seed(args.seed)
    np.random.seed(args.seed)
    settings = _settings_from_args(args)
    settings.validate_runtime_paths()
    startup_report(settings=settings)
    run_spec = {
        "profile": _run_profile_for_args(args),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "gate_threshold": None,
        "max_examples": args.max_examples,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "embedding_model": settings.retrieval.embedding_model,
        "reranker": settings.retrieval.reranker,
        "ocr_engine": settings.recovery.ocr_engine,
        "vlm_backend": settings.recovery.vlm_backend,
        "vlm_model": settings.vlm_request_model(),
        "vlm_cost_rates": vlm_cost_rates(settings.recovery.vlm_backend),
        "model_provenance": settings.model_provenance(),
    }
    repo = load_benchmark_repository(settings.project_root, args.dataset, args.split)
    run_spec["manifest_sha256"] = getattr(repo, "manifest_sha256", None)

    if args.mode in {"colpali", "visrag"}:
        if args.shard_index is not None or args.num_shards is not None:
            raise SystemExit("Shard flags are not supported for visual modes.")
        _validate_baseline(
            args.baseline,
            label=f"--mode {args.mode}",
            run_spec=run_spec,
        )
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_visual_baseline_to_result(
            settings,
            args.mode,
            args.out,
            args.max_examples,
            run_spec,
            args.dataset,
            args.split,
            resume=args.resume,
            publish=False,
        )
        payload = _apply_baseline_harm(
            payload,
            args.out,
            args.baseline,
            run_spec=run_spec,
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    text_out = _sharded_out_path(args.out, args.shard_index, args.num_shards)
    text_kwargs = _text_run_kwargs(args.resume, args.shard_index, args.num_shards)

    if args.mode == "faar":
        _validate_baseline(
            args.baseline,
            label="--mode faar",
            run_spec=run_spec,
        )
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_profile_to_result(
            settings, "faar_full", text_out, f"FAAR {args.dataset} {args.split}", args.max_examples, run_spec, args.dataset, args.split, publish=False, **text_kwargs
        )
        payload = _apply_baseline_harm(
            payload,
            text_out,
            args.baseline,
            run_spec=run_spec,
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    if args.ablate:
        ablation_profile = ABLATION_PROFILE_MAP[args.ablate]
        _validate_baseline(
            args.baseline,
            label=f"--ablate {args.ablate}",
            run_spec=run_spec,
        )
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_profile_to_result(
            settings, ablation_profile, text_out, args.ablate, args.max_examples, run_spec, args.dataset, args.split, publish=False, **text_kwargs
        )
        payload = _apply_baseline_harm(
            payload,
            text_out,
            args.baseline,
            run_spec=run_spec,
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    key = (args.gate or "on", args.recovery or "off")
    if key not in BASELINE_MAP:
        raise SystemExit(f"Unsupported baseline combination: gate={args.gate}, recovery={args.recovery}")
    baseline_id, profile, label = BASELINE_MAP[key]
    if baseline_id == "B0" and args.baseline is not None:
        raise SystemExit("B0 defines harm_rate and does not accept --baseline.")
    if baseline_id in {"B1", "B2"}:
        _validate_baseline(
            args.baseline,
            label=label,
            run_spec=run_spec,
        )
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
    payload = _run_profile_to_result(settings, profile, text_out, label, args.max_examples, run_spec, args.dataset, args.split, publish=baseline_id == "B0", **text_kwargs)
    payload = _apply_baseline_harm(
        payload,
        text_out,
        args.baseline,
        run_spec=run_spec,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
