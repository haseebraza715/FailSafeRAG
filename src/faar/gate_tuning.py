from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .metrics import exact_match
from .run_io import atomic_write_text, canonical_digest

THETA_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)

GATE_PRECISION_MIN = 0.75
GATE_RECALL_MIN = 0.70

GATE_SOURCE_METRIC = "em"


@dataclass(frozen=True)
class GateExample:
    example_id: str
    top_reranker_score: float
    needs_recovery: bool


def load_gate_examples(path: Path) -> list[GateExample]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected result rows in {path}.")

    examples: list[GateExample] = []
    for row in rows:
        gate = row.get("gate") or {}
        score = row.get("top_reranker_score", gate.get("top_reranker_score"))
        if score is None:
            raise ValueError(f"Missing top_reranker_score for example {row.get('example_id', '<unknown>')}.")
        metrics = row.get("metrics") or {}
        if "em" in metrics:
            needs_recovery = float(metrics["em"]) < 1.0
        else:
            needs_recovery = exact_match(
                str(row.get("predicted_answer", row.get("answer", ""))),
                str(row.get("gold_answer", row.get("correct_answer", ""))),
            ) < 1.0
        examples.append(
            GateExample(str(row.get("example_id", "")), float(score), needs_recovery)
        )
    if not examples:
        raise ValueError("No rows available for gate threshold tuning.")
    return examples


def metrics_at_theta(examples: Iterable[GateExample], theta: float) -> dict[str, float | int]:
    values = list(examples)
    # Low top-1 reranker confidence means the controller should enter recovery.
    tp = sum(example.needs_recovery and example.top_reranker_score < theta for example in values)
    fp = sum(not example.needs_recovery and example.top_reranker_score < theta for example in values)
    fn = sum(example.needs_recovery and example.top_reranker_score >= theta for example in values)
    tn = sum(not example.needs_recovery and example.top_reranker_score >= theta for example in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "theta": theta,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def search_threshold(examples: Iterable[GateExample], theta_grid: Iterable[float] = THETA_GRID) -> dict[str, Any]:
    values = list(examples)
    grid = [metrics_at_theta(values, theta) for theta in theta_grid]
    winner = max(grid, key=lambda item: (float(item["f1"]), float(item["precision"]), float(item["recall"]), -float(item["theta"])))
    return {
        "signal": "BGE-reranker-v2-m3 top-1 score",
        "positive_class": "B0 exact-match error; route to recovery when score < theta",
        "validation_examples": len(values),
        "grid": grid,
        "winner": winner,
        "pass_criteria": {
            "precision_at_least": GATE_PRECISION_MIN,
            "recall_at_least": GATE_RECALL_MIN,
            "passed": winner["precision"] >= GATE_PRECISION_MIN and winner["recall"] >= GATE_RECALL_MIN,
        },
    }


def require_validation_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    run_spec = payload.get("run_spec") if isinstance(payload, dict) else None
    if not isinstance(run_spec, dict):
        raise ValueError("Gate threshold tuning accepts only a result JSON with a run_spec object.")
    if payload.get("smoke") is True or payload.get("paper_result") is False:
        raise ValueError(
            "Gate threshold tuning accepts only a real paper B0 result; "
            "smoke/development results are forbidden."
        )
    if run_spec.get("split") != "val":
        raise ValueError(
            "Gate threshold tuning accepts only a result JSON produced with run_spec.split='val'; test results are forbidden."
        )
    if run_spec.get("profile") != "naive_rag":
        raise ValueError(
            "Gate threshold tuning accepts only a B0 baseline result JSON with run_spec.profile='naive_rag'."
        )
    dataset = run_spec.get("dataset")
    if not (isinstance(dataset, str) and dataset.strip()):
        raise ValueError("Gate threshold tuning requires a non-empty run_spec.dataset.")
    model_provenance = run_spec.get("model_provenance")
    if not (isinstance(model_provenance, dict) and model_provenance):
        raise ValueError("Gate threshold tuning requires a non-empty run_spec.model_provenance object.")
    if "manifest_sha256" in run_spec and not (
        isinstance(run_spec["manifest_sha256"], str) and run_spec["manifest_sha256"].strip()
    ):
        raise ValueError("Gate threshold tuning requires run_spec.manifest_sha256 to be present when declared.")
    return payload


def _gate_relevant_source_digest(payload: dict[str, Any]) -> str:
    """Digest of the exact content the gate threshold is derived from.

    Covers run_spec plus the per-row gate signal (top reranker score) and the
    scored metrics. Volatile fields (timestamps, run ids, API usage, VLM
    provenance) are excluded so a semantically identical re-run of the
    validation B0 does not invalidate the lock, while any change to the
    threshold inputs still does.
    """
    run_spec = payload.get("run_spec")
    rows = payload.get("rows")
    if not isinstance(run_spec, dict) or not isinstance(rows, list):
        raise ValueError("Gate source must be a result payload with run_spec and rows.")
    gate_rows = [
        {
            "example_id": str(row.get("example_id", "")),
            "top_reranker_score": row.get(
                "top_reranker_score", (row.get("gate") or {}).get("top_reranker_score")
            ),
            "gate": row.get("gate"),
            "metrics": row.get("metrics"),
        }
        for row in rows
    ]
    return canonical_digest({"run_spec": run_spec, "rows": gate_rows})


def write_locked_threshold(path: Path, search: dict[str, Any], *, source: Path | None = None) -> None:
    if not bool((search.get("pass_criteria") or {}).get("passed")):
        raise ValueError("Gate threshold cannot be locked because validation precision/recall did not pass.")
    if source is None:
        declared = search.get("source_results")
        if not declared:
            raise ValueError("Gate threshold lock requires the source B0 result path via source= or search['source_results'].")
        source = Path(declared).resolve()
    else:
        source = Path(source).resolve()
    source_payload = require_validation_payload(source)
    source_spec = source_payload["run_spec"]
    winner = search["winner"]
    payload = {
        "source_split": "val",
        "source_path": str(source),
        "source_digest": _gate_relevant_source_digest(source_payload),
        "dataset": source_spec["dataset"],
        "split": source_spec["split"],
        "model_provenance": dict(source_spec["model_provenance"]),
        "gate_source_metric": GATE_SOURCE_METRIC,
        "signal": search["signal"],
        "threshold": winner["theta"],
        "precision": winner["precision"],
        "recall": winner["recall"],
        "f1": winner["f1"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _text_relevant_provenance(model_provenance: dict[str, Any]) -> dict[str, Any]:
    return {key: model_provenance[key] for key in ("embedding", "reranker") if key in model_provenance}


def _verify_locked_provenance(
    payload: dict[str, Any],
    path: Path,
    *,
    dataset: str | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> None:
    if payload.get("source_split") != "val":
        raise ValueError(
            f"Locked gate threshold at {path} must declare source_split='val'; test data cannot tune the gate."
        )
    source_path = payload.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError(
            f"Locked gate threshold at {path} records no source_path provenance; re-tune with scripts/experiments/tune_gate.py."
        )
    source = Path(source_path)
    if not source.is_file():
        raise ValueError(
            f"Locked gate threshold at {path} references missing source result file: {source_path}"
        )
    recorded_digest = payload.get("source_digest")
    if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
        raise ValueError(f"Locked gate threshold at {path} records no source_digest provenance.")
    try:
        source_payload = json.loads(source.read_text(encoding="utf-8"))
        actual_digest = _gate_relevant_source_digest(source_payload)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Locked gate threshold at {path} cannot re-read its source result {source_path}: {exc}"
        ) from exc
    if actual_digest != recorded_digest:
        raise ValueError(
            f"Locked gate threshold at {path} source_digest no longer matches {source_path}; "
            "the source B0 scores/metrics were modified or the lock was tampered with."
        )
    recorded_split = payload.get("split")
    if recorded_split != "val":
        raise ValueError(
            f"Locked gate threshold at {path} records split={recorded_split!r}; the gate threshold must be tuned on split='val'."
        )
    recorded_dataset = payload.get("dataset")
    if not (isinstance(recorded_dataset, str) and recorded_dataset.strip()):
        raise ValueError(f"Locked gate threshold at {path} records no dataset provenance.")
    recorded_mp = payload.get("model_provenance")
    if not (isinstance(recorded_mp, dict) and recorded_mp):
        raise ValueError(f"Locked gate threshold at {path} records no model_provenance provenance.")
    if dataset is not None and recorded_dataset != dataset:
        raise ValueError(
            f"Locked gate threshold at {path} was tuned on dataset {recorded_dataset!r}; the current run uses dataset {dataset!r}."
        )
    if model_provenance is not None:
        recorded_text = _text_relevant_provenance(recorded_mp)
        current_text = _text_relevant_provenance(model_provenance)
        if recorded_text != current_text:
            raise ValueError(
                f"Locked gate threshold at {path} retrieval model provenance does not match the current run; "
                "re-tune the gate with the same embedding and reranker configuration."
            )


def require_paper_gate_threshold(
    path: Path,
    *,
    dataset: str | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"Gate-dependent paper runs require a locked gate threshold at {path}; run scripts/experiments/tune_gate.py on validation results first."
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Locked gate threshold at {path} must be a JSON object.")
    _verify_locked_provenance(payload, path, dataset=dataset, model_provenance=model_provenance)
    try:
        threshold = float(payload["threshold"])
        precision = float(payload["precision"])
        recall = float(payload["recall"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Locked gate threshold at {path} has missing or invalid numeric fields.") from exc
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (threshold, precision, recall)):
        raise ValueError(f"Locked gate threshold at {path} must contain finite values in [0, 1].")
    if precision < GATE_PRECISION_MIN or recall < GATE_RECALL_MIN:
        raise ValueError(
            f"Locked gate threshold at {path} fails the paper bar: precision={precision} (>= {GATE_PRECISION_MIN}) "
            f"and recall={recall} (>= {GATE_RECALL_MIN}) are required."
        )
    return payload
