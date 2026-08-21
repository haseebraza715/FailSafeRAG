#!/usr/bin/env python3
"""Merge per-shard baseline outputs into one unsharded result file.

Multi-GPU usage: launch one scripts/experiments/run.py process per GPU with
--shard-index i --num-shards N, then merge the shard outputs:

    python cluster/merge_shards.py --out results/b2.json --baseline results/b0.json \
        results/b2_shard1of8.json ... results/b2_shard8of8.json

The merged file is validated against the baseline run_spec, gets an
unsharded run_spec (shard_index/num_shards reset to null), recomputed
summary metrics and harm_rate, and is written atomically so it can be
validated by cluster/run_baselines.py on later resumes.

The full shard set must be supplied: every index in range(num_shards) must be
declared exactly once. A B0 merge must match the exact locked manifest hash
and selected example ids; later stages must match B0. A partial set is
rejected before shard metadata is cleared or the merged file is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.final_analysis import _harm_rate
from faar.results_aggregator import summarize_api_usage, summarize_examples
from faar.run_io import atomic_write_text

RUN_SPEC_MATCH_KEYS = (
    "profile",
    "dataset",
    "split",
    "seed",
    "max_examples",
    "embedding_model",
    "reranker",
    "ocr_engine",
    "model_provenance",
    "manifest_sha256",
)

EXPECTED_LABELS = {
    "naive_rag": "Text-only RAG",
    "faar_always_vlm": "Always-VLM",
    "faar_no_diagnosis": "Random recovery",
}


def _require_scored_metrics(row: dict[str, Any], key: str, path: Path) -> None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or metrics.get(key) is None:
        raise SystemExit(f"{path} row {row.get('example_id')!r} has no metrics.{key}.")
    try:
        score = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path} row {row.get('example_id')!r} has invalid metrics.{key}.") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SystemExit(f"{path} row {row.get('example_id')!r} has metrics.{key} outside [0, 1].")


def _load_shard(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"shard is not readable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise SystemExit(f"shard must be a result payload with a rows list: {path}")
    run_spec = payload.get("run_spec")
    if not isinstance(run_spec, dict):
        raise SystemExit(f"shard has no run_spec provenance: {path}")
    return payload


def _validate_shard_declaration(path: Path, spec: dict[str, Any], reference_spec: dict[str, Any]) -> int:
    """Validate one shard's index/count declaration; return its shard_index."""
    for key in ("shard_index", "num_shards"):
        if key not in spec:
            raise SystemExit(f"shard {path} run_spec is missing {key}; not a sharded output.")
    index = spec["shard_index"]
    num_shards = spec["num_shards"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise SystemExit(
            f"shard {path} run_spec.shard_index must be a non-boolean integer; received {index!r}."
        )
    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise SystemExit(
            f"shard {path} run_spec.num_shards must be a non-boolean integer; received {num_shards!r}."
        )
    if num_shards < 1:
        raise SystemExit(
            f"shard {path} declares num_shards={num_shards}; expected a positive integer."
        )
    if num_shards != reference_spec["num_shards"]:
        raise SystemExit(
            f"shard {path} declares num_shards={num_shards}; expected {reference_spec['num_shards']}."
        )
    if not 0 <= index < num_shards:
        raise SystemExit(
            f"shard {path} declares shard_index={index}; expected an index in [0, {num_shards})."
        )
    return index


def _validate_shard_set(
    shard_paths: list[Path], payloads: list[dict[str, Any]], num_shards: int
) -> None:
    """Require exactly one shard per index in range(num_shards), no duplicates."""
    declared: dict[int, Path] = {}
    duplicates: list[str] = []
    for path, payload in zip(shard_paths, payloads):
        index = int(payload["run_spec"]["shard_index"])
        if index in declared:
            duplicates.append(f"shard index {index} declared by {declared[index]} and {path}")
        else:
            declared[index] = path
    missing = [index for index in range(num_shards) if index not in declared]
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing shard indices {missing}")
        details.extend(duplicates)
        raise SystemExit(
            f"incomplete shard set for num_shards={num_shards}: " + "; ".join(details)
        )


def _validated_baseline_ids(payload: dict, path: Path) -> list[str]:
    """Derive expected example ids from a B0 result, validating its structure.

    The payload must be a JSON object whose rows is a list of objects, each
    with a unique nonempty example_id. A malformed baseline is never treated
    as an authoritative expected-ID source.
    """
    if not isinstance(payload, dict):
        raise SystemExit(f"baseline result must be a JSON object: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"baseline result has no rows list: {path}")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"baseline row {index} is not an object: {path}")
        value = row.get("example_id")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"baseline row {index} has a missing or empty example_id: {path}")
        ids.append(value.strip())
    if len(ids) != len(set(ids)):
        raise SystemExit(f"baseline contains duplicate example_ids: {path}")
    return ids


def _validate_baseline_provenance(
    payload: dict[str, Any], path: Path, reference_spec: dict[str, Any]
) -> None:
    if payload.get("profile") != "naive_rag":
        raise SystemExit(f"baseline must be a naive_rag B0 result: {path}")
    spec = payload.get("run_spec")
    if not isinstance(spec, dict):
        raise SystemExit(f"baseline result has no run_spec provenance: {path}")
    if spec.get("profile") != "naive_rag":
        raise SystemExit(f"baseline run_spec.profile must be naive_rag: {path}")
    if spec.get("shard_index") is not None or spec.get("num_shards") is not None:
        raise SystemExit(f"baseline must be an unsharded merged B0 result: {path}")
    for key in RUN_SPEC_MATCH_KEYS:
        if key == "profile":
            continue
        if key not in spec:
            raise SystemExit(f"baseline {path} run_spec is missing {key}.")
        if spec[key] != reference_spec[key]:
            raise SystemExit(
                f"baseline {path} run_spec.{key}={spec[key]!r} does not match "
                f"shard run_spec.{key}={reference_spec[key]!r}."
            )


def _normalise_dataset(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "")


def _validated_manifest_ids(path: Path, reference_spec: dict[str, Any]) -> list[str]:
    """Derive the exact run selection from the manifest named by run_spec."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"manifest is unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"manifest must be a JSON object: {path}")
    expected_hash = reference_spec.get("manifest_sha256")
    actual_hash = hashlib.sha256(raw).hexdigest()
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise SystemExit(
            f"manifest SHA-256 does not match the shard run_spec: {path} "
            f"(expected {expected_hash!r}, found {actual_hash})."
        )
    manifest_dataset = payload.get("dataset")
    if not isinstance(manifest_dataset, str) or _normalise_dataset(manifest_dataset) != _normalise_dataset(
        str(reference_spec.get("dataset", ""))
    ):
        raise SystemExit(
            f"manifest dataset {manifest_dataset!r} does not match shard dataset "
            f"{reference_spec.get('dataset')!r}: {path}"
        )
    manifest_split = payload.get("split")
    if manifest_split != reference_spec.get("split"):
        raise SystemExit(
            f"manifest split {manifest_split!r} does not match shard split "
            f"{reference_spec.get('split')!r}: {path}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit(f"manifest has no records list: {path}")
    ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"manifest record {index} is not an object: {path}")
        value = record.get("example_id")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"manifest record {index} has a missing or empty example_id: {path}")
        ids.append(value.strip())
    if len(ids) != len(set(ids)):
        raise SystemExit(f"manifest contains duplicate example_ids: {path}")
    max_examples = reference_spec.get("max_examples")
    if max_examples is not None:
        if isinstance(max_examples, bool) or not isinstance(max_examples, int):
            raise SystemExit(
                "shard run_spec.max_examples must be a non-boolean integer or null; "
                f"received {max_examples!r}."
            )
        ids = sorted(ids)[: max(0, max_examples)]
    else:
        ids = sorted(ids)
    return ids


def _validate_expected_id_source(expected_example_ids: list[str]) -> None:
    for index, value in enumerate(expected_example_ids):
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"expected example id at index {index} is empty.")
    if len(expected_example_ids) != len(set(expected_example_ids)):
        raise SystemExit("expected example ids contain duplicates.")


def _validate_expected_ids(
    merged_rows: list[dict[str, Any]], expected_example_ids: list[str]
) -> None:
    _validate_expected_id_source(expected_example_ids)
    merged_ids = [str(row["example_id"]) for row in merged_rows]
    expected_set = set(expected_example_ids)
    merged_set = set(merged_ids)
    missing = [item for item in expected_example_ids if item not in merged_set]
    extra = [item for item in merged_ids if item not in expected_set]
    if missing or extra:
        raise SystemExit(
            f"merged rows do not match the expected example ids from the baseline "
            f"(missing {len(missing)}, extra {len(extra)})."
        )


def _validate_merged_not_empty(rows: list[dict[str, Any]], expected_example_ids: list[str] | None) -> None:
    if rows:
        return
    if expected_example_ids is None:
        raise SystemExit(
            "merged shards contain no rows; refusing to publish an entirely empty result. "
            "An empty merge is only allowed when an authoritative expected-ID source "
            "explicitly declares an empty selection."
        )


def merge_shards(
    shard_paths: list[Path],
    *,
    expected_example_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not shard_paths:
        raise SystemExit("at least one shard file is required")
    payloads = [_load_shard(path) for path in shard_paths]
    reference_spec = payloads[0]["run_spec"]
    for index, (path, payload) in enumerate(zip(shard_paths, payloads)):
        spec = payload["run_spec"]
        for key in RUN_SPEC_MATCH_KEYS:
            if key not in spec:
                raise SystemExit(f"shard {path} run_spec is missing {key}.")
            if index > 0 and spec[key] != reference_spec[key]:
                raise SystemExit(
                    f"shard {path} run_spec.{key}={spec[key]!r} does not match "
                    f"{shard_paths[0]} run_spec.{key}={reference_spec[key]!r}."
                )
        _validate_shard_declaration(path, spec, reference_spec)
        if payload.get("profile") != spec["profile"]:
            raise SystemExit(
                f"shard {path} top-level profile {payload.get('profile')!r} does not match "
                f"run_spec.profile {spec['profile']!r}."
            )
        expected_label = EXPECTED_LABELS.get(str(spec["profile"]))
        if expected_label is not None and payload.get("label") != expected_label:
            raise SystemExit(
                f"shard {path} label {payload.get('label')!r} does not match "
                f"profile {spec['profile']!r} label {expected_label!r}."
            )
        if index > 0 and payload.get("label") != payloads[0].get("label"):
            raise SystemExit(
                f"shard {path} label {payload.get('label')!r} does not match "
                f"{shard_paths[0]} label {payloads[0].get('label')!r}."
            )
    _validate_shard_set(shard_paths, payloads, reference_spec["num_shards"])
    seen: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for path, payload in zip(shard_paths, payloads):
        for row in payload["rows"]:
            if not isinstance(row, dict):
                raise SystemExit(f"shard {path} contains a row that is not an object.")
            example_id = str(row.get("example_id", ""))
            if not example_id:
                raise SystemExit(f"shard {path} contains a row without example_id.")
            if (row.get("action_outcome") or {}).get("status") == "failed" or row.get("error"):
                raise SystemExit(
                    f"shard {path} contains failed row {example_id!r}; "
                    "failed examples cannot be merged or scored."
                )
            _require_scored_metrics(row, "em", path)
            _require_scored_metrics(row, "f1", path)
            if example_id in seen:
                raise SystemExit(
                    f"duplicate example_id {example_id!r} in shards {seen[example_id]} and {path}."
                )
            seen[example_id] = path
        rows.extend(payload["rows"])
    if expected_example_ids is not None:
        _validate_expected_ids(rows, expected_example_ids)
    else:
        _validate_merged_not_empty(rows, expected_example_ids)
    examples_summary = summarize_examples(rows)
    api = summarize_api_usage(rows)
    merged = {
        "label": payloads[0].get("label", "merged"),
        "profile": payloads[0].get("profile"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_spec": {**reference_spec, "shard_index": None, "num_shards": None},
        "summary": {
            "EM": examples_summary["em"],
            "F1": examples_summary["f1"],
            "vlm_rate": examples_summary["vlm_rate"],
            "harm_rate": 0.0 if reference_spec["profile"] == "naive_rag" else None,
            "api_requests": api["api_requests"],
            "prompt_tokens": api["prompt_tokens"],
            "completion_tokens": api["completion_tokens"],
            "cost_usd": api["cost_usd"],
            "runtime_sec": round(
                sum(float((payload.get("summary") or {}).get("runtime_sec") or 0.0) for payload in payloads),
                4,
            ),
        },
        "rows": sorted(rows, key=lambda row: str(row.get("example_id", ""))),
        "merged_from": [str(path) for path in shard_paths],
    }
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Merged unsharded output path")
    parser.add_argument("--baseline", type=Path, default=None, help="Matching B0 result used to compute harm_rate and validate merged example ids")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Locked benchmark manifest used to validate a B0 merge",
    )
    parser.add_argument("shards", nargs="+", type=Path, help="Shard result JSON files")
    args = parser.parse_args(argv)
    first_payload = _load_shard(args.shards[0])
    is_b0 = first_payload["run_spec"].get("profile") == "naive_rag"
    if is_b0:
        if args.manifest is None:
            raise SystemExit("B0 shard merge requires --manifest as its authoritative expected-ID source.")
        if args.baseline is not None:
            raise SystemExit("B0 shard merge does not accept --baseline; pass its locked --manifest.")
    elif args.baseline is None:
        raise SystemExit("Non-B0 shard merge requires --baseline to validate example ids and harm_rate.")
    if args.manifest is not None and not is_b0:
        raise SystemExit("--manifest is only accepted for B0 shard merges; use --baseline for later stages.")
    expected_example_ids = None
    baseline_payload = None
    if args.manifest is not None:
        if not args.manifest.is_file():
            raise SystemExit(f"manifest does not exist: {args.manifest}")
        expected_example_ids = _validated_manifest_ids(args.manifest, first_payload["run_spec"])
    if args.baseline is not None:
        if not args.baseline.is_file():
            raise SystemExit(f"baseline result does not exist: {args.baseline}")
        try:
            baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"baseline result is unreadable JSON: {args.baseline}: {exc}") from exc
        expected_example_ids = _validated_baseline_ids(baseline_payload, args.baseline)
        _validate_baseline_provenance(
            baseline_payload, args.baseline, first_payload["run_spec"]
        )
    payload = merge_shards(args.shards, expected_example_ids=expected_example_ids)
    if baseline_payload is not None:
        try:
            payload["summary"]["harm_rate"] = _harm_rate(payload["rows"], baseline_payload["rows"])
        except ValueError as exc:
            raise SystemExit(f"cannot compute harm_rate against baseline {args.baseline}: {exc}") from exc
    atomic_write_text(args.out, json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
