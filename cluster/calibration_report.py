#!/usr/bin/env python3
"""Calibration reporting and runtime/storage projections.

`build` consumes a calibration checkpoint (the bounded 108-page single-document
run produced by scripts/data/prepare_benchmark_assets.py, or a prepare_assets.py shard
checkpoint) and emits a machine-readable summary:

- commit SHA, non-secret configuration, and locked split/manifest hashes;
- GPU model and total/free VRAM from an optional preflight report;
- peak allocated/reserved VRAM and peak process RSS when measurable;
- per-stage and total wall runtime;
- documents and pages attempted/completed/failed;
- rendered-image and OCR-output storage, and HF cache growth when measurable;
- per-page OCR throughput distributions (min/p50/p95/max/mean);
- failure and retry counts, and whether the run resumed from a checkpoint.

`project` consumes that summary and projects validation (549 documents / 7,037
pages) and test (567 documents / 6,849 pages) preparation, keeping Docling
document-level scaling and page-level rendering/OCR scaling as separate
components. It never merges them into a single rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.dataset_paths import SPLIT_RELATIVE_PATH
from faar.run_io import atomic_write_text

SCHEMA_VERSION = 1
DEFAULT_HEADROOM_FRACTION = 0.25
DEFAULT_WALLTIME_HOURS_PER_SHARD = 24.0
DEFAULT_VAL = {"documents": 549, "pages": 7037}
DEFAULT_TEST = {"documents": 567, "pages": 6849}
RELEVANT_ENV_KEYS = (
    "FAAR_GPU_BUDGET_GB",
    "FAAR_MIN_GPU_FREE_GB",
    "FAAR_MAX_RSS_GB",
    "FAAR_MAX_GPU_MEMORY_FRACTION",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "HF_HOME",
    "VLM_BACKEND",
)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)

    def quantile(q: float) -> float:
        position = q * (n - 1)
        lower = int(position)
        upper = min(lower + 1, n - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "n": n,
        "min": round(ordered[0], 4),
        "p50": round(quantile(0.5), 4),
        "p95": round(quantile(0.95), 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / n, 4),
    }


def _git_commit_sha(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _non_secret_environment() -> dict[str, str | None]:
    return {key: os.getenv(key) for key in RELEVANT_ENV_KEYS if os.getenv(key) is not None}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def build_summary(
    *,
    checkpoint_path: Path,
    project_root: Path,
    preflight_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint = _load_json(checkpoint_path, "checkpoint")
    completed = checkpoint.get("completed")
    failed = checkpoint.get("failed")
    if not isinstance(completed, dict):
        raise SystemExit(f"checkpoint has no completed section: {checkpoint_path}")
    doc_entries = [
        (doc_rel, entry)
        for doc_rel, entry in completed.items()
        if isinstance(entry, dict) and isinstance(entry.get("metrics"), dict)
    ]
    if not doc_entries:
        raise SystemExit(f"checkpoint contains no completed document metrics: {checkpoint_path}")
    docs = [entry for _doc_rel, entry in doc_entries]

    pages_attempted = sum(len(entry.get("page_ids") or []) for entry in docs)
    pages_completed = pages_attempted
    docling_total = sum(float(entry["metrics"].get("docling_runtime_sec") or 0.0) for entry in docs)
    render_total = sum(float(entry["metrics"].get("render_runtime_sec") or 0.0) for entry in docs)
    ocr_total = sum(float(entry["metrics"].get("got_ocr_runtime_sec_total") or 0.0) for entry in docs)
    peak_rss = max(
        (int(entry["metrics"].get("peak_rss_bytes") or 0) for entry in docs),
        default=0,
    )
    cuda_allocated = max(
        (entry["metrics"].get("cuda_peak_allocated_bytes") for entry in docs if entry["metrics"].get("cuda_peak_allocated_bytes")),
        default=None,
    )
    cuda_reserved = max(
        (entry["metrics"].get("cuda_peak_reserved_bytes") for entry in docs if entry["metrics"].get("cuda_peak_reserved_bytes")),
        default=None,
    )
    per_page_ocr: list[float] = []
    png_bytes_total = 0
    ocr_bytes_total = 0
    per_page_bytes: dict[str, dict[str, int]] = {}
    for doc_rel, entry in doc_entries:
        per_page = entry["metrics"].get("per_page_got_ocr_runtime_sec") or {}
        per_page_ocr.extend(float(value) for value in per_page.values() if isinstance(value, (int, float)))
        storage = entry["metrics"].get("storage") or {}
        png_bytes_total += int(storage.get("png_bytes_total") or 0)
        ocr_bytes_total += int(storage.get("ocr_bytes_total") or 0)
        for page_id, sizes in (storage.get("per_page_bytes") or {}).items():
            # Key per document so page numbers never collide across documents.
            per_page_bytes[f"{doc_rel}:p{page_id}"] = sizes

    calibration = checkpoint.get("calibration") if isinstance(checkpoint.get("calibration"), dict) else {}
    cache_before = calibration.get("cache_bytes_before")
    cache_after = calibration.get("cache_bytes_after")
    cache_growth = None
    if isinstance(cache_before, int) and isinstance(cache_after, int):
        cache_growth = max(0, cache_after - cache_before)

    attempt_records = checkpoint.get("attempt_records") if isinstance(checkpoint.get("attempt_records"), list) else []
    timing_complete = calibration.get("timing_complete")
    if timing_complete is None:
        timing_complete = True
        for record in attempt_records:
            if not isinstance(record, dict):
                continue
            if record.get("state") in {"running", "unclean"} and not isinstance(record.get("elapsed_sec"), (int, float)):
                timing_complete = False
                break
    measured_wall = calibration.get("measured_wall_sec")
    if not isinstance(measured_wall, (int, float)):
        measured_wall = calibration.get("attempts_elapsed_sec")
    if not isinstance(measured_wall, (int, float)):
        measured_wall = calibration.get("runtime_sec_total_wall") or 0.0
    exact_total = calibration.get("runtime_sec_total_wall") if timing_complete else None

    gpu: dict[str, Any] = {"model": None, "total_memory_bytes": None, "free_memory_bytes_at_start": None}
    if preflight_path is not None:
        preflight = _load_json(preflight_path, "preflight report")
        devices = (preflight.get("torch") or {}).get("devices") or []
        if devices:
            gpu = {
                "model": devices[0].get("name"),
                "total_memory_bytes": devices[0].get("total_memory_bytes"),
                "free_memory_bytes_at_start": devices[0].get("free_memory_bytes"),
            }

    split_checksum = _sha256(project_root / SPLIT_RELATIVE_PATH)
    qas_checksum = _sha256(project_root / "OHR-Bench/data/qas_v2.json")
    split = str(checkpoint.get("split") or checkpoint.get("dataset") or "")
    manifest_path = project_root / "data/benchmark_assets/ohrbench" / f"{split}.json"
    manifest_checksum = _sha256(manifest_path) if split else None

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration-summary",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "commit_sha": _git_commit_sha(project_root),
        "configuration": _non_secret_environment(),
        "gpu": gpu,
        "vram_peak": {"allocated_bytes": cuda_allocated, "reserved_bytes": cuda_reserved},
        "process": {"peak_rss_bytes": peak_rss or None},
        "runtime": {
            "docling_sec": round(docling_total, 4),
            "render_sec": round(render_total, 4),
            "ocr_sec": round(ocr_total, 4),
            "total_wall_sec": None if exact_total is None else round(float(exact_total), 4),
            "measured_wall_sec": round(float(measured_wall or 0.0), 4),
            "timing_complete": bool(timing_complete),
            "attempts": calibration.get("attempts") if calibration.get("attempts") is not None else len(attempt_records),
            "attempt_records": attempt_records,
        },
        "documents": {
            "attempted": len(docs),
            "completed": len(docs),
            "failed": len(failed) if isinstance(failed, dict) else 0,
        },
        "pages": {"attempted": pages_attempted, "completed": pages_completed},
        "storage": {
            "png_bytes_total": png_bytes_total,
            "ocr_bytes_total": ocr_bytes_total,
            "per_page_bytes": per_page_bytes,
        },
        "cache": {
            "path": calibration.get("cache_path"),
            "bytes_before": cache_before,
            "bytes_after": cache_after,
            "growth_bytes": cache_growth,
        },
        "throughput": {"ocr_page_runtime_sec": _percentiles(per_page_ocr)},
        "failures": {
            "count": len(failed) if isinstance(failed, dict) else 0,
            "retried": len(checkpoint.get("retried_documents") or []),
        },
        "resumed": bool(checkpoint.get("resumed") or calibration.get("resumed")),
        "checksums": {
            "split_sha256": split_checksum,
            "qas_v2_sha256": qas_checksum,
            "manifest_sha256": manifest_checksum,
        },
        "source_checkpoint": str(checkpoint_path),
    }
    return summary


def project_preparation(
    summary: dict[str, Any],
    *,
    headroom_fraction: float,
    walltime_hours_per_shard: float,
    val_documents: int,
    val_pages: int,
    test_documents: int,
    test_pages: int,
) -> dict[str, Any]:
    runtime = summary.get("runtime") or {}
    timing_complete = runtime.get("timing_complete")
    if timing_complete is False:
        raise SystemExit(
            "calibration timing is incomplete because an attempt ended in an unclean shutdown. "
            "Projection refuses an incomplete wall-time total and does not accept scheduler timing. "
            "Apply scheduler timing with scripts/data/prepare_benchmark_assets.py --scheduler-elapsed-sec, "
            "rebuild the calibration summary, then run projection again."
        )
    throughput = (summary.get("throughput") or {}).get("ocr_page_runtime_sec") or {}
    storage = summary.get("storage") or {}
    documents_completed = max(1, int((summary.get("documents") or {}).get("completed") or 1))
    pages_completed = max(1, int((summary.get("pages") or {}).get("completed") or 1))

    docling_per_doc = float(runtime.get("docling_sec") or 0.0) / documents_completed
    render_per_page = float(runtime.get("render_sec") or 0.0) / pages_completed
    ocr_mean = float(throughput.get("mean") or 0.0) if throughput.get("mean") is not None else 0.0
    ocr_p50 = float(throughput.get("p50") or 0.0) if throughput.get("p50") is not None else ocr_mean
    ocr_p95 = float(throughput.get("p95") or 0.0) if throughput.get("p95") is not None else ocr_mean
    png_per_page = int(storage.get("png_bytes_total") or 0) / pages_completed
    ocr_per_page = int(storage.get("ocr_bytes_total") or 0) / pages_completed

    def project(documents: int, pages: int, label: str) -> dict[str, Any]:
        docling_sec = documents * docling_per_doc
        render_sec = pages * render_per_page
        ocr_sec_mean = pages * ocr_mean
        ocr_sec_p95 = pages * ocr_p95
        total_mean = docling_sec + render_sec + ocr_sec_mean
        total_p95 = docling_sec + render_sec + ocr_sec_p95
        total_headroom = total_mean * (1.0 + headroom_fraction)
        storage_raw = pages * (png_per_page + ocr_per_page)
        shards = max(1, math.ceil(total_headroom / (walltime_hours_per_shard * 3600.0)))
        return {
            "label": label,
            "documents": documents,
            "pages": pages,
            "components": {
                "docling_sec": round(docling_sec, 1),
                "render_sec": round(render_sec, 1),
                "ocr_sec": {"mean": round(ocr_sec_mean, 1), "p95": round(ocr_sec_p95, 1)},
            },
            "total_sec": {"mean": round(total_mean, 1), "p95": round(total_p95, 1)},
            "total_with_headroom_sec": round(total_headroom, 1),
            "storage_bytes": {"raw": int(storage_raw), "with_headroom": int(storage_raw * (1.0 + headroom_fraction))},
            "suggested_shards_for_walltime": shards,
        }

    val = project(val_documents, val_pages, "val")
    test = project(test_documents, test_pages, "test")

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "preparation-projection",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_summary": summary.get("source_checkpoint"),
        "rates": {
            "docling_sec_per_document": round(docling_per_doc, 4),
            "render_sec_per_page": round(render_per_page, 4),
            "ocr_sec_per_page": {
                "mean": round(ocr_mean, 4),
                "p50": round(ocr_p50, 4),
                "p95": round(ocr_p95, 4),
            },
            "storage_bytes_per_page": {"png": int(png_per_page), "ocr": int(ocr_per_page)},
            "pages_per_document_observed": round(pages_completed / documents_completed, 2),
        },
        "headroom_fraction": headroom_fraction,
        "projections": {"val": val, "test": test},
        "sharding": {
            "walltime_hours_per_shard": walltime_hours_per_shard,
            "suggested_shards_val": val["suggested_shards_for_walltime"],
            "suggested_shards_test": test["suggested_shards_for_walltime"],
            "walltime_hours_for_4_shards_val": round(val["total_with_headroom_sec"] / 4.0 / 3600.0, 1),
            "walltime_hours_for_4_shards_test": round(test["total_with_headroom_sec"] / 4.0 / 3600.0, 1),
        },
        "assumptions": [
            f"Docling scales per document at {docling_per_doc:.2f} s/doc (single-document measurement).",
            f"Page rendering scales per page at {render_per_page:.4f} s/page.",
            f"GOT-OCR scales per page at mean {ocr_mean:.4f} s/page (p95 {ocr_p95:.4f} s/page).",
            f"Storage scales per page at {int(png_per_page + ocr_per_page)} bytes/page (PNG + OCR).",
            f"Headroom of {headroom_fraction:.0%} applied to total runtime and storage.",
            "Shard assignment is deterministic contiguous chunks of the sorted document list; "
            "no overlap between shards and full coverage when combined.",
        ],
        "warnings": [
            "A single-document calibration may not represent all document sizes; Docling time varies "
            "with document structure and page counts.",
            "Run a representative multi-document pilot (50-100 questions, multiple document types) "
            "before committing to shard counts or wall-time for full validation.",
            "Docling (document-level) and page-level rendering/OCR are projected separately and must "
            "not be collapsed into one per-page rate.",
            "No real CUDA cluster measurements exist yet; these projections are estimates from local "
            "or single-document runs only.",
        ],
        "timing": {
            "complete": timing_complete is not False,
        },
    }
    return result


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "commit_sha": summary.get("commit_sha"),
                "gpu": summary.get("gpu", {}).get("model"),
                "runtime_sec": summary.get("runtime"),
                "peak_rss_bytes": (summary.get("process") or {}).get("peak_rss_bytes"),
                "throughput": summary.get("throughput"),
                "resumed": summary.get("resumed"),
                "checksums": summary.get("checksums"),
            },
            indent=2,
        )
    )


def _print_projection(projection: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "kind": projection["kind"],
                "rates": projection["rates"],
                "projections": projection["projections"],
                "sharding": projection["sharding"],
                "warnings": projection["warnings"],
            },
            indent=2,
        )
    )


def build_command(args: argparse.Namespace) -> int:
    project_root = args.project_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    summary = build_summary(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        preflight_path=args.preflight.expanduser().resolve() if args.preflight else None,
    )
    _print_summary(summary)
    if args.out:
        atomic_write_text(args.out, json.dumps(summary, indent=2) + "\n")
        print(f"[calibration-report] summary written to {args.out}", flush=True)
    return 0


def project_command(args: argparse.Namespace) -> int:
    summary = _load_json(args.summary.expanduser().resolve(), "calibration summary")
    if summary.get("kind") != "calibration-summary":
        raise SystemExit(f"{args.summary} is not a calibration-summary report")
    projection = project_preparation(
        summary,
        headroom_fraction=args.headroom_fraction,
        walltime_hours_per_shard=args.walltime_hours_per_shard,
        val_documents=args.val_documents,
        val_pages=args.val_pages,
        test_documents=args.test_documents,
        test_pages=args.test_pages,
    )
    _print_projection(projection)
    if args.out:
        atomic_write_text(args.out, json.dumps(projection, indent=2) + "\n")
        print(f"[calibration-report] projection written to {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a calibration summary from a checkpoint")
    build_parser.add_argument("--checkpoint", type=Path, required=True)
    build_parser.add_argument("--preflight", type=Path, default=None)
    build_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    build_parser.add_argument("--out", type=Path, default=None)
    build_parser.set_defaults(handler=build_command)

    project_parser = subparsers.add_parser("project", help="Project validation/test preparation from a summary")
    project_parser.add_argument("--summary", type=Path, required=True)
    project_parser.add_argument("--out", type=Path, default=None)
    project_parser.add_argument("--headroom-fraction", type=float, default=DEFAULT_HEADROOM_FRACTION)
    project_parser.add_argument("--walltime-hours-per-shard", type=float, default=DEFAULT_WALLTIME_HOURS_PER_SHARD)
    project_parser.add_argument("--val-documents", type=int, default=DEFAULT_VAL["documents"])
    project_parser.add_argument("--val-pages", type=int, default=DEFAULT_VAL["pages"])
    project_parser.add_argument("--test-documents", type=int, default=DEFAULT_TEST["documents"])
    project_parser.add_argument("--test-pages", type=int, default=DEFAULT_TEST["pages"])
    project_parser.set_defaults(handler=project_command)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
