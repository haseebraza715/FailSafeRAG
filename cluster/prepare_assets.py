#!/usr/bin/env python3
"""Resumable, sharded, multi-document OHR-Bench asset preparation.

Pipeline stages per document (identical to scripts/data/prepare_benchmark_assets.py):
  1. Safe PDF extract/copy from zip or filesystem
  2. Docling structured audit Markdown
  3. Deterministic page PNG rendering
  4. Pinned GOT-OCR text extraction

Design invariants:

- Documents come only from the immutable OHR split + qas_v2.json pair and are
  ordered stably (sorted unique doc_name), so shard boundaries are identical
  on every machine and every retry.
- Shards are contiguous, non-overlapping chunks of the sorted document list
  using the same validated select_shard semantics as cluster/run_baselines.py.
  Combining all shards covers the whole split exactly once.
- Documents are processed strictly one at a time; only the current document's
  page inventory and outputs are held in memory. The full dataset and the full
  rendered image set are never loaded at once.
- --max-documents / --max-pages bound the assignment deterministically so
  calibration subsets are reproducible across retries.
- The immutable split and QA source are verified against the committed
  checksum lock before any work starts.
- A document is recorded as completed only after every stage output has been
  validated. Checkpoints and shard manifests are published atomically, and a
  structurally invalid checkpoint fails closed.
- Recoverable per-document errors are recorded and the run continues, then the
  process exits non-zero with the exact failed document list. Fatal RAM/CUDA
  exhaustion and graceful termination save the checkpoint and abort.
- Completed documents are never silently overwritten: they are re-verified
  against their recorded outputs and provenance before being skipped.

The validation split is the default; the test split requires --split test
explicitly. Train is intentionally not prepared by this runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.asset_preparation import (
    execute_document_preparation,
    export_docling_markdown,
    hash_source_pdf,
    load_locked_docling,
    load_locked_got_ocr,
    sha256_file,
)
from faar.dataset_paths import (
    SPLIT_RELATIVE_PATH,
    DatasetPathError,
    env_raw,
    load_project_dotenv,
    resolve_dataset_paths,
)
from faar.ohr_inventory import load_resolved_ohr_document_inventory
from faar.operations import check_termination, install_graceful_termination_handler
from faar.resource_limits import enforce_memory_budget, is_fatal_resource_error
from faar.run_io import atomic_write_text, select_shard, shard_label

SCHEMA_VERSION = 1
SPLIT_CHOICES = ("val", "test")


def verify_split_checksums(project_root: Path) -> None:
    """Fail closed unless the immutable split and QA source match the committed lock."""
    lock_path = project_root / "config/split_checksums.json"
    if not lock_path.is_file():
        raise SystemExit(
            f"Missing committed split checksum lock: {lock_path}. "
            "Cannot verify the immutable benchmark split; refusing to prepare assets."
        )
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Split checksum lock is unreadable JSON: {lock_path}: {exc}") from exc
    if not isinstance(lock, dict):
        raise SystemExit(f"Split checksum lock has an invalid structure: {lock_path}")
    for relative, key in (
        (SPLIT_RELATIVE_PATH.as_posix(), "split_sha256"),
        ("OHR-Bench/data/qas_v2.json", "qas_v2_sha256"),
    ):
        source = project_root / relative
        expected = lock.get(key)
        if not source.is_file():
            raise SystemExit(f"Immutable split source is missing: {source}")
        if not isinstance(expected, str) or not expected:
            raise SystemExit(f"Split checksum lock is missing {key}: {lock_path}")
        actual = sha256_file(source)
        if actual != expected:
            raise SystemExit(
                f"Immutable split integrity check failed for {relative}: "
                f"actual sha256 {actual} does not match the committed lock {expected}. "
                "Do not modify the benchmark split."
            )


def load_split_documents(project_root: Path, split: str) -> tuple[list[str], int]:
    """Return (sorted unique doc_names, example count) for the locked split."""
    split_payload = json.loads((project_root / SPLIT_RELATIVE_PATH).read_text(encoding="utf-8"))
    example_ids = split_payload.get("splits", {}).get(split)
    if not isinstance(example_ids, list) or not example_ids:
        raise SystemExit(f"{SPLIT_RELATIVE_PATH} has no non-empty {split!r} selection.")
    rows = json.loads((project_root / "OHR-Bench/data/qas_v2.json").read_text(encoding="utf-8"))
    by_id = {str(row.get("ID")): row for row in rows}
    missing = [example_id for example_id in example_ids if example_id not in by_id]
    if missing:
        raise SystemExit(f"qas_v2.json is missing {len(missing)} split examples (first: {missing[:5]}).")
    doc_names = sorted({str(by_id[example_id]["doc_name"]) for example_id in example_ids})
    if not doc_names:
        raise SystemExit(f"split {split!r} produced no documents.")
    return doc_names, len(example_ids)


def bound_documents(
    doc_names: list[str],
    page_ids_by_doc: dict[str, list[int]],
    *,
    max_documents: int | None = None,
    max_pages: int | None = None,
) -> tuple[list[str], bool]:
    """Deterministically bound an already-ordered shard by count/page budgets."""
    selected: list[str] = []
    page_total = 0
    bounds_hit = False
    for doc in doc_names:
        if max_documents is not None and len(selected) >= max_documents:
            bounds_hit = True
            break
        doc_pages = len(page_ids_by_doc.get(doc, []))
        if max_pages is not None and page_total + doc_pages > max_pages:
            bounds_hit = True
            break
        selected.append(doc)
        page_total += doc_pages
    return selected, bounds_hit


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"completed": {}, "failed": {}, "updated_at_utc": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Preparation checkpoint is unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Preparation checkpoint has an invalid structure: {path}")
    for key in ("completed", "failed"):
        if key in payload and not isinstance(payload[key], dict):
            raise SystemExit(f"Preparation checkpoint has an invalid {key!r} section: {path}")
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _safe_output_path(value: Any, project_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("output path must be a non-empty string")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"output path must not contain '..': {value!r}")
    resolved = path if path.is_absolute() else project_root / path
    return resolved.expanduser().resolve()


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _recorded_hash_matches(path: Path, recorded: Any) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    try:
        return sha256_file(path) == recorded
    except OSError:
        return False


def _completed_doc_valid(
    project_root: Path,
    doc_rel: str,
    entry: Any,
    *,
    page_ids: list[int],
    source_sha256: str,
    got_ocr: dict[str, str],
    docling: dict[str, str],
) -> bool:
    """Skip a completed document only when provenance identity and hashes match."""
    if not isinstance(entry, dict):
        return False
    if list(entry.get("page_ids") or []) != list(page_ids):
        return False
    if entry.get("pdf_sha256") != source_sha256:
        return False
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return False
    try:
        pdf_path = _safe_output_path(outputs.get("pdf"), project_root)
        audit_path = _safe_output_path(outputs.get("docling_audit"), project_root)
        provenance_path = _safe_output_path(outputs.get("provenance"), project_root)
        image_values = outputs.get("images")
        ocr_values = outputs.get("ocr")
        if not isinstance(image_values, list) or not image_values:
            return False
        if not isinstance(ocr_values, list) or not ocr_values:
            return False
        image_paths = [_safe_output_path(value, project_root) for value in image_values]
        ocr_paths = [_safe_output_path(value, project_root) for value in ocr_values]
    except (ValueError, OSError):
        return False
    required = [pdf_path, audit_path, provenance_path, *image_paths, *ocr_paths]
    try:
        if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
            return False
    except OSError:
        return False
    provenance = _read_json_file(provenance_path)
    if provenance is None:
        return False
    if provenance.get("doc_name") != doc_rel:
        return False
    if list(provenance.get("page_ids") or []) != list(page_ids):
        return False
    if provenance.get("pdf_sha256") != source_sha256:
        return False
    if provenance.get("got_ocr_repository") != got_ocr.get("repository"):
        return False
    if provenance.get("got_ocr_revision") != got_ocr.get("revision"):
        return False
    stored_docling = provenance.get("docling_models") if isinstance(provenance.get("docling_models"), dict) else {}
    if stored_docling.get("repository") != docling.get("repository"):
        return False
    if stored_docling.get("revision") != docling.get("revision"):
        return False
    if not _recorded_hash_matches(pdf_path, provenance.get("pdf_sha256")):
        return False
    audit_prov = provenance.get("docling_audit") if isinstance(provenance.get("docling_audit"), dict) else {}
    if audit_prov.get("repository") != docling.get("repository"):
        return False
    if audit_prov.get("revision") != docling.get("revision"):
        return False
    if audit_prov.get("pdf_sha256") != source_sha256:
        return False
    if not _recorded_hash_matches(audit_path, audit_prov.get("audit_sha256")):
        return False
    pages = provenance.get("pages") if isinstance(provenance.get("pages"), dict) else {}
    if len(image_paths) != len(page_ids) or len(ocr_paths) != len(page_ids):
        return False
    for page_id, image_path, ocr_path in zip(page_ids, image_paths, ocr_paths):
        page_prov = pages.get(str(page_id)) if isinstance(pages.get(str(page_id)), dict) else {}
        if page_prov.get("pdf_sha256") != source_sha256:
            return False
        if page_prov.get("got_ocr_repository") != got_ocr.get("repository"):
            return False
        if page_prov.get("got_ocr_revision") != got_ocr.get("revision"):
            return False
        if not _recorded_hash_matches(image_path, page_prov.get("png_sha256")):
            return False
        if not _recorded_hash_matches(ocr_path, page_prov.get("ocr_sha256")):
            return False
    return True


def _source_pdf_sha256(
    *,
    project_root: Path,
    doc_rel: str,
    pdf_root: Path | None,
    pdf_zip: Path | None,
    inventory_dir: Path | None,
) -> str:
    """Hash the current source PDF. Missing or unreadable sources raise."""
    resolved_inventory = inventory_dir or (project_root / "OHR-Bench/data/retrieval_base/gt")
    return hash_source_pdf(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_root=pdf_root,
        pdf_zip=pdf_zip,
        inventory_dir=resolved_inventory,
    )


def run_shard(
    *,
    project_root: Path,
    split: str,
    doc_names: list[str],
    page_ids_by_doc: dict[str, list[int]],
    out_root: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    shard_index: int = 0,
    num_shards: int = 1,
    pdf_root: Path | None = None,
    pdf_zip: Path | None = None,
    inventory_dir: Path | None = None,
    max_documents: int | None = None,
    max_pages: int | None = None,
    resume: bool = False,
    extract_got_ocr_fn: Callable[..., str] | None = None,
    export_docling_fn: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    shard_docs, bounds_hit = bound_documents(
        doc_names, page_ids_by_doc, max_documents=max_documents, max_pages=max_pages
    )
    identity = {
        "split": split,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "max_documents": max_documents,
        "max_pages": max_pages,
    }
    locked_got = load_locked_got_ocr(project_root)
    locked_docling = load_locked_docling(project_root)
    checkpoint = load_checkpoint(checkpoint_path)
    for key, expected in identity.items():
        if key in checkpoint and checkpoint[key] != expected:
            raise SystemExit(
                f"checkpoint {checkpoint_path} was written for {key}={checkpoint[key]!r}; "
                f"this run is {key}={expected!r}. Refusing to resume mismatched work."
            )
    had_prior = bool(checkpoint.get("completed") or checkpoint.get("failed"))
    if had_prior and not resume:
        raise SystemExit(
            f"checkpoint {checkpoint_path} already contains prepared work; "
            "pass --resume to continue or move the checkpoint aside."
        )
    checkpoint.update(identity)
    checkpoint.setdefault("completed", {})
    failed = checkpoint.setdefault("failed", {})
    retried = sorted(failed)
    for doc in retried:
        del failed[doc]
    checkpoint["resumed"] = bool(had_prior)
    checkpoint.setdefault("started_at_utc", datetime.now(UTC).isoformat())
    save_checkpoint(checkpoint_path, checkpoint)

    failed_docs: list[str] = []
    completed_pages = 0
    skipped_docs: list[str] = []
    enforce_memory_budget("multi-document preparation start")
    for doc in shard_docs:
        try:
            check_termination()
            enforce_memory_budget(f"asset preparation before document {doc}")
            entry = checkpoint["completed"].get(doc)
            if entry is not None:
                try:
                    source_sha = _source_pdf_sha256(
                        project_root=project_root,
                        doc_rel=doc,
                        pdf_root=pdf_root,
                        pdf_zip=pdf_zip,
                        inventory_dir=inventory_dir,
                    )
                except FileNotFoundError as exc:
                    raise SystemExit(
                        f"Cannot verify source PDF for completed document {doc}: {exc}. "
                        "Refusing to skip completed work or publish a shard manifest."
                    ) from exc
                if _completed_doc_valid(
                    project_root,
                    doc,
                    entry,
                    page_ids=page_ids_by_doc[doc],
                    source_sha256=source_sha,
                    got_ocr=locked_got,
                    docling=locked_docling,
                ):
                    completed_pages += len(page_ids_by_doc[doc])
                    skipped_docs.append(doc)
                    continue
                del checkpoint["completed"][doc]
            result = execute_document_preparation(
                project_root=project_root,
                doc_rel=doc,
                page_ids=page_ids_by_doc[doc],
                out_root=out_root,
                pdf_root=pdf_root,
                pdf_zip=pdf_zip,
                inventory_dir=inventory_dir,
                extract_got_ocr_fn=extract_got_ocr_fn,
                export_docling_fn=export_docling_fn or export_docling_markdown,
            )
        except (SystemExit, KeyboardInterrupt) as exc:
            code = exc.code if isinstance(exc, SystemExit) else 130
            if not (isinstance(code, int) and code >= 128):
                raise
            save_checkpoint(checkpoint_path, checkpoint)
            raise
        except Exception as exc:
            save_checkpoint(checkpoint_path, checkpoint)
            if is_fatal_resource_error(exc):
                raise
            checkpoint["failed"][doc] = {
                "error": str(exc),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            }
            save_checkpoint(checkpoint_path, checkpoint)
            failed_docs.append(doc)
            continue
        enforce_memory_budget(f"asset preparation after document {doc}")
        checkpoint["completed"][doc] = {
            "page_ids": page_ids_by_doc[doc],
            "pdf_sha256": result.pdf_sha256,
            "outputs": result.outputs,
            "metrics": result.metrics,
        }
        completed_pages += len(page_ids_by_doc[doc])
        save_checkpoint(checkpoint_path, checkpoint)
    enforce_memory_budget("multi-document preparation end")

    finished_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ohr-asset-shard-manifest",
        "dataset": "ohrbench",
        "split": split,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "created_at_utc": checkpoint.get("started_at_utc"),
        "finished_at_utc": finished_at,
        "resumed": bool(checkpoint.get("resumed")),
        "bounds": {
            "max_documents": max_documents,
            "max_pages": max_pages,
            "bounds_hit": bounds_hit,
        },
        "documents_assigned": shard_docs,
        "documents_completed": sorted(checkpoint["completed"]),
        "documents_failed": {doc: checkpoint["failed"][doc] for doc in sorted(checkpoint["failed"])},
        "retried_documents": retried,
        "pages_assigned": sum(len(page_ids_by_doc[doc]) for doc in shard_docs),
        "pages_completed": completed_pages,
        "runtime_sec": round(time.perf_counter() - started, 3),
        "checkpoint": checkpoint_path.as_posix(),
        "out_root": out_root.as_posix(),
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return {
        "manifest": manifest,
        "failed": failed_docs,
        "skipped": skipped_docs,
        "pages_completed": completed_pages,
    }


def merge_shard_manifests(
    manifest_paths: list[Path], *, expected_documents: list[str] | None = None
) -> dict[str, Any]:
    """Validate a complete, non-overlapping shard set and merge the summaries."""
    if not manifest_paths:
        raise SystemExit("at least one shard manifest is required")
    manifests: list[dict[str, Any]] = []
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"shard manifest is unreadable JSON: {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("kind") != "ohr-asset-shard-manifest":
            raise SystemExit(f"shard manifest is invalid: {path}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(f"shard manifest has an unsupported schema version: {path}")
        manifests.append(payload)

    reference = manifests[0]
    for path, payload in zip(manifest_paths, manifests):
        for key in ("split", "num_shards", "dataset"):
            if payload.get(key) != reference.get(key):
                raise SystemExit(
                    f"shard manifest {path} has {key}={payload.get(key)!r}; "
                    f"expected {reference.get(key)!r}."
                )

    declared: dict[int, Path] = {}
    assigned: dict[str, Path] = {}
    duplicates: list[str] = []
    for path, payload in zip(manifest_paths, manifests):
        index = payload.get("shard_index")
        num_shards = payload.get("num_shards")
        if isinstance(index, bool) or not isinstance(index, int):
            raise SystemExit(f"shard manifest {path} has an invalid shard_index: {index!r}.")
        if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards < 1:
            raise SystemExit(f"shard manifest {path} has an invalid num_shards: {num_shards!r}.")
        if not 0 <= index < num_shards:
            raise SystemExit(
                f"shard manifest {path} declares shard_index={index} for num_shards={num_shards}; "
                "expected an index in [0, num_shards)."
            )
        if index in declared:
            duplicates.append(f"shard index {index} declared by {declared[index]} and {path}")
        declared[index] = path
        documents = payload.get("documents_assigned")
        if not isinstance(documents, list):
            raise SystemExit(f"shard manifest {path} has no documents_assigned list.")
        for doc in documents:
            if not isinstance(doc, str) or not doc:
                raise SystemExit(f"shard manifest {path} contains an invalid document entry.")
            if doc in assigned:
                duplicates.append(f"document {doc!r} assigned by {assigned[doc]} and {path}")
            assigned[doc] = path
    missing_indices = [index for index in range(reference["num_shards"]) if index not in declared]
    if missing_indices or duplicates:
        details = []
        if missing_indices:
            details.append(f"missing shard indices {missing_indices}")
        details.extend(duplicates)
        raise SystemExit("incomplete shard manifest set: " + "; ".join(details))

    if expected_documents is not None:
        expected_set = set(expected_documents)
        assigned_set = set(assigned)
        missing_docs = sorted(expected_set - assigned_set)
        extra_docs = sorted(assigned_set - expected_set)
        if missing_docs or extra_docs:
            raise SystemExit(
                f"shard manifests do not match the expected document set "
                f"(missing {len(missing_docs)}, extra {len(extra_docs)})."
            )

    incomplete = [
        doc
        for path, payload in zip(manifest_paths, manifests)
        for doc in payload.get("documents_assigned", [])
        if doc not in (payload.get("documents_completed") or [])
    ]
    if incomplete:
        raise SystemExit(
            f"shard manifests declare uncompleted documents; refusing to merge: {sorted(set(incomplete))}"
        )

    return {
        "split": reference["split"],
        "dataset": reference["dataset"],
        "num_shards": reference["num_shards"],
        "documents": sorted(assigned),
        "documents_failed": {
            doc: payload["documents_failed"][doc]
            for payload in manifests
            for doc in (payload.get("documents_failed") or {})
        },
        "pages_completed": sum(payload.get("pages_completed", 0) for payload in manifests),
        "runtime_sec": round(sum(float(payload.get("runtime_sec") or 0.0) for payload in manifests), 3),
        "bounds_hit": any(payload.get("bounds", {}).get("bounds_hit") for payload in manifests),
    }


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{name} must be a non-boolean integer; received {value!r}.")
    if value < 1:
        raise SystemExit(f"{name} must be a positive integer; received {value}.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{name} must be a non-boolean integer; received {value!r}.")
    if value < 0:
        raise SystemExit(f"{name} must be a non-negative integer; received {value}.")


def _validate_resource_config() -> None:
    budget = os.getenv("FAAR_GPU_BUDGET_GB")
    fraction = os.getenv("FAAR_MAX_GPU_MEMORY_FRACTION")
    if budget and budget.strip() and fraction and fraction.strip():
        raise SystemExit(
            "Set only one of FAAR_GPU_BUDGET_GB or FAAR_MAX_GPU_MEMORY_FRACTION; "
            "two GPU limits are ambiguous."
        )
    for name in ("FAAR_GPU_BUDGET_GB", "FAAR_MIN_GPU_FREE_GB", "FAAR_MAX_RSS_GB"):
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise SystemExit(f"{name} must be a positive finite number of GiB; received {raw!r}.") from exc
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{name} must be a positive finite number of GiB; received {value}.")


def _default_thread_limits() -> None:
    if os.getenv("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // 2))
    if os.getenv("MKL_NUM_THREADS") is None:
        os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=["ohrbench"], default="ohrbench")
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="val",
        help="Split to prepare (default val). Test requires --split test explicitly.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pdf-root", type=Path, default=None)
    parser.add_argument("--pdf-zip", type=Path, default=None)
    parser.add_argument("--document-inventory", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args(argv)

    _validate_non_negative_int("--shard-index", args.shard_index)
    _validate_positive_int("--num-shards", args.num_shards)
    _validate_positive_int("--max-documents", args.max_documents)
    _validate_positive_int("--max-pages", args.max_pages)
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise SystemExit(f"project root is not a directory: {project_root}")
    load_project_dotenv(project_root)
    _validate_resource_config()
    _default_thread_limits()
    install_graceful_termination_handler()
    verify_split_checksums(project_root)
    doc_names, example_count = load_split_documents(project_root, args.split)

    try:
        dataset_paths = resolve_dataset_paths(
            project_root=project_root,
            inventory=args.document_inventory or env_raw("FAAR_DOCUMENT_INVENTORY"),
            pdf_root=args.pdf_root or env_raw("FAAR_PDF_ROOT"),
            pdf_zip=args.pdf_zip or env_raw("FAAR_PDF_ZIP"),
        )
    except DatasetPathError as exc:
        raise SystemExit(str(exc)) from exc
    pdf_root = dataset_paths.pdf_root
    pdf_zip = dataset_paths.pdf_zip
    inventory_dir = dataset_paths.inventory_dir
    shard_docs = select_shard(doc_names, args.shard_index, args.num_shards)
    page_ids_by_doc, _resolutions = load_resolved_ohr_document_inventory(inventory_dir, set(shard_docs))
    missing_inventory = sorted(set(shard_docs) - set(page_ids_by_doc))
    if missing_inventory:
        raise SystemExit(
            f"No complete inventory for {len(missing_inventory)} shard documents "
            f"(first: {missing_inventory[:10]})."
        )
    selected, bounds_hit = bound_documents(
        shard_docs, page_ids_by_doc, max_documents=args.max_documents, max_pages=args.max_pages
    )

    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    label = shard_label(args.shard_index, args.num_shards)
    checkpoint_path = (args.checkpoint or (out_root / f"checkpoint_{label}.json")).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    manifest_path = out_root / f"shard_manifest_{label}.json"

    if args.dry_run:
        locked = load_locked_got_ocr(project_root)
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "dataset": args.dataset,
                    "split": args.split,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "shard_label": label,
                    "documents_assigned": selected,
                    "documents_skipped_by_bounds": sorted(set(shard_docs) - set(selected)),
                    "bounds": {"max_documents": args.max_documents, "max_pages": args.max_pages, "bounds_hit": bounds_hit},
                    "page_count": sum(len(page_ids_by_doc[doc]) for doc in selected),
                    "example_count": example_count,
                    "got_ocr": locked,
                    "checkpoint": checkpoint_path.as_posix(),
                    "manifest": manifest_path.as_posix(),
                    "stages": ["extract_pdf", "docling_audit", "render_pages", "got_ocr"],
                },
                indent=2,
            )
        )
        return 0

    if args.cache_root is not None:
        cache_root = args.cache_root if args.cache_root.is_absolute() else project_root / args.cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root))
    out_root.mkdir(parents=True, exist_ok=True)

    summary = run_shard(
        project_root=project_root,
        split=args.split,
        doc_names=selected,
        page_ids_by_doc=page_ids_by_doc,
        out_root=out_root,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        pdf_root=pdf_root,
        pdf_zip=pdf_zip,
        inventory_dir=inventory_dir,
        max_documents=args.max_documents,
        max_pages=args.max_pages,
        resume=args.resume,
    )
    manifest = summary["manifest"]
    print(
        json.dumps(
            {
                "mode": "execute",
                "dataset": args.dataset,
                "split": args.split,
                "shard_label": label,
                "resumed": manifest["resumed"],
                "documents_assigned": len(manifest["documents_assigned"]),
                "documents_completed": len(manifest["documents_completed"]),
                "documents_failed": manifest["documents_failed"],
                "retried_documents": manifest["retried_documents"],
                "pages_completed": manifest["pages_completed"],
                "runtime_sec": manifest["runtime_sec"],
                "bounds_hit": manifest["bounds"]["bounds_hit"],
                "checkpoint": manifest["checkpoint"],
                "manifest": manifest_path.as_posix(),
            },
            indent=2,
        )
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
