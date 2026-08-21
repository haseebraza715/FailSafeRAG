from __future__ import annotations

"""
Resumable FAAR benchmark asset preparation.

Pipeline stages per document:
  1. Safe PDF extract/copy from zip or filesystem
  2. Docling structured audit Markdown
  3. Deterministic page PNG rendering
  4. Pinned GOT-OCR text extraction

Full-dataset execution remains disabled. Use `--smoke-doc` for one document.
"""

import argparse
import json
import os
import signal
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.asset_paths import to_relative_project_path
from faar.asset_preparation import (
    RENDER_SCALE,
    STAGE_ORDER,
    execute_document_preparation,
    hash_source_pdf,
    load_locked_docling,
    load_locked_got_ocr,
    page_image_name,
    page_ocr_name,
    plan_document_work,
    resolve_pdf_source,
)
from faar.dataset_paths import DatasetPathError, env_raw, load_project_dotenv, resolve_dataset_paths
from faar.ohr_inventory import (
    diagnose_ohr_inventory_gaps,
    load_resolved_ohr_document_inventory,
    resolve_ohr_inventory_path,
)
from faar.operations import install_graceful_termination_handler
from faar.run_io import atomic_write_text

__all__ = ["page_image_name", "page_ocr_name"]

CHECKPOINT_IDENTITY_SCHEMA = 1
IDENTITY_FIELDS = (
    "schema_version",
    "dataset",
    "smoke_doc",
    "page_ids",
    "source_pdf_sha256",
    "got_ocr",
    "docling",
    "render_scale",
)
CALIBRATION_108_DOC = "manual/User_Manual_1500S_Classic_EN"
CALIBRATION_108_PAGE_COUNT = 108
CALIBRATION_108_PAGE_IDS = list(range(CALIBRATION_108_PAGE_COUNT))


def _rel(path: Path, project_root: Path) -> str:
    try:
        return to_relative_project_path(path, project_root)
    except OSError:
        return path.as_posix()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return {"completed": {}, "updated_at_utc": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Preparation checkpoint is unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("completed"), dict):
        raise SystemExit(f"Preparation checkpoint has an invalid structure: {path}")
    return payload


def save_checkpoint(path: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _huggingface_cache_bytes() -> int | None:
    raw = os.getenv("HF_HOME")
    cache_root = Path(raw).expanduser() if raw and raw.strip() else Path.home() / ".cache/huggingface"
    if not cache_root.is_dir():
        return 0
    total = 0
    try:
        for path in cache_root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        return None
    return total


def _inventory_page_ids(project_root: Path, doc_rel: str, inventory_dir: Path, page_ids_arg: str | None) -> list[int]:
    if page_ids_arg:
        return [int(part.strip()) for part in page_ids_arg.split(",") if part.strip()]
    inventory, _resolutions = load_resolved_ohr_document_inventory(inventory_dir, {doc_rel})
    if doc_rel not in inventory:
        path, resolved, kind = resolve_ohr_inventory_path(inventory_dir, doc_rel)
        raise SystemExit(
            f"No complete gt inventory for {doc_rel!r} (resolved={resolved!r}, diagnosis={kind}, path={path})."
        )
    return inventory[doc_rel]


def validate_calibration_108(*, smoke_doc: str, page_ids: list[int] | None) -> None:
    """Refuse the bounded calibration unless the exact 108-page document is selected."""
    problems: list[str] = []
    if smoke_doc != CALIBRATION_108_DOC:
        problems.append(f"document is {smoke_doc!r}; required {CALIBRATION_108_DOC!r}")
    if not page_ids:
        problems.append("page inventory is missing")
    else:
        try:
            ids = [int(page_id) for page_id in page_ids]
        except (TypeError, ValueError):
            problems.append("page inventory contains non-integer page indices")
            ids = None
        if ids is not None and ids != CALIBRATION_108_PAGE_IDS:
            expected = set(CALIBRATION_108_PAGE_IDS)
            if len(ids) != CALIBRATION_108_PAGE_COUNT:
                problems.append(f"page count is {len(ids)}; required {CALIBRATION_108_PAGE_COUNT}")
            seen: set[int] = set()
            duplicates: list[int] = []
            for page_id in ids:
                if page_id in seen:
                    duplicates.append(page_id)
                else:
                    seen.add(page_id)
            if duplicates:
                problems.append(f"duplicate page indices: {sorted(set(duplicates))}")
            missing = [page_id for page_id in CALIBRATION_108_PAGE_IDS if page_id not in seen]
            extra = sorted(page_id for page_id in seen if page_id not in expected)
            if missing:
                preview = missing[:12]
                suffix = "..." if len(missing) > 12 else ""
                problems.append(f"missing page indices: {preview}{suffix}")
            if extra:
                preview = extra[:12]
                suffix = "..." if len(extra) > 12 else ""
                problems.append(f"unexpected page indices: {preview}{suffix}")
            unique_sorted = sorted(seen)
            if unique_sorted and unique_sorted != list(range(unique_sorted[0], unique_sorted[-1] + 1)):
                problems.append("page indices are not contiguous")
            if ids != unique_sorted:
                problems.append("page indices are not strictly increasing from 0 through 107")
    if problems:
        raise SystemExit(
            "Bounded 108-page calibration lock failed before processing: "
            + "; ".join(problems)
            + ". Refusing to start calibration."
        )


def build_run_identity(
    *,
    dataset: str,
    smoke_doc: str,
    page_ids: list[int],
    source_pdf_sha256: str,
    got_ocr: dict[str, str],
    docling: dict[str, str],
    render_scale: float = RENDER_SCALE,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA,
        "dataset": dataset,
        "smoke_doc": smoke_doc,
        "page_ids": list(page_ids),
        "source_pdf_sha256": source_pdf_sha256,
        "got_ocr": {"repository": got_ocr["repository"], "revision": got_ocr["revision"]},
        "docling": {"repository": docling["repository"], "revision": docling["revision"]},
        "render_scale": render_scale,
    }


def _identity_mismatches(stored: Any, requested: dict[str, Any]) -> list[str]:
    if not isinstance(stored, dict):
        return ["run_identity"]
    return [field for field in IDENTITY_FIELDS if stored.get(field) != requested.get(field)]


def _checkpoint_has_work(checkpoint: dict[str, Any]) -> bool:
    if checkpoint.get("completed"):
        return True
    if checkpoint.get("failed"):
        return True
    if checkpoint.get("attempt_records"):
        return True
    if int(checkpoint.get("attempts") or 0) > 0:
        return True
    return False


def _ensure_checkpoint_identity(
    checkpoint: dict[str, Any],
    requested: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    stored = checkpoint.get("run_identity")
    if stored is None and not _checkpoint_has_work(checkpoint):
        checkpoint["run_identity"] = requested
        return
    if stored is None:
        raise SystemExit(
            f"checkpoint {checkpoint_path} has no run identity and cannot be migrated. "
            "Use a new --checkpoint path for this calibration run."
        )
    mismatches = _identity_mismatches(stored, requested)
    if mismatches:
        details = ", ".join(
            f"{field}: stored={stored.get(field)!r} requested={requested.get(field)!r}"
            for field in mismatches
        )
        raise SystemExit(
            f"checkpoint {checkpoint_path} identity mismatch ({details}). "
            "Refusing to combine calibration work from different runs. Use a new --checkpoint path."
        )


def _attempt_records(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    records = checkpoint.get("attempt_records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    checkpoint["attempt_records"] = []
    return checkpoint["attempt_records"]


def _mark_unclean_running_attempts(checkpoint: dict[str, Any]) -> bool:
    found = False
    now = datetime.now(UTC).isoformat()
    for record in _attempt_records(checkpoint):
        if record.get("state") == "running":
            record["state"] = "unclean"
            record["interrupted_at_utc"] = now
            found = True
    return found


def _timing_from_attempts(records: list[dict[str, Any]]) -> tuple[bool, float]:
    measured = 0.0
    complete = True
    for record in records:
        state = record.get("state")
        elapsed = record.get("elapsed_sec")
        if state == "running":
            complete = False
        if isinstance(elapsed, (int, float)):
            measured += float(elapsed)
        elif state in {"unclean", "interrupted", "completed"}:
            complete = False
        if state == "unclean" and not isinstance(elapsed, (int, float)):
            complete = False
    return complete, measured


def _close_attempt(
    record: dict[str, Any],
    *,
    state: str,
    elapsed_sec: float,
    exit_code: int | None = None,
    signal_number: int | None = None,
) -> None:
    record["state"] = state
    record["elapsed_sec"] = elapsed_sec
    record["elapsed_source"] = "monotonic"
    if state == "interrupted":
        record["interrupted_at_utc"] = datetime.now(UTC).isoformat()
    if exit_code is not None:
        record["exit_code"] = exit_code
    if signal_number is not None:
        record["signal"] = signal_number


def _exit_signal(exc: BaseException) -> tuple[int | None, int | None]:
    if isinstance(exc, SystemExit) and isinstance(exc.code, int):
        code = exc.code
        if code >= 128:
            return code, code - 128
        return code, None
    if isinstance(exc, KeyboardInterrupt):
        return 130, int(signal.SIGINT)
    return None, None


def _apply_scheduler_elapsed(checkpoint: dict[str, Any], elapsed_sec: float) -> None:
    if elapsed_sec <= 0 or not (elapsed_sec == elapsed_sec) or elapsed_sec == float("inf"):
        raise SystemExit("--scheduler-elapsed-sec must be a positive finite number of seconds.")
    for record in reversed(_attempt_records(checkpoint)):
        if record.get("state") == "unclean" and not isinstance(record.get("elapsed_sec"), (int, float)):
            record["elapsed_sec"] = elapsed_sec
            record["elapsed_source"] = "scheduler"
            return
    raise SystemExit(
        "no unclean attempt is missing elapsed time; --scheduler-elapsed-sec was not applied. "
        "Use this flag only to record scheduler wall time for a hard-killed attempt."
    )


def _sync_timing_fields(checkpoint: dict[str, Any], last_attempt_sec: float | None = None) -> None:
    records = _attempt_records(checkpoint)
    complete, measured = _timing_from_attempts(records)
    checkpoint["attempts"] = len(records)
    checkpoint["attempts_elapsed_sec"] = measured
    checkpoint["timing_complete"] = complete
    calibration = checkpoint.setdefault("calibration", {})
    calibration["attempts"] = len(records)
    calibration["attempts_elapsed_sec"] = measured
    calibration["timing_complete"] = complete
    calibration["measured_wall_sec"] = measured
    if complete:
        calibration["runtime_sec_total_wall"] = measured
    else:
        calibration["runtime_sec_total_wall"] = None
    if last_attempt_sec is not None:
        calibration["last_attempt_sec"] = last_attempt_sec
    calibration["resumed"] = bool(checkpoint.get("resumed"))
    calibration["cache_bytes_before"] = checkpoint.get("cache_bytes_before")
    calibration["cache_path"] = checkpoint.get("cache_path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute one-document PDF → Docling → PNG → GOT-OCR asset preparation."
    )
    parser.add_argument("--dataset", required=True, choices=["ohrbench", "mpdocvqa", "arxivqa"])
    parser.add_argument("--project-root", type=Path, default=None, help="Repository root (default: current directory).")
    parser.add_argument("--pdf-root", type=Path, help="Optional filesystem root containing source PDFs.")
    parser.add_argument(
        "--pdf-zip",
        type=Path,
        default=None,
        help="PDF archive (default: data/ohr_bench_raw/pdfs.zip).",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for audit/images/ocr.")
    parser.add_argument(
        "--smoke-doc",
        type=str,
        help="Document id for one-document planning/execution (e.g. academic/DUDE_...).",
    )
    parser.add_argument(
        "--page-ids",
        type=str,
        default=None,
        help="Optional explicit page ids. Default: complete gt inventory pages for the document.",
    )
    parser.add_argument(
        "--document-inventory",
        type=Path,
        default=None,
        help="OHR gt inventory directory (default: OHR-Bench/data/retrieval_base/gt).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the one-document pipeline (local Docling/GOT-OCR). Full-dataset mode remains blocked.",
    )
    parser.add_argument(
        "--require-calibration-108",
        action="store_true",
        help=(
            "Refuse to run unless --smoke-doc is manual/User_Manual_1500S_Classic_EN "
            "and the selected pages are exactly 0 through 107. Required for the "
            "bounded supervisor calibration; other one-document prep stays unlocked."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, help="Resumable checkpoint JSON path.")
    parser.add_argument(
        "--scheduler-elapsed-sec",
        type=float,
        default=None,
        help=(
            "Scheduler-reported wall seconds for an unclean (SIGKILL) attempt. "
            "Does not invent elapsed time from the start timestamp."
        ),
    )
    parser.add_argument(
        "--diagnose-inventory",
        action="store_true",
        help="Print OHR inventory gap diagnosis for qas_v2 and exit.",
    )
    args = parser.parse_args(argv)
    install_graceful_termination_handler()

    project_root = (args.project_root or Path.cwd()).expanduser().resolve()
    load_project_dotenv(project_root)
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

    if args.diagnose_inventory:
        diagnose_zip = pdf_zip or (project_root / "data/ohr_bench_raw/pdfs.zip")
        pdf_names: set[str] = set()
        if diagnose_zip.is_file():
            with zipfile.ZipFile(diagnose_zip) as archive:
                pdf_names = {name for name in archive.namelist() if name.endswith(".pdf")}
        report = diagnose_ohr_inventory_gaps(
            qas_path=project_root / "OHR-Bench/data/qas_v2.json",
            inventory_dir=inventory_dir,
            pdf_names=pdf_names,
        )
        print(json.dumps(report, indent=2))
        return

    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (out_root / "prepare_checkpoint.json")
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not args.smoke_doc:
        raise SystemExit(
            "Full-dataset preparation is intentionally not enabled. "
            "Pass --smoke-doc <doc_rel> to plan or execute one document."
        )
    if args.dataset != "ohrbench":
        raise SystemExit("One-document execution is currently implemented for --dataset ohrbench only.")

    doc_rel = args.smoke_doc.strip().removesuffix(".pdf")
    page_ids = _inventory_page_ids(project_root, doc_rel, inventory_dir, args.page_ids)
    if args.require_calibration_108:
        validate_calibration_108(smoke_doc=doc_rel, page_ids=page_ids)
    elif not page_ids:
        raise SystemExit("--page-ids / inventory must contain at least one page id.")

    kind, source = resolve_pdf_source(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_root=pdf_root,
        pdf_zip=pdf_zip,
        inventory_dir=inventory_dir,
    )
    if kind == "missing":
        raise SystemExit(f"PDF for {doc_rel!r} was not found in --pdf-root or pdfs.zip.")
    try:
        source_sha256 = hash_source_pdf(
            project_root=project_root,
            doc_rel=doc_rel,
            pdf_root=pdf_root,
            pdf_zip=pdf_zip,
            inventory_dir=inventory_dir,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    pdf_path = out_root / "pdfs" / f"{doc_rel}.pdf"
    locked = load_locked_got_ocr(project_root)
    locked_docling = load_locked_docling(project_root)
    requested_identity = build_run_identity(
        dataset=args.dataset,
        smoke_doc=doc_rel,
        page_ids=page_ids,
        source_pdf_sha256=source_sha256,
        got_ocr=locked,
        docling=locked_docling,
    )
    work = plan_document_work(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_path=pdf_path,
        page_ids=page_ids,
        out_root=out_root,
        got_ocr=locked,
    )
    work["pdf_source"] = {
        "kind": kind,
        "source": source.as_posix() if isinstance(source, Path) else None,
    }
    checkpoint = load_checkpoint(checkpoint_path)
    _ensure_checkpoint_identity(checkpoint, requested_identity, checkpoint_path)
    unclean = _mark_unclean_running_attempts(checkpoint)
    if args.scheduler_elapsed_sec is not None:
        _apply_scheduler_elapsed(checkpoint, args.scheduler_elapsed_sec)
    elif unclean:
        checkpoint["timing_complete"] = False
    checkpoint.setdefault("completed", {})
    checkpoint["run_identity"] = requested_identity
    checkpoint["dataset"] = args.dataset
    checkpoint["smoke_doc"] = doc_rel
    checkpoint["plan"] = work
    checkpoint["resumed"] = _checkpoint_has_work(checkpoint)
    checkpoint["cache_path"] = str(
        (Path(os.getenv("HF_HOME")) if os.getenv("HF_HOME") else Path.home() / ".cache/huggingface").expanduser()
    )
    _sync_timing_fields(checkpoint)

    plan_path = out_root / f"prepare_plan_{doc_rel.replace('/', '__')}.json"
    atomic_write_text(plan_path, json.dumps(work, indent=2) + "\n")

    if not args.execute:
        save_checkpoint(checkpoint_path, checkpoint)
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "dataset": args.dataset,
                    "smoke_doc": doc_rel,
                    "page_ids": page_ids,
                    "page_count": len(page_ids),
                    "pdf_source": work["pdf_source"],
                    "got_ocr": locked,
                    "docling": locked_docling,
                    "expected_outputs": {
                        "pdf": work["pdf"],
                        "docling_audit": work["stages"]["docling_audit"]["output"],
                        "images": work["stages"]["render_pages"]["outputs"],
                        "ocr": work["stages"]["got_ocr"]["outputs"],
                    },
                    "plan": _rel(plan_path, project_root),
                    "checkpoint": _rel(checkpoint_path, project_root),
                    "stages": STAGE_ORDER,
                    "timing_complete": checkpoint.get("timing_complete"),
                },
                indent=2,
            )
        )
        return

    runtime_started = time.perf_counter()
    if checkpoint.get("cache_bytes_before") is None:
        checkpoint["cache_bytes_before"] = _huggingface_cache_bytes()
    records = _attempt_records(checkpoint)
    attempt = {
        "id": len(records) + 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "state": "running",
        "elapsed_sec": None,
        "elapsed_source": None,
        "interrupted_at_utc": None,
        "exit_code": None,
        "signal": None,
    }
    records.append(attempt)
    checkpoint["attempt_records"] = records
    checkpoint["attempt_started_at_utc"] = attempt["started_at_utc"]
    _sync_timing_fields(checkpoint)
    save_checkpoint(checkpoint_path, checkpoint)
    try:
        result = execute_document_preparation(
            project_root=project_root,
            doc_rel=doc_rel,
            page_ids=page_ids,
            out_root=out_root,
            pdf_root=pdf_root,
            pdf_zip=pdf_zip,
            inventory_dir=inventory_dir,
        )
    except BaseException as exc:
        elapsed = time.perf_counter() - runtime_started
        exit_code, signal_number = _exit_signal(exc)
        _close_attempt(
            attempt,
            state="interrupted",
            elapsed_sec=elapsed,
            exit_code=exit_code,
            signal_number=signal_number,
        )
        _sync_timing_fields(checkpoint, last_attempt_sec=elapsed)
        save_checkpoint(checkpoint_path, checkpoint)
        raise
    elapsed = time.perf_counter() - runtime_started
    _close_attempt(attempt, state="completed", elapsed_sec=elapsed, exit_code=0)
    _sync_timing_fields(checkpoint, last_attempt_sec=elapsed)
    checkpoint["calibration"]["cache_bytes_after"] = _huggingface_cache_bytes()
    checkpoint["completed"][doc_rel] = {
        "page_ids": page_ids,
        "pdf_sha256": result.pdf_sha256,
        "outputs": result.outputs,
        "metrics": result.metrics,
    }
    checkpoint["plan"] = plan_document_work(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_path=result.pdf_path,
        page_ids=page_ids,
        out_root=out_root,
        got_ocr=locked,
    )
    save_checkpoint(checkpoint_path, checkpoint)
    result_path = out_root / f"prepare_result_{doc_rel.replace('/', '__')}.json"
    atomic_write_text(
        result_path,
        json.dumps(
            {
                "doc_name": result.doc_name,
                "page_ids": result.page_ids,
                "pdf_sha256": result.pdf_sha256,
                "got_ocr_repository": result.got_ocr_repository,
                "got_ocr_revision": result.got_ocr_revision,
                "device": result.device,
                "outputs": result.outputs,
                "metrics": result.metrics,
            },
            indent=2,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "mode": "execute",
                "dataset": args.dataset,
                "smoke_doc": doc_rel,
                "page_count": len(page_ids),
                "device": result.device,
                "got_ocr_revision": result.got_ocr_revision,
                "result": _rel(result_path, project_root),
                "outputs": result.outputs,
                "metrics": {
                    "pdf_sha256": result.pdf_sha256,
                    "render_runtime_sec": result.metrics.get("render_runtime_sec"),
                    "docling_runtime_sec": result.metrics.get("docling_runtime_sec"),
                    "got_ocr_runtime_sec_total": result.metrics.get("got_ocr_runtime_sec_total"),
                    "per_page_got_ocr_runtime_sec": result.metrics.get("per_page_got_ocr_runtime_sec"),
                    "peak_rss_bytes": result.metrics.get("peak_rss_bytes"),
                    "storage": result.metrics.get("storage"),
                },
                "timing_complete": checkpoint.get("timing_complete"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
