from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path
from typing import Any

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PACKAGED_LOCK = Path(__file__).resolve().parent / "model_revisions.json"
_ARTIFACTS: dict[tuple[str, str], Path] = {}
_CONVERTERS: dict[tuple[str, str], Any] = {}


class PdfPreprocessingError(RuntimeError):
    """Raised when Docling cannot produce a usable document representation."""


def validate_docling_revision(revision: str, *, source: Path | str) -> str:
    text = str(revision).strip()
    if not _IMMUTABLE_REVISION.fullmatch(text):
        raise PdfPreprocessingError(
            f"Docling revision must be exactly 40 lowercase hexadecimal characters ({source})."
        )
    return text


def _read_docling_lock(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PdfPreprocessingError(f"Docling model lock is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PdfPreprocessingError(f"Docling model lock is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PdfPreprocessingError(f"Docling model lock has an invalid structure: {path}")
    entry = (payload.get("models") or {}).get("docling") or {}
    if not isinstance(entry, dict):
        raise PdfPreprocessingError(f"config/model_revisions.json docling lock is incomplete: {path}")
    repository = str(entry.get("repository", "")).strip()
    revision = str(entry.get("revision", "")).strip()
    if not repository or not revision:
        raise PdfPreprocessingError(f"config/model_revisions.json docling lock is incomplete: {path}")
    return repository, validate_docling_revision(revision, source=path)


def resolve_docling_lock(*, project_root: Path | None = None) -> tuple[str, str]:
    """Resolve the Docling repository and immutable revision.

    An explicit project root uses only that project's config/model_revisions.json
    and never falls back to another lock. The packaged lock is used only when
    no project root exists (standalone package use).
    """
    if project_root is not None:
        lock_path = project_root.expanduser().resolve() / "config/model_revisions.json"
        return _read_docling_lock(lock_path)
    return _read_docling_lock(_PACKAGED_LOCK)


def _docling_artifacts_path(repository: str, revision: str) -> Path:
    key = (repository, revision)
    cached = _ARTIFACTS.get(key)
    if cached is not None:
        return cached
    snapshot_download = import_module("huggingface_hub").snapshot_download
    path = Path(snapshot_download(repo_id=repository, revision=revision))
    _ARTIFACTS[key] = path
    return path


def _docling_converter(repository: str, revision: str) -> Any:
    """One DocumentConverter per immutable repository/revision pair."""
    key = (repository, revision)
    cached = _CONVERTERS.get(key)
    if cached is not None:
        return cached
    document_converter = import_module("docling.document_converter")
    converter = document_converter.DocumentConverter(
        artifacts_path=_docling_artifacts_path(repository, revision)
    )
    _CONVERTERS[key] = converter
    return converter


def reset_docling_converter() -> None:
    """Drop cached converters (test isolation; new HF cache locations)."""
    _CONVERTERS.clear()
    _ARTIFACTS.clear()


def convert_pdf_with_docling(
    pdf_path: Path,
    *,
    repository: str,
    revision: str,
) -> Any:
    """Convert one PDF with the pinned Docling runtime and return the conversion result."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Docling input PDF is missing: {pdf_path}")
    revision = validate_docling_revision(revision, source="caller")
    converter = _docling_converter(str(repository).strip(), revision)

    document_model = import_module("docling.datamodel.document")
    if hasattr(document_model, "DocumentConversionInput"):
        conversion_input = document_model.DocumentConversionInput.from_paths([pdf_path])
        results = list(converter.convert(conversion_input))
        if not results:
            raise PdfPreprocessingError(f"Docling returned no conversion result for {pdf_path}")
        result = results[0]
    else:
        result = converter.convert(pdf_path)

    status = getattr(result, "status", None)
    status_name = getattr(status, "name", str(status) if status is not None else "")
    pages = getattr(result, "pages", None)
    document = getattr(result, "document", None)
    if status_name and status_name != "SUCCESS":
        raise PdfPreprocessingError(f"Docling conversion status {status_name!r} for {pdf_path}")
    if document is None and not pages:
        raise PdfPreprocessingError(f"Docling returned no document/pages for {pdf_path}")
    return result


def _result_to_markdown(result: Any) -> str:
    if hasattr(result, "render_as_markdown"):
        return str(result.render_as_markdown())
    document = getattr(result, "document", None)
    if document is not None and hasattr(document, "export_to_markdown"):
        return str(document.export_to_markdown())
    return ""


def _result_to_audit_payload(result: Any, pdf_path: Path) -> dict[str, Any]:
    status = getattr(result, "status", None)
    pages = getattr(result, "pages", []) or []
    page_summaries = []
    for page in pages:
        page_summaries.append(
            {
                "page_no": getattr(page, "page_no", None),
                "page_hash": getattr(page, "page_hash", None),
                "width": getattr(getattr(page, "size", None), "width", None),
                "height": getattr(getattr(page, "size", None), "height", None),
                "cell_count": len(getattr(page, "cells", []) or []),
            }
        )
    payload: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "status": getattr(status, "name", str(status) if status is not None else None),
        "page_count": len(pages),
        "pages": page_summaries,
        "errors": [str(err) for err in (getattr(result, "errors", []) or [])],
    }
    return payload


def export_docling_markdown(
    pdf_path: Path,
    output_path: Path,
    *,
    repository: str | None = None,
    revision: str | None = None,
    project_root: Path | None = None,
) -> Path:
    """Convert a PDF and persist Docling's structured audit Markdown (+ JSON sidecar)."""
    if repository is None or revision is None:
        repository, revision = resolve_docling_lock(project_root=project_root)
    else:
        repository = str(repository).strip()
        revision = validate_docling_revision(revision, source="caller")
        if not repository:
            raise PdfPreprocessingError("Docling repository is empty.")
    pdf_path = pdf_path.expanduser().resolve()
    result = convert_pdf_with_docling(pdf_path, repository=repository, revision=revision)
    markdown = _result_to_markdown(result).strip()
    audit = _result_to_audit_payload(result, pdf_path)
    audit["docling_repository"] = repository
    audit["docling_revision"] = revision
    if not markdown:
        # Image-heavy pages may yield empty markdown while still converting successfully.
        markdown = (
            f"<!-- docling audit: status={audit.get('status')} "
            f"pages={audit.get('page_count')} -->\n"
        )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown if markdown.endswith("\n") else markdown + "\n", encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    try:
        from .asset_paths import to_relative_project_path

        root = project_root.expanduser().resolve() if project_root is not None else None
        audit["pdf_path"] = to_relative_project_path(pdf_path, root) if root is not None else pdf_path.name
    except Exception:
        audit["pdf_path"] = pdf_path.name
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit.get("page_count", 0) <= 0 and getattr(result, "document", None) is None:
        raise PdfPreprocessingError(f"Docling produced no pages for {pdf_path}")
    return output_path
