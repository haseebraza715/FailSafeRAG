from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .asset_paths import AssetPathError, resolve_project_asset, to_relative_project_path

SUPPORTED_DATASETS = {"mpdocvqa", "arxivqa"}

_ALIASES = {
    "id": ("example_id", "question_id", "qa_id", "id", "ID"),
    "question": ("question", "questions", "query"),
    "answer": ("correct_answer", "answers", "answer", "ground_truth", "gold_answer"),
    "document": ("doc_name", "document_id", "doc_id", "document", "document_name", "file_name"),
    "page": ("page_ids", "page_id", "page_no", "page", "page_number", "evidence_page_no", "answer_page_idx"),
    "image": ("image_paths", "image_path", "page_images", "page_image", "image"),
    "ocr": ("ocr_text_paths", "ocr_text_path", "ocr_paths", "ocr_path", "ocr_file", "ocr"),
}

_PLACEHOLDER_VALUES = {
    "-",
    "?",
    "n/a",
    "na",
    "none",
    "null",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
}


class ExternalAssetError(ValueError):
    """Raised when external benchmark data cannot form a paper-ready manifest."""


def normalise_dataset_name(dataset: str) -> str:
    key = dataset.lower().replace("-", "").replace("_", "")
    aliases = {"mpdocvqa": "mpdocvqa", "arxivqa": "arxivqa"}
    try:
        return aliases[key]
    except KeyError as exc:
        raise ExternalAssetError(
            f"Unsupported external dataset {dataset!r}; expected MP-DocVQA or ArXivQA."
        ) from exc


def _first(row: dict[str, Any], kind: str) -> Any:
    for key in _ALIASES[kind]:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    return (
        lowered in _PLACEHOLDER_VALUES
        or "placeholder" in lowered
        or lowered.startswith(("path/to/", "your/path", "your_"))
        or bool(re.fullmatch(r"<[^>]+>", text))
    )


def _required_text(row: dict[str, Any], kind: str, row_label: str) -> str:
    value = _first(row, kind)
    if kind == "answer" and isinstance(value, list):
        value = next((item for item in value if not _is_placeholder(item)), None)
    if isinstance(value, dict):
        value = value.get("id") or value.get("name") or value.get("value")
    if _is_placeholder(value):
        raise ExternalAssetError(f"{row_label} has a missing or placeholder {kind}.")
    return str(value).strip()


def _page_number(value: Any, row_label: str) -> int:
    if isinstance(value, dict):
        value = _first(value, "page")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExternalAssetError(f"{row_label} has invalid page value {value!r}; expected an integer.") from exc


def _resolve_existing_file(value: Any, asset_root: Path, row_label: str, kind: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("uri")
    if _is_placeholder(value):
        raise ExternalAssetError(f"{row_label} has a missing or placeholder {kind} path.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = asset_root / path
    path = path.resolve()
    if not path.is_file():
        raise ExternalAssetError(f"{row_label} references a missing {kind} file: {path}")
    return path


def _extract_val_rows(payload: Any) -> list[dict[str, Any]]:
    split_proven = False
    rows: Any = payload
    if isinstance(payload, dict):
        declared_split = payload.get("split") or payload.get("dataset_split")
        if declared_split is not None:
            if str(declared_split).strip().lower() != "val":
                raise ExternalAssetError(f"External benchmark source must be exactly the val split, not {declared_split!r}.")
            split_proven = True
        if "val" in payload:
            rows = payload["val"]
            split_proven = True
        else:
            rows = payload.get("records", payload.get("data", payload.get("examples")))
    if not isinstance(rows, list) or not rows:
        raise ExternalAssetError("External benchmark source must contain a non-empty list of val records.")
    if not all(isinstance(row, dict) for row in rows):
        raise ExternalAssetError("Every external benchmark record must be a JSON object.")

    row_splits = {
        str(row.get("split") or row.get("dataset_split")).strip().lower()
        for row in rows
        if row.get("split") or row.get("dataset_split")
    }
    if row_splits:
        if row_splits != {"val"}:
            raise ExternalAssetError(f"External benchmark records must all be val; found splits {sorted(row_splits)}.")
        if len(row_splits) == 1 and all(row.get("split") or row.get("dataset_split") for row in rows):
            split_proven = True
    if not split_proven:
        raise ExternalAssetError(
            "Could not verify the exact val split. Use a top-level 'val' key, set top-level split='val', "
            "or mark every record split='val'."
        )
    return rows


def _evidence_page_ids(row: dict[str, Any], example_id: str) -> list[int]:
    """Evidence/answer pages for evaluation only; never used to build the retrieval corpus."""
    if "evidence_page_no" in row or "evidence_page_ids" in row or "answer_page_idx" in row:
        values = _as_list(row.get("evidence_page_ids") or row.get("evidence_page_no") or row.get("answer_page_idx"))
    else:
        values = _as_list(_first(row, "page"))
    if not values:
        raise ExternalAssetError(f"{example_id} is missing evidence/answer page ids.")
    return [_page_number(value, example_id) for value in values]


def _document_inventory(payload: Any, asset_root: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Complete document-page inventory, separate from QA evidence pages.

    Accepted shapes:
      {"documents": [{"doc_id": "...", "pages": [{"page_id": 1, "image_path": "...", "ocr_text_path": "..."}, ...]}]}
      {"document_pages": [{"doc_name": "...", "page_id": 1, "image_path": "...", "ocr_text_path": "..."}, ...]}
    """
    if not isinstance(payload, dict):
        raise ExternalAssetError(
            "Complete document pages cannot be proven: source must be a JSON object with a documents inventory."
        )

    inventory: dict[str, dict[int, dict[str, Any]]] = {}

    documents = payload.get("documents")
    if isinstance(documents, list) and documents:
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise ExternalAssetError("documents entries must be JSON objects.")
            doc_id = _required_text(document, "document", f"documents[{index}]")
            pages = document.get("pages") or document.get("document_pages")
            if not isinstance(pages, list) or not pages:
                raise ExternalAssetError(f"Document {doc_id!r} has an empty pages inventory.")
            for page in pages:
                if not isinstance(page, dict):
                    raise ExternalAssetError(f"Document {doc_id!r} page entries must be JSON objects.")
                page_id = _page_number(_first(page, "page"), doc_id)
                image_path = _resolve_existing_file(_first(page, "image"), asset_root, doc_id, "image")
                ocr_path = _resolve_existing_file(_first(page, "ocr"), asset_root, doc_id, "OCR")
                inventory.setdefault(doc_id, {})[page_id] = {
                    "page_id": page_id,
                    "image_path": image_path,
                    "ocr_text_path": ocr_path,
                }
    else:
        document_pages = payload.get("document_pages")
        if not isinstance(document_pages, list) or not document_pages:
            raise ExternalAssetError(
                "Complete document pages cannot be proven. Provide top-level 'documents' (with full page lists) "
                "or 'document_pages'. QA evidence pages alone are not a retrieval corpus."
            )
        for index, page in enumerate(document_pages):
            if not isinstance(page, dict):
                raise ExternalAssetError("document_pages entries must be JSON objects.")
            doc_id = _required_text(page, "document", f"document_pages[{index}]")
            page_id = _page_number(_first(page, "page"), doc_id)
            image_path = _resolve_existing_file(_first(page, "image"), asset_root, doc_id, "image")
            ocr_path = _resolve_existing_file(_first(page, "ocr"), asset_root, doc_id, "OCR")
            inventory.setdefault(doc_id, {})[page_id] = {
                "page_id": page_id,
                "image_path": image_path,
                "ocr_text_path": ocr_path,
            }

    return {
        doc_id: [pages[page_id] for page_id in sorted(pages)]
        for doc_id, pages in sorted(inventory.items())
    }


def build_external_asset_manifest(
    source: Path,
    dataset: str,
    *,
    asset_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Normalize an MP-DocVQA/ArXivQA val JSON file into FAAR's asset schema."""
    dataset_key = normalise_dataset_name(dataset)
    source = source.resolve()
    if not source.is_file():
        raise ExternalAssetError(f"External benchmark source does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalAssetError(f"Could not read external benchmark JSON {source}: {exc}") from exc
    rows = _extract_val_rows(payload)
    root = (asset_root or source.parent).resolve()
    if not root.is_dir():
        raise ExternalAssetError(f"Asset root does not exist or is not a directory: {root}")
    project = (project_root or root).resolve()

    documents = _document_inventory(payload, root)
    if not documents:
        raise ExternalAssetError("Complete document pages cannot be proven: documents inventory is empty.")

    records: list[dict[str, Any]] = []
    corpus_by_id: dict[str, dict[str, Any]] = {}
    seen_examples: set[str] = set()

    for row_index, row in enumerate(rows):
        row_label = f"record {row_index}"
        example_id = _required_text(row, "id", row_label)
        if example_id in seen_examples:
            raise ExternalAssetError(f"Duplicate example id {example_id!r}.")
        seen_examples.add(example_id)
        question = _required_text(row, "question", example_id)
        answer = _required_text(row, "answer", example_id)
        document_id = _required_text(row, "document", example_id)
        if document_id not in documents:
            raise ExternalAssetError(
                f"{example_id} references document {document_id!r} with no complete document-page inventory."
            )

        evidence_pages = _evidence_page_ids(row, example_id)
        inventory_page_ids = {int(page["page_id"]) for page in documents[document_id]}
        missing_evidence = [page_id for page_id in evidence_pages if page_id not in inventory_page_ids]
        if missing_evidence:
            raise ExternalAssetError(
                f"{example_id} evidence pages {missing_evidence} are absent from the complete document inventory."
            )

        # Corpus is every inventoried page of the document — never evidence-only.
        for page in documents[document_id]:
            page_id = int(page["page_id"])
            corpus_id = f"{document_id}:p{page_id}"
            try:
                image_rel = to_relative_project_path(page["image_path"], project)
                ocr_rel = to_relative_project_path(page["ocr_text_path"], project)
            except AssetPathError as exc:
                raise ExternalAssetError(str(exc)) from exc
            corpus_page = {
                "corpus_id": corpus_id,
                "doc_name": document_id,
                "page_id": page_id,
                "ocr_text_path": ocr_rel,
                "image_path": image_rel,
            }
            existing = corpus_by_id.get(corpus_id)
            if existing is not None and existing != corpus_page:
                raise ExternalAssetError(f"Conflicting assets for corpus page {corpus_id!r}.")
            corpus_by_id[corpus_id] = corpus_page

        records.append(
            {
                "example_id": example_id,
                "doc_name": document_id,
                "question": question,
                "correct_answer": answer,
                "page_ids": evidence_pages,
                "corpus_ids": [f"{document_id}:p{page_id}" for page_id in sorted(inventory_page_ids)],
                "metadata": {
                    "dataset": dataset_key,
                    "split": "val",
                    "document_id": document_id,
                    "evidence_page_ids": evidence_pages,
                },
            }
        )

    try:
        source_rel = to_relative_project_path(source, project)
    except AssetPathError:
        source_rel = str(source)

    return {
        "dataset": dataset_key,
        "split": "val",
        "source": source_rel,
        "records": records,
        "corpus_pages": [corpus_by_id[key] for key in sorted(corpus_by_id)],
        "document_inventory": {
            doc_id: [int(page["page_id"]) for page in pages] for doc_id, pages in documents.items()
        },
    }


def validate_manifest_payload(payload: dict[str, Any], *, project_root: Path | None = None) -> None:
    """Validate a normalized payload without relying on the source JSON."""
    if payload.get("split") != "val":
        raise ExternalAssetError("External asset manifest split must be exactly 'val'.")
    if normalise_dataset_name(str(payload.get("dataset", ""))) not in SUPPORTED_DATASETS:
        raise ExternalAssetError("External asset manifest has an unsupported dataset.")
    records = payload.get("records")
    pages = payload.get("corpus_pages")
    inventory = payload.get("document_inventory")
    if not isinstance(records, list) or not records or not isinstance(pages, list) or not pages:
        raise ExternalAssetError("External asset manifest requires non-empty records and corpus_pages lists.")
    if not isinstance(inventory, dict) or not inventory:
        raise ExternalAssetError(
            "Complete document pages cannot be proven: document_inventory is required and must be non-empty."
        )

    page_ids = {page.get("corpus_id") for page in pages if isinstance(page, dict)}
    pages_by_doc: dict[str, set[int]] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ExternalAssetError("corpus_pages entries must be JSON objects.")
        doc_name = str(page.get("doc_name", ""))
        pages_by_doc.setdefault(doc_name, set()).add(int(page["page_id"]))
        if project_root is not None:
            for key in ("image_path", "ocr_text_path"):
                value = page.get(key)
                if value:
                    resolve_project_asset(str(value), project_root)

    for doc_id, expected_pages in inventory.items():
        expected = {int(page_id) for page_id in expected_pages}
        actual = pages_by_doc.get(str(doc_id), set())
        if actual != expected:
            raise ExternalAssetError(
                f"Corpus pages for document {doc_id!r} do not match complete inventory "
                f"(expected {sorted(expected)}, found {sorted(actual)})."
            )

    for record in records:
        if not isinstance(record, dict):
            raise ExternalAssetError("records entries must be JSON objects.")
        doc_name = str(record.get("doc_name", ""))
        evidence = [int(page_id) for page_id in record.get("page_ids", [])]
        inventory_pages = {int(page_id) for page_id in inventory.get(doc_name, [])}
        if not evidence or not set(evidence).issubset(inventory_pages):
            raise ExternalAssetError("Every query evidence page must belong to the complete document inventory.")
        expected_corpus_ids = {f"{doc_name}:p{page_id}" for page_id in inventory_pages}
        if set(record.get("corpus_ids", [])) != expected_corpus_ids:
            raise ExternalAssetError(
                "Every query record must reference every inventoried page of its document via corpus_ids."
            )
        if not expected_corpus_ids.issubset(page_ids):
            raise ExternalAssetError("Every query record must reference registered corpus_pages.")
