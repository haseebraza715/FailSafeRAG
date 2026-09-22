from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .arxivqa_prepare import (
    STAGED_ROW_FINGERPRINT_ALGORITHM,
    inspect_pdf,
    qa_rows_content_fingerprint,
)
from .asset_preparation import sha256_file


class ArXivQARemapError(ValueError):
    """Raised when ArXivQA cannot be mapped to a complete paper corpus."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArXivQARemapError(f"Could not read JSON source {path}: {exc}") from exc


def _rows(payload: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("val") or payload.get("records") or payload.get("data")
    if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
        raise ArXivQARemapError(f"{label} must contain a non-empty list of JSON objects.")
    return payload


def _text(row: dict[str, Any], keys: tuple[str, ...], label: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ArXivQARemapError(f"{label} is missing one of: {', '.join(keys)}.")


def _page_id(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArXivQARemapError(f"{label} has invalid page id {value!r}.") from exc


def _resolve_project_path(path_value: str | Path, *, project_root: Path | None, label: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    if project_root is None:
        raise ArXivQARemapError(f"{label} is project-relative and requires project_root: {path}")
    return project_root / path


def _load_locked_got_ocr(project_root: Path) -> dict[str, str]:
    lock_path = project_root / "config" / "model_revisions.json"
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        locked = payload["models"]["got_ocr"]
        repository = str(locked["repository"])
        revision = str(locked["revision"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArXivQARemapError(
            "Remap requires pinned GOT-OCR repository/revision from config/model_revisions.json."
        ) from exc
    if not repository or not revision:
        raise ArXivQARemapError("Pinned GOT-OCR repository/revision in model_revisions.json is incomplete.")
    return {"repository": repository, "revision": revision}


def _require_project_validation_policy(payload: Any, qa_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ArXivQARemapError("ArXivQA QA source must retain its locked source provenance.")
    lock = payload.get("source_lock")
    policy = payload.get("project_role_policy")
    inputs = payload.get("remapping_inputs")
    if not isinstance(lock, dict) or not isinstance(policy, dict) or not isinstance(inputs, dict):
        raise ArXivQARemapError("ArXivQA QA source lacks locked provenance or project-role policy.")
    if (
        payload.get("upstream_split") != "test"
        or payload.get("supervisor_approved_project_role") != "validation"
        or policy.get("allows_upstream_test_as_project_validation") is not True
        or not str(policy.get("policy_id") or "")
    ):
        raise ArXivQARemapError("ArXivQA QA source lacks the explicit upstream-test-to-project-validation policy.")
    vidore = lock.get("vidore_source")
    official = lock.get("official_source")
    if not isinstance(vidore, dict) or not isinstance(official, dict):
        raise ArXivQARemapError("ArXivQA QA source has incomplete locked source evidence.")
    expected_rows = vidore.get("expected_rows")
    fingerprint = lock.get("staged_row_fingerprint_sha256")
    if (
        not isinstance(expected_rows, int)
        or len(qa_rows) != expected_rows
        or inputs.get("row_count") != expected_rows
        or not isinstance(fingerprint, str)
        or official.get("upstream_split") != "test"
    ):
        raise ArXivQARemapError("ArXivQA QA source does not prove the locked complete ViDoRe subset.")
    try:
        recomputed = qa_rows_content_fingerprint(qa_rows)
    except Exception as exc:
        raise ArXivQARemapError(
            f"ArXivQA QA source content cannot be fingerprinted with the shared staged-row algorithm: {exc}"
        ) from exc
    if recomputed != fingerprint:
        raise ArXivQARemapError(
            "ArXivQA QA source content fingerprint does not match the locked staged-row fingerprint."
        )
    if inputs.get("staged_row_fingerprint_sha256") != fingerprint:
        raise ArXivQARemapError("ArXivQA QA source remapping_inputs fingerprint disagrees with source_lock.")
    return {
        "upstream_split": payload["upstream_split"],
        "supervisor_approved_project_role": payload["supervisor_approved_project_role"],
        "project_role_policy": policy,
        "source_lock": {
            **lock,
            "staged_row_fingerprint_sha256": fingerprint,
            "staged_row_fingerprint_algorithm": STAGED_ROW_FINGERPRINT_ALGORITHM,
        },
        "content_fingerprint_sha256": recomputed,
    }


def _verify_ocr_ready_paper(
    paper: dict[str, Any],
    *,
    paper_id: str,
    project_root: Path,
    locked_got_ocr: dict[str, str],
    inventory_readiness: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        inventory_readiness.get("state") != "verified_pinned_got_ocr"
        or inventory_readiness.get("repository") != locked_got_ocr["repository"]
        or inventory_readiness.get("revision") != locked_got_ocr["revision"]
    ):
        raise ArXivQARemapError(
            "Paper inventory OCR readiness is not bound to config/model_revisions.json GOT-OCR pins."
        )
    provenance_rel = paper.get("got_ocr_provenance")
    if not provenance_rel:
        raise ArXivQARemapError(f"Paper {paper_id!r} is missing got_ocr_provenance.")
    provenance_path = _resolve_project_path(str(provenance_rel), project_root=project_root, label="got_ocr_provenance")
    provenance = _read_json(provenance_path)
    if (
        not isinstance(provenance, dict)
        or provenance.get("paper_id") != paper_id
        or provenance.get("pdf_sha256") != paper.get("pdf_sha256")
        or provenance.get("got_ocr_repository") != locked_got_ocr["repository"]
        or provenance.get("got_ocr_revision") != locked_got_ocr["revision"]
    ):
        raise ArXivQARemapError(f"Pinned GOT-OCR provenance does not match paper {paper_id!r}.")
    provenance_pages = provenance.get("pages")
    if not isinstance(provenance_pages, dict):
        raise ArXivQARemapError(f"Pinned GOT-OCR provenance has no pages for {paper_id!r}.")

    page_count = paper.get("page_count")
    if not isinstance(page_count, int) or page_count <= 0:
        raise ArXivQARemapError(f"Paper {paper_id!r} has no valid page_count.")
    pages = paper.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ArXivQARemapError(f"Paper {paper_id!r} has no complete page inventory.")

    pdf_path_value = paper.get("pdf_path")
    if pdf_path_value:
        pdf_path = _resolve_project_path(str(pdf_path_value), project_root=project_root, label="pdf_path")
        try:
            actual_page_count = inspect_pdf(pdf_path)
        except Exception as exc:
            raise ArXivQARemapError(f"Paper {paper_id!r} PDF cannot be verified: {exc}") from exc
        actual_pdf_sha = sha256_file(pdf_path)
        if page_count != actual_page_count or paper.get("pdf_sha256") != actual_pdf_sha:
            raise ArXivQARemapError(f"Paper {paper_id!r} PDF page count or SHA-256 does not match.")

    normalized_pages: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for page in pages:
        if not isinstance(page, dict):
            raise ArXivQARemapError(f"Paper {paper_id!r} has a non-object page entry.")
        page_id = _page_id(page.get("page_id", page.get("page_number")), paper_id)
        if page_id in seen_pages:
            raise ArXivQARemapError(f"Paper {paper_id!r} repeats page {page_id}.")
        image_path = page.get("image_path") or page.get("page_image")
        ocr_path = page.get("ocr_text_path") or page.get("ocr_path") or page.get("ocr_file")
        image_sha = page.get("image_sha256")
        ocr_sha = page.get("ocr_sha256")
        if not image_path or not ocr_path or not image_sha or not ocr_sha:
            raise ArXivQARemapError(
                f"Paper {paper_id!r} page {page_id} needs hashed image and pinned GOT-OCR paths."
            )
        resolved_image = _resolve_project_path(str(image_path), project_root=project_root, label="image_path")
        resolved_ocr = _resolve_project_path(str(ocr_path), project_root=project_root, label="ocr_text_path")
        if not resolved_image.is_file() or resolved_image.stat().st_size <= 0:
            raise ArXivQARemapError(f"Paper {paper_id!r} page {page_id} image is missing.")
        if not resolved_ocr.is_file() or resolved_ocr.stat().st_size <= 0:
            raise ArXivQARemapError(f"Paper {paper_id!r} page {page_id} OCR text is missing.")
        actual_image_sha = sha256_file(resolved_image)
        actual_ocr_sha = sha256_file(resolved_ocr)
        if image_sha != actual_image_sha:
            raise ArXivQARemapError(f"Paper {paper_id!r} page {page_id} image hash does not match.")
        if ocr_sha != actual_ocr_sha:
            raise ArXivQARemapError(f"Paper {paper_id!r} page {page_id} OCR hash does not match.")
        page_provenance = provenance_pages.get(str(page_id))
        if not isinstance(page_provenance, dict) or page_provenance.get("ocr_sha256") != actual_ocr_sha:
            raise ArXivQARemapError(f"Pinned GOT-OCR hash does not match for {paper_id!r} page {page_id}.")
        seen_pages.add(page_id)
        normalized_pages.append(
            {
                "page_id": page_id,
                "image_path": str(image_path),
                "ocr_text_path": str(ocr_path),
            }
        )

    ids = [page["page_id"] for page in normalized_pages]
    if ids != list(range(page_count)):
        raise ArXivQARemapError(
            f"Paper {paper_id!r} pages must be exactly 0..{page_count - 1}; found {ids}."
        )
    return sorted(normalized_pages, key=lambda item: item["page_id"])


def build_full_paper_source(
    qa_source: Path,
    paper_inventory: Path,
    figure_mapping: Path,
    *,
    output: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build a strict full-paper ArXivQA source for external asset registration.

    The figure mapping is deliberately explicit. It must identify the paper and
    answer page for every QA row; retrieval later receives every page in the
    paper inventory, not only the mapped evidence page.
    """
    root = (project_root or Path.cwd()).resolve()
    qa_payload = _read_json(qa_source)
    qa_rows = _rows(qa_payload, "ArXivQA source")
    qa_provenance = _require_project_validation_policy(qa_payload, qa_rows)
    inventory_payload = _read_json(paper_inventory)
    if not isinstance(inventory_payload, dict) or inventory_payload.get("state") != "ocr_ready":
        raise ArXivQARemapError(
            "Paper inventory is preparation-only. Verify every pinned GOT-OCR file and provenance before remapping."
        )
    inventory_rows = inventory_payload.get("papers")
    if not isinstance(inventory_rows, list) or not inventory_rows:
        raise ArXivQARemapError("Paper inventory must contain a non-empty 'papers' list.")
    readiness = inventory_payload.get("ocr_readiness")
    if not isinstance(readiness, dict):
        raise ArXivQARemapError("Paper inventory is missing OCR readiness provenance.")
    locked_got_ocr = _load_locked_got_ocr(root)
    mapping_rows = _rows(_read_json(figure_mapping), "figure mapping")

    papers: dict[str, dict[str, Any]] = {}
    for index, paper in enumerate(inventory_rows):
        if not isinstance(paper, dict):
            raise ArXivQARemapError(f"papers[{index}] must be a JSON object.")
        paper_id = _text(paper, ("paper_id", "doc_id", "id"), f"papers[{index}]")
        normalized_pages = _verify_ocr_ready_paper(
            paper,
            paper_id=paper_id,
            project_root=root,
            locked_got_ocr=locked_got_ocr,
            inventory_readiness=readiness,
        )
        papers[paper_id] = {"paper_id": paper_id, "pages": normalized_pages}

    qa_paper_ids: dict[str, str] = {}
    for index, row in enumerate(qa_rows):
        qa_id = _text(row, ("example_id", "qa_id", "question_id", "id"), f"ArXivQA row {index}")
        paper_id = _text(row, ("paper_id", "doc_id", "document_id"), qa_id)
        qa_paper_ids[qa_id] = paper_id

    mappings: dict[str, dict[str, Any]] = {}
    for index, mapping in enumerate(mapping_rows):
        qa_id = _text(mapping, ("qa_id", "example_id", "id"), f"figure mapping[{index}]")
        if qa_id in mappings:
            raise ArXivQARemapError(f"Figure mapping repeats QA id {qa_id!r}.")
        paper_id = _text(mapping, ("paper_id", "doc_id", "document_id"), qa_id)
        staged_paper_id = qa_paper_ids.get(qa_id)
        if staged_paper_id is None:
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} does not match any staged QA row.")
        if paper_id != staged_paper_id:
            raise ArXivQARemapError(
                f"Figure mapping {qa_id!r} paper_id {paper_id!r} does not equal staged QA paper_id {staged_paper_id!r}."
            )
        if paper_id not in papers:
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} references unknown paper {paper_id!r}.")
        raw_pages = mapping.get("evidence_page_ids") or mapping.get("answer_page_ids") or mapping.get("page_ids")
        if not isinstance(raw_pages, list) or not raw_pages:
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} needs explicit evidence_page_ids.")
        evidence_pages = [_page_id(value, qa_id) for value in raw_pages]
        known_pages = {page["page_id"] for page in papers[paper_id]["pages"]}
        if not set(evidence_pages).issubset(known_pages):
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} points outside paper {paper_id!r}'s page inventory.")
        confirmation = mapping.get("human_confirmation")
        if (
            not isinstance(confirmation, dict)
            or not str(confirmation.get("confirmed_by") or "").strip()
            or not str(confirmation.get("confirmed_at_utc") or "").strip()
            or confirmation.get("method") != "human"
        ):
            raise ArXivQARemapError(
                f"Figure mapping {qa_id!r} needs explicit human_confirmation before it can become evidence."
            )
        mappings[qa_id] = {
            "paper_id": paper_id,
            "evidence_page_ids": evidence_pages,
            "figure_id": mapping.get("figure_id") or mapping.get("figure_image"),
            "human_confirmation": confirmation,
        }

    records: list[dict[str, Any]] = []
    for index, row in enumerate(qa_rows):
        qa_id = _text(row, ("example_id", "qa_id", "question_id", "id"), f"ArXivQA row {index}")
        mapping = mappings.get(qa_id)
        if mapping is None:
            raise ArXivQARemapError(f"ArXivQA row {qa_id!r} has no explicit figure-to-paper mapping.")
        paper_id = mapping["paper_id"]
        question = _text(row, ("question", "questions", "query"), qa_id)
        answer = _text(row, ("answer", "answers", "ground_truth", "correct_answer"), qa_id)
        if isinstance(row.get("answers"), list):
            answer = _text({"answer": next((item for item in row["answers"] if str(item).strip()), None)}, ("answer",), qa_id)
        records.append(
            {
                "id": qa_id,
                "question": question,
                "answer": answer,
                "document": paper_id,
                "evidence_page_ids": mapping["evidence_page_ids"],
                "metadata": {
                    "remapping_method": "full_paper",
                    "source_qa_id": qa_id,
                    "figure_id": mapping["figure_id"],
                    "human_confirmation": mapping["human_confirmation"],
                },
            }
        )

    payload = {
        "split": "val",
        "upstream_split": qa_provenance["upstream_split"],
        "supervisor_approved_project_role": qa_provenance["supervisor_approved_project_role"],
        "project_role_policy": qa_provenance["project_role_policy"],
        "source_lock": qa_provenance["source_lock"],
        "remapping": {
            "method": "full_paper",
            "qa_source": str(qa_source),
            "paper_inventory": str(paper_inventory),
            "figure_mapping": str(figure_mapping),
            "content_fingerprint_sha256": qa_provenance["content_fingerprint_sha256"],
            "got_ocr": locked_got_ocr,
        },
        "data": records,
        "documents": [
            {"doc_id": paper["paper_id"], "pages": paper["pages"]}
            for paper in sorted(papers.values(), key=lambda item: item["paper_id"])
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
