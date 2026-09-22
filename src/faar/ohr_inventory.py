from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def _page_ids_from_inventory_payload(payload: Any) -> list[int]:
    rows = payload if isinstance(payload, list) else payload.get("pages", payload.get("data", []))
    if not isinstance(rows, list) or not rows:
        return []
    return sorted(
        {
            int(row.get("page_idx", row.get("page_no", row.get("page_id", 0))))
            for row in rows
            if isinstance(row, dict)
        }
    )


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _skeleton(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii_fold(text).lower())


def _candidate_inventory_names(doc_name: str) -> list[str]:
    """Return deterministic alternate inventory keys for known OHR naming drift."""
    candidates = [doc_name]
    # qas_v2 uses textbook_needrop_*; retrieval_base/gt and pdfs.zip use jiaocai_needrop_*.
    if doc_name.startswith("textbook/textbook_needrop_en_"):
        candidates.append(doc_name.replace("textbook/textbook_needrop_en_", "textbook/jiaocai_needrop_en_", 1))
    # Strip accidental trailing .pdf before page-suffix document ids.
    if doc_name.endswith(".pdf"):
        candidates.append(doc_name[: -len(".pdf")])
    folded = _ascii_fold(doc_name)
    if folded != doc_name:
        candidates.append(folded)
    return list(dict.fromkeys(candidates))


def resolve_ohr_inventory_path(inventory_dir: Path, doc_name: str) -> tuple[Path | None, str | None, str]:
    """
    Resolve a qas doc_name to an on-disk inventory JSON.

    Returns (path, resolved_key, diagnosis) where diagnosis is one of:
      exact | alias | unicode_alias | missing
    """
    inventory_dir = inventory_dir.expanduser().resolve()
    for candidate in _candidate_inventory_names(doc_name):
        path = inventory_dir / f"{candidate}.json"
        if path.is_file():
            kind = "exact" if candidate == doc_name else "alias"
            if candidate != doc_name and _skeleton(doc_name) == _skeleton(candidate):
                kind = "unicode_alias"
            return path, candidate, kind

    # Unique alnum-skeleton match inside the same category (handles mojibake).
    category = doc_name.split("/", 1)[0] if "/" in doc_name else ""
    needle = _skeleton(Path(doc_name).name)
    matches: list[Path] = []
    search_root = inventory_dir / category if category else inventory_dir
    if search_root.is_dir():
        for path in search_root.glob("*.json"):
            if _skeleton(path.stem) == needle:
                matches.append(path)
    if len(matches) == 1:
        resolved = str(matches[0].relative_to(inventory_dir).with_suffix(""))
        return matches[0], resolved, "unicode_alias"
    return None, None, "missing"


def diagnose_ohr_inventory_gaps(
    *,
    qas_path: Path,
    inventory_dir: Path,
    pdf_names: set[str] | None = None,
) -> dict[str, Any]:
    """Classify qas documents missing a direct inventory filename match."""
    rows = json.loads(qas_path.read_text(encoding="utf-8"))
    docs = sorted({str(row["doc_name"]) for row in rows})
    naming_mismatches: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for doc_name in docs:
        path, resolved, kind = resolve_ohr_inventory_path(inventory_dir, doc_name)
        if kind == "exact":
            continue
        pdf_key = f"{resolved or doc_name}.pdf"
        pdf_alt = f"{doc_name}.pdf"
        has_pdf = False
        if pdf_names is not None:
            has_pdf = pdf_key in pdf_names or pdf_alt in pdf_names
            for candidate in _candidate_inventory_names(doc_name):
                if f"{candidate}.pdf" in pdf_names:
                    has_pdf = True
                    break
        entry = {
            "qas_doc_name": doc_name,
            "resolved_inventory_key": resolved,
            "diagnosis": kind,
            "pdf_present": has_pdf,
        }
        if path is not None:
            page_ids = _page_ids_from_inventory_payload(json.loads(path.read_text(encoding="utf-8")))
            entry["page_ids"] = page_ids
            entry["page_count"] = len(page_ids)
            naming_mismatches.append(entry)
        else:
            unavailable.append(entry)
    return {
        "naming_mismatches": naming_mismatches,
        "unavailable": unavailable,
        "qas_document_count": len(docs),
    }


def load_resolved_ohr_document_inventory(
    inventory_dir: Path,
    selected_documents: set[str],
) -> tuple[dict[str, list[int]], dict[str, dict[str, str]]]:
    """
    Load complete page ids keyed by the original qas doc_name.

    Page counts always come from real inventory JSON; never fabricated.
    """
    inventory: dict[str, list[int]] = {}
    resolutions: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for doc_name in sorted(selected_documents):
        path, resolved, kind = resolve_ohr_inventory_path(inventory_dir, doc_name)
        if path is None or resolved is None:
            missing.append(doc_name)
            continue
        page_ids = _page_ids_from_inventory_payload(json.loads(path.read_text(encoding="utf-8")))
        if not page_ids:
            missing.append(doc_name)
            continue
        inventory[doc_name] = page_ids
        resolutions[doc_name] = {"resolved_key": resolved, "diagnosis": kind, "inventory_path": str(path)}
    return inventory, resolutions
