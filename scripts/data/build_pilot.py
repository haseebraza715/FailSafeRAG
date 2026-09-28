"""Build a frozen OHR-Bench development pilot and a local human inspection packet.

Reads the locked QA source, split, PDF archive, retrieval-base text and the
asset-audit manifest. It never modifies them. Outputs (under the config's
`output_dir`):

  selection_record.json      seed, pool, allocation, exclusions, source hashes, digests
  runtime_manifest.json      what a later system may read (no gold labels, no gt text)
  evaluation_manifest.json   gold answers, evidence labels/contexts, gt references, flags
  inspection/index.html      static packet (no scripts, local images only)
  inspection/annotations.csv blank observation template
  inspection/render_record.json renderer provenance and measured cost

Selection uses document identity and metadata only. Gold labels are used only
to choose which pages the inspection packet displays.

Exit codes:
  0  pilot built, or an existing frozen pilot reproduced exactly
  1  inputs invalid or stale (hash/audit mismatch), or the eligible pool is too small
  3  an existing frozen pilot differs from what these inputs and config produce;
     give the config a new pilot_id/output_dir instead of replacing it

Usage:
  python scripts/data/build_pilot.py --project-root . --config config/pilots/ohr_dev_v1.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "stratified-hash-rank-v1"
INSPECTION_ALGORITHM = "purposive-quota-hash-rank-v1"
EXIT_INVALID = 1
EXIT_FROZEN_MISMATCH = 3
OBSERVATION_CATEGORIES = [
    "word_corruption",
    "lost_or_misleading_structure",
    "missing_content",
    "apparently_adequate_ocr",
    "uncertain_or_mixed",
]
RUNTIME_FORBIDDEN_KEYS = {"answers", "answer", "evidence_page_no", "evidence_pages", "evidence_context", "gt_text", "failure_label", "annotations"}

_spec = spec_from_file_location("audit_assets", Path(__file__).resolve().parent / "audit_assets.py")
audit = module_from_spec(_spec)
_spec.loader.exec_module(audit)


class PilotError(RuntimeError):
    def __init__(self, message: str, code: int = EXIT_INVALID) -> None:
        super().__init__(message)
        self.code = code


def digest(payload: Any) -> str:
    return audit.canonical_sha256(payload)


def rank_key(seed: int, *parts: str) -> str:
    """Order-independent deterministic rank: sha256 over the seed and an identity string."""
    return hashlib.sha256("\x1f".join([str(seed), *parts]).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- inputs


def load_inputs(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load sources and refuse to proceed when the audit no longer describes them."""
    manifest_path = project_root / config["audit_manifest"]
    manifest, err = audit.read_json(manifest_path)
    if err or not isinstance(manifest, dict):
        raise PilotError(f"audit manifest {config['audit_manifest']} is {err or 'malformed'}; run scripts/data/audit_assets.py first")
    if audit.content_digest(manifest) != manifest.get("content_sha256"):
        raise PilotError("audit manifest content_sha256 does not match its content")
    if manifest.get("audit_execution", {}).get("integrity_failures"):
        raise PilotError(f"audit reports integrity failures: {manifest['audit_execution']['integrity_failures']}")

    lock, err = audit.read_json(project_root / config["pdf_source_lock"])
    if err or not isinstance(lock, dict):
        raise PilotError(f"PDF source lock {config['pdf_source_lock']} is {err or 'malformed'}")
    archive_lock = lock["pdf_archive"]
    archive_path = project_root / archive_lock["local_path"]
    if not archive_path.is_file():
        raise PilotError(f"PDF archive {archive_lock['local_path']} is missing")

    checks = {
        config["qa_source"]: (project_root / config["qa_source"], manifest["hashes"]["files"].get(config["qa_source"], {}).get("sha256")),
        config["split"]: (project_root / config["split"], manifest["hashes"]["files"].get(config["split"], {}).get("sha256")),
        archive_lock["local_path"]: (archive_path, archive_lock["sha256"]),
    }
    hashes = {}
    for rel, (path, expected) in checks.items():
        actual = audit.cached_sha256(path)
        if actual != expected:
            raise PilotError(f"{rel} SHA-256 {actual} does not match the audit/lock value {expected}; rerun the audit")
        hashes[rel] = actual
    trees = {}
    for source in (config["noisy_text_source"], config["reference_text_source"]):
        rel = f"{audit.RETRIEVAL_BASE_REL}/{source}"
        actual = audit.tree_digest(project_root / rel)
        if actual != manifest["hashes"]["tree_digests"].get(rel):
            raise PilotError(f"{rel} tree digest differs from the audit manifest; rerun the audit")
        trees[rel] = actual

    qas = json.loads((project_root / config["qa_source"]).read_text(encoding="utf-8"))
    split = json.loads((project_root / config["split"]).read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "lock": lock,
        "lock_sha256": audit.sha256_file(project_root / config["pdf_source_lock"]),
        "archive_path": archive_path,
        "qas": qas,
        "split": split,
        "hashes": hashes,
        "trees": trees,
    }


def canonical_ids(manifest: dict[str, Any], docs: set[str]) -> dict[str, str]:
    """Raw QA doc name -> canonical id: the NFC form of the gt inventory key it resolves to.

    Uses the audit's recorded resolutions (exact, documented alias, Unicode NFC);
    unresolved names keep their NFC raw name.
    """
    resolved = {r["qa_doc_name"]: r["resolved_key"] for r in manifest["ohr"]["coverage"]["text_sources"]["gt"]["alias_resolutions"]}
    return {d: audit._nfc(resolved.get(d) or d) for d in docs}


def text_file_index(project_root: Path, source: str) -> dict[str, Path]:
    root = project_root / audit.RETRIEVAL_BASE_REL / source
    return {audit._nfc(p.relative_to(root).with_suffix("").as_posix()): p for p in root.rglob("*.json") if p.is_file()}


# ---------------------------------------------------------------- selection


def build_pool(inputs: dict[str, Any]) -> dict[str, Any]:
    manifest, qas, split = inputs["manifest"], inputs["qas"], inputs["split"]
    split_of = {qid: name for name, ids in split["splits"].items() for qid in ids}
    raw_docs = {str(r["doc_name"]) for r in qas}
    canon = canonical_ids(manifest, raw_docs)
    splits_by_doc: dict[str, set[str]] = defaultdict(set)
    raw_by_doc: dict[str, set[str]] = defaultdict(set)
    for r in qas:
        cid = canon[str(r["doc_name"])]
        splits_by_doc[cid].add(split_of.get(str(r["ID"]), "unassigned"))
        raw_by_doc[cid].add(str(r["doc_name"]))
    train_exclusive = sorted(d for d, s in splits_by_doc.items() if s == {"train"})
    pdf_docs = manifest["ohr"]["coverage"]["pdf_per_document"]
    gt_unresolved = set(manifest["ohr"]["coverage"]["text_sources"]["gt"]["unresolved_documents"])
    eligible, exclusions = [], []
    for cid in train_exclusive:
        reasons = []
        raws = sorted(raw_by_doc[cid])
        if len(raws) > 1:
            reasons.append(f"several raw QA names map to this document: {raws}")
        statuses = {pdf_docs.get(r, {}).get("status", "not_in_audit") for r in raws}
        if statuses != {"verified_complete"}:
            reasons.append(f"source PDF status {sorted(statuses)}")
        elif any(len(pdf_docs[r]["candidates"]) != 1 for r in raws):
            reasons.append("source PDF has more than one candidate")
        if any(r in gt_unresolved for r in raws):
            reasons.append("no gt reference for evaluation")
        if reasons:
            exclusions.append({"doc_id": cid, "raw_doc_names": raws, "reasons": reasons})
        else:
            eligible.append(cid)
    return {
        "canon": canon,
        "split_of": split_of,
        "splits_by_doc": {d: sorted(s) for d, s in splits_by_doc.items()},
        "raw_by_doc": {d: sorted(s) for d, s in raw_by_doc.items()},
        "train_exclusive": train_exclusive,
        "eligible": eligible,
        "exclusions": exclusions,
    }


def doc_type(doc_id: str) -> str:
    return doc_id.split("/", 1)[0]


def allocate(counts: dict[str, int], total: int, minimum: int) -> list[dict[str, Any]]:
    """Largest-remainder allocation after a per-stratum minimum.

    Each available stratum first receives min(minimum, available) slots when the
    total allows it. Remaining slots follow proportional quotas of the pool;
    ties in the remainder break by larger pool size, then stratum name.
    """
    strata = sorted(counts)
    alloc = {s: 0 for s in strata}
    if minimum and total >= minimum * len(strata):
        for s in strata:
            alloc[s] = min(minimum, counts[s])
    remaining = total - sum(alloc.values())
    pool = sum(counts.values())
    quotas = {s: total * counts[s] / pool for s in strata}
    while remaining > 0:
        open_strata = [s for s in strata if alloc[s] < counts[s]]
        if not open_strata:
            break
        # Deficit against the proportional quota; ties -> larger pool, then name.
        best = max(open_strata, key=lambda s: (quotas[s] - alloc[s], counts[s], [-ord(c) for c in s]))
        alloc[best] += 1
        remaining -= 1
    return [{"doc_type": s, "eligible_documents": counts[s], "proportional_quota": round(quotas[s], 4), "allocated": alloc[s]} for s in strata]


def select_documents(eligible: list[str], config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, str]]:
    sel = config["selection"]
    if len(eligible) < sel["documents"]:
        raise PilotError(f"eligible pool has {len(eligible)} documents; the config asks for {sel['documents']}. Not expanding beyond the train-exclusive pool.")
    by_type: dict[str, list[str]] = defaultdict(list)
    for d in eligible:
        by_type[doc_type(d)].append(d)
    allocation = allocate({t: len(v) for t, v in by_type.items()}, sel["documents"], sel.get("minimum_per_available_stratum", 0))
    ranks = {d: rank_key(sel["seed"], "select", d) for d in eligible}
    chosen = []
    for row in allocation:
        members = sorted(by_type[row["doc_type"]], key=lambda d: (ranks[d], d))
        chosen.extend(members[: row["allocated"]])
    return sorted(chosen), allocation, {d: ranks[d] for d in chosen}


# ---------------------------------------------------------------- manifests


def page_bins(n: int) -> str:
    for hi, label in ((1, "1"), (5, "2-5"), (20, "6-20"), (50, "21-50")):
        if n <= hi:
            return label
    return ">50"


def question_flags(row: dict[str, Any], ocr_pages: dict[int, str]) -> dict[str, Any]:
    raw = row["evidence_page_no"]
    pages = sorted(set(raw if isinstance(raw, list) else [raw]))
    statuses = sorted({ocr_pages.get(p, "missing") for p in pages})
    return {
        "evidence_pages": pages,
        "list_valued_evidence": isinstance(raw, list),
        "multi_page_evidence": len(pages) > 1,
        "question_scripts": audit.scripts_present(str(row["questions"])),
        "evidence_page_ocr_status": "ok" if statuses == ["ok"] else ("+".join(s for s in statuses if s != "ok")),
    }


def build_documents(project_root: Path, inputs: dict[str, Any], pool: dict[str, Any], selected: list[str], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per selected document: PDF identity re-verified from the archive, and every PDF page with its OCR status."""
    manifest = inputs["manifest"]
    pdf_docs = manifest["ohr"]["coverage"]["pdf_per_document"]
    noisy = text_file_index(project_root, config["noisy_text_source"])
    reference = text_file_index(project_root, config["reference_text_source"])
    docs: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(inputs["archive_path"]) as archive:
        for cid in selected:
            raw = pool["raw_by_doc"][cid][0]
            rec = pdf_docs[raw]
            data = archive.read(rec["selected"]["member"])
            sha = hashlib.sha256(data).hexdigest()
            pages = audit.pdf_page_count(data)
            if sha != rec["pdf_sha256"] or pages != rec["pdf_pages"]:
                raise PilotError(f"{cid}: archive member differs from the audit record (sha {sha}, pages {pages})")
            noisy_path = noisy.get(cid)
            noisy_inv = audit.load_page_inventory(noisy_path) if noisy_path else {"pages": {}, "page_ids": [], "problems": ["file missing"]}
            ref_path = reference[cid]
            ref_inv = audit.load_page_inventory(ref_path)
            page_rows = []
            ocr_status = {}
            for idx in range(pages):
                if idx not in noisy_inv["pages"]:
                    status, text_sha = "missing", None
                else:
                    text = noisy_inv["pages"][idx]
                    status = "empty" if not text.strip() else "ok"
                    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                ocr_status[idx] = status
                page_rows.append({"page_idx": idx, "pdf_page_number": idx + 1, "ocr_status": status, "ocr_text_sha256": text_sha})
            docs[cid] = {
                "doc_id": cid,
                "qa_doc_names": pool["raw_by_doc"][cid],
                "doc_type": doc_type(cid),
                "excerpt_document": bool(audit.PAGE_SUFFIX_RE.match(cid.rsplit("/", 1)[-1])),
                "pdf": {"archive": inputs["lock"]["pdf_archive"]["local_path"], "member": rec["selected"]["member"], "sha256": sha, "bytes": len(data), "page_count": pages},
                "noisy_text": {
                    "source": config["noisy_text_source"],
                    "path": audit.display_path(noisy_path, project_root) if noisy_path else None,
                    "sha256": audit.sha256_file(noisy_path) if noisy_path else None,
                    "status": "present" if noisy_path else "missing",
                    "problems": noisy_inv["problems"],
                },
                "pages": page_rows,
                "_ocr_status": ocr_status,
                "_ocr_text": noisy_inv["pages"],
                "_ref_path": ref_path,
                "_ref_inv": ref_inv,
            }
    return docs


def build_manifests(project_root: Path, inputs: dict[str, Any], pool: dict[str, Any], docs: dict[str, dict[str, Any]], config: dict[str, Any]) -> tuple[dict, dict, list[dict[str, Any]]]:
    rows = sorted((r for r in inputs["qas"] if pool["canon"][str(r["doc_name"])] in docs), key=lambda r: str(r["ID"]))
    lock = inputs["lock"]
    runtime = {
        "schema_version": SCHEMA_VERSION,
        "kind": "runtime",
        "pilot_id": config["pilot_id"],
        "contract": "inputs a later system may read: question text, document identity, complete PDF page inventory and noisy-text references. No gold answers, evidence labels, evidence contexts, gt text or annotations.",
        "page_convention": "page_idx is 0-based and equals the PDF page index; pdf_page_number = page_idx + 1",
        "ocr_status_values": {"ok": "noisy text present and non-empty", "empty": "noisy text entry present but blank", "missing": "no noisy text entry for this PDF page"},
        "sources": {
            "pdf_archive": {"local_path": lock["pdf_archive"]["local_path"], "sha256": lock["pdf_archive"]["sha256"], "lock": config["pdf_source_lock"]},
            "noisy_text": {
                "source": config["noisy_text_source"],
                "root": f"{audit.RETRIEVAL_BASE_REL}/{config['noisy_text_source']}",
                "tree_digest": inputs["trees"][f"{audit.RETRIEVAL_BASE_REL}/{config['noisy_text_source']}"],
                "upstream_reference": config["pdf_source_lock"],
                "limitation": "MinerU engine version and settings are unknown; bytes match the upstream repository",
            },
        },
        "documents": [{k: v for k, v in docs[d].items() if not k.startswith("_")} for d in sorted(docs)],
        "questions": [{"question_id": str(r["ID"]), "question": r["questions"], "doc_id": pool["canon"][str(r["doc_name"])]} for r in rows],
    }
    evaluation_questions = {}
    flags_list = []
    for r in rows:
        cid = pool["canon"][str(r["doc_name"])]
        d = docs[cid]
        flags = question_flags(r, d["_ocr_status"])
        flags_list.append({"question_id": str(r["ID"]), "doc_id": cid, "evidence_source": r.get("evidence_source", ""), **flags})
        evaluation_questions[str(r["ID"])] = {
            "doc_id": cid,
            "answers": r["answers"],
            "answer_form": r.get("answer_form"),
            "evidence_page_no_raw": r["evidence_page_no"],
            "evidence_context": r.get("evidence_context"),
            "evidence_source": r.get("evidence_source"),
            "doc_type": r.get("doc_type"),
            "gt_reference": {"path": audit.display_path(d["_ref_path"], project_root), "sha256": audit.sha256_file(d["_ref_path"]), "page_ids": d["_ref_inv"]["page_ids"]},
            **flags,
        }
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "kind": "evaluation",
        "pilot_id": config["pilot_id"],
        "contract": "evaluation and inspection only; never an input to retrieval, gating, routing or recovery",
        "reference_text": {"source": config["reference_text_source"], "tree_digest": inputs["trees"][f"{audit.RETRIEVAL_BASE_REL}/{config['reference_text_source']}"]},
        "questions": evaluation_questions,
    }
    return runtime, evaluation, flags_list


def assert_runtime_boundary(runtime: dict[str, Any]) -> None:
    stack = [runtime]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            bad = RUNTIME_FORBIDDEN_KEYS & set(node)
            if bad:
                raise PilotError(f"runtime manifest contains forbidden keys {sorted(bad)}")
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, str) and "/retrieval_base/gt" in node:
            raise PilotError("runtime manifest references ground-truth text")


# ---------------------------------------------------------------- composition and families


def composition(doc_ids: list[str], doc_pages: dict[str, int], doc_ocr: dict[str, dict[str, int]], qflags: list[dict[str, Any]]) -> dict[str, Any]:
    ids = set(doc_ids)
    qs = [q for q in qflags if q["doc_id"] in ids]
    return {
        "documents": len(ids),
        "questions": len(qs),
        "pages": sum(doc_pages[d] for d in ids),
        "doc_type": dict(sorted(Counter(doc_type(d) for d in ids).items())),
        "page_count_bins": dict(sorted(Counter(page_bins(doc_pages[d]) for d in ids).items())),
        "documents_with_missing_ocr_pages": sum(1 for d in ids if doc_ocr[d]["missing"]),
        "documents_with_empty_ocr_pages": sum(1 for d in ids if doc_ocr[d]["empty"]),
        "ocr_pages": {k: sum(doc_ocr[d][k] for d in ids) for k in ("ok", "empty", "missing")},
        "evidence_source": dict(sorted(Counter(q["evidence_source"] for q in qs).items())),
        "evidence_page_ocr_status": dict(sorted(Counter(q["evidence_page_ocr_status"] for q in qs).items())),
        "question_scripts": dict(sorted(Counter("+".join(q["question_scripts"]) or "none" for q in qs).items())),
        "list_valued_evidence": sum(q["list_valued_evidence"] for q in qs),
        "multi_page_evidence": sum(q["multi_page_evidence"] for q in qs),
    }


def pool_profiles(project_root: Path, inputs: dict[str, Any], pool: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Page counts, OCR conditions and question flags for every eligible document (metadata only)."""
    pdf_docs = inputs["manifest"]["ohr"]["coverage"]["pdf_per_document"]
    noisy = text_file_index(project_root, config["noisy_text_source"])
    pages, ocr, qflags = {}, {}, []
    status_by_doc: dict[str, dict[int, str]] = {}
    for cid in pool["eligible"]:
        n = pdf_docs[pool["raw_by_doc"][cid][0]]["pdf_pages"]
        pages[cid] = n
        inv = audit.load_page_inventory(noisy[cid])["pages"] if cid in noisy else {}
        status = {i: ("missing" if i not in inv else ("empty" if not inv[i].strip() else "ok")) for i in range(n)}
        status_by_doc[cid] = status
        ocr[cid] = dict(Counter(status.values()))
        for k in ("ok", "empty", "missing"):
            ocr[cid].setdefault(k, 0)
    for r in inputs["qas"]:
        cid = pool["canon"][str(r["doc_name"])]
        if cid in status_by_doc:
            qflags.append({"question_id": str(r["ID"]), "doc_id": cid, "evidence_source": r.get("evidence_source", ""), **question_flags(r, status_by_doc[cid])})
    return pages, ocr, qflags


OMNIDOC_PAGE_RE = re.compile(r"^omnidocbench_(?P<collection>[a-z_]+?)_(?P<source>[0-9a-f]{32})_(?P<page>\d+)$")


def source_family(cid: str) -> tuple[str, str] | None:
    """(family key, rule) for explicit page-excerpt names; None when no rule applies."""
    head, _, name = cid.rpartition("/")
    m = audit.PAGE_SUFFIX_RE.match(name)
    if m:
        return f"{head}/{m['source']}", "pdf_page_suffix"
    m = OMNIDOC_PAGE_RE.match(name)
    if m:
        return f"{head}/omnidocbench_{m['collection']}_{m['source']}", "omnidocbench_hex_page"
    return None


def family_overlap(selected: list[str], pool: dict[str, Any]) -> dict[str, Any]:
    """Excerpt families shared within the pilot and with validation/test documents."""
    def family(cid: str) -> str | None:
        f = source_family(cid)
        return f[0] if f else None

    eval_docs = {d for d, s in pool["splits_by_doc"].items() if set(s) & {"val", "test"}}
    families: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"pilot": [], "val_or_test": []})
    for d in selected:
        f = family(d)
        if f:
            families[f]["pilot"].append(d)
    for d in eval_docs:
        f = family(d)
        if f in families:
            families[f]["val_or_test"].append(d)
    rows = [
        {"family": f, "rule": source_family(v["pilot"][0])[1], "pilot_documents": sorted(v["pilot"]), "val_or_test_documents": sorted(v["val_or_test"])}
        for f, v in sorted(families.items())
    ]
    return {
        "rules": {
            "pdf_page_suffix": "'<publication>.pdf_<page>': page excerpt of one publication; family strips '_<page>'",
            "omnidocbench_hex_page": "'omnidocbench_<collection>_<32-hex id>_<page>': page of one source; family is collection + id",
        },
        "confidence": "uncertain: inferred from naming conventions, not upstream metadata; no other grouping (e.g. company filings, GNHK writers) is attempted",
        "pilot_documents_in_families": sum(len(r["pilot_documents"]) for r in rows),
        "families_with_several_pilot_documents": [r for r in rows if len(r["pilot_documents"]) > 1],
        "families_shared_with_val_or_test": [r for r in rows if r["val_or_test_documents"]],
        "families": rows,
    }


# ---------------------------------------------------------------- inspection selection and rendering


def inspection_categories(q: dict[str, Any]) -> list[str]:
    cats = []
    if q["evidence_page_ocr_status"] != "ok":
        cats.append("ocr_gap")
    if q["multi_page_evidence"]:
        cats.append("multi_page_evidence")
    if q["evidence_source"] == "multi":
        cats.append("multi_evidence_source")
    if "han" in q["question_scripts"]:
        cats.append("han_script")
    if q["evidence_source"] in ("table", "formula", "chart", "reading_order", "text"):
        cats.append(q["evidence_source"])
    return cats


def select_inspection(qflags: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Fill category quotas in order from hash-ranked questions, then top up to the case target.

    A candidate whose new pages would exceed the remaining page budget is skipped
    and the next-ranked candidate is tried. The pilot itself is never changed.
    """
    spec = config["inspection"]
    seed = config["selection"]["seed"]
    ranked = sorted(qflags, key=lambda q: (rank_key(seed, "inspect", q["question_id"]), q["question_id"]))
    chosen: list[dict[str, Any]] = []
    pages: set[tuple[str, int]] = set()
    skipped_for_budget: list[str] = []
    report = []

    def try_add(q: dict[str, Any], reason: str) -> bool:
        need = {(q["doc_id"], p) for p in q["evidence_pages"]} - pages
        if len(pages) + len(need) > spec["page_budget"]:
            skipped_for_budget.append(q["question_id"])
            return False
        pages.update(need)
        chosen.append({**q, "categories": inspection_categories(q), "selected_for": reason})
        return True

    for category, quota in spec["category_quotas"]:
        available = [q for q in ranked if category in inspection_categories(q)]
        taken = sum(1 for c in chosen if category in c["categories"])
        for q in available:
            if taken >= quota or len(chosen) >= spec["cases"]:
                break
            if any(c["question_id"] == q["question_id"] for c in chosen):
                continue
            if try_add(q, category):
                taken += 1
        report.append({"category": category, "quota": quota, "available_in_pilot": len(available), "cases_covering": sum(1 for c in chosen if category in c["categories"])})
    for q in ranked:
        if len(chosen) >= spec["cases"]:
            break
        if not any(c["question_id"] == q["question_id"] for c in chosen):
            try_add(q, "top_up")
    return {
        "algorithm": INSPECTION_ALGORITHM,
        "procedure": "questions ranked by sha256(seed, 'inspect', question_id); category quotas filled in the configured order (a case counts toward every category it has); remaining slots topped up in rank order; a case whose new evidence pages exceed the page budget is skipped for the next-ranked one",
        "cases": chosen,
        "category_report": report,
        "unavailable_categories": [r["category"] for r in report if r["available_in_pilot"] == 0],
        "skipped_for_page_budget": sorted(set(skipped_for_budget)),
        "unique_pages": sorted([d, p] for d, p in pages),
        "warning": "purposive inspection sample; its category proportions are not prevalence estimates",
    }


def render_pages(archive_path: Path, docs: dict[str, dict[str, Any]], pages: list[list[Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    import PIL  # noqa: PLC0415
    import pypdfium2  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    by_doc: dict[str, list[int]] = defaultdict(list)
    for d, p in pages:
        by_doc[d].append(p)
    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        for d in sorted(by_doc):
            member = docs[d]["pdf"]["member"]
            data = archive.read(member)
            if hashlib.sha256(data).hexdigest() != docs[d]["pdf"]["sha256"]:
                raise PilotError(f"{d}: archive member hash changed before rendering")
            pdf = pypdfium2.PdfDocument(data)
            try:
                for p in sorted(set(by_doc[d])):
                    t0 = time.perf_counter()
                    image = pdf[p].render(scale=dpi / 72).to_pil()
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    png = buf.getvalue()
                    name = f"{hashlib.sha256(d.encode('utf-8')).hexdigest()[:12]}_p{p}.png"
                    (out_dir / name).write_bytes(png)
                    rows.append({"doc_id": d, "page_idx": p, "file": f"pages/{name}", "width": image.width, "height": image.height, "bytes": len(png), "png_sha256": hashlib.sha256(png).hexdigest(), "source_member": member, "source_sha256": docs[d]["pdf"]["sha256"], "render_sec": round(time.perf_counter() - t0, 3)})
            finally:
                pdf.close()
    elapsed = time.perf_counter() - started
    sizes = sorted(r["bytes"] for r in rows)
    return {
        "renderer": {"package": "pypdfium2", "version": pypdfium2.V_PYPDFIUM2, "pdfium_build": str(pypdfium2.V_LIBPDFIUM), "pillow": PIL.__version__},
        "dpi": dpi,
        "dpi_note": "150 DPI is an inspection choice, not a frozen model-input setting",
        "image_format": "png",
        "crops": [],
        "pages": rows,
        "unique_pages": len(rows),
        "total_image_bytes": sum(sizes),
        "per_page_bytes": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]} if sizes else {},
        "elapsed_sec_including_archive_reads": round(elapsed, 3),
        "note": "measured on a small purposive sample; not an estimate of full-dataset storage or compute",
    }


# ---------------------------------------------------------------- HTML packet


def build_html(config: dict[str, Any], docs: dict[str, dict[str, Any]], evaluation: dict[str, Any], runtime_q: dict[str, dict[str, Any]], inspection: dict[str, Any], render: dict[str, Any]) -> str:
    e = html.escape
    images = {(r["doc_id"], r["page_idx"]): r for r in render["pages"]}
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{e(config['pilot_id'])} inspection packet</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;max-width:1400px;color:#111;background:#fff}"
        ".warn{border:2px solid #b45309;background:#fff7ed;padding:10px;margin:12px 0}"
        ".case{border-top:3px solid #333;margin-top:28px;padding-top:8px}.page{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;margin:12px 0}"
        "img{max-width:100%;border:1px solid #999}pre{white-space:pre-wrap;word-break:break-word;background:#f4f4f4;padding:8px;max-height:700px;overflow:auto;font-size:13px}"
        ".state{font-weight:bold;color:#9a3412}details{margin:8px 0}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "@media(max-width:800px){.page{grid-template-columns:1fr}}</style></head><body>",
        f"<h1>{e(config['pilot_id'])}: human inspection packet</h1>",
        "<div class='warn'><strong>Inspection only.</strong> Pages shown here were chosen using gold evidence labels so a person can look at the evidence. "
        "This does not demonstrate retrieval performance and must never become a runtime evidence shortcut. "
        "The cases are a purposive sample; their category proportions are not prevalence estimates. "
        "No labels are pre-filled, no retrieval has run, and an OCR gap is not evidence of its cause.</div>",
        "<p>Record observations in <code>annotations.csv</code> (one row per case and page). Categories: "
        + ", ".join(e(c) for c in OBSERVATION_CATEGORIES)
        + ". Several may apply; add an uncertainty note.</p>",
        f"<p>Page convention: <code>page_idx</code> is 0-based and equals the PDF page index; the PDF page number shown is page_idx + 1. "
        f"Rendering: {e(render['renderer']['package'])} {e(render['renderer']['version'])}, {render['dpi']} DPI, PNG ({e(render['dpi_note'])}).</p>",
        "<h2>Cases</h2><table><tr><th>Case</th><th>Question</th><th>Document</th><th>Categories</th></tr>",
    ]
    for i, c in enumerate(inspection["cases"], 1):
        parts.append(f"<tr><td><a href='#case-{i}'>{i}</a></td><td>{e(c['question_id'])}</td><td>{e(c['doc_id'])}</td><td>{e(', '.join(c['categories']))}</td></tr>")
    parts.append("</table>")
    for i, c in enumerate(inspection["cases"], 1):
        qid = c["question_id"]
        d = docs[c["doc_id"]]
        ev = evaluation["questions"][qid]
        parts += [
            f"<section class='case' id='case-{i}'><h2>Case {i}: {e(qid)}</h2>",
            f"<p><strong>Question:</strong> {e(runtime_q[qid]['question'])}</p>",
            f"<p><strong>Document:</strong> {e(d['doc_id'])} ({d['pdf']['page_count']} pages; member <code>{e(d['pdf']['member'])}</code>, sha256 <code>{e(d['pdf']['sha256'][:16])}…</code>)</p>",
            f"<p><strong>Categories:</strong> {e(', '.join(c['categories']))} (selected for: {e(c['selected_for'])}). Evidence pages shown (0-based page_idx): {e(str(c['evidence_pages']))}</p>",
        ]
        for p in c["evidence_pages"]:
            img = images.get((d["doc_id"], p))
            status = d["_ocr_status"].get(p, "missing")
            if status == "missing":
                ocr = "<p class='state'>MinerU: no text entry for this page (missing).</p>"
            elif status == "empty":
                ocr = "<p class='state'>MinerU: entry present but empty.</p>"
            else:
                ocr = f"<pre>{e(d['_ocr_text'][p])}</pre>"
            img_html = f"<img src='{e(img['file'])}' alt='{e(d['doc_id'])} page_idx {p}' width='{img['width']}' height='{img['height']}' loading='lazy'>" if img else "<p class='state'>Image not rendered (page budget).</p>"
            gt_text = d["_ref_inv"]["pages"].get(p)
            parts += [
                f"<h3>page_idx {p} (PDF page {p + 1}) — OCR status: {e(status)}</h3>",
                f"<div class='page'><div>{img_html}</div><div><h4>MinerU text (noisy input)</h4>{ocr}</div></div>",
                "<details><summary>Optional ground-truth reference for this page (evaluation only)</summary>"
                + (f"<pre>{e(gt_text)}</pre>" if gt_text is not None else "<p class='state'>No gt entry.</p>")
                + "</details>",
            ]
        parts += [
            "<details><summary>Optional gold labels (evaluation only)</summary>",
            f"<p><strong>Answer:</strong> {e(str(ev['answers']))} ({e(str(ev['answer_form']))}); evidence source: {e(str(ev['evidence_source']))}</p>",
            f"<pre>{e(str(ev['evidence_context']))}</pre></details>",
            "<p><strong>Observations</strong> (fill in annotations.csv): "
            + " · ".join(f"☐ {e(cat)}" for cat in OBSERVATION_CATEGORIES)
            + " · uncertainty note: ________</p></section>",
        ]
    parts.append("</body></html>")
    return "\n".join(parts)


def write_annotations(path: Path, inspection: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "question_id", "doc_id", "page_idx", *OBSERVATION_CATEGORIES, "uncertainty_note", "notes", "annotator"])
        for i, c in enumerate(inspection["cases"], 1):
            for p in c["evidence_pages"]:
                writer.writerow([i, c["question_id"], c["doc_id"], p, *[""] * len(OBSERVATION_CATEGORIES), "", "", ""])


# ---------------------------------------------------------------- orchestration


def build(project_root: Path, config_path: Path, *, render: bool = True) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("selection", {}).get("algorithm") != SELECTION_ALGORITHM:
        raise PilotError(f"config selection algorithm must be {SELECTION_ALGORITHM}")
    out = project_root / config["output_dir"]
    inputs = load_inputs(project_root, config)
    pool = build_pool(inputs)
    selected, allocation, ranks = select_documents(pool["eligible"], config)
    docs = build_documents(project_root, inputs, pool, selected, config)
    runtime, evaluation, qflags = build_manifests(project_root, inputs, pool, docs, config)
    assert_runtime_boundary(runtime)
    for q in runtime["questions"]:
        if pool["split_of"].get(q["question_id"]) != "train":
            raise PilotError(f"question {q['question_id']} is not in train")

    pages, ocr, pool_q = pool_profiles(project_root, inputs, pool, config)
    selection = {
        "schema_version": SCHEMA_VERSION,
        "pilot_id": config["pilot_id"],
        "config": config,
        "config_sha256": digest(config),
        "algorithm": {
            "name": SELECTION_ALGORITHM,
            "identity_rule": "canonical doc id = NFC of the gt inventory key the QA doc_name resolves to (exact, documented alias, or Unicode NFC), as recorded by the asset audit",
            "pool_rule": "documents all of whose questions are in the train split",
            "eligibility_rule": "one raw QA name, exactly one source PDF candidate with audit status verified_complete, and a gt reference; missing or empty OCR never excludes",
            "allocation": "per-stratum minimum of 1 where feasible, then largest remainder against proportional quotas; ties -> larger pool, then stratum name",
            "within_stratum": "ascending sha256(seed, 'select', doc_id); independent of input order",
            "inputs_used": "document identity, doc type and source status only; never answers, evidence strings, string-match results or model outputs",
        },
        "seed": config["selection"]["seed"],
        "pool": {
            "train_exclusive_documents": len(pool["train_exclusive"]),
            "eligible_documents": len(pool["eligible"]),
            "eligible_sha256": digest(pool["eligible"]),
            "source_exclusions": pool["exclusions"],
        },
        "allocation": allocation,
        "selected_documents": [{"doc_id": d, "rank_key": ranks[d], "doc_type": doc_type(d)} for d in selected],
        "selected_question_ids": [q["question_id"] for q in runtime["questions"]],
        "composition": {
            "eligible_pool": composition(pool["eligible"], pages, ocr, pool_q),
            "pilot": composition(selected, pages, ocr, pool_q),
            "note": "sample descriptions, not accuracy results",
        },
        "source_family_overlap": family_overlap(selected, pool),
        "sources": {
            "qa_sha256": inputs["hashes"][config["qa_source"]],
            "split_sha256": inputs["hashes"][config["split"]],
            "pdf_archive_sha256": inputs["hashes"][inputs["lock"]["pdf_archive"]["local_path"]],
            "pdf_source_lock": config["pdf_source_lock"],
            "pdf_source_lock_sha256": inputs["lock_sha256"],
            "text_tree_digests": inputs["trees"],
            "audit_manifest": config["audit_manifest"],
            "audit_content_sha256": inputs["manifest"]["content_sha256"],
        },
        "manifest_digests": {"runtime": digest(runtime), "evaluation": digest(evaluation)},
    }
    selection["selection_digest"] = digest(selection)
    record_path = out / "selection_record.json"
    status = "created"
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("selection_digest") != selection["selection_digest"]:
            raise PilotError(
                f"{record_path} holds a different frozen pilot (digest {existing.get('selection_digest')}, new {selection['selection_digest']}). "
                "Use a new pilot_id and output_dir instead of replacing it.",
                EXIT_FROZEN_MISMATCH,
            )
        for name, key in (("runtime_manifest.json", "runtime"), ("evaluation_manifest.json", "evaluation")):
            on_disk = json.loads((out / name).read_text(encoding="utf-8"))
            if digest(on_disk) != existing["manifest_digests"][key]:
                raise PilotError(f"{out / name} was modified after freezing", EXIT_FROZEN_MISMATCH)
        status = "reproduced"
    else:
        write_json(out / "runtime_manifest.json", runtime)
        write_json(out / "evaluation_manifest.json", evaluation)
        selection_out = dict(selection)
        selection_out["volatile"] = {"generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), "repository": audit.git_info(project_root)}
        write_json(record_path, selection_out)

    result = {"status": status, "selection_digest": selection["selection_digest"], "documents": len(selected), "questions": len(runtime["questions"]), "pages": sum(d["pdf"]["page_count"] for d in docs.values())}
    if render:
        inspection = select_inspection(qflags, config)
        idir = out / "inspection"
        cases_path = idir / "inspection_cases.json"
        if cases_path.exists():
            previous = json.loads(cases_path.read_text(encoding="utf-8"))
            if previous.get("inspection_digest") != digest(inspection):
                raise PilotError(f"{cases_path} holds a different inspection sample; use a new pilot_id/output_dir", EXIT_FROZEN_MISMATCH)
        record = render_pages(inputs["archive_path"], docs, inspection["unique_pages"], idir / "pages", config["inspection"]["dpi"])
        rq = {q["question_id"]: q for q in runtime["questions"]}
        (idir / "index.html").write_text(build_html(config, docs, evaluation, rq, inspection, record), encoding="utf-8")
        if not (idir / "annotations.csv").exists():  # never overwrite human observations
            write_annotations(idir / "annotations.csv", inspection)
        write_json(idir / "inspection_cases.json", {"schema_version": SCHEMA_VERSION, "pilot_id": config["pilot_id"], "selection_digest": selection["selection_digest"], **{k: v for k, v in inspection.items()}, "inspection_digest": digest(inspection)})
        write_json(idir / "render_record.json", {"schema_version": SCHEMA_VERSION, "selection_digest": selection["selection_digest"], **record, "volatile": ["elapsed_sec_including_archive_reads", "pages[].render_sec"]})
        result.update({"inspection_cases": len(inspection["cases"]), "rendered_pages": record["unique_pages"], "image_bytes": record["total_image_bytes"], "render_sec": record["elapsed_sec_including_archive_reads"]})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a frozen OHR development pilot and inspection packet (no models, no OCR, no network).")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true", help="Freeze or validate the selection without rendering the inspection packet.")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        result = build(root, config, render=not args.no_render)
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
