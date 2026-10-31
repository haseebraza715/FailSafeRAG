"""Shared inputs for the ohr_dev_v1 failure analysis of the gpt-6.1-terra baseline.

Everything here reads files that already exist. Nothing calls a model. The
join puts runtime records (question, retrieved evidence, reply) next to
evaluation-only records (gold answer, evidence pages, clean text) so that a
person can read a case. Nothing built here feeds any runtime component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = ROOT / "results" / "pilots" / "ohr_dev_v1"
BASELINE_RUN = "2026-10-31-ohr-dev-v1-baseline-terra-r1"
BASELINE_DIR = ROOT / "results" / "development" / BASELINE_RUN
OUT_DIR = ROOT / "results" / "analysis" / "2026-10-31-ohr-dev-v1-terra-failures"
PDF_ARCHIVE = ROOT / "data" / "ohr_bench_raw" / "pdfs.zip"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def page_texts(path: Path) -> dict[int, str]:
    """Page index to text, joining repeated indexes with a newline, as the pilot loader does."""
    pages: dict[int, str] = {}
    for row in read_json(path):
        if not isinstance(row, dict) or not isinstance(row.get("page_idx"), int):
            continue
        text = row.get("text") if isinstance(row.get("text"), str) else ""
        idx = row["page_idx"]
        pages[idx] = pages[idx] + "\n" + text if idx in pages else text
    return pages


def load_cases() -> list[dict[str, Any]]:
    """One record per pilot question, in runtime-manifest order, joining run and evaluation files."""
    runtime = read_json(PILOT_DIR / "runtime_manifest.json")
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    docs = {d["doc_id"]: d for d in runtime["documents"]}
    preds = {p["question_id"]: p for p in read_jsonl(BASELINE_DIR / "predictions.jsonl")}
    scores = {s["question_id"]: s for s in read_jsonl(BASELINE_DIR / "scores.jsonl")}
    requests = {r["question_id"]: r for r in read_jsonl(BASELINE_DIR / "requests.jsonl")}
    cases = []
    for q in runtime["questions"]:
        qid = q["question_id"]
        ev = evaluation[qid]
        doc = docs[q["doc_id"]]
        pred, score, req = preds[qid], scores[qid], requests[qid]
        noisy = page_texts(ROOT / doc["noisy_text"]["path"]) if doc["noisy_text"]["status"] == "present" else {}
        clean = page_texts(ROOT / ev["gt_reference"]["path"])
        response = None
        if pred["attempt_ids"]:
            response = read_json(BASELINE_DIR / "responses" / f"{pred['attempt_ids'][-1]}.json")
        cases.append(
            {
                "question_id": qid,
                "doc_id": q["doc_id"],
                "doc_type": ev["doc_type"],
                "question": q["question"],
                "gold": ev["answers"],
                "answer_form": ev["answer_form"],
                "evidence_source": ev["evidence_source"],
                "evidence_context": ev["evidence_context"],
                "evidence_pages": ev["evidence_pages"],
                "evidence_page_ocr_status": ev["evidence_page_ocr_status"],
                "question_scripts": ev["question_scripts"],
                "multi_page_evidence": ev["multi_page_evidence"],
                "page_count": doc["pdf"]["page_count"],
                "pdf_member": doc["pdf"]["member"],
                "pdf_sha256": doc["pdf"]["sha256"],
                "status": pred["status"],
                "answer": pred["answer"],
                "abstained": pred["abstained"],
                "no_evidence_reason": pred["no_evidence_reason"],
                "retrieved": [{"rank": e["rank"], "chunk_id": e["chunk_id"], "page_idx": e["page_idx"]} for e in pred["evidence"]],
                "user_prompt": req.get("user"),
                "raw_reply": None if response is None else response["provider_response"]["text"],
                "em": score["em"],
                "f1": score["f1"],
                "noisy_pages": noisy,
                "clean_pages": clean,
            }
        )
    return cases
