"""DIAGNOSTIC that reads evaluation data: search the prepared prompts for gold strings and reference passages.

This script opens ``evaluation_manifest.json`` (gold answers, evidence contexts, evidence pages) and the clean
reference text. It exists to check, after the fact, that runtime preparation did not let evaluation data into a
prompt. Its findings must never feed runtime preparation, retrieval, gating, diagnosis or recovery, and its output
must never be read by the code that prepares or sends requests. Run it only when you want this diagnostic. The
runtime-only audit is ``audit_prompts.py``.

For every sent request the script separates the prompt into the evidence text and everything else (the system
message, the template text, the block headers and fences, and the question), then searches:

    gold answer         a gold string of 4 or more characters, after whitespace collapsing and casefolding,
                        inside the text outside the evidence; the hit is traced to the question, or to the template
                        and headers, which would be a leak
    gold in evidence    the same string inside the evidence blocks, with the ranks. A hit there is OCR text of the
                        document, not a leak, because every chunk text is a substring of its MinerU page
    evidence context    the evidence_context field (20 or more characters) outside the evidence
    reference passage   any 40-character window of the prompt text outside the evidence and the question that
                        occurs in the clean reference text of the document
    field names         evaluation field names in the prompt
    page coverage       whether the prompt holds a chunk from each gold evidence page. This is not evidence coverage:
                        a gold page among the retrieved pages does not show that the needed passage was retrieved.

The output holds question ids, counts and ranks. It holds no gold answer, question or document text.

Usage:
    python scripts/audits/audit_prompt_leakage_diagnostic.py --dry-run-dir .local/work/dry-run --out .local/work/audits
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAILED_CHECK,
    REPO_ROOT,
    norm_ws,
    prepare_out,
    read_json,
    read_jsonl,
    require_dir,
    require_file,
    resolve,
    write_json,
)
from audit_prompts import parse_user_message  # noqa: E402

BANNER = (
    "DIAGNOSTIC that reads evaluation data. Never feed its output into runtime preparation, retrieval, gating, "
    "diagnosis or recovery."
)
DEFAULT_DRY_RUN = ".local/work/dry-run"
DEFAULT_OUT = ".local/work/audits"
DEFAULT_EVALUATION = "results/pilots/ohr_dev_v1/evaluation_manifest.json"
FIELD_NAMES = (
    "answer_form",
    "evidence_context",
    "evidence_source",
    "question_scripts",
    "multi_page_evidence",
    "list_valued_evidence",
    "evidence_page_ocr_status",
    "gt_reference",
    "reference_text",
    "annotations",
    "evidence_pages",
    "answers",
)
MIN_GOLD_CHARS = 4
MIN_CONTEXT_CHARS = 20
WINDOW = 40
STEP = 20


def _fold(text: str) -> str:
    return norm_ws(text).casefold()


def load_reference_text(evaluation: dict[str, Any], project_root: Path, doc_ids: set[str]) -> dict[str, str]:
    """Read the clean reference text of each document that has a sent question. Fail when a file is missing."""
    paths: dict[str, str] = {}
    for entry in evaluation["questions"].values():
        if entry["doc_id"] in doc_ids:
            paths.setdefault(entry["doc_id"], entry["gt_reference"]["path"])
    texts: dict[str, str] = {}
    for doc_id, rel in paths.items():
        path = require_file(project_root / rel, "clean reference text", "or pass --skip-reference")
        rows = read_json(path, "clean reference text")
        texts[doc_id] = _fold(" ".join(r.get("text", "") if isinstance(r, dict) else str(r) for r in rows))
    return texts


def outside_evidence_text(record: dict[str, Any], message: dict[str, Any]) -> tuple[str, str]:
    """Split a request into the text outside the evidence (question removed) and the question text."""
    user = record["user"]
    q_start, q_end = message["question_span"]
    cut = sorted([(q_start, q_end)] + [(s, e) for kind, s, e in message["spans"] if kind == "evidence"])
    pieces = [record["system"]]
    cursor = 0
    for start, end in cut:
        pieces.append(user[cursor:start])
        cursor = end
    pieces.append(user[cursor:])
    return "\n".join(pieces), user[q_start:q_end]


def diagnose(
    records: Sequence[dict[str, Any]],
    evaluation: dict[str, Any],
    reference: dict[str, str] | None,
) -> dict[str, Any]:
    found: dict[str, list[Any]] = defaultdict(list)
    short_gold: list[str] = []
    page_rows: list[dict[str, Any]] = []
    for record in records:
        if record["action"] != "send":
            continue
        qid = record["question_id"]
        entry = evaluation["questions"].get(qid)
        if entry is None:
            found["question_not_in_evaluation_manifest"].append(qid)
            continue
        message = parse_user_message(record["user"], record["evidence"])
        if message is None or message["problems"]:
            found["prompt_does_not_parse"].append(qid)
            continue
        outside, question = outside_evidence_text(record, message)
        outside_folded = _fold(outside)
        question_folded = _fold(question)
        evidence_folded = [_fold(e["text"]) for e in record["evidence"]]

        gold = _fold(str(entry["answers"])).rstrip(".")
        if len(gold) < MIN_GOLD_CHARS:
            short_gold.append(qid)
        else:
            if gold in outside_folded:
                found["gold_outside_evidence_and_question"].append(qid)
            if gold in question_folded:
                found["gold_inside_question_text"].append(qid)
            ranks = [i for i, text in enumerate(evidence_folded, start=1) if gold in text]
            if ranks:
                found["gold_inside_evidence"].append((qid, ranks))
        contexts = entry["evidence_context"] if isinstance(entry["evidence_context"], list) else [entry["evidence_context"]]
        if any(len(_fold(c)) >= MIN_CONTEXT_CHARS and _fold(c) in outside_folded for c in contexts):
            found["evidence_context_outside_evidence"].append(qid)
        for name in FIELD_NAMES:
            if name in outside or name in question:
                found["evaluation_field_name_in_prompt"].append((qid, name))
        if reference is not None:
            text = reference.get(record["doc_id"], "")
            windows = (outside_folded[i : i + WINDOW] for i in range(0, max(0, len(outside_folded) - WINDOW), STEP))
            if any(w and w in text for w in windows):
                found["reference_passage_outside_evidence_and_question"].append(qid)

        got_pages = sorted({e["page_idx"] for e in record["evidence"]})
        page_rows.append(
            {
                "question_id": qid,
                "all_gold_pages_in_prompt": all(p in got_pages for p in entry["evidence_pages"]),
                "any_gold_page_in_prompt": any(p in got_pages for p in entry["evidence_pages"]),
            }
        )
    hard_keys = (
        "gold_outside_evidence_and_question",
        "evidence_context_outside_evidence",
        "evaluation_field_name_in_prompt",
        "reference_passage_outside_evidence_and_question",
        "question_not_in_evaluation_manifest",
        "prompt_does_not_parse",
    )
    failures = {k: found[k] for k in hard_keys if found.get(k)}
    sent = len(page_rows)
    return {
        "pass": not failures,
        "failures": failures,
        "sent_questions": sent,
        "gold_shorter_than_4_characters_not_searched": short_gold,
        "gold_inside_question_text": found.get("gold_inside_question_text", []),
        "gold_inside_evidence_questions": len(found.get("gold_inside_evidence", [])),
        "gold_inside_evidence": [[qid, ranks] for qid, ranks in found.get("gold_inside_evidence", [])],
        "gold_inside_evidence_note": "OCR text of the document; every chunk is a substring of its MinerU page",
        "reference_searched": reference is not None,
        "page_coverage": {
            "all_gold_pages_in_prompt": sum(1 for r in page_rows if r["all_gold_pages_in_prompt"]),
            "not_all_gold_pages_in_prompt": [r["question_id"] for r in page_rows if not r["all_gold_pages_in_prompt"]],
            "no_gold_page_in_prompt": [r["question_id"] for r in page_rows if not r["any_gold_page_in_prompt"]],
            "note": "page coverage overstates evidence coverage: a gold page among the retrieved pages does not "
            "show that the needed passage was retrieved",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=BANNER)
    parser.add_argument("--dry-run-dir", default=DEFAULT_DRY_RUN, help=f"dry-run directory (default: {DEFAULT_DRY_RUN})")
    parser.add_argument(
        "--evaluation-manifest", default=DEFAULT_EVALUATION, help=f"evaluation manifest (default: {DEFAULT_EVALUATION})"
    )
    parser.add_argument(
        "--project-root", default=str(REPO_ROOT), help="checkout that holds OHR-Bench/ (default: this repository)"
    )
    parser.add_argument("--skip-reference", action="store_true", help="do not search the clean reference text")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)

    print(BANNER)
    project_root = resolve(args.project_root)
    dry_run = require_dir(resolve(args.dry_run_dir), "dry-run directory", "run_pilot_live.py dry-run writes it")
    records = read_jsonl(dry_run / "prepared_requests.jsonl", "prepared_requests.jsonl")
    evaluation = read_json(resolve(args.evaluation_manifest, project_root), "evaluation manifest")
    reference = None
    if not args.skip_reference:
        sent_docs = {r["doc_id"] for r in records if r["action"] == "send"}
        reference = load_reference_text(evaluation, project_root, sent_docs)
    out = prepare_out(resolve(args.out))

    result = diagnose(records, evaluation, reference)
    write_json(out / "audit_prompt_leakage_diagnostic.json", {"notice": BANNER, **result})

    print(f"sent questions {result['sent_questions']}")
    print(f"{'PASS' if result['pass'] else 'FAIL'}  no gold string, evidence context, reference passage or field name "
          f"outside the evidence and the question")
    for key, value in result["failures"].items():
        print(f"      {key}: {value}")
    print(f"gold strings inside evidence blocks (OCR text): {result['gold_inside_evidence_questions']} questions")
    print(f"gold strings inside question text: {len(result['gold_inside_question_text'])}")
    print(f"gold shorter than {MIN_GOLD_CHARS} characters, not searched: {len(result['gold_shorter_than_4_characters_not_searched'])}")
    cov = result["page_coverage"]
    print(f"prompts holding every gold page: {cov['all_gold_pages_in_prompt']} of {result['sent_questions']}; "
          f"missing a gold page: {[q[:8] for q in cov['not_all_gold_pages_in_prompt']]}")
    return 0 if result["pass"] else EXIT_FAILED_CHECK


if __name__ == "__main__":
    raise SystemExit(main())
