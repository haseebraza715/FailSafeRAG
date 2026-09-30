"""Count the categories in the structured agent review of the 20 inspection cases.

The input is ``docs/reports/ohr-dev-v1-agent-review-2026-09-30.cases.json``. It is an agent review. It is not
human annotation and not ground truth, and its counts describe a purposive sample of 20 cases, not the dataset.

The script prints the number of sent cases, how many of the sent cases hold the answer in the retrieved evidence
(yes, partly, no), how OCR fared, which gold answers the agents doubted, and the likely origin of a miss.
With ``--check-markdown`` it also compares every row with the summary table of the markdown report and exits 1
on any difference, so the JSON and the table cannot drift apart.

Usage:
    python scripts/audits/case_review_counts.py [--cases PATH] [--check-markdown PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXIT_FAILED_CHECK, fail, prepare_out, read_json, require_file, resolve, write_json

DEFAULT_CASES = "docs/reports/ohr-dev-v1-agent-review-2026-09-30.cases.json"
ALLOWED = {
    "ocr_preserves": {"survives", "damaged", "lost", "uncertain"},
    "retrieval_contains": {"yes", "partly", "no", "not_sent"},
    "gold_questionable": {"yes", "no", "uncertain"},
}
REQUIRED = (
    "case",
    "question_id",
    "sent",
    "ocr_preserves",
    "retrieval_contains",
    "likely_origin",
    "gold_questionable",
    "condition",
)


def validate(review: dict[str, Any]) -> list[str]:
    """Return the problems in the structure of the review. An empty list means the structure is sound."""
    problems: list[str] = []
    if review.get("record_type") != "AGENT REVIEW":
        problems.append("record_type is not 'AGENT REVIEW'")
    if review.get("not_human_annotation") is not True or review.get("not_ground_truth") is not True:
        problems.append("the header must say not_human_annotation and not_ground_truth are true")
    cases = review.get("cases")
    if not isinstance(cases, list) or not cases:
        return [*problems, "cases is missing or empty"]
    numbers = [c.get("case") for c in cases]
    if numbers != list(range(1, len(cases) + 1)):
        problems.append(f"case numbers are not 1..{len(cases)} in order: {numbers}")
    for c in cases:
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            problems.append(f"case {c.get('case')}: missing {missing}")
            continue
        for field, allowed in ALLOWED.items():
            if c[field] not in allowed:
                problems.append(f"case {c['case']}: {field}={c[field]!r} is not one of {sorted(allowed)}")
        if not isinstance(c["sent"], bool):
            problems.append(f"case {c['case']}: sent must be true or false")
        elif c["sent"] == (c["retrieval_contains"] == "not_sent"):
            problems.append(f"case {c['case']}: sent={c['sent']} contradicts retrieval_contains={c['retrieval_contains']!r}")
    return problems


def count(review: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the categories. Raises ``ValueError`` when the structure is unsound."""
    problems = validate(review)
    if problems:
        raise ValueError("; ".join(problems))
    cases = review["cases"]
    sent = [c for c in cases if c["sent"]]

    def tally(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(r[field] for r in rows).items()))

    def which(rows: Sequence[dict[str, Any]], field: str, value: str) -> list[int]:
        return [r["case"] for r in rows if r[field] == value]

    return {
        "record_type": review["record_type"],
        "cases": len(cases),
        "sent": len(sent),
        "not_sent": len(cases) - len(sent),
        "not_sent_cases": [c["case"] for c in cases if not c["sent"]],
        "retrieval_contains_among_sent": tally(sent, "retrieval_contains"),
        "retrieval_partly_cases": which(sent, "retrieval_contains", "partly"),
        "retrieval_no_cases": which(sent, "retrieval_contains", "no"),
        "ocr_preserves_all": tally(cases, "ocr_preserves"),
        "ocr_preserves_among_sent": tally(sent, "ocr_preserves"),
        "gold_questionable_all": tally(cases, "gold_questionable"),
        "gold_questionable_yes_cases": which(cases, "gold_questionable", "yes"),
        "gold_questionable_uncertain_cases": which(cases, "gold_questionable", "uncertain"),
        "gold_questionable_yes_among_sent": which(sent, "gold_questionable", "yes"),
        "likely_origin_all": tally(cases, "likely_origin"),
        "likely_origin_among_sent": tally(sent, "likely_origin"),
    }


_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.+)\|\s*$")


def parse_markdown_table(text: str) -> dict[int, dict[str, str]]:
    """Read the summary table of the markdown report: case number to its cells (backticks removed)."""
    rows: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        match = _ROW.match(line)
        if not match:
            continue
        cells = [cell.strip().replace("`", "") for cell in match.group(2).split("|")]
        if len(cells) != 7:
            continue
        keys = ("question_id", "condition", "sent", "ocr", "holds", "origin", "gold")
        rows[int(match.group(1))] = dict(zip(keys, cells, strict=True))
    return rows


def compare_with_markdown(review: dict[str, Any], text: str) -> list[str]:
    """Compare every case with its row of the markdown summary table."""
    table = parse_markdown_table(text)
    problems: list[str] = []
    if sorted(table) != [c["case"] for c in review["cases"]]:
        problems.append(f"the table lists cases {sorted(table)} and the JSON lists {[c['case'] for c in review['cases']]}")
    for c in review["cases"]:
        row = table.get(c["case"])
        if row is None:
            continue
        n = c["case"]
        if not c["question_id"].startswith(row["question_id"]):
            problems.append(f"case {n}: question_id {c['question_id'][:8]} differs from the table's {row['question_id']}")
        if c["condition"] != row["condition"]:
            problems.append(f"case {n}: condition {c['condition']!r} differs from {row['condition']!r}")
        if c["sent"] != row["sent"].startswith("yes"):
            problems.append(f"case {n}: sent={c['sent']} differs from the table's {row['sent']!r}")
        if c["ocr_preserves"] != row["ocr"]:
            problems.append(f"case {n}: ocr_preserves {c['ocr_preserves']!r} differs from {row['ocr']!r}")
        holds = row["holds"].replace("not sent", "not_sent")
        word, _, rank = holds.partition(", rank ")
        if c["retrieval_contains"] != word:
            problems.append(f"case {n}: retrieval_contains {c['retrieval_contains']!r} differs from {word!r}")
        if rank and c.get("retrieval_rank") != int(rank):
            problems.append(f"case {n}: retrieval_rank {c.get('retrieval_rank')} differs from the table's {rank}")
        if c["likely_origin"] != row["origin"]:
            problems.append(f"case {n}: likely_origin {c['likely_origin']!r} differs from {row['origin']!r}")
        if c["gold_questionable"] != row["gold"]:
            problems.append(f"case {n}: gold_questionable {c['gold_questionable']!r} differs from {row['gold']!r}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default=DEFAULT_CASES, help=f"structured agent review (default: {DEFAULT_CASES})")
    parser.add_argument("--check-markdown", metavar="PATH", help="markdown report whose summary table must agree")
    parser.add_argument("--out", help="directory for case_review_counts.json; refused under results/pilots/")
    args = parser.parse_args(argv)

    review = read_json(resolve(args.cases), "structured agent review")
    try:
        counts = count(review)
    except ValueError as exc:
        fail(f"structured agent review is unsound: {exc}", code=EXIT_FAILED_CHECK)

    print("AGENT REVIEW counts (agent judgement on a purposive sample of 20 cases, not dataset rates)")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    status = 0
    if args.check_markdown:
        md_path = require_file(resolve(args.check_markdown), "markdown report")
        problems = compare_with_markdown(review, md_path.read_text(encoding="utf-8"))
        for problem in problems:
            print(f"FAIL  {problem}")
        if problems:
            status = EXIT_FAILED_CHECK
        else:
            print(f"PASS  all {counts['cases']} rows agree with the summary table of {md_path.name}")
    if args.out:
        out = prepare_out(resolve(args.out))
        write_json(out / "case_review_counts.json", counts)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
