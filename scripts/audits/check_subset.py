"""Check that a derived runtime manifest is an exact subset of the frozen one.

A derived manifest is an exact subset when:

* it has the same top-level fields as the frozen manifest, and every field other than ``documents`` and
  ``questions`` has the same value;
* every document entry equals the frozen entry with the same ``doc_id``, and the documents keep the frozen order;
* every question equals the frozen question with the same ``question_id``, with no duplicates, in the frozen order
  (the derived ids are a subsequence of the frozen ids);
* every question's ``doc_id`` has a document entry.

The script prints the derived file's sha256, its question count and the digest of its question ids. The digest is
the sha256 of the ids in manifest order joined by a newline. It exits 1 when the manifest is not an exact subset.

Usage:
    python scripts/audits/check_subset.py --derived .local/work/livecheck/runtime_manifest.livecheck-v1.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXIT_FAILED_CHECK, ids_digest, read_json, resolve, sha256_file

DEFAULT_FROZEN = "results/pilots/ohr_dev_v1/runtime_manifest.json"
LISTS = ("documents", "questions")


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    remaining = iter(haystack)
    return all(item in remaining for item in needle)


def check_subset(frozen: dict[str, Any], derived: dict[str, Any]) -> list[str]:
    """Return the reasons ``derived`` is not an exact subset of ``frozen``. An empty list means it is."""
    problems: list[str] = []
    if set(frozen) != set(derived):
        problems.append(f"top-level fields differ: frozen-only {sorted(set(frozen) - set(derived))}, "
                        f"derived-only {sorted(set(derived) - set(frozen))}")
    for key in sorted(set(frozen) & set(derived)):
        if key not in LISTS and frozen[key] != derived[key]:
            problems.append(f"top-level field {key!r} differs")
    for key in LISTS:
        if not isinstance(derived.get(key), list):
            problems.append(f"{key!r} is missing or not a list")
    if problems:
        return problems

    frozen_docs = {d["doc_id"]: d for d in frozen["documents"]}
    derived_doc_ids = [d["doc_id"] for d in derived["documents"]]
    if len(set(derived_doc_ids)) != len(derived_doc_ids):
        problems.append("a document appears twice")
    for doc in derived["documents"]:
        if doc["doc_id"] not in frozen_docs:
            problems.append(f"document {doc['doc_id']!r} is not in the frozen manifest")
        elif doc != frozen_docs[doc["doc_id"]]:
            problems.append(f"document {doc['doc_id']!r} differs from the frozen entry")
    if not _is_subsequence(derived_doc_ids, [d["doc_id"] for d in frozen["documents"]]):
        problems.append("documents are not in frozen order")

    frozen_questions = {q["question_id"]: q for q in frozen["questions"]}
    derived_ids = [q["question_id"] for q in derived["questions"]]
    if len(set(derived_ids)) != len(derived_ids):
        problems.append("a question appears twice")
    for question in derived["questions"]:
        qid = question["question_id"]
        if qid not in frozen_questions:
            problems.append(f"question {qid} is not in the frozen manifest")
            continue
        if question != frozen_questions[qid]:
            problems.append(f"question {qid} differs from the frozen entry")
        if question["doc_id"] not in derived_doc_ids:
            problems.append(f"question {qid} uses document {question['doc_id']!r}, which the derived manifest lacks")
    known = [qid for qid in derived_ids if qid in frozen_questions]
    if not _is_subsequence(known, [q["question_id"] for q in frozen["questions"]]):
        problems.append("questions are not in frozen manifest order")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--derived", required=True, help="derived runtime manifest to check")
    parser.add_argument("--frozen", default=DEFAULT_FROZEN, help=f"frozen runtime manifest (default: {DEFAULT_FROZEN})")
    args = parser.parse_args(argv)

    frozen_path = resolve(args.frozen)
    derived_path = resolve(args.derived)
    frozen = read_json(frozen_path, "frozen runtime manifest")
    derived = read_json(derived_path, "derived runtime manifest")

    problems = check_subset(frozen, derived)
    ids = [q["question_id"] for q in derived.get("questions", [])]
    print(f"derived manifest   {derived_path}")
    print(f"sha256             {sha256_file(derived_path)}")
    print(f"frozen sha256      {sha256_file(frozen_path)}")
    print(f"questions          {len(ids)} of {len(frozen['questions'])}")
    print(f"documents          {len(derived.get('documents', []))} of {len(frozen['documents'])}")
    print(f"question_ids digest {ids_digest(ids)}")
    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        return EXIT_FAILED_CHECK
    print("PASS  exact subset of the frozen manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
