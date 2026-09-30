"""Select the questions of the small live check by the fixed rule ``faar-live-check-v1``.

The rule reads runtime information only: question text, retrieved evidence text, the page count of the
document, the input bound and whether the dry run sends or skips the question. It never opens
``evaluation_manifest.json`` and never reads an answer, a label or a score.

Rank of a question: ``sha256("faar-live-check-v1|" + question_id)`` as lowercase hex, ascending.
The strata run in this order, and each draws only from questions not chosen earlier:

    L  the sendable question with the largest input bound (a tie goes to the lowest rank)   1
    H  sendable, the question text has a Han character, lowest ranks                        2
    M  sendable, no Han in the question, the document has 3 or more pages, lowest rank      1
    E  sendable, no Han in the question or the evidence, single-page document, lowest ranks 2
    S  skipped: the lowest-ranked ``no_text_chunks`` question, and the lowest-ranked
       ``no_text_content`` question                                                        1 + 1

Outputs, written only to ``--out`` (never under ``results/pilots/``):

    selected_question_ids.json          the ids in manifest order
    runtime_manifest.livecheck-v1.json  the frozen runtime manifest cut down to those questions and
                                        the documents they use, so it is an exact subset of it
    selection_trace.json                the stratum of each id and its rank

Usage:
    python scripts/audits/select_live_check.py --dry-run-dir .local/work/dry-run --out .local/work/livecheck
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EXIT_FAILED_CHECK,
    HAN,
    fail,
    ids_digest,
    prepare_out,
    read_json,
    read_jsonl,
    require_dir,
    resolve,
    sha256_file,
    write_json,
)

RULE = "faar-live-check-v1"
DEFAULT_MANIFEST = "results/pilots/ohr_dev_v1/runtime_manifest.json"
DEFAULT_DRY_RUN = ".local/work/dry-run"


class SelectionError(ValueError):
    """The prepared requests cannot fill a stratum."""


def rank_of(question_id: str) -> str:
    return hashlib.sha256(f"{RULE}|{question_id}".encode()).hexdigest()


def _rank(row: dict[str, Any]) -> str:
    return rank_of(row["question_id"])


def _has_han_question(row: dict[str, Any]) -> bool:
    return bool(HAN.search(row["question"]))


def _has_han_evidence(row: dict[str, Any]) -> bool:
    return any(HAN.search(block["text"]) for block in row["evidence"])


def select(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Apply the rule to the prepared requests. Return ``{question_id: stratum}`` in selection order.

    Raises ``SelectionError`` when a stratum has too few candidates.
    """
    sends = [r for r in rows if r["action"] == "send"]
    chosen: dict[str, str] = {}

    def take(stratum: str, pool: Sequence[dict[str, Any]], count: int, key: Any = _rank) -> None:
        picked = sorted((r for r in pool if r["question_id"] not in chosen), key=key)[:count]
        if len(picked) < count:
            raise SelectionError(
                f"stratum {stratum} needs {count} questions and the prepared requests give {len(picked)}"
            )
        for row in picked:
            chosen[row["question_id"]] = stratum

    take("L", sends, 1, key=lambda r: (-r["input_token_upper_bound"], _rank(r)))
    take("H", [r for r in sends if _has_han_question(r)], 2)
    take("M", [r for r in sends if not _has_han_question(r) and r["ocr_condition"]["pages_total"] >= 3], 1)
    take(
        "E",
        [
            r
            for r in sends
            if not _has_han_question(r) and not _has_han_evidence(r) and r["ocr_condition"]["pages_total"] == 1
        ],
        2,
    )
    take("S", [r for r in rows if r["skip_reason"] == "no_text_chunks"], 1)
    take("S", [r for r in rows if r["skip_reason"] == "no_text_content"], 1)
    return chosen


def derive_manifest(frozen: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    """Cut the frozen runtime manifest to the selected questions and the documents they use."""
    derived = dict(frozen)
    questions = [q for q in frozen["questions"] if q["question_id"] in selected]
    needed = {q["doc_id"] for q in questions}
    derived["questions"] = questions
    derived["documents"] = [d for d in frozen["documents"] if d["doc_id"] in needed]
    return derived


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """The serialisation the readiness report's manifest hash was taken over."""
    return (json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dry-run-dir",
        default=DEFAULT_DRY_RUN,
        help=f"directory with prepared_requests.jsonl from run_pilot_live.py dry-run (default: {DEFAULT_DRY_RUN})",
    )
    parser.add_argument(
        "--runtime-manifest", default=DEFAULT_MANIFEST, help=f"frozen runtime manifest (default: {DEFAULT_MANIFEST})"
    )
    parser.add_argument("--out", required=True, help="output directory; refused under results/pilots/")
    args = parser.parse_args(argv)

    dry_run = require_dir(resolve(args.dry_run_dir), "dry-run directory", "run_pilot_live.py dry-run writes it")
    rows = read_jsonl(dry_run / "prepared_requests.jsonl", "prepared_requests.jsonl")
    manifest_path = resolve(args.runtime_manifest)
    frozen = read_json(manifest_path, "runtime manifest")
    out = prepare_out(resolve(args.out))

    manifest_ids = [q["question_id"] for q in frozen["questions"]]
    if [r["question_id"] for r in rows] != manifest_ids:
        fail("prepared_requests.jsonl does not list the runtime manifest's questions in manifest order")

    try:
        chosen = select(rows)
    except SelectionError as exc:
        fail(str(exc), code=EXIT_FAILED_CHECK)

    ids = [qid for qid in manifest_ids if qid in chosen]
    derived = derive_manifest(frozen, set(ids))
    manifest_out = out / "runtime_manifest.livecheck-v1.json"
    manifest_out.write_bytes(manifest_bytes(derived))
    write_json(out / "selected_question_ids.json", ids)
    by_id = {r["question_id"]: r for r in rows}
    trace = [
        {
            "question_id": qid,
            "stratum": chosen[qid],
            "rank_sha256": rank_of(qid),
            "action": by_id[qid]["action"],
            "skip_reason": by_id[qid]["skip_reason"],
            "input_token_upper_bound": by_id[qid]["input_token_upper_bound"],
            "pages_total": by_id[qid]["ocr_condition"]["pages_total"],
        }
        for qid in ids
    ]
    write_json(out / "selection_trace.json", {"rule": RULE, "selected": trace})

    print(f"rule {RULE}: {len(ids)} questions ({sum(1 for t in trace if t['action'] == 'send')} sent)")
    for t in trace:
        print(
            f"  {t['question_id']}  {t['stratum']}  {t['action']:4s}  pages={t['pages_total']}"
            f"  bound={t['input_token_upper_bound']}  rank={t['rank_sha256'][:8]}"
        )
    print(f"documents {len(derived['documents'])}")
    print(f"derived manifest {manifest_out}")
    print(f"derived manifest sha256 {sha256_file(manifest_out)}")
    print(f"frozen manifest sha256  {sha256_file(manifest_path)}")
    print(f"question_ids digest     {ids_digest(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
