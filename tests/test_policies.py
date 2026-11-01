"""Failure list:

1. Random assignments differ from the preregistered hash.
2. An unflagged question or skipped repair changes the initial answer.
3. A blocked repair silently keeps the initial answer.
4. A missing or invalid chosen repair silently falls back.
"""

import hashlib

import pytest

from faar.policies import RANDOM_SEEDS, REPAIRS, assemble, random_repair


def test_random_assignment():
    for seed in RANDOM_SEEDS:
        for qid in ("q1", "other question", "é"):
            expected = REPAIRS[int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(), 16) % 3]
            assert random_repair(seed, qid) == expected


def test_unflagged_and_skip_keep_initial():
    initial = {"status": "answered", "answer": "initial"}
    assert all(row is initial for row, _ in assemble("q", initial, False, "", {}).values())
    skipped = {name: {"status": "skipped", "skip_reason": "no_text"} for name in REPAIRS}
    policies = assemble("q", initial, True, "correct_text", skipped)
    assert policies["faar"] == (initial, "kept_initial:skip:no_text")


def test_blocked_remains_failed():
    failed = {"status": "execution_failed", "answer": ""}
    policies = assemble("q", {}, True, "correct_text", dict.fromkeys(REPAIRS, failed))
    assert policies["faar"] == (failed, "repair:correct_text")


def test_missing_and_invalid_repairs_fail():
    with pytest.raises(ValueError, match="missing"):
        assemble("q", {}, True, "correct_text", {})
    with pytest.raises(ValueError, match="unknown"):
        assemble("q", {}, True, "bad", {"invoke_vlm": {"status": "answered"}})
