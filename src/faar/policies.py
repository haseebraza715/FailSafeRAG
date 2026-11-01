"""Assemble policies from a shared initial answer and recorded repair outcomes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

REPAIRS = ("retry_retrieval", "correct_text", "invoke_vlm")
RANDOM_SEEDS = (20261101, 20261102, 20261103)


def random_repair(seed: int, question_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()
    return REPAIRS[int(digest, 16) % 3]


def assemble(
    question_id: str,
    initial: Mapping[str, Any],
    flagged: bool,
    route: str,
    repairs: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, tuple[Mapping[str, Any], str]]:
    """Return each policy's recorded row and source. Gold answers are never inputs."""
    choices = {
        "gated_visual": "invoke_vlm",
        "faar": route,
        **{f"random_{seed}": random_repair(seed, question_id) for seed in RANDOM_SEEDS},
        "all_retry_retrieval": "retry_retrieval",
        "all_correct_text": "correct_text",
    }
    result = {"no_recovery": (initial, "initial")}
    for policy, repair in choices.items():
        if not flagged:
            result[policy] = (initial, "kept_initial:not_flagged")
            continue
        if repair not in REPAIRS:
            raise ValueError(f"unknown repair: {repair}")
        row = repairs.get(repair)
        if row is None:
            raise ValueError(f"missing {repair} row for {question_id}")
        status = row.get("status")
        if status == "skipped":
            reason = row.get("skip_reason")
            if not reason:
                raise ValueError(f"missing skip reason for {question_id}: {repair}")
            result[policy] = (initial, f"kept_initial:skip:{reason}")
        elif status in ("answered", "execution_failed"):
            result[policy] = (row, f"repair:{repair}")
        else:
            raise ValueError(f"invalid repair status: {status!r}")
    return result
