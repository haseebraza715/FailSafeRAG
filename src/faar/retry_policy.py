"""Retry decisions for failed answer-model attempts (contract rule 5).

The policy maps one ``ProviderError`` to one action. It never sleeps, never
reads a clock and never touches the global random state. The run driver
receives ``Decision.delay_seconds`` and calls its own injected ``sleep``.

Failure list (each item has a test in ``tests/test_retry_policy.py``)
======================================================================

  1. An error whose outcome is ``unknown`` may have been processed and billed.
     Retrying it could bill twice. The action is ``reconcile``, for every kind
     and for retryable and non-retryable errors alike.
  2. An authentication (HTTP 401 and 403), unknown-model or quota error fails
     every later request too. The action is ``stop_run`` on the first
     occurrence, even when the error claims to be retryable. The provider has
     no separate permission kind: it maps 403 to ``auth``.
  3. A retryable error with outcome ``not_sent`` or ``rejected`` is retried
     only while attempts remain. At ``max_attempts`` the action is
     ``fail_question``. The policy never allows attempt ``max_attempts + 1``.
  4. Any other error is not retryable and gives ``fail_question``. A
     ``retryable`` flag that is not exactly ``True`` counts as False.
  5. Delays must not grow without limit. Each delay is at most
     ``max_delay_seconds``.
  6. Jitter must be reproducible. The same seed and attempt number give the
     same delay in every process and run. Two seeds give different delays.
  7. Jitter must only add to the delay, so the delay is never below the plain
     exponential value (unless the cap applies).
  8. ``attempts_so_far`` below 1, or not an integer, is a caller bug and
     raises. The parameters are checked when the policy is built: a
     non-positive ``max_attempts`` or ``base_delay_seconds``, a ``factor``
     below 1, a ``jitter_fraction`` outside [0, 1] and non-finite values all
     raise.
  9. Importing or using the module must not sleep or consume global random
     numbers.
 10. ``describe()`` must return every parameter that changes a decision, so
     the run identity changes when the policy changes.

Semantics
=========

``attempts_so_far`` counts the attempts already made for this question,
including the one that just failed. After the first failure it is 1.

The delay before attempt ``n + 1`` is

    raw    = base_delay_seconds * factor ** (n - 1)
    u      = a number in [0, 1) taken from sha256(f"{seed}:{n}")
    delay  = min(max_delay_seconds, raw * (1 + jitter_fraction * u))

The existing visual-fallback code in ``faar.recovery`` uses the same shape
(base 2 seconds, doubling) with global-random jitter of up to 0.25 seconds.
This policy replaces that jitter with a seeded one so a fake run repeats.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from faar.live_contract import OUTCOME_NOT_SENT, OUTCOME_REJECTED, OUTCOME_UNKNOWN, ProviderError

POLICY_ID = "exponential-backoff-seeded-jitter-v1"

ACTION_RETRY = "retry"
ACTION_FAIL_QUESTION = "fail_question"
ACTION_RECONCILE = "reconcile"
ACTION_STOP_RUN = "stop_run"
ACTIONS = (ACTION_RETRY, ACTION_FAIL_QUESTION, ACTION_RECONCILE, ACTION_STOP_RUN)

# Error kinds after which every later request would fail the same way. The provider adapter
# (faar.answer_providers) emits exactly these names. This module does not import it.
STOP_RUN_KINDS = ("auth", "unknown_model", "quota")
RETRYABLE_OUTCOMES = (OUTCOME_NOT_SENT, OUTCOME_REJECTED)


@dataclass(frozen=True)
class Decision:
    """What the driver does next. ``delay_seconds`` is 0.0 unless the action is ``retry``."""

    action: str
    delay_seconds: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "delay_seconds": self.delay_seconds, "reason": self.reason}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    factor: float = 2.0
    jitter_fraction: float = 0.25
    seed: int = 0
    max_delay_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not _is_int(self.max_attempts) or self.max_attempts < 1:
            raise ValueError(f"max_attempts must be an integer of at least 1, got {self.max_attempts!r}")
        if not _finite(self.base_delay_seconds) or self.base_delay_seconds <= 0:
            raise ValueError(f"base_delay_seconds must be positive and finite, got {self.base_delay_seconds!r}")
        if not _finite(self.factor) or self.factor < 1:
            raise ValueError(f"factor must be finite and at least 1, got {self.factor!r}")
        if not _finite(self.jitter_fraction) or not 0 <= self.jitter_fraction <= 1:
            raise ValueError(f"jitter_fraction must be in [0, 1], got {self.jitter_fraction!r}")
        if not _is_int(self.seed):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if not _finite(self.max_delay_seconds) or self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                f"max_delay_seconds must be finite and at least base_delay_seconds, got {self.max_delay_seconds!r}"
            )

    def _unit(self, attempts_so_far: int) -> float:
        """Deterministic number in [0, 1) from the seed and the attempt number."""
        digest = hashlib.sha256(f"{self.seed}:{attempts_so_far}".encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def delay_for(self, attempts_so_far: int) -> float:
        """Seconds to wait after the ``attempts_so_far``-th failed attempt."""
        if not _is_int(attempts_so_far) or attempts_so_far < 1:
            raise ValueError(f"attempts_so_far must be an integer of at least 1, got {attempts_so_far!r}")
        try:
            raw = self.base_delay_seconds * self.factor ** (attempts_so_far - 1)
        except OverflowError:
            raw = math.inf
        jittered = raw * (1 + self.jitter_fraction * self._unit(attempts_so_far))
        return float(min(self.max_delay_seconds, jittered))

    def decide(self, error: ProviderError, attempts_so_far: int) -> Decision:
        """Choose the next action after a failed attempt."""
        if not _is_int(attempts_so_far) or attempts_so_far < 1:
            raise ValueError(f"attempts_so_far must be an integer of at least 1, got {attempts_so_far!r}")
        kind = error.kind
        if error.outcome == OUTCOME_UNKNOWN:
            return Decision(
                ACTION_RECONCILE,
                0.0,
                f"outcome unknown ({kind}): the request may have been processed, so it is never resent automatically",
            )
        if kind in STOP_RUN_KINDS:
            return Decision(ACTION_STOP_RUN, 0.0, f"{kind} error: every later request would fail the same way")
        if error.retryable is True and error.outcome in RETRYABLE_OUTCOMES:
            if attempts_so_far < self.max_attempts:
                return Decision(
                    ACTION_RETRY,
                    self.delay_for(attempts_so_far),
                    f"retryable {kind} error ({error.outcome}), attempt {attempts_so_far} of {self.max_attempts}",
                )
            return Decision(
                ACTION_FAIL_QUESTION,
                0.0,
                f"retryable {kind} error but all {self.max_attempts} attempts are used",
            )
        return Decision(ACTION_FAIL_QUESTION, 0.0, f"non-retryable {kind} error ({error.outcome})")

    def describe(self) -> dict[str, Any]:
        """Every parameter that changes a decision, for the run identity."""
        return {
            "policy": POLICY_ID,
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "factor": self.factor,
            "jitter_fraction": self.jitter_fraction,
            "seed": self.seed,
            "max_delay_seconds": self.max_delay_seconds,
            "stop_run_kinds": list(STOP_RUN_KINDS),
            "retryable_outcomes": list(RETRYABLE_OUTCOMES),
        }
