"""Tests for faar.retry_policy. The failure list is in that module's docstring.

Each test carries the number of the failure it covers. No test sleeps.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

import faar.retry_policy as retry_policy
from faar.live_contract import PROVIDER_OUTCOMES, ProviderError
from faar.retry_policy import STOP_RUN_KINDS, Decision, RetryPolicy

# The kinds the provider adapter emits that the tests below use, with their (outcome, retryable) pair and the
# action the policy takes on the first failure.
PROVIDER_KINDS = {
    "connect": ("not_sent", True),
    "client_config": ("not_sent", False),
    "timeout": ("unknown", False),
    "malformed_response": ("unknown", False),
    "rate_limit": ("rejected", True),
    "quota": ("rejected", False),
    "not_found": ("rejected", False),
}
EXPECTED_ACTION = {
    "connect": "retry",
    "client_config": "fail_question",
    "timeout": "reconcile",
    "malformed_response": "reconcile",
    "rate_limit": "retry",
    "quota": "stop_run",
    "not_found": "fail_question",
}


def err(kind: str = "server_error", outcome: str = "rejected", retryable: bool = True, status: int | None = None):
    return ProviderError("boom", kind=kind, outcome=outcome, retryable=retryable, http_status=status)


# 1: unknown outcomes are never retried.
@pytest.mark.parametrize("kind", ["server_error", "auth"])  # an unknown outcome wins over the stop-run kinds
@pytest.mark.parametrize("retryable", [True, False])
@pytest.mark.parametrize("attempts", [1, 3])
def test_unknown_outcome_reconciles(kind, retryable, attempts):
    decision = RetryPolicy().decide(err(kind, "unknown", retryable), attempts)
    assert decision.action == "reconcile"
    assert decision.delay_seconds == 0.0


# 2: auth, unknown-model and quota errors stop the run on the first occurrence.
@pytest.mark.parametrize("kind", ["auth", "unknown_model", "quota"])
@pytest.mark.parametrize(("outcome", "retryable"), [("rejected", False), ("not_sent", True)])
def test_stop_run_kinds(kind, outcome, retryable):
    decision = RetryPolicy().decide(err(kind, outcome, retryable, 401), 1)
    assert decision.action == "stop_run"
    assert decision.delay_seconds == 0.0


@pytest.mark.parametrize("attempts", [1, 3])
@pytest.mark.parametrize("kind", sorted(PROVIDER_KINDS))
def test_decision_for_a_provider_kind_at_the_first_and_the_last_attempt(kind, attempts):
    # Only a retry turns into fail_question at the attempt limit. Every other action is unchanged.
    outcome, retryable = PROVIDER_KINDS[kind]
    decision = RetryPolicy().decide(err(kind, outcome, retryable), attempts)
    expected = EXPECTED_ACTION[kind]
    if expected == "retry" and attempts == 3:
        expected = "fail_question"
    assert decision.action == expected
    assert decision.reason
    assert (decision.action == "retry") == (decision.delay_seconds > 0)


# 3: retryable errors retry while attempts remain and fail the question at the limit.
@pytest.mark.parametrize(("kind", "outcome"), [("connect", "not_sent"), ("server_error", "rejected")])
def test_retryable_errors_retry_until_attempts_run_out(kind, outcome):
    policy = RetryPolicy()
    assert [policy.decide(err(kind, outcome), n).action for n in (1, 2, 3, 4, 10)] == [
        "retry",
        "retry",
        "fail_question",
        "fail_question",
        "fail_question",
    ]


@pytest.mark.parametrize("max_attempts", [1, 5])
def test_attempts_are_bounded(max_attempts):
    policy = RetryPolicy(max_attempts=max_attempts)
    made = 0
    while True:
        made += 1
        decision = policy.decide(err(), made)
        if decision.action != "retry":
            break
        assert made < 100
    assert decision.action == "fail_question"
    assert made == max_attempts


# 4: everything else fails the question.
@pytest.mark.parametrize(("kind", "outcome"), [("client_config", "not_sent"), ("bad_request", "rejected")])
def test_non_retryable_errors_fail_the_question(kind, outcome):
    for n in (1, 2, 3):
        decision = RetryPolicy().decide(err(kind, outcome, retryable=False, status=400), n)
        assert decision.action == "fail_question"
        assert decision.delay_seconds == 0.0


def test_retryable_flag_must_be_exactly_true():
    error = err()
    error.retryable = "yes"  # type: ignore[assignment]
    assert RetryPolicy().decide(error, 1).action == "fail_question"
    error.retryable = 1  # type: ignore[assignment]
    assert RetryPolicy().decide(error, 1).action == "fail_question"


# 5 to 7: delays are bounded, reproducible and never below the plain exponential value.
def test_delay_without_jitter_is_exponential():
    policy = RetryPolicy(jitter_fraction=0.0, max_attempts=6)
    assert [policy.delay_for(n) for n in range(1, 6)] == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_delay_is_capped():
    policy = RetryPolicy(max_attempts=50, jitter_fraction=0.25, max_delay_seconds=60.0)
    assert all(policy.delay_for(n) <= 60.0 for n in range(1, 50))
    assert policy.delay_for(49) == 60.0
    assert policy.delay_for(5000) == 60.0  # 2 ** 4999 overflows a float; the cap still holds


def test_jitter_stays_inside_its_band_and_only_adds():
    for seed in range(50):
        policy = RetryPolicy(seed=seed, max_attempts=6, max_delay_seconds=1000.0)
        for n in range(1, 6):
            raw = 2.0 * 2.0 ** (n - 1)
            delay = policy.delay_for(n)
            assert raw <= delay < raw * 1.25


def test_delays_are_deterministic_across_calls_instances_and_processes():
    first = [RetryPolicy(seed=7).delay_for(n) for n in range(1, 4)]
    assert first == [RetryPolicy(seed=7).delay_for(n) for n in range(1, 4)]
    code = (
        "from faar.retry_policy import RetryPolicy;"
        "import json;print(json.dumps([RetryPolicy(seed=7).delay_for(n) for n in range(1,4)]))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(retry_policy.__file__).resolve().parents[1])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert json.loads(out.stdout) == first


def test_delay_values_are_pinned():
    # A change to the jitter function changes fake runs, so pin the numbers (computed from sha256 of "seed:attempt").
    assert [RetryPolicy(seed=0).delay_for(n) for n in (1, 2, 3)] == pytest.approx(
        [2.4669441927799403, 4.574839226063698, 8.928337446877887], rel=1e-12
    )
    assert [RetryPolicy(seed=7).delay_for(n) for n in (1, 2)] == pytest.approx(
        [2.42114874437817, 4.552957740962898], rel=1e-12
    )


def test_different_seeds_and_attempts_give_different_jitter():
    a = {RetryPolicy(seed=s).delay_for(1) for s in range(20)}
    assert len(a) > 15
    policy = RetryPolicy(jitter_fraction=1.0, factor=1.0, max_attempts=10)
    assert len({policy.delay_for(n) for n in range(1, 10)}) > 5


def test_decision_carries_the_policy_delay():
    policy = RetryPolicy(seed=3)
    decision = policy.decide(err(), 2)
    assert decision.action == "retry"
    assert decision.delay_seconds == policy.delay_for(2)


# 9: the module never sleeps and never touches the global random state.
def test_decision_does_not_consume_global_random_numbers():
    random.seed(42)
    expected = random.random()
    random.seed(42)
    RetryPolicy().decide(err(), 1)
    assert random.random() == expected


def test_module_never_sleeps(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("retry_policy must not sleep")

    monkeypatch.setattr(time, "sleep", fail)
    policy = RetryPolicy()
    for kind, outcome, retryable, attempts in itertools.product(
        ["server_error", "auth"], PROVIDER_OUTCOMES, [True, False], [1, 3]
    ):
        policy.decide(err(kind, outcome, retryable), attempts)


# 8: invalid input.
@pytest.mark.parametrize("attempts", [0, 1.0, True])
def test_attempts_so_far_must_be_a_positive_integer(attempts):
    with pytest.raises(ValueError):
        RetryPolicy().decide(err(), attempts)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RetryPolicy().delay_for(attempts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"base_delay_seconds": 0},
        {"base_delay_seconds": math.nan},
        {"factor": 0.5},
        {"factor": math.inf},
        {"jitter_fraction": -0.1},
        {"jitter_fraction": 1.1},
        {"seed": True},
        {"max_delay_seconds": 1.0},
        {"max_delay_seconds": math.inf},
    ],
)
def test_policy_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


# 10: describe.
def test_describe_lists_every_parameter_and_is_json_ready():
    description = RetryPolicy().describe()
    assert json.loads(json.dumps(description)) == description
    assert description["stop_run_kinds"] == list(STOP_RUN_KINDS)
    assert description["policy"] == retry_policy.POLICY_ID


def test_describe_changes_when_any_parameter_changes():
    changes = [
        {"max_attempts": 4},
        {"base_delay_seconds": 3.0},
        {"factor": 3.0},
        {"jitter_fraction": 0.5},
        {"seed": 1},
        {"max_delay_seconds": 30.0},
    ]
    for kwargs in changes:
        assert RetryPolicy(**kwargs).describe() != RetryPolicy().describe(), kwargs


def test_decision_as_dict():
    assert Decision("retry", 1.0, "why").as_dict() == {"action": "retry", "delay_seconds": 1.0, "reason": "why"}
