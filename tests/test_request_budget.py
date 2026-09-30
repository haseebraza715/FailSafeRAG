"""Tests for faar.request_budget. The failure list is in that module's docstring.

Each test carries the number of the failure it covers. Nothing here opens a
network connection or calls a model. Prices are simulated or invented.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from faar.live_contract import PriceTable, ProviderUsage
from faar.request_budget import (
    BudgetError,
    LedgerError,
    SafetyLedger,
    input_token_upper_bound,
    measured_cost,
    micro_to_amount,
    request_cost_upper_bound,
)

# Invented rates that are exact in decimal: $2.50 per million input, $1.25 cached, $10.00 output.
REAL = PriceTable("openai", "test-model", "USD", 2.5, 1.25, 10.0, "invented for tests", "2026-09-29", False)
SIM = replace(REAL, currency="SIM", simulated=True)
NO_CACHE = replace(REAL, cached_input_per_million=None)


def msgs(*contents: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text} for text in contents]


def dispatch(attempt: int, upper: float, question: str = "q1") -> dict:
    return {
        "event": "dispatch_started",
        "question_id": question,
        "request_id": f"{question}-r",
        "attempt": attempt,
        "attempt_id": f"{question}-r-a{attempt}",
        "cost_upper_bound": upper,
        "ledger_before": {"measured": 0.0, "reserved": 0.0},
    }


def saved(attempt: int, cost: float | None, question: str = "q1", usage: dict | None = None) -> dict:
    return {
        "event": "response_saved",
        "question_id": question,
        "attempt_id": f"{question}-r-a{attempt}",
        "measured_cost": cost,
        "usage": usage if usage is not None else {},
    }


def failed(attempt: int, outcome: str, question: str = "q1") -> dict:
    return {
        "event": "attempt_failed",
        "question_id": question,
        "attempt_id": f"{question}-r-a{attempt}",
        "kind": "server_error",
        "outcome": outcome,
        "retryable": True,
    }


def unknown(attempt: int, question: str = "q1") -> dict:
    return {"event": "outcome_unknown", "question_id": question, "attempt_id": f"{question}-r-a{attempt}", "kind": "timeout"}


def reconciled(attempt: int, resolution: str = "allow_new_attempt", question: str = "q1") -> dict:
    return {
        "event": "reconciled",
        "attempt_id": f"{question}-r-a{attempt}",
        "resolution": resolution,
        "note": "checked the provider dashboard",
    }


# ---------------------------------------------------------------------------
# Input bound (failure list 1 to 5)
# ---------------------------------------------------------------------------


def test_input_bound_formula():
    # 1: content bytes, plus 4 per message, plus 16.
    assert input_token_upper_bound(msgs("abc")) == 3 + 4 + 16
    assert input_token_upper_bound(msgs("abc", "de")) == 3 + 2 + 2 * 4 + 16
    assert input_token_upper_bound(msgs("")) == 4 + 16


@pytest.mark.parametrize("bad", [None, b"bytes", 12, 1.5, ["a", "b"], [{"type": "image_url"}], {"text": "x"}])
def test_input_bound_refuses_non_string_content(bad):
    # 1
    with pytest.raises(TypeError):
        input_token_upper_bound([{"role": "user", "content": bad}])


@pytest.mark.parametrize("bad", ["text", b"text", None, [None], ["text"], [{"role": "user"}], [42]])
def test_input_bound_refuses_malformed_messages(bad):
    # 2
    with pytest.raises(TypeError):
        input_token_upper_bound(bad)


def test_input_bound_refuses_empty_message_list():
    # 3
    with pytest.raises(ValueError):
        input_token_upper_bound([])


def test_input_bound_refuses_lone_surrogate():
    with pytest.raises(ValueError):
        input_token_upper_bound(msgs("bad \ud800 text"))


def test_input_bound_counts_bytes_of_multibyte_text():
    # 4: three bytes per Han character, four for a supplementary-plane emoji.
    assert input_token_upper_bound(msgs("中文问答")) == 12 + 4 + 16
    assert input_token_upper_bound(msgs("😀")) == 4 + 4 + 16
    assert input_token_upper_bound(msgs("é")) == 2 + 4 + 16


def _load_ranks(name: str):
    directory = os.environ.get("FAAR_TIKTOKEN_RANKS_DIR")
    if not directory or not (Path(directory) / name).is_file():
        pytest.skip(f"set FAAR_TIKTOKEN_RANKS_DIR to a directory with {name} (base64 rank format)")
    tiktoken = pytest.importorskip("tiktoken")
    from tiktoken.load import load_tiktoken_bpe

    return tiktoken, load_tiktoken_bpe(str(Path(directory) / name))


GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SAMPLES = [
    "Total revenue for fiscal 2023 was $1,234.5 million, up 12% year over year.",
    "中文问答评测：请根据证据回答问题，不要翻译。营业收入为人民币三百二十万元。",
    "日本語のテキストと English mixed 한국어 with numbers 12345678901234567890.",
    "😀😀😀 emoji ​​ zero-width and \t\t tabs\n\n\n newlines",
    "x" * 500,
    "".join(chr(random.Random(7).randrange(0x4E00, 0x9FFF)) for _ in range(300)),
]


@pytest.mark.parametrize("ranks_file", ["gpt2.tiktoken", "multilingual.tiktoken"])
def test_input_bound_dominates_a_real_byte_level_bpe_count(ranks_file):
    # 5: needs local vocabulary files, so it skips when they are absent. tiktoken is not a project dependency.
    tiktoken, ranks = _load_ranks(ranks_file)
    encoding = tiktoken.Encoding(name=ranks_file, pat_str=GPT2_PAT, mergeable_ranks=ranks, special_tokens={})
    for text in SAMPLES:
        messages = msgs(text, text[::-1])
        counted = sum(len(encoding.encode(m["content"])) for m in messages)
        assert counted + 4 * len(messages) <= input_token_upper_bound(messages), text[:30]


def test_input_bound_property_against_a_toy_byte_level_bpe():
    # 5: a byte-level merge process cannot make a token shorter than one byte, so tokens <= bytes.
    rng = random.Random(3)
    for _ in range(200):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200)))
        tokens = [bytes([b]) for b in data]
        for _ in range(rng.randrange(0, 50)):  # random merges of adjacent tokens
            if len(tokens) < 2:
                break
            i = rng.randrange(len(tokens) - 1)
            tokens[i : i + 2] = [tokens[i] + tokens[i + 1]]
        assert all(len(t) >= 1 for t in tokens)
        assert len(tokens) <= len(data)


# ---------------------------------------------------------------------------
# Request cost bound (6 to 9)
# ---------------------------------------------------------------------------


def test_request_cost_upper_bound_value():
    # 1,000,000 input tokens at $2.50 plus 128 output tokens at $10 per million.
    assert request_cost_upper_bound(1_000_000, 128, REAL) == pytest.approx(2.5 + 0.00128)
    # Exact: 12,000 * 2.5 + 128 * 10 = 31,280 micro-units.
    assert request_cost_upper_bound(12_000, 128, REAL) == 0.03128


def test_request_cost_upper_bound_uses_full_input_rate_not_cached():
    assert request_cost_upper_bound(1000, 1, REAL) == request_cost_upper_bound(1000, 1, NO_CACHE)


def test_request_cost_upper_bound_prices_at_dearer_cached_rate_if_a_table_has_one():
    dearer = replace(REAL, cached_input_per_million=3.125)  # a cache-write surcharge shape
    assert request_cost_upper_bound(1_000_000, 1, dearer) > request_cost_upper_bound(1_000_000, 1, REAL)


@pytest.mark.parametrize("field", ["input_per_million", "output_per_million"])
@pytest.mark.parametrize("bad", [None, -1.0, math.nan, math.inf, -math.inf, True, "2.5"])
def test_request_cost_upper_bound_refuses_bad_rates(field, bad):
    # 6
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, 10, replace(REAL, **{field: bad}))


@pytest.mark.parametrize("bad", [-0.5, math.nan, math.inf, True])
def test_request_cost_upper_bound_refuses_bad_cached_rate(bad):
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, 10, replace(REAL, cached_input_per_million=bad))


def test_zero_rate_refused_for_real_table_and_allowed_for_simulated():
    # 6: a zero from a typo would switch the ceiling off.
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, 10, replace(REAL, output_per_million=0.0))
    free = replace(SIM, output_per_million=0.0)
    assert request_cost_upper_bound(1_000_000, 5, free) == 2.5


def test_request_cost_upper_bound_refuses_bad_table_metadata():
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, 10, replace(REAL, currency=""))
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, 10, replace(REAL, simulated=1))  # type: ignore[arg-type]
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, 10, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, None, 1.5, True, math.nan])
def test_request_cost_upper_bound_refuses_bad_max_output(bad):
    # 7
    with pytest.raises(BudgetError):
        request_cost_upper_bound(10, bad, REAL)


@pytest.mark.parametrize("bad", [-1, None, 10.0, True, math.nan, math.inf])
def test_request_cost_upper_bound_refuses_bad_input_bound(bad):
    # 8
    with pytest.raises(BudgetError):
        request_cost_upper_bound(bad, 10, REAL)


def test_request_cost_upper_bound_never_below_exact_product():
    # 9: awkward rates that are not exact in binary floating point.
    rng = random.Random(11)
    for _ in range(500):
        rate_in = round(rng.uniform(0.01, 30), rng.randrange(1, 6))
        rate_out = round(rng.uniform(0.01, 90), rng.randrange(1, 6))
        table = replace(REAL, input_per_million=rate_in, output_per_million=rate_out, cached_input_per_million=None)
        n_in, n_out = rng.randrange(0, 200_000), rng.randrange(1, 4096)
        exact = (Decimal(n_in) * Decimal(repr(rate_in)) + Decimal(n_out) * Decimal(repr(rate_out))) / 1_000_000
        got = Decimal(repr(request_cost_upper_bound(n_in, n_out, table)))
        assert got >= exact
        assert got - exact < Decimal("0.000001")


# ---------------------------------------------------------------------------
# Measured cost (10 to 13)
# ---------------------------------------------------------------------------


def test_measured_cost_plain_usage():
    usage = ProviderUsage(input_tokens=1000, output_tokens=200)
    assert measured_cost(usage, REAL) == pytest.approx(0.0025 + 0.002)


@pytest.mark.parametrize(
    "usage",
    [
        ProviderUsage(),
        ProviderUsage(input_tokens=10),
        ProviderUsage(output_tokens=10),
        ProviderUsage(cached_input_tokens=5, reasoning_tokens=5),
    ],
)
def test_measured_cost_is_none_without_both_token_counts(usage):
    # 10
    assert measured_cost(usage, REAL) is None


def test_measured_cost_zero_tokens_reported_is_zero_not_none():
    assert measured_cost(ProviderUsage(input_tokens=0, output_tokens=0), REAL) == 0.0


def test_measured_cost_cached_discount_only_when_reported_and_rate_exists():
    # 11
    base = ProviderUsage(input_tokens=1000, output_tokens=0)
    cached = ProviderUsage(input_tokens=1000, cached_input_tokens=400, output_tokens=0)
    zero_cached = ProviderUsage(input_tokens=1000, cached_input_tokens=0, output_tokens=0)
    assert measured_cost(base, REAL) == pytest.approx(1000 * 2.5 / 1e6)
    assert measured_cost(cached, REAL) == pytest.approx((600 * 2.5 + 400 * 1.25) / 1e6)
    assert measured_cost(zero_cached, REAL) == measured_cost(base, REAL)
    # No cached rate in the table: everything at the full rate.
    assert measured_cost(cached, NO_CACHE) == measured_cost(base, NO_CACHE)


def test_measured_cost_does_not_double_count_reasoning_tokens():
    # 12: reasoning tokens sit inside output_tokens.
    with_reasoning = ProviderUsage(input_tokens=100, output_tokens=700, reasoning_tokens=600)
    without = ProviderUsage(input_tokens=100, output_tokens=700)
    assert measured_cost(with_reasoning, REAL) == measured_cost(without, REAL)
    assert measured_cost(with_reasoning, REAL) == pytest.approx((100 * 2.5 + 700 * 10.0) / 1e6)


def test_measured_cost_none_when_counts_contradict():
    # 12 and 13
    assert measured_cost(ProviderUsage(input_tokens=10, cached_input_tokens=11, output_tokens=1), REAL) is None
    assert measured_cost(ProviderUsage(input_tokens=10, output_tokens=5, reasoning_tokens=6), REAL) is None


@pytest.mark.parametrize(
    "usage",
    [
        ProviderUsage(input_tokens=-1, output_tokens=1),
        ProviderUsage(input_tokens=1, output_tokens=-1),
        ProviderUsage(input_tokens=1.5, output_tokens=1),  # type: ignore[arg-type]
        ProviderUsage(input_tokens=True, output_tokens=1),  # type: ignore[arg-type]
        ProviderUsage(input_tokens=1, output_tokens="9"),  # type: ignore[arg-type]
        ProviderUsage(input_tokens=1, output_tokens=1, cached_input_tokens=-1),
    ],
)
def test_measured_cost_none_for_untrusted_counts(usage):
    # 13
    assert measured_cost(usage, REAL) is None


def test_measured_cost_never_below_exact_cost():
    rng = random.Random(5)
    for _ in range(300):
        table = replace(
            REAL,
            input_per_million=round(rng.uniform(0.01, 30), 4),
            output_per_million=round(rng.uniform(0.01, 90), 4),
            cached_input_per_million=round(rng.uniform(0.01, 10), 4),
        )
        n_in = rng.randrange(0, 100_000)
        cached = rng.randrange(0, n_in + 1)
        n_out = rng.randrange(0, 2000)
        usage = ProviderUsage(input_tokens=n_in, cached_input_tokens=cached, output_tokens=n_out)
        exact = (
            Decimal(n_in - cached) * Decimal(repr(table.input_per_million))
            + Decimal(cached) * Decimal(repr(table.cached_input_per_million))
            + Decimal(n_out) * Decimal(repr(table.output_per_million))
        ) / 1_000_000
        got = Decimal(repr(measured_cost(usage, table)))
        assert exact <= got < exact + Decimal("0.000001")


def test_measured_cost_refuses_bad_prices():
    with pytest.raises(BudgetError):
        measured_cost(ProviderUsage(input_tokens=1, output_tokens=1), replace(REAL, input_per_million=math.nan))


# ---------------------------------------------------------------------------
# Ledger scenarios (14 to 20)
# ---------------------------------------------------------------------------


def ledger(events, prices=REAL):
    return SafetyLedger.from_events(events, prices)


def test_empty_log_costs_nothing():
    led = ledger([])
    assert (led.measured, led.reserved, led.committed_upper) == (0.0, 0.0, 0.0)
    assert led.per_question == {}


def test_measured_response_counts_measured_and_releases_its_bound():
    usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": None, "reasoning_tokens": None}
    led = ledger([dispatch(1, 0.01), saved(1, 0.0035, usage=usage)])
    assert led.measured == 0.0035
    assert led.reserved == 0.0
    assert led.committed_upper == 0.0035
    assert led.anomalies == ()


def test_dispatch_without_resolution_stays_reserved():
    # 14: a crash between dispatch_started and any result.
    led = ledger([dispatch(1, 0.01)])
    assert (led.measured, led.reserved, led.committed_upper) == (0.0, 0.01, 0.01)
    assert led.attempts[0].state == "unresolved"


def test_outcome_unknown_stays_reserved():
    # 14
    led = ledger([dispatch(1, 0.01), unknown(1)])
    assert led.reserved == 0.01
    assert led.attempts[0].state == "outcome_unknown"


def test_rejected_failure_stays_reserved():
    # 14
    led = ledger([dispatch(1, 0.01), failed(1, "rejected")])
    assert led.reserved == 0.01
    assert led.measured == 0.0


def test_not_sent_failure_reserves_nothing():
    # 15
    led = ledger([dispatch(1, 0.01), failed(1, "not_sent")])
    assert led.reserved == 0.0
    assert led.committed_upper == 0.0
    assert led.attempts[0].state == "not_sent"


def test_saved_response_with_missing_usage_reserves_the_bound_and_invents_nothing():
    # 14
    led = ledger([dispatch(1, 0.01), saved(1, None, usage={"input_tokens": None, "output_tokens": None})])
    assert led.measured == 0.0
    assert led.reserved == 0.01
    assert led.attempts[0].state == "response_without_measured_cost"


def test_reconciliation_does_not_release_reserved_cost():
    # 16
    before = ledger([dispatch(1, 0.01), unknown(1)])
    for resolution in ("allow_new_attempt", "mark_failed"):
        after = ledger([dispatch(1, 0.01), unknown(1), reconciled(1, resolution)])
        assert after.reserved == before.reserved == 0.01
        assert after.committed_upper == before.committed_upper
        assert after.attempts[0].reconciled == resolution
        assert after.attempts[0].state == "outcome_unknown"


def test_event_mix_across_questions_and_attempts():
    events = [
        {"event": "invocation_started", "safety_ceiling": {"amount": 1.0, "currency": "USD", "simulated": False}},
        dispatch(1, 0.01, "q1"),
        failed(1, "not_sent", "q1"),  # 0
        dispatch(2, 0.01, "q1"),
        failed(2, "rejected", "q1"),  # 0.01 reserved
        dispatch(3, 0.01, "q1"),
        saved(3, 0.004, "q1"),  # 0.004 measured
        dispatch(1, 0.02, "q2"),
        unknown(1, "q2"),  # 0.02 reserved
        dispatch(1, 0.03, "q3"),  # 0.03 reserved, unresolved
        {"event": "invocation_ended", "reason": "interrupted"},
    ]
    led = ledger(events)
    assert led.measured == 0.004
    assert led.reserved == pytest.approx(0.06)
    assert led.committed_upper == pytest.approx(0.064)
    assert led.measured_usd == led.measured and led.reserved_usd == led.reserved
    assert led.committed_upper_usd == led.committed_upper
    q1, q2, q3 = (led.per_question[q] for q in ("q1", "q2", "q3"))
    assert (q1.attempts, q1.measured, q1.reserved) == (3, 0.004, 0.01)
    assert (q2.attempts, q2.measured, q2.reserved) == (1, 0.0, 0.02)
    assert (q3.attempts, q3.measured, q3.reserved) == (1, 0.0, 0.03)
    assert sum(a.measured_micro + a.reserved_micro for a in led.attempts) == led.committed_upper_micro
    assert led.as_dict()["per_question"]["q2"]["reserved"] == 0.02


@pytest.mark.parametrize(
    "events",
    [
        [saved(1, 0.1)],  # response for an attempt never dispatched
        [failed(1, "rejected")],
        [unknown(1)],
        [reconciled(1)],
        [dispatch(1, 0.01), dispatch(1, 0.01)],  # same attempt dispatched twice
        [dispatch(1, 0.01), saved(1, 0.1), saved(1, 0.1)],  # resolved twice
        [dispatch(1, 0.01), failed(1, "rejected"), saved(1, 0.1)],
        [dispatch(1, 0.01), unknown(1), failed(1, "not_sent")],  # would release a reserved attempt
        [dispatch(1, 0.01), failed(1, "unknown")],  # attempt_failed takes not_sent or rejected only
        [dispatch(1, 0.01), failed(1, "other")],
        [{"event": "surprise"}],
        [dispatch(1, 0.01), reconciled(1, "something_else")],
        [dispatch(1, None)],  # type: ignore[arg-type]
        [dispatch(1, math.nan)],
        [dispatch(1, -0.01)],
        [dispatch(1, math.inf)],
        [dispatch(1, 0.01), saved(1, -0.5)],
        [dispatch(1, 0.01), saved(1, math.nan)],
        [{"event": "dispatch_started", "cost_upper_bound": 0.01}],  # no attempt_id
        ["not a mapping"],
    ],
)
def test_corrupt_logs_are_refused(events):
    # 17
    with pytest.raises(LedgerError):
        ledger(events)


def test_ledger_error_is_a_budget_error_and_a_value_error():
    assert issubclass(LedgerError, BudgetError) and issubclass(BudgetError, ValueError)


def test_simulated_table_refused_against_a_real_ceiling_and_the_reverse():
    # 18
    with pytest.raises(BudgetError):
        ledger([]).can_reserve(0.01, 1.0, ceiling_simulated=True)
    with pytest.raises(BudgetError):
        ledger([], SIM).can_reserve(0.01, 1.0, ceiling_simulated=False)
    assert ledger([], SIM).can_reserve(0.01, 1.0, ceiling_simulated=True)
    assert ledger([], REAL).can_reserve(0.01, 1.0, ceiling_simulated=False)
    with pytest.raises(BudgetError):
        ledger([]).can_reserve(0.01, 1.0, ceiling_simulated=0)  # type: ignore[arg-type]


def test_currency_mismatch_refused():
    # 18
    with pytest.raises(BudgetError):
        ledger([]).can_reserve(0.01, 1.0, ceiling_simulated=False, ceiling_currency="EUR")
    assert ledger([]).can_reserve(0.01, 1.0, ceiling_simulated=False, ceiling_currency="USD")


def test_invocation_started_must_agree_with_the_price_table():
    # 18
    def started(simulated, currency="USD"):
        return {"event": "invocation_started", "safety_ceiling": {"amount": 1, "currency": currency, "simulated": simulated}}

    assert ledger([started(False)]).simulated is False
    with pytest.raises(LedgerError):
        ledger([started(True)])
    with pytest.raises(LedgerError):
        ledger([started(True)], REAL)
    with pytest.raises(LedgerError):
        ledger([started(False, "EUR")])
    assert ledger([started(True, "SIM")], SIM).simulated is True
    with pytest.raises(LedgerError):
        ledger([started(False, "SIM")], SIM)


def test_bad_price_table_refused_when_building_the_ledger():
    with pytest.raises(BudgetError):
        ledger([], replace(REAL, output_per_million=None))


def test_dispatch_is_refused_before_the_ceiling_would_be_exceeded():
    # 19: simulate a driver. Each attempt reserves 0.4 and resolves with measured cost 0.3 or stays reserved.
    ceiling = 1.0
    events: list[dict] = []
    sent = []
    for n in range(1, 8):
        led = ledger(events, SIM)
        if not led.can_reserve(0.4, ceiling, ceiling_simulated=True):
            break
        events.append(dispatch(n, 0.4, f"q{n}"))
        sent.append(n)
        if n == 1:
            events.append(saved(n, 0.3, f"q{n}"))
        else:
            events.append(unknown(n, f"q{n}"))  # unknown outcomes stay at 0.4
        # The ledger never exceeds the ceiling after a permitted dispatch.
        assert ledger(events, SIM).committed_upper <= ceiling
    # committed: 0.3, then 0.7, then 1.1 would not fit. Two attempts after the first are refused.
    assert sent == [1, 2]
    final = ledger(events, SIM)
    assert final.committed_upper == pytest.approx(0.7)
    assert not final.can_reserve(0.4, ceiling, ceiling_simulated=True)
    assert final.can_reserve(0.3, ceiling, ceiling_simulated=True)


def test_exact_boundary_equality_is_allowed_and_one_micro_unit_over_is_not():
    # 19
    led = ledger([dispatch(1, 0.7), unknown(1)])
    assert led.can_reserve(0.3, 1.0, ceiling_simulated=False)  # 0.7 + 0.3 == 1.0
    assert not led.can_reserve(0.300001, 1.0, ceiling_simulated=False)
    assert led.can_reserve(0.299999, 1.0, ceiling_simulated=False)
    # A float sum would give 0.1 + 0.2 == 0.30000000000000004 > 0.3.
    assert 0.1 + 0.2 > 0.3
    tenth = ledger([dispatch(1, 0.1), unknown(1)])
    assert tenth.can_reserve(0.2, 0.3, ceiling_simulated=False)


def test_boundary_is_independent_of_event_order():
    costs = [0.1, 0.2, 0.3, 0.05, 0.15]
    rng = random.Random(1)
    totals = set()
    for _ in range(20):
        rng.shuffle(costs)
        led = ledger([e for i, c in enumerate(costs) for e in (dispatch(1, c, f"q{i}"), unknown(1, f"q{i}"))])
        totals.add(led.committed_upper_micro)
        assert led.can_reserve(0.2, 1.0, ceiling_simulated=False)
        assert not led.can_reserve(0.200001, 1.0, ceiling_simulated=False)
    assert totals == {800_000}


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, None, True, "1"])
def test_can_reserve_refuses_bad_amounts(bad):
    led = ledger([])
    with pytest.raises(BudgetError):
        led.can_reserve(bad, 1.0, ceiling_simulated=False)
    with pytest.raises(BudgetError):
        led.can_reserve(0.1, bad, ceiling_simulated=False)


def test_rounding_costs_round_up_and_ceiling_rounds_down():
    # Rounding rule: attempt costs up, ceiling down.
    led = ledger([dispatch(1, 0.0000001), unknown(1)])  # 0.1 micro-unit rounds up to 1
    assert led.reserved_micro == 1
    assert led.reserved == 0.000001
    assert not ledger([]).can_reserve(0.0000001, 0.0000009, ceiling_simulated=False)  # 1 micro > floor(0.9)=0
    assert ledger([]).can_reserve(0.0000001, 0.0000011, ceiling_simulated=False)  # 1 micro <= floor(1.1)=1
    assert ledger([]).can_reserve(0.0, 0.0, ceiling_simulated=False)


def test_float_views_round_trip_through_micro_units():
    rng = random.Random(9)
    for _ in range(2000):
        micro = rng.randrange(0, 999_999_999)
        led = ledger([dispatch(1, micro_to_amount(micro)), unknown(1)])
        assert led.reserved_micro == micro


def test_recorded_measured_cost_that_disagrees_with_usage_is_flagged_and_counted_high():
    # 20
    usage = {"input_tokens": 1000, "output_tokens": 100}
    real_cost = measured_cost(ProviderUsage(**usage), REAL)  # 0.0035
    lower = ledger([dispatch(1, 0.01), saved(1, 0.001, usage=usage)])
    assert lower.measured == real_cost
    assert any("differs" in note for note in lower.anomalies)
    higher = ledger([dispatch(1, 0.01), saved(1, 0.009, usage=usage)])
    assert higher.measured == 0.009
    assert any("differs" in note for note in higher.anomalies)


def test_measured_cost_above_its_upper_bound_is_flagged_and_counted():
    # 20
    usage = {"input_tokens": 1000, "output_tokens": 100}
    led = ledger([dispatch(1, 0.001), saved(1, 0.0035, usage=usage)])
    assert led.measured == 0.0035
    assert any("exceeds" in note for note in led.anomalies)


def test_measured_cost_recorded_without_reproducible_usage_is_flagged():
    led = ledger([dispatch(1, 0.01), saved(1, 0.002, usage={})])
    assert led.measured == 0.002
    assert any("cannot reproduce" in note for note in led.anomalies)


def test_ledger_is_immutable():
    led = ledger([])
    with pytest.raises(Exception):
        led.measured_micro = 5  # type: ignore[misc]


def test_ledger_accepts_a_generator():
    led = ledger(e for e in [dispatch(1, 0.01), unknown(1)])
    assert led.reserved == 0.01


def test_end_to_end_bound_covers_measured_cost():
    # The per-attempt bound from the input bound is at least the measured cost for any usage within the limits.
    messages = msgs("system rules", "question and evidence " * 40)
    bound_in = input_token_upper_bound(messages)
    upper = request_cost_upper_bound(bound_in, 128, REAL)
    usage = ProviderUsage(input_tokens=bound_in, output_tokens=128, cached_input_tokens=0)
    assert measured_cost(usage, REAL) <= upper


def test_a_reopened_question_keeps_its_earlier_reserved_cost() -> None:
    """Reopening a failed question releases nothing; later attempts add their own bounds."""
    events = [
        dispatch(1, 0.5),
        failed(1, "rejected"),
        {"event": "question_reopened", "question_id": "q1", "request_id": "q1-r", "attempts_before": 1, "note": "fixed"},
        dispatch(2, 0.5),
        saved(2, 0.1),
    ]
    ledger = SafetyLedger.from_events(events, REAL)
    assert ledger.reserved == 0.5
    assert ledger.measured == 0.1


# ---------------------------------------------------------------------------
# Service tier: a response whose tier is not verified is not counted as measured cost
# ---------------------------------------------------------------------------

TIERED = replace(REAL, service_tier="default")
USAGE = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": None, "reasoning_tokens": None}


def saved_in_tier(attempt: int, tier: object, question: str = "q1", cost: float | None = 0.0035) -> dict:
    event = saved(attempt, cost, question=question, usage=USAGE)
    event["returned_service_tier"] = tier
    return event


def test_a_standard_tier_response_counts_as_measured_cost():
    led = ledger([dispatch(1, 0.01), saved_in_tier(1, "default")], TIERED)
    assert (led.measured, led.reserved) == (0.0035, 0.0)
    assert led.unverified_tier == {}


@pytest.mark.parametrize("tier", ["priority", "flex", "scale", "auto", "fast", "", None])
def test_a_response_from_another_tier_is_kept_out_of_measured_and_stays_reserved_at_the_standard_bound(tier):
    led = ledger([dispatch(1, 0.01), saved_in_tier(1, tier)], TIERED)
    assert led.measured == 0.0
    assert led.reserved == 0.01 and led.committed_upper == 0.01
    assert led.attempts[0].state == "unverified_service_tier"
    block = led.unverified_tier
    assert block["attempts"] == 1 and block["attempt_ids"] == ["q1-r-a1"]
    assert block["usage"] == {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 100, "reasoning_tokens": 0}
    assert block["standard_rate_cost"] == 0.0035
    assert led.anomalies == ()


def test_a_response_with_no_tier_key_in_a_tier_aware_run_is_unverified():
    event = saved(1, 0.0035, usage=USAGE)
    led = ledger([dispatch(1, 0.01), event], TIERED)
    assert led.measured == 0.0 and led.unverified_tier["attempts"] == 1


def test_unverified_costs_sum_and_measured_keeps_only_verified_attempts():
    events = [
        dispatch(1, 0.01, "q1"),
        saved_in_tier(1, "default", "q1"),
        dispatch(1, 0.01, "q2"),
        saved_in_tier(1, "priority", "q2"),
        dispatch(1, 0.01, "q3"),
        saved_in_tier(1, "flex", "q3"),
    ]
    led = ledger(events, TIERED)
    assert led.measured == 0.0035 and led.reserved == 0.02 and led.committed_upper == 0.0235
    assert led.unverified_tier["attempts"] == 2 and led.unverified_tier["standard_rate_cost"] == 0.007
    assert led.unverified_tier["usage"]["input_tokens"] == 2000


def test_a_legacy_price_table_without_a_tier_is_not_checked():
    led = ledger([dispatch(1, 0.01), saved_in_tier(1, "priority")], REAL)
    assert led.measured == 0.0035 and led.unverified_tier == {}
    assert "unverified_tier" not in led.as_dict()


def test_the_ledger_dict_names_the_unverified_block_only_when_there_is_one():
    led = ledger([dispatch(1, 0.01), saved_in_tier(1, "priority")], TIERED)
    assert led.as_dict()["unverified_tier"]["attempts"] == 1
