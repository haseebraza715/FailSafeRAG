"""Conservative spending accounting for the answer-model path.

The safety ceiling stops a run before it spends more than the lead allowed. The
ledger in this module errs high. When it cannot tell what an attempt cost, it
counts the attempt at its per-attempt upper bound. It never guesses a lower
number. Nothing here reads a network, a clock or a credential.

Failure list (each item has a test in ``tests/test_request_budget.py``)
=======================================================================

Input bound
  1. A message content that is not a string (None, bytes, list of parts, number)
     would make the bound meaningless. The function raises ``TypeError``.
  2. A message that is not a mapping, or has no ``content``, raises.
  3. An empty message list raises. A request with no messages is not a request.
  4. Multi-byte text (Han script, emoji) must count bytes, not characters.
  5. The bound must be at least the real token count of byte-level BPE
     encodings (cl100k, o200k, gpt2 vocabularies).

Request cost bound
  6. A missing, negative, NaN or infinite rate raises. A price table that is
     not simulated and has a zero input or output rate raises, because a typo
     of zero would switch the ceiling off.
  7. ``max_output_tokens`` that is not a positive integer raises.
  8. An input bound that is negative, non-integer or a bool raises.
  9. Float rounding must never make the bound smaller than the exact product.

Measured cost
 10. Usage without both input and output tokens gives ``None``, never a guess.
 11. Cached tokens are discounted only when the usage reports them and the
     price table has a cached rate.
 12. Reasoning tokens are already inside the output tokens. Adding them again
     would overcharge. Usage that reports more reasoning than output tokens, or
     more cached than input tokens, contradicts that and gives ``None``.
 13. Negative, non-integer or bool counts give ``None``.

Ledger
 14. A dispatched attempt with no measured cost must stay reserved: unresolved,
     ``outcome_unknown``, ``attempt_failed`` with outcome ``rejected``, and a
     saved response whose ``measured_cost`` is null.
 15. A ``not_sent`` failure reserves nothing.
 16. ``reconciled`` must not release reserved cost.
 17. A response, failure or reconciliation for an attempt that was never
     dispatched, a second resolution of one attempt, a repeated dispatch, an
     unknown event name, or an unusable ``cost_upper_bound`` or ``measured_cost``
     means the log is corrupt. The ledger raises instead of guessing.
 18. A simulated price table must not meet a real ceiling, and the reverse. A
     currency mismatch is refused too. An ``invocation_started`` event that
     disagrees with the price table is refused.
 19. The ceiling test must be exact at the boundary and must not depend on
     float summation order.
 20. A measured cost above the attempt's own upper bound, or a recorded cost
     that disagrees with the usage, is reported in ``anomalies`` and counted
     at the higher figure.

Service tier
 21. A response whose returned service tier is not verified as Standard must not
     count as measured cost at the price table's Standard rates. The ledger keeps
     its attempt reserved at the Standard-rate upper bound and reports its usage
     and its Standard-rate cost apart, in ``unverified_tier``. That bound is not an
     upper bound on the real charge when the tier bills more. Only a price table
     that declares a ``service_tier`` is checked, so older records read as before.

Rounding rule
=============

The ledger holds every amount as an integer count of micro-units, one
millionth of the price table's currency unit. A rate is a price per million
tokens, so ``tokens * rate`` is already a micro-unit amount and no division
occurs. The steps are:

* Rates and recorded floats convert through ``Decimal(repr(x))``. That is the
  shortest decimal that round-trips the float, so ``0.1`` is one tenth and not
  ``0.1000000000000000055``.
* Each attempt cost rounds UP to a whole micro-unit.
* The ceiling rounds DOWN to a whole micro-unit.
* Sums are exact integer sums.
* A value handed back as a float is ``micro / 1_000_000``. Converting that
  float again gives the same integer for any amount below about 1e9 units.

Rounding up per attempt can overstate a run by less than one micro-unit per
attempt. For a run of 70 questions that is below 0.0002 units.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

from faar.live_contract import (
    EVENT_ATTEMPT_FAILED,
    EVENT_DISPATCH_STARTED,
    EVENT_INVOCATION_ENDED,
    EVENT_INVOCATION_STARTED,
    EVENT_OUTCOME_UNKNOWN,
    EVENT_QUESTION_REOPENED,
    EVENT_RECONCILED,
    EVENT_RESPONSE_SAVED,
    OUTCOME_NOT_SENT,
    OUTCOME_REJECTED,
    RETURNED_SERVICE_TIERS_ACCEPTED,
    PriceTable,
    ProviderUsage,
)

MICRO = 1_000_000
BYTES_PER_MESSAGE_OVERHEAD = 4
BYTES_FIXED_OVERHEAD = 16

# Attempt states in the ledger.
STATE_UNRESOLVED = "unresolved"
STATE_MEASURED = "measured"
STATE_NOT_SENT = "not_sent"
STATE_REJECTED = "rejected"
STATE_OUTCOME_UNKNOWN = "outcome_unknown"
STATE_NO_MEASURED_COST = "response_without_measured_cost"
STATE_UNVERIFIED_TIER = "unverified_service_tier"
RESERVED_STATES = (STATE_UNRESOLVED, STATE_REJECTED, STATE_OUTCOME_UNKNOWN, STATE_NO_MEASURED_COST, STATE_UNVERIFIED_TIER)

UNVERIFIED_TIER_NOTE = (
    "These responses did not report the Standard service tier, so their usage is not counted as measured cost. "
    "standard_rate_cost prices that usage at the Standard rates of the price table. It is not an actual-cost claim: "
    "a tier that bills more (the pricing page lists Fast gpt-4o at 1.7 times Standard) charged more. Each attempt "
    "stays reserved at its Standard-rate upper bound, and that bound is not an upper bound on the real charge. "
    "Check the provider's usage export."
)
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")

RECONCILE_RESOLUTIONS = ("allow_new_attempt", "mark_failed")


class BudgetError(ValueError):
    """An input the accounting refuses to price."""


class LedgerError(BudgetError):
    """The event log is inconsistent, so the ledger cannot say what was spent."""


# ---------------------------------------------------------------------------
# Number handling
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decimal(value: Any, name: str) -> Decimal:
    """Exact decimal for a finite real number. Refuses bool, NaN, infinity and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{name} must be a finite number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise BudgetError(f"{name} must be finite, got {value!r}")
    return Decimal(value) if _is_int(value) else Decimal(repr(value))


def _ceil_micro(amount: Decimal) -> int:
    """Whole micro-units at or above ``amount`` (already expressed in micro-units)."""
    return int(amount.to_integral_value(rounding=ROUND_CEILING))


def _amount_to_micro(value: Any, name: str, *, up: bool) -> int:
    """Convert a currency amount to micro-units, rounding up (a cost) or down (a ceiling)."""
    micro = _decimal(value, name) * MICRO
    if micro < 0:
        raise BudgetError(f"{name} must not be negative, got {value!r}")
    rounding = ROUND_CEILING if up else ROUND_FLOOR
    return int(micro.to_integral_value(rounding=rounding))


def micro_to_amount(micro: int) -> float:
    """Float view of a micro-unit count, for records and display only."""
    return micro / MICRO


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def check_prices(prices: PriceTable) -> None:
    """Raise ``BudgetError`` unless the table can price a request safely."""
    if not isinstance(prices, PriceTable):
        raise BudgetError("prices must be a PriceTable")
    if not isinstance(prices.currency, str) or not prices.currency:
        raise BudgetError("price table needs a currency")
    if not isinstance(prices.simulated, bool):
        raise BudgetError("price table 'simulated' must be a bool")
    for name in ("input_per_million", "output_per_million"):
        rate = _decimal(getattr(prices, name), name)
        if rate < 0:
            raise BudgetError(f"{name} must not be negative, got {rate}")
        if rate == 0 and not prices.simulated:
            raise BudgetError(f"{name} is zero in a table that is not simulated")
    cached = prices.cached_input_per_million
    if cached is not None and _decimal(cached, "cached_input_per_million") < 0:
        raise BudgetError(f"cached_input_per_million must not be negative, got {cached!r}")


def _rates(prices: PriceTable) -> tuple[Decimal, Decimal | None, Decimal]:
    check_prices(prices)
    cached = prices.cached_input_per_million
    return (
        _decimal(prices.input_per_million, "input_per_million"),
        None if cached is None else _decimal(cached, "cached_input_per_million"),
        _decimal(prices.output_per_million, "output_per_million"),
    )


# ---------------------------------------------------------------------------
# Input bound
# ---------------------------------------------------------------------------


def input_token_upper_bound(messages: Sequence[Mapping[str, str]]) -> int:
    """Upper bound on the input tokens of a chat request.

    The bound is the UTF-8 byte length of every message content, plus 4 per
    message, plus 16.

    Why it holds. A byte-level BPE tokeniser (the GPT-2, cl100k and o200k
    families) starts from single bytes and only merges them, and every token
    stands for at least one byte. So a text of N bytes never yields more than N
    tokens. The 4 per message covers the role and separator tokens the chat
    format wraps around each content. The 16 covers the tokens that prime the
    reply and leaves room for format changes.

    When it does not hold.

    * A tokeniser that is not byte-level. One whose normaliser can lengthen the
      text before tokenising (some Unicode compatibility mappings turn one
      3-byte character into four 3-byte characters) can produce more tokens
      than input bytes.
    * A model that adds many special tokens per message or per request.
    * Non-text content such as images. This function refuses those: content
      must be a string.

    The run configuration must therefore declare ``tokenizer_bound:
    "utf8-bytes"`` for the model in use (contract rule 7). The bound also
    overstates the count by a large factor for English text (about 4 bytes per
    token), which makes the ceiling check cautious.
    """
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence of mappings")
    if len(messages) == 0:
        raise ValueError("messages is empty")
    total = BYTES_FIXED_OVERHEAD
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or "content" not in message:
            raise TypeError(f"message {index} must be a mapping with a 'content' key")
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError(f"message {index} content must be a string, got {type(content).__name__}")
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError as exc:  # a lone surrogate cannot be sent as UTF-8 either
            raise ValueError(f"message {index} content is not valid Unicode text: {exc}") from exc
        total += len(encoded) + BYTES_PER_MESSAGE_OVERHEAD
    return total


# ---------------------------------------------------------------------------
# Cost of one request
# ---------------------------------------------------------------------------


def request_cost_upper_bound_micro(input_upper: int, max_output_tokens: int, prices: PriceTable) -> int:
    """Upper bound of one attempt in micro-units, rounded up."""
    if not _is_int(input_upper) or input_upper < 0:
        raise BudgetError(f"input_upper must be a non-negative integer, got {input_upper!r}")
    if not _is_int(max_output_tokens) or max_output_tokens <= 0:
        raise BudgetError(f"max_output_tokens must be a positive integer, got {max_output_tokens!r}")
    input_rate, cached_rate, output_rate = _rates(prices)
    # Every input token is priced at the dearest input rate the table lists.
    # This is the full uncached rate unless the table has a cached rate above it.
    worst_input_rate = input_rate if cached_rate is None else max(input_rate, cached_rate)
    return _ceil_micro(Decimal(input_upper) * worst_input_rate + Decimal(max_output_tokens) * output_rate)


def request_cost_upper_bound(input_upper: int, max_output_tokens: int, prices: PriceTable) -> float:
    """Largest cost one attempt can have, in the price table's currency.

    Input tokens are priced at the full uncached input rate (or the cached rate
    when a table lists one above it), and ``max_output_tokens`` at the output
    rate. For a reasoning model ``max_output_tokens`` must be the cap that
    includes reasoning tokens, because those are billed as output. See
    ``measured_cost``.
    """
    return micro_to_amount(request_cost_upper_bound_micro(input_upper, max_output_tokens, prices))


def measured_cost_micro(usage: ProviderUsage, prices: PriceTable) -> int | None:
    """Cost of a response in micro-units, rounded up, or ``None`` when usage cannot be trusted."""
    input_rate, cached_rate, output_rate = _rates(prices)
    counts = (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, usage.reasoning_tokens)
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    if any(count is not None and (not _is_int(count) or count < 0) for count in counts):
        return None
    if usage.cached_input_tokens is not None and usage.cached_input_tokens > usage.input_tokens:
        return None
    if usage.reasoning_tokens is not None and usage.reasoning_tokens > usage.output_tokens:
        return None
    cached = usage.cached_input_tokens if (usage.cached_input_tokens is not None and cached_rate is not None) else 0
    fresh = usage.input_tokens - cached
    total = Decimal(fresh) * input_rate + Decimal(usage.output_tokens) * output_rate
    if cached:
        total += Decimal(cached) * cached_rate  # type: ignore[operator]
    return _ceil_micro(total)


def measured_cost(usage: ProviderUsage, prices: PriceTable) -> float | None:
    """Cost of a response from the usage the provider reported.

    Returns ``None`` unless both input and output tokens were reported, and
    also when the counts contradict each other. The caller then reserves the
    attempt's upper bound.

    Cached input tokens are billed at the cached rate only when the usage
    reports ``cached_input_tokens`` and the table has a cached rate. Otherwise
    all input is billed at the full rate.

    Reasoning tokens are not added. OpenAI counts them inside the output
    tokens and bills them as output tokens. The reasoning guide says they "are
    billed as output tokens" and shows ``reasoning_tokens`` inside
    ``output_tokens_details``, a breakdown of ``output_tokens``
    (https://developers.openai.com/api/docs/guides/reasoning, read 2026-09-29).
    Chat Completions reports the same split as
    ``completion_tokens_details.reasoning_tokens`` inside ``completion_tokens``.
    Adding ``reasoning_tokens`` to the output count would bill them twice.
    """
    micro = measured_cost_micro(usage, prices)
    return None if micro is None else micro_to_amount(micro)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptAccount:
    """What one dispatched attempt counts for."""

    attempt_id: str
    question_id: str | None
    state: str
    upper_micro: int
    measured_micro: int
    reserved_micro: int
    reconciled: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "question_id": self.question_id,
            "state": self.state,
            "cost_upper_bound": micro_to_amount(self.upper_micro),
            "measured": micro_to_amount(self.measured_micro),
            "reserved": micro_to_amount(self.reserved_micro),
            "reconciled": self.reconciled,
        }


@dataclass(frozen=True)
class QuestionAccount:
    """Spending of one question over all its attempts."""

    question_id: str | None
    attempts: int
    measured_micro: int
    reserved_micro: int

    @property
    def measured(self) -> float:
        return micro_to_amount(self.measured_micro)

    @property
    def reserved(self) -> float:
        return micro_to_amount(self.reserved_micro)

    @property
    def committed_upper(self) -> float:
        return micro_to_amount(self.measured_micro + self.reserved_micro)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "attempts": self.attempts,
            "measured": self.measured,
            "reserved": self.reserved,
            "committed_upper": self.committed_upper,
        }


@dataclass(frozen=True)
class SafetyLedger:
    """Upper bound on what a run has spent, rebuilt from its event log.

    ``committed_upper`` is ``measured + reserved``. Use ``can_reserve`` before
    each dispatch. Build a new ledger from the events after every event you
    append, or keep one in step with them.
    """

    currency: str
    simulated: bool
    measured_micro: int
    reserved_micro: int
    attempts: tuple[AttemptAccount, ...] = ()
    per_question: Mapping[str | None, QuestionAccount] = field(default_factory=dict)
    anomalies: tuple[str, ...] = ()
    # Responses whose returned service tier is not verified as Standard. Empty when there are none.
    unverified_tier: Mapping[str, Any] = field(default_factory=dict)

    # -- amounts (floats are views of the integer micro-unit fields) --------

    @property
    def measured(self) -> float:
        return micro_to_amount(self.measured_micro)

    @property
    def reserved(self) -> float:
        return micro_to_amount(self.reserved_micro)

    @property
    def committed_upper(self) -> float:
        return micro_to_amount(self.committed_upper_micro)

    @property
    def committed_upper_micro(self) -> int:
        return self.measured_micro + self.reserved_micro

    # Names used by the contract. They carry the price table's currency, which is USD only for USD tables.
    @property
    def measured_usd(self) -> float:
        return self.measured

    @property
    def reserved_usd(self) -> float:
        return self.reserved

    @property
    def committed_upper_usd(self) -> float:
        return self.committed_upper

    def can_reserve(
        self,
        upper: float,
        ceiling: float,
        *,
        ceiling_simulated: bool,
        ceiling_currency: str | None = None,
    ) -> bool:
        """True when ``committed_upper + upper <= ceiling``, compared exactly in micro-units.

        ``upper`` rounds up and ``ceiling`` rounds down to a whole micro-unit.
        ``ceiling_simulated`` says whether the ceiling is in simulated currency.
        It must equal the price table's flag, so a simulated table never meets a
        real ceiling and the reverse. Pass ``ceiling_currency`` to check the
        currency as well.
        """
        if not isinstance(ceiling_simulated, bool):
            raise BudgetError("ceiling_simulated must be a bool")
        if ceiling_simulated != self.simulated:
            raise BudgetError(
                f"ceiling simulated={ceiling_simulated} does not match price table simulated={self.simulated}"
            )
        if ceiling_currency is not None and ceiling_currency != self.currency:
            raise BudgetError(f"ceiling currency {ceiling_currency!r} does not match price table {self.currency!r}")
        upper_micro = _amount_to_micro(upper, "upper", up=True)
        ceiling_micro = _amount_to_micro(ceiling, "ceiling", up=False)
        return self.committed_upper_micro + upper_micro <= ceiling_micro

    def as_dict(self) -> dict[str, Any]:
        data = {
            "currency": self.currency,
            "simulated": self.simulated,
            "measured": self.measured,
            "reserved": self.reserved,
            "committed_upper": self.committed_upper,
            "rounding": "micro-units, attempt costs up, ceiling down",
            "per_question": {str(key): account.as_dict() for key, account in self.per_question.items()},
            "anomalies": list(self.anomalies),
        }
        if self.unverified_tier:
            data["unverified_tier"] = dict(self.unverified_tier)
        return data

    @classmethod
    def from_events(cls, events: Iterable[Mapping[str, Any]], prices: PriceTable) -> SafetyLedger:
        """Rebuild the ledger from the contract's event records, in log order.

        * ``measured``: the ``measured_cost`` of every ``response_saved`` event
          that has one.
        * ``reserved``: the ``cost_upper_bound`` of every dispatched attempt
          without a measured cost. That covers an attempt with no later event,
          ``outcome_unknown``, ``attempt_failed`` with outcome ``rejected``,
          ``response_saved`` with ``measured_cost`` null, and ``response_saved``
          whose ``returned_service_tier`` is not Standard when the price table
          declares a service tier (see ``unverified_tier``).
        * A ``not_sent`` failure costs nothing.
        * ``reconciled`` records the resolution and releases nothing.

        Raises ``LedgerError`` on an inconsistent log and ``BudgetError`` on
        an unusable price table.
        """
        check_prices(prices)
        state: dict[str, _Attempt] = {}
        for position, event in enumerate(events, start=1):
            _apply_event(state, event, prices, position)
        return _build(state, prices)


@dataclass
class _Attempt:
    question_id: str | None
    upper_micro: int
    state: str = STATE_UNRESOLVED
    measured_micro: int = 0
    reconciled: str | None = None
    notes: list[str] = field(default_factory=list)
    usage: Mapping[str, Any] | None = None
    standard_micro: int | None = None


def _attempt_id(event: Mapping[str, Any], position: int) -> str:
    attempt_id = event.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise LedgerError(f"event {position} ({event.get('event')}) has no attempt_id")
    return attempt_id


def _resolve(state: Mapping[str, _Attempt], event: Mapping[str, Any], position: int) -> tuple[str, _Attempt]:
    attempt_id = _attempt_id(event, position)
    attempt = state.get(attempt_id)
    if attempt is None:
        raise LedgerError(f"event {position} ({event['event']}) refers to {attempt_id}, which was never dispatched")
    if attempt.state != STATE_UNRESOLVED:
        raise LedgerError(f"event {position} ({event['event']}) resolves {attempt_id} a second time")
    return attempt_id, attempt


def _apply_event(state: dict[str, _Attempt], event: Mapping[str, Any], prices: PriceTable, position: int) -> None:
    if not isinstance(event, Mapping):
        raise LedgerError(f"event {position} is not a mapping")
    name = event.get("event")
    if name == EVENT_INVOCATION_ENDED:
        return
    if name == EVENT_INVOCATION_STARTED:
        ceiling = event.get("safety_ceiling")
        if isinstance(ceiling, Mapping):
            if ceiling.get("simulated") is not prices.simulated:
                raise LedgerError(
                    f"event {position}: ceiling simulated={ceiling.get('simulated')!r} "
                    f"does not match price table simulated={prices.simulated}"
                )
            if ceiling.get("currency") not in (None, prices.currency):
                raise LedgerError(
                    f"event {position}: ceiling currency {ceiling.get('currency')!r} "
                    f"does not match price table {prices.currency!r}"
                )
        return
    if name == EVENT_DISPATCH_STARTED:
        attempt_id = _attempt_id(event, position)
        if attempt_id in state:
            raise LedgerError(f"event {position}: {attempt_id} was dispatched twice")
        try:
            upper = _amount_to_micro(event.get("cost_upper_bound"), "cost_upper_bound", up=True)
        except BudgetError as exc:
            raise LedgerError(f"event {position}: {exc}") from exc
        state[attempt_id] = _Attempt(question_id=event.get("question_id"), upper_micro=upper)
        return
    if name == EVENT_RESPONSE_SAVED:
        attempt_id, attempt = _resolve(state, event, position)
        recorded = event.get("measured_cost")
        if prices.service_tier is not None and event.get("returned_service_tier") not in RETURNED_SERVICE_TIERS_ACCEPTED:
            # The rates apply to the Standard tier only. Keep the usage, count nothing as measured, stay reserved.
            attempt.state = STATE_UNVERIFIED_TIER
            attempt.usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
            try:
                recorded_micro = None if recorded is None else _amount_to_micro(recorded, "measured_cost", up=True)
            except BudgetError as exc:
                raise LedgerError(f"event {position}: {exc}") from exc
            recomputed_micro = _recomputed(event, prices)
            known = [m for m in (recorded_micro, recomputed_micro) if m is not None]
            attempt.standard_micro = max(known) if known else None
            return
        if recorded is None:
            attempt.state = STATE_NO_MEASURED_COST
            return
        try:
            micro = _amount_to_micro(recorded, "measured_cost", up=True)
        except BudgetError as exc:
            raise LedgerError(f"event {position}: {exc}") from exc
        attempt.state = STATE_MEASURED
        recomputed = _recomputed(event, prices)
        if recomputed is None:
            attempt.notes.append(f"{attempt_id}: measured_cost recorded but usage cannot reproduce it")
        elif recomputed != micro:
            attempt.notes.append(f"{attempt_id}: recorded measured_cost differs from the cost of its usage")
        attempt.measured_micro = max(micro, recomputed or 0)
        return
    if name == EVENT_ATTEMPT_FAILED:
        _, attempt = _resolve(state, event, position)
        outcome = event.get("outcome")
        if outcome == OUTCOME_NOT_SENT:
            attempt.state = STATE_NOT_SENT
        elif outcome == OUTCOME_REJECTED:
            attempt.state = STATE_REJECTED
        else:
            raise LedgerError(f"event {position}: attempt_failed outcome {outcome!r} is not not_sent or rejected")
        return
    if name == EVENT_OUTCOME_UNKNOWN:
        _, attempt = _resolve(state, event, position)
        attempt.state = STATE_OUTCOME_UNKNOWN
        return
    if name == EVENT_RECONCILED:
        attempt_id = _attempt_id(event, position)
        if attempt_id not in state:
            raise LedgerError(f"event {position}: reconciled {attempt_id}, which was never dispatched")
        resolution = event.get("resolution")
        if resolution not in RECONCILE_RESOLUTIONS:
            raise LedgerError(f"event {position}: unknown reconcile resolution {resolution!r}")
        state[attempt_id].reconciled = resolution
        return
    if name == EVENT_QUESTION_REOPENED:
        # Reopening a failed question changes no cost: earlier attempts keep their measured or
        # reserved amounts, and later attempts are priced when they are dispatched.
        return
    raise LedgerError(f"event {position}: unknown event {name!r}")


def _recomputed(event: Mapping[str, Any], prices: PriceTable) -> int | None:
    """Cost recomputed from the event's usage, or None when the usage cannot give one."""
    usage = event.get("usage")
    if not isinstance(usage, Mapping):
        return None
    keys = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
    return measured_cost_micro(ProviderUsage(**{key: usage.get(key) for key in keys}), prices)


def _build(state: Mapping[str, _Attempt], prices: PriceTable) -> SafetyLedger:
    attempts: list[AttemptAccount] = []
    per_question: dict[str | None, list[int]] = {}
    anomalies: list[str] = []
    measured_total = 0
    reserved_total = 0
    unverified_ids: list[str] = []
    for attempt_id, attempt in state.items():
        if attempt.state == STATE_UNVERIFIED_TIER:
            unverified_ids.append(attempt_id)
        reserved = attempt.upper_micro if attempt.state in RESERVED_STATES else 0
        measured = attempt.measured_micro if attempt.state == STATE_MEASURED else 0
        anomalies.extend(attempt.notes)
        if attempt.state == STATE_MEASURED and measured > attempt.upper_micro:
            anomalies.append(f"{attempt_id}: measured cost exceeds its upper bound")
        attempts.append(
            AttemptAccount(
                attempt_id=attempt_id,
                question_id=attempt.question_id,
                state=attempt.state,
                upper_micro=attempt.upper_micro,
                measured_micro=measured,
                reserved_micro=reserved,
                reconciled=attempt.reconciled,
            )
        )
        totals = per_question.setdefault(attempt.question_id, [0, 0, 0])
        totals[0] += 1
        totals[1] += measured
        totals[2] += reserved
        measured_total += measured
        reserved_total += reserved
    return SafetyLedger(
        currency=prices.currency,
        simulated=prices.simulated,
        measured_micro=measured_total,
        reserved_micro=reserved_total,
        attempts=tuple(attempts),
        per_question={
            key: QuestionAccount(question_id=key, attempts=n, measured_micro=m, reserved_micro=r)
            for key, (n, m, r) in per_question.items()
        },
        anomalies=tuple(anomalies),
        unverified_tier=_unverified_block(state, unverified_ids),
    )


def _unverified_block(state: Mapping[str, _Attempt], attempt_ids: Sequence[str]) -> dict[str, Any]:
    """Usage and Standard-rate cost of the attempts whose returned tier is not verified. Empty when there are none."""
    if not attempt_ids:
        return {}
    usage = dict.fromkeys(USAGE_KEYS, 0)
    priced = [state[attempt_id].standard_micro for attempt_id in attempt_ids]
    for attempt_id in attempt_ids:
        reported = state[attempt_id].usage or {}
        for key in USAGE_KEYS:
            value = reported.get(key)
            if _is_int(value):
                usage[key] += value
    return {
        "attempts": len(attempt_ids),
        "attempt_ids": list(attempt_ids),
        "usage": usage,
        "standard_rate_cost": micro_to_amount(sum(m for m in priced if m is not None)),
        "unpriceable_attempts": sum(1 for m in priced if m is None),
        "note": UNVERIFIED_TIER_NOTE,
    }
