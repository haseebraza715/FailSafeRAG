"""Shared types for the answer-model execution path.

The prompt builder, the providers, the spending ledger and the run driver all
exchange these objects. Keeping them in one module stops each part from
inventing its own record format. This module defines data only: it imports no
provider SDK and opens no connection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CONTRACT_VERSION = "1"

# Execution modes. A fake run is an engineering check. A live run is a
# development pilot and needs explicit enablement (see faar.live_runner).
MODE_FAKE = "fake"
MODE_LIVE = "live"
MODES = (MODE_FAKE, MODE_LIVE)

# Events appended to attempts.jsonl, one JSON object per line.
EVENT_INVOCATION_STARTED = "invocation_started"
EVENT_DISPATCH_STARTED = "dispatch_started"
EVENT_RESPONSE_SAVED = "response_saved"
EVENT_ATTEMPT_FAILED = "attempt_failed"
EVENT_OUTCOME_UNKNOWN = "outcome_unknown"
EVENT_RECONCILED = "reconciled"
EVENT_INVOCATION_ENDED = "invocation_ended"
# A person reopens an execution_failed question after fixing its cause. The question
# gets up to max_attempts further attempts; earlier attempts and their costs stay.
EVENT_QUESTION_REOPENED = "question_reopened"
EVENTS = (
    EVENT_INVOCATION_STARTED,
    EVENT_DISPATCH_STARTED,
    EVENT_RESPONSE_SAVED,
    EVENT_ATTEMPT_FAILED,
    EVENT_OUTCOME_UNKNOWN,
    EVENT_RECONCILED,
    EVENT_INVOCATION_ENDED,
    EVENT_QUESTION_REOPENED,
)

# Request parameters a provider config may set. Everything else is refused, because an
# unknown parameter can change billing (service tiers, tools, extra modalities) outside
# the cost bound.
ALLOWED_OPENAI_PARAMS = ("temperature", "top_p", "seed", "stop", "presence_penalty", "frequency_penalty")

# Provider-side storage policy. The ordinary live baseline sends `store=false` on every request and does not
# rely on the account default (OpenAI documents Chat Completions as stored by default for new accounts).
# `store=false` does not mean zero retention: OpenAI keeps abuse-monitoring logs for up to 30 days unless the
# organization has an approved Zero Data Retention or Modified Abuse Monitoring control. Nor does it settle
# whether the dataset terms allow sending the documents at all.
STORAGE_DISABLED = "disabled"  # sends store=false
STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP = "enabled_for_attempt_lookup"  # sends store=true plus metadata; needs explicit authorization
STORAGE_POLICIES = (STORAGE_DISABLED, STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP)

# Service tier. The baseline requests the Standard tier explicitly, so a project-level setting (for example
# Fast mode) cannot change the price. The Chat Completions reference (read 2026-09-30) documents the request
# value "default" as "standard pricing and performance for the selected model", and says the response echoes
# the tier actually used when the request sets one. openai-python 1.68.2 types the request field as
# Literal["auto", "default"].
SERVICE_TIER_STANDARD = "default"
# A saved response whose service_tier is anything but SERVICE_TIER_STANDARD is a safety violation, including
# a missing or unrecognised value: the price table's rates apply only to the Standard tier, so its usage
# cannot be priced as verified cost.
RETURNED_SERVICE_TIERS_ACCEPTED = (SERVICE_TIER_STANDARD,)

# What a run is for. It is fixed at initialisation and recorded in the identity, the run config, the summary,
# the score summary and the registry. An engineering_check is never eligible as a baseline, whatever it calls.
RUN_KIND_ENGINEERING = "engineering_check"
RUN_KIND_DEVELOPMENT = "development_pilot"
RUN_KINDS = (RUN_KIND_ENGINEERING, RUN_KIND_DEVELOPMENT)

# How a provider error was resolved, from the provider's side.
# not_sent: the request provably never left the process (for example a DNS or connect failure).
# rejected: the provider answered with an error status; no answer was produced.
# unknown: the request may have reached the provider and may have been processed and billed.
OUTCOME_NOT_SENT = "not_sent"
OUTCOME_REJECTED = "rejected"
OUTCOME_UNKNOWN = "unknown"
PROVIDER_OUTCOMES = (OUTCOME_NOT_SENT, OUTCOME_REJECTED, OUTCOME_UNKNOWN)

# Output status of a saved response (study brief section 15.7).
OUTPUT_OK = "ok"
OUTPUT_EMPTY = "empty"
OUTPUT_TRUNCATED = "truncated"
OUTPUT_REFUSAL = "refusal"
OUTPUT_ABSTAINED = "abstained"
OUTPUT_STATUSES = (OUTPUT_OK, OUTPUT_EMPTY, OUTPUT_TRUNCATED, OUTPUT_REFUSAL, OUTPUT_ABSTAINED)


@dataclass(frozen=True)
class EvidenceBlock:
    """One retrieved chunk as the prompt shows it. Text comes from the noisy MinerU input only."""

    rank: int
    chunk_id: str
    doc_id: str
    page_idx: int
    text: str


@dataclass(frozen=True)
class PromptPayload:
    """The exact messages for one question, with the hashes that identify them."""

    template_id: str
    template_sha256: str
    system: str
    user: str
    prompt_sha256: str
    evidence_sha256: str
    evidence_chars: int

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]


@dataclass(frozen=True)
class ProviderRequest:
    """What the run driver hands to a provider for one attempt."""

    request_id: str
    attempt_id: str
    messages: tuple[dict[str, str], ...]
    params: dict[str, Any]
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class ProviderUsage:
    """Token counts exactly as the provider reported them. None means not reported, never zero by default."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderResponse:
    """A response the provider returned. raw is the JSON-serialisable payload, kept unchanged."""

    text: str | None
    finish_reason: str | None
    refusal: str | None
    returned_model: str | None
    response_id: str | None
    usage: ProviderUsage
    raw: dict[str, Any] = field(default_factory=dict)
    # The service tier the provider reports it used, exactly as returned. None means the response had none.
    returned_service_tier: str | None = None


class ProviderError(Exception):
    """A failed attempt. outcome says whether the provider may have processed it."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        outcome: str,
        retryable: bool,
        http_status: int | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        if outcome not in PROVIDER_OUTCOMES:
            raise ValueError(f"unknown provider outcome {outcome!r}")
        super().__init__(message)
        self.kind = kind
        self.outcome = outcome
        self.retryable = retryable
        self.http_status = http_status
        self.raw = raw or {}


@dataclass(frozen=True)
class PriceTable:
    """Rates used to price usage. simulated=True marks fictional rates for fake-provider runs."""

    provider: str
    model: str
    currency: str
    input_per_million: float
    cached_input_per_million: float | None
    output_per_million: float
    source: str
    source_date: str
    simulated: bool
    # The service tier these rates apply to. None only in records written before the tier was recorded.
    service_tier: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["service_tier"] is None:
            del data["service_tier"]  # records written before the field existed keep their exact form
        return data
