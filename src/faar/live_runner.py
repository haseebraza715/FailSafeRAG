"""Answer-model run driver: durable records, resume, run lock, exports.

This module drives one answer model over the runtime questions of a frozen
pilot. It never opens an evaluation file during ``run``, ``dry-run``,
``status``, ``export`` or ``reconcile``. Only ``score`` reads the evaluation manifest.
Nothing here authorises a paid request: a fake provider serves every test, and
live mode refuses to start unless every check in :func:`check_live_enablement` passes.

Data flow::

    runtime manifest + MinerU text  (faar.pilot_runner.retrieve_runtime_questions)
      -> prepared requests (prompt, evidence hashes, token and cost bounds)      requests.jsonl
      -> dispatch loop: one event line per step, fsync before the next step      attempts.jsonl
      -> one response file per answered attempt                                  responses/<attempt_id>.json
      -> export, rebuilt from the three records above                            predictions.jsonl, run_summary.json
      -> score: join to the evaluation manifest                                  scores.jsonl, score_summary.json

Run directory::

    run_config.json    identity, provenance, mode and kind; written once, never rewritten
    requests.jsonl     one prepared request per runtime question, manifest order; written once
    attempts.jsonl     append-only events; one JSON object per line
    responses/         one file per saved response
    run.lock           held with flock while an invocation runs
    predictions.jsonl  run_summary.json   rebuilt by export
    scores.jsonl  score_summary.json      written by score

Crash windows (the contract's rule 3). Every step that changes the world is
written and fsynced first, so a restart can tell what happened:

1. Before ``dispatch_started`` is on disk, nothing was sent. The request is pending.
2. ``dispatch_started`` is on disk and nothing resolves it. The request may have
   reached the provider. The next invocation appends ``outcome_unknown`` and
   the question waits for ``reconcile``. Other questions continue.
3. A response arrived and the process died before the response file existed. The
   answer is lost and the attempt is unknown, as in window 2. If the process died
   after the response file but before ``response_saved``, the next invocation
   appends ``response_saved`` from that file and resends nothing.
4. ``response_saved`` is on disk. ``export`` rebuilds the predictions from the records.

Safety stop. A saved response that broke a safety assumption ends dispatch for good (see ``END_SAFETY_STOP``
and :func:`find_safety_violations`). The check reads the records before every dispatch, and a restart repeats
it before it builds a provider.

There is no exactly-once claim. A crash between ``send`` returning and the next
fsync can always lose a response.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
import socket
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .answer_providers import SimulatedCrash  # noqa: F401  (re-exported: crash hooks and the fake provider raise it)
from .live_contract import (
    ALLOWED_OPENAI_PARAMS,
    CONTRACT_VERSION,
    EVENT_ATTEMPT_FAILED,
    EVENT_DISPATCH_STARTED,
    EVENT_INVOCATION_ENDED,
    EVENT_INVOCATION_STARTED,
    EVENT_OUTCOME_UNKNOWN,
    EVENT_QUESTION_REOPENED,
    EVENT_RECONCILED,
    EVENT_RESPONSE_SAVED,
    EVENTS,
    MODE_FAKE,
    MODE_LIVE,
    MODES,
    OUTCOME_UNKNOWN,
    EvidenceBlock,
    PriceTable,
    PromptPayload,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)
from .pilot_runner import (
    DEFAULT_TEXT_POLICY,
    FAILURE_MESSAGE_LIMIT,
    RetrievalRun,
    RunnerRefusal,
    _display_path,
    _dumps,
    _package_versions,
    git_provenance,
    retrieve_runtime_questions,
    sha256_bytes,
    sha256_file,
    validate_run_id,
)
from .run_io import _measurement_code_digest, canonical_digest

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_EXECUTION_FAILED = 2
EXIT_INTERNAL_ERROR = 3
EXIT_BUDGET_LIMITED = 4
EXIT_NEEDS_ATTENTION = 5
# A safety or model assumption failed (see "Safety stop" below). Distinct from every code above: the run
# cannot go on and no later command of the same run directory can lift the stop.
EXIT_SAFETY_STOPPED = 6

SCHEMA_VERSION = 1
KIND_FAKE = "engineering_check"
KIND_LIVE = "development_pilot"
LABEL_FAKE = (
    "Engineering check with a fake answer provider. No request leaves the process and nothing here measures a model."
)
LABEL_LIVE = "Development pilot with a real answer model under engineering retrieval settings. Not the approved protocol."

RUN_CONFIG_NAME = "run_config.json"
REQUESTS_NAME = "requests.jsonl"
ATTEMPTS_NAME = "attempts.jsonl"
RESPONSES_DIR = "responses"
LOCK_NAME = "run.lock"
PREDICTIONS_NAME = "predictions.jsonl"
RUN_SUMMARY_NAME = "run_summary.json"
SCORES_NAME = "scores.jsonl"
SCORE_SUMMARY_NAME = "score_summary.json"
PREPARED_NAME = "prepared_requests.jsonl"
PREVIEW_NAME = "prompt_preview.md"
DRY_RUN_SUMMARY_NAME = "dry_run_summary.json"
UNCOMMITTED_PREFIX = "attempts.uncommitted."
KNOWN_RUN_FILES = frozenset(
    {
        RUN_CONFIG_NAME,
        REQUESTS_NAME,
        ATTEMPTS_NAME,
        RESPONSES_DIR,
        LOCK_NAME,
        PREDICTIONS_NAME,
        RUN_SUMMARY_NAME,
        SCORES_NAME,
        SCORE_SUMMARY_NAME,
    }
)

LIVE_ENV_NAME = "FAAR_ALLOW_LIVE_REQUESTS"
LIVE_ENV_VALUE = "I_UNDERSTAND_THIS_SPENDS_MONEY"
# The OpenAI SDK (1.68) reads the first three when its client is built without them. A live run must not
# inherit a redirected base URL or an organization or project it did not choose. The SDK does not read
# OPENAI_ORGANIZATION, but other SDK versions and tools do, so it is refused as well.
REDIRECTING_ENV_NAMES = ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION")
TOKENIZER_BOUND = "utf8-bytes"
TOKEN_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens")

RUN_COMPLETE = "complete"
RUN_BUDGET_LIMITED = "budget_limited"
RUN_NEEDS_RECONCILIATION = "needs_reconciliation"
RUN_INCOMPLETE = "incomplete"
RUN_STOPPED = "stopped"
RUN_SAFETY_STOPPED = "safety_stopped"
RUN_STATES = (RUN_COMPLETE, RUN_BUDGET_LIMITED, RUN_NEEDS_RECONCILIATION, RUN_INCOMPLETE, RUN_STOPPED, RUN_SAFETY_STOPPED)

END_COMPLETED = "completed"
END_BUDGET = "budget_exhausted"
END_STOP = "stop_run"
END_RECONCILIATION = "needs_reconciliation"
END_INTERRUPTED = "interrupted"
END_CIRCUIT = "circuit_breaker"
END_SAFETY_STOP = "safety_stop"
END_REASONS = (END_COMPLETED, END_BUDGET, END_STOP, END_RECONCILIATION, END_INTERRUPTED, END_CIRCUIT, END_SAFETY_STOP)

# Safety stop. A saved response can show that an assumption behind the cost bound or the measurement failed:
# it reports more input tokens than the request's input upper bound, it comes from a model other than the
# configured one, or the safety ledger reports an anomaly. The driver keeps that response and all its
# evidence, then dispatches nothing more. The violation is a function of the durable records (events, plus a
# response file whose event was lost), so every restart, status and export rebuilds it. Nothing in the same
# run directory clears it: not a raised ceiling, not reconcile, not reopen. A successor run with corrected code
# or configuration is the only way on, and the identity checks already demand a new run for that.
# Stopping cannot undo a charge that was already incurred. The ceiling guarantee rests on provider-reported
# usage and on valid bounds, and this check can only act on what the provider reports.
#
# The event log has no hash chain. The stop protects against the code's own behaviour and honest operation.
# Loading a run refuses a run_config.json whose identity no longer matches its identity_sha256, and refuses a
# response_saved event that differs from its hash-pinned response file. A file edit that keeps every one of
# those cross-checks consistent (for example rewriting the event, the response file and its hash together) is
# out of scope: nothing here detects it, and nothing here claims to.
VIOLATION_INPUT_BOUND = "input_bound_exceeded"
VIOLATION_RETURNED_MODEL = "returned_model_mismatch"
# Every anomaly kind that SafetyLedger reports is blocking today: a measured cost that differs from the cost
# recomputed from its usage, a measured cost above its own upper bound, and a recorded cost whose usage cannot
# reproduce it. A kind that request_budget adds later blocks too, on purpose. Relaxing that is a decision for
# the lead, made here and tested.
VIOLATION_LEDGER_ANOMALY = "ledger_anomaly"
FAILURE_SAFETY_STOP = "safety_stop"
SAFETY_LIMIT_NOTE = (
    "Stopping cannot undo a charge already incurred. The ceiling guarantee depends on provider-reported usage "
    "and on valid bounds."
)
SAFETY_MODEL_NOTE = (
    "The returned-model check is an exact match, so the configured model must be the dated snapshot the provider "
    "returns (for example gpt-4o-2024-11-20). A configured alias such as gpt-4o stops the run after the first response."
)
SAFETY_NO_CLEAR_NOTE = (
    "Resuming, raising the safety ceiling, reconcile and reopen do not clear the violation. "
    "Start a successor run (a new run directory) with corrected code or configuration."
)

# Circuit breaker. An invocation stops after this many consecutive counted attempts: an unknown outcome, a
# failure that ended its question (decision fail_question, whether rejected or never sent), or an exception that
# is not a ProviderError. Only a saved response resets the count. A retried attempt neither adds to it nor resets
# it, so a network outage that fails question after question still trips it. The number is part of the run
# identity (the retry section).
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_COUNTS = (
    "outcome unknown",
    "any failure with decision fail_question, rejected or not_sent",
    "exception that is not a ProviderError",
)
CIRCUIT_BREAKER_RESETS = ("saved response",)

STATUS_ANSWERED = "answered"
STATUS_NO_EVIDENCE = "no_evidence"
STATUS_EXECUTION_FAILED = "execution_failed"

REQUEST_SKIP = "skip"
REQUEST_ANSWERED = "answered"
REQUEST_FAILED = "failed"
REQUEST_UNKNOWN = "unknown"
REQUEST_PENDING = "pending"

ACTION_SEND = "send"
ACTION_SKIP = "skip"
SKIP_PROMPT_OVER_LIMIT = "prompt_over_limit"
# Extra skip reasons for a question whose preparation raised. Both export as execution_failed at stage "prepare".
SKIP_RETRIEVAL_FAILED = "retrieval_failed"
SKIP_PROMPT_BUILD_FAILED = "prompt_build_failed"
PREPARE_FAILURE_SKIPS = (SKIP_RETRIEVAL_FAILED, SKIP_PROMPT_BUILD_FAILED)

RESOLUTION_ALLOW = "allow_new_attempt"
RESOLUTION_FAIL = "mark_failed"
RESOLUTIONS = (RESOLUTION_ALLOW, RESOLUTION_FAIL)

DECISION_RETRY = "retry"
DECISION_FAIL = "fail_question"
DECISION_STOP = "stop_run"

# Names of the points where a crash hook is called. See the module docstring for the windows.
CRASH_BEFORE_DISPATCH = "before_dispatch_started"  # window 1
CRASH_AFTER_DISPATCH = "after_dispatch_started"  # window 2
CRASH_AFTER_SEND = "after_send"  # window 3, response in memory only
CRASH_AFTER_RESPONSE_FILE = "after_response_file"  # window 3, file on disk, no event
CRASH_AFTER_SAVED = "after_response_saved"  # window 4 starts here
CRASH_BEFORE_EXPORT = "before_export"  # window 4, after the last event
CRASH_POINTS = (
    CRASH_BEFORE_DISPATCH,
    CRASH_AFTER_DISPATCH,
    CRASH_AFTER_SEND,
    CRASH_AFTER_RESPONSE_FILE,
    CRASH_AFTER_SAVED,
    CRASH_BEFORE_EXPORT,
)

# SimulatedCrash (from faar.answer_providers) derives from BaseException, so no ``except Exception`` in the
# driver swallows it. The driver writes nothing after it, as after a real crash. A crash hook raises it, and so
# does the fake provider's crash_after_send step.

EXPECTED_RESUME_NOTE = "The run directory is a record. Keep it and start a new run directory for changed settings."


CrashHook = Callable[[str, Mapping[str, Any]], None]


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Everything about the answer model that is part of the run identity.

    ``endpoint`` is the API base URL the client sends to, for example ``https://api.openai.com/v1``. It is
    not a request path: the SDK adds ``/chat/completions``. The fake config uses ``fake://in-process``.
    """

    provider: str
    model: str
    endpoint: str
    params: Mapping[str, Any]
    max_output_tokens: int
    max_input_tokens: int
    timeout_seconds: float
    tokenizer_bound: str
    prices: PriceTable
    token_limit_param: str = "max_tokens"

    def identity_block(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "params": dict(self.params),
            "max_output_tokens": self.max_output_tokens,
            "max_input_tokens": self.max_input_tokens,
            "timeout_seconds": self.timeout_seconds,
            "tokenizer_bound": self.tokenizer_bound,
            "token_limit_param": self.token_limit_param,
        }


FAKE_CONFIG = ProviderConfig(
    provider="fake",
    model="fake-answer-model-1",
    endpoint="fake://in-process",
    params={"temperature": 0},
    max_output_tokens=128,
    max_input_tokens=12_000,
    timeout_seconds=60.0,
    tokenizer_bound=TOKENIZER_BOUND,
    prices=PriceTable(
        provider="fake",
        model="fake-answer-model-1",
        currency="USD",
        input_per_million=2.5,
        cached_input_per_million=1.25,
        output_per_million=10.0,
        source="simulated: fictional rates for engineering checks",
        source_date="2026-09-29",
        simulated=True,
    ),
)

# Used by dry-run when no --provider-config is given. These are the study brief's proposed
# values (section 15.5 option A and section 15.7 limits), which the lead has not approved.
PROVISIONAL_DRY_RUN_CONFIG = ProviderConfig(
    provider="openai",
    model="gpt-4o-2024-11-20",
    endpoint="https://api.openai.com/v1",
    params={"temperature": 0},
    max_output_tokens=128,
    max_input_tokens=12_000,
    timeout_seconds=60.0,
    tokenizer_bound=TOKENIZER_BOUND,
    prices=PriceTable(
        provider="openai",
        model="gpt-4o-2024-11-20",
        currency="USD",
        input_per_million=2.5,
        cached_input_per_million=1.25,
        output_per_million=10.0,
        source="study brief section 15.5, option A (OpenAI model page)",
        source_date="2026-09-29",
        simulated=False,
    ),
)


def _finite_positive(value: Any, what: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerRefusal(f"provider config: {what} must be a number, got {value!r}")
    if integer and not isinstance(value, int):
        raise RunnerRefusal(f"provider config: {what} must be an integer, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise RunnerRefusal(f"provider config: {what} must be finite and above zero, got {value!r}")


_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
FAKE_ENDPOINT_SCHEME = "fake"


def _check_endpoint(endpoint: str, *, simulated: bool) -> None:
    """The endpoint is an API base URL: https (http only on the loopback), a host, no credentials, query or fragment."""
    try:
        parts = urlsplit(endpoint)
        host = parts.hostname
        parts.port  # noqa: B018  (raises ValueError for a bad port)
    except ValueError as exc:
        raise RunnerRefusal(f"provider config: endpoint {endpoint!r} is not a valid URL ({exc})") from exc
    if simulated and parts.scheme == FAKE_ENDPOINT_SCHEME:
        return
    problem = None
    if parts.scheme not in ("https", "http") or not host:
        problem = "it must be an https URL with a host"
    elif parts.scheme == "http" and host not in _LOOPBACK_HOSTS:
        problem = "plain http is allowed only for localhost, 127.0.0.1 and ::1"
    elif parts.username is not None or parts.password is not None:
        problem = "it must not hold credentials"
    elif parts.query or parts.fragment:
        problem = "it must not hold a query or a fragment"
    elif parts.path.rstrip("/").endswith("/chat/completions"):
        problem = "it is the API base URL such as https://api.openai.com/v1, not the chat completions path"
    if problem:
        raise RunnerRefusal(f"provider config: endpoint {endpoint!r} is refused: {problem}")


def parse_provider_config(payload: Any, *, simulated: bool = False) -> ProviderConfig:
    """Validate a provider config object. Raises :class:`RunnerRefusal` on any missing or odd field."""
    if not isinstance(payload, dict):
        raise RunnerRefusal("provider config is not a JSON object")
    required = (
        "provider",
        "model",
        "endpoint",
        "params",
        "max_output_tokens",
        "timeout_seconds",
        "tokenizer_bound",
        "prices",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RunnerRefusal(f"provider config lacks fields: {missing}")
    for key in ("provider", "model", "endpoint"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise RunnerRefusal(f"provider config: {key} must be a non-empty string")
    if not isinstance(payload["params"], dict):
        raise RunnerRefusal("provider config: params must be an object")
    unknown_params = sorted(set(payload["params"]) - set(ALLOWED_OPENAI_PARAMS))
    if unknown_params:
        raise RunnerRefusal(
            f"provider config: params {unknown_params} are not allowed. An unknown request parameter can change "
            f"billing outside the cost bound. Allowed: {list(ALLOWED_OPENAI_PARAMS)}"
        )
    _check_endpoint(payload["endpoint"], simulated=simulated)
    _finite_positive(payload["max_output_tokens"], "max_output_tokens", integer=True)
    max_input = payload.get("max_input_tokens", 12_000)
    _finite_positive(max_input, "max_input_tokens", integer=True)
    _finite_positive(payload["timeout_seconds"], "timeout_seconds")
    if payload["tokenizer_bound"] != TOKENIZER_BOUND:
        raise RunnerRefusal(
            f"provider config: tokenizer_bound must be {TOKENIZER_BOUND!r} "
            f"(the input bound counts UTF-8 bytes), got {payload['tokenizer_bound']!r}"
        )
    prices = payload["prices"]
    if not isinstance(prices, dict):
        raise RunnerRefusal("provider config: prices must be an object")
    for key in ("currency", "source", "source_date"):
        if not isinstance(prices.get(key), str) or not prices[key]:
            raise RunnerRefusal(f"provider config: prices.{key} must be a non-empty string")
    _finite_positive(prices.get("input_per_million"), "prices.input_per_million")
    _finite_positive(prices.get("output_per_million"), "prices.output_per_million")
    cached = prices.get("cached_input_per_million")
    if cached is not None:
        _finite_positive(cached, "prices.cached_input_per_million")
    token_limit_param = payload.get("token_limit_param", "max_tokens")
    if token_limit_param not in TOKEN_LIMIT_PARAMS:
        raise RunnerRefusal(f"provider config: token_limit_param must be one of {TOKEN_LIMIT_PARAMS}, got {token_limit_param!r}")
    price_table = PriceTable(
        provider=payload["provider"],
        model=payload["model"],
        currency=prices["currency"],
        input_per_million=float(prices["input_per_million"]),
        cached_input_per_million=None if cached is None else float(cached),
        output_per_million=float(prices["output_per_million"]),
        source=prices["source"],
        source_date=prices["source_date"],
        simulated=simulated,
    )
    return ProviderConfig(
        provider=payload["provider"],
        model=payload["model"],
        endpoint=payload["endpoint"],
        params=dict(payload["params"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        max_input_tokens=int(max_input),
        timeout_seconds=float(payload["timeout_seconds"]),
        tokenizer_bound=payload["tokenizer_bound"],
        prices=price_table,
        token_limit_param=token_limit_param,
    )


def load_provider_config(path: Path, *, simulated: bool = False) -> ProviderConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"cannot read the provider config {path}: {exc}") from exc
    return parse_provider_config(payload, simulated=simulated)


# ---------------------------------------------------------------------------
# Collaborators behind one seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Services:
    """The prompt builder, the budget arithmetic, the ledger and the retry policy, as plain callables.

    ``Services.default()`` wires the real modules. A test can pass its own
    values. Nothing here touches the network.
    """

    template_id: str
    template_sha256: str
    build_prompt: Callable[[str, Sequence[EvidenceBlock]], PromptPayload]
    parse_reply: Callable[[str | None, str | None, str | None], Mapping[str, Any]]
    estimate_tokens: Callable[[str], Mapping[str, Any]]
    render_preview: Callable[..., str]
    input_token_upper_bound: Callable[[Sequence[Mapping[str, str]]], int]
    request_cost_upper_bound: Callable[[int, int, PriceTable], float]
    measured_cost: Callable[[Any, PriceTable], float | None]
    ledger_from_events: Callable[[Sequence[Mapping[str, Any]], PriceTable], Any]
    retry_policy: Any

    def retry_parameters(self) -> dict[str, Any]:
        policy = self.retry_policy
        if hasattr(policy, "describe"):
            return dict(policy.describe())
        if dataclasses.is_dataclass(policy) and not isinstance(policy, type):
            return dataclasses.asdict(policy)
        return dict(vars(policy))

    @classmethod
    def default(cls) -> Services:
        try:
            from . import answer_prompt, prompt_preview, request_budget, retry_policy
        except ImportError as exc:  # pragma: no cover - the modules ship with this package
            raise RunnerRefusal(f"the answer-model modules are not available in this checkout: {exc}") from exc

        def ledger_from_events(events: Sequence[Mapping[str, Any]], prices: PriceTable) -> Any:
            # The ledger treats question_reopened as costless: the attempts it follows are already counted.
            try:
                return request_budget.SafetyLedger.from_events(events, prices)
            except request_budget.BudgetError as exc:
                raise RunnerRefusal(f"the safety ledger refused the run's records: {exc}") from exc

        return cls(
            template_id=answer_prompt.TEMPLATE_ID,
            template_sha256=answer_prompt.TEMPLATE_SHA256,
            build_prompt=answer_prompt.build_prompt,
            parse_reply=answer_prompt.parse_reply,
            estimate_tokens=answer_prompt.estimate_tokens,
            render_preview=prompt_preview.render_preview,
            input_token_upper_bound=request_budget.input_token_upper_bound,
            request_cost_upper_bound=request_budget.request_cost_upper_bound,
            measured_cost=request_budget.measured_cost,
            ledger_from_events=ledger_from_events,
            retry_policy=retry_policy.RetryPolicy(),
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Durable writes
# ---------------------------------------------------------------------------


def sync_fd(fd: int) -> None:
    """Flush a file descriptor to stable storage.

    ``os.fsync`` is the portable call. On macOS it hands the data to the drive but does not ask the drive
    to empty its own write cache, so a power cut can still lose it. ``fcntl.F_FULLFSYNC`` asks for that.
    The second call is best effort: a filesystem that refuses it (some network shares) leaves the plain
    ``os.fsync`` result, which is the strongest guarantee available there.
    """
    os.fsync(fd)
    if sys.platform == "darwin":
        command = getattr(fcntl, "F_FULLFSYNC", None)
        if command is not None:
            try:
                fcntl.fcntl(fd, command)
            except OSError:
                pass


def _fsync_dir(directory: Path) -> None:
    """Sync a directory entry after a create, rename or truncate. Best effort: some filesystems refuse it."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        sync_fd(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file, sync it, rename it over ``path`` and sync the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            sync_fd(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _line(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _jsonl(records: Sequence[Any]) -> str:
    return "".join(_line(record) + "\n" for record in records)


def _sha_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _refuse(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerRefusal(message)


def _diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Dotted paths at which two JSON values differ. Used to say what changed in a refused resume."""
    if isinstance(a, dict) and isinstance(b, dict):
        paths: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            paths.extend(_diff_paths(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else str(key)))
        return paths
    return [] if a == b else [prefix]


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _protected_top_level(root_relative: Sequence[str]) -> bool:
    protected = {"config", "ohr-bench", "annotation", "experiments", "logs"}
    if not root_relative:
        return False
    if root_relative[0] in protected:
        return True
    return len(root_relative) >= 2 and root_relative[0] == "results" and root_relative[1] == "pilots"


def refuse_output_location(out_dir: Path, project_root: Path) -> None:
    """Refuse a dry-run output directory in a frozen or shared location (case-insensitive)."""
    resolved = _absolute(out_dir).resolve()
    folded = [part.casefold() for part in resolved.parts]
    in_pilots = any(folded[i] == "results" and folded[i + 1] == "pilots" for i in range(len(folded) - 1))
    _refuse(not in_pilots, f"refusing {out_dir}: output must not be written under results/pilots/.")
    root = [part.casefold() for part in project_root.resolve().parts]
    if folded[: len(root)] == root:
        _refuse(
            not _protected_top_level(folded[len(root) :]),
            f"refusing {out_dir}: config/, OHR-Bench/, annotation/, experiments/, logs/ and results/pilots/ hold frozen or shared files.",
        )


def refuse_frozen_pilots_path(run_dir: Path) -> None:
    """Refuse any path with a ``results/pilots`` pair, in any letter case, anywhere in it."""
    folded = [part.casefold() for part in _absolute(run_dir).resolve().parts]
    in_pilots = any(folded[i] == "results" and folded[i + 1] == "pilots" for i in range(len(folded) - 1))
    _refuse(not in_pilots, f"refusing {run_dir}: runs must not be written under results/pilots/.")


def require_initialised_run(run_dir: Path) -> Path:
    """Check a run directory before a command creates or opens ``run.lock`` in it.

    ``RunLock`` creates its file, so a mistyped path would gain a ``run.lock`` (and a frozen directory would
    be written to). Every command that takes the lock calls this first: the path must not sit under
    ``results/pilots/`` and the directory must hold ``run_config.json``. Nothing is created or changed.
    """
    run_dir = _absolute(run_dir)
    refuse_frozen_pilots_path(run_dir)
    _refuse(run_dir.is_dir(), f"{run_dir} is not a run directory")
    _refuse((run_dir / RUN_CONFIG_NAME).is_file(), f"{run_dir} holds no {RUN_CONFIG_NAME}; it is not an initialised run.")
    return run_dir


def is_scored(run_dir: Path) -> bool:
    """True once ``score`` wrote either score file. A scored run is final."""
    return (run_dir / SCORES_NAME).exists() or (run_dir / SCORE_SUMMARY_NAME).exists()


def refuse_if_scored(run_dir: Path, what: str) -> None:
    _refuse(
        not is_scored(run_dir),
        f"{run_dir} has been scored, so {what} is refused. Scoring is final for a run: the scores describe these "
        "predictions. Start a new run directory to continue.",
    )


def refuse_run_directory(run_dir: Path, project_root: Path, mode: str) -> None:
    """A fake run lives under results/engineering/<run_id>/, a live run under results/development/<run_id>/.

    Outside the project root any directory is allowed. Anywhere, a ``results/pilots`` pair is refused
    (case-insensitive, because ``results/Pilots`` reaches the frozen directory on a case-insensitive filesystem).
    """
    refuse_frozen_pilots_path(run_dir)
    folded = [part.casefold() for part in _absolute(run_dir).resolve().parts]
    root = [part.casefold() for part in project_root.resolve().parts]
    if folded[: len(root)] == root:
        area = "engineering" if mode == MODE_FAKE else "development"
        inside = folded[len(root) :]
        _refuse(
            len(inside) >= 3 and inside[0] == "results" and inside[1] == area,
            f"refusing {run_dir}: a {mode} run inside the project must be under results/{area}/<run_id>/.",
        )


# ---------------------------------------------------------------------------
# Run lock
# ---------------------------------------------------------------------------


class RunLock:
    """An exclusive ``flock`` on ``run.lock``, held until :meth:`release` or process exit.

    The kernel drops the lock when the process dies, so a crash never leaves a stale lock.
    """

    def __init__(self, run_dir: Path, invocation_id: str) -> None:
        self.path = run_dir / LOCK_NAME
        self.invocation_id = invocation_id
        self._fd: int | None = None

    def acquire(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_lock_holder(fd)
            os.close(fd)
            raise RunnerRefusal(
                f"another invocation holds {self.path} ({holder}). Wait for it to finish; nothing was dispatched."
            ) from None
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        content = {"pid": os.getpid(), "host": socket.gethostname(), "invocation_id": self.invocation_id, "started_at": _now()}
        os.ftruncate(fd, 0)
        os.pwrite(fd, (_line(content) + "\n").encode("utf-8"), 0)
        os.fsync(fd)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _read_lock_holder(fd: int) -> str:
    try:
        raw = os.pread(fd, 4096, 0).decode("utf-8", errors="replace").strip()
        info = json.loads(raw) if raw else {}
        return f"pid {info.get('pid')} on {info.get('host')}, invocation {info.get('invocation_id')}"
    except (OSError, ValueError):
        return "holder unknown"


def lock_is_held(run_dir: Path) -> bool | None:
    """Probe the lock without keeping it. ``None`` when there is no lock file.

    A shared, non-blocking ``flock`` fails while an invocation holds the exclusive lock.
    """
    path = run_dir / LOCK_NAME
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

_REQUIRED_EVENT_KEYS: dict[str, tuple[str, ...]] = {
    EVENT_INVOCATION_STARTED: ("safety_ceiling", "ceiling_change", "mode", "identity_sha256"),
    EVENT_DISPATCH_STARTED: ("question_id", "request_id", "attempt", "attempt_id", "cost_upper_bound", "ledger_before"),
    EVENT_RESPONSE_SAVED: (
        "question_id",
        "request_id",
        "attempt",
        "attempt_id",
        "response_file",
        "response_sha256",
        "usage",
        "measured_cost",
        "output_status",
        "answer",
        "abstained",
    ),
    EVENT_ATTEMPT_FAILED: (
        "question_id",
        "request_id",
        "attempt",
        "attempt_id",
        "kind",
        "outcome",
        "retryable",
        "message",
        "decision",
    ),
    EVENT_OUTCOME_UNKNOWN: ("question_id", "request_id", "attempt", "attempt_id", "kind", "message"),
    EVENT_RECONCILED: ("question_id", "request_id", "attempt", "attempt_id", "resolution", "note"),
    EVENT_INVOCATION_ENDED: ("reason", "ledger"),
    EVENT_QUESTION_REOPENED: ("question_id", "request_id", "attempts_before", "note"),
}
_BASE_EVENT_KEYS = ("event", "invocation_id", "seq", "at")


@dataclass(frozen=True)
class EventFile:
    """The committed events of ``attempts.jsonl`` and any uncommitted tail."""

    events: tuple[dict[str, Any], ...]
    committed_bytes: int
    uncommitted_tail: bytes


def read_event_file(path: Path) -> EventFile:
    """Read ``attempts.jsonl``. A final line without a newline is uncommitted and ignored.

    Any other malformed line, an unknown event, a missing key or a broken ``seq`` raises
    :class:`RunnerRefusal`. The file is not changed.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return EventFile((), 0, b"")
    except OSError as exc:
        raise RunnerRefusal(f"cannot read {path}: {exc}") from exc
    end = raw.rfind(b"\n") + 1
    committed, tail = raw[:end], raw[end:]
    events: list[dict[str, Any]] = []
    for number, chunk in enumerate(committed.split(b"\n")[:-1], start=1):
        try:
            event = json.loads(chunk.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunnerRefusal(f"{path} line {number} is not valid JSON ({exc}); the log is corrupt. Nothing was changed.") from exc
        _refuse(isinstance(event, dict), f"{path} line {number} is not a JSON object; the log is corrupt.")
        missing = [key for key in _BASE_EVENT_KEYS if key not in event]
        _refuse(not missing, f"{path} line {number} lacks {missing}; the log is corrupt.")
        _refuse(event["event"] in EVENTS, f"{path} line {number} has unknown event {event['event']!r}; the log is corrupt.")
        _refuse(event["seq"] == number, f"{path} line {number} has seq {event['seq']!r}; a line is missing or repeated.")
        lacking = [key for key in _REQUIRED_EVENT_KEYS[event["event"]] if key not in event]
        _refuse(not lacking, f"{path} line {number} ({event['event']}) lacks {lacking}; the log is corrupt.")
        events.append(event)
    return EventFile(tuple(events), end, tail)


class EventLog:
    """Appends events to ``attempts.jsonl``: one ``O_APPEND`` write per line, then a sync.

    Open it only while holding the run lock.

    An interrupt can land at any point of :meth:`append`. The order is chosen so that no point leaves a
    second line with the same ``seq``:

    1. Write the whole line. If this raises (a signal, a full disk), cut the file back to its earlier
       length, so the line was never part of the log.
    2. Move ``_seq`` on and call ``on_append``. From here the line is part of the log, whatever happens next.
    3. Sync in a ``finally``, so a failing callback does not skip it.

    A ``KeyboardInterrupt`` can still arrive between the last byte and step 2. The run driver handles that by
    reading the log back from the file (:func:`_resume_after_interrupt`) and calling :meth:`resume_at`.
    """

    def __init__(self, path: Path, invocation_id: str, next_seq: int, on_append: Callable[[dict[str, Any]], None]) -> None:
        self.path = path
        self.invocation_id = invocation_id
        self._seq = next_seq
        self._on_append = on_append
        self._broken = False
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        _fsync_dir(path.parent)

    def resume_at(self, next_seq: int) -> None:
        """Set the next ``seq`` after the driver re-read the file. Also clears a failed rollback."""
        self._seq = next_seq
        self._broken = False

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        _refuse(not self._broken, f"{self.path} may end in a partial line after a failed write; no event was appended.")
        record: dict[str, Any] = {"event": event, "invocation_id": self.invocation_id, "seq": self._seq, "at": _now()}
        record.update(fields)
        data = (_line(record) + "\n").encode("utf-8")
        size_before = os.fstat(self._fd).st_size
        try:
            view = memoryview(data)
            while view:
                written = os.write(self._fd, view)
                view = view[written:]
        except BaseException:
            try:
                os.ftruncate(self._fd, size_before)
            except OSError:
                self._broken = True
            raise
        self._seq += 1
        try:
            self._on_append(record)
        finally:
            sync_fd(self._fd)
        return record

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Identity and prepared requests
# ---------------------------------------------------------------------------


def build_identity(
    *,
    mode: str,
    retrieval_run: RetrievalRun,
    config: ProviderConfig,
    services: Services,
    provider_descriptor: Mapping[str, Any],
    cli_script_sha256: str | None,
    scientific_budget: Any = None,
) -> dict[str, Any]:
    """Content that decides the answers. No timestamps, no random ids, no safety ceiling."""
    return {
        "contract_version": CONTRACT_VERSION,
        "mode": mode,
        "pilot_id": retrieval_run.pilot_id,
        "runtime_manifest_sha256": retrieval_run.runtime_manifest_sha256,
        "document_noisy_text_sha256": dict(retrieval_run.document_noisy_text_sha256),
        "retrieval": dict(retrieval_run.retrieval),
        "prompt": {"template_id": services.template_id, "template_sha256": services.template_sha256},
        "provider": {**config.identity_block(), "adapter": dict(provider_descriptor)},
        "prices": config.prices.as_dict(),
        "retry_policy": {
            **services.retry_parameters(),
            "circuit_breaker": {
                "consecutive_failures": CIRCUIT_BREAKER_THRESHOLD,
                "counts": list(CIRCUIT_BREAKER_COUNTS),
                "resets": list(CIRCUIT_BREAKER_RESETS),
            },
        },
        "scientific_budget": scientific_budget,
        "code": {"measurement_code_digest": _measurement_code_digest(), "cli_script_sha256": cli_script_sha256},
    }


def identity_sha256(identity: Mapping[str, Any]) -> str:
    return canonical_digest(dict(identity))


def _request_id(identity_hash: str, question_id: str, prompt_hash: str | None) -> str:
    return hashlib.sha256(f"{identity_hash}\0{question_id}\0{prompt_hash or ''}".encode()).hexdigest()[:32]


def prepare_requests(
    retrieval_run: RetrievalRun,
    *,
    config: ProviderConfig,
    services: Services,
    identity_hash: str,
) -> list[dict[str, Any]]:
    """Build one prepared-request record per runtime question, in manifest order.

    Each record holds the full evidence text under ``evidence[i]["text"]``. Use
    :func:`strip_evidence_text` for the form stored in ``requests.jsonl``.
    """
    records: list[dict[str, Any]] = []
    for outcome in retrieval_run.questions:
        question = outcome.question
        record: dict[str, Any] = {
            "question_id": question.question_id,
            "doc_id": question.doc_id,
            "question": question.question,
            "action": ACTION_SKIP,
            "skip_reason": None,
            "evidence": [],
            "evidence_sha256": None,
            "prompt_sha256": None,
            "template_id": services.template_id,
            "system": None,
            "user": None,
            "input_token_upper_bound": None,
            "max_output_tokens": None,
            "cost_upper_bound": None,
            "cost_currency": config.prices.currency,
            "cost_simulated": config.prices.simulated,
            "request_id": None,
            "token_estimate": None,
            "no_evidence_reason": outcome.no_evidence_reason,
            "query_retrieval_tokens": outcome.query_retrieval_tokens,
            "ocr_condition": dict(outcome.ocr_condition),
            "prepare_failure": None,
        }
        if outcome.failure is not None:
            record["skip_reason"] = SKIP_RETRIEVAL_FAILED
            record["prepare_failure"] = dict(outcome.failure)
            record["request_id"] = _request_id(identity_hash, question.question_id, None)
            records.append(record)
            continue
        if not outcome.hits:
            record["skip_reason"] = outcome.no_evidence_reason
            record["request_id"] = _request_id(identity_hash, question.question_id, None)
            records.append(record)
            continue
        blocks: list[EvidenceBlock] = []
        evidence: list[dict[str, Any]] = []
        for rank, hit in enumerate(outcome.hits, start=1):
            if hit.chunk.doc_name != question.doc_id:
                raise RunnerRefusal(
                    f"question {question.question_id}: evidence chunk {hit.chunk.chunk_id!r} belongs to "
                    f"{hit.chunk.doc_name!r}, not to {question.doc_id!r}. Nothing was prepared."
                )
            blocks.append(EvidenceBlock(rank, hit.chunk.chunk_id, hit.chunk.doc_name, hit.chunk.page_id, hit.chunk.text))
            evidence.append(
                {
                    "rank": rank,
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": hit.chunk.doc_name,
                    "page_idx": hit.chunk.page_id,
                    "pdf_page_number": hit.chunk.page_id + 1,
                    "chars": len(hit.chunk.text),
                    "text_sha256": _sha_text(hit.chunk.text),
                    "fused_score": float(hit.fused_score),
                    "bm25_score": float(hit.bm25_score),
                    "dense_score": float(hit.dense_score),
                    "text": hit.chunk.text,
                }
            )
        try:
            payload = services.build_prompt(question.question, blocks)
        except ValueError as exc:
            record["skip_reason"] = SKIP_PROMPT_BUILD_FAILED
            record["evidence"] = evidence
            record["prepare_failure"] = {"stage": "prompt", "type": type(exc).__name__, "message": str(exc)[:FAILURE_MESSAGE_LIMIT]}
            record["request_id"] = _request_id(identity_hash, question.question_id, None)
            records.append(record)
            continue
        _refuse(
            payload.template_id == services.template_id and payload.template_sha256 == services.template_sha256,
            f"question {question.question_id}: the prompt names template {payload.template_id}/{payload.template_sha256}, "
            f"the run identity names {services.template_id}/{services.template_sha256}",
        )
        try:
            upper = services.input_token_upper_bound(payload.messages())
            cost = services.request_cost_upper_bound(upper, config.max_output_tokens, config.prices)
        except (ValueError, TypeError) as exc:
            raise RunnerRefusal(
                f"question {question.question_id}: the input bound or cost bound cannot be computed ({exc}). "
                "A missing or non-finite price, max_output_tokens or input bound refuses the run (contract rule 6)."
            ) from exc
        _refuse(
            isinstance(cost, float) and math.isfinite(cost) and cost >= 0,
            f"question {question.question_id}: the cost bound {cost!r} is not a finite number",
        )
        over = upper > config.max_input_tokens
        record.update(
            {
                "action": ACTION_SKIP if over else ACTION_SEND,
                "skip_reason": SKIP_PROMPT_OVER_LIMIT if over else None,
                "evidence": evidence,
                "evidence_sha256": payload.evidence_sha256,
                "prompt_sha256": payload.prompt_sha256,
                "system": payload.system,
                "user": payload.user,
                "input_token_upper_bound": upper,
                "max_output_tokens": config.max_output_tokens,
                "cost_upper_bound": cost,
                "request_id": _request_id(identity_hash, question.question_id, payload.prompt_sha256),
                "token_estimate": dict(services.estimate_tokens(payload.system + "\n\n" + payload.user)),
                "no_evidence_reason": None,
            }
        )
        records.append(record)
    _refuse(
        [r["question_id"] for r in records] == [q.question.question_id for q in retrieval_run.questions],
        "prepared requests do not match the runtime questions",
    )
    _refuse(len({r["request_id"] for r in records}) == len(records), "two prepared requests share a request_id")
    return records


def strip_evidence_text(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The stored form of a prepared request: message text stays, per-chunk text goes (it is inside ``user``)."""
    stored = []
    for record in records:
        copy = dict(record)
        copy["evidence"] = [{k: v for k, v in item.items() if k != "text"} for item in record["evidence"]]
        stored.append(copy)
    return stored


# ---------------------------------------------------------------------------
# Reconstructing state from the records
# ---------------------------------------------------------------------------


@dataclass
class AttemptState:
    attempt: int
    attempt_id: str
    dispatch: dict[str, Any]
    outcome: str | None = None  # saved | failed | unknown
    resolution_event: dict[str, Any] | None = None
    reconciled: dict[str, Any] | None = None


@dataclass
class RequestState:
    """One question's attempts, folded from the events.

    ``reopen_marks`` holds, for each ``question_reopened`` event, how many attempts existed before it. The
    attempt limit and the failure test look only at the attempts after the latest mark (the window). Earlier
    attempts stay in ``attempts``, and their costs stay in the ledger.
    """

    request: dict[str, Any]
    attempts: list[AttemptState] = field(default_factory=list)
    reopen_marks: list[int] = field(default_factory=list)

    @property
    def request_id(self) -> str:
        return self.request["request_id"]

    @property
    def question_id(self) -> str:
        return self.request["question_id"]

    @property
    def window_attempts(self) -> list[AttemptState]:
        return self.attempts[(self.reopen_marks[-1] if self.reopen_marks else 0) :]

    @property
    def attempts_in_window(self) -> int:
        """Attempts since the latest reopen (all attempts when the question was never reopened)."""
        return len(self.window_attempts)

    @property
    def saved(self) -> AttemptState | None:
        return next((a for a in self.attempts if a.outcome == "saved"), None)

    @property
    def unresolved_unknown(self) -> AttemptState | None:
        return next((a for a in self.attempts if a.outcome == "unknown" and a.reconciled is None), None)

    @property
    def orphans(self) -> list[AttemptState]:
        return [a for a in self.attempts if a.outcome is None]

    @property
    def marked_failed(self) -> AttemptState | None:
        return next(
            (a for a in self.window_attempts if a.reconciled is not None and a.reconciled["resolution"] == RESOLUTION_FAIL),
            None,
        )

    @property
    def status(self) -> str:
        if self.request["action"] == ACTION_SKIP:
            return REQUEST_SKIP
        if self.saved is not None:
            return REQUEST_ANSWERED
        if self.unresolved_unknown is not None or self.orphans:
            return REQUEST_UNKNOWN
        if self.marked_failed is not None:
            return REQUEST_FAILED
        window = self.window_attempts
        last = window[-1] if window else None
        if last is not None and last.outcome == "failed":
            event = last.resolution_event
            # A stop_run error on the last allowed attempt also ends the question (it never gets another).
            if event["decision"] == DECISION_FAIL or event.get("attempts_exhausted") is True:
                return REQUEST_FAILED
        return REQUEST_PENDING

    @property
    def next_attempt(self) -> int:
        return len(self.attempts) + 1


@dataclass
class InvocationInfo:
    invocation_id: str
    started: dict[str, Any]
    ended: dict[str, Any] | None = None


@dataclass
class RunView:
    """The run as its files describe it. Built by :func:`load_run`."""

    run_dir: Path
    config: dict[str, Any]
    requests: list[dict[str, Any]]
    requests_bytes: bytes
    events: list[dict[str, Any]]
    committed_bytes: int
    uncommitted_tail: bytes
    states: dict[str, RequestState]
    invocations: list[InvocationInfo]
    # Filled by assess_safety(). None until then, so a caller that skips the assessment fails loudly
    # instead of reporting a safety-stopped run as complete.
    safety_violations: list[dict[str, Any]] | None = None

    @property
    def ordered_states(self) -> list[RequestState]:
        return [self.states[r["request_id"]] for r in self.requests]

    def by_status(self, status: str) -> list[RequestState]:
        return [s for s in self.ordered_states if s.status == status]

    @property
    def last_end_reason(self) -> str | None:
        """Why the latest invocation ended. ``interrupted`` when it wrote no ``invocation_ended``."""
        if not self.invocations:
            return None
        last = self.invocations[-1]
        return last.ended["reason"] if last.ended is not None else END_INTERRUPTED

    @property
    def effective_ceiling(self) -> dict[str, Any] | None:
        return self.invocations[-1].started["safety_ceiling"] if self.invocations else None

    @property
    def run_state(self) -> str:
        if self.safety_violations is None:
            raise RuntimeError("assess_safety(view, services) must run before run_state is read")
        if self.safety_violations:
            return RUN_SAFETY_STOPPED
        pending = self.by_status(REQUEST_PENDING)
        if self.by_status(REQUEST_UNKNOWN):
            return RUN_NEEDS_RECONCILIATION
        if pending:
            reason = self.last_end_reason
            if reason == END_BUDGET:
                return RUN_BUDGET_LIMITED
            if reason in (END_STOP, END_CIRCUIT):
                return RUN_STOPPED
            return RUN_INCOMPLETE
        return RUN_COMPLETE


def _read_requests(run_dir: Path) -> tuple[list[dict[str, Any]], bytes]:
    path = run_dir / REQUESTS_NAME
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunnerRefusal(f"cannot read {path}: {exc}") from exc
    try:
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"{path} is corrupt: {exc}") from exc
    return records, raw


def reconstruct(requests: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]) -> tuple[dict[str, RequestState], list[InvocationInfo]]:
    """Fold the events into per-request and per-invocation state. Raises on any inconsistency."""
    states = {r["request_id"]: RequestState(r) for r in requests}
    attempts_by_id: dict[str, AttemptState] = {}
    invocations: list[InvocationInfo] = []
    by_invocation: dict[str, InvocationInfo] = {}

    def state_for(event: Mapping[str, Any]) -> RequestState:
        state = states.get(event["request_id"])
        _refuse(state is not None, f"event {event['seq']} names request {event['request_id']!r}, which requests.jsonl does not hold")
        assert state is not None
        _refuse(
            state.question_id == event["question_id"],
            f"event {event['seq']} names question {event['question_id']!r} for a request that belongs to {state.question_id!r}",
        )
        return state

    def attempt_for(event: Mapping[str, Any], state: RequestState) -> AttemptState:
        attempt = attempts_by_id.get(event["attempt_id"])
        _refuse(attempt is not None, f"event {event['seq']} ({event['event']}) resolves attempt {event['attempt_id']!r}, which has no dispatch_started")
        assert attempt is not None
        _refuse(attempt in state.attempts, f"event {event['seq']} resolves an attempt of another request")
        return attempt

    for event in events:
        kind = event["event"]
        if kind == EVENT_INVOCATION_STARTED:
            info = InvocationInfo(event["invocation_id"], event)
            _refuse(event["invocation_id"] not in by_invocation, f"invocation {event['invocation_id']} starts twice")
            by_invocation[event["invocation_id"]] = info
            invocations.append(info)
        elif kind == EVENT_INVOCATION_ENDED:
            info = by_invocation.get(event["invocation_id"])
            _refuse(info is not None and info.ended is None, f"event {event['seq']}: invocation_ended without a matching invocation_started")
            assert info is not None
            info.ended = event
        elif kind == EVENT_DISPATCH_STARTED:
            state = state_for(event)
            _refuse(
                state.status == REQUEST_PENDING,
                f"event {event['seq']}: dispatch_started for question {event['question_id']} in state {state.status}",
            )
            _refuse(event["attempt"] == state.next_attempt, f"event {event['seq']}: attempt {event['attempt']} follows {len(state.attempts)} attempts")
            _refuse(event["attempt_id"] == f"{event['request_id']}-a{event['attempt']}", f"event {event['seq']}: attempt_id does not match request_id and attempt")
            attempt = AttemptState(event["attempt"], event["attempt_id"], event)
            state.attempts.append(attempt)
            attempts_by_id[event["attempt_id"]] = attempt
        elif kind in (EVENT_RESPONSE_SAVED, EVENT_ATTEMPT_FAILED, EVENT_OUTCOME_UNKNOWN):
            state = state_for(event)
            attempt = attempt_for(event, state)
            _refuse(attempt.outcome is None, f"event {event['seq']}: attempt {attempt.attempt_id} is resolved twice")
            attempt.outcome = {EVENT_RESPONSE_SAVED: "saved", EVENT_ATTEMPT_FAILED: "failed", EVENT_OUTCOME_UNKNOWN: "unknown"}[kind]
            attempt.resolution_event = event
            if kind == EVENT_ATTEMPT_FAILED:
                _refuse(event["decision"] in (DECISION_RETRY, DECISION_FAIL, DECISION_STOP), f"event {event['seq']}: unknown decision {event['decision']!r}")
        elif kind == EVENT_RECONCILED:
            state = state_for(event)
            attempt = attempt_for(event, state)
            _refuse(attempt.outcome == "unknown" and attempt.reconciled is None, f"event {event['seq']}: attempt {attempt.attempt_id} is not awaiting reconciliation")
            _refuse(event["resolution"] in RESOLUTIONS, f"event {event['seq']}: unknown resolution {event['resolution']!r}")
            attempt.reconciled = event
        elif kind == EVENT_QUESTION_REOPENED:
            state = state_for(event)
            _refuse(
                state.status == REQUEST_FAILED,
                f"event {event['seq']}: question_reopened for question {event['question_id']} in state {state.status}, not {REQUEST_FAILED}",
            )
            _refuse(
                event["attempts_before"] == len(state.attempts),
                f"event {event['seq']}: question_reopened says {event['attempts_before']} earlier attempts, the question has {len(state.attempts)}",
            )
            state.reopen_marks.append(len(state.attempts))
    for state in states.values():
        _refuse(len([a for a in state.attempts if a.outcome == "saved"]) <= 1, f"question {state.question_id} has two saved responses")
    return states, invocations


def _response_path(run_dir: Path, attempt_id: str) -> Path:
    return run_dir / RESPONSES_DIR / f"{attempt_id}.json"


def _event_differences(event: Mapping[str, Any], fields: Mapping[str, Any]) -> list[str]:
    """Names of the ``response_saved`` fields on which the event differs from the fields rebuilt from its file.

    Values are compared as JSON text, so ``0`` does not pass for ``False`` and ``1`` does not pass for ``1.0``.
    """
    return sorted(
        key
        for key, value in fields.items()
        if key not in event or json.dumps(event[key], sort_keys=True) != json.dumps(value, sort_keys=True)
    )


def verify_saved_responses(run_dir: Path, view_states: Mapping[str, RequestState]) -> None:
    """Every ``response_saved`` event must match its response file, and the file must have the recorded hash.

    The file is the stronger evidence: it holds the provider's whole payload and its hash is in the event.
    The event's copy of the fields that the safety stop reads (``returned_model``, ``usage``,
    ``input_bound_exceeded``, ``measured_cost``, ``answer``, ``output_status`` and the rest of
    :func:`saved_event_fields`) is rebuilt from the file, and any difference refuses the run.

    ``attempts.jsonl`` has no hash chain, so this cannot detect an edit that keeps every cross-check
    consistent, for example one that rewrites the event, the response file and its recorded hash together.
    The stop protects against the code's own behaviour and honest operation, not against file editing.
    """
    for state in view_states.values():
        saved = state.saved
        if saved is None:
            continue
        event = saved.resolution_event
        assert event is not None
        path = run_dir / event["response_file"]
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RunnerRefusal(f"{path} is missing or unreadable ({exc}); event {event['seq']} records it as saved.") from exc
        actual = sha256_bytes(raw)
        _refuse(
            actual == event["response_sha256"],
            f"{path} has sha256 {actual}, event {event['seq']} records {event['response_sha256']}. The response file changed.",
        )
        try:
            fields = saved_event_fields(json.loads(raw.decode("utf-8")), actual)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise RunnerRefusal(f"{path} is not a response file ({exc!r}); event {event['seq']} records it as saved.") from exc
        differences = _event_differences(event, fields)
        _refuse(
            not differences,
            f"event {event['seq']} differs from its response file {path} in {', '.join(differences)}. "
            "The response file is the stronger evidence, so the event log was edited or is wrong. Nothing was dispatched.",
        )


def load_run(run_dir: Path, *, verify: bool = True) -> RunView:
    """Read a run directory without changing it. Raises :class:`RunnerRefusal` for a partial or corrupt run."""
    run_dir = _absolute(run_dir)
    _refuse(run_dir.is_dir(), f"{run_dir} is not a run directory")
    _refuse((run_dir / RUN_CONFIG_NAME).exists(), f"{run_dir} holds no {RUN_CONFIG_NAME}; it is not an initialised run.")
    try:
        config = json.loads((run_dir / RUN_CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"{run_dir / RUN_CONFIG_NAME} is corrupt: {exc}") from exc
    _refuse(
        isinstance(config, dict)
        and isinstance(config.get("identity"), dict)
        and canonical_digest(config["identity"]) == config.get("identity_sha256"),
        f"{run_dir / RUN_CONFIG_NAME} holds an identity that does not match its identity_sha256; the stored identity "
        "was edited. Nothing was dispatched. Keep this directory as the record and start a successor run.",
    )
    requests, requests_bytes = _read_requests(run_dir)
    _refuse(
        sha256_bytes(requests_bytes) == config.get("requests_sha256"),
        f"{run_dir / REQUESTS_NAME} does not match the hash in {RUN_CONFIG_NAME}; the prepared requests changed.",
    )
    event_file = read_event_file(run_dir / ATTEMPTS_NAME)
    states, invocations = reconstruct(requests, list(event_file.events))
    if verify:
        verify_saved_responses(run_dir, states)
    return RunView(
        run_dir,
        config,
        requests,
        requests_bytes,
        list(event_file.events),
        event_file.committed_bytes,
        event_file.uncommitted_tail,
        states,
        invocations,
    )


def check_run_directory(run_dir: Path) -> str:
    """Classify a directory: ``new`` (missing or empty), ``prepared`` (requests only) or ``run`` (has run_config.json)."""
    if not run_dir.exists():
        return "new"
    _refuse(run_dir.is_dir(), f"{run_dir} exists and is not a directory")
    names = {entry.name for entry in run_dir.iterdir()}
    stray = sorted(
        n
        for n in names
        if n not in KNOWN_RUN_FILES and not n.startswith(UNCOMMITTED_PREFIX) and not (n.startswith(".") and n.endswith(".tmp"))
    )
    _refuse(not stray, f"{run_dir} holds unrecognised entries {stray}; it is not a run directory. Pick a new run directory.")
    if RUN_CONFIG_NAME in names:
        return "run"
    written = names - {LOCK_NAME} - {n for n in names if n.startswith(".") and n.endswith(".tmp")}
    if not written:
        return "new"
    _refuse(
        written == {REQUESTS_NAME},
        f"{run_dir} holds a partial run ({sorted(written)} without {RUN_CONFIG_NAME}). Keep it and choose a new run directory.",
    )
    return "prepared"


# ---------------------------------------------------------------------------
# Live enablement
# ---------------------------------------------------------------------------


def check_live_enablement(
    *,
    mode: str,
    has_provider_config: bool,
    safety_ceiling: float | None,
    environ: Mapping[str, str],
) -> None:
    """Refuse live mode unless every requirement is met. Runs before anything is prepared, and reads no credential."""
    if mode != MODE_LIVE:
        return
    problems: list[str] = []
    if not has_provider_config:
        problems.append("--provider-config PATH")
    if safety_ceiling is None:
        problems.append("--safety-ceiling AMOUNT")
    if environ.get(LIVE_ENV_NAME) != LIVE_ENV_VALUE:
        problems.append(f"environment {LIVE_ENV_NAME}={LIVE_ENV_VALUE}")
    if problems:
        raise RunnerRefusal(
            "live mode needs all of: --mode live, " + ", ".join(problems) + ". Nothing was prepared and no client was built."
        )
    refuse_redirecting_environment(environ)


def refuse_redirecting_environment(environ: Mapping[str, str]) -> None:
    """Refuse when the environment would change where the key is sent or as whom. Reads no credential.

    The SDK reads ``os.environ``, not the mapping a caller passes, so both are checked.

    Proxy and CA variables (``HTTPS_PROXY``, ``HTTP_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``SSL_CERT_FILE``,
    ``SSL_CERT_DIR``) are not refused, because the live client never reads them: ``build_live_provider`` builds it
    with ``trust_env=False``, which also ignores the operating system's proxy settings. A run behind a proxy
    therefore fails to connect, as a connect error that sent nothing, and does not send through the proxy.
    """
    present = sorted({name for source in (environ, os.environ) for name in REDIRECTING_ENV_NAMES if name in source})
    if present:
        raise RunnerRefusal(
            f"live mode refuses to run with {', '.join(present)} set in the environment: they can send the key "
            "to another URL or bill another organization or project. Unset them and set the base URL in the "
            "provider config's endpoint. Nothing was prepared and no client was built."
        )


@dataclass(frozen=True)
class ProviderContext:
    """What a provider factory may use: the prepared requests and how many attempts each already has.

    A fresh invocation starts a fresh provider, so a scripted fake must skip the steps that earlier
    attempts consumed. ``prior_attempts`` maps ``request_id`` to that count.
    """

    requests: Sequence[Mapping[str, Any]]
    prior_attempts: Mapping[str, int]


ProviderFactory = Callable[[ProviderContext], Any]


def load_fake_script(script_path: Path) -> tuple[dict[str, list[Any]], Any, str]:
    """Read a fake script. Returns steps by question_id, the default step and the file's sha256.

    The file is a JSON object with ``script`` (question_id to a list of step objects) and an optional
    ``default`` step (default: ``{"kind": "answer"}``). Step objects take the fields of ``FakeStep``.
    """
    from .answer_providers import FakeStep

    try:
        raw = script_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"cannot read the fake script {script_path}: {exc}") from exc
    _refuse(isinstance(payload, dict), f"fake script {script_path} is not a JSON object")
    unknown = sorted(set(payload) - {"default", "script"})
    _refuse(not unknown, f"fake script {script_path} has unknown keys {unknown}; expected script and default")

    def step(value: Any) -> Any:
        _refuse(isinstance(value, dict) and "kind" in value, f"fake script step {value!r} needs a kind")
        try:
            return FakeStep(**value)
        except (TypeError, ValueError) as exc:
            raise RunnerRefusal(f"fake script step {value!r}: {exc}") from exc

    scripted = payload.get("script", {})
    _refuse(isinstance(scripted, dict), "fake script: script must map question ids to lists of steps")
    steps = {qid: [step(item) for item in items] for qid, items in scripted.items()}
    return steps, step(payload.get("default", {"kind": "answer"})), sha256_bytes(raw)


def build_fake_provider(
    steps_by_question: Mapping[str, Sequence[Any]], default: Any, config: ProviderConfig, context: ProviderContext
) -> Any:
    """A ``FakeProvider`` for this invocation, scripted by question and aligned with earlier attempts."""
    from .answer_providers import FakeProvider, script_by_question

    try:
        keyed = script_by_question(context.requests, steps_by_question)
    except ValueError as exc:
        raise RunnerRefusal(f"fake script: {exc}") from exc
    aligned = {rid: steps[context.prior_attempts.get(rid, 0) :] for rid, steps in keyed.items()}
    return FakeProvider(aligned, default=default, model=config.model)


def expected_base_url(endpoint: str) -> str:
    """The client's ``base_url`` for a config endpoint: the SDK adds a trailing slash."""
    return endpoint if endpoint.endswith("/") else endpoint + "/"


def build_live_provider(
    config: ProviderConfig, environ: Mapping[str, str], *, http_client: Any = None
) -> Any:
    """Build the real provider. Reads ``OPENAI_API_KEY`` here and nowhere else.

    ``http_client`` is for tests, which pass an ``httpx.Client`` on a mock transport. The CLI never sets it.
    The HTTP client must be direct, and the builder refuses one that is not (``http_client_problems``):

    * It must not follow redirects. The SDK's own default client does (openai-python 1.68.2,
      ``_base_client.py:754``), and a 307 or 308 makes it post the prompt again to the ``Location`` URL, outside
      the driver's accounting.
    * It must not trust the environment. With ``trust_env=True`` (the httpx default) httpx sends requests through
      ``HTTPS_PROXY``, ``HTTP_PROXY`` or ``ALL_PROXY``, through the operating system's proxy settings
      (``urllib.request.getproxies`` reads them on macOS), and takes its CA bundle from ``SSL_CERT_FILE`` or
      ``SSL_CERT_DIR``. Any of these would carry the prompt and the key through a route the run does not record.
    * It must have no proxy of its own.

    So the builder passes ``openai.DefaultHttpxClient(follow_redirects=False, trust_env=False)``, which keeps the
    SDK's other defaults. Consequences: proxy variables are ignored, so a machine that can reach OpenAI only
    through a proxy gets connect errors, and a custom CA bundle or a TLS-intercepting proxy is not supported.

    The client sends to ``config.endpoint`` and to nothing else: the redirecting environment variables are
    refused first, and the built client's ``base_url`` must equal the value recorded in the run identity.
    The provider keeps the default storage policy (``store=false``) and the Standard service tier on every
    request. ``store=false`` is not zero retention, and it does not settle the dataset's licensing.
    """
    refuse_redirecting_environment(environ)
    import openai

    from .answer_providers import OpenAIChatProvider, http_client_problems

    key = environ.get("OPENAI_API_KEY")
    _refuse(bool(key), "OPENAI_API_KEY is not set")
    if http_client is None:
        http_client = openai.DefaultHttpxClient(follow_redirects=False, trust_env=False)
    problems = http_client_problems(http_client)
    _refuse(
        not problems,
        "the HTTP client is not direct: " + "; ".join(problems) + ". A redirect would send the prompt to another URL, and "
        "a proxy or CA setting from the environment would route the prompt and the key outside the run's records. "
        "Nothing was sent.",
    )
    client = openai.OpenAI(
        api_key=key, base_url=config.endpoint, max_retries=0, timeout=config.timeout_seconds, http_client=http_client
    )
    _refuse(
        str(client.base_url) == expected_base_url(config.endpoint),
        f"the client base_url {str(client.base_url)!r} differs from the config endpoint {config.endpoint!r}; nothing was sent",
    )
    return OpenAIChatProvider(
        config.model, dict(config.params), client=client, token_limit_param=config.token_limit_param
    )


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    project_root: Path
    run_dir: Path
    mode: str = MODE_FAKE
    pilot_id: str = "ohr_dev_v1"
    runtime_manifest_path: Path | None = None
    run_id: str | None = None
    safety_ceiling: float | None = None
    raise_safety_ceiling: float | None = None
    authorization_note: str | None = None
    config: ProviderConfig | None = None
    cli_script: Path | None = None
    code_root: Path | None = None
    command: Sequence[str] | None = None


@dataclass(frozen=True)
class LiveResult:
    exit_code: int
    message: str
    summary: Mapping[str, Any] | None = None


def _validate_amount(value: float | None, what: str) -> float:
    _refuse(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0,
        f"{what} must be a finite amount above zero, got {value!r}",
    )
    return float(value)  # type: ignore[arg-type]


def _resolve_config(options: RunOptions) -> ProviderConfig:
    if options.mode == MODE_FAKE:
        return options.config or FAKE_CONFIG
    _refuse(options.config is not None, "live mode needs a provider config")
    assert options.config is not None
    return options.config


def _initialise(
    run_dir: Path,
    *,
    options: RunOptions,
    run_id: str,
    identity: dict[str, Any],
    identity_hash: str,
    requests_text: str,
    retrieval_run: RetrievalRun,
    config: ProviderConfig,
) -> None:
    """Write ``requests.jsonl`` and then ``run_config.json``. The config marks the run as initialised."""
    atomic_write_text(run_dir / REQUESTS_NAME, requests_text)
    git = git_provenance(options.code_root, exclude=[run_dir])
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "pilot_id": retrieval_run.pilot_id,
        "mode": options.mode,
        "kind": KIND_FAKE if options.mode == MODE_FAKE else KIND_LIVE,
        "label": LABEL_FAKE if options.mode == MODE_FAKE else LABEL_LIVE,
        "contract_version": CONTRACT_VERSION,
        "created_at": _now(),
        "identity": identity,
        "identity_sha256": identity_hash,
        "requests_sha256": _sha_text(requests_text),
        "provenance": {
            **git,
            "code_root": _display_path(options.code_root, options.project_root) if options.code_root else None,
            "environment": _package_versions(),
            "command": list(options.command) if options.command is not None else None,
        },
        "runtime_manifest": {
            "path": _display_path(retrieval_run.runtime_manifest_path, options.project_root),
            "sha256": retrieval_run.runtime_manifest_sha256,
        },
        "inputs_read": ["runtime manifest", "MinerU JSON file of each declared document"],
        "inputs_never_read": [
            "evaluation_manifest.json (only the score subcommand opens it)",
            "selection_record.json",
            "inspection/",
            "OHR-Bench/data/qas*.json",
            "gt text",
            "annotations",
        ],
        "safety_ceiling_note": "The safety ceiling is an operating limit and is not part of the identity. Each invocation_started records it.",
    }
    atomic_write_text(run_dir / RUN_CONFIG_NAME, _dumps(run_config) + "\n")



PROVIDER_RAW_LIMIT = 20_000
# Fields that say what a reply cost, which request it was, and why it failed. They come first when a mapping
# has more keys than _RAW_MAX_KEYS, so a wide payload cannot push them out.
_RAW_PRIORITY_KEYS = ("usage", "id", "model", "reason", "http_status", "x_request_id", "error_code", "error_type", "payload", "body")
_RAW_MAX_KEYS = 16
_RAW_MAX_DEPTH = 2
_RAW_FIELD_LIMIT = 1_500
_RAW_KEY_LIMIT = 100


def _raw_key_rank(key: Any) -> tuple[int, int, str]:
    name = str(key)
    return (0, _RAW_PRIORITY_KEYS.index(name), name) if name in _RAW_PRIORITY_KEYS else (1, 0, name)


def _shrink_raw(value: Any, field_limit: int, depth: int) -> Any:
    """``value`` with every part longer than ``field_limit`` characters cut and marked.

    A mapping is cut key by key, so its small fields stay structured JSON. Below ``depth`` levels, or for any
    other type, a long part becomes ``{"truncated": true, "chars": N, "text": <prefix>}``. A mapping with more
    than ``_RAW_MAX_KEYS`` keys keeps the priority keys and then the first keys in sorted order, and names how
    many it left out under ``omitted_keys``.
    """
    if len(_line(value)) <= field_limit:
        return value
    if isinstance(value, dict) and depth > 0:
        keys = sorted(value, key=_raw_key_rank)
        shrunk = {str(k)[:_RAW_KEY_LIMIT]: _shrink_raw(value[k], field_limit, depth - 1) for k in keys[:_RAW_MAX_KEYS]}
        if len(keys) > _RAW_MAX_KEYS:
            shrunk["omitted_keys"] = {"count": len(keys) - _RAW_MAX_KEYS, "names": [str(k)[:60] for k in keys[_RAW_MAX_KEYS : _RAW_MAX_KEYS + 8]]}
        return shrunk
    text = _line(value)
    return {"truncated": True, "chars": len(text), "text": text[:field_limit]}


def _bounded_raw(raw: Any) -> Any:
    """The provider payload of an unknown outcome, kept as evidence for reconciliation and billing checks.

    A payload that serialises and fits in ``PROVIDER_RAW_LIMIT`` characters is stored as it is. A larger one
    keeps its small top-level fields as structured JSON, and for a nested ``payload`` its small fields too, so
    ``usage``, ``id`` and ``model`` survive next to a large body. Only the large values are cut, each marked
    with ``truncated``, its original ``chars`` and a text prefix. If the result is still over the limit the
    per-field limit is halved, and as a last resort the record holds a prefix of the whole text plus the
    ``usage``, ``id`` and ``model`` it could find. The stored record never exceeds ``PROVIDER_RAW_LIMIT``.
    """
    if not raw:
        return None
    try:
        text = _line(raw)
    except (TypeError, ValueError):
        return {"unserialisable": True, "text": repr(raw)[:PROVIDER_RAW_LIMIT - 100]}
    if len(text) <= PROVIDER_RAW_LIMIT:
        return json.loads(text)
    limit = _RAW_FIELD_LIMIT
    while limit >= 16:
        shrunk = _shrink_raw(json.loads(text), limit, _RAW_MAX_DEPTH)
        if isinstance(shrunk, dict) and len(_line(shrunk)) <= PROVIDER_RAW_LIMIT:
            return shrunk
        limit //= 2
    parsed = json.loads(text)
    nested = parsed.get("payload") if isinstance(parsed, dict) and isinstance(parsed.get("payload"), dict) else {}
    kept: dict[str, Any] = {}
    for key in ("usage", "id", "model"):
        found = parsed.get(key) if isinstance(parsed, dict) else None
        found = nested.get(key) if found is None else found
        if found is not None and len(_line(found)) <= _RAW_FIELD_LIMIT:
            kept[key] = found
    return {"truncated": True, "chars": len(text), **kept, "text": text[: PROVIDER_RAW_LIMIT // 2]}


class _Driver:
    """One invocation. Holds the run lock for its whole life."""

    def __init__(
        self,
        *,
        view: RunView,
        log: EventLog,
        services: Services,
        config: ProviderConfig,
        provider: Any,
        ceiling: float,
        sleep: Callable[[float], None],
        crash_hook: CrashHook | None,
        clock: Callable[[], float],
    ) -> None:
        self.view = view
        self.log = log
        self.services = services
        self.config = config
        self.provider = provider
        self.ceiling = ceiling
        self.sleep = sleep
        self.crash_hook = crash_hook
        self.clock = clock
        self.consecutive_failures = 0
        self.violations: list[dict[str, Any]] = []

    def safety_stopped(self) -> bool:
        """True when the durable records hold a safety violation. Rebuilt from the events every time it is asked."""
        self.violations = assess_safety(self.view, self.services)
        return bool(self.violations)

    def breaker_record(self) -> dict[str, int]:
        return {"consecutive_failures": self.consecutive_failures, "threshold": CIRCUIT_BREAKER_THRESHOLD}

    def hook(self, point: str, **info: Any) -> None:
        if self.crash_hook is not None:
            self.crash_hook(point, info)

    def ledger(self) -> Any:
        return self.services.ledger_from_events(self.view.events, self.config.prices)

    def ledger_totals(self) -> dict[str, float]:
        ledger = self.ledger()
        return {
            "measured": ledger.measured,
            "reserved": ledger.reserved,
            "committed_upper": ledger.committed_upper,
        }

    # -- steps ---------------------------------------------------------------

    def run_all(self) -> str:
        """Dispatch every pending request in manifest order. Returns the reason the invocation ended."""
        for state in self.view.ordered_states:
            if state.status != REQUEST_PENDING:
                continue
            outcome = self.process(state)
            if outcome in (END_BUDGET, END_STOP, END_CIRCUIT, END_SAFETY_STOP):
                return outcome
        # The last response may be the offending one. Nothing is left to dispatch, but the run still ends as a safety stop.
        if self.safety_stopped():
            return END_SAFETY_STOP
        return END_RECONCILIATION if self.view.by_status(REQUEST_UNKNOWN) else END_COMPLETED

    def process(self, state: RequestState) -> str | None:
        """Send one request until it resolves. Returns an end reason when the whole invocation must stop."""
        request = state.request
        while True:
            # Before every dispatch, including a retry: a saved response that broke a safety assumption ends the invocation.
            if self.safety_stopped():
                return END_SAFETY_STOP
            attempt = state.next_attempt
            attempt_id = f"{state.request_id}-a{attempt}"
            limit = self.services.retry_parameters().get("max_attempts")
            _refuse(
                not isinstance(limit, int) or state.attempts_in_window < limit,
                f"question {state.question_id} already has {state.attempts_in_window} attempts since its last reopen, the retry policy's limit",
            )
            upper = request["cost_upper_bound"]
            if not self.ledger().can_reserve(
                upper, self.ceiling, ceiling_simulated=self.config.prices.simulated, ceiling_currency=self.config.prices.currency
            ):
                return END_BUDGET
            self.hook(CRASH_BEFORE_DISPATCH, question_id=state.question_id, attempt=attempt)
            dispatch = self.log.append(
                EVENT_DISPATCH_STARTED,
                question_id=state.question_id,
                request_id=state.request_id,
                attempt=attempt,
                attempt_id=attempt_id,
                cost_upper_bound=upper,
                ledger_before=self.ledger_totals(),
            )
            attempt_state = AttemptState(attempt, attempt_id, dispatch)
            state.attempts.append(attempt_state)
            self.hook(CRASH_AFTER_DISPATCH, question_id=state.question_id, attempt=attempt)
            provider_request = ProviderRequest(
                request_id=state.request_id,
                attempt_id=attempt_id,
                messages=tuple(
                    {"role": role, "content": request[key]} for role, key in (("system", "system"), ("user", "user"))
                ),
                params=dict(self.config.params),
                max_output_tokens=self.config.max_output_tokens,
                timeout_seconds=self.config.timeout_seconds,
            )
            started = self.clock()
            try:
                response = self.provider.send(provider_request)
            except ProviderError as error:
                latency = int((self.clock() - started) * 1000)
                step = self._record_error(state, attempt_state, error, latency)
                counted = step == "unknown" or step == DECISION_FAIL
                if counted:
                    self.consecutive_failures += 1
                if step == DECISION_RETRY:
                    continue
                if step == DECISION_STOP:
                    return END_STOP
                return self._breaker_tripped()
            except Exception as error:  # the provider broke its contract: the request may have been processed
                latency = int((self.clock() - started) * 1000)
                self._record_unknown(
                    state, attempt_state, kind="provider_exception", message=f"{type(error).__name__}: {error}", latency=latency
                )
                self.consecutive_failures += 1
                return self._breaker_tripped()
            latency = int((self.clock() - started) * 1000)
            self.hook(CRASH_AFTER_SEND, question_id=state.question_id, attempt=attempt)
            self._save_response(state, attempt_state, response, latency)
            self.consecutive_failures = 0
            return None

    def _breaker_tripped(self) -> str | None:
        return END_CIRCUIT if self.consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD else None

    def _record_unknown(
        self, state: RequestState, attempt: AttemptState, *, kind: str, message: str, latency: int | None, **extra: Any
    ) -> None:
        event = self.log.append(
            EVENT_OUTCOME_UNKNOWN,
            question_id=state.question_id,
            request_id=state.request_id,
            attempt=attempt.attempt,
            attempt_id=attempt.attempt_id,
            kind=kind,
            message=message[:FAILURE_MESSAGE_LIMIT],
            latency_ms=latency,
            **extra,
        )
        attempt.outcome, attempt.resolution_event = "unknown", event

    def _record_error(self, state: RequestState, attempt: AttemptState, error: ProviderError, latency: int) -> str:
        """Record a provider error. Returns ``retry``, ``fail_question``, ``stop_run`` or ``unknown``."""
        if error.outcome == OUTCOME_UNKNOWN:
            self._record_unknown(
                state, attempt, kind=error.kind, message=str(error), latency=latency, provider_raw=_bounded_raw(error.raw)
            )
            return "unknown"
        decision = self.services.retry_policy.decide(error, state.attempts_in_window)
        action = decision.action
        _refuse(
            action in (DECISION_RETRY, DECISION_FAIL, DECISION_STOP),
            f"the retry policy returned {action!r} for a {error.outcome} error; expected retry, fail_question or stop_run",
        )
        limit = self.services.retry_parameters().get("max_attempts")
        # A stop_run error ends the invocation, but the attempt still counts toward the limit. On the last
        # allowed attempt the question ends too, so repeated stop_run errors cannot buy a fourth attempt.
        exhausted = action == DECISION_STOP and isinstance(limit, int) and state.attempts_in_window >= limit
        event = self.log.append(
            EVENT_ATTEMPT_FAILED,
            question_id=state.question_id,
            request_id=state.request_id,
            attempt=attempt.attempt,
            attempt_id=attempt.attempt_id,
            kind=error.kind,
            outcome=error.outcome,
            http_status=error.http_status,
            retryable=error.retryable,
            message=str(error)[:FAILURE_MESSAGE_LIMIT],
            latency_ms=latency,
            decision=action,
            delay_seconds=decision.delay_seconds if action == DECISION_RETRY else None,
            attempts_exhausted=exhausted,
            # The provider's request id and error body, for a later check of what the provider recorded.
            provider_raw=_bounded_raw(error.raw),
        )
        attempt.outcome, attempt.resolution_event = "failed", event
        if action == DECISION_RETRY:
            self.sleep(decision.delay_seconds)
        return action

    def _save_response(self, state: RequestState, attempt: AttemptState, response: ProviderResponse, latency: int) -> None:
        parsed = dict(self.services.parse_reply(response.text, response.finish_reason, response.refusal))
        try:
            measured = self.services.measured_cost(response.usage, self.config.prices)
        except (ValueError, TypeError):
            measured = None
        upper = state.request["input_token_upper_bound"]
        reported_input = response.usage.input_tokens
        payload = {
            "schema_version": SCHEMA_VERSION,
            "question_id": state.question_id,
            "request_id": state.request_id,
            "attempt": attempt.attempt,
            "attempt_id": attempt.attempt_id,
            "saved_by_invocation": self.log.invocation_id,
            "returned_model": response.returned_model,
            "response_id": response.response_id,
            "finish_reason": response.finish_reason,
            "usage": response.usage.as_dict(),
            "measured_cost": measured,
            "input_bound_exceeded": reported_input is not None and reported_input > upper,
            "latency_ms": latency,
            "output_status": parsed["output_status"],
            "answer": parsed["answer"],
            "abstained": parsed["abstained"],
            "provider_response": {
                "text": response.text,
                "finish_reason": response.finish_reason,
                "refusal": response.refusal,
                "raw": response.raw,
            },
        }
        text = _dumps(payload) + "\n"
        path = _response_path(self.view.run_dir, attempt.attempt_id)
        atomic_write_text(path, text)
        self.hook(CRASH_AFTER_RESPONSE_FILE, question_id=state.question_id, attempt=attempt.attempt)
        event = self.log.append(EVENT_RESPONSE_SAVED, **saved_event_fields(payload, _sha_text(text)))
        attempt.outcome, attempt.resolution_event = "saved", event
        self.hook(CRASH_AFTER_SAVED, question_id=state.question_id, attempt=attempt.attempt)


def saved_event_fields(payload: Mapping[str, Any], file_sha256: str) -> dict[str, Any]:
    """The ``response_saved`` fields, derived from the response file so that recovery can rebuild them."""
    return {
        "question_id": payload["question_id"],
        "request_id": payload["request_id"],
        "attempt": payload["attempt"],
        "attempt_id": payload["attempt_id"],
        "response_file": f"{RESPONSES_DIR}/{payload['attempt_id']}.json",
        "response_sha256": file_sha256,
        "returned_model": payload["returned_model"],
        "response_id": payload["response_id"],
        "finish_reason": payload["finish_reason"],
        "usage": payload["usage"],
        "measured_cost": payload["measured_cost"],
        "input_bound_exceeded": payload["input_bound_exceeded"],
        "latency_ms": payload["latency_ms"],
        "output_status": payload["output_status"],
        "answer": payload["answer"],
        "abstained": payload["abstained"],
    }


# ---------------------------------------------------------------------------
# Recovery at the start of an invocation
# ---------------------------------------------------------------------------


def quarantine_uncommitted_tail(run_dir: Path, view: RunView, invocation_id: str) -> dict[str, Any] | None:
    """Move an uncommitted final line out of ``attempts.jsonl`` so appends start on a clean line.

    The bytes are kept in ``attempts.uncommitted.<invocation_id>``. Returns a record of what was moved.
    """
    tail = view.uncommitted_tail
    if not tail:
        return None
    target = run_dir / f"{UNCOMMITTED_PREFIX}{invocation_id}"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, tail)
        sync_fd(fd)
    finally:
        os.close(fd)
    log_fd = os.open(run_dir / ATTEMPTS_NAME, os.O_WRONLY)
    try:
        os.ftruncate(log_fd, view.committed_bytes)
        sync_fd(log_fd)
    finally:
        os.close(log_fd)
    _fsync_dir(run_dir)
    return {
        "file": target.name,
        "bytes": len(tail),
        "sha256": sha256_bytes(tail),
    }


def _read_recoverable_response(run_dir: Path, state: RequestState, attempt: AttemptState) -> tuple[dict[str, Any], bytes] | None:
    """The parsed response file of an attempt that has ``dispatch_started`` and no resolving event, or None.

    None when the file is absent or is not valid JSON. Raises :class:`RunnerRefusal` for a file that parses
    but names another attempt, question or request, or lacks a field that ``response_saved`` copies from it.
    """
    path = _response_path(run_dir, attempt.attempt_id)
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    _refuse(isinstance(payload, dict), f"response file {path} is not a JSON object")
    for key, expected in (
        ("attempt_id", attempt.attempt_id),
        ("attempt", attempt.attempt),
        ("question_id", state.question_id),
        ("request_id", state.request_id),
    ):
        _refuse(
            key in payload and json.dumps(payload[key]) == json.dumps(expected),
            f"response file {path} does not describe attempt {attempt.attempt_id}: {key} is {payload.get(key)!r}, expected {expected!r}",
        )
    try:
        saved_event_fields(payload, sha256_bytes(raw))
    except KeyError as exc:
        raise RunnerRefusal(f"response file {path} lacks the field {exc}, so its event cannot be rebuilt from it") from exc
    return payload, raw


def recover_open_attempts(
    view: RunView, log: EventLog, run_dir: Path, *, tail: Mapping[str, Any] | None = None
) -> list[str]:
    """Resolve every attempt that has ``dispatch_started`` and nothing after it.

    If its response file exists and parses, append ``response_saved`` from the file, so the
    answer is kept and nothing is resent (crash window 3, file written). Otherwise append
    ``outcome_unknown`` citing the invocation that dispatched it (windows 2 and 3, no file).
    ``tail`` describes an uncommitted tail that the caller moved aside. It is recorded on the first event
    appended, because a caller that writes no ``invocation_started`` has no other event to carry it.
    Returns the attempt ids handled.
    """
    handled: list[str] = []
    carried: dict[str, Any] = {} if tail is None else {"recovered_uncommitted_tail": dict(tail)}
    for state in view.ordered_states:
        for attempt in state.orphans:
            found = _read_recoverable_response(run_dir, state, attempt)
            payload, raw = found if found is not None else (None, b"")
            if payload is not None:
                event = log.append(
                    EVENT_RESPONSE_SAVED,
                    **saved_event_fields(payload, sha256_bytes(raw)),
                    recovered=True,
                    recovered_from_invocation_id=attempt.dispatch["invocation_id"],
                    **carried,
                )
                attempt.outcome = "saved"
            else:
                event = log.append(
                    EVENT_OUTCOME_UNKNOWN,
                    question_id=state.question_id,
                    request_id=state.request_id,
                    attempt=attempt.attempt,
                    attempt_id=attempt.attempt_id,
                    kind="orphaned_dispatch",
                    message=(
                        f"dispatch_started (event {attempt.dispatch['seq']}, invocation {attempt.dispatch['invocation_id']}) "
                        "has no response, failure or unknown outcome. The request may have been processed."
                    ),
                    latency_ms=None,
                    recovered=True,
                    recovered_from_invocation_id=attempt.dispatch["invocation_id"],
                    **carried,
                )
                attempt.outcome = "unknown"
            carried = {}
            attempt.resolution_event = event
            handled.append(attempt.attempt_id)
    return handled


# ---------------------------------------------------------------------------
# Safety stop
# ---------------------------------------------------------------------------


def _recoverable_saved_events(view: RunView) -> list[dict[str, Any]]:
    """``response_saved`` events that recovery would append for response files whose event was lost.

    A response file written before the crash is a durable record of what the provider returned. Reading it
    here lets a restart, ``status`` and ``export`` see its violation before ``recover_open_attempts`` runs.
    Nothing is written.
    """
    events: list[dict[str, Any]] = []
    for state in view.ordered_states:
        for attempt in state.orphans:
            found = _read_recoverable_response(view.run_dir, state, attempt)
            if found is None:
                continue
            payload, raw = found
            fields = saved_event_fields(payload, sha256_bytes(raw))
            events.append({"event": EVENT_RESPONSE_SAVED, **fields, "recovered": True})
    return events


def fold_recoverable_responses(view: RunView) -> RunView:
    """A copy of ``view`` in which every attempt with a readable response file and no event counts as saved.

    ``status`` and ``export`` read the run without writing to ``attempts.jsonl``. This gives them the state that
    :func:`recover_open_attempts` would write, so a response the provider returned is reported as an answer and
    not as an attempt awaiting reconciliation. The events are in memory only. Nothing is written.
    """
    recoverable = {event["attempt_id"]: event for event in _recoverable_saved_events(view)}
    if not recoverable:
        return view
    states = {}
    for request_id, state in view.states.items():
        attempts = [
            dataclasses.replace(a, outcome="saved", resolution_event=recoverable[a.attempt_id]) if a.attempt_id in recoverable else a
            for a in state.attempts
        ]
        states[request_id] = dataclasses.replace(state, attempts=attempts)
    return dataclasses.replace(view, events=[*view.events, *recoverable.values()], states=states)


def _violation(condition: str, attempt_id: str | None, question_id: str | None, detail: str) -> dict[str, Any]:
    return {"condition": condition, "attempt_id": attempt_id, "question_id": question_id, "detail": detail}


def find_safety_violations(view: RunView, services: Services) -> list[dict[str, Any]]:
    """Every safety violation in the durable records, in event order, then ledger anomalies.

    Read only, and a function of the records alone: a restart, ``status`` and ``export`` get the same list
    the running driver saw. Three conditions, each named in the result:

    * ``input_bound_exceeded``: a ``response_saved`` event reports more input tokens than the request's
      input upper bound. A response with no reported input tokens shows no exceedance. Its cost stays
      reserved at the upper bound in the ledger, so the ceiling still holds.
    * ``returned_model_mismatch``: the returned model fails :func:`returned_model_matches` against the
      model in the run identity.
    * ``ledger_anomaly``: ``SafetyLedger.anomalies`` is not empty. All anomaly kinds block.
    """
    requested = view.config["identity"]["provider"]["model"]
    by_request = {request["request_id"]: request for request in view.requests}
    events = list(view.events) + _recoverable_saved_events(view)
    found: list[dict[str, Any]] = []
    for event in events:
        if event["event"] != EVENT_RESPONSE_SAVED:
            continue
        attempt_id, question_id = event["attempt_id"], event["question_id"]
        bound = by_request.get(event["request_id"], {}).get("input_token_upper_bound")
        reported = (event.get("usage") or {}).get("input_tokens")
        counts = (int, float)
        over = (
            isinstance(reported, counts)
            and not isinstance(reported, bool)
            and isinstance(bound, counts)
            and not isinstance(bound, bool)
            and reported > bound
        )
        if over or event.get("input_bound_exceeded") is True:
            found.append(
                _violation(
                    VIOLATION_INPUT_BOUND,
                    attempt_id,
                    question_id,
                    f"the response reported {reported} input tokens, above the input upper bound {bound}",
                )
            )
        if not returned_model_matches(requested, event.get("returned_model")):
            found.append(
                _violation(
                    VIOLATION_RETURNED_MODEL,
                    attempt_id,
                    question_id,
                    f"the response came from model {event.get('returned_model')!r}, the run requested {requested!r}",
                )
            )
    prices = PriceTable(**view.config["identity"]["prices"])
    question_of = {a.attempt_id: s.question_id for s in view.states.values() for a in s.attempts}
    for text in services.ledger_from_events(events, prices).anomalies:
        # SafetyLedger writes "<attempt_id>: <what is wrong>". An anomaly in another form has no attempt id.
        head = str(text).split(": ", 1)[0]
        attempt_id = head if head in question_of else None
        found.append(_violation(VIOLATION_LEDGER_ANOMALY, attempt_id, question_of.get(head), str(text)))
    return found


def assess_safety(view: RunView, services: Services) -> list[dict[str, Any]]:
    """Set ``view.safety_violations`` from the records and return it. Call it again after events are appended."""
    view.safety_violations = find_safety_violations(view, services)
    return view.safety_violations


def describe_violations(violations: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"{v['condition']} at {v['attempt_id'] or 'no single attempt'}: {v['detail']}" for v in violations
    )


def model_note(violations: Sequence[Mapping[str, Any]]) -> str:
    """The exact-match warning, only when a returned-model mismatch is among the violations."""
    return SAFETY_MODEL_NOTE if any(v["condition"] == VIOLATION_RETURNED_MODEL for v in violations) else ""


def safety_stop_message(run_dir: Path, violations: Sequence[Mapping[str, Any]]) -> str:
    return (
        f"run in {run_dir} is safety_stopped, so no request was sent. Violations: {describe_violations(violations)}. "
        f"{SAFETY_NO_CLEAR_NOTE} {SAFETY_LIMIT_NOTE} {model_note(violations)}".rstrip()
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _failure(kind: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"stage": "dispatch", "type": kind, "message": message[:FAILURE_MESSAGE_LIMIT], **extra}


def build_prediction(state: RequestState, *, run_state: str, last_reason: str | None) -> dict[str, Any]:
    """One terminal record for a question. Statuses are only ``answered``, ``no_evidence`` and ``execution_failed``."""
    request = state.request
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "question_id": request["question_id"],
        "doc_id": request["doc_id"],
        "request_id": request["request_id"],
        "status": STATUS_EXECUTION_FAILED,
        "answer": None,
        "abstained": False,
        "output_status": None,
        "no_evidence_reason": None,
        "skip_reason": request["skip_reason"],
        "unserved": False,
        "awaiting_reconciliation": False,
        "attempt_ids": [a.attempt_id for a in state.attempts],
        "reopen_count": len(state.reopen_marks),
        "query_retrieval_tokens": request["query_retrieval_tokens"],
        "evidence": [{k: v for k, v in item.items() if k != "text"} for item in request["evidence"]],
        "ocr_condition": request["ocr_condition"],
        "failure": None,
        "returned_model": None,
        "response_id": None,
        "finish_reason": None,
        "usage": None,
        "measured_cost": None,
    }
    status = state.status
    if status == REQUEST_SKIP:
        if request["skip_reason"] in PREPARE_FAILURE_SKIPS:
            failure = request["prepare_failure"]
            record["failure"] = {"stage": "prepare", "type": failure["type"], "message": failure["message"], "step": failure["stage"]}
        elif request["skip_reason"] == SKIP_PROMPT_OVER_LIMIT:
            record["failure"] = {
                "stage": "prepare",
                "type": SKIP_PROMPT_OVER_LIMIT,
                "message": f"input token upper bound {request['input_token_upper_bound']} is over the input limit; nothing was sent",
            }
        else:
            record.update(
                status=STATUS_NO_EVIDENCE, answer="", abstained=True, no_evidence_reason=request["no_evidence_reason"]
            )
    elif status == REQUEST_ANSWERED:
        saved = state.saved
        assert saved is not None and saved.resolution_event is not None
        event = saved.resolution_event
        record.update(
            status=STATUS_ANSWERED,
            answer=event["answer"],
            abstained=event["abstained"],
            output_status=event["output_status"],
            returned_model=event["returned_model"],
            response_id=event["response_id"],
            finish_reason=event["finish_reason"],
            usage=event["usage"],
            measured_cost=event["measured_cost"],
        )
    elif status == REQUEST_FAILED:
        marked = state.marked_failed
        if marked is not None:
            assert marked.reconciled is not None
            record["failure"] = _failure(
                "outcome_unknown_marked_failed", marked.reconciled["note"], attempt_id=marked.attempt_id
            )
        else:
            last = state.attempts[-1].resolution_event
            assert last is not None
            record["failure"] = _failure(
                last["kind"], last["message"], http_status=last.get("http_status"), attempts=len(state.attempts)
            )
            if last.get("attempts_exhausted") is True:
                record["failure"]["attempts_exhausted"] = True
    elif status == REQUEST_UNKNOWN:
        record["awaiting_reconciliation"] = True
        open_attempt = state.unresolved_unknown or state.orphans[0]
        record["failure"] = _failure(
            "outcome_unknown",
            "the request may have been processed; run reconcile for this attempt",
            attempt_id=open_attempt.attempt_id,
        )
    else:
        record["unserved"] = True
        kind = "budget_exhausted" if last_reason == END_BUDGET else "not_attempted"
        message = "no request was sent for this question"
        if state.reopen_marks:
            message = "a person reopened this question; no attempt has been made since"
        if run_state == RUN_SAFETY_STOPPED:
            kind = FAILURE_SAFETY_STOP
            message = "no request was sent for this question because the run stopped on a safety violation; see run_summary.json"
        record["failure"] = _failure(kind, message, attempts=len(state.attempts))
    return record


VALID_BASELINE_MEANS = (
    "valid_baseline checks mechanical completeness only. It is true when the run is live, its state is complete, "
    "no question is execution_failed, every returned model equals the requested model, no response reported more "
    "input tokens than the input bound and the safety ledger reports no anomaly. It does not mean the prompt, "
    "the model, the retrieval settings or the budget are approved."
)


def returned_model_matches(requested: str, returned: str | None) -> bool:
    """The rule for a saved response: the returned model string must equal the requested one exactly.

    A requested dated snapshot (``gpt-4o-2024-11-20``) is returned unchanged, so it matches. No alias mapping
    is defined: a run that requests an alias and gets a dated snapshot back counts as a mismatch, so a
    baseline must name the snapshot it wants. A response without a returned model cannot be checked and
    counts as a mismatch.
    """
    return bool(returned) and returned == requested


def summarise_run(view: RunView, predictions_text: str, services: Services) -> dict[str, Any]:
    predictions = [json.loads(line) for line in predictions_text.splitlines()]
    counts = {STATUS_ANSWERED: 0, STATUS_NO_EVIDENCE: 0, STATUS_EXECUTION_FAILED: 0}
    output_status: dict[str, int] = {}
    for record in predictions:
        counts[record["status"]] += 1
        if record["output_status"] is not None:
            output_status[record["output_status"]] = output_status.get(record["output_status"], 0) + 1
    prices = PriceTable(**view.config["identity"]["prices"])
    ledger = services.ledger_from_events(view.events, prices)
    run_state = view.run_state
    mode = view.config["mode"]
    failed = counts[STATUS_EXECUTION_FAILED]
    attempts = [a for s in view.ordered_states for a in s.attempts]
    by_outcome = {"saved": 0, "failed": 0, "unknown": 0, "open": 0}
    for attempt in attempts:
        by_outcome[attempt.outcome or "open"] += 1
    if run_state == RUN_SAFETY_STOPPED:
        exit_code = EXIT_SAFETY_STOPPED
    elif run_state == RUN_COMPLETE:
        exit_code = EXIT_OK if failed == 0 else EXIT_EXECUTION_FAILED
    elif run_state == RUN_BUDGET_LIMITED:
        exit_code = EXIT_BUDGET_LIMITED
    else:
        exit_code = EXIT_NEEDS_ATTENTION
    last_ceiling = view.effective_ceiling
    requested_model = view.config["identity"]["provider"]["model"]
    saved_events = [a.resolution_event for a in attempts if a.outcome == "saved" and a.resolution_event]
    returned_models: dict[str, int] = {}
    for event in saved_events:
        key = event.get("returned_model") or "(none)"
        returned_models[key] = returned_models.get(key, 0) + 1
    mismatches = sum(1 for event in saved_events if not returned_model_matches(requested_model, event.get("returned_model")))
    bound_exceeded = sum(1 for event in saved_events if event.get("input_bound_exceeded"))
    anomalies = list(ledger.anomalies)
    violations = list(view.safety_violations or [])
    blockers = []
    if violations:
        blockers.append(f"the run stopped on {len(violations)} safety violation(s): {describe_violations(violations)}")
    if mode != MODE_LIVE:
        blockers.append(f"the run is a {mode} run, not a live run")
    if run_state != RUN_COMPLETE:
        blockers.append(f"the run state is {run_state}, not complete")
    if failed:
        blockers.append(f"{failed} question(s) are execution_failed")
    if mismatches:
        blockers.append(f"{mismatches} response(s) came from a returned model that differs from the requested {requested_model}")
    if bound_exceeded:
        blockers.append(f"{bound_exceeded} response(s) reported more input tokens than the input bound")
    if anomalies:
        blockers.append(f"the safety ledger reports {len(anomalies)} anomaly(ies)")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": view.config["run_id"],
        "pilot_id": view.config["pilot_id"],
        "mode": mode,
        "kind": view.config["kind"],
        "label": view.config["label"],
        "identity_sha256": view.config["identity_sha256"],
        "run_state": run_state,
        # True only when nothing blocks it. The lead sets any failure-rate limit and approves the prompt, model, retrieval and budget.
        "valid_baseline": not blockers,
        "valid_baseline_blockers": blockers,
        "valid_baseline_means": VALID_BASELINE_MEANS,
        "exit_code": exit_code,
        "counts": {
            "questions": len(predictions),
            "answered": counts[STATUS_ANSWERED],
            "no_evidence": counts[STATUS_NO_EVIDENCE],
            "execution_failed": failed,
            "abstained": sum(1 for r in predictions if r["abstained"]),
            "model_abstained": sum(1 for r in predictions if r["status"] == STATUS_ANSWERED and r["abstained"]),
            "unserved": sum(1 for r in predictions if r["unserved"]),
            "awaiting_reconciliation": sum(1 for r in predictions if r["awaiting_reconciliation"]),
            "prompt_over_limit": sum(1 for r in predictions if r["skip_reason"] == SKIP_PROMPT_OVER_LIMIT),
        },
        "output_status": dict(sorted(output_status.items())),
        "attempts": {"total": len(attempts), **by_outcome},
        "input_bound_exceeded": bound_exceeded,
        **({"safety_violations": violations} if violations else {}),  # absent for a clean run, so earlier exports stay byte-identical
        "requested_model": requested_model,
        "returned_model_mismatch": mismatches,
        "returned_models": dict(sorted(returned_models.items())),
        "cost": {
            "currency": prices.currency,
            "simulated": prices.simulated,
            "measured": ledger.measured,
            "reserved": ledger.reserved,
            "committed_upper": ledger.committed_upper,
            "anomalies": anomalies,
        },
        "safety_ceiling": last_ceiling,
        "invocations": [
            {
                "invocation_id": info.invocation_id,
                "started_at": info.started["at"],
                "safety_ceiling": info.started["safety_ceiling"],
                "ceiling_change": info.started["ceiling_change"],
                "ended_reason": info.ended["reason"] if info.ended else END_INTERRUPTED,
            }
            for info in view.invocations
        ],
        "uncommitted_tail_bytes": len(view.uncommitted_tail),
        "predictions_sha256": _sha_text(predictions_text),
        "requests_sha256": view.config["requests_sha256"],
    }


def export_view(view: RunView, services: Services) -> tuple[str, dict[str, Any]]:
    """Rebuild ``predictions.jsonl`` text and the run summary from the records. No timestamp, so a repeat is byte-identical."""
    assess_safety(view, services)
    run_state = view.run_state
    last_reason = view.last_end_reason
    predictions = [build_prediction(s, run_state=run_state, last_reason=last_reason) for s in view.ordered_states]
    _refuse(
        [p["question_id"] for p in predictions] == [r["question_id"] for r in view.requests],
        "predictions do not hold one record per runtime question in manifest order",
    )
    text = _jsonl(predictions)
    return text, summarise_run(view, text, services)


def write_export(view: RunView, services: Services) -> tuple[str, dict[str, Any]]:
    text, summary = export_view(view, services)
    atomic_write_text(view.run_dir / PREDICTIONS_NAME, text)
    atomic_write_text(view.run_dir / RUN_SUMMARY_NAME, _dumps(summary) + "\n")
    return text, summary


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def provider_descriptor(mode: str, config: ProviderConfig | None = None) -> dict[str, Any]:
    """Credential-free description of the adapter, part of the run identity.

    A live descriptor holds ``base_url``, the value the client's ``str(client.base_url)`` will have. The
    driver checks that after it builds the client.
    """
    if mode == MODE_FAKE:
        return {"adapter": "faar.answer_providers.FakeProvider"}
    descriptor: dict[str, Any] = {"adapter": "faar.answer_providers.OpenAIChatProvider"}
    if config is not None:
        descriptor["base_url"] = expected_base_url(config.endpoint)
    try:
        import importlib.metadata

        descriptor["openai_sdk"] = importlib.metadata.version("openai")
    except Exception:
        descriptor["openai_sdk"] = None
    return descriptor


def wire_provider(
    options: RunOptions, *, fake_script: Path | None, environ: Mapping[str, str]
) -> tuple[ProviderFactory, dict[str, Any]]:
    """Return a lazy provider factory and the adapter descriptor. No provider and no client is built here.

    In fake mode the script file is read and validated now, so a bad script refuses before anything is
    prepared, and its sha256 joins the run identity: a changed script is a different run.
    """
    config = _resolve_config(options)
    if options.mode == MODE_FAKE:
        _refuse(fake_script is not None, "--fake-script PATH is required with --mode fake")
        assert fake_script is not None
        steps, default, script_sha = load_fake_script(fake_script)
        descriptor = {**provider_descriptor(MODE_FAKE), "fake_script_sha256": script_sha}
        return (lambda context: build_fake_provider(steps, default, config, context)), descriptor
    return (lambda context: build_live_provider(config, environ)), provider_descriptor(MODE_LIVE, config)


def _ceiling_record(amount: float, prices: PriceTable) -> dict[str, Any]:
    return {"amount": amount, "currency": prices.currency, "simulated": prices.simulated}


def decide_ceiling(
    *, prior: Mapping[str, Any] | None, requested: float, raised: float | None, note: str | None, prices: PriceTable
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the effective ceiling record and the ``ceiling_change`` record for this invocation.

    The first invocation sets the ceiling. Later invocations must repeat the current ceiling in
    ``--safety-ceiling``. Raising it needs ``--raise-safety-ceiling`` and a note, and the request
    must name the current ceiling, so a stale command cannot raise a ceiling that moved. Lowering
    needs no authorisation and is recorded.
    """
    note = note.strip() if note else None
    if prior is None:
        _refuse(raised is None, "--raise-safety-ceiling needs an earlier invocation; this run has none. Use --safety-ceiling.")
        _refuse(note is None, "--authorization-note belongs with --raise-safety-ceiling")
        return _ceiling_record(requested, prices), None
    _refuse(prior["currency"] == prices.currency, f"the ceiling was set in {prior['currency']}, the price table uses {prices.currency}")
    current = float(prior["amount"])
    if raised is not None:
        _refuse(bool(note), "--raise-safety-ceiling needs --authorization-note TEXT saying who approved the higher ceiling and why")
        _refuse(
            requested == current,
            f"--safety-ceiling {requested} does not match the current ceiling {current}. "
            "Repeat the current ceiling next to --raise-safety-ceiling.",
        )
        _refuse(raised > current, f"--raise-safety-ceiling {raised} must be above the current ceiling {current}")
        return _ceiling_record(raised, prices), {"from": current, "to": raised, "authorization_note": note}
    _refuse(note is None, "--authorization-note belongs with --raise-safety-ceiling")
    if requested == current:
        return _ceiling_record(current, prices), None
    _refuse(
        requested < current,
        f"--safety-ceiling {requested} is above the current ceiling {current}. "
        "Use --raise-safety-ceiling with --authorization-note to raise it.",
    )
    return _ceiling_record(requested, prices), {"from": current, "to": requested, "authorization_note": None}


def _cleanup_fresh_directory(run_dir: Path, created: bool) -> None:
    """After a refusal on a directory this invocation created, remove the empty shell and its lock file."""
    if not created:
        return
    try:
        lock = run_dir / LOCK_NAME
        if lock.exists() and {p.name for p in run_dir.iterdir()} == {LOCK_NAME}:
            lock.unlink()
        run_dir.rmdir()
    except OSError:
        pass


def _run_message(summary: Mapping[str, Any], run_dir: Path) -> str:
    counts = summary["counts"]
    cost = summary["cost"]
    tag = " (simulated)" if cost["simulated"] else ""
    message = (
        f"run {summary['run_id']} in {run_dir}: state {summary['run_state']}; {counts['questions']} questions: "
        f"{counts['answered']} answered, {counts['no_evidence']} no_evidence, {counts['execution_failed']} execution_failed "
        f"({counts['unserved']} unserved, {counts['awaiting_reconciliation']} awaiting reconciliation); "
        f"cost measured {cost['measured']:.6f}, committed upper {cost['committed_upper']:.6f} {cost['currency']}{tag}."
    )
    if summary.get("safety_violations"):
        message += (
            f" Safety stop, no further request was sent: {describe_violations(summary['safety_violations'])}. "
            f"{SAFETY_NO_CLEAR_NOTE} {SAFETY_LIMIT_NOTE} {model_note(summary['safety_violations'])}".rstrip()
        )
    return message


def execute_run(
    options: RunOptions,
    *,
    services: Services | None = None,
    provider_factory: ProviderFactory | None = None,
    descriptor: Mapping[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    crash_hook: CrashHook | None = None,
    clock: Callable[[], float] = time.monotonic,
    environ: Mapping[str, str] | None = None,
) -> LiveResult:
    """Start or resume a run: prepare, check identity, dispatch pending requests, export.

    Raises :class:`RunnerRefusal` before any dispatch when the run must not start or continue.
    ``provider_factory`` receives a :class:`ProviderContext` and is called only when at least one request is pending, after every check passed.
    """
    environ = os.environ if environ is None else environ
    _refuse(options.mode in MODES, f"mode must be one of {MODES}, got {options.mode!r}")
    ceiling_requested = _validate_amount(options.safety_ceiling, "--safety-ceiling")
    raised = None if options.raise_safety_ceiling is None else _validate_amount(options.raise_safety_ceiling, "--raise-safety-ceiling")
    check_live_enablement(
        mode=options.mode,
        has_provider_config=options.config is not None,
        safety_ceiling=options.safety_ceiling,
        environ=environ,
    )
    project_root = options.project_root.resolve()
    run_dir = _absolute(options.run_dir)
    refuse_run_directory(run_dir, project_root, options.mode)
    run_id = validate_run_id(options.run_id if options.run_id is not None else run_dir.name)
    config = _resolve_config(options)
    invocation_id = uuid.uuid4().hex

    created = not run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(run_dir, invocation_id)
    lock.acquire()  # a refusal here leaves the directory as it is: another invocation owns it
    try:
        try:
            return _execute_locked(
                options=options,
                run_dir=run_dir,
                run_id=run_id,
                config=config,
                services=services,
                provider_factory=provider_factory,
                descriptor=descriptor or {"adapter": "unspecified"},
                sleep=sleep,
                crash_hook=crash_hook,
                clock=clock,
                invocation_id=invocation_id,
                ceiling_requested=ceiling_requested,
                raised=raised,
                project_root=project_root,
            )
        except RunnerRefusal:
            if created and not (run_dir / RUN_CONFIG_NAME).exists() and not (run_dir / REQUESTS_NAME).exists():
                lock.release()
                _cleanup_fresh_directory(run_dir, created)
            raise
    finally:
        lock.release()


def _resume_after_interrupt(view: RunView, log: EventLog, run_dir: Path, invocation_id: str) -> None:
    """Rebuild the driver's picture of the run from the file, then resolve every open attempt.

    A Ctrl-C can land after a line reached the file and before the driver noticed (in the sync, or between
    the last byte and the bookkeeping). The file is the truth: read it back, move any partial final line
    aside, continue the ``seq`` after the last committed line, and only then append ``outcome_unknown`` or
    ``response_saved`` for what is still open. ``view`` is updated in place because the log's callback holds it.
    """
    fresh = load_run(run_dir, verify=False)
    if fresh.uncommitted_tail:
        quarantine_uncommitted_tail(run_dir, fresh, invocation_id)
    view.events[:] = fresh.events
    view.invocations[:] = fresh.invocations
    view.states = fresh.states
    view.committed_bytes = fresh.committed_bytes
    view.uncommitted_tail = b""
    log.resume_at(len(fresh.events) + 1)
    recover_open_attempts(view, log, run_dir)


def _record_stopped_restart(run_dir: Path, view: RunView, services: Services, invocation_id: str) -> dict[str, Any]:
    """Make the durable records of a safety-stopped run complete, then return its summary. Sends nothing.

    A crash between a response file and its ``response_saved`` event leaves an open attempt. Only the file
    tells that the provider answered, so the append-only recovery runs here, under the lock the caller
    holds, exactly as ``reconcile`` runs it. Then ``predictions.jsonl`` and ``run_summary.json`` are written
    when they differ from what the records say, so the exports name the stop. A stopped run with nothing open
    and current exports is not touched.
    """
    if any(state.orphans for state in view.ordered_states):
        tail = quarantine_uncommitted_tail(run_dir, view, invocation_id)

        def track(event: dict[str, Any]) -> None:
            view.events.append(event)

        log = EventLog(run_dir / ATTEMPTS_NAME, invocation_id, len(view.events) + 1, track)
        try:
            recover_open_attempts(view, log, run_dir, tail=tail)
        finally:
            log.close()
        view = load_run(run_dir)
        assess_safety(view, services)
    text, summary = export_view(view, services)
    summary_text = _dumps(summary) + "\n"
    if _read_text_or_none(run_dir / PREDICTIONS_NAME) != text:
        atomic_write_text(run_dir / PREDICTIONS_NAME, text)
    if _read_text_or_none(run_dir / RUN_SUMMARY_NAME) != summary_text:
        atomic_write_text(run_dir / RUN_SUMMARY_NAME, summary_text)
    return summary


def _execute_locked(
    *,
    options: RunOptions,
    run_dir: Path,
    run_id: str,
    config: ProviderConfig,
    services: Services | None,
    provider_factory: ProviderFactory | None,
    descriptor: Mapping[str, Any],
    sleep: Callable[[float], None],
    crash_hook: CrashHook | None,
    clock: Callable[[], float],
    invocation_id: str,
    ceiling_requested: float,
    raised: float | None,
    project_root: Path,
) -> LiveResult:
    services = services or Services.default()
    directory = check_run_directory(run_dir)
    _refuse(
        not (run_dir / SCORES_NAME).exists() and not (run_dir / SCORE_SUMMARY_NAME).exists(),
        f"{run_dir} has been scored. Scoring is final for a run; start a new run directory to continue.",
    )
    retrieval_run = retrieve_runtime_questions(
        project_root=project_root,
        pilot_id=options.pilot_id,
        runtime_manifest_path=options.runtime_manifest_path,
        text_policy=DEFAULT_TEXT_POLICY,
    )
    script_sha = sha256_file(options.cli_script) if options.cli_script is not None else None
    identity = build_identity(
        mode=options.mode,
        retrieval_run=retrieval_run,
        config=config,
        services=services,
        provider_descriptor=descriptor,
        cli_script_sha256=script_sha,
    )
    identity_hash = identity_sha256(identity)
    records = prepare_requests(retrieval_run, config=config, services=services, identity_hash=identity_hash)
    requests_text = _jsonl(strip_evidence_text(records))

    if directory == "run":
        view = load_run(run_dir)
        stored = view.config
        if stored.get("identity_sha256") != identity_hash:
            changed = _diff_paths(stored.get("identity"), identity)
            raise RunnerRefusal(
                f"{run_dir} was started with a different identity. Changed: {', '.join(changed[:12]) or 'unknown'}. "
                f"Nothing was dispatched. {EXPECTED_RESUME_NOTE}"
            )
        _refuse(stored.get("run_id") == run_id, f"{run_dir} holds run_id {stored.get('run_id')!r}, this invocation names {run_id!r}")
        _refuse(
            view.requests_bytes.decode("utf-8") == requests_text,
            f"{run_dir} holds prepared requests that regenerate differently (prompt or evidence changed). "
            f"Nothing was dispatched. {EXPECTED_RESUME_NOTE}",
        )
        services.ledger_from_events(view.events, config.prices)  # a corrupt ledger refuses here, before any dispatch
        # A stored safety violation ends the run before any provider exists and before the ceiling is decided,
        # so a raised ceiling is neither recorded nor able to lift the stop. The only writes are the append-only
        # recovery of a response file whose event was lost, and the exports rebuilt from the records.
        violations = assess_safety(view, services)
        if violations:
            stopped_summary = _record_stopped_restart(run_dir, view, services, invocation_id)
            return LiveResult(EXIT_SAFETY_STOPPED, safety_stop_message(run_dir, violations), stopped_summary)
        prior = view.effective_ceiling
        stored_requests: Sequence[Mapping[str, Any]] = view.requests
        pending_count = len(view.by_status(REQUEST_PENDING))
        prior_attempts = {rid: len(state.attempts) for rid, state in view.states.items()}
    else:
        prior = None
        stored_requests = json.loads("[" + ",".join(requests_text.splitlines()) + "]")
        pending_count = sum(1 for r in stored_requests if r["action"] == ACTION_SEND)
        prior_attempts = {}

    effective, change = decide_ceiling(
        prior=prior,
        requested=ceiling_requested,
        raised=raised,
        note=options.authorization_note,
        prices=config.prices,
    )

    # Every check that can refuse has run except those inside the provider factory, which builds nothing
    # until now. In live mode the factory reads the credential here, and only here.
    provider = None
    if pending_count:
        _refuse(provider_factory is not None, "no provider is available and requests are pending")
        assert provider_factory is not None
        provider = provider_factory(ProviderContext(stored_requests, prior_attempts))

    if directory != "run":
        _initialise(
            run_dir,
            options=options,
            run_id=run_id,
            identity=identity,
            identity_hash=identity_hash,
            requests_text=requests_text,
            retrieval_run=retrieval_run,
            config=config,
        )
        view = load_run(run_dir)

    # From here on, events are appended.
    tail = quarantine_uncommitted_tail(run_dir, view, invocation_id)

    invocations: list[InvocationInfo] = view.invocations

    def track(event: dict[str, Any]) -> None:
        view.events.append(event)
        if event["event"] == EVENT_INVOCATION_STARTED:
            invocations.append(InvocationInfo(event["invocation_id"], event))
        elif event["event"] == EVENT_INVOCATION_ENDED:
            invocations[-1].ended = event

    log = EventLog(run_dir / ATTEMPTS_NAME, invocation_id, len(view.events) + 1, track)
    try:
        log.append(
            EVENT_INVOCATION_STARTED,
            safety_ceiling=effective,
            ceiling_change=change,
            mode=options.mode,
            identity_sha256=identity_hash,
            recovered_uncommitted_tail=tail,
        )
        recover_open_attempts(view, log, run_dir)
        driver = _Driver(
            view=view,
            log=log,
            services=services,
            config=config,
            provider=provider,
            ceiling=effective["amount"],
            sleep=sleep,
            crash_hook=crash_hook,
            clock=clock,
        )
        try:
            reason = driver.run_all()
        except KeyboardInterrupt:
            _resume_after_interrupt(view, log, run_dir, invocation_id)
            reason = END_INTERRUPTED
        ended_extra = {"safety_violations": driver.violations} if reason == END_SAFETY_STOP else {}
        log.append(
            EVENT_INVOCATION_ENDED, reason=reason, ledger=driver.ledger_totals(), circuit_breaker=driver.breaker_record(), **ended_extra
        )
    finally:
        log.close()
    if crash_hook is not None:
        crash_hook(CRASH_BEFORE_EXPORT, {})
    final = load_run(run_dir)
    _, summary = write_export(final, services)
    return LiveResult(summary["exit_code"], _run_message(summary, run_dir), summary)


def read_lock_holder(run_dir: Path) -> str | None:
    """Who wrote ``run.lock`` last, read without creating or locking it. ``None`` when there is no file."""
    try:
        fd = os.open(run_dir / LOCK_NAME, os.O_RDONLY)
    except OSError:
        return None
    try:
        return _read_lock_holder(fd)
    finally:
        os.close(fd)


def run_status(run_dir: Path, services: Services | None = None) -> dict[str, Any]:
    """Describe a run from its files. Changes nothing and takes no lock.

    While a live invocation holds the lock, an attempt that has ``dispatch_started`` and no result is
    ``in_flight``: its own process is still waiting for the answer. It is not unknown, and the run is
    ``active``. Without a lock holder the same attempt is unknown, because its process is gone.
    """
    services = services or Services.default()
    run_dir = _absolute(run_dir)
    view = fold_recoverable_responses(load_run(run_dir))
    _, summary = export_view(view, services)
    lock_held = lock_is_held(run_dir)
    active = lock_held is True
    in_flight = [attempt.attempt_id for state in view.ordered_states if active and state.unresolved_unknown is None for attempt in state.orphans]
    in_flight_questions = {s.question_id for s in view.ordered_states if active and s.unresolved_unknown is None and s.orphans}
    by_status = {status: len(view.by_status(status)) for status in (REQUEST_SKIP, REQUEST_ANSWERED, REQUEST_FAILED, REQUEST_UNKNOWN, REQUEST_PENDING)}
    unknown = [
        (s.unresolved_unknown or s.orphans[0]).attempt_id for s in view.by_status(REQUEST_UNKNOWN) if s.question_id not in in_flight_questions
    ]
    by_status[REQUEST_UNKNOWN] -= len(in_flight_questions)
    by_status["in_flight"] = len(in_flight_questions)
    counts = dict(summary["counts"])
    counts["awaiting_reconciliation"] -= len(in_flight_questions)
    counts["in_flight"] = len(in_flight_questions)
    actions: list[str] = []
    if active:
        actions.append("wait for the running invocation to finish, then check the status again")
    if unknown:
        actions.append("reconcile each attempt in awaiting_reconciliation after checking the provider's records")
    safety = summary.get("safety_violations", [])
    if safety and not active:
        actions.append(
            f"safety stop: {describe_violations(safety)}. {SAFETY_NO_CLEAR_NOTE} Keep this directory as the record of what "
            f"happened. {SAFETY_LIMIT_NOTE} {model_note(safety)}".rstrip()
        )
    if view.last_end_reason == END_CIRCUIT and not active and not safety:
        actions.append(
            f"the circuit breaker stopped the last invocation after {CIRCUIT_BREAKER_THRESHOLD} consecutive failing attempts: "
            "find the shared cause in attempts.jsonl before running again"
        )
    if by_status[REQUEST_PENDING] and not active and not safety:
        actions.append("run again to serve the pending questions")
    if by_status[REQUEST_FAILED] and not active and not safety and not is_scored(run_dir):
        actions.append("after fixing the cause, reopen an execution_failed question to give it further attempts")
    return {
        "run_id": view.config["run_id"],
        "mode": view.config["mode"],
        "kind": view.config["kind"],
        "run_state": "active" if active else summary["run_state"],
        "exit_code": None if active else summary["exit_code"],
        "requests": by_status,
        "awaiting_reconciliation": unknown,
        "in_flight": in_flight,
        "counts": counts,
        "cost": summary["cost"],
        "safety_ceiling": summary["safety_ceiling"],
        "safety_violations": safety,
        "attempts": summary["attempts"],
        "last_invocation_ended": view.last_end_reason,
        "uncommitted_tail_bytes": len(view.uncommitted_tail),
        "lock_held": lock_held,
        "lock_holder": read_lock_holder(run_dir) if active else None,
        "scored": is_scored(run_dir),
        "predictions_current": _file_matches(run_dir / PREDICTIONS_NAME, summary["predictions_sha256"]),
        "next": actions,
    }


def _file_matches(path: Path, digest: str) -> bool:
    try:
        return sha256_file(path) == digest
    except OSError:
        return False


def format_status(status: Mapping[str, Any]) -> str:
    lines = [
        f"run {status['run_id']} ({status['mode']}, {status['kind']}): {status['run_state']}",
        "requests: " + ", ".join(f"{k} {v}" for k, v in status["requests"].items()),
        f"attempts: {status['attempts']['total']} (saved {status['attempts']['saved']}, failed {status['attempts']['failed']}, "
        f"unknown {status['attempts']['unknown']}, open {status['attempts']['open']})",
        "cost: measured {measured:.6f}, reserved {reserved:.6f}, committed upper {committed_upper:.6f} {currency}{sim}".format(
            **status["cost"], sim=" (simulated)" if status["cost"]["simulated"] else ""
        ),
        f"safety ceiling: {status['safety_ceiling']['amount'] if status['safety_ceiling'] else None}",
        f"last invocation ended: {status['last_invocation_ended']}",
        f"lock held by a running invocation: {status['lock_held']}",
        f"scored: {'yes, the run is final' if status['scored'] else 'no'}",
        f"predictions.jsonl matches the records: {status['predictions_current']}",
    ]
    if status["uncommitted_tail_bytes"]:
        lines.append(f"attempts.jsonl ends with {status['uncommitted_tail_bytes']} uncommitted bytes; the next run moves them aside")
    if status["run_state"] == "active":
        lines.insert(
            1,
            f"the run is active: an invocation holds the lock ({status['lock_holder']}). Counts below are a snapshot of its files.",
        )
    if status["safety_violations"]:
        lines.insert(
            1,
            f"SAFETY STOP: {len(status['safety_violations'])} violation(s). No further request will be sent for this run. "
            f"{SAFETY_LIMIT_NOTE} {model_note(status['safety_violations'])}".rstrip(),
        )
    for attempt_id in status["in_flight"]:
        lines.append(f"in flight: {attempt_id} (dispatched, no result yet; its process is still running)")
    for attempt_id in status["awaiting_reconciliation"]:
        lines.append(f"awaiting reconciliation: {attempt_id}")
    for step in status["next"]:
        lines.append(f"next: {step}")
    return "\n".join(lines)


def export_run(run_dir: Path, services: Services | None = None) -> LiveResult:
    """Rebuild ``predictions.jsonl`` and ``run_summary.json`` from the records. Safe to repeat.

    After ``score`` the exports are final. A repeat that would write byte-identical files succeeds and
    writes nothing. Any other export of a scored run is refused.
    """
    services = services or Services.default()
    run_dir = require_initialised_run(run_dir)
    lock = RunLock(run_dir, uuid.uuid4().hex)
    lock.acquire()
    try:
        view = fold_recoverable_responses(load_run(run_dir))
        if is_scored(run_dir):
            text, summary = export_view(view, services)
            summary_text = _dumps(summary) + "\n"
            same = (
                _read_text_or_none(run_dir / PREDICTIONS_NAME) == text
                and _read_text_or_none(run_dir / RUN_SUMMARY_NAME) == summary_text
            )
            refuse_if_scored_unless(same, run_dir, "an export that would change the exports")
            return LiveResult(summary["exit_code"], "exports unchanged (the run is scored). " + _run_message(summary, run_dir), summary)
        _, summary = write_export(view, services)
    finally:
        lock.release()
    return LiveResult(summary["exit_code"], "exported. " + _run_message(summary, run_dir), summary)


def refuse_if_scored_unless(condition: bool, run_dir: Path, what: str) -> None:
    if not condition:
        refuse_if_scored(run_dir, what)


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def reconcile_attempt(
    *,
    run_dir: Path,
    attempt_id: str,
    resolution: str,
    note: str,
    services: Services | None = None,
    crash_hook: CrashHook | None = None,
) -> LiveResult:
    """Record how an unknown attempt was resolved. It never contacts the provider.

    Look at the provider's own records first (dashboard, usage export) to see whether the
    request was processed. ``allow_new_attempt`` puts the question back in line for a new
    attempt, and the unknown attempt's cost stays counted at its upper bound.
    ``mark_failed`` ends the question as ``execution_failed``.
    """
    services = services or Services.default()
    run_dir = _absolute(run_dir)
    _refuse(resolution in RESOLUTIONS, f"resolution must be one of {RESOLUTIONS}")
    _refuse(bool(note and note.strip()), "--note TEXT is required: say what you checked and what you found")
    run_dir = require_initialised_run(run_dir)
    invocation_id = uuid.uuid4().hex
    lock = RunLock(run_dir, invocation_id)
    lock.acquire()
    try:
        refuse_if_scored(run_dir, "reconcile")
        view = load_run(run_dir)
        tail = quarantine_uncommitted_tail(run_dir, view, invocation_id)
        target = next((s for s in view.ordered_states if any(a.attempt_id == attempt_id for a in s.attempts)), None)
        _refuse(target is not None, f"no attempt {attempt_id!r} in {run_dir}")
        assert target is not None

        def track(event: dict[str, Any]) -> None:
            view.events.append(event)

        log = EventLog(run_dir / ATTEMPTS_NAME, invocation_id, len(view.events) + 1, track)
        try:
            recover_open_attempts(view, log, run_dir)
            attempt = next(a for a in target.attempts if a.attempt_id == attempt_id)
            _refuse(
                attempt.outcome == "unknown" and attempt.reconciled is None,
                f"attempt {attempt_id} is not awaiting reconciliation (state: {attempt.outcome or 'open'}"
                f"{', already reconciled' if attempt.reconciled else ''})",
            )
            if resolution == RESOLUTION_ALLOW:
                limit = view.config["identity"]["retry_policy"].get("max_attempts")
                _refuse(
                    limit is None or target.attempts_in_window < limit,
                    f"question {target.question_id} already has {target.attempts_in_window} attempts since its last reopen, the retry policy's limit. "
                    f"Use {RESOLUTION_FAIL}.",
                )
            if crash_hook is not None:
                crash_hook("before_reconciled", {"attempt_id": attempt_id})
            log.append(
                EVENT_RECONCILED,
                question_id=target.question_id,
                request_id=target.request_id,
                attempt=attempt.attempt,
                attempt_id=attempt_id,
                resolution=resolution,
                note=note.strip(),
                recovered_uncommitted_tail=tail,
            )
        finally:
            log.close()
        final = load_run(run_dir)
        _, summary = write_export(final, services)
    finally:
        lock.release()
    return LiveResult(
        EXIT_OK,
        f"attempt {attempt_id} reconciled as {resolution}. " + _run_message(summary, run_dir),
        summary,
    )


def reopen_question(
    *,
    run_dir: Path,
    question_id: str,
    note: str,
    services: Services | None = None,
    crash_hook: CrashHook | None = None,
) -> LiveResult:
    """Put an ``execution_failed`` question back in line for up to ``max_attempts`` further attempts.

    Use it after fixing the cause of the failures (a network fault, a config error, a provider outage).
    It appends ``question_reopened``, changes no earlier event, and never contacts the provider. The
    earlier attempts and their costs stay in the records and in the ledger. The attempt limit counts from
    the reopen event. A person decides this: the driver never reopens a question by itself.

    Refused for a question that is answered, still pending (an unserved or budget-limited question simply
    resumes with ``run``), awaiting reconciliation (use ``reconcile``) or never sent. Also refused without a
    note, while another invocation holds the lock, and once the run is scored.
    """
    services = services or Services.default()
    _refuse(bool(note and note.strip()), "--note TEXT is required: say what you fixed and why the question may be tried again")
    run_dir = require_initialised_run(run_dir)
    invocation_id = uuid.uuid4().hex
    lock = RunLock(run_dir, invocation_id)
    lock.acquire()
    try:
        refuse_if_scored(run_dir, "reopen")
        view = load_run(run_dir)
        target = next((s for s in view.ordered_states if s.question_id == question_id), None)
        _refuse(target is not None, f"no question {question_id!r} in {run_dir}")
        assert target is not None
        tail = quarantine_uncommitted_tail(run_dir, view, invocation_id)

        def track(event: dict[str, Any]) -> None:
            view.events.append(event)

        log = EventLog(run_dir / ATTEMPTS_NAME, invocation_id, len(view.events) + 1, track)
        try:
            recover_open_attempts(view, log, run_dir)
            status = target.status
            _refuse(
                status == REQUEST_FAILED,
                _reopen_refusal(target, status),
            )
            last = target.window_attempts[-1] if target.window_attempts else None
            if crash_hook is not None:
                crash_hook("before_reopened", {"question_id": question_id})
            log.append(
                EVENT_QUESTION_REOPENED,
                question_id=target.question_id,
                request_id=target.request_id,
                attempts_before=len(target.attempts),
                note=note.strip(),
                previous_failure=None if last is None or last.resolution_event is None else last.resolution_event.get("kind"),
                recovered_uncommitted_tail=tail,
            )
        finally:
            log.close()
        final = load_run(run_dir)
        _, summary = write_export(final, services)
    finally:
        lock.release()
    return LiveResult(
        EXIT_OK,
        f"question {question_id} reopened: up to {services.retry_parameters().get('max_attempts')} further attempts on the next run. "
        + _run_message(summary, run_dir),
        summary,
    )


def _reopen_refusal(state: RequestState, status: str) -> str:
    who = f"question {state.question_id}"
    if status == REQUEST_ANSWERED:
        return f"{who} is answered. A saved response is never reopened."
    if status == REQUEST_UNKNOWN:
        return f"{who} has an attempt with an unknown outcome. Use reconcile, after checking the provider's records."
    if status == REQUEST_PENDING:
        return f"{who} is pending, not execution_failed. Run again to serve it (an unserved or budget-limited question resumes by itself)."
    return f"{who} was never sent ({state.request['skip_reason']}). Its request cannot change in this run; a changed request needs a new run."


def score_run_live(
    *,
    project_root: Path,
    run_dir: Path,
    evaluation_manifest_path: Path | None = None,
    services: Services | None = None,
) -> LiveResult:
    """Export, then join predictions to the evaluation manifest with ``faar.ohr_scoring.score_predictions``.

    This is the only command that opens the evaluation manifest. It scores only a run whose state is
    ``complete``. Scoring is final: existing score files are never overwritten (identical bytes report
    "already scored", different bytes are refused), and a scored run's exports are never rewritten.
    """
    import importlib
    import unicodedata

    services = services or Services.default()
    project_root = project_root.resolve()
    run_dir = require_initialised_run(run_dir)
    lock = RunLock(run_dir, uuid.uuid4().hex)
    lock.acquire()
    try:
        view = load_run(run_dir)
        scored = is_scored(run_dir)
        assess_safety(view, services)
        _refuse(
            scored or view.run_state != RUN_SAFETY_STOPPED,
            f"score needs a complete run, and this run is {RUN_SAFETY_STOPPED}: "
            f"{describe_violations(view.safety_violations or [])}. {SAFETY_NO_CLEAR_NOTE} Nothing was written.",
        )
        if not scored and view.run_state != RUN_COMPLETE:
            folded = fold_recoverable_responses(view)
            assess_safety(folded, services)
            _refuse(
                folded.run_state != RUN_COMPLETE,
                f"score needs a complete run, and this run is {view.run_state} only because a response file was written "
                "but its event was not (the process stopped in between). Run the same `run` command once to record "
                "the recovered response; it sends nothing. Nothing was written.",
            )
        _refuse(
            scored or view.run_state == RUN_COMPLETE,
            f"score needs a complete run, and this run is {view.run_state}. Serve the pending questions, "
            "reconcile the unknown attempts or reopen the failed questions first. Nothing was written.",
        )
        if scored:
            # Scores are final: rebuild in memory and compare, never rewrite the exports.
            predictions_text, run_summary = export_view(view, services)
            _refuse(
                _read_text_or_none(run_dir / PREDICTIONS_NAME) == predictions_text
                and _read_text_or_none(run_dir / RUN_SUMMARY_NAME) == _dumps(run_summary) + "\n",
                f"{run_dir} has been scored and its exports no longer match the records. Nothing was written.",
            )
        else:
            predictions_text, run_summary = write_export(view, services)
        pilot_id = view.config["pilot_id"]
        path = evaluation_manifest_path or project_root / "results" / "pilots" / pilot_id / "evaluation_manifest.json"
        try:
            evaluation_bytes = path.read_bytes()
            evaluation = json.loads(evaluation_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunnerRefusal(f"cannot read the evaluation manifest {path}: {exc}") from exc
        _refuse(isinstance(evaluation, dict) and isinstance(evaluation.get("questions"), dict), f"{path} has no questions object")
        _refuse(evaluation.get("pilot_id") == pilot_id, f"{path} is for pilot {evaluation.get('pilot_id')!r}, the run is for {pilot_id!r}")
        try:
            scoring = importlib.import_module("faar.ohr_scoring")
        except ImportError as exc:
            raise RunnerRefusal(f"faar.ohr_scoring is not available in this checkout: {exc}") from exc
        predictions = [json.loads(line) for line in predictions_text.splitlines()]
        try:
            result = scoring.score_predictions(predictions, evaluation["questions"])
        except ValueError as exc:
            raise RunnerRefusal(f"scoring refused the join: {exc}") from exc
        scores_text = _jsonl(result["rows"])
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": view.config["run_id"],
            "pilot_id": pilot_id,
            "mode": view.config["mode"],
            "kind": view.config["kind"],
            "label": view.config["label"],
            "run_state": run_summary["run_state"],
            "valid_baseline": run_summary["valid_baseline"],
            "valid_baseline_blockers": run_summary["valid_baseline_blockers"],
            "valid_baseline_means": run_summary["valid_baseline_means"],
            "scorer": scoring.scorer_identity(),
            "scorer_environment": {**scoring.scorer_dependencies(), "unicodedata": unicodedata.unidata_version},
            "evaluation_manifest": {"path": _display_path(path, project_root), "sha256": sha256_bytes(evaluation_bytes)},
            "identity_sha256": view.config["identity_sha256"],
            "predictions_sha256": run_summary["predictions_sha256"],
            "counts": result["counts"],
            "aggregates": result["aggregates"],
            "scores_sha256": _sha_text(scores_text),
        }
        summary_text = _dumps(summary) + "\n"
        failed = result["counts"].get("execution_failed", 0)
        aggregate = result["aggregates"]["all_questions"]
        line = (
            f"{result['counts']['questions']} questions, execution_failed {failed}, run state {run_summary['run_state']}; "
            f"all-question EM {aggregate['em']}, F1 {aggregate['f1']}"
        )
        exit_code = EXIT_OK if failed == 0 else EXIT_EXECUTION_FAILED
        scores_path, summary_path = run_dir / SCORES_NAME, run_dir / SCORE_SUMMARY_NAME
        if scores_path.exists() or summary_path.exists():
            same = (
                scores_path.exists()
                and summary_path.exists()
                and scores_path.read_text(encoding="utf-8") == scores_text
                and summary_path.read_text(encoding="utf-8") == summary_text
            )
            _refuse(same, f"{run_dir} already holds scores that differ from a fresh scoring. Nothing was written.")
            return LiveResult(exit_code, f"already scored, verified identical: {line}. Nothing rewritten.", summary)
        atomic_write_text(scores_path, scores_text)
        atomic_write_text(summary_path, summary_text)
    finally:
        lock.release()
    return LiveResult(exit_code, f"scores written to {run_dir}: {line}.", summary)


def _estimate_total(records: Sequence[Mapping[str, Any]]) -> int | None:
    total, seen = 0, False
    for record in records:
        estimate = record.get("token_estimate")
        if isinstance(estimate, Mapping):
            for key in ("estimate", "tokens"):
                value = estimate.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    total += value
                    seen = True
                    break
    return total if seen else None


def dry_run(
    *,
    project_root: Path,
    out_dir: Path,
    pilot_id: str = "ohr_dev_v1",
    runtime_manifest_path: Path | None = None,
    config: ProviderConfig | None = None,
    config_source: str | None = None,
    services: Services | None = None,
) -> LiveResult:
    """Prepare every request, write the preview and a small summary. No provider, no key, no network.

    Without ``config`` the provisional values of the study brief (option A, unapproved) fix the
    output limit and prices, and the summary says so.
    """
    project_root = project_root.resolve()
    out_dir = _absolute(out_dir)
    refuse_output_location(out_dir, project_root)
    services = services or Services.default()
    provisional = config is None
    config = config or PROVISIONAL_DRY_RUN_CONFIG
    retrieval_run = retrieve_runtime_questions(
        project_root=project_root,
        pilot_id=pilot_id,
        runtime_manifest_path=runtime_manifest_path,
        text_policy=DEFAULT_TEXT_POLICY,
    )
    identity = build_identity(
        mode="dry-run",
        retrieval_run=retrieval_run,
        config=config,
        services=services,
        provider_descriptor={"adapter": "none (dry run)"},
        cli_script_sha256=None,
    )
    identity_hash = identity_sha256(identity)
    records = prepare_requests(retrieval_run, config=config, services=services, identity_hash=identity_hash)
    prepared_text = _jsonl(records)
    preview = services.render_preview(records, title=f"Prompt preview for pilot {retrieval_run.pilot_id} (dry run)")
    sends = [r for r in records if r["action"] == ACTION_SEND]
    skip_reasons: dict[str, int] = {}
    for record in records:
        if record["action"] == ACTION_SKIP:
            skip_reasons[record["skip_reason"]] = skip_reasons.get(record["skip_reason"], 0) + 1
    bounds = [r["input_token_upper_bound"] for r in sends]
    costs = [r["cost_upper_bound"] for r in sends]
    max_attempts = services.retry_parameters().get("max_attempts")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dry_run",
        "provider_calls": 0,
        "pilot_id": retrieval_run.pilot_id,
        "runtime_manifest_sha256": retrieval_run.runtime_manifest_sha256,
        "template": {"template_id": services.template_id, "template_sha256": services.template_sha256},
        "provider_config": {
            "source": config_source or ("built-in provisional values from study brief 15.5 and 15.7 (not approved)" if provisional else "provider config file"),
            **config.identity_block(),
            "prices": config.prices.as_dict(),
        },
        "dry_run_identity_sha256": identity_hash,
        "counts": {
            "questions": len(records),
            "send": len(sends),
            "skip": len(records) - len(sends),
            "skip_reasons": dict(sorted(skip_reasons.items())),
        },
        "input_token_upper_bound": {"total": sum(bounds), "max": max(bounds, default=0), "limit": config.max_input_tokens},
        "cost_upper_bound": {
            "currency": config.prices.currency,
            "simulated": config.prices.simulated,
            "per_attempt_total": sum(costs),
            "per_attempt_max": max(costs, default=0.0),
            "worst_case_all_attempts": sum(costs) * max_attempts if isinstance(max_attempts, int) else None,
            "max_attempts": max_attempts,
        },
        "heuristic_token_estimate_total": _estimate_total(records),
        "hashes": {
            "prepared_requests_sha256": _sha_text(prepared_text),
            "prompt_preview_sha256": _sha_text(preview),
        },
    }
    summary_text = _dumps(summary) + "\n"
    outputs = {PREPARED_NAME: prepared_text, PREVIEW_NAME: preview, DRY_RUN_SUMMARY_NAME: summary_text}
    existing = [name for name in outputs if (out_dir / name).exists()]
    if existing:
        same = all((out_dir / name).read_text(encoding="utf-8") == outputs[name] for name in outputs if (out_dir / name).exists())
        _refuse(
            same and len(existing) == len(outputs),
            f"{out_dir} already holds {existing} that differ from this dry run. Nothing was overwritten; use a new directory.",
        )
        return LiveResult(EXIT_OK, f"dry run verified identical to {out_dir}. Nothing rewritten.", summary)
    for name, text in outputs.items():
        atomic_write_text(out_dir / name, text)
    counts = summary["counts"]
    return LiveResult(
        EXIT_OK,
        f"dry run written to {out_dir}: {counts['questions']} questions, {counts['send']} to send, {counts['skip']} skipped "
        f"{counts['skip_reasons']}; per-attempt cost upper bound {summary['cost_upper_bound']['per_attempt_total']:.6f} "
        f"{config.prices.currency}. No provider was called.",
        summary,
    )


def iter_committed_events(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the committed events of a run in order, for tests and ad-hoc inspection."""
    yield from read_event_file(_absolute(run_dir) / ATTEMPTS_NAME).events

