"""Answer-model providers for the live answer path: the interface, a fake and an OpenAI adapter.

The run driver talks to an `AnswerProvider`. This module offers two:

* `FakeProvider` replays a scripted sequence of outcomes. It reads no clock, uses no
  randomness and opens no connection, so a fake run is exactly repeatable. It is an
  engineering check tool. Its identity says `simulated: true`.
* `OpenAIChatProvider` maps a `ProviderRequest` onto `client.chat.completions.create`
  and maps the reply, or the SDK exception, back to the shared types in
  `faar.live_contract`. The caller builds and passes the client. This module never
  reads a credential, never builds a client and never sends anything itself.

Importing this module imports no provider SDK and no HTTP library. `openai` and
`httpx` load inside `classify_openai_error` and the adapter constructor.

Key for scripted steps
----------------------
`FakeProvider` keys its script by `ProviderRequest.request_id`, the value the driver
already has. `script_by_question` builds that mapping from prepared-request records
(`question_id` and `request_id`) and a mapping keyed by `question_id`. Each key holds
a sequence of `FakeStep`. Attempt N of a request consumes step N. When the sequence
is used up, or the key is absent, the `default` step applies.

Failure list
------------
Written before the code. Each item has a test in `tests/test_answer_providers.py`.

Fake provider
  F1. Two providers with the same script give identical results (no clock, no randomness).
  F2. The script advances per request, not globally: one request's retries do not shift another's.
  F3. A used-up or absent script falls back to `default`, and that fallback is deterministic.
  F4. Every send is recorded before it can fail, including the send that simulates a crash.
  F5. `crash_after_send` raises a `BaseException`, so an `except Exception` in the driver cannot swallow it.
  F6. Each step kind returns the fields the driver and parser branch on: text, finish_reason,
      refusal, usage (None when missing, never a silent zero), outcome and retryable flags.
  F7. `returned_model` can differ from the requested model, and both stay visible.
  F8. Fake usage never exceeds `max_output_tokens` and is derived from message bytes only.
  F9. An unknown step kind, or a key in `script_by_question` with no matching request, fails loudly.
  F10. `identity()` carries `simulated: true`, `model_calls: false`, and changes when the script changes.

OpenAI adapter
  O1. Success maps text, finish_reason, refusal, returned model, response id and usage; raw keeps
      `system_fingerprint` and the whole payload.
  O2. Absent `usage`, absent `prompt_tokens_details` and absent `completion_tokens_details` give None,
      not zero.
  O3. `finish_reason: length` and `message.refusal` reach the response unchanged.
  O4. A 200 reply that is not a chat completion is not silently treated as an answer. An HTML
      body, a JSON object without a non-empty `choices` list, a choice that is not an object and
      a choice without a `message` object raise `ProviderError(kind="malformed_response",
      outcome="unknown", retryable=False)` with the payload in `raw`. A message whose `content`
      is null stays a response (a refusal string is the usual reason), and the reply parser
      classifies it.
  O5. A status counts as `rejected` only when a source shows the provider did not process the request.
      A retry recommendation does not show that. OpenAI's error-codes page, `Retry-After` and the
      SDK's default retry set (408, 409, 429, 5xx, and `x-should-retry: true`) say when a retry is
      worth trying. They do not say the first request was not processed or billed. Every other
      retry candidate is `unknown` and not retryable, so the run driver never resends it and the
      question waits for `reconcile`. The rows and their sources are in the comment above
      `_classify_status`. Summary: 408 and 503 with code `server_is_overloaded` are `rejected` and
      retryable. 429 (rate limit), 409, 500, 502, 503 with any other body, other 5xx and the
      gateway timeouts 504, 522 and 524 are `unknown`. A quota or billing 429 is `rejected` and
      stops the run. An unknown status outcome keeps status, ids, headers and a bounded body in
      `ProviderError.raw`.
  O6. 400 is `rejected` and not retryable. 401 and 403 are kind `auth`. 404 model-not-found is
      kind `unknown_model`.
  O7. A connect failure is `not_sent` and retryable. A read timeout, a dropped connection and any
      error we cannot place are `unknown` and not retryable.
  O8. The SDK's own retries never run: a client with `max_retries != 0` is refused, because a hidden
      retry would break attempt accounting.
  O8b. Redirects are never followed. A client whose httpx client follows redirects, or whose setting cannot
      be read, is refused at construction (the SDK's default client follows them). A 3xx reply that
      reaches the adapter is `unknown` and not retryable, kind `redirect`, with the `location` header
      (clipped) and the request id in `raw`. The request reached a server, and nothing shows it was not
      processed, so a resend or a follow could bill twice and could send the prompt to another host.
  O9. The request carries the model, messages, `max_tokens` and per-request timeout, and nothing
      else unless the caller configured it. `params` and `request.params` may hold only the keys
      in `faar.live_contract.ALLOWED_OPENAI_PARAMS`. Any other key (`service_tier`, `tools`,
      `modalities`, `reasoning_effort`, a key the adapter sets itself) is refused before dispatch,
      because an unknown parameter can change billing outside the cost bound.
  O10. A test cannot reach the network: the mock handler must see the request, and the conftest
       blocks non-loopback sockets.
  O11. Importing the module builds no client and imports neither `openai` nor `httpx`.
  O12. `identity()` never touches the API key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Protocol, runtime_checkable

from faar.live_contract import (
    ALLOWED_OPENAI_PARAMS,
    OUTCOME_NOT_SENT,
    OUTCOME_REJECTED,
    OUTCOME_UNKNOWN,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)

# Kept equal to `faar.answer_prompt.ABSTENTION_TOKEN`. The fake must not import the prompt module.
ABSTENTION_TEXT = "NO_ANSWER"

# ProviderError.kind values. The retry policy branches on these.
# stop_run kinds: auth, unknown_model, quota. An unknown outcome goes to reconcile. Other errors follow retryable.
KIND_RATE_LIMIT = "rate_limit"
KIND_SERVER_ERROR = "server_error"  # HTTP 5xx except 504, 522 and 524
KIND_GATEWAY_TIMEOUT = "gateway_timeout"  # HTTP 504, 522, 524
# Gateway statuses that can follow a request the provider accepted: 504 (gateway timeout) and the
# Cloudflare origin timeouts 522 and 524.
_GATEWAY_TIMEOUT_STATUSES = (504, 522, 524)
# HTTP 408 (`rejected`, retryable) and HTTP 409 (`unknown`, not retryable). The name is older than the 409
# decision and is kept: tests/test_retry_policy.py and the fake's 408 step use the string "transient_status".
KIND_TRANSIENT_STATUS = "transient_status"
KIND_REDIRECT = "redirect"  # HTTP 3xx: the request reached a server, which sent us elsewhere
KIND_QUOTA = "quota"
KIND_BAD_REQUEST = "bad_request"
KIND_CLIENT_ERROR = "client_error"  # other 4xx
KIND_AUTH = "auth"
KIND_UNKNOWN_MODEL = "unknown_model"
KIND_NOT_FOUND = "not_found"  # 404 that is not a missing model
KIND_CONNECT = "connect"
KIND_CONNECT_TIMEOUT = "connect_timeout"
KIND_POOL_TIMEOUT = "pool_timeout"
KIND_CLIENT_CONFIG = "client_config"
KIND_TIMEOUT = "timeout"
KIND_CONNECTION_LOST = "connection_lost"
KIND_MALFORMED_RESPONSE = "malformed_response"
KIND_UNEXPECTED = "unexpected"
KIND_AMBIGUOUS = "ambiguous"

STOP_RUN_KINDS = (KIND_AUTH, KIND_UNKNOWN_MODEL, KIND_QUOTA)


class SimulatedCrash(BaseException):
    """Raised by a fake `crash_after_send` step to model the process dying after dispatch.

    It derives from `BaseException` so that `except Exception` blocks cannot absorb it.
    """


@runtime_checkable
class AnswerProvider(Protocol):
    def identity(self) -> dict[str, Any]:
        """Content-only description of this provider (no timestamps, no secrets)."""

    def send(self, request: ProviderRequest) -> ProviderResponse:
        """Send one attempt. Raise `ProviderError` on failure."""


# --------------------------------------------------------------------------- fake provider

FAKE_STEP_KINDS = (
    "answer",
    "abstain",
    "empty",
    "truncated",
    "refusal",
    "retryable_error",
    "non_retryable_error",
    "auth_error",
    "connect_error",
    "missing_usage",
    "timeout_unknown",
    "ambiguous",
    "crash_after_send",
)

FAKE_BYTES_PER_TOKEN = 4
_DEFAULT_ANSWER = "FAKE ANSWER"
_DEFAULT_TRUNCATED = "FAKE PARTIAL ANSWER"
_DEFAULT_REFUSAL = "FAKE REFUSAL"


@dataclass(frozen=True)
class FakeStep:
    """One scripted outcome for one attempt.

    `text` overrides the reply text of answer-like kinds. `returned_model` overrides the model
    the fake reports back. `http_status` overrides the status of `retryable_error` (default 408)
    and `non_retryable_error` (default 400). `message` overrides the refusal text or error message.

    `retryable_error` exercises the driver's retry path. It is a `rejected`, retryable failure whatever
    status it carries. The real adapter yields that only for HTTP 408 and for 503 with code
    `server_is_overloaded`, so the fake matches the real adapter only for those two. A real 429 or 502 is
    `unknown` and not retryable. Use `timeout_unknown` or `ambiguous` to script an unknown outcome.
    """

    kind: str
    text: str | None = None
    returned_model: str | None = None
    http_status: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in FAKE_STEP_KINDS:
            raise ValueError(f"unknown fake step kind {self.kind!r}; expected one of {FAKE_STEP_KINDS}")


@dataclass(frozen=True)
class FakeCall:
    """One recorded `send`."""

    seq: int
    request_id: str
    attempt_id: str
    step_kind: str


def script_by_question(
    prepared: Iterable[Mapping[str, Any]],
    steps_by_question: Mapping[str, Sequence[FakeStep]],
) -> dict[str, list[FakeStep]]:
    """Re-key a script from `question_id` to `request_id` using prepared-request records.

    Every question in `steps_by_question` must appear in `prepared`; a typo would otherwise
    leave a scripted failure unused and the test green.
    """
    request_by_question = {str(record["question_id"]): str(record["request_id"]) for record in prepared}
    missing = sorted(set(steps_by_question) - set(request_by_question))
    if missing:
        raise ValueError(f"script names questions with no prepared request: {missing}")
    return {request_by_question[question_id]: list(steps) for question_id, steps in steps_by_question.items()}


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _message_bytes(messages: Iterable[Mapping[str, Any]]) -> int:
    return sum(len(str(message.get("content", "")).encode("utf-8")) for message in messages)


def _fake_error(step: FakeStep, *, kind: str, outcome: str, retryable: bool, status: int | None, message: str) -> ProviderError:
    return ProviderError(
        step.message or message,
        kind=kind,
        outcome=outcome,
        retryable=retryable,
        http_status=status,
        raw={"simulated": True, "step_kind": step.kind},
    )


class FakeProvider:
    """Deterministic scripted provider. See the module docstring for the script format."""

    def __init__(
        self,
        script: Mapping[str, Sequence[FakeStep]],
        *,
        default: FakeStep,
        model: str,
    ) -> None:
        self._script = {key: tuple(steps) for key, steps in script.items()}
        self._default = default
        self._model = model
        self._cursor: dict[str, int] = {}
        self.calls: list[FakeCall] = []

    def identity(self) -> dict[str, Any]:
        script_payload = {
            "script": {key: [asdict(step) for step in steps] for key, steps in sorted(self._script.items())},
            "default": asdict(self._default),
        }
        digest = hashlib.sha256(json.dumps(script_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return {
            "provider": "fake",
            "simulated": True,
            "requested_model": self._model,
            "endpoint": None,
            "params": {},
            "sdk": None,
            "engineering_only": True,
            "model_calls": False,
            "fake_script_sha256": digest.hexdigest(),
        }

    def _next_step(self, request_id: str) -> FakeStep:
        index = self._cursor.get(request_id, 0)
        self._cursor[request_id] = index + 1
        steps = self._script.get(request_id, ())
        return steps[index] if index < len(steps) else self._default

    def send(self, request: ProviderRequest) -> ProviderResponse:
        step = self._next_step(request.request_id)
        self.calls.append(FakeCall(len(self.calls) + 1, request.request_id, request.attempt_id, step.kind))
        kind = step.kind
        if kind == "crash_after_send":
            raise SimulatedCrash(f"simulated crash after send of {request.attempt_id}")
        if kind == "retryable_error":
            status = step.http_status or 408
            raise _fake_error(
                step,
                kind=KIND_TRANSIENT_STATUS if status == 408 else KIND_RATE_LIMIT if status == 429 else KIND_SERVER_ERROR,
                outcome=OUTCOME_REJECTED,
                retryable=True,
                status=status,
                message=f"simulated retryable HTTP {status}",
            )
        if kind == "non_retryable_error":
            status = step.http_status or 400
            raise _fake_error(
                step,
                kind=KIND_BAD_REQUEST,
                outcome=OUTCOME_REJECTED,
                retryable=False,
                status=status,
                message=f"simulated non-retryable HTTP {status}",
            )
        if kind == "auth_error":
            raise _fake_error(
                step, kind=KIND_AUTH, outcome=OUTCOME_REJECTED, retryable=False, status=401, message="simulated HTTP 401"
            )
        if kind == "connect_error":
            raise _fake_error(
                step,
                kind=KIND_CONNECT,
                outcome=OUTCOME_NOT_SENT,
                retryable=True,
                status=None,
                message="simulated connect failure",
            )
        if kind == "timeout_unknown":
            raise _fake_error(
                step,
                kind=KIND_TIMEOUT,
                outcome=OUTCOME_UNKNOWN,
                retryable=False,
                status=None,
                message="simulated read timeout after send",
            )
        if kind == "ambiguous":
            raise _fake_error(
                step,
                kind=KIND_AMBIGUOUS,
                outcome=OUTCOME_UNKNOWN,
                retryable=False,
                status=None,
                message="simulated connection loss after send",
            )
        return self._reply(step, request)

    def _reply(self, step: FakeStep, request: ProviderRequest) -> ProviderResponse:
        kind = step.kind
        text: str | None
        refusal: str | None = None
        finish_reason = "stop"
        if kind == "abstain":
            text = ABSTENTION_TEXT
        elif kind == "empty":
            text = ""
        elif kind == "truncated":
            text = step.text if step.text is not None else _DEFAULT_TRUNCATED
            finish_reason = "length"
        elif kind == "refusal":
            text = None
            refusal = step.message or _DEFAULT_REFUSAL
        else:  # answer, missing_usage
            text = step.text if step.text is not None else _DEFAULT_ANSWER

        input_tokens = _ceil_div(_message_bytes(request.messages), FAKE_BYTES_PER_TOKEN)
        if kind == "truncated":
            output_tokens = request.max_output_tokens
        else:
            body = text if text is not None else refusal or ""
            output_tokens = min(_ceil_div(len(body.encode("utf-8")), FAKE_BYTES_PER_TOKEN), request.max_output_tokens)
        if kind == "missing_usage":
            usage = ProviderUsage()
            raw_usage: dict[str, Any] | None = None
        else:
            usage = ProviderUsage(
                input_tokens=input_tokens,
                cached_input_tokens=0,
                output_tokens=output_tokens,
                reasoning_tokens=0,
            )
            raw_usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        returned_model = step.returned_model or self._model
        response_id = f"fake-{request.attempt_id}"
        raw: dict[str, Any] = {
            "simulated": True,
            "object": "fake.chat.completion",
            "id": response_id,
            "model": returned_model,
            "system_fingerprint": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": text, "refusal": refusal},
                }
            ],
            "usage": raw_usage,
        }
        return ProviderResponse(
            text=text,
            finish_reason=finish_reason,
            refusal=refusal,
            returned_model=returned_model,
            response_id=response_id,
            usage=usage,
            raw=raw,
        )


# --------------------------------------------------------------------------- OpenAI adapter

# Not used by the adapter any more: it checks `ALLOWED_OPENAI_PARAMS`. `faar.live_runner.parse_provider_config`
# still imports this name for its own check. Delete it once that check uses the allowlist too.
_MANAGED_PARAM_KEYS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "timeout",
        "n",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "parallel_tool_calls",
        "extra_headers",
        "extra_query",
        "extra_body",
        "store",
        "metadata",
    }
)
_TOKEN_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens")
_MESSAGE_LIMIT = 1000


def _check_allowed_params(params: Mapping[str, Any], where: str) -> None:
    """Refuse any key outside ``ALLOWED_OPENAI_PARAMS``. The adapter sets model, messages, the token limit and
    the timeout itself, so those keys are not allowed here either."""
    extra = sorted(str(key) for key in params if key not in ALLOWED_OPENAI_PARAMS)
    if extra:
        raise ValueError(f"{where} keys not allowed: {extra}; allowed keys: {list(ALLOWED_OPENAI_PARAMS)}")


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sdk_version() -> str | None:
    try:
        return importlib_metadata.version("openai")
    except importlib_metadata.PackageNotFoundError:
        return None


class OpenAIChatProvider:
    """Adapter for `client.chat.completions.create`. Non-streaming, no tools.

    `client` is an already-built `openai.OpenAI` (the caller decides where credentials and the
    transport come from). It must have `max_retries=0`: the SDK's hidden retries would send extra
    requests that the run driver could neither count nor bound. Its httpx client must not follow
    redirects: a followed redirect is a second request, possibly to another host, that the driver
    could neither count nor bound. The constructor refuses a client that does.

    `token_limit_param` names the request field that carries `max_output_tokens`. The default,
    `max_tokens`, is deprecated by OpenAI and rejected by o-series models, which need
    `max_completion_tokens`. The choice is part of `identity()`.

    `tag_attempts=True` is opt-in and untested against the live API. It sends
    `X-Client-Request-Id: <attempt_id>`, `store=True` and `metadata` holding the attempt and request
    ids, so an operator can look an unknown attempt up with `GET /v1/chat/completions?metadata[...]`.
    It stores the prompt and reply on OpenAI's side, so enabling it is a data-handling decision.
    """

    def __init__(
        self,
        model: str,
        params: Mapping[str, Any],
        *,
        client: Any,
        token_limit_param: str = "max_tokens",
        tag_attempts: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if token_limit_param not in _TOKEN_LIMIT_PARAMS:
            raise ValueError(f"token_limit_param must be one of {_TOKEN_LIMIT_PARAMS}")
        _check_allowed_params(params, "params")
        retries = getattr(client, "max_retries", None)
        if retries != 0:
            raise ValueError(f"client must be built with max_retries=0, found {retries!r}")
        # openai-python 1.68.2 keeps the httpx client in `client._client` (_base_client.py:825) and
        # httpx keeps the setting in `follow_redirects` (httpx/_client.py:213). The SDK's own default client
        # sets it to True (_base_client.py:754), so the caller must pass `http_client`. Anything but an
        # explicit False, including a client we cannot inspect, is refused.
        follows = getattr(getattr(client, "_client", None), "follow_redirects", None)
        if follows is not False:
            raise ValueError(
                "client must not follow redirects: pass http_client=httpx.Client(follow_redirects=False), "
                f"found follow_redirects={follows!r}"
            )
        self._model = model
        self._params = dict(params)
        self._client = client
        self._token_limit_param = token_limit_param
        self._tag_attempts = tag_attempts

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "simulated": False,
            "requested_model": self._model,
            "endpoint": str(getattr(self._client, "base_url", "")) or None,
            "api": "chat.completions",
            "params": dict(self._params),
            "token_limit_param": self._token_limit_param,
            "tag_attempts": self._tag_attempts,
            "sdk": {"name": "openai", "version": _sdk_version()},
            "engineering_only": False,
            "model_calls": True,
        }

    def send(self, request: ProviderRequest) -> ProviderResponse:
        _check_allowed_params(request.params or {}, "request.params")
        if request.params and dict(request.params) != self._params:
            raise ValueError("request.params differ from the params this provider was built with")
        if not isinstance(request.max_output_tokens, int) or request.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if not request.timeout_seconds or request.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        call: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(message) for message in request.messages],
            self._token_limit_param: request.max_output_tokens,
            "timeout": request.timeout_seconds,
            **self._params,
        }
        if self._tag_attempts:
            call["store"] = True
            call["metadata"] = {"faar_attempt_id": request.attempt_id, "faar_request_id": request.request_id}
            if request.attempt_id.isascii() and len(request.attempt_id) <= 512:
                call["extra_headers"] = {"X-Client-Request-Id": request.attempt_id}
        try:
            completion = self._client.chat.completions.create(**call)
        except Exception as exc:
            raise classify_openai_error(exc) from exc
        return self._to_response(completion)

    @staticmethod
    def _to_response(completion: Any) -> ProviderResponse:
        dump = getattr(completion, "model_dump", None)
        if not callable(dump):
            # The SDK returns the body text when a 200 reply is not JSON. A reply arrived, so the
            # attempt may have been billed, but there is nothing to record as an answer.
            raise ProviderError(
                f"200 reply was not a chat completion ({type(completion).__name__})",
                kind=KIND_MALFORMED_RESPONSE,
                outcome=OUTCOME_UNKNOWN,
                retryable=False,
                raw={"body_excerpt": str(completion)[:2000]},
            )
        raw: dict[str, Any] = dump(mode="json")
        choices = raw.get("choices")
        reason = None
        if not isinstance(choices, list) or not choices:
            reason = "choices is missing or empty"
        elif not isinstance(choices[0], Mapping):
            reason = "choices[0] is not an object"
        elif not isinstance(choices[0].get("message"), Mapping):
            reason = "choices[0] has no message object"
        if reason is not None:
            # A reply arrived, so the attempt may have been billed, but it holds no assistant message to
            # record as an answer. The raw payload (with any usage) stays on the error for the operator.
            raise ProviderError(
                f"200 reply was not a usable chat completion: {reason}",
                kind=KIND_MALFORMED_RESPONSE,
                outcome=OUTCOME_UNKNOWN,
                retryable=False,
                raw={"reason": reason, "payload": raw},
            )
        choice = _mapping(choices[0])
        message = _mapping(choice.get("message"))
        usage = _mapping(raw.get("usage"))
        usage_out = ProviderUsage(
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            cached_input_tokens=_int_or_none(_mapping(usage.get("prompt_tokens_details")).get("cached_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            reasoning_tokens=_int_or_none(_mapping(usage.get("completion_tokens_details")).get("reasoning_tokens")),
        )
        text = message.get("content")
        refusal = message.get("refusal")
        finish_reason = choice.get("finish_reason")
        returned_model = raw.get("model")
        response_id = raw.get("id")
        return ProviderResponse(
            text=text if isinstance(text, str) else None,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            refusal=refusal if isinstance(refusal, str) else None,
            returned_model=returned_model if isinstance(returned_model, str) else None,
            response_id=response_id if isinstance(response_id, str) else None,
            usage=usage_out,
            raw=raw,
        )


# --------------------------------------------------------------------------- error classification

_QUOTA_MARKERS = ("insufficient_quota", "quota", "spend limit", "credit balance", "usage limit")
# 429 error codes on the error-codes page for billing, spend and usage limits. The page says retrying
# them does not restore access, so they stop the run. The message text is not documented, so match the code.
_QUOTA_CODES = (
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "insufficient_quota",
)
# The only error code of a 503 that OpenAI documents as a refusal to process the request.
_OVERLOAD_CODE = "server_is_overloaded"
_BODY_LIMIT = 4000
_HEADER_LIMIT = 200


def _clip(text: str) -> str:
    return text if len(text) <= _MESSAGE_LIMIT else text[:_MESSAGE_LIMIT] + "..."


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _bounded_body(body: Any) -> Any:
    """The response body for `ProviderError.raw`, cut to `_BODY_LIMIT` characters.

    A parsed JSON body stays a JSON value while it fits. A longer one, and any non-JSON text, is
    stored as text with a `truncated` marker, so the event log line stays small and keeps the ids.
    """
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        try:
            text = json.dumps(body, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(body)
        if len(text) <= _BODY_LIMIT:
            return _json_safe(body)
    else:
        text = str(body)
        if len(text) <= _BODY_LIMIT:
            return text
    return {"truncated": True, "length": len(text), "text": text[:_BODY_LIMIT]}


def _header(headers: Any, name: str) -> str | None:
    value = headers.get(name)
    return None if value is None else str(value)[:_HEADER_LIMIT]


# Which HTTP statuses count as `rejected` and which as `unknown`.
#
# `rejected` needs proof that the provider did not process the request. A recommendation to try again
# is not proof. OpenAI documents no idempotency key for Chat Completions, so a resend after an
# unprocessed-looking failure can bill twice. Without proof the outcome is `unknown`: the run driver
# never resends the request, keeps its reservation and waits for `reconcile`.
#
# Sources, all retrieved 2026-09-29:
#   E1  OpenAI, Error codes, https://developers.openai.com/api/docs/guides/error-codes. Entries exist for
#       400 (service_tier), 401, 403, 429 (credit_balance_exhausted, rate limit, slow_down, spend and
#       usage limits), 500 and 503 (server_is_overloaded). There is no 502, 504, 408 or 409 entry and no
#       statement on billing failed requests. 500: "Retry your request after a brief wait". 429: "follow
#       the Retry-After header". Both are retry recommendations. Billing, spend and quota errors: "Retrying
#       billing, spend, or quota errors won't restore API access".
#   E2  Same page, section "503 - Model temporarily overloaded": the model "does not have enough capacity
#       to process your request at the moment". It is the only OpenAI text that says a failed request was
#       not processed, and it covers type service_unavailable_error with code server_is_overloaded only.
#   E3  OpenAI, Rate limits, https://developers.openai.com/api/docs/guides/rate-limits. Section "Retrying
#       with exponential backoff": "unsuccessful requests contribute to your per-minute limit". Section
#       "What are some steps I can take to mitigate this?": the limit is computed from max_tokens and a
#       character-count estimate, which suggests a check before generation. The guide never says a 429
#       request was not processed or not billed.
#   E4  RFC 9110, https://www.rfc-editor.org/rfc/rfc9110.html. 15.5.9 (408): the server "did not receive a
#       complete request message", and the client MAY repeat it. 15.5.10 (409): "the user might be able to
#       resolve the conflict and resubmit". 15.6.1 (500): an "unexpected condition that prevented it from
#       fulfilling the request". 15.6.3 (502): a gateway "received an invalid response from an inbound
#       server". 15.6.4 (503): "unable to handle the request due to a temporary overload or scheduled
#       maintenance". 15.6.5 (504): no "timely response from an upstream server". None of the 5xx
#       definitions says the request was not applied. 9.2.2: a client "SHOULD NOT automatically retry a
#       request with a non-idempotent method unless it has some means to know that the request semantics
#       are actually idempotent ... or some means to detect that the original request was never applied".
#   E5  openai-python 1.68.2, `_base_client.py` `_should_retry`: it resends on `x-should-retry: true`, on
#       408, 409, 429 and on every status from 500. That is a client default for retrying. This adapter
#       turns it off with `max_retries=0`, and the code is not documentation that the request was not
#       processed.
#   E6  RFC 9110 15.5.1 (400): the server "cannot or will not process the request". 15.5.2 (401): "the
#       request has not been applied". 15.5.4 (403): the server "refuses to fulfill it". 15.5.21 (422):
#       the server "was unable to process the contained instructions". 15.5.5 (404) and the rest of 15.5
#       are client errors. A non-retryable `rejected` never causes a resend, so a wrong `rejected` here
#       adds no second charge. The reservation is kept for every `rejected` attempt anyway.
#   E7  RFC 9110 15.4 (3xx): "further action needs to be taken by the user agent in order to fulfill the
#       request". 15.4.4 (303): the redirect target gives "an indirect response to the original request",
#       and it is "primarily used" to send the output of a POST action elsewhere, so the POST was
#       processed. No 3xx definition says the request was not applied.
#
#   status                                      outcome   retry  basis
#   408                                         rejected  yes    E4 15.5.9
#   503 with code server_is_overloaded          rejected  yes    E2
#   500, 502, 503 otherwise, other 5xx          unknown   no     E1, E4, E5: recommendation only
#   504, 522, 524                               unknown   no     a gateway can give up after the provider finished
#   429 rate limit                              unknown   no     E1 and E3 recommend waiting, none says "not processed"
#   429 quota, billing, spend, usage limit      rejected  no     E1, stops the run
#   409                                         unknown   no     E4 15.5.10: a person resolves a conflict
#   3xx (redirects are never followed)          unknown   no     E7, E4 9.2.2: the request reached a server
#   400, 401, 403, 404, 422, other 4xx          rejected  no     E6
#
# 429 is the one row that could go the other way. If OpenAI states that a rate-limited request is never
# processed, change the rate-limit branch below and `STATUS_CASES` in tests/test_answer_providers.py.
def _classify_status(exc: Any) -> ProviderError:
    status = int(exc.status_code)
    code = exc.code if isinstance(exc.code, str) else None
    error_type = exc.type if isinstance(exc.type, str) else None
    message = _clip(str(exc.message))
    headers = exc.response.headers
    lowered = message.lower()
    # Diagnostics for reconciliation. The headers are the provider's response headers, never the request's,
    # so the Authorization header and the key are not in here.
    raw = {
        "error_class": type(exc).__name__,
        "http_status": status,
        "x_request_id": _header(headers, "x-request-id"),
        "retry_after": _header(headers, "retry-after"),
        "x_should_retry": _header(headers, "x-should-retry"),
        "error_code": code,
        "error_type": error_type,
        "body": _bounded_body(exc.body),
    }
    server_forbids_retry = headers.get("x-should-retry") == "false"

    def build(kind: str, outcome: str, retryable: bool) -> ProviderError:
        # Only a `rejected` outcome can be retryable. `x-should-retry: true` never changes an outcome.
        allowed = retryable and outcome == OUTCOME_REJECTED and not server_forbids_retry
        return ProviderError(message, kind=kind, outcome=outcome, retryable=allowed, http_status=status, raw=raw)

    is_missing_model = code == "model_not_found" or (status == 404 and code is None and "model" in lowered)
    if is_missing_model and status in (400, 403, 404):
        return build(KIND_UNKNOWN_MODEL, OUTCOME_REJECTED, False)
    if 300 <= status < 400:
        # The request reached a server. With redirects switched off (the constructor checks) the reply
        # arrives here. Nothing shows the request was not processed, and a resend would go to the same
        # address. The `location` header stays for the operator, clipped.
        raw["location"] = _header(headers, "location")
        return build(KIND_REDIRECT, OUTCOME_UNKNOWN, False)
    if status in (401, 403):
        return build(KIND_AUTH, OUTCOME_REJECTED, False)
    if status == 429:
        marked = " ".join(filter(None, (code, error_type, lowered)))
        if code in _QUOTA_CODES or any(marker in marked for marker in _QUOTA_MARKERS):
            return build(KIND_QUOTA, OUTCOME_REJECTED, False)
        return build(KIND_RATE_LIMIT, OUTCOME_UNKNOWN, False)
    if status == 408:
        return build(KIND_TRANSIENT_STATUS, OUTCOME_REJECTED, True)
    if status == 409:
        return build(KIND_TRANSIENT_STATUS, OUTCOME_UNKNOWN, False)
    if status in _GATEWAY_TIMEOUT_STATUSES:
        # A gateway can time out after the provider finished the request, so the attempt may have been
        # processed and billed.
        return build(KIND_GATEWAY_TIMEOUT, OUTCOME_UNKNOWN, False)
    if status == 503 and code == _OVERLOAD_CODE:
        return build(KIND_SERVER_ERROR, OUTCOME_REJECTED, True)
    if status >= 500:
        # 500, 502 and other 5xx say the provider or a gateway failed. They do not say the request was
        # not processed, so nothing is resent. `x-should-retry: true` and OpenAI's "retry after a brief
        # wait" are recommendations to try again, not proof.
        return build(KIND_SERVER_ERROR, OUTCOME_UNKNOWN, False)
    if status == 404:
        return build(KIND_NOT_FOUND, OUTCOME_REJECTED, False)
    if status in (400, 422):
        return build(KIND_BAD_REQUEST, OUTCOME_REJECTED, False)
    return build(KIND_CLIENT_ERROR, OUTCOME_REJECTED, False)


def _classify_transport(exc: Exception, cause: BaseException | None, httpx: Any, *, timed_out: bool) -> ProviderError:
    """Place a connection-level failure by the httpx exception the SDK wrapped."""
    detail = _clip(f"{type(cause).__name__}: {cause}" if cause is not None else str(exc))
    raw = {"error_class": type(exc).__name__, "cause_class": type(cause).__name__ if cause is not None else None}

    def build(kind: str, outcome: str, retryable: bool) -> ProviderError:
        return ProviderError(detail, kind=kind, outcome=outcome, retryable=retryable, raw=raw)

    # The request provably did not leave the process.
    if isinstance(cause, httpx.ConnectTimeout):
        return build(KIND_CONNECT_TIMEOUT, OUTCOME_NOT_SENT, True)
    if isinstance(cause, httpx.PoolTimeout):
        return build(KIND_POOL_TIMEOUT, OUTCOME_NOT_SENT, True)
    if isinstance(cause, (httpx.ConnectError, httpx.ProxyError)):
        return build(KIND_CONNECT, OUTCOME_NOT_SENT, True)
    if isinstance(cause, (httpx.UnsupportedProtocol, httpx.LocalProtocolError, httpx.InvalidURL)):
        return build(KIND_CLIENT_CONFIG, OUTCOME_NOT_SENT, False)
    # The request may have reached the provider.
    if timed_out or isinstance(cause, httpx.TimeoutException):
        return build(KIND_TIMEOUT, OUTCOME_UNKNOWN, False)
    return build(KIND_CONNECTION_LOST, OUTCOME_UNKNOWN, False)


def classify_openai_error(exc: BaseException) -> ProviderError:
    """Map an exception raised by `client.chat.completions.create` to a `ProviderError`.

    Conservative rule: say `not_sent` only for failures that happen before any request byte is
    written; say `rejected` only when a status code arrived; everything else is `unknown`.
    """
    import httpx
    import openai

    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, openai.APIStatusError):
        return _classify_status(exc)
    if isinstance(exc, openai.APIResponseValidationError):
        return ProviderError(
            _clip(str(exc)),
            kind=KIND_MALFORMED_RESPONSE,
            outcome=OUTCOME_UNKNOWN,
            retryable=False,
            http_status=getattr(exc, "status_code", None),
            raw={"error_class": type(exc).__name__, "body": _json_safe(exc.body)},
        )
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return _classify_transport(exc, exc.__cause__, httpx, timed_out=isinstance(exc, openai.APITimeoutError))
    if isinstance(exc, httpx.TransportError):  # a caller that bypassed the SDK wrapper
        return _classify_transport(exc, exc, httpx, timed_out=False)
    return ProviderError(
        _clip(f"{type(exc).__name__}: {exc}"),
        kind=KIND_UNEXPECTED,
        outcome=OUTCOME_UNKNOWN,
        retryable=False,
        raw={"error_class": type(exc).__name__},
    )
