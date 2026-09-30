"""Tests for faar.answer_providers.

The failure list these tests cover is in the docstring of `faar/answer_providers.py`
(items F1-F10 for the fake provider, O1-O13 for the OpenAI adapter). No test sends a
request: the OpenAI client gets an `httpx.MockTransport`, and the conftest blocks
non-loopback sockets besides.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
import pytest

from faar import answer_providers as ap
from faar.answer_providers import (
    FAKE_STEP_KINDS,
    AnswerProvider,
    FakeProvider,
    FakeStep,
    OpenAIChatProvider,
    SimulatedCrash,
    classify_openai_error,
    script_by_question,
)
from faar.live_contract import (
    ALLOWED_OPENAI_PARAMS,
    SERVICE_TIER_STANDARD,
    STORAGE_DISABLED,
    STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP,
    STORAGE_POLICIES,
    ProviderError,
    ProviderRequest,
    ProviderUsage,
)

ROOT = Path(__file__).resolve().parents[1]


def make_request(request_id: str = "r1", attempt: int = 1, *, max_output_tokens: int = 64, params: dict | None = None) -> ProviderRequest:
    return ProviderRequest(
        request_id=request_id,
        attempt_id=f"{request_id}-a{attempt}",
        messages=({"role": "system", "content": "sys"}, {"role": "user", "content": "question? " * 5}),
        params=params or {},
        max_output_tokens=max_output_tokens,
        timeout_seconds=30.0,
    )


# --------------------------------------------------------------------------- fake provider


def fake(script: dict[str, list[FakeStep]] | None = None, default: str = "answer", model: str = "fake-model") -> FakeProvider:
    return FakeProvider(script or {}, default=FakeStep(default), model=model)


def test_every_step_kind_is_covered_by_a_case_below() -> None:
    covered = {
        "answer", "abstain", "empty", "truncated", "refusal", "retryable_error", "non_retryable_error",
        "auth_error", "connect_error", "missing_usage", "timeout_unknown", "ambiguous", "crash_after_send",
    }
    assert covered == set(FAKE_STEP_KINDS)


def test_unknown_step_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        FakeStep("succeeds_sometimes")


def test_fake_answer_fields_and_deterministic_usage() -> None:
    request = make_request()
    response = fake().send(request)
    assert response.text == "FAKE ANSWER"
    assert response.finish_reason == "stop"
    assert response.refusal is None
    assert response.returned_model == "fake-model"
    assert response.response_id == "fake-r1-a1"
    total_bytes = len("sys") + len("question? " * 5)
    expected_input = -(-total_bytes // 4)
    assert response.usage == ProviderUsage(
        input_tokens=expected_input, cached_input_tokens=0, output_tokens=3, reasoning_tokens=0
    )
    assert response.raw["simulated"] is True
    assert response.raw["usage"]["total_tokens"] == expected_input + 3


def test_fake_usage_counts_utf8_bytes_and_caps_output() -> None:
    request = ProviderRequest("r", "r-a1", ({"role": "user", "content": "é" * 8},), {}, 2, 5.0)
    response = FakeProvider({}, default=FakeStep("answer", text="x" * 100), model="m").send(request)
    assert response.usage.input_tokens == 4  # 16 bytes / 4
    assert response.usage.output_tokens == 2  # capped by max_output_tokens


def test_fake_custom_text_abstain_empty() -> None:
    assert fake({"r1": [FakeStep("answer", text="Paris")]}).send(make_request()).text == "Paris"
    abstain = fake(default="abstain").send(make_request())
    assert abstain.text == "NO_ANSWER" and abstain.finish_reason == "stop"
    empty = fake(default="empty").send(make_request())
    assert empty.text == "" and empty.usage.output_tokens == 0


def test_fake_truncated_sets_length_and_spends_the_limit() -> None:
    response = fake(default="truncated").send(make_request(max_output_tokens=7))
    assert response.finish_reason == "length"
    assert response.text
    assert response.usage.output_tokens == 7


def test_fake_refusal_sets_refusal_and_no_text() -> None:
    response = fake(default="refusal").send(make_request())
    assert response.text is None
    assert response.refusal == "FAKE REFUSAL"
    assert response.raw["choices"][0]["message"]["refusal"] == "FAKE REFUSAL"


def test_fake_missing_usage_is_none_not_zero() -> None:
    response = fake(default="missing_usage").send(make_request())
    assert response.text == "FAKE ANSWER"
    assert response.usage == ProviderUsage()
    assert response.usage.as_dict() == {
        "input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
    }
    assert response.raw["usage"] is None


@pytest.mark.parametrize(
    ("step", "kind", "outcome", "retryable", "status"),
    [
        (FakeStep("retryable_error"), "transient_status", "rejected", True, 408),
        (FakeStep("retryable_error", http_status=503), "server_error", "rejected", True, 503),
        (FakeStep("non_retryable_error"), "bad_request", "rejected", False, 400),
        (FakeStep("auth_error"), "auth", "rejected", False, 401),
        (FakeStep("connect_error"), "connect", "not_sent", True, None),
        (FakeStep("timeout_unknown"), "timeout", "unknown", False, None),
        (FakeStep("ambiguous"), "ambiguous", "unknown", False, None),
    ],
)
def test_fake_error_kinds(step: FakeStep, kind: str, outcome: str, retryable: bool, status: int | None) -> None:
    provider = fake({"r1": [step]})
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    error = info.value
    assert (error.kind, error.outcome, error.retryable, error.http_status) == (kind, outcome, retryable, status)
    assert error.raw["simulated"] is True
    assert len(provider.calls) == 1  # recorded even though it failed


def test_fake_crash_after_send_records_the_call_and_escapes_except_exception() -> None:
    provider = fake({"r1": [FakeStep("crash_after_send")]})
    swallowed = False
    with pytest.raises(SimulatedCrash):
        try:
            provider.send(make_request())
        except Exception:  # pragma: no cover - must not run
            swallowed = True
    assert not swallowed
    assert not issubclass(SimulatedCrash, Exception)
    assert [(c.request_id, c.attempt_id, c.step_kind) for c in provider.calls] == [("r1", "r1-a1", "crash_after_send")]


def test_fake_script_advances_per_request_and_falls_back_to_default() -> None:
    script = {"r1": [FakeStep("retryable_error"), FakeStep("answer", text="second")]}
    provider = fake(script, default="abstain")
    with pytest.raises(ProviderError):
        provider.send(make_request("r1", 1))
    # Another request is unaffected by r1's progress and uses the default.
    assert provider.send(make_request("r2", 1)).text == "NO_ANSWER"
    assert provider.send(make_request("r1", 2)).text == "second"
    assert provider.send(make_request("r1", 3)).text == "NO_ANSWER"  # script used up
    assert [(c.seq, c.request_id, c.attempt_id, c.step_kind) for c in provider.calls] == [
        (1, "r1", "r1-a1", "retryable_error"),
        (2, "r2", "r2-a1", "abstain"),
        (3, "r1", "r1-a2", "answer"),
        (4, "r1", "r1-a3", "abstain"),
    ]


def test_fake_is_deterministic_across_instances() -> None:
    script = {"r1": [FakeStep("connect_error"), FakeStep("answer")]}

    def run() -> list[Any]:
        provider = fake(script)
        out: list[Any] = []
        for attempt in (1, 2, 3):
            try:
                response = provider.send(make_request("r1", attempt))
                out.append(("ok", response.text, response.response_id, response.usage.as_dict(), json.dumps(response.raw, sort_keys=True)))
            except ProviderError as error:
                out.append(("err", error.kind, str(error)))
        return out

    assert run() == run()


def test_fake_requested_and_returned_model_are_separate() -> None:
    provider = fake({"r1": [FakeStep("answer", returned_model="fake-model-2026-01-01")]}, model="fake-model")
    response = provider.send(make_request())
    assert provider.identity()["requested_model"] == "fake-model"
    assert response.returned_model == "fake-model-2026-01-01"
    assert response.raw["model"] == "fake-model-2026-01-01"


def test_fake_identity_is_simulated_and_tracks_the_script() -> None:
    a = fake({"r1": [FakeStep("answer")]}).identity()
    b = fake({"r1": [FakeStep("abstain")]}).identity()
    c = fake({"r1": [FakeStep("answer")]}).identity()
    assert a["provider"] == "fake" and a["simulated"] is True
    assert a["model_calls"] is False and a["engineering_only"] is True
    assert a["fake_script_sha256"] == c["fake_script_sha256"] != b["fake_script_sha256"]
    json.dumps(a)  # JSON-serialisable


def test_script_by_question_maps_and_rejects_unknown_questions() -> None:
    prepared = [{"question_id": "q1", "request_id": "rq1"}, {"question_id": "q2", "request_id": "rq2"}]
    script = script_by_question(prepared, {"q2": [FakeStep("abstain")]})
    assert list(script) == ["rq2"]
    with pytest.raises(ValueError, match="q9"):
        script_by_question(prepared, {"q9": [FakeStep("abstain")]})


def test_fake_and_adapter_satisfy_the_protocol() -> None:
    client, _ = mock_client(lambda request: httpx.Response(200, json=completion_payload()))
    assert isinstance(fake(), AnswerProvider)
    assert isinstance(OpenAIChatProvider("m", {}, client=client), AnswerProvider)


# --------------------------------------------------------------------------- OpenAI adapter on a mock transport


def completion_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test-2026-01-01",
        "system_fingerprint": "fp_test",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Paris", "refusal": None}}
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 9,
            "total_tokens": 129,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }
    payload.update(overrides)
    return payload


def mock_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "test-not-a-key",
    follow_redirects: bool = False,
    trust_env: bool = False,
) -> tuple[openai.OpenAI, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = openai.OpenAI(
        api_key=api_key,
        http_client=httpx.Client(transport=httpx.MockTransport(recording), follow_redirects=follow_redirects, trust_env=trust_env),
        max_retries=0,
    )
    return client, seen


def adapter(
    handler: Callable[[httpx.Request], httpx.Response],
    params: dict | None = None,
    *,
    api_key: str = "test-not-a-key",
    **kwargs: Any,
) -> tuple[OpenAIChatProvider, list[httpx.Request]]:
    client, seen = mock_client(handler, api_key=api_key)
    return OpenAIChatProvider("gpt-test", params or {}, client=client, **kwargs), seen


def error_body(message: str, *, code: str | None = None, type_: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def test_success_maps_every_field_and_keeps_the_raw_payload() -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload(), headers={"x-request-id": "req_abc"}))
    response = provider.send(make_request())
    assert len(seen) == 1  # the mock saw the request; nothing else could be reached
    assert response.text == "Paris"
    assert response.finish_reason == "stop"
    assert response.refusal is None
    assert response.returned_model == "gpt-test-2026-01-01"
    assert response.response_id == "chatcmpl-test-1"
    assert response.usage == ProviderUsage(input_tokens=120, cached_input_tokens=100, output_tokens=9, reasoning_tokens=4)
    assert response.raw["system_fingerprint"] == "fp_test"
    assert response.raw["choices"][0]["message"]["content"] == "Paris"
    json.dumps(response.raw)


def test_request_body_and_headers() -> None:
    provider, seen = adapter(
        lambda request: httpx.Response(200, json=completion_payload()), {"temperature": 0, "seed": 7}
    )
    provider.send(make_request(max_output_tokens=33))
    request = seen[0]
    body = json.loads(request.content)
    assert request.method == "POST" and request.url.path.endswith("/chat/completions")
    assert body == {
        "model": "gpt-test",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question? " * 5}],
        "max_tokens": 33,
        "temperature": 0,
        "seed": 7,
        "store": False,
        "service_tier": "default",
    }
    assert "x-client-request-id" not in request.headers
    assert request.extensions["timeout"] == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}
    assert request.headers["authorization"] == "Bearer test-not-a-key"


def test_max_completion_tokens_option_and_attempt_lookup_storage() -> None:
    provider, seen = adapter(
        lambda request: httpx.Response(200, json=completion_payload()),
        token_limit_param="max_completion_tokens",
        storage=STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP,
    )
    provider.send(make_request("abc", 2, max_output_tokens=10))
    body = json.loads(seen[0].content)
    assert body["max_completion_tokens"] == 10 and "max_tokens" not in body
    assert body["store"] is True
    assert body["metadata"] == {"faar_attempt_id": "abc-a2", "faar_request_id": "abc"}
    assert seen[0].headers["x-client-request-id"] == "abc-a2"
    identity = provider.identity()
    assert identity["token_limit_param"] == "max_completion_tokens"
    assert identity["storage"] == STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP and identity["store"] is True


def test_missing_usage_and_missing_details_are_none() -> None:
    payload = completion_payload()
    del payload["usage"]
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    response = provider.send(make_request())
    assert response.text == "Paris"
    assert response.usage == ProviderUsage()

    payload = completion_payload(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    usage = provider.send(make_request()).usage
    assert usage == ProviderUsage(input_tokens=10, cached_input_tokens=None, output_tokens=2, reasoning_tokens=None)


def test_length_finish_reason_reaches_the_response() -> None:
    payload = completion_payload()
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["message"]["content"] = "Par"
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    response = provider.send(make_request())
    assert (response.finish_reason, response.text) == ("length", "Par")


def test_refusal_reaches_the_response() -> None:
    payload = completion_payload()
    payload["choices"][0]["message"] = {"role": "assistant", "content": None, "refusal": "I can't help with that."}
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    response = provider.send(make_request())
    assert response.text is None
    assert response.refusal == "I can't help with that."


def _choice_without_message() -> dict[str, Any]:
    return {"index": 0, "finish_reason": "stop"}


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("empty choices", completion_payload(choices=[])),
        ("no choices key", {"id": "chatcmpl-x", "object": "chat.completion", "model": "m"}),
        ("json object that is not a completion", {"foo": 1}),
        ("choices is a string", completion_payload(choices="x")),
        ("choice is not an object", completion_payload(choices=[1])),
        ("choice without message", completion_payload(choices=[_choice_without_message()])),
        ("message is null", completion_payload(choices=[{**_choice_without_message(), "message": None}])),
        ("message is a string", completion_payload(choices=[{**_choice_without_message(), "message": "hi"}])),
    ],
)
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_200_reply_without_a_choice_message_is_a_malformed_response_not_an_answer(label: str, body: dict) -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    error = info.value
    assert len(seen) == 1, label
    assert (error.kind, error.outcome, error.retryable) == ("malformed_response", "unknown", False), label
    assert error.raw["payload"] is not None and error.raw["reason"], label  # the reply is kept for the operator
    if "usage" in body:
        assert error.raw["payload"]["usage"]["prompt_tokens"] == 120  # the reply was billed; the raw keeps the usage


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_a_malformed_200_puts_usage_id_and_model_at_the_top_of_raw() -> None:
    """A later bound on the payload's size must not drop the evidence of what the reply cost."""
    body = completion_payload(choices=[])
    provider, _ = adapter(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    raw = info.value.raw
    assert raw["usage"] == raw["payload"]["usage"]  # the SDK's dump adds None-valued detail fields
    assert (raw["usage"]["prompt_tokens"], raw["usage"]["completion_tokens"], raw["usage"]["total_tokens"]) == (120, 9, 129)
    assert (raw["id"], raw["model"]) == ("chatcmpl-test-1", "gpt-test-2026-01-01")
    provider, _ = adapter(lambda request: httpx.Response(200, json={"foo": 1}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert (info.value.raw["usage"], info.value.raw["id"], info.value.raw["model"]) == (None, None, None)


def test_null_content_with_a_refusal_string_stays_a_response() -> None:
    payload = completion_payload()
    payload["choices"][0]["message"] = {"role": "assistant", "content": None, "refusal": "No."}
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    response = provider.send(make_request())
    assert (response.text, response.refusal) == (None, "No.")


def test_message_with_null_content_and_no_refusal_stays_a_response() -> None:
    payload = completion_payload()
    payload["choices"][0]["message"] = {"role": "assistant", "content": None}
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    response = provider.send(make_request())
    assert response.text is None and response.refusal is None  # the parser records this as an empty reply


def test_non_json_200_body_is_a_malformed_response_not_an_answer() -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1
    assert (info.value.kind, info.value.outcome, info.value.retryable) == ("malformed_response", "unknown", False)


# Classification of HTTP statuses. `rejected` means the provider or a documented rule shows the request was
# not processed. `unknown` means it may have been processed and billed, so it is never resent automatically.
# A retry recommendation (OpenAI's error-codes page, the SDK's default retry set, `Retry-After`) is not
# proof that the request was not processed. The evidence for each row is in the comment above
# `_classify_status` in faar/answer_providers.py.
STATUS_CASES = [
    # 429: no OpenAI statement says a rate-limited request was not processed or billed.
    (429, error_body("Rate limit reached for requests", code="rate_limit_exceeded", type_="requests"), "rate_limit", "unknown", False),
    (429, error_body("Slow down", code="slow_down", type_="rate_limit_error"), "rate_limit", "unknown", False),
    # Billing, spend and quota errors: documented as errors that retrying cannot fix. They stop the run.
    # The documented codes must match even when the message text differs.
    (429, error_body("You exceeded your current quota", code="insufficient_quota", type_="insufficient_quota"), "quota", "rejected", False),
    (429, error_body("Project reached its enforced monthly spend limit"), "quota", "rejected", False),
    (429, error_body("Request refused.", code="credit_balance_exhausted"), "quota", "rejected", False),
    (429, error_body("Request refused.", code="organization_spend_limit_exceeded"), "quota", "rejected", False),
    (429, error_body("Request refused.", code="project_spend_limit_exceeded"), "quota", "rejected", False),
    (429, error_body("Request refused.", code="organization_usage_limit_exceeded"), "quota", "rejected", False),
    # 5xx: an upstream failure does not show that the request was not processed.
    (500, error_body("The server had an error while processing your request", type_="server_error"), "server_error", "unknown", False),
    (502, error_body("Bad gateway", type_="server_error"), "server_error", "unknown", False),
    (501, error_body("Not implemented"), "server_error", "unknown", False),
    (599, error_body("Odd gateway status"), "server_error", "unknown", False),
    (503, error_body("Service unavailable", type_="server_error"), "server_error", "unknown", False),
    (504, error_body("Gateway timeout", type_="server_error"), "gateway_timeout", "unknown", False),
    (522, error_body("Origin connection timed out"), "gateway_timeout", "unknown", False),
    (524, error_body("Origin took too long"), "gateway_timeout", "unknown", False),
    # 503 with the documented overload code: "does not have enough capacity to process your request".
    (
        503,
        error_body("The model is overloaded", code="server_is_overloaded", type_="service_unavailable_error"),
        "server_error",
        "rejected",
        True,
    ),
    # 408: RFC 9110 section 15.5.9 says the server did not receive a complete request.
    (408, error_body("Request timed out"), "transient_status", "rejected", True),
    # 409: RFC 9110 section 15.5.10 lets the user resolve a conflict and resubmit. It is not a blind retry.
    (409, error_body("Conflict"), "transient_status", "unknown", False),
    (400, error_body("Unsupported parameter: 'max_tokens'", code="unsupported_parameter"), "bad_request", "rejected", False),
    (422, error_body("Unprocessable"), "bad_request", "rejected", False),
    (401, error_body("Incorrect API key provided", code="invalid_api_key"), "auth", "rejected", False),
    (403, error_body("Country, region, or territory not supported"), "auth", "rejected", False),
    (404, error_body("The model `gpt-nope` does not exist", code="model_not_found"), "unknown_model", "rejected", False),
    (404, error_body("The model `gpt-nope` does not exist or you do not have access to it."), "unknown_model", "rejected", False),
    (404, error_body("Unknown URL"), "not_found", "rejected", False),
    (418, error_body("teapot"), "client_error", "rejected", False),
]


@pytest.mark.parametrize(("status", "body", "kind", "outcome", "retryable"), STATUS_CASES)
def test_status_errors_are_classified(status: int, body: dict, kind: str, outcome: str, retryable: bool) -> None:
    provider, seen = adapter(lambda request: httpx.Response(status, json=body, headers={"x-request-id": "req_9", "retry-after": "3"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1, "the SDK must not retry on its own"
    error = info.value
    assert (error.kind, error.outcome, error.retryable, error.http_status) == (kind, outcome, retryable, status)
    assert error.raw["x_request_id"] == "req_9"
    assert error.raw["retry_after"] == "3"
    assert "test-not-a-key" not in json.dumps(error.raw) + str(error)

# --------------------------------------------------------------------------- redirects (M2)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_reply_is_one_unknown_attempt_and_is_not_followed(status: int) -> None:
    """The request reached a server. Nothing shows it was not processed, so no resend and no follow."""
    target = "http://127.0.0.1:9/elsewhere"
    provider, seen = adapter(lambda request: httpx.Response(status, headers={"location": target, "x-request-id": "req_3xx"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1, "a redirect must be neither followed nor retried"
    error = info.value
    assert (error.kind, error.outcome, error.retryable, error.http_status) == ("redirect", "unknown", False, status)
    assert error.raw["location"] == target
    assert error.raw["x_request_id"] == "req_3xx"
    assert error.raw["http_status"] == status


def test_a_redirect_location_is_clipped_and_a_redirect_without_location_is_still_unknown() -> None:
    provider, seen = adapter(lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1:9/" + "a" * 5000}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(info.value.raw["location"]) <= 200
    provider, seen = adapter(lambda request: httpx.Response(300))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert (info.value.outcome, info.value.retryable, info.value.raw["location"]) == ("unknown", False, None)


def test_a_redirect_goes_to_reconcile_not_to_a_retry() -> None:
    from faar.retry_policy import RetryPolicy

    provider, seen = adapter(lambda request: httpx.Response(307, headers={"location": "http://127.0.0.1:9/x"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert RetryPolicy().decide(info.value, 1).action == "reconcile"
    assert len(seen) == 1


def test_a_client_that_follows_redirects_is_refused_at_construction() -> None:
    client, seen = mock_client(lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1:9/x"}), follow_redirects=True)
    with pytest.raises(ValueError, match="follow redirects"):
        OpenAIChatProvider("gpt-test", {}, client=client)
    assert seen == []


def test_the_sdk_default_http_client_follows_redirects_and_is_refused() -> None:
    client = openai.OpenAI(api_key="test-not-a-key", max_retries=0)
    with pytest.raises(ValueError, match="follow redirects"):
        OpenAIChatProvider("gpt-test", {}, client=client)


def test_a_client_whose_redirect_setting_cannot_be_read_is_refused() -> None:
    class Opaque:
        max_retries = 0
        api_key = "test-not-a-key"

    with pytest.raises(ValueError, match="follow redirects"):
        OpenAIChatProvider("gpt-test", {}, client=Opaque())


def test_a_following_client_would_have_sent_a_second_request() -> None:
    """Why the refusal exists: with follow_redirects=True the same reply costs two requests."""
    client, seen = mock_client(
        lambda request: httpx.Response(200, json=completion_payload())
        if request.url.path.endswith("/elsewhere")
        else httpx.Response(307, headers={"location": "http://127.0.0.1:9/elsewhere"}),
        follow_redirects=True,
    )
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert len(seen) == 2

# --------------------------------------------------------------------------- API key scrub (L4)

ECHOED_KEY = "sk-test-echoed-0123456789"


def test_an_api_key_echoed_by_a_proxy_is_replaced_in_the_message_and_the_raw() -> None:
    header_echo = {"x-request-id": "req_echo", "x-debug": f"Bearer {ECHOED_KEY}", "location": f"http://127.0.0.1:9/?key={ECHOED_KEY}"}
    cases = [
        (401, {"json": error_body(f"Incorrect API key provided: {ECHOED_KEY}.", code="invalid_api_key")}),
        (502, {"json": {"error": {"message": "bad", "type": "server_error", "code": None, "detail": {"echo": [f"Authorization: Bearer {ECHOED_KEY}"]}}}}),
        (502, {"text": f"<html>upstream saw {ECHOED_KEY}</html>"}),
        (500, {"json": error_body("ok", code=f"code-{ECHOED_KEY}", type_=f"type-{ECHOED_KEY}")}),
        (302, {"headers": header_echo}),
    ]
    for status, kwargs in cases:
        provider, seen = adapter(lambda request, s=status, k=kwargs: httpx.Response(s, **k), api_key=ECHOED_KEY)
        with pytest.raises(ProviderError) as info:
            provider.send(make_request())
        error = info.value
        assert len(seen) == 1
        assert ECHOED_KEY not in str(error), (status, kwargs)
        assert ECHOED_KEY not in json.dumps(error.raw), (status, kwargs)
        assert "[redacted-api-key]" in str(error) + json.dumps(error.raw), (status, kwargs)
        assert error.outcome in ("rejected", "unknown") and error.http_status == status


def test_a_key_that_straddles_a_size_limit_is_still_removed() -> None:
    """The scrub runs before the message and body are cut to their limits, so no fragment of the key survives a cut.

    Each case first runs with a different client key, to show that the cut really lands inside the echoed key.
    """
    message_prefix = "Error code: 401 - {'error': {'message': '"  # how the SDK renders the message
    message = "x" * (995 - len(message_prefix)) + ECHOED_KEY
    body = {"error": {"message": "y" * (3995 - len('{"message": "')) + ECHOED_KEY, "type": "server_error", "code": None}}
    cases = [
        (401, error_body(message), lambda error: str(error)),
        (502, body, lambda error: json.dumps(error.raw)),
    ]
    for status, payload, stored in cases:
        control, _ = adapter(lambda request, s=status, p=payload: httpx.Response(s, json=p), api_key="a-different-key-1234")
        with pytest.raises(ProviderError) as info:
            control.send(make_request())
        assert ECHOED_KEY[:4] in stored(info.value), "the cut must land inside the key for this test to mean anything"
        provider, _ = adapter(lambda request, s=status, p=payload: httpx.Response(s, json=p), api_key=ECHOED_KEY)
        with pytest.raises(ProviderError) as info:
            provider.send(make_request())
        assert ECHOED_KEY[:4] not in stored(info.value).replace("[redacted-api-key]", ""), status


def test_an_api_key_in_a_transport_error_or_a_malformed_reply_is_replaced() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"proxy rejected {ECHOED_KEY}", request=request)

    provider, _ = adapter(refuse, api_key=ECHOED_KEY)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert ECHOED_KEY not in str(info.value) + json.dumps(info.value.raw)
    provider, _ = adapter(lambda request: httpx.Response(200, text=f"<html>{ECHOED_KEY}</html>", headers={"content-type": "text/html"}), api_key=ECHOED_KEY)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert ECHOED_KEY not in str(info.value) + json.dumps(info.value.raw)
    body = completion_payload(choices=[], id=f"id-{ECHOED_KEY}")
    provider, _ = adapter(lambda request: httpx.Response(200, json=body), api_key=ECHOED_KEY)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert ECHOED_KEY not in str(info.value) + json.dumps(info.value.raw)


def test_a_key_shorter_than_eight_characters_is_not_scrubbed() -> None:
    """A short value would also match ordinary text, so the scrub leaves it alone."""
    provider, _ = adapter(lambda request: httpx.Response(401, json=error_body("bad key abc1234")), api_key="abc1234")
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert "abc1234" in str(info.value) and "[redacted-api-key]" not in str(info.value)


def test_the_provider_never_exposes_the_key() -> None:
    provider, _ = adapter(lambda request: httpx.Response(200, json=completion_payload()), api_key=ECHOED_KEY)
    assert ECHOED_KEY not in json.dumps(provider.identity()) and ECHOED_KEY not in repr(provider)


def test_only_documented_not_processed_statuses_are_retryable() -> None:
    """Every status that the retry policy would resend by itself is listed here on purpose."""
    retryable = sorted({(status, body["error"]["code"]) for status, body, _, _, can_retry in STATUS_CASES if can_retry})
    assert retryable == [(408, None), (503, "server_is_overloaded")]


def test_a_502_is_one_unknown_attempt_and_the_retry_policy_reconciles_it() -> None:
    from faar.retry_policy import RetryPolicy

    provider, seen = adapter(lambda request: httpx.Response(502, text="<html>Bad gateway</html>", headers={"x-request-id": "req_502"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1
    decision = RetryPolicy().decide(info.value, 1)
    assert decision.action == "reconcile" and decision.delay_seconds == 0.0


@pytest.mark.parametrize("status", [500, 502, 503, 409, 429])
def test_x_should_retry_true_does_not_make_an_unknown_outcome_retryable(status: int) -> None:
    """The SDK resends on `x-should-retry: true`. A header is a recommendation, not proof that nothing was processed."""
    provider, _ = adapter(lambda request: httpx.Response(status, json=error_body("later"), headers={"x-should-retry": "true"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert (info.value.outcome, info.value.retryable) == ("unknown", False)
    assert info.value.raw["x_should_retry"] == "true"


def test_an_unknown_status_outcome_keeps_the_diagnostics_and_no_secret() -> None:
    body = error_body("upstream said no", code="upstream_error", type_="server_error")
    headers = {"x-request-id": "req_diag", "retry-after": "7", "x-should-retry": "true", "openai-organization": "org-x"}
    provider, _ = adapter(lambda request: httpx.Response(502, json=body, headers=headers))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    raw = info.value.raw
    assert raw["http_status"] == 502
    assert raw["x_request_id"] == "req_diag"
    assert raw["retry_after"] == "7"
    assert raw["x_should_retry"] == "true"
    assert raw["error_code"] == "upstream_error"
    assert raw["error_type"] == "server_error"
    assert raw["body"] == body["error"]  # the SDK unwraps the "error" object
    text = json.dumps(raw).lower()
    assert "test-not-a-key" not in text and "authorization" not in text and "bearer" not in text


def test_a_huge_error_body_is_bounded_and_keeps_the_ids() -> None:
    huge = {"error": {"message": "x" * 200_000, "type": "server_error", "code": None}}
    provider, _ = adapter(lambda request: httpx.Response(502, json=huge, headers={"x-request-id": "req_big"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    raw = info.value.raw
    assert raw["x_request_id"] == "req_big" and raw["http_status"] == 502
    assert len(json.dumps(raw)) < 12_000
    assert raw["body"]["truncated"] is True


def test_a_huge_non_json_error_body_is_bounded() -> None:
    provider, _ = adapter(lambda request: httpx.Response(502, text="<html>" + "y" * 200_000 + "</html>"))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(json.dumps(info.value.raw)) < 12_000
    assert info.value.raw["http_status"] == 502


def test_x_should_retry_false_downgrades_a_retryable_status() -> None:
    body = error_body("overloaded", code="server_is_overloaded", type_="service_unavailable_error")
    provider, _ = adapter(lambda request: httpx.Response(503, json=body, headers={"x-should-retry": "false"}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert info.value.retryable is False and info.value.outcome == "rejected"


def test_504_with_a_non_json_body_is_still_a_gateway_timeout_with_unknown_outcome() -> None:
    provider, seen = adapter(lambda request: httpx.Response(504, text="<html>gateway timeout</html>"))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1
    error = info.value
    assert (error.kind, error.outcome, error.retryable, error.http_status) == ("gateway_timeout", "unknown", False, 504)


def test_connect_error_from_the_transport_is_not_sent_and_retryable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name resolution failed", request=request)

    provider, seen = adapter(refuse)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1
    error = info.value
    assert (error.kind, error.outcome, error.retryable, error.http_status) == ("connect", "not_sent", True, None)
    assert error.raw["error_class"] == "APIConnectionError"
    assert error.raw["cause_class"] == "ConnectError"


def test_read_timeout_from_the_transport_is_unknown_and_not_retryable() -> None:
    def hang(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no bytes for 30 s", request=request)

    provider, seen = adapter(hang)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert len(seen) == 1
    error = info.value
    assert (error.kind, error.outcome, error.retryable) == ("timeout", "unknown", False)
    assert error.raw["error_class"] == "APITimeoutError"


@pytest.mark.parametrize(
    ("exc_factory", "kind", "outcome", "retryable"),
    [
        (lambda r: httpx.ConnectTimeout("connect timed out", request=r), "connect_timeout", "not_sent", True),
        (lambda r: httpx.PoolTimeout("pool exhausted", request=r), "pool_timeout", "not_sent", True),
        (lambda r: httpx.ProxyError("proxy refused CONNECT", request=r), "connect", "not_sent", True),
        (lambda r: httpx.UnsupportedProtocol("ftp://", request=r), "client_config", "not_sent", False),
        (lambda r: httpx.WriteTimeout("write timed out", request=r), "timeout", "unknown", False),
        (lambda r: httpx.ReadError("connection reset", request=r), "connection_lost", "unknown", False),
        (lambda r: httpx.WriteError("broken pipe", request=r), "connection_lost", "unknown", False),
        (lambda r: httpx.RemoteProtocolError("Server disconnected without sending a response.", request=r), "connection_lost", "unknown", False),
        (lambda r: httpx.CloseError("close failed", request=r), "connection_lost", "unknown", False),
    ],
)
def test_transport_failures_are_placed_by_where_they_can_happen(exc_factory: Callable, kind: str, outcome: str, retryable: bool) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exc_factory(request)

    provider, _ = adapter(fail)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert (info.value.kind, info.value.outcome, info.value.retryable) == (kind, outcome, retryable)


def test_unplaceable_exception_is_unknown() -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("something else")

    provider, _ = adapter(explode)
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert (info.value.kind, info.value.outcome, info.value.retryable) == ("connection_lost", "unknown", False)
    # An exception that never came through the SDK wrapper:
    direct = classify_openai_error(ValueError("bad"))
    assert (direct.kind, direct.outcome, direct.retryable) == ("unexpected", "unknown", False)


def test_classify_passes_provider_errors_through_and_handles_raw_httpx_errors() -> None:
    original = ProviderError("x", kind="k", outcome="rejected", retryable=False)
    assert classify_openai_error(original) is original
    assert classify_openai_error(httpx.ConnectError("no route")).outcome == "not_sent"
    assert classify_openai_error(httpx.ReadTimeout("slow")).outcome == "unknown"


def test_client_with_sdk_retries_is_refused() -> None:
    client = openai.OpenAI(
        api_key="test-not-a-key",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=completion_payload()))),
        max_retries=2,
    )
    with pytest.raises(ValueError, match="max_retries=0"):
        OpenAIChatProvider("gpt-test", {}, client=client)


UNALLOWED_PARAMS = [
    "service_tier",
    "reasoning_effort",
    "modalities",
    "logprobs",
    "web_search_options",
    "prediction",
    # keys the adapter sets itself
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "timeout",
    "stream",
    "tools",
    "n",
    "store",
    "metadata",
    "extra_headers",
    "extra_body",
    # anything else
    "not_a_real_parameter",
    "Temperature",
]


@pytest.mark.parametrize("key", UNALLOWED_PARAMS)
def test_params_outside_the_allowlist_are_refused_before_any_dispatch(key: str) -> None:
    client, seen = mock_client(lambda request: httpx.Response(200, json=completion_payload()))
    with pytest.raises(ValueError, match="not allowed"):
        OpenAIChatProvider("gpt-test", {key: 1}, client=client)
    assert seen == []


@pytest.mark.parametrize("key", ALLOWED_OPENAI_PARAMS)
def test_every_allowed_param_is_accepted_and_sent(key: str) -> None:
    value: Any = ["\n"] if key == "stop" else 0
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()), {key: value})
    provider.send(make_request(params={key: value}))
    assert json.loads(seen[0].content)[key] == value


def test_one_unallowed_key_among_allowed_ones_refuses_the_whole_set() -> None:
    client, _ = mock_client(lambda request: httpx.Response(200, json=completion_payload()))
    with pytest.raises(ValueError, match="service_tier"):
        OpenAIChatProvider("gpt-test", {"temperature": 0, "service_tier": "flex"}, client=client)


@pytest.mark.parametrize("key", ["service_tier", "reasoning_effort", "modalities", "logprobs", "web_search_options", "prediction"])
def test_request_params_outside_the_allowlist_are_refused_and_nothing_is_sent(key: str) -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()), {"temperature": 0})
    with pytest.raises(ValueError, match="not allowed"):
        provider.send(make_request(params={"temperature": 0, key: "x"}))
    with pytest.raises(ValueError, match="not allowed"):
        provider.send(make_request(params={key: "x"}))
    assert seen == []


def test_request_params_must_match_the_configured_params_and_nothing_is_sent_otherwise() -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()), {"temperature": 0})
    with pytest.raises(ValueError, match="differ"):
        provider.send(make_request(params={"temperature": 1}))
    provider.send(make_request(params={"temperature": 0}))
    with pytest.raises(ValueError):
        provider.send(make_request(max_output_tokens=0))
    assert len(seen) == 1


def test_identity_reports_the_configuration_without_touching_credentials() -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()), {"temperature": 0})
    identity = provider.identity()
    assert identity["provider"] == "openai" and identity["simulated"] is False
    assert identity["requested_model"] == "gpt-test"
    assert identity["endpoint"] == "https://api.openai.com/v1/"
    assert identity["params"] == {"temperature": 0}
    assert identity["storage"] == STORAGE_DISABLED and identity["store"] is False
    assert identity["service_tier"] == SERVICE_TIER_STANDARD == "default"
    assert "tag_attempts" not in identity
    assert identity["sdk"] == {"name": "openai", "version": openai.__version__}
    assert identity["model_calls"] is True and identity["engineering_only"] is False
    assert "test-not-a-key" not in json.dumps(identity)
    assert seen == []  # identity sends nothing


def test_one_attempt_is_one_http_request() -> None:
    provider, seen = adapter(lambda request: httpx.Response(429, json=error_body("slow down")))
    for _ in range(3):
        with pytest.raises(ProviderError):
            provider.send(make_request())
    assert len(seen) == 3


def test_importing_the_module_loads_no_sdk_and_builds_no_client() -> None:
    code = (
        "import sys, faar.answer_providers;"
        "bad = [m for m in ('openai', 'httpx', 'requests', 'anthropic') if m in sys.modules];"
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_module_source_never_reads_credentials() -> None:
    source = Path(ap.__file__).read_text(encoding="utf-8")
    assert "environ" not in source and "getenv" not in source and "OPENAI_API_KEY" not in source.split('"""', 2)[2]


@pytest.mark.parametrize("status", [522, 524])
def test_cloudflare_origin_timeouts_are_unknown_outcomes_like_504(status: int) -> None:
    """A 522 or 524 comes from a gateway in front of the provider, which may have finished and billed the request."""
    provider, seen = adapter(lambda request: httpx.Response(status, text="<html>origin timeout</html>"))
    with pytest.raises(ProviderError) as caught:
        provider.send(make_request())
    error = caught.value
    assert seen, "the mock transport saw the request"
    assert (error.kind, error.outcome, error.retryable, error.http_status) == ("gateway_timeout", "unknown", False, status)


# --------------------------------------------------------------------------- storage policy (O14)
#
# Ways the storage policy could fail, written before the code.
#   S1. The default request leaves `store` out, so the account default decides (OpenAI documents Chat
#       Completions as stored by default for new accounts) and the prompt is stored without anyone choosing it.
#   S2. A retried attempt loses `store: false`, because the flag was set on one code path only.
#   S3. `tag_attempts` and `storage` disagree and the request follows one while the identity reports the other.
#   S4. An unknown storage value is accepted and silently treated as one of the two policies.
#   S5. Turning provider storage off removes a local diagnostic (response id, x-request-id, attempt id).
#   S6. `store` gets in through `params`, past the allowlist.


def body_of(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_the_default_storage_policy_sends_store_false_and_no_attempt_metadata() -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()))
    provider.send(make_request("abc", 1))
    body = body_of(seen[0])
    assert body["store"] is False
    assert "metadata" not in body and "x-client-request-id" not in seen[0].headers
    identity = provider.identity()
    assert identity["storage"] == STORAGE_DISABLED and identity["store"] is False


def test_a_retried_attempt_sends_store_false_both_times() -> None:
    """S2: a 408 is retryable; the second request is a new attempt and must carry the same policy."""
    replies = iter([httpx.Response(408, json=error_body("timeout", type_="server_error")), httpx.Response(200, json=completion_payload())])
    provider, seen = adapter(lambda request: next(replies))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request("abc", 1))
    assert info.value.retryable is True
    provider.send(make_request("abc", 2))
    assert [body_of(request)["store"] for request in seen] == [False, False]
    assert [body_of(request)["service_tier"] for request in seen] == ["default", "default"]


def test_attempt_lookup_storage_sends_store_true_metadata_and_the_client_request_id() -> None:
    provider, seen = adapter(
        lambda request: httpx.Response(200, json=completion_payload()), storage=STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP
    )
    provider.send(make_request("abc", 3))
    body = body_of(seen[0])
    assert body["store"] is True
    assert body["metadata"] == {"faar_attempt_id": "abc-a3", "faar_request_id": "abc"}
    assert seen[0].headers["x-client-request-id"] == "abc-a3"
    assert body["service_tier"] == "default"


@pytest.mark.parametrize("storage", STORAGE_POLICIES)
def test_local_diagnostics_do_not_depend_on_provider_storage(storage: str) -> None:
    """S5: the response id and the x-request-id header stay available under both policies."""
    provider, _ = adapter(
        lambda request: httpx.Response(200, json=completion_payload(), headers={"x-request-id": "req_ok"}), storage=storage
    )
    response = provider.send(make_request("abc", 1))
    assert response.response_id == "chatcmpl-test-1"
    failing, _ = adapter(
        lambda request: httpx.Response(502, json=error_body("bad gateway", type_="server_error"), headers={"x-request-id": "req_bad"}),
        storage=storage,
    )
    with pytest.raises(ProviderError) as info:
        failing.send(make_request("abc", 1))
    assert info.value.raw["x_request_id"] == "req_bad"


@pytest.mark.parametrize("storage", ["enabled", "STORAGE_DISABLED", "", None, True, 1, ["disabled"]])
def test_an_unknown_storage_value_is_refused_at_construction(storage: Any) -> None:
    client, seen = mock_client(lambda request: httpx.Response(200, json=completion_payload()))
    with pytest.raises(ValueError, match="storage"):
        OpenAIChatProvider("gpt-test", {}, client=client, storage=storage)
    assert seen == []


def test_tag_attempts_is_gone_so_it_cannot_contradict_the_storage_policy() -> None:
    """S3: one field decides. The old flag is not accepted as an alias, so it cannot disagree with `storage`."""
    client, _ = mock_client(lambda request: httpx.Response(200, json=completion_payload()))
    with pytest.raises(TypeError, match="tag_attempts"):
        OpenAIChatProvider("gpt-test", {}, client=client, tag_attempts=True)  # type: ignore[call-arg]


def test_store_stays_refused_in_params_under_both_policies() -> None:
    """S6"""
    for storage in STORAGE_POLICIES:
        client, _ = mock_client(lambda request: httpx.Response(200, json=completion_payload()))
        with pytest.raises(ValueError, match="not allowed"):
            OpenAIChatProvider("gpt-test", {"store": False}, client=client, storage=storage)
        provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()), storage=storage)
        with pytest.raises(ValueError, match="not allowed"):
            provider.send(make_request(params={"store": True}))
        assert seen == []


# --------------------------------------------------------------------------- service tier (O15)
#
#   T1. The request leaves `service_tier` out, so a project-level setting (for example Fast mode) can move the
#       price outside the price table.
#   T2. The adapter rewrites the returned tier (for example lower-cases it, or maps an unknown one to "default"),
#       and the safety check downstream never sees the violation.
#   T3. The SDK types the response field as Literal["scale", "default"] and could reject or change "priority".
#   T4. The malformed-200 error keeps usage, id and model but drops the tier.


def test_every_request_asks_for_the_standard_tier() -> None:
    provider, seen = adapter(lambda request: httpx.Response(200, json=completion_payload()))
    provider.send(make_request())
    assert body_of(seen[0])["service_tier"] == SERVICE_TIER_STANDARD == "default"
    assert provider.identity()["service_tier"] == "default"


def test_the_sdk_types_the_request_tier_as_auto_or_default() -> None:
    """The value is documented for the Chat Completions API (openai-python 1.68.2 types it auto|default)."""
    from openai.types.chat import completion_create_params

    hint = str(completion_create_params.CompletionCreateParamsBase.__annotations__["service_tier"])
    assert "default" in hint and "auto" in hint


TIER_CASES = ["default", "priority", "flex", "scale", "fast", "DEFAULT"]


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
@pytest.mark.parametrize("returned", TIER_CASES)
def test_a_returned_tier_is_recorded_exactly_as_the_provider_sent_it(returned: str) -> None:
    provider, _ = adapter(lambda request: httpx.Response(200, json=completion_payload(service_tier=returned)))
    response = provider.send(make_request())
    assert response.returned_service_tier == returned  # never normalised
    assert response.raw["service_tier"] == returned  # the payload is unchanged
    assert response.text == "Paris"  # the SDK's narrower response type did not reject the reply


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
@pytest.mark.parametrize("label", ["missing", "null", "integer", "list", "object", "boolean"])
def test_a_missing_or_non_string_tier_is_none(label: str) -> None:
    payload = completion_payload()
    if label != "missing":
        payload["service_tier"] = {"null": None, "integer": 5, "list": ["default"], "object": {"tier": "default"}, "boolean": True}[label]
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    response = provider.send(make_request())
    assert response.returned_service_tier is None
    assert response.text == "Paris"


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
@pytest.mark.parametrize("returned", ["default", "priority", None])
def test_a_malformed_200_keeps_the_tier_beside_usage_id_and_model(returned: str | None) -> None:
    """T4"""
    payload = completion_payload(choices=[], service_tier=returned)
    provider, _ = adapter(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    raw = info.value.raw
    assert raw["service_tier"] == returned
    assert raw["payload"]["service_tier"] == returned
    assert (raw["id"], raw["model"]) == ("chatcmpl-test-1", "gpt-test-2026-01-01")


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_a_malformed_200_without_a_tier_has_a_none_tier_in_raw() -> None:
    provider, _ = adapter(lambda request: httpx.Response(200, json={"foo": 1}))
    with pytest.raises(ProviderError) as info:
        provider.send(make_request())
    assert "service_tier" in info.value.raw and info.value.raw["service_tier"] is None


# --------------------------------------------------------------------------- direct transport (O16)
#
#   R1. An injected client reads HTTPS_PROXY, HTTP_PROXY, ALL_PROXY or the operating system's proxy settings
#       (`trust_env=True`), so the prompt and the key go through a proxy the run never recorded.
#   R2. `trust_env` is truthy but not `False` (0, None), or cannot be read, and the check lets it through.
#   R3. A client with an explicit proxy or mount is accepted because its `trust_env` is False.
#   R4. Only the constructor path is checked; the SDK's own default client (trust_env=True) slips through.


def test_a_client_that_reads_the_proxy_environment_is_refused_at_construction() -> None:
    client, seen = mock_client(lambda request: httpx.Response(200, json=completion_payload()), trust_env=True)
    with pytest.raises(ValueError, match="trust_env"):
        OpenAIChatProvider("gpt-test", {}, client=client)
    assert seen == []


def test_the_sdk_default_client_is_refused_for_both_settings() -> None:
    client = openai.OpenAI(api_key="test-not-a-key", max_retries=0)
    with pytest.raises(ValueError) as info:
        OpenAIChatProvider("gpt-test", {}, client=client)
    assert "follow redirects" in str(info.value) and "trust_env" in str(info.value)


class _StubInner:
    def __init__(self, **attributes: Any) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class _StubClient:
    max_retries = 0
    api_key = "test-not-a-key"

    def __init__(self, inner: Any) -> None:
        self._client = inner


@pytest.mark.parametrize(
    "attributes",
    [
        {"follow_redirects": False, "trust_env": 0},
        {"follow_redirects": False, "trust_env": None},
        {"follow_redirects": False, "trust_env": "False"},
        {"follow_redirects": False},  # trust_env cannot be read
        {"trust_env": False},  # follow_redirects cannot be read
        {"follow_redirects": 0, "trust_env": False},
        {},
    ],
)
def test_a_setting_that_is_not_exactly_false_or_cannot_be_read_is_refused(attributes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OpenAIChatProvider("gpt-test", {}, client=_StubClient(_StubInner(**attributes)))


def test_a_client_without_an_http_client_is_refused() -> None:
    with pytest.raises(ValueError):
        OpenAIChatProvider("gpt-test", {}, client=_StubClient(None))


def test_a_client_with_an_explicit_proxy_is_refused_even_when_trust_env_is_false() -> None:
    """R3: `proxy=` adds a mount that routes every request through it, whatever trust_env says."""
    client = openai.OpenAI(
        api_key="test-not-a-key",
        http_client=httpx.Client(follow_redirects=False, trust_env=False, proxy="http://127.0.0.1:9"),
        max_retries=0,
    )
    with pytest.raises(ValueError, match="proxy"):
        OpenAIChatProvider("gpt-test", {}, client=client)


def test_a_client_whose_own_transport_goes_through_a_proxy_is_refused() -> None:
    import httpx

    from faar.answer_providers import http_client_problems

    proxied = httpx.Client(trust_env=False, follow_redirects=False, transport=httpx.HTTPTransport(proxy="http://127.0.0.1:9"))
    direct = httpx.Client(trust_env=False, follow_redirects=False, transport=httpx.HTTPTransport())
    assert http_client_problems(proxied), "a proxy inside the client's own transport must be refused"
    assert http_client_problems(direct) == []
