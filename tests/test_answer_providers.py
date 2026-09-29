"""Tests for faar.answer_providers.

The failure list these tests cover is in the docstring of `faar/answer_providers.py`
(items F1-F10 for the fake provider, O1-O12 for the OpenAI adapter). No test sends a
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
from faar.live_contract import ALLOWED_OPENAI_PARAMS, ProviderError, ProviderRequest, ProviderUsage

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
        (FakeStep("retryable_error"), "rate_limit", "rejected", True, 429),
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


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[openai.OpenAI, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = openai.OpenAI(
        api_key="test-not-a-key",
        http_client=httpx.Client(transport=httpx.MockTransport(recording)),
        max_retries=0,
    )
    return client, seen


def adapter(handler: Callable[[httpx.Request], httpx.Response], params: dict | None = None, **kwargs: Any) -> tuple[OpenAIChatProvider, list[httpx.Request]]:
    client, seen = mock_client(handler)
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
    }
    assert "x-client-request-id" not in request.headers
    assert request.extensions["timeout"] == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}
    assert request.headers["authorization"] == "Bearer test-not-a-key"


def test_max_completion_tokens_option_and_tag_attempts() -> None:
    provider, seen = adapter(
        lambda request: httpx.Response(200, json=completion_payload()),
        token_limit_param="max_completion_tokens",
        tag_attempts=True,
    )
    provider.send(make_request("abc", 2, max_output_tokens=10))
    body = json.loads(seen[0].content)
    assert body["max_completion_tokens"] == 10 and "max_tokens" not in body
    assert body["store"] is True
    assert body["metadata"] == {"faar_attempt_id": "abc-a2", "faar_request_id": "abc"}
    assert seen[0].headers["x-client-request-id"] == "abc-a2"
    identity = provider.identity()
    assert identity["token_limit_param"] == "max_completion_tokens" and identity["tag_attempts"] is True


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


@pytest.mark.parametrize(
    ("status", "body", "kind", "outcome", "retryable"),
    [
        (429, error_body("Rate limit reached for requests", code="rate_limit_exceeded", type_="requests"), "rate_limit", "rejected", True),
        (429, error_body("You exceeded your current quota", code="insufficient_quota", type_="insufficient_quota"), "quota", "rejected", False),
        (429, error_body("Project reached its enforced monthly spend limit"), "quota", "rejected", False),
        (500, error_body("The server had an error", type_="server_error"), "server_error", "rejected", True),
        (502, error_body("Bad gateway", type_="server_error"), "server_error", "rejected", True),
        (503, error_body("Model overloaded", type_="server_error"), "server_error", "rejected", True),
        (504, error_body("Gateway timeout", type_="server_error"), "gateway_timeout", "unknown", False),
        (408, error_body("Request timed out"), "transient_status", "rejected", True),
        (409, error_body("Conflict"), "transient_status", "rejected", True),
        (400, error_body("Unsupported parameter: 'max_tokens'", code="unsupported_parameter"), "bad_request", "rejected", False),
        (422, error_body("Unprocessable"), "bad_request", "rejected", False),
        (401, error_body("Incorrect API key provided", code="invalid_api_key"), "auth", "rejected", False),
        (403, error_body("Country, region, or territory not supported"), "auth", "rejected", False),
        (404, error_body("The model `gpt-nope` does not exist", code="model_not_found"), "unknown_model", "rejected", False),
        (404, error_body("The model `gpt-nope` does not exist or you do not have access to it."), "unknown_model", "rejected", False),
        (404, error_body("Unknown URL"), "not_found", "rejected", False),
        (418, error_body("teapot"), "client_error", "rejected", False),
    ],
)
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


def test_x_should_retry_false_downgrades_a_retryable_status() -> None:
    provider, _ = adapter(lambda request: httpx.Response(500, json=error_body("boom"), headers={"x-should-retry": "false"}))
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
