"""Driver-level tests for uncertain HTTP outcomes of the OpenAI adapter.

`faar.live_runner.execute_run` drives a real `OpenAIChatProvider` over an `httpx.MockTransport`. The
mock records every request, so each test counts what a provider would have received. Nothing here
sends a request, and the API key is the literal string `test-not-a-key`.

Ways the pair could fail, written before the tests:

  H1. An HTTP 502 is resent by the retry policy. A gateway that returns 502 may have passed the
      request on, and OpenAI documents no idempotency key for Chat Completions, so the resend can bill twice.
  H2. An ordinary resume sends a request whose outcome is unknown.
  H3. The `outcome_unknown` event drops the status, the request id, the retry headers or the error
      body, or stores the Authorization header or the key.
  H4. An unknown attempt's reservation disappears after a restart, after `reconcile allow_new_attempt`
      or after `reconcile mark_failed`.
  H5. `allow_new_attempt` permits more than one further dispatch, or none.
  H6. A failure that provably never left the machine (`httpx.ConnectError`) stops being retried.
  H7. A read timeout or a 200 reply without a chat completion becomes retryable.
  H8. A documented refusal to process (503 `server_is_overloaded`) stops being retried, or an
      undocumented one (429, 500, 409) starts being retried.
  H9. A gateway outage of many 502 replies sends more requests than the circuit breaker allows.
  H10. The client that `build_live_provider` builds follows a 307 or 308 reply and posts the prompt to the
      Location URL, outside the driver's accounting. Each such reply must reach the transport once, and the
      attempt must be recorded as unknown.

The status table and its sources are in the comment above `_classify_status` in
`faar/answer_providers.py`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from test_live_runner import (
    WARRANTY_QUESTION,
    Project,
    by_question,
    default_project,
    events_of,
    options,
    requests_of,
    summary_of,
)

from faar import live_runner as lr
from faar.answer_providers import OpenAIChatProvider
from faar.request_budget import SafetyLedger

API_KEY = "test-not-a-key"
DESCRIPTOR = {"adapter": "faar.answer_providers.OpenAIChatProvider", "transport": "httpx.MockTransport"}

Step = Callable[[httpx.Request], httpx.Response]


def completion(text: str = "twelve months") -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": lr.FAKE_CONFIG.model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text, "refusal": None}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
    }


def ok() -> Step:
    return lambda request: httpx.Response(200, json=completion(), headers={"x-request-id": "req_ok"})


def status(code: int, *, body: Any = None, headers: dict[str, str] | None = None, text: str | None = None) -> Step:
    def respond(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(code, text=text, headers=headers)
        return httpx.Response(code, json=body if body is not None else {"error": {"message": f"HTTP {code}", "type": "server_error", "code": None}}, headers=headers)

    return respond


def raises(factory: Callable[[httpx.Request], Exception]) -> Step:
    def respond(request: httpx.Request) -> httpx.Response:
        raise factory(request)

    return respond


BAD_GATEWAY_HEADERS = {"x-request-id": "req_502_diag", "retry-after": "5", "x-should-retry": "true"}


def bad_gateway() -> Step:
    body = {"error": {"message": "upstream returned an invalid response", "type": "server_error", "code": "bad_gateway"}}
    return status(502, body=body, headers=BAD_GATEWAY_HEADERS)


class Wire:
    """A mock transport with one scripted step per attempt of each question. It counts every request it receives."""

    def __init__(self, script: dict[str, list[Step]] | None = None) -> None:
        self.script = {key: list(steps) for key, steps in (script or {}).items()}
        self.seen: list[str] = []
        self.urls: list[str] = []
        self.authorization: list[str | None] = []
        self.providers = 0

    @staticmethod
    def question_of(request: httpx.Request) -> str:
        user = json.loads(request.content)["messages"][1]["content"]
        for question_id, (question, marker) in {
            "q1": (WARRANTY_QUESTION, "Alpha"),
            "q2": (WARRANTY_QUESTION, "Beta"),
            "q3": ("How much does the buyer pay?", "Alpha"),
            "q4": ("How many years does the lease run?", "Gamma"),
            "q6": ("How much does the buyer pay?", "Beta"),
        }.items():
            if question in user and marker in user:
                return question_id
        raise AssertionError(f"request matches no question: {user[:200]!r}")

    def handler(self, request: httpx.Request) -> httpx.Response:
        question_id = self.question_of(request)
        index = self.seen.count(question_id)
        self.seen.append(question_id)
        self.urls.append(str(request.url))
        self.authorization.append(request.headers.get("authorization"))
        steps = self.script.get(question_id, [])
        step = steps[index] if index < len(steps) else ok()
        return step(request)

    def count(self, question_id: str) -> int:
        return self.seen.count(question_id)

    def factory(self, context: lr.ProviderContext) -> OpenAIChatProvider:
        self.providers += 1
        client = openai.OpenAI(
            api_key=API_KEY,
            base_url="http://127.0.0.1:9/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )
        return OpenAIChatProvider(lr.FAKE_CONFIG.model, dict(lr.FAKE_CONFIG.params), client=client)

    def live_factory(self, context: lr.ProviderContext) -> OpenAIChatProvider:
        """The provider the live CLI builds, on this mock transport. Only the transport is a test seam."""
        self.providers += 1
        config = dataclasses.replace(lr.FAKE_CONFIG, endpoint="http://127.0.0.1:9/v1")
        client = httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=False)
        return lr.build_live_provider(config, {"OPENAI_API_KEY": API_KEY}, http_client=client)


class Session:
    """One run directory driven over a shared `Wire`, with the delays the driver asked to sleep."""

    def __init__(self, project: Project, wire: Wire, name: str = "run-a") -> None:
        self.project, self.wire, self.name = project, wire, name
        self.sleeps: list[float] = []

    @property
    def run_dir(self) -> Path:
        return self.project.run_dir(self.name)

    def run(self, *, live_build: bool = False) -> lr.LiveResult:
        return lr.execute_run(
            options(self.project, self.name),
            provider_factory=self.wire.live_factory if live_build else self.wire.factory,
            descriptor=DESCRIPTOR,
            sleep=self.sleeps.append,
            environ={},
        )

    def reconcile(self, attempt_id: str, resolution: str) -> lr.LiveResult:
        return lr.reconcile_attempt(run_dir=self.run_dir, attempt_id=attempt_id, resolution=resolution, note="checked the usage export")

    def events(self, name: str, question_id: str = "q1") -> list[dict[str, Any]]:
        request_id = next(r["request_id"] for r in requests_of(self.run_dir) if r["question_id"] == question_id)
        return [e for e in events_of(self.run_dir) if e["event"] == name and e.get("request_id") == request_id]

    def upper_bound(self, question_id: str = "q1") -> float:
        return next(r["cost_upper_bound"] for r in requests_of(self.run_dir) if r["question_id"] == question_id)

    def ledger(self) -> SafetyLedger:
        """The ledger rebuilt from the event log on disk, as a restarted process would see it."""
        return SafetyLedger.from_events(list(lr.iter_committed_events(self.run_dir)), lr.FAKE_CONFIG.prices)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return default_project(tmp_path / "project")


def session(project: Project, script: dict[str, list[Step]] | None = None) -> Session:
    return Session(project, Wire(script))


# ---------------------------------------------------------------------------
# HTTP 502
# ---------------------------------------------------------------------------


def test_a_502_is_dispatched_once_and_is_not_repeated_automatically(project: Project) -> None:
    """H1: the retry policy must not resend a 502. Other questions continue."""
    s = session(project, {"q1": [bad_gateway(), ok()]})
    result = s.run()
    assert s.wire.count("q1") == 1, "a second q1 request would be an automatic resend"
    assert s.wire.seen == ["q1", "q2", "q3", "q4", "q6"]
    assert s.sleeps == [], "no retry delay was requested"
    assert len(s.events("dispatch_started")) == 1
    assert len(s.events("outcome_unknown")) == 1
    assert s.events("attempt_failed") == []
    assert lr.run_status(s.run_dir)["requests"]["unknown"] == 1
    assert result.summary["run_state"] == "needs_reconciliation"


def test_an_ordinary_resume_sends_nothing_for_a_502(project: Project) -> None:
    """H2."""
    s = session(project, {"q1": [bad_gateway(), ok()]})
    s.run()
    before = list(s.wire.seen)
    resumed = s.run()
    assert s.wire.seen == before, "a resume must not dispatch the unknown attempt again"
    assert resumed.summary["run_state"] == "needs_reconciliation"
    assert s.wire.count("q1") == 1


def test_the_unknown_event_keeps_the_diagnostics_and_no_secret(project: Project) -> None:
    """H3."""
    s = session(project, {"q1": [bad_gateway()]})
    s.run()
    (event,) = s.events("outcome_unknown")
    raw = event["provider_raw"]
    assert event["kind"] == "server_error"
    assert raw["http_status"] == 502
    assert raw["x_request_id"] == "req_502_diag"
    assert raw["retry_after"] == "5"
    assert raw["x_should_retry"] == "true"
    assert raw["error_code"] == "bad_gateway"
    assert raw["error_type"] == "server_error"
    assert raw["body"]["message"] == "upstream returned an invalid response"
    log = (s.run_dir / lr.ATTEMPTS_NAME).read_text(encoding="utf-8").lower()
    assert API_KEY not in log and "authorization" not in log and "bearer" not in log
    assert s.wire.authorization[0] == f"Bearer {API_KEY}", "the mock did see the header, so the absence above means something"


def test_a_502_with_a_html_body_keeps_a_bounded_body(project: Project) -> None:
    s = session(project, {"q1": [status(502, text="<html>" + "z" * 100_000 + "</html>", headers={"x-request-id": "req_html"})]})
    s.run()
    (event,) = s.events("outcome_unknown")
    raw = event["provider_raw"]
    assert raw["x_request_id"] == "req_html" and raw["http_status"] == 502
    assert raw["body"]["truncated"] is True
    assert len(json.dumps(raw)) < 12_000


def test_the_reservation_of_a_502_survives_a_restart(project: Project) -> None:
    """H4: rebuilt from the event log on disk, the unknown attempt still holds its upper bound."""
    s = session(project, {"q1": [bad_gateway()]})
    s.run()
    reserved = s.ledger().reserved
    assert reserved == pytest.approx(s.upper_bound())
    (account,) = [a for a in s.ledger().attempts if a.attempt_id.endswith("-a1") and a.reserved_micro > 0]
    assert account.state == "outcome_unknown"
    s.run()  # a restart
    assert s.ledger().reserved == pytest.approx(reserved)
    assert summary_of(s.run_dir)["cost"]["reserved"] == pytest.approx(reserved)


def test_the_reservation_stays_counted_after_allow_new_attempt(project: Project) -> None:
    """H4."""
    s = session(project, {"q1": [bad_gateway(), ok()]})
    s.run()
    (event,) = s.events("outcome_unknown")
    reserved_before = s.ledger().reserved
    s.reconcile(event["attempt_id"], lr.RESOLUTION_ALLOW)
    assert s.ledger().reserved == pytest.approx(reserved_before)
    s.run()
    # The second attempt was measured, and the first still holds its upper bound.
    assert s.ledger().reserved == pytest.approx(s.upper_bound())
    assert by_question(s.run_dir)["q1"]["status"] == "answered"


def test_the_reservation_stays_counted_after_mark_failed(project: Project) -> None:
    """H4, and the question ends without another dispatch."""
    s = session(project, {"q1": [bad_gateway(), ok()]})
    s.run()
    (event,) = s.events("outcome_unknown")
    s.reconcile(event["attempt_id"], lr.RESOLUTION_FAIL)
    assert s.ledger().reserved == pytest.approx(s.upper_bound())
    s.run()
    assert s.wire.count("q1") == 1
    assert by_question(s.run_dir)["q1"]["status"] == "execution_failed"
    assert s.ledger().reserved == pytest.approx(s.upper_bound())


def test_allow_new_attempt_permits_exactly_one_further_dispatch(project: Project) -> None:
    """H5: after the reconcile the next run sends q1 once. A second 502 is unknown again and is not resent."""
    s = session(project, {"q1": [bad_gateway(), bad_gateway(), ok()]})
    s.run()
    assert s.wire.count("q1") == 1
    (first,) = s.events("outcome_unknown")
    s.reconcile(first["attempt_id"], lr.RESOLUTION_ALLOW)
    assert s.wire.count("q1") == 1, "reconcile never contacts the provider"
    s.run()
    assert s.wire.count("q1") == 2, "exactly one further dispatch"
    assert len(s.events("outcome_unknown")) == 2
    s.run()
    assert s.wire.count("q1") == 2, "the second 502 waits for its own reconcile"
    assert s.ledger().reserved == pytest.approx(2 * s.upper_bound())


def test_the_one_further_dispatch_can_answer(project: Project) -> None:
    s = session(project, {"q1": [bad_gateway(), ok()]})
    s.run()
    (first,) = s.events("outcome_unknown")
    s.reconcile(first["attempt_id"], lr.RESOLUTION_ALLOW)
    result = s.run()
    assert s.wire.count("q1") == 2
    assert by_question(s.run_dir)["q1"]["status"] == "answered"
    assert result.summary["run_state"] == "complete"
    s.run()
    assert s.wire.count("q1") == 2


def test_a_gateway_outage_stops_after_three_dispatches(project: Project) -> None:
    """H9: three consecutive unknown outcomes trip the circuit breaker."""
    s = session(project, {q: [bad_gateway()] for q in ("q1", "q2", "q3", "q4", "q6")})
    result = s.run()
    assert s.wire.seen == ["q1", "q2", "q3"]
    assert result.summary["run_state"] == "needs_reconciliation"
    s.run()
    assert s.wire.seen == ["q1", "q2", "q3", "q4", "q6"], "a resume sends only the questions that were never dispatched"


# ---------------------------------------------------------------------------
# Failures that are still retried
# ---------------------------------------------------------------------------


def connect_error() -> Step:
    return raises(lambda request: httpx.ConnectError("connection refused", request=request))


def test_a_connect_error_is_retried_within_the_attempt_limit(project: Project) -> None:
    """H6: the request never left the machine, so a resend is safe. It costs nothing in the ledger."""
    s = session(project, {"q1": [connect_error(), connect_error(), ok()]})
    result = s.run()
    assert s.wire.count("q1") == 3
    assert len(s.sleeps) == 2 and all(delay > 0 for delay in s.sleeps)
    assert len(s.events("attempt_failed")) == 2
    assert {e["outcome"] for e in s.events("attempt_failed")} == {"not_sent"}
    assert by_question(s.run_dir)["q1"]["status"] == "answered"
    assert s.ledger().reserved == 0
    assert result.summary["run_state"] == "complete"


def test_connect_errors_stop_at_the_attempt_limit(project: Project) -> None:
    s = session(project, {"q1": [connect_error()] * 5})
    s.run()
    assert s.wire.count("q1") == 3, "max_attempts is 3, so no fourth request"
    assert by_question(s.run_dir)["q1"]["status"] == "execution_failed"
    assert s.ledger().reserved == 0


# ---------------------------------------------------------------------------
# Failures that stay unknown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "step", "kind"),
    [
        ("read timeout", raises(lambda r: httpx.ReadTimeout("no bytes", request=r)), "timeout"),
        ("connection lost", raises(lambda r: httpx.RemoteProtocolError("Server disconnected", request=r)), "connection_lost"),
        ("html 200", lambda r: httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"}), "malformed_response"),
        ("200 without choices", lambda r: httpx.Response(200, json={"id": "x", "object": "chat.completion"}), "malformed_response"),
        ("504", status(504, text="<html>timeout</html>"), "gateway_timeout"),
    ],
)
def test_other_uncertain_outcomes_are_one_dispatch_and_stay_unknown(project: Project, label: str, step: Step, kind: str) -> None:
    """H7."""
    s = session(project, {"q1": [step, ok()]})
    s.run()
    assert s.wire.count("q1") == 1, label
    (event,) = s.events("outcome_unknown")
    assert event["kind"] == kind, label
    assert s.events("attempt_failed") == [], label
    s.run()
    assert s.wire.count("q1") == 1, label
    assert s.ledger().reserved == pytest.approx(s.upper_bound()), label


# ---------------------------------------------------------------------------
# Which statuses are retried
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "body"),
    [
        (429, {"error": {"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"}}),
        (500, {"error": {"message": "The server had an error while processing your request", "type": "server_error", "code": None}}),
        (503, {"error": {"message": "Service unavailable", "type": "server_error", "code": None}}),
        (409, {"error": {"message": "Conflict", "type": "invalid_request_error", "code": None}}),
    ],
)
def test_statuses_without_proof_of_no_processing_are_not_resent(project: Project, code: int, body: dict) -> None:
    """H8: OpenAI recommends retrying these, but no source says the first request was not processed."""
    s = session(project, {"q1": [status(code, body=body, headers={"retry-after": "1"}), ok()]})
    s.run()
    assert s.wire.count("q1") == 1
    assert len(s.events("outcome_unknown")) == 1 and s.events("attempt_failed") == []
    assert s.sleeps == []


def test_a_documented_overload_503_is_retried_and_billed_at_most_once(project: Project) -> None:
    """H8: "does not have enough capacity to process your request" is a refusal to process."""
    body = {"error": {"message": "The model is overloaded", "type": "service_unavailable_error", "code": "server_is_overloaded"}}
    s = session(project, {"q1": [status(503, body=body, headers={"retry-after": "1"}), ok()]})
    s.run()
    assert s.wire.count("q1") == 2
    (failed,) = s.events("attempt_failed")
    assert (failed["outcome"], failed["decision"], failed["http_status"]) == ("rejected", "retry", 503)
    assert by_question(s.run_dir)["q1"]["status"] == "answered"
    # The rejected first attempt keeps its reservation. Only the second attempt is measured.
    assert s.ledger().reserved == pytest.approx(s.upper_bound())


def test_a_408_is_retried_within_the_attempt_limit(project: Project) -> None:
    s = session(project, {"q1": [status(408), status(408), status(408), ok()]})
    s.run()
    assert s.wire.count("q1") == 3
    assert by_question(s.run_dir)["q1"]["status"] == "execution_failed"
    assert s.ledger().reserved == pytest.approx(3 * s.upper_bound())


def test_a_run_without_failures_reserves_nothing(project: Project) -> None:
    """A guard for the ledger arithmetic above: measured answers hold no reservation and cost less than their bound."""
    s = session(project)
    s.run()
    ledger = s.ledger()
    assert ledger.reserved == 0 and ledger.measured > 0
    assert ledger.measured < s.upper_bound()


def test_a_rejected_attempt_keeps_the_provider_request_id_for_later_checks(project: Project) -> None:
    """A rejected attempt may still need a check with the provider, so its request id and body are kept."""
    body = {"error": {"message": "The model is overloaded", "type": "service_unavailable_error", "code": "server_is_overloaded"}}
    s = session(project, {"q1": [status(503, body=body, headers={"x-request-id": "req_abc123"}), ok()]})
    s.run()
    (failed,) = s.events("attempt_failed")
    assert failed["provider_raw"]["x_request_id"] == "req_abc123"
    assert failed["provider_raw"]["error_code"] == "server_is_overloaded"
    assert "authorization" not in json.dumps(failed).lower() and "test-not-a-key" not in json.dumps(failed)


# ---------------------------------------------------------------------------
# Redirects (H10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [307, 308])
def test_a_redirect_is_not_followed_and_the_attempt_is_unknown(project: Project, code: int) -> None:
    """H10: a 307 or 308 makes an SDK client that follows redirects post the prompt again, to the Location URL."""
    elsewhere = "http://other.invalid/v1/elsewhere"
    redirect: Step = lambda request: httpx.Response(code, headers={"location": elsewhere})  # noqa: E731
    s = session(project, {"q1": [redirect, ok()]})
    result = s.run(live_build=True)
    assert s.wire.count("q1") == 1, "the redirect target was not requested"
    assert not any("other.invalid" in url for url in s.wire.urls)
    assert len(s.events("dispatch_started")) == 1
    assert s.events("attempt_failed") == []
    (event,) = s.events("outcome_unknown")
    assert event["kind"] == "redirect" and event["provider_raw"]["http_status"] == code
    assert result.summary["run_state"] == "needs_reconciliation"
    assert s.ledger().reserved == pytest.approx(s.upper_bound())
    s.run(live_build=True)
    assert s.wire.count("q1") == 1, "a resume does not send it again"


# ---------------------------------------------------------------------------
# Bounded provider payloads (L4)
#
# Ways `_bounded_raw` could fail, written before the change.
#   B1. A large malformed 200 loses `usage`, the billing evidence, because the payload was cut as one string
#       and `usage` sorts last.
#   B2. The bounded record still grows with the payload: many keys, long key names, a large list or string.
#   B3. A payload that fits the bound changes shape.
#   B4. A truncated value carries no marker, so a reader takes the prefix for the whole value.
# ---------------------------------------------------------------------------

USAGE = {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43}


def test_a_large_malformed_200_keeps_its_usage_id_and_model_in_the_event(project: Project) -> None:
    """B1: a 3 MB body with no choices is an unknown outcome, and the event still says what it cost."""
    payload = {"id": "chatcmpl-big", "object": "chat.completion", "created": 1, "model": "gpt-4o-2024-11-20", "choices": [], "usage": USAGE, "junk": "A" * 3_000_000}
    s = session(project, {"q1": [lambda request: httpx.Response(200, json=payload)]})
    s.run()
    (event,) = s.events("outcome_unknown")
    raw = event["provider_raw"]
    assert event["kind"] == "malformed_response"
    counts = lambda usage: {k: usage[k] for k in USAGE}  # noqa: E731  the SDK adds its own empty detail fields
    assert counts(raw["usage"]) == USAGE and raw["id"] == "chatcmpl-big" and raw["model"] == "gpt-4o-2024-11-20"
    assert counts(raw["payload"]["usage"]) == USAGE and raw["payload"]["id"] == "chatcmpl-big"
    assert raw["payload"]["junk"]["truncated"] is True and raw["payload"]["junk"]["chars"] > 3_000_000
    assert len(lr._line(raw)) <= lr.PROVIDER_RAW_LIMIT
    line = next(text for text in (s.run_dir / lr.ATTEMPTS_NAME).read_text(encoding="utf-8").splitlines() if '"chatcmpl-big"' in text)
    assert len(line) < 3 * lr.PROVIDER_RAW_LIMIT


def test_a_payload_within_the_bound_is_stored_unchanged() -> None:
    """B3."""
    payload = {"reason": "no choices", "usage": USAGE, "payload": {"id": "x", "choices": []}}
    assert lr._bounded_raw(payload) == payload
    assert lr._bounded_raw({}) is None and lr._bounded_raw(None) is None


@pytest.mark.parametrize(
    "payload",
    [
        {f"key_{i}": "v" * 5_000 for i in range(200)},
        {"k" * 5_000 + str(i): 1 for i in range(50)},
        {"usage": USAGE, "payload": {f"k{i}": ["x" * 1_000] * 50 for i in range(60)}},
        {"usage": USAGE, "payload": {"deep": {"deeper": {"deepest": "z" * 500_000}}}},
        ["x" * 100_000] * 5,
        "s" * 500_000,
    ],
    ids=["many_keys", "long_key_names", "wide_payload", "deep_payload", "list", "string"],
)
def test_a_bounded_payload_stays_within_the_bound_and_marks_what_it_cut(payload: Any) -> None:
    """B2, B4."""
    bounded = lr._bounded_raw(payload)
    text = lr._line(bounded)
    assert len(text) <= lr.PROVIDER_RAW_LIMIT
    assert "truncated" in text or "omitted" in text
    if isinstance(payload, dict) and "usage" in payload:
        assert bounded["usage"] == USAGE, "the billing evidence survives"


def test_an_unserialisable_payload_is_kept_as_bounded_text() -> None:
    bounded = lr._bounded_raw({"x": object()})
    assert bounded["unserialisable"] is True and len(bounded["text"]) <= lr.PROVIDER_RAW_LIMIT
