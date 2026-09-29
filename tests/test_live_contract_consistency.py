"""Constants that two modules of the answer-model path define separately must agree.

Failure list: the fake provider abstains with a string that the reply parser does
not treat as an abstention; the provider and the retry policy disagree on which
error kinds stop a run; an error kind that the adapter marks with outcome ``unknown``
(a 504 gateway timeout, a malformed 200 reply) gets a retry instead of a reconcile;
the adapter accepts a request parameter that ``faar.live_contract`` does not allow.
"""

import httpx
import openai
import pytest

from faar import answer_prompt, answer_providers, live_contract, retry_policy


def test_fake_abstention_text_is_the_prompt_abstention_token() -> None:
    assert answer_providers.ABSTENTION_TEXT == answer_prompt.ABSTENTION_TOKEN
    parsed = answer_prompt.parse_reply(answer_providers.ABSTENTION_TEXT, "stop", None)
    assert parsed == {"answer": "", "abstained": True, "output_status": "abstained"}


def test_provider_and_retry_policy_agree_on_run_stopping_kinds() -> None:
    assert set(answer_providers.STOP_RUN_KINDS) == set(retry_policy.STOP_RUN_KINDS)


@pytest.mark.parametrize(
    ("status", "kind"),
    [(504, "gateway_timeout")],
)
def test_unknown_outcome_kinds_from_the_adapter_go_to_reconcile_not_retry(status: int, kind: str) -> None:
    client = openai.OpenAI(
        api_key="test-not-a-key",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, text="gateway"))),
        max_retries=0,
    )
    provider = answer_providers.OpenAIChatProvider("gpt-test", {}, client=client)
    request = live_contract.ProviderRequest("r1", "r1-a1", ({"role": "user", "content": "q"},), {}, 8, 5.0)
    with pytest.raises(live_contract.ProviderError) as info:
        provider.send(request)
    assert info.value.kind == kind
    decision = retry_policy.RetryPolicy().decide(info.value, 1)
    assert decision.action == retry_policy.ACTION_RECONCILE


def test_adapter_accepts_exactly_the_contract_parameter_allowlist() -> None:
    client = openai.OpenAI(
        api_key="test-not-a-key",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))),
        max_retries=0,
    )
    for key in live_contract.ALLOWED_OPENAI_PARAMS:
        answer_providers.OpenAIChatProvider("gpt-test", {key: 0}, client=client)
    with pytest.raises(ValueError):
        answer_providers.OpenAIChatProvider("gpt-test", {"service_tier": "flex"}, client=client)
