"""Constants that two modules of the answer-model path define separately must agree.

Failure list: the fake provider abstains with a string that the reply parser does
not treat as an abstention; the provider and the retry policy disagree on which
error kinds stop a run.
"""

from faar import answer_prompt, answer_providers, retry_policy


def test_fake_abstention_text_is_the_prompt_abstention_token() -> None:
    assert answer_providers.ABSTENTION_TEXT == answer_prompt.ABSTENTION_TOKEN
    parsed = answer_prompt.parse_reply(answer_providers.ABSTENTION_TEXT, "stop", None)
    assert parsed == {"answer": "", "abstained": True, "output_status": "abstained"}


def test_provider_and_retry_policy_agree_on_run_stopping_kinds() -> None:
    assert set(answer_providers.STOP_RUN_KINDS) == set(retry_policy.STOP_RUN_KINDS)
