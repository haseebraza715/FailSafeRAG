"""Service tier, storage policy and the returned-tier safety stop of the answer-model run driver.

Every test uses the fake provider. Nothing here sends a request.

Ways the driver could fail, written before the code. Each test below covers one.

Configuration and identity
  T1. A config asks for a tier other than Standard, or for storage, and the run starts.
  T2. A real price table with no declared tier is accepted, so its rates could be applied to another tier.
  T3. The tier or the storage policy is missing from the identity, so a run with another policy resumes.
Returned tier
  T4. A response from a tier other than Standard, or with no tier, is accepted and priced at Standard rates.
  T5. The run dispatches after such a response, or a restart, a raised ceiling or a reconcile clears the stop.
  T6. The offending response or its usage is dropped, or counted as verified measured cost.
  T7. The summary or status calls the Standard-rate bound an upper bound of the real charge.
  T8. A run recorded before the tier existed stops, or its exports change.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_live_runner import (
    QUESTIONS,
    SENT,
    Project,
    Rig,
    _ended,
    attempt_id_of,
    by_question,
    conditions,
    default_project,
    events_of,
    file_snapshot,
    go,
    options,
    requests_of,
    stop_violations,
    summary_of,
    valid_live_config,
)

from faar import live_runner as lr
from faar.answer_providers import FakeStep
from faar.live_contract import (
    RETURNED_SERVICE_TIERS_ACCEPTED,
    SERVICE_TIER_STANDARD,
    STORAGE_DISABLED,
    STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP,
)
from faar.pilot_runner import RunnerRefusal

ROOT = Path(__file__).resolve().parents[1]
CONDITION = "returned_service_tier_unverified"
SAFETY_EXIT = 6



@pytest.fixture
def project(tmp_path: Path) -> Project:
    return default_project(tmp_path / "project")

def tier_rig(tier: str | None, question: str = "q2") -> Rig:
    return Rig(steps={question: [FakeStep("answer", text="twelve months", service_tier=tier)]})


# ---------------------------------------------------------------------------
# T1 to T3: configuration and identity
# ---------------------------------------------------------------------------


def with_prices(**changes: Any) -> dict[str, Any]:
    payload = valid_live_config()
    payload["prices"] = {**payload["prices"], "service_tier": "default", **changes}
    return payload


def test_a_config_defaults_to_the_standard_tier_and_disabled_storage() -> None:
    config = lr.parse_provider_config(with_prices())
    assert (config.service_tier, config.storage) == (SERVICE_TIER_STANDARD, STORAGE_DISABLED)
    assert config.prices.service_tier == SERVICE_TIER_STANDARD
    for builtin in (lr.FAKE_CONFIG, lr.PROVISIONAL_DRY_RUN_CONFIG):
        assert (builtin.service_tier, builtin.storage) == (SERVICE_TIER_STANDARD, STORAGE_DISABLED)
        assert builtin.prices.service_tier == SERVICE_TIER_STANDARD


def test_explicit_standard_tier_and_store_false_are_accepted() -> None:
    config = lr.parse_provider_config({**with_prices(), "service_tier": "default", "store": False})
    assert (config.service_tier, config.storage) == ("default", "disabled")


@pytest.mark.parametrize("tier", ["priority", "flex", "scale", "auto", "fast", "", None, 1])
def test_any_requested_tier_but_standard_is_refused(tier: object) -> None:
    with pytest.raises(RunnerRefusal, match="service_tier"):
        lr.parse_provider_config({**with_prices(), "service_tier": tier})


@pytest.mark.parametrize("store", [True, None, "false", 0, 1])
def test_store_must_be_false_when_present(store: object) -> None:
    with pytest.raises(RunnerRefusal, match="store"):
        lr.parse_provider_config({**with_prices(), "store": store})


def test_store_true_names_the_missing_authorization() -> None:
    with pytest.raises(RunnerRefusal, match="authorization"):
        lr.parse_provider_config({**with_prices(), "store": True})


def test_a_real_price_table_must_declare_the_standard_tier() -> None:
    payload = valid_live_config()
    del payload["prices"]["service_tier"]
    with pytest.raises(RunnerRefusal, match=r"prices\.service_tier"):
        lr.parse_provider_config(payload)
    for tier in ("priority", "auto", None, ""):
        with pytest.raises(RunnerRefusal, match=r"prices\.service_tier"):
            lr.parse_provider_config(with_prices(service_tier=tier))


def test_a_simulated_price_table_may_omit_the_tier() -> None:
    payload = valid_live_config()
    del payload["prices"]["service_tier"]
    config = lr.parse_provider_config(payload, simulated=True)
    assert config.prices.service_tier == SERVICE_TIER_STANDARD


def test_a_simulated_price_table_may_not_declare_another_tier() -> None:
    with pytest.raises(RunnerRefusal, match=r"prices\.service_tier"):
        lr.parse_provider_config(with_prices(service_tier="priority"), simulated=True)


def test_the_identity_carries_the_tier_and_the_storage_policy(project: Project) -> None:
    go(project, Rig())
    config = json.loads((project.run_dir("run-a") / "run_config.json").read_text())
    provider = config["identity"]["provider"]
    assert provider["service_tier"] == "default" and provider["storage"] == "disabled"
    assert config["identity"]["prices"]["service_tier"] == "default"


def test_the_dry_run_summary_names_the_tier_and_storage(project: Project) -> None:
    result = lr.dry_run(project_root=project.root, out_dir=project.root / "dry", pilot_id="fix_v1")
    config = result.summary["provider_config"]
    assert config["service_tier"] == "default" and config["storage"] == "disabled"
    assert config["prices"]["service_tier"] == "default"


def test_storage_enabled_is_refused_for_any_run_this_cli_starts(project: Project) -> None:
    import dataclasses

    config = dataclasses.replace(lr.FAKE_CONFIG, storage=STORAGE_ENABLED_FOR_ATTEMPT_LOOKUP)
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="future explicit authorization"):
        lr.execute_run(
            options(project, config=config), provider_factory=rig.factory, descriptor=rig.descriptor, environ={}
        )
    assert rig.factory_calls == 0 and not project.run_dir("run-a").exists()


def test_another_tier_in_a_config_object_is_refused_before_anything_starts(project: Project) -> None:
    import dataclasses

    config = dataclasses.replace(lr.FAKE_CONFIG, service_tier="priority")
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="service_tier"):
        lr.execute_run(
            options(project, config=config), provider_factory=rig.factory, descriptor=rig.descriptor, environ={}
        )
    assert rig.factory_calls == 0 and not project.run_dir("run-a").exists()


# ---------------------------------------------------------------------------
# T4 to T7: the returned-tier stop
# ---------------------------------------------------------------------------


def test_the_accepted_returned_tiers_are_the_standard_tier_alone() -> None:
    assert RETURNED_SERVICE_TIERS_ACCEPTED == ("default",)


def test_a_standard_tier_run_completes_and_saves_the_returned_tier(project: Project) -> None:
    rig = Rig()
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert result.exit_code == 0 and rig.sent(run_dir) == SENT
    saved = [e for e in events_of(run_dir) if e["event"] == "response_saved"]
    assert len(saved) == len(SENT) and {e["returned_service_tier"] for e in saved} == {"default"}
    for event in saved:
        response = json.loads((run_dir / event["response_file"]).read_text())
        assert response["returned_service_tier"] == "default"
    summary = summary_of(run_dir)
    assert "unverified_tier" not in summary["cost"] and "safety_violations" not in summary


@pytest.mark.parametrize("tier", ["priority", "flex", "scale", "auto", "fast", "premium-2027", "DEFAULT", " default", ""])
def test_a_response_from_another_tier_stops_dispatch_after_two_requests(project: Project, tier: str) -> None:
    rig = tier_rig(tier)
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2"], "nothing is dispatched after the offending response"
    assert sum(len(p.calls) for p in rig.providers) == 2
    assert result.exit_code == SAFETY_EXIT and _ended(run_dir)["reason"] == "safety_stop"
    assert conditions(stop_violations(run_dir)) == {(attempt_id_of(run_dir, "q2"), CONDITION)}
    detail = stop_violations(run_dir)[0]["detail"]
    assert repr(tier) in detail and "default" in detail
    saved = next(e for e in events_of(run_dir) if e["event"] == "response_saved" and e["question_id"] == "q2")
    assert saved["returned_service_tier"] == tier
    response = json.loads((run_dir / saved["response_file"]).read_text())
    assert response["returned_service_tier"] == tier and response["usage"] == saved["usage"] != {}
    assert response["provider_response"]["text"] == "twelve months"


def test_a_response_with_no_tier_stops_dispatch_and_says_the_tier_is_missing(project: Project) -> None:
    rig = tier_rig(None)
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2"] and result.exit_code == SAFETY_EXIT
    violation = stop_violations(run_dir)[0]
    assert violation["condition"] == CONDITION and violation["attempt_id"] == attempt_id_of(run_dir, "q2")
    assert "missing" in violation["detail"]
    saved = next(e for e in events_of(run_dir) if e["event"] == "response_saved" and e["question_id"] == "q2")
    assert saved["returned_service_tier"] is None


def test_the_offending_answer_is_kept_and_the_unserved_questions_are_never_abstentions(project: Project) -> None:
    go(project, tier_rig("priority"))
    run_dir = project.run_dir("run-a")
    predictions = by_question(run_dir)
    assert list(predictions) == QUESTIONS
    assert predictions["q2"]["status"] == "answered" and predictions["q2"]["usage"] is not None
    for qid in ("q3", "q4", "q6"):
        assert predictions[qid]["failure"]["type"] == "safety_stop" and predictions[qid]["abstained"] is False
    summary = summary_of(run_dir)
    assert summary["run_state"] == "safety_stopped" and summary["valid_baseline"] is False
    assert any(CONDITION in blocker for blocker in summary["valid_baseline_blockers"])


def test_a_restart_builds_no_provider_and_dispatches_nothing(project: Project) -> None:
    go(project, tier_rig("priority"))
    run_dir = project.run_dir("run-a")
    before = {k: v for k, v in file_snapshot(run_dir).items() if k != "run.lock"}
    fresh = tier_rig("priority")
    result = go(project, fresh)
    assert result.exit_code == SAFETY_EXIT and "cannot undo a charge" in result.message
    assert fresh.factory_calls == 0 and fresh.providers == []
    assert {k: v for k, v in file_snapshot(run_dir).items() if k != "run.lock"} == before
    bare = lr.execute_run(
        options(project), provider_factory=None, descriptor=tier_rig("priority").descriptor, sleep=lambda s: None, environ={}
    )
    assert bare.exit_code == SAFETY_EXIT


def test_a_raised_ceiling_reconcile_and_reopen_do_not_clear_the_tier_stop_and_score_refuses(project: Project) -> None:
    go(project, tier_rig("priority"))
    run_dir = project.run_dir("run-a")
    later = tier_rig("priority")
    raised = go(project, later, safety_ceiling=100.0, raise_safety_ceiling=500.0, authorization_note="lead approved")
    assert raised.exit_code == SAFETY_EXIT and later.factory_calls == 0
    assert summary_of(run_dir)["safety_ceiling"]["amount"] == 100.0
    with pytest.raises(RunnerRefusal):
        lr.reopen_question(run_dir=run_dir, question_id="q1", note="try again")
    assert lr.export_run(run_dir).exit_code == SAFETY_EXIT
    assert lr.run_status(run_dir)["run_state"] == "safety_stopped"
    with pytest.raises(RunnerRefusal, match="safety"):
        lr.score_run_live(project_root=project.root, run_dir=run_dir)
    assert not (run_dir / "scores.jsonl").exists()


def test_a_crash_between_the_response_file_and_its_event_still_stops_on_restart(project: Project) -> None:
    from test_live_runner import crash_at

    from faar.answer_providers import SimulatedCrash

    with pytest.raises(SimulatedCrash):
        go(project, tier_rig("priority"), crash_hook=crash_at(lr.CRASH_AFTER_RESPONSE_FILE, "q2"))
    run_dir = project.run_dir("run-a")
    restart = tier_rig("priority")
    result = go(project, restart)
    assert result.exit_code == SAFETY_EXIT and restart.factory_calls == 0
    assert (attempt_id_of(run_dir, "q2"), CONDITION) in conditions(result.summary["safety_violations"])
    recovered = next(e for e in events_of(run_dir) if e["event"] == "response_saved" and e["question_id"] == "q2")
    assert recovered["returned_service_tier"] == "priority" and recovered.get("recovered") is True


def test_a_saved_event_that_differs_from_its_response_file_in_the_tier_is_refused(project: Project) -> None:
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    lines = (run_dir / "attempts.jsonl").read_text().splitlines()
    edited = []
    for line in lines:
        event = json.loads(line)
        if event["event"] == "response_saved" and event["question_id"] == "q2":
            event["returned_service_tier"] = "flex"
        edited.append(json.dumps(event, separators=(",", ":")))
    (run_dir / "attempts.jsonl").write_text("\n".join(edited) + "\n")
    with pytest.raises(RunnerRefusal, match="returned_service_tier"):
        lr.load_run(run_dir)


# The real adapter on a mock transport: the tier the wire reports reaches the stop.


def wire_step(tier: object) -> Any:
    import httpx
    from test_live_http_outcomes import completion

    def respond(request: httpx.Request) -> httpx.Response:
        body = completion()
        if tier == "absent":
            body.pop("service_tier", None)
        else:
            body["service_tier"] = tier
        return httpx.Response(200, json=body, headers={"x-request-id": "req_tier"})

    return respond


@pytest.mark.parametrize("tier", ["priority", "flex", "scale", "auto", "absent", None])
def test_a_tier_reported_on_the_wire_stops_the_run_after_two_requests(project: Project, tier: object) -> None:
    from test_live_http_outcomes import Session, Wire

    wire = Wire({"q2": [wire_step(tier)]})
    session = Session(project, wire)
    result = session.run()
    assert wire.seen == ["q1", "q2"], "no request leaves after the offending response"
    assert result.exit_code == SAFETY_EXIT
    violations = result.summary["safety_violations"]
    assert [v["condition"] for v in violations] == [CONDITION]
    saved = next(e for e in events_of(session.run_dir) if e["event"] == "response_saved" and e["question_id"] == "q2")
    assert saved["returned_service_tier"] == (None if tier in ("absent", None) else tier)
    assert all(body["service_tier"] == "default" and body["store"] is False for body in wire.bodies)
    restart = Session(project, Wire())
    again = restart.run()
    assert again.exit_code == SAFETY_EXIT and restart.wire.providers == 0 and restart.wire.seen == []


def test_the_standard_tier_reported_on_the_wire_completes_the_run(project: Project) -> None:
    from test_live_http_outcomes import Session, Wire

    wire = Wire()
    result = Session(project, wire).run()
    assert result.exit_code == 0 and wire.seen == SENT
    assert result.summary["returned_service_tiers"] == {"default": len(SENT)}


# T6, T7: cost honesty


def stopped_run(project: Project, tier: str | None = "priority") -> tuple[Path, dict[str, Any]]:
    go(project, tier_rig(tier))
    run_dir = project.run_dir("run-a")
    return run_dir, summary_of(run_dir)


def test_the_unverified_response_is_kept_out_of_measured_cost_and_reported_separately(project: Project) -> None:
    run_dir, summary = stopped_run(project)
    events = {e["question_id"]: e for e in events_of(run_dir) if e["event"] == "response_saved"}
    q1_cost = events["q1"]["measured_cost"]
    q2_usage = events["q2"]["usage"]
    cost = summary["cost"]
    assert cost["measured"] == q1_cost, "only the verified response is measured cost"
    block = cost["unverified_tier"]
    assert block["attempts"] == 1 and block["attempt_ids"] == [attempt_id_of(run_dir, "q2")]
    assert block["usage"]["input_tokens"] == q2_usage["input_tokens"]
    assert block["usage"]["output_tokens"] == q2_usage["output_tokens"]
    assert block["standard_rate_cost"] > 0
    assert "not an actual-cost claim" in block["note"] and "not an upper bound" in block["note"]
    upper_q2 = next(r for r in requests_of(run_dir) if r["question_id"] == "q2")["cost_upper_bound"]
    assert cost["reserved"] == pytest.approx(upper_q2), "the response stays reserved at its Standard-rate bound"
    assert cost["committed_upper"] == pytest.approx(q1_cost + upper_q2)


def test_the_prediction_of_the_unverified_response_has_no_measured_cost_but_keeps_its_usage(project: Project) -> None:
    run_dir, _ = stopped_run(project)
    predictions = by_question(run_dir)
    assert predictions["q2"]["measured_cost"] is None and predictions["q2"]["usage"]["input_tokens"] > 0
    assert predictions["q1"]["measured_cost"] is not None


def test_status_and_the_run_message_say_the_standard_bound_may_understate_the_charge(project: Project) -> None:
    run_dir, _ = stopped_run(project)
    status = lr.run_status(run_dir)
    text = " ".join(lr.format_status(status).split())
    assert "unverified tier" in text and "not an upper bound" in text and "1.7" in text
    assert "cannot undo a charge" in text
    message = " ".join(lr.export_run(run_dir).message.split())
    assert "not an upper bound" in message and "unverified" in message


def test_a_missing_tier_is_reported_the_same_way(project: Project) -> None:
    _, summary = stopped_run(project, None)
    assert summary["cost"]["unverified_tier"]["attempts"] == 1


def test_a_run_without_an_unverified_response_has_no_such_block(project: Project) -> None:
    go(project, Rig())
    assert "unverified_tier" not in summary_of(project.run_dir("run-a"))["cost"]
    assert "unverified" not in lr.format_status(lr.run_status(project.run_dir("run-a")))


# ---------------------------------------------------------------------------
# T8: runs recorded before the tier existed
# ---------------------------------------------------------------------------

LEGACY_RUNS = ("2026-09-29-ohr-dev-v1-fake-provider-r4", "2026-09-29-ohr-dev-v1-fake-provider-r3")


def legacy_source(name: str) -> Path:
    """The recorded run. Its requests.jsonl is git-ignored, so a fresh checkout may lack it (the test skips)."""
    import os

    for base in (ROOT / "results" / "engineering", *([Path(os.environ["FAAR_LEGACY_RUNS_ROOT"])] if "FAAR_LEGACY_RUNS_ROOT" in os.environ else [])):
        if (base / name / "requests.jsonl").is_file():
            return base / name
    pytest.skip(f"{name} has no requests.jsonl in this checkout")


@pytest.mark.parametrize("name", LEGACY_RUNS)
def test_recorded_runs_load_report_and_export_byte_for_byte_as_before(tmp_path: Path, name: str) -> None:
    copy = tmp_path / "engineering" / name
    shutil.copytree(legacy_source(name), copy)
    before = file_snapshot(copy)
    view = lr.load_run(copy)
    assert "service_tier" not in view.config["identity"]["provider"]
    status = lr.run_status(copy)
    assert status["safety_violations"] == [] and status["run_state"] == "complete"
    assert "unverified_tier" not in status["cost"] and "run_kind" not in status
    assert lr.format_status(status)
    result = lr.export_run(copy)
    assert result.exit_code == 0 and "unchanged" in result.message
    after = file_snapshot(copy)
    assert {k: v for k, v in after.items() if k != "run.lock"} == {k: v for k, v in before.items() if k != "run.lock"}
    text, summary = lr.export_view(lr.fold_recoverable_responses(view), lr.Services.default())
    assert text.encode() == before["predictions.jsonl"]
    assert (lr._dumps(summary) + "\n").encode() == before["run_summary.json"]
    assert "run_kind" not in summary and "unverified_tier" not in summary["cost"]


def test_a_run_whose_identity_declares_no_tier_is_not_stopped_for_a_response_with_no_tier(project: Project) -> None:
    """The check follows the identity: a legacy identity has no service_tier, so no response is judged by it."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    view = lr.load_run(run_dir)
    legacy = lr.load_run(run_dir)
    for key in ("service_tier", "storage"):
        legacy.config["identity"]["provider"].pop(key)
    legacy.config["identity"]["prices"].pop("service_tier")
    stripped = [
        {k: v for k, v in e.items() if k != "returned_service_tier"} if e["event"] == "response_saved" else e for e in legacy.events
    ]
    legacy.events[:] = stripped
    assert lr.find_safety_violations(legacy, lr.Services.default()) == []
    assert lr.find_safety_violations(view, lr.Services.default()) == []


def malformed_step(tier: object) -> Any:
    import httpx
    from test_live_http_outcomes import completion

    def respond(request: httpx.Request) -> httpx.Response:
        body = completion()
        body["choices"] = []
        if tier == "absent":
            body.pop("service_tier", None)
        else:
            body["service_tier"] = tier
        return httpx.Response(200, json=body, headers={"x-request-id": "req_malformed"})

    return respond


@pytest.mark.parametrize("tier", ["priority", "flex", ""])
def test_an_unusable_reply_that_reports_another_tier_stops_dispatch(project: Project, tier: str) -> None:
    from test_live_http_outcomes import Session, Wire

    wire = Wire({"q2": [malformed_step(tier)]})
    result = Session(project, wire).run()
    assert wire.seen == ["q1", "q2"], "no request leaves after a reply that reports a non-Standard tier"
    assert result.exit_code == SAFETY_EXIT
    assert [v["condition"] for v in result.summary["safety_violations"]] == [CONDITION]
    again = Session(project, Wire()).run()
    assert again.exit_code == SAFETY_EXIT


def test_an_unusable_reply_with_no_tier_waits_for_reconciliation_and_is_not_a_tier_stop(project: Project) -> None:
    from test_live_http_outcomes import Session, Wire

    wire = Wire({"q2": [malformed_step("absent")]})
    result = Session(project, wire).run()
    assert result.exit_code != SAFETY_EXIT
    assert "safety_violations" not in result.summary
