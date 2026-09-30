"""Check the operational success criteria of a finished live-check run directory. Offline, read-only.

The script reads ``run_config.json``, ``requests.jsonl``, ``attempts.jsonl``, ``run_summary.json``,
``predictions.jsonl`` and the response files of one run directory. It checks mechanics only, never answer quality.
It sends nothing and reads no credential. It exits 0 when no check fails and 1 otherwise.

Each row is PASS, FAIL or INFO. A field that the runner records only in newer code (``returned_service_tier``,
``run_kind``, ``pilot_manifest``, the storage policy) is read where it is present and reported as ``absent``
otherwise. An absent field is INFO, not a failure. A field that is present with a wrong value is a FAIL.

The rows follow the criteria of the readiness report (section 5):

    1  run_config.json records mode live, the model, the endpoint and the parameters {"temperature": 0}
    2  each sent question has exactly one dispatch and one saved response; skipped questions have none
    3  every saved response has integer token counts, and measured_cost equals usage times the configured rates
    4  every returned model equals the requested model exactly, and no input bound is exceeded
    5  no attempt_failed, outcome_unknown or safety-violation event
    6  run_state is complete and no question is execution_failed
    7  the ledger has no anomaly and nothing reserved; the summary's measured cost equals the ledger's
    8  the returned service tier is default or absent in every response

Usage:
    python scripts/audits/verify_live_check.py RUN_DIR [--expect-run-kind engineering_check] \\
        [--expect-manifest-sha256 HEX] [--allow-fake]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXIT_FAILED_CHECK, fail, read_json, read_jsonl, require_dir, require_file, resolve  # noqa: E402

from faar.live_contract import PriceTable, ProviderUsage  # noqa: E402
from faar.request_budget import BudgetError, LedgerError, SafetyLedger, measured_cost_micro  # noqa: E402

ABSENT = "absent"
STANDARD_TIER = "default"


class Row(NamedTuple):
    status: str  # PASS, FAIL or INFO
    name: str
    detail: str = ""


def find_field(name: str, *sources: Mapping[str, Any]) -> Any:
    """Return the first value stored under ``name`` at the top of a source or one level down, else ``ABSENT``."""
    for source in sources:
        if name in source:
            return source[name]
    for source in sources:
        for value in source.values():
            if isinstance(value, Mapping) and name in value:
                return value[name]
    return ABSENT


def _brief(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _read_events(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path, path.name)


def verify(
    run_dir: Path,
    *,
    allow_fake: bool = False,
    expect_model: str = "gpt-4o-2024-11-20",
    expect_base_url: str = "https://api.openai.com/v1/",
    expect_sends: int = 6,
    expect_skips: int = 2,
    expect_run_kind: str = "engineering_check",
    expect_manifest_sha256: str | None = None,
) -> list[Row]:
    config = read_json(run_dir / "run_config.json", "run_config.json")
    requests = _read_events(require_file(run_dir / "requests.jsonl", "requests.jsonl"))
    events = _read_events(require_file(run_dir / "attempts.jsonl", "attempts.jsonl"))
    summary = read_json(run_dir / "run_summary.json", "run_summary.json")
    predictions_path = require_file(run_dir / "predictions.jsonl", "predictions.jsonl")
    predictions = _read_events(predictions_path)
    identity = config["identity"]
    provider = identity["provider"]
    prices = PriceTable(**identity["prices"])
    rows: list[Row] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        rows.append(Row("PASS" if ok else "FAIL", name, detail))

    def info(name: str, detail: str) -> None:
        rows.append(Row("INFO", name, detail))

    # 1. provider identity
    check("mode is live" if not allow_fake else "mode (fake allowed)", config["mode"] == "live" or allow_fake, config["mode"])
    check("provider is openai", provider["provider"] == "openai", provider["provider"])
    check("requested model", provider["model"] == expect_model, provider["model"])
    base = provider.get("adapter", {}).get("base_url")
    check("client base_url recorded and exact", allow_fake or base == expect_base_url, str(base))
    check("endpoint in config", provider["endpoint"] == "https://api.openai.com/v1", provider["endpoint"])
    check("params are only temperature=0", provider["params"] == {"temperature": 0}, json.dumps(provider["params"]))

    # fields that newer runner code records
    run_kind = find_field("run_kind", config, summary, identity)
    info("kind (recorded by the runner at the top of run_config.json)", str(config.get("kind", ABSENT)))
    if run_kind == ABSENT:
        info("run_kind", ABSENT)
    else:
        check(f"run_kind is {expect_run_kind}", run_kind == expect_run_kind, str(run_kind))
    pilot_manifest = find_field("pilot_manifest", config, summary, identity)
    info("pilot_manifest", ABSENT if pilot_manifest == ABSENT else _brief(pilot_manifest))
    storage = find_field("storage_policy", provider, identity, config, summary)
    info("storage_policy", str(storage))
    price_tier = prices.service_tier
    if price_tier is None:
        info("price table service_tier", ABSENT)
    else:
        check("price table service_tier is default", price_tier == STANDARD_TIER or allow_fake, str(price_tier))
    manifest_hash = find_field("runtime_manifest_sha256", identity)
    recorded_run_manifest = config.get("runtime_manifest", {})
    info("runtime manifest sha256 (identity)", str(manifest_hash))
    if expect_manifest_sha256:
        check(
            "runtime manifest hash equals the expected hash",
            manifest_hash == expect_manifest_sha256 and recorded_run_manifest.get("sha256", manifest_hash) == manifest_hash,
            f"identity {manifest_hash}",
        )

    # 2. request set and dispatch counts
    sends = [r for r in requests if r["action"] == "send"]
    skips = [r for r in requests if r["action"] == "skip"]
    check("send count", len(sends) == expect_sends, str(len(sends)))
    check("skip count", len(skips) == expect_skips, str(len(skips)))
    dispatched = Counter(e["question_id"] for e in events if e["event"] == "dispatch_started")
    send_ids = {r["question_id"] for r in sends}
    check("no question dispatched more than once", all(v == 1 for v in dispatched.values()), json.dumps(dispatched.most_common(3)))
    check("every sendable question dispatched", set(dispatched) == send_ids, f"{len(dispatched)} dispatched")
    check("skipped questions never dispatched", not ({r["question_id"] for r in skips} & set(dispatched)))

    # 3 and 4. saved responses
    saved = [e for e in events if e["event"] == "response_saved"]
    check(
        "one saved response per sendable question",
        {e["question_id"] for e in saved} == send_ids and len(saved) == len(sends),
        str(len(saved)),
    )
    bad_usage: list[str] = []
    bad_cost: list[str] = []
    bad_model: list[tuple[str, Any]] = []
    unreadable: list[str] = []
    over_bound: list[str] = []
    finish: Counter[str] = Counter()
    tier_values: Counter[str] = Counter()
    tier_sources: Counter[str] = Counter()
    for event in saved:
        qid = event["question_id"]
        usage = event.get("usage") or {}
        if not isinstance(usage.get("input_tokens"), int) or not isinstance(usage.get("output_tokens"), int):
            bad_usage.append(qid)
        else:
            recomputed = measured_cost_micro(
                ProviderUsage(*(usage.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"))),
                prices,
            )
            recorded = (
                None
                if event.get("measured_cost") is None
                else int((Decimal(repr(event["measured_cost"])) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
            )
            if recomputed is None or recorded != recomputed:
                bad_cost.append(qid)
            request = next((r for r in sends if r["question_id"] == qid), None)
            if request is not None and usage["input_tokens"] > request["input_token_upper_bound"]:
                over_bound.append(qid)
        if event.get("returned_model") != expect_model:
            bad_model.append((qid, event.get("returned_model")))
        try:
            response = json.loads((run_dir / event["response_file"]).read_text(encoding="utf-8"))["provider_response"]
        except (OSError, KeyError, ValueError):
            unreadable.append(qid)
            continue
        finish[str(response.get("finish_reason"))] += 1
        # The tier the provider returned: the event field, the response record, then the raw payload.
        if "returned_service_tier" in event:
            tier_values[str(event["returned_service_tier"])] += 1
            tier_sources["event.returned_service_tier"] += 1
        elif "returned_service_tier" in response:
            tier_values[str(response["returned_service_tier"])] += 1
            tier_sources["provider_response.returned_service_tier"] += 1
        elif "service_tier" in response.get("raw", {}):
            tier_values[str(response["raw"]["service_tier"])] += 1
            tier_sources["provider_response.raw.service_tier"] += 1
        else:
            tier_values["None"] += 1
            tier_sources[ABSENT] += 1
    check("every response file is readable", not unreadable, str(unreadable))
    check("usage (input and output tokens) present on every response", not bad_usage, str(bad_usage))
    check("measured_cost equals usage x rates (micro-unit rounding up)", not bad_cost, str(bad_cost))
    check("returned model equals requested model exactly", not bad_model, str(bad_model))
    check("no input bound exceeded", not over_bound, str(over_bound))
    info("finish reasons", json.dumps(finish))
    info("service tier source", json.dumps(tier_sources))
    check(
        "returned service tier is default or absent",
        set(tier_values) <= {STANDARD_TIER, "None"} or allow_fake,
        json.dumps(tier_values),
    )

    # 5 and 6. failures, unknowns, run state
    failed = [e for e in events if e["event"] == "attempt_failed"]
    unknown = [e for e in events if e["event"] == "outcome_unknown"]
    check("no attempt_failed events (explain any)", not failed, str([(e.get("kind"), e.get("http_status")) for e in failed]))
    check("no outcome_unknown events (explain any)", not unknown, str([e.get("kind") for e in unknown]))
    check("run_state complete", summary["run_state"] == "complete", summary["run_state"])
    check("no safety violations", not summary.get("safety_violations"), _brief(summary.get("safety_violations", "")))
    check("no execution_failed", summary["counts"]["execution_failed"] == 0, json.dumps(summary["counts"]))

    # 7. ledger and export consistency
    try:
        ledger = SafetyLedger.from_events(events, prices)
    except (LedgerError, BudgetError) as exc:
        check("ledger rebuilds from the events", False, str(exc))
        return rows
    check("ledger has no anomalies", not ledger.anomalies, str(ledger.anomalies))
    check(
        "summary measured equals ledger",
        abs(summary["cost"]["measured"] - ledger.measured) < 1e-9,
        f"{summary['cost']['measured']} vs {ledger.measured}",
    )
    check("nothing reserved at the end", ledger.reserved_micro == 0, str(ledger.reserved))
    check(
        "sum of per-response measured cost equals ledger measured",
        abs(sum(e.get("measured_cost") or 0 for e in saved) - ledger.measured) < 1e-6,
    )
    check(
        "predictions has one record per request, same order",
        [p["question_id"] for p in predictions] == [r["question_id"] for r in requests],
    )
    no_evidence = [p for p in predictions if p["status"] == "no_evidence"]
    check(
        "skipped questions exported as no_evidence with empty answer",
        len(no_evidence) == expect_skips and all(p["answer"] == "" and p["abstained"] for p in no_evidence),
        str(len(no_evidence)),
    )
    check(
        "predictions_sha256 recorded matches file",
        summary["predictions_sha256"] == hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    )
    ceiling = summary["safety_ceiling"]["amount"] if summary.get("safety_ceiling") else None
    check("committed upper within ceiling", ceiling is not None and ledger.committed_upper <= ceiling, f"{ledger.committed_upper} <= {ceiling}")
    info(
        "totals",
        f"measured {ledger.measured:.6f} {prices.currency}; committed_upper {ledger.committed_upper:.6f}; "
        f"{len(dispatched)} questions, {sum(dispatched.values())} dispatches",
    )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="live-check run directory, for example results/development/<run_id>")
    parser.add_argument("--allow-fake", action="store_true", help="accept a fake run (to test this script)")
    parser.add_argument("--expect-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--expect-base-url", default="https://api.openai.com/v1/")
    parser.add_argument("--expect-sends", type=int, default=6)
    parser.add_argument("--expect-skips", type=int, default=2)
    parser.add_argument("--expect-run-kind", default="engineering_check", help="checked only when the run records run_kind")
    parser.add_argument("--expect-manifest-sha256", help="hash of the derived runtime manifest the run must use")
    args = parser.parse_args(argv)

    run_dir = require_dir(resolve(args.run_dir), "run directory")
    try:
        rows = verify(
            run_dir,
            allow_fake=args.allow_fake,
            expect_model=args.expect_model,
            expect_base_url=args.expect_base_url,
            expect_sends=args.expect_sends,
            expect_skips=args.expect_skips,
            expect_run_kind=args.expect_run_kind,
            expect_manifest_sha256=args.expect_manifest_sha256,
        )
    except KeyError as exc:
        fail(f"{run_dir} lacks the record field {exc}; is it a run directory from run_pilot_live.py?", code=EXIT_FAILED_CHECK)
    width = max(len(r.name) for r in rows)
    for row in rows:
        print(f"{row.status:4s}  {row.name:{width}s}  {row.detail}")
    return EXIT_FAILED_CHECK if any(r.status == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
