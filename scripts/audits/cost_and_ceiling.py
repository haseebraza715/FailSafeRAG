"""Cost bounds, an expected-cost estimate and a safety-ceiling replay for a prepared dry run.

Offline. The script reads ``prepared_requests.jsonl`` and ``dry_run_summary.json`` from a dry-run directory. It
sends nothing and reads no credential. The arithmetic comes from ``faar.request_budget``, the module the runner uses.

Bounds (facts about the prepared requests, given the price table in the dry-run summary):
    input bound         ``input_token_upper_bound`` recomputed from the saved messages (UTF-8 bytes + 4 per message + 16)
    one-attempt bound   sum over sent requests of ``request_cost_upper_bound_micro``
    N-attempt bound     the one-attempt bound times ``--max-attempts``

Expected cost (an estimate, not a bound). It rests on the assumptions below, all set by arguments and all
written into the output:
    input tokens        the repository's heuristic (``answer_prompt.estimate_tokens``), with a low and a high
                        variant from assumed characters per token. It is not a tokenizer count.
    output tokens       ``--output-tokens`` per request, with a low and a high variant
    retries             ``--retry-rate`` extra billed attempts per request, as a multiplier on the total
    A second estimate uses the study brief's assumptions (``--brief-output-tokens``, ``--brief-retry-rate``).

Fast tier (optional): ``--fast-multiplier 1.7`` scales all three rates and recomputes the bounds and the estimate.

Ceiling replay: the driver's dispatch rule, run against ``SafetyLedger`` and ``SafetyLedger.can_reserve`` with
scripted outcomes per question (answered, response without usage, unknown, rejected). It shows where a run would
stop at ``--ceiling``. A reservation is a bound, not a charge, and the ceiling does not see the service tier.

Usage:
    python scripts/audits/cost_and_ceiling.py --dry-run-dir .local/work/dry-run --out .local/work/audits \\
        [--subset-ids selected_question_ids.json] [--fast-multiplier 1.7]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAILED_CHECK,
    HAN,
    fail,
    prepare_out,
    read_json,
    read_jsonl,
    require_dir,
    resolve,
    write_json,
)

from faar.answer_prompt import estimate_tokens  # noqa: E402
from faar.live_contract import PriceTable, ProviderUsage  # noqa: E402
from faar.request_budget import (  # noqa: E402
    MICRO,
    SafetyLedger,
    input_token_upper_bound,
    measured_cost_micro,
    request_cost_upper_bound_micro,
)

DEFAULT_DRY_RUN = ".local/work/dry-run"
DEFAULT_OUT = ".local/work/audits"

Plan = Callable[[int], list[str]]


def load_prices(summary: dict[str, Any]) -> PriceTable:
    prices = summary["provider_config"]["prices"]
    return PriceTable(**prices)


def scaled(prices: PriceTable, factor: float) -> PriceTable:
    """The same table with every rate multiplied by ``factor`` (for the Fast-tier scenario)."""
    cached = prices.cached_input_per_million
    return replace(
        prices,
        input_per_million=prices.input_per_million * factor,
        cached_input_per_million=None if cached is None else cached * factor,
        output_per_million=prices.output_per_million * factor,
    )


def token_scenarios(row: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    """Input-token estimates for one sent request: low, central (the repository heuristic) and high."""
    text = row["system"] + "\n\n" + row["user"]
    han = len(HAN.findall(text))
    other = len(text) - han
    central = estimate_tokens(text)["estimate"]
    low = round(other / args.chars_per_token_low + han * args.cjk_tokens_low)
    high = round(other / args.chars_per_token_high + han * args.cjk_tokens_high) + args.framing_tokens
    return {"low": low, "central": central, "high": high}


def usage_cost_micro(input_tokens: int, output_tokens: int, prices: PriceTable) -> int:
    micro = measured_cost_micro(ProviderUsage(input_tokens, 0, output_tokens, 0), prices)
    assert micro is not None
    return micro


def bounds(sends: Sequence[dict[str, Any]], prices: PriceTable, max_attempts: int) -> dict[str, Any]:
    input_bounds = [input_token_upper_bound([{"content": r["system"]}, {"content": r["user"]}]) for r in sends]
    per_request = [request_cost_upper_bound_micro(b, r["max_output_tokens"], prices) for b, r in zip(input_bounds, sends)]
    total = sum(per_request)
    return {
        "input_bound_tokens": {
            "min": min(input_bounds),
            "median": sorted(input_bounds)[len(input_bounds) // 2],
            "max": max(input_bounds),
            "total": sum(input_bounds),
        },
        "one_attempt_usd": total / MICRO,
        "max_request_usd": max(per_request) / MICRO,
        f"{max_attempts}_attempts_usd": max_attempts * total / MICRO,
        "_input_bounds": input_bounds,
        "_per_request_micro": per_request,
    }


def expected(
    tokens: Sequence[dict[str, int]], prices: PriceTable, output_tokens: int, retry_rate: float, scenario: str
) -> float:
    base = sum(usage_cost_micro(t[scenario], output_tokens, prices) for t in tokens) / MICRO
    return base * (1 + retry_rate)


def simulate(
    sends: Sequence[dict[str, Any]],
    per_request_micro: Sequence[int],
    tokens: Sequence[dict[str, int]],
    prices: PriceTable,
    ceiling: float,
    plan: Plan,
    output_tokens: int,
    scenario: str = "central",
) -> dict[str, Any]:
    """Replay the driver's dispatch rule with the real ledger.

    ``plan(i)`` lists the outcomes of question ``i``'s attempts in order. Each is ``answered`` (usage and
    measured cost), ``no_usage`` (a saved response without measured cost), ``unknown`` or ``rejected``. A question
    ends at its first ``answered`` or ``no_usage`` attempt, or when its list ends. Before each dispatch the replay
    asks ``SafetyLedger.can_reserve`` and stops the run when the answer is no.
    """
    events: list[dict[str, Any]] = []
    done = 0
    for i, (row, upper) in enumerate(zip(sends, per_request_micro)):
        for attempt, outcome in enumerate(plan(i), start=1):
            ledger = SafetyLedger.from_events(events, prices)
            if not ledger.can_reserve(upper / MICRO, ceiling, ceiling_simulated=prices.simulated):
                return _replay_result(ledger, done, i, len(sends))
            attempt_id = f"q{i}-a{attempt}"
            events.append(
                {
                    "event": "dispatch_started",
                    "attempt_id": attempt_id,
                    "question_id": row["question_id"],
                    "cost_upper_bound": upper / MICRO,
                }
            )
            if outcome == "answered":
                cost = usage_cost_micro(tokens[i][scenario], output_tokens, prices)
                usage = {
                    "input_tokens": tokens[i][scenario],
                    "cached_input_tokens": 0,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": 0,
                }
                # An answered attempt in this replay is a Standard-tier response. Without the returned tier the
                # ledger reserves it at its bound whenever the price table names a tier.
                events.append(
                    {
                        "event": "response_saved",
                        "attempt_id": attempt_id,
                        "measured_cost": cost / MICRO,
                        "usage": usage,
                        "returned_service_tier": prices.service_tier,
                    }
                )
            elif outcome == "no_usage":
                events.append({"event": "response_saved", "attempt_id": attempt_id, "measured_cost": None, "usage": None})
            elif outcome == "unknown":
                events.append({"event": "outcome_unknown", "attempt_id": attempt_id})
            elif outcome == "rejected":
                events.append({"event": "attempt_failed", "attempt_id": attempt_id, "outcome": "rejected"})
            else:
                raise ValueError(f"unknown outcome {outcome!r}")
        done += 1
    return _replay_result(SafetyLedger.from_events(events, prices), done, None, len(sends))


def _replay_result(ledger: SafetyLedger, done: int, stopped_at: int | None, total: int) -> dict[str, Any]:
    return {
        "completed_questions": done,
        "of": total,
        "stopped_at_question_index": stopped_at,
        "measured_usd": ledger.measured,
        "reserved_usd": ledger.reserved,
        "committed_upper_usd": ledger.committed_upper,
    }


def ceiling_replays(
    sends: Sequence[dict[str, Any]],
    per_request_micro: Sequence[int],
    tokens: Sequence[dict[str, int]],
    prices: PriceTable,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out: list[dict[str, Any]] = []

    def run(name: str, plan: Plan, output_tokens: int | None = None, scenario: str = "central") -> None:
        result = simulate(
            sends, per_request_micro, tokens, prices, args.ceiling, plan, output_tokens or args.output_tokens, scenario
        )
        out.append({"scenario": name, **result})

    run("all answered (central input, base output tokens)", lambda i: ["answered"])
    run(
        "all answered (high input, high output tokens)",
        lambda i: ["answered"],
        output_tokens=args.output_tokens_high,
        scenario="high",
    )
    run("every response saved without usage (reserved at bound)", lambda i: ["no_usage"])
    run("one rejected attempt per question, then answered", lambda i: ["rejected", "answered"])
    run("two rejected attempts per question, then answered", lambda i: ["rejected", "rejected", "answered"])
    run(
        "two rejected attempts per question, low output tokens",
        lambda i: ["rejected", "rejected", "answered"],
        output_tokens=args.output_tokens_low,
    )
    run("one unknown attempt per question, reconciled, then answered", lambda i: ["unknown", "answered"])
    run("two unknown attempts per question, then answered", lambda i: ["unknown", "unknown", "answered"])

    thresholds: dict[str, int] = {}
    for extra in (1, 2):
        for where in ("first", "last"):
            best = 0
            for m in range(len(sends) + 1):
                chosen = set(range(m)) if where == "first" else set(range(len(sends) - m, len(sends)))

                def plan(i: int, chosen: set[int] = chosen, extra: int = extra) -> list[str]:
                    return ["unknown"] * extra + ["answered"] if i in chosen else ["answered"]

                if simulate(sends, per_request_micro, tokens, prices, args.ceiling, plan, args.output_tokens)[
                    "stopped_at_question_index"
                ] is None:
                    best = m
            thresholds[f"max questions with {extra} extra reserved attempt(s), placed {where}"] = best
    return {"ceiling_usd": args.ceiling, "scenarios": out, "completion_thresholds": thresholds}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run-dir", default=DEFAULT_DRY_RUN, help=f"dry-run directory (default: {DEFAULT_DRY_RUN})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--subset-ids", help="JSON list of question ids; only those requests are costed")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--ceiling", type=float, default=2.0, help="safety ceiling in USD for the replay")
    parser.add_argument("--output-tokens", type=int, default=15, help="assumed output tokens per request (central)")
    parser.add_argument("--output-tokens-low", type=int, default=5)
    parser.add_argument("--output-tokens-high", type=int, default=40)
    parser.add_argument("--retry-rate", type=float, default=0.0, help="assumed extra billed attempts per request")
    parser.add_argument("--brief-output-tokens", type=int, default=20, help="study brief output-token assumption")
    parser.add_argument("--brief-retry-rate", type=float, default=0.10, help="study brief billed-retry assumption")
    parser.add_argument("--chars-per-token-low", type=float, default=4.5, help="non-Han characters per token, low case")
    parser.add_argument("--chars-per-token-high", type=float, default=3.0, help="non-Han characters per token, high case")
    parser.add_argument("--cjk-tokens-low", type=float, default=0.6, help="tokens per Han character, low case")
    parser.add_argument("--cjk-tokens-high", type=float, default=1.0, help="tokens per Han character, high case")
    parser.add_argument("--framing-tokens", type=int, default=20, help="tokens added to the high case for chat framing")
    parser.add_argument("--fast-multiplier", type=float, help="optional Fast-tier rate multiplier scenario, for example 1.7")
    args = parser.parse_args(argv)

    dry_run = require_dir(resolve(args.dry_run_dir), "dry-run directory", "run_pilot_live.py dry-run writes it")
    rows = read_jsonl(dry_run / "prepared_requests.jsonl", "prepared_requests.jsonl")
    summary = read_json(dry_run / "dry_run_summary.json", "dry_run_summary.json")
    out = prepare_out(resolve(args.out))

    subset: set[str] | None = None
    if args.subset_ids:
        subset = set(read_json(resolve(args.subset_ids), "subset id list"))
        unknown = subset - {r["question_id"] for r in rows}
        if unknown:
            fail(f"--subset-ids names {len(unknown)} question(s) absent from the dry run", code=EXIT_FAILED_CHECK)
        rows = [r for r in rows if r["question_id"] in subset]
    sends = [r for r in rows if r["action"] == "send"]
    if not sends:
        fail("no sendable request in the selection", code=EXIT_FAILED_CHECK)

    prices = load_prices(summary)
    checks: dict[str, bool] = {}
    b = bounds(sends, prices, args.max_attempts)
    checks["input bounds equal the saved input_token_upper_bound"] = b["_input_bounds"] == [
        r["input_token_upper_bound"] for r in sends
    ]
    checks["per-request cost bounds equal the saved cost_upper_bound"] = [
        round(m / MICRO, 6) for m in b["_per_request_micro"]
    ] == [round(r["cost_upper_bound"], 6) for r in sends]
    if subset is None:
        totals = summary["cost_upper_bound"]
        checks["one-attempt bound equals dry_run_summary.json"] = abs(b["one_attempt_usd"] - totals["per_attempt_total"]) < 5e-7
        checks["N-attempt bound equals dry_run_summary.json"] = (
            abs(b[f"{args.max_attempts}_attempts_usd"] - totals["worst_case_all_attempts"]) < 5e-7
            if totals["max_attempts"] == args.max_attempts
            else True
        )
        checks["input bound total equals dry_run_summary.json"] = (
            b["input_bound_tokens"]["total"] == summary["input_token_upper_bound"]["total"]
        )
        limit = summary["input_token_upper_bound"]["limit"]
        checks[f"no input bound above the limit ({limit})"] = b["input_bound_tokens"]["max"] <= limit

    tokens = [token_scenarios(r, args) for r in sends]
    heuristic_saved = [r["token_estimate"]["estimate"] for r in sends]
    checks["heuristic tokens equal the saved token_estimate"] = [t["central"] for t in tokens] == heuristic_saved

    def estimate_block(p: PriceTable) -> dict[str, Any]:
        return {
            "central_usd": expected(tokens, p, args.output_tokens, args.retry_rate, "central"),
            "range_low_usd": expected(tokens, p, args.output_tokens_low, args.retry_rate, "low"),
            "range_high_usd": expected(tokens, p, args.output_tokens_high, args.retry_rate, "high"),
            "study_brief_assumptions_usd": expected(tokens, p, args.brief_output_tokens, args.brief_retry_rate, "central"),
        }

    result: dict[str, Any] = {
        "kind": "offline cost estimate; a bound is a fact about the requests, an expected cost is an estimate",
        "requests": {"questions": len(rows), "sent": len(sends), "skipped": len(rows) - len(sends)},
        "prices": prices.as_dict(),
        "assumptions": {
            "output_tokens": {"low": args.output_tokens_low, "central": args.output_tokens, "high": args.output_tokens_high},
            "retry_rate": args.retry_rate,
            "study_brief": {"output_tokens": args.brief_output_tokens, "retry_rate": args.brief_retry_rate},
            "input_tokens": {
                "central": "answer_prompt.estimate_tokens heuristic, not a tokenizer count",
                "chars_per_token_low": args.chars_per_token_low,
                "chars_per_token_high": args.chars_per_token_high,
                "han_tokens_low": args.cjk_tokens_low,
                "han_tokens_high": args.cjk_tokens_high,
                "framing_tokens_added_to_high": args.framing_tokens,
            },
            "max_attempts": args.max_attempts,
            "sensitivity_note": "the low and high input variants are a sensitivity check, not an envelope",
        },
        "heuristic_input_tokens": {name: sum(t[name] for t in tokens) for name in ("low", "central", "high")},
        "bounds": {k: v for k, v in b.items() if not k.startswith("_")},
        "expected": estimate_block(prices),
        "fast_tier": None,
        "ceiling_replay": ceiling_replays(sends, b["_per_request_micro"], tokens, prices, args),
        "checks": checks,
    }
    if args.fast_multiplier:
        fast = scaled(prices, args.fast_multiplier)
        fb = bounds(sends, fast, args.max_attempts)
        result["fast_tier"] = {
            "multiplier": args.fast_multiplier,
            "one_attempt_usd": fb["one_attempt_usd"],
            f"{args.max_attempts}_attempts_usd": fb[f"{args.max_attempts}_attempts_usd"],
            "expected": estimate_block(fast),
        }

    write_json(out / "cost_and_ceiling.json", result)

    print(f"requests: {len(rows)} ({len(sends)} sent); prices {prices.input_per_million}/"
          f"{prices.cached_input_per_million}/{prices.output_per_million} per 1M (input/cached/output)")
    print(f"input bound tokens: {result['bounds']['input_bound_tokens']}")
    print(f"one-attempt bound:  ${result['bounds']['one_attempt_usd']:.6f}")
    print(f"{args.max_attempts}-attempt bound:   ${result['bounds'][f'{args.max_attempts}_attempts_usd']:.6f}")
    print(f"heuristic input tokens (low/central/high): {result['heuristic_input_tokens']}")
    e = result["expected"]
    print(f"expected (ESTIMATE): ${e['central_usd']:.6f}, range "
          f"${e['range_low_usd']:.6f} to ${e['range_high_usd']:.6f}; study brief assumptions ${e['study_brief_assumptions_usd']:.6f}")
    if result["fast_tier"]:
        ft = result["fast_tier"]
        print(f"fast tier x{ft['multiplier']}: one attempt ${ft['one_attempt_usd']:.6f}, "
              f"{args.max_attempts} attempts ${ft[f'{args.max_attempts}_attempts_usd']:.6f}, "
              f"expected ${ft['expected']['central_usd']:.6f}")
    print(f"ceiling replay at ${args.ceiling:.2f}:")
    for s in result["ceiling_replay"]["scenarios"]:
        stop = "completes" if s["stopped_at_question_index"] is None else f"stops after {s['completed_questions']}"
        print(f"  {s['scenario']:62s} {stop:20s} committed ${s['committed_upper_usd']:.6f}")
    for name, value in result["ceiling_replay"]["completion_thresholds"].items():
        print(f"  {name}: {value}")
    ok = True
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    return 0 if ok else EXIT_FAILED_CHECK


if __name__ == "__main__":
    raise SystemExit(main())
