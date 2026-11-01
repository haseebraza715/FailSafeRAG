"""Run the visual repair (``invoke_vlm``) on a frozen pilot, after a no-recovery run on the same pilot.

Design: docs/research/visual-repair-design.md. Every question gets one repair request: the no-recovery run's exact
user message (or, for a question that run did not send, the no-text message), plus images of the pages chosen by
``retrieved-then-gaps-v1`` from the runtime manifest and the retrieved evidence. The repaired answer replaces the
initial one. ``prepare`` and ``run`` never open the evaluation manifest, clean text or scores; ``score`` does, after
the run, and only to score.

Commands::

    python scripts/experiments/run_visual_repair.py prepare --pilot-id ohr_dev_v2 \\
        --baseline-run-dir results/development/<no-recovery run> --run-dir results/development/<run_id> \\
        [--questions-file <file with one question id per line>]
    FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY python scripts/experiments/run_visual_repair.py run \\
        --run-dir results/development/<run_id> --cap 2.50
    python scripts/experiments/run_visual_repair.py export --run-dir results/development/<run_id>
    python scripts/experiments/run_visual_repair.py score --run-dir results/development/<run_id>

``run`` resumes. A question with a saved response is never sent again. Before each request the runner checks that
measured cost plus reserved cost plus this request's upper bound stays within ``--cap``. An attempt whose outcome is
unknown keeps its upper bound reserved for good. It blocks its question, except under the resend rule
``one-resend-connection-lost-v1`` when the run's identity names it: an attempt that failed with kind
``connection_lost`` may be sent once more. A second unknown outcome blocks the question. No other unknown outcome
(a timeout, a 5xx, a redirect) is ever resent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from faar.answer_prompt import get_template, parse_reply
from faar.live_contract import ProviderError, ProviderRequest
from faar.live_runner import build_live_provider, load_provider_config, refuse_redirecting_environment
from faar.request_budget import input_token_upper_bound, measured_cost, request_cost_upper_bound
from faar.visual_repair import (
    DEFAULT_BUDGET,
    FORBIDDEN_PATH_PARTS,
    GATE_ID,
    IMAGE_DETAIL,
    IMAGE_NOTE,
    IMAGE_TOKEN_BOUND,
    MODEL_JPEG_QUALITY,
    MODEL_LONG_SIDE,
    NO_TEXT_USER_PREFIX,
    POLICY_ID,
    RENDER_DPI,
    PageRenderer,
    RepairError,
    check_runtime_path,
    repair_messages,
    repair_text,
    select_pages,
)

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_CONFIG = ROOT / ".local" / "work" / "provider-config.gpt-6.1-terra.json"
PDF_ARCHIVE = ROOT / "data" / "ohr_bench_raw" / "pdfs.zip"
LIVE_ENV_NAME, LIVE_ENV_VALUE = "FAAR_ALLOW_LIVE_REQUESTS", "I_UNDERSTAND_THIS_SPENDS_MONEY"
MAX_CONSECUTIVE_FAILURES = 3
RESEND_RULES = ("none", "one-resend-connection-lost-v1")
RESENDABLE_KIND = "connection_lost"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _read_bytes(path: Path) -> bytes:
    check_runtime_path(path)
    return path.read_bytes()


def _read_json(path: Path) -> Any:
    return json.loads(_read_bytes(path).decode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _read_bytes(path).decode("utf-8").splitlines() if line.strip()]


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- prepare


def load_baseline(root: Path, pilot_id: str, baseline_run_dir: Path) -> dict[str, Any]:
    """The frozen runtime manifest and a valid no-recovery run on it, checked against each other."""
    runtime_path = root / "results" / "pilots" / pilot_id / "runtime_manifest.json"
    runtime_bytes = _read_bytes(runtime_path)
    runtime = json.loads(runtime_bytes)
    if runtime.get("pilot_id") != pilot_id:
        raise RepairError(f"{runtime_path} is for pilot {runtime.get('pilot_id')!r}")
    baseline_config = _read_json(baseline_run_dir / "run_config.json")
    baseline_identity = baseline_config.get("identity", baseline_config)
    baseline_summary = _read_json(baseline_run_dir / "run_summary.json")
    if baseline_summary.get("pilot_id", pilot_id) != pilot_id:
        raise RepairError(f"{baseline_run_dir.name} is not a run on {pilot_id}")
    manifest_sha = _sha(runtime_bytes)
    if json.dumps(baseline_identity, sort_keys=True).find(manifest_sha) < 0:
        raise RepairError(f"{baseline_run_dir.name} does not record the runtime manifest of {pilot_id} ({manifest_sha[:12]})")
    if baseline_summary.get("valid_baseline") is not True:
        raise RepairError(f"{baseline_run_dir.name} is not a valid no-recovery run (valid_baseline is not true)")
    requests_bytes = _read_bytes(baseline_run_dir / "requests.jsonl")
    requests = {r["question_id"]: r for r in (json.loads(line) for line in requests_bytes.decode("utf-8").splitlines() if line.strip())}
    template_ids = {r["template_id"] for r in requests.values()}
    if len(template_ids) != 1:
        raise RepairError(f"{baseline_run_dir.name} mixes templates {sorted(template_ids)}")
    for q in runtime["questions"]:
        base = requests.get(q["question_id"])
        if base is None:
            raise RepairError(f"{baseline_run_dir.name} has no request for question {q['question_id']}")
        if base["question"] != q["question"] or base["doc_id"] != q["doc_id"]:
            raise RepairError(f"{q['question_id']}: the no-recovery request differs from the runtime manifest")
    return {"runtime": runtime, "runtime_sha256": manifest_sha, "requests": requests, "requests_sha256": _sha(requests_bytes),
            "template": get_template(template_ids.pop())}


def write_identity(run_dir: Path, identity: dict[str, Any], records: list[dict[str, Any]] | None) -> None:
    """Write ``run_config.json`` once and, unless ``records`` is None, ``requests.jsonl``; refuse a changed identity."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != identity:
            diff = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
            raise RepairError(f"{config_path} exists with a different identity ({', '.join(diff)}); a changed run needs a new run_id")
    else:
        config_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if records is not None:
        (run_dir / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def prepare(
    *,
    root: Path,
    pilot_id: str,
    baseline_run_dir: Path,
    run_dir: Path,
    run_id: str,
    budget: int,
    resend_rule: str,
    provider_config: Path,
    archive: Path,
    questions_file: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write ``run_config.json`` and ``requests.jsonl``. Sends nothing. Refuses a changed identity in an existing run."""
    if resend_rule not in RESEND_RULES:
        raise RepairError(f"unknown resend rule {resend_rule!r}")
    loaded = load_baseline(root, pilot_id, baseline_run_dir)
    runtime, manifest_sha, baseline, template = loaded["runtime"], loaded["runtime_sha256"], loaded["requests"], loaded["template"]
    questions = None
    if questions_file is not None:
        questions_bytes = _read_bytes(questions_file)
        questions = {line.strip() for line in questions_bytes.decode("utf-8").splitlines() if line.strip()}
        if not questions:
            raise RepairError("questions file is empty")
        unknown = questions - {q["question_id"] for q in runtime["questions"]}
        if unknown:
            raise RepairError(f"{len(unknown)} question ids are not in {pilot_id}")
    config = load_provider_config(provider_config)
    documents = {d["doc_id"]: d for d in runtime["documents"]}
    renderer = PageRenderer(archive, documents)

    records = []
    for q in runtime["questions"]:
        qid = q["question_id"]
        if questions is not None and qid not in questions:
            continue
        base = baseline[qid]
        doc = documents[q["doc_id"]]
        system, user, retrieved = repair_text(base, template)
        pages = select_pages(doc["pages"], retrieved, budget)
        images = [renderer.page(q["doc_id"], p) for p in pages]
        text_messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        bound = input_token_upper_bound(text_messages) + IMAGE_TOKEN_BOUND * len(images)
        records.append({
            "question_id": qid, "doc_id": q["doc_id"], "action": "send",
            "initial_action": base["action"], "initial_skip_reason": base.get("skip_reason"),
            "retrieved_pages": retrieved, "selected_pages": pages,
            "images": [{"page_idx": im.page_idx, "sha256": im.sha256, "size": list(im.size), "bytes": len(im.data)} for im in images],
            "system": system, "user": user,
            "prompt_sha256": _canonical({"messages": text_messages, "images": [im.sha256 for im in images], "detail": IMAGE_DETAIL}),
            "input_token_upper_bound": bound, "max_output_tokens": config.max_output_tokens,
            "cost_upper_bound": request_cost_upper_bound(bound, config.max_output_tokens, config.prices),
        })

    code_files = [Path(__file__).resolve()] + [ROOT / "src" / "faar" / n for n in
                                               ("visual_repair.py", "answer_prompt.py", "answer_providers.py", "request_budget.py")]
    identity = {
        "run_id": run_id, "kind": "development_pilot", "repair": "invoke_vlm", "pilot_id": pilot_id,
        "runtime_manifest_sha256": manifest_sha,
        "baseline_run": baseline_run_dir.name, "baseline_requests_sha256": loaded["requests_sha256"],
        "template_id": template.template_id, "template_sha256": template.sha256,
        "image_note": IMAGE_NOTE, "no_text_user_template": NO_TEXT_USER_PREFIX + template.closing_line,
        "page_policy": POLICY_ID, "image_budget": budget, "gate": GATE_ID,
        "render": {"dpi": RENDER_DPI, "long_side_max": MODEL_LONG_SIDE, "format": "JPEG", "quality": MODEL_JPEG_QUALITY,
                   "detail": IMAGE_DETAIL, "archive_sha256": _sha(_read_bytes(archive))},
        "image_token_bound": IMAGE_TOKEN_BOUND,
        "provider_config_sha256": _sha(_read_bytes(provider_config)),
        "resend_rule": resend_rule, "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
        "forbidden_path_parts": list(FORBIDDEN_PATH_PARTS),
        "requests_sha256": _canonical(records),
        "code": {_rel(p, ROOT): _sha(p.read_bytes()) for p in code_files},
    }
    if questions is not None:
        identity["questions_file_sha256"] = _sha(questions_bytes)
        identity["question_count"] = len(questions)
    write_identity(run_dir, identity, records)
    return records, identity


# --------------------------------------------------------------------------- ledger


def _events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "attempts.jsonl"
    return _read_jsonl(path) if path.exists() else []


def _append(run_dir: Path, event: dict[str, Any]) -> None:
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ledger(events: list[dict[str, Any]], resend_rule: str) -> dict[str, Any]:
    """Measured and reserved cost, saved and blocked questions, and per-question unknown outcomes.

    Every attempt that may have reached the provider without a saved response stays reserved at its upper bound,
    including one that the resend rule allows to be sent again.
    """
    measured, reserved = 0.0, 0.0
    saved: set[str] = set()
    unknown: dict[str, list[str]] = {}
    open_attempts: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["event"] == "dispatch":
            open_attempts[e["attempt_id"]] = e
        elif e["event"] in ("saved", "failed"):
            start = open_attempts.pop(e["attempt_id"])
            if e["event"] == "saved":
                saved.add(e["question_id"])
                if e.get("measured_cost") is None:
                    reserved += start["cost_upper_bound"]
                else:
                    measured += e["measured_cost"]
            elif e.get("outcome") != "not_sent":
                reserved += start["cost_upper_bound"]
                unknown.setdefault(e["question_id"], []).append(e.get("kind") or "unknown")
    for start in open_attempts.values():  # dispatched, then the process died: outcome unknown
        reserved += start["cost_upper_bound"]
        unknown.setdefault(start["question_id"], []).append("no_terminal_event")
    blocked = set()
    for qid, kinds in unknown.items():
        if qid in saved:
            continue
        resendable = resend_rule == "one-resend-connection-lost-v1" and kinds == [RESENDABLE_KIND]
        if not resendable:
            blocked.add(qid)
    return {"measured": measured, "reserved": reserved, "saved": saved, "blocked": blocked, "unknown": unknown}


# --------------------------------------------------------------------------- run


def run(
    *,
    root: Path,
    run_dir: Path,
    cap: float,
    provider: Any,
    config: Any,
    archive: Path,
) -> dict[str, Any]:
    identity = _read_json(run_dir / "run_config.json")
    records = _read_jsonl(run_dir / "requests.jsonl")
    if _canonical(records) != identity["requests_sha256"]:
        raise RepairError("requests.jsonl differs from the run identity")
    runtime = _read_json(root / "results" / "pilots" / identity["pilot_id"] / "runtime_manifest.json")
    renderer = PageRenderer(archive, {d["doc_id"]: d for d in runtime["documents"]})
    invocation = uuid.uuid4().hex
    _append(run_dir, {"event": "invocation_started", "invocation": invocation, "cap": cap, "run_id": identity["run_id"]})
    failures, end_reason = 0, "complete"
    for record in records:
        state = ledger(_events(run_dir), identity["resend_rule"])
        qid = record["question_id"]
        if record.get("action", "send") != "send" or qid in state["saved"] or qid in state["blocked"]:
            continue
        if state["measured"] + state["reserved"] + record["cost_upper_bound"] > cap:
            end_reason = "cap_reached"
            break
        if "selected_pages" in record:
            images = [renderer.page(record["doc_id"], p) for p in record["selected_pages"]]
            messages = repair_messages(record, images)
        else:  # a text-only repair: the prepared system and user messages, unchanged
            messages = ({"role": "system", "content": record["system"]}, {"role": "user", "content": record["user"]})
        attempt_id = f"{qid[:8]}-{uuid.uuid4().hex[:8]}"
        resend_of = state["unknown"].get(qid)
        request = ProviderRequest(qid, attempt_id, messages, dict(config.params), config.max_output_tokens, config.timeout_seconds)
        _append(run_dir, {"event": "dispatch", "question_id": qid, "attempt_id": attempt_id,
                          "cost_upper_bound": record["cost_upper_bound"], "resend": bool(resend_of)})
        started = time.monotonic()
        try:
            response = provider.send(request)
        except ProviderError as exc:
            _append(run_dir, {"event": "failed", "question_id": qid, "attempt_id": attempt_id, "kind": exc.kind,
                              "http_status": exc.http_status, "outcome": exc.outcome, "message": str(exc)[:500]})
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                end_reason = "consecutive_failures"
                break
            continue
        failures = 0
        latency = int((time.monotonic() - started) * 1000)
        cost = measured_cost(response.usage, config.prices)
        parsed = parse_reply(response.text, response.finish_reason, response.refusal)
        payload = {
            "question_id": qid, "attempt_id": attempt_id, "returned_model": response.returned_model,
            "returned_service_tier": response.returned_service_tier, "response_id": response.response_id,
            "finish_reason": response.finish_reason, "usage": response.usage.as_dict(), "measured_cost": cost,
            "latency_ms": latency, **parsed, "raw": response.raw,
        }
        out = run_dir / "responses" / f"{attempt_id}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _append(run_dir, {"event": "saved", "question_id": qid, "attempt_id": attempt_id, "measured_cost": cost,
                          "response_sha256": _sha(out.read_bytes()), "returned_model": response.returned_model,
                          "returned_service_tier": response.returned_service_tier})
        if response.returned_model != config.model or response.returned_service_tier != "default":
            end_reason = "safety_stop"
            break
    state = ledger(_events(run_dir), identity["resend_rule"])
    summary = {"end_reason": end_reason, "measured": round(state["measured"], 6), "reserved": round(state["reserved"], 6),
               "saved": len(state["saved"]), "blocked": sorted(state["blocked"])}
    _append(run_dir, {"event": "invocation_ended", "invocation": invocation, **summary})
    return summary


# --------------------------------------------------------------------------- export and score


def export(run_dir: Path) -> list[dict[str, Any]]:
    """Write ``predictions.jsonl``: one row per question, answered or execution_failed. Reads no evaluation file."""
    identity = _read_json(run_dir / "run_config.json")
    records = _read_jsonl(run_dir / "requests.jsonl")
    events = _events(run_dir)
    state = ledger(events, identity["resend_rule"])
    saved = {e["question_id"]: e for e in events if e["event"] == "saved"}
    rows = []
    for r in records:
        qid = r["question_id"]
        attempts = [e["attempt_id"] for e in events if e["event"] == "dispatch" and e["question_id"] == qid]
        row = {"question_id": qid, "doc_id": r["doc_id"], "attempt_ids": attempts, "images": len(r.get("selected_pages", [])),
               "unknown_outcomes": len(state["unknown"].get(qid, []))}
        if r.get("action", "send") != "send":
            row.update(status="skipped", skip_reason=r.get("skip_reason"), answer="", abstained=False, measured_cost=0.0, usage=None)
        elif qid in saved:
            resp = _read_json(run_dir / "responses" / f"{saved[qid]['attempt_id']}.json")
            row.update(status="answered", answer=resp["answer"], abstained=resp["abstained"],
                       measured_cost=resp["measured_cost"], usage=resp["usage"])
        else:
            row.update(status="execution_failed", answer="", abstained=False, measured_cost=None, usage=None,
                       blocked=qid in state["blocked"])
        rows.append(row)
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    summary = {
        "run_id": identity["run_id"], "questions": len(rows),
        "answered": sum(x["status"] == "answered" for x in rows),
        "execution_failed": sum(x["status"] == "execution_failed" for x in rows),
        "blocked": sorted(state["blocked"]),
        "questions_with_unknown_outcomes": len(state["unknown"]),
        "resent": sum(1 for e in events if e["event"] == "dispatch" and e.get("resend")),
        "abstained": sum(x["abstained"] for x in rows),
        "images": sum(x["images"] for x in rows),
        "measured_cost_usd": round(state["measured"], 6), "reserved_cost_usd": round(state["reserved"], 6),
        "returned_models": sorted({e.get("returned_model") for e in events if e["event"] == "saved"}),
        "returned_service_tiers": sorted({e.get("returned_service_tier") for e in events if e["event"] == "saved"}),
        "skipped": sum(x["status"] == "skipped" for x in rows),
        "complete": all(x["status"] in ("answered", "skipped") for x in rows),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows


def score(root: Path, run_dir: Path) -> dict[str, Any]:
    """Join the exported predictions to the evaluation manifest. Runs after the repair, never inside it."""
    from faar.ohr_scoring import exact_match_score, f1_score

    identity = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / "results" / "pilots" / identity["pilot_id"] / "evaluation_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for p in (json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()):
        gold = evaluation["questions"][p["question_id"]]["answers"]
        rows.append({"question_id": p["question_id"], "status": p["status"], "answer": p["answer"],
                     "em": exact_match_score(p["answer"], gold), "f1": f1_score(p["answer"], gold)})
    (run_dir / "scores.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    n = len(rows)
    summary = {"run_id": identity["run_id"], "questions": n, "em": sum(x["em"] for x in rows) / n, "f1": sum(x["f1"] for x in rows) / n,
               "execution_failed": sum(x["status"] == "execution_failed" for x in rows)}
    (run_dir / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--pilot-id", required=True)
    p.add_argument("--baseline-run-dir", type=Path, required=True)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--resend-rule", choices=RESEND_RULES, default="none")
    p.add_argument("--run-id")
    p.add_argument("--questions-file", type=Path)
    r = sub.add_parser("run")
    r.add_argument("--cap", type=float, required=True, help="hard ceiling in USD for this run, counting reserved cost")
    sub.add_parser("export")
    sub.add_parser("score")
    for s in sub.choices.values():
        s.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    if args.command == "prepare":
        baseline = args.baseline_run_dir if args.baseline_run_dir.is_absolute() else ROOT / args.baseline_run_dir
        questions_file = args.questions_file
        if questions_file is not None and not questions_file.is_absolute():
            questions_file = ROOT / questions_file
        records, identity = prepare(root=ROOT, pilot_id=args.pilot_id, baseline_run_dir=baseline, run_dir=run_dir,
                                    run_id=args.run_id or run_dir.name, budget=args.budget, resend_rule=args.resend_rule,
                                    provider_config=PROVIDER_CONFIG, archive=PDF_ARCHIVE, questions_file=questions_file)
        print(json.dumps({"run_dir": _rel(run_dir, ROOT), "requests": len(records),
                          "images": sum(len(x["selected_pages"]) for x in records),
                          "image_bytes": sum(i["bytes"] for x in records for i in x["images"]),
                          "cost_upper_bound_usd": round(sum(x["cost_upper_bound"] for x in records), 4)}, indent=2))
    elif args.command == "run":
        if os.environ.get(LIVE_ENV_NAME) != LIVE_ENV_VALUE:
            raise SystemExit(f"live requests need {LIVE_ENV_NAME}={LIVE_ENV_VALUE}; nothing was sent")
        refuse_redirecting_environment(os.environ)
        identity = _read_json(run_dir / "run_config.json")
        if _sha(_read_bytes(PROVIDER_CONFIG)) != identity["provider_config_sha256"]:
            raise SystemExit("the provider config differs from the run identity; nothing was sent")
        config = load_provider_config(PROVIDER_CONFIG)
        provider = build_live_provider(config, os.environ)
        print(json.dumps(run(root=ROOT, run_dir=run_dir, cap=args.cap, provider=provider, config=config, archive=PDF_ARCHIVE)))
    elif args.command == "export":
        rows = export(run_dir)
        print(json.dumps({"questions": len(rows), "answered": sum(x["status"] == "answered" for x in rows)}))
    else:
        print(json.dumps(score(ROOT, run_dir), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
