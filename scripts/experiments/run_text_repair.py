"""Run a text repair (``retry_retrieval`` or ``correct_text``) after a no-recovery run on the same pilot.

    PYTHONPATH=src .local/venv-prebaseline/bin/python scripts/experiments/run_text_repair.py prepare \\
        --repair retry_retrieval --rule exclude-seen-v1 --pilot-id ohr_long_dev_v1 \\
        --baseline-run-dir results/development/<no-recovery run> --run-dir results/development/<run_id> \\
        [--questions-file <file with one question id per line>]
    ... prepare --repair correct_text --corrector symspell-en-82765 ...
    FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY ... run_visual_repair.py run --run-dir <run_dir> --cap 0.50
    ... run_visual_repair.py export --run-dir <run_dir>

``prepare`` replays the no-recovery retrieval offline, checks it against the evidence that run sent, builds the new
evidence and writes the exact prompts. It sends nothing. ``run``, ``export`` and ``score`` are those of
``run_visual_repair.py``: the same ledger, cap, resend rule and boundary checks. A question the repair cannot serve
(no text evidence, or nothing new to retrieve) is a ``skip`` record with its reason; the policy that uses the repair
then keeps the initial answer for it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import run_visual_repair as rvr

from faar.answer_prompt import build_prompt
from faar.live_runner import load_provider_config
from faar.request_budget import input_token_upper_bound, request_cost_upper_bound
from faar.text_repair import (
    CORRECTORS,
    EXPANSION_RULES,
    blocks_from_hits,
    corrected_blocks,
    corrector,
    replay_retrieval,
    retry_hits,
)

ROOT = Path(__file__).resolve().parents[2]
REPAIRS = ("retry_retrieval", "correct_text")


def prepare(
    *,
    root: Path,
    repair: str,
    method: str,
    pilot_id: str,
    baseline_run_dir: Path,
    run_dir: Path,
    run_id: str,
    resend_rule: str,
    provider_config: Path,
    questions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if repair not in REPAIRS:
        raise rvr.RepairError(f"unknown repair {repair!r}; known: {REPAIRS}")
    if resend_rule not in rvr.RESEND_RULES:
        raise rvr.RepairError(f"unknown resend rule {resend_rule!r}")
    loaded = rvr.load_baseline(root, pilot_id, baseline_run_dir)
    template = loaded["template"]
    config = load_provider_config(provider_config)
    replay = replay_retrieval(root, pilot_id, loaded["requests"])
    method_identity: dict[str, Any]
    if repair == "correct_text":
        correct, method_identity = corrector(method)
    elif method in EXPANSION_RULES:
        method_identity = {"name": method, "top_k": replay["settings"].top_k}
    else:
        raise rvr.RepairError(f"unknown expansion rule {method!r}; known: {EXPANSION_RULES}")
    if questions is not None:
        unknown = questions - set(loaded["requests"])
        if unknown:
            raise rvr.RepairError(f"{len(unknown)} question ids are not in {pilot_id}")

    records = []
    for q in loaded["runtime"]["questions"]:
        qid = q["question_id"]
        if questions is not None and qid not in questions:
            continue
        base = loaded["requests"][qid]
        record: dict[str, Any] = {"question_id": qid, "doc_id": q["doc_id"], "repair": repair, "action": "skip", "skip_reason": None}
        initial = replay["hits"][qid]
        if base["action"] != "send":
            record["skip_reason"] = "no_text_evidence"
            records.append(record)
            continue
        if repair == "retry_retrieval":
            retriever = replay["indexes"][q["doc_id"]].retriever()
            hits = retry_hits(method, q["question"], initial, retriever, replay["settings"].top_k)
            if not hits:
                record["skip_reason"] = "nothing_new_retrieved"
                records.append(record)
                continue
            blocks = blocks_from_hits(hits)
            record["changed_blocks"] = sum(h.chunk.chunk_id not in {i.chunk.chunk_id for i in initial} for h in hits)
        else:
            blocks, changed = corrected_blocks(blocks_from_hits(initial), correct)
            record["changed_blocks"] = changed
            if changed == 0:
                record["skip_reason"] = "correction_changed_nothing"
                records.append(record)
                continue
        payload = build_prompt(q["question"], blocks, template)
        messages = [{"role": "system", "content": payload.system}, {"role": "user", "content": payload.user}]
        bound = input_token_upper_bound(messages)
        record.update({
            "action": "send", "system": payload.system, "user": payload.user,
            "evidence": [{"rank": b.rank, "chunk_id": b.chunk_id, "page_idx": b.page_idx, "text_sha256": hashlib.sha256(b.text.encode("utf-8")).hexdigest()} for b in blocks],
            "prompt_sha256": rvr._canonical({"messages": messages}),
            "input_token_upper_bound": bound, "max_output_tokens": config.max_output_tokens,
            "cost_upper_bound": request_cost_upper_bound(bound, config.max_output_tokens, config.prices),
        })
        records.append(record)

    code_files = [Path(__file__).resolve(), rvr.Path(rvr.__file__).resolve()] + [
        ROOT / "src" / "faar" / n for n in ("text_repair.py", "answer_prompt.py", "answer_providers.py", "request_budget.py", "pilot_runner.py", "retrieval.py", "symspell.py")]
    identity = {
        "run_id": run_id, "kind": "development_pilot", "repair": repair, "method": method_identity, "pilot_id": pilot_id,
        "runtime_manifest_sha256": loaded["runtime_sha256"], "baseline_run": baseline_run_dir.name,
        "baseline_requests_sha256": loaded["requests_sha256"], "retrieval": replay["retrieval"],
        "template_id": template.template_id, "template_sha256": template.sha256,
        "provider_config_sha256": rvr._sha(rvr._read_bytes(provider_config)), "resend_rule": resend_rule,
        "max_consecutive_failures": rvr.MAX_CONSECUTIVE_FAILURES, "forbidden_path_parts": list(rvr.FORBIDDEN_PATH_PARTS),
        "questions": "all" if questions is None else sorted(questions),
        "requests_sha256": rvr._canonical(records),
        "code": {rvr._rel(p, ROOT): rvr._sha(p.read_bytes()) for p in code_files},
    }
    rvr.write_identity(run_dir, identity, records)
    return records, identity


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["prepare"])
    p.add_argument("--repair", choices=REPAIRS, required=True)
    p.add_argument("--rule", choices=EXPANSION_RULES, help="query expansion rule for retry_retrieval")
    p.add_argument("--corrector", choices=CORRECTORS, help="corrector for correct_text")
    p.add_argument("--pilot-id", required=True)
    p.add_argument("--baseline-run-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--run-id")
    p.add_argument("--resend-rule", choices=rvr.RESEND_RULES, default="one-resend-connection-lost-v1")
    p.add_argument("--questions-file", type=Path)
    args = p.parse_args()
    method = args.rule if args.repair == "retry_retrieval" else args.corrector
    if method is None:
        p.error("--rule is required for retry_retrieval and --corrector for correct_text")
    resolve = lambda path: path if path.is_absolute() else ROOT / path  # noqa: E731
    questions = None
    if args.questions_file:
        questions = {line.strip() for line in resolve(args.questions_file).read_text(encoding="utf-8").splitlines() if line.strip()}
    run_dir = resolve(args.run_dir)
    records, _ = prepare(root=ROOT, repair=args.repair, method=method, pilot_id=args.pilot_id, baseline_run_dir=resolve(args.baseline_run_dir),
                         run_dir=run_dir, run_id=args.run_id or run_dir.name, resend_rule=args.resend_rule,
                         provider_config=rvr.PROVIDER_CONFIG, questions=questions)
    sent = [r for r in records if r["action"] == "send"]
    reasons: dict[str, int] = {}
    for r in records:
        if r["skip_reason"]:
            reasons[r["skip_reason"]] = reasons.get(r["skip_reason"], 0) + 1
    print(json.dumps({"run_dir": rvr._rel(run_dir, ROOT), "records": len(records), "send": len(sent), "skip_reasons": reasons,
                      "cost_upper_bound_usd": round(sum(r["cost_upper_bound"] for r in sent), 4)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
