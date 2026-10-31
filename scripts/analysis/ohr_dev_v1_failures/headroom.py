"""Headroom runs for the gpt-6.1-terra baseline: the same 70 questions, model, settings and prompt template,
with a better evidence source in place of the retrieved MinerU text.

These are DIAGNOSTIC REFERENCE conditions. Each uses an evaluation-only input (clean ground-truth text or the
gold evidence-page labels) to bound what a repair could gain at most. None of them is a runtime method, and
nothing they produce may feed retrieval, gating, diagnosis or recovery.

Conditions
----------
clean_text         Clean OHR-Bench ground-truth text replaces MinerU; the same chunker, retriever and top-k pick
                   the evidence. Upper bound for a perfect text repair (``correct_text``) that could also change
                   what retrieval finds.
gold_page_text     All MinerU chunks of the gold evidence page(s), in page and chunk order, replace retrieval.
                   Upper bound for a perfect retrieval repair (``retry_retrieval``) on unchanged OCR.
gold_page_image    The baseline's own retrieved MinerU text (the exact baseline user message) plus the image of
                   every gold evidence page. Upper bound for the visual repair (``invoke_vlm``) with perfect page
                   selection, as study brief section 4 defines that repair (same text, images added).
gold_page_image_only
                   The gold evidence page image(s) and no OCR text. Shows what the page alone gives the model.

The prompt is the baseline template ``faar-answer-draft-v1``. Text conditions use ``build_prompt`` unchanged.
Image conditions append one paragraph to the system message (IMAGE_NOTE) and, where no text block exists, use
NO_TEXT_USER_TEMPLATE; that is the input-format difference the brief allows (section 5).

Commands (one output directory per condition under results/development/<run_id>/)::

    python headroom.py prepare --condition clean_text
    FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY python headroom.py run --condition clean_text --cap 0.30
    python headroom.py export --condition clean_text

``prepare`` sends nothing. ``run`` resumes: a question with a saved response is never sent again, and an attempt
whose outcome is unknown blocks its question until a person looks at it. Before each request the runner checks
that measured cost plus reserved cost plus this request's upper bound stays within ``--cap``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from common import BASELINE_DIR, OUT_DIR, PILOT_DIR, ROOT, read_json, read_jsonl
from faar.answer_prompt import ABSTENTION_TOKEN, SYSTEM_TEMPLATE, TEMPLATE_ID, TEMPLATE_SHA256, build_prompt, parse_reply
from faar.live_contract import EvidenceBlock, ProviderError, ProviderRequest
from faar.live_runner import build_live_provider, load_provider_config, refuse_redirecting_environment
from faar.ohr_scoring import exact_match_score, f1_score
from faar.pilot_runner import (
    DEFAULT_TEXT_POLICY,
    LoadedDocument,
    _DocumentIndex,
    _retrieve_question,
    engineering_retrieval_settings,
    load_runtime_manifest,
    retrieve_runtime_questions,
)
from faar.request_budget import input_token_upper_bound, measured_cost, request_cost_upper_bound

CONDITIONS = ("clean_text", "gold_page_text", "gold_page_image", "gold_page_image_only")
RUN_IDS = {c: f"2026-10-31-ohr-dev-v1-headroom-{c.replace('_', '-')}-terra-r1" for c in CONDITIONS}
PROVIDER_CONFIG = ROOT / ".local" / "work" / "provider-config.gpt-6.1-terra.json"
LIVE_ENV_NAME, LIVE_ENV_VALUE = "FAAR_ALLOW_LIVE_REQUESTS", "I_UNDERSTAND_THIS_SPENDS_MONEY"

IMAGE_DETAIL = "high"
# Upper bound per attached image, in input tokens. OpenAI's high-detail cost for an image whose long side is at
# most 2048 px is at most a few thousand tokens under every published formula; 8,000 is a deliberate overstatement
# so the cap check stays conservative. Measured usage replaces the bound once the response is saved.
IMAGE_TOKEN_BOUND = 8000
IMAGE_NOTE = (
    "\n\nThe user message also contains images of document pages. The images are evidence too. Read them together "
    "with any text blocks, and answer from what the page shows."
)
NO_TEXT_USER_TEMPLATE = (
    "Question:\n{question}\n\nNo text evidence is available. The evidence is the page image(s) below.\n\n"
    f"Answer the question using only the evidence. If the evidence lacks the answer, reply {ABSTENTION_TOKEN}."
)

MAX_CONSECUTIVE_FAILURES = 3


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def run_dir(condition: str) -> Path:
    return ROOT / "results" / "development" / RUN_IDS[condition]


# --------------------------------------------------------------------------- evidence per condition


def _clean_text_retrieval() -> dict[str, Any]:
    """Run the baseline retrieval code on clean ground-truth pages instead of MinerU pages."""
    manifest, _ = load_runtime_manifest(PILOT_DIR / "runtime_manifest.json", expected_pilot_id="ohr_dev_v1")
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    gt_path = {ev["doc_id"]: ev["gt_reference"]["path"] for ev in evaluation.values()}
    settings = engineering_retrieval_settings()
    out = {}
    indexes = {}
    for doc in manifest.documents:
        rows = read_json(ROOT / gt_path[doc.doc_id])
        texts: dict[int, str] = {}
        for row in rows:
            idx, text = row.get("page_idx"), row.get("text")
            if isinstance(idx, int) and isinstance(text, str):
                texts[idx] = texts[idx] + "\n" + text if idx in texts else text
        page_texts, status = {}, {}
        for page in doc.pages:
            if page.page_idx not in texts:
                status[page.page_idx] = "missing"
            else:
                page_texts[page.page_idx] = texts[page.page_idx]
                status[page.page_idx] = "ok" if texts[page.page_idx].strip() else "empty"
        loaded = LoadedDocument(doc, page_texts, status, sum(1 for i in texts if i not in status))
        indexes[doc.doc_id] = (loaded, _DocumentIndex(loaded, settings, DEFAULT_TEXT_POLICY))
    for q in manifest.questions:
        loaded, index = indexes[q.doc_id]
        out[q.question_id] = _retrieve_question(q, loaded, index)
    return out


def _gold_page_chunks() -> dict[str, list[Any]]:
    """MinerU chunks of each question's gold evidence pages, chunked exactly as the baseline chunked them."""
    retrieval = retrieve_runtime_questions(project_root=ROOT)
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    out = {}
    for outcome in retrieval.questions:
        q = outcome.question
        pages = set(evaluation[q.question_id]["evidence_pages"])
        chunks = [c for c in retrieval.loaded_documents[q.doc_id].chunks if c.page_id in pages]
        out[q.question_id] = sorted(chunks, key=lambda c: (c.page_id, c.chunk_id))
    return out


def _page_images(doc_id: str, pages: list[int]) -> list[dict[str, Any]]:
    render = read_json(OUT_DIR / "render_record.json")
    by_page = {(r["doc_id"], r["page_idx"]): r for r in render["pages"]}
    images = []
    for p in pages:
        row = by_page[(doc_id, p)]
        data = (OUT_DIR / row["model_jpg"]).read_bytes()
        if _sha(data) != row["model_jpg_sha256"]:
            raise SystemExit(f"{row['model_jpg']} does not match render_record.json; rerun render_pages.py")
        images.append({"page_idx": p, "file": row["model_jpg"], "sha256": row["model_jpg_sha256"], "size": row["model_jpg_size"], "data": data})
    return images


def _blocks(hits_or_chunks: list[Any], doc_id: str) -> list[EvidenceBlock]:
    blocks = []
    for rank, item in enumerate(hits_or_chunks, start=1):
        chunk = getattr(item, "chunk", item)
        blocks.append(EvidenceBlock(rank, chunk.chunk_id, chunk.doc_name, chunk.page_id, chunk.text))
    return blocks


def prepare(condition: str) -> list[dict[str, Any]]:
    """Build one request record per question in runtime-manifest order. Reads evaluation files by design (diagnostic)."""
    runtime = read_json(PILOT_DIR / "runtime_manifest.json")
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    baseline = {r["question_id"]: r for r in read_jsonl(BASELINE_DIR / "requests.jsonl")}
    config = load_provider_config(PROVIDER_CONFIG)
    clean = _clean_text_retrieval() if condition == "clean_text" else None
    gold_chunks = _gold_page_chunks() if condition == "gold_page_text" else None
    records = []
    for q in runtime["questions"]:
        qid, doc_id, question = q["question_id"], q["doc_id"], q["question"]
        ev = evaluation[qid]
        record: dict[str, Any] = {"question_id": qid, "doc_id": doc_id, "condition": condition, "action": "skip", "skip_reason": None,
                                  "evidence": [], "images": [], "system": None, "user": None}
        system, user_text, blocks, images = SYSTEM_TEMPLATE, None, [], []
        if condition == "clean_text":
            outcome = clean[qid]
            if outcome.failure is not None or not outcome.hits:
                record["skip_reason"] = outcome.no_evidence_reason or "retrieval_failed"
            else:
                blocks = _blocks(list(outcome.hits), doc_id)
        elif condition == "gold_page_text":
            if gold_chunks[qid]:
                blocks = _blocks(gold_chunks[qid], doc_id)
            else:
                record["skip_reason"] = "no_text_chunks_on_gold_pages"
        else:
            images = _page_images(doc_id, ev["evidence_pages"])
            system = SYSTEM_TEMPLATE + IMAGE_NOTE
            base = baseline[qid]
            if condition == "gold_page_image" and base["action"] == "send":
                user_text = base["user"]
                record["evidence"] = [{k: e[k] for k in ("rank", "chunk_id", "page_idx", "text_sha256")} for e in base["evidence"]]
            else:
                user_text = NO_TEXT_USER_TEMPLATE.format(question=question)
        if blocks:
            payload = build_prompt(question, blocks)
            if payload.template_id != TEMPLATE_ID or payload.template_sha256 != TEMPLATE_SHA256:
                raise SystemExit("prompt template changed")
            system, user_text = payload.system, payload.user
            record["evidence"] = [{"rank": b.rank, "chunk_id": b.chunk_id, "page_idx": b.page_idx, "chars": len(b.text),
                                   "text_sha256": _sha(b.text.encode("utf-8"))} for b in blocks]
        if user_text is None:
            records.append(record)
            continue
        text_messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        bound = input_token_upper_bound(text_messages) + IMAGE_TOKEN_BOUND * len(images)
        record.update({
            "action": "send", "system": system, "user": user_text,
            "images": [{k: im[k] for k in ("page_idx", "file", "sha256", "size")} for im in images],
            "prompt_sha256": _canonical({"messages": text_messages, "images": [im["sha256"] for im in images], "detail": IMAGE_DETAIL if images else None}),
            "input_token_upper_bound": bound,
            "max_output_tokens": config.max_output_tokens,
            "cost_upper_bound": request_cost_upper_bound(bound, config.max_output_tokens, config.prices),
        })
        records.append(record)
    return records


def _messages(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if not record["images"]:
        return ({"role": "system", "content": record["system"]}, {"role": "user", "content": record["user"]})
    content: list[dict[str, Any]] = [{"type": "text", "text": record["user"]}]
    for image in record["images"]:
        data = (OUT_DIR / image["file"]).read_bytes()
        if _sha(data) != image["sha256"]:
            raise SystemExit(f"{image['file']} changed since prepare")
        content.append({"type": "text", "text": f"Image of page {image['page_idx'] + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), "detail": IMAGE_DETAIL}})
    return ({"role": "system", "content": record["system"]}, {"role": "user", "content": content})


# --------------------------------------------------------------------------- run, with a hard cost cap


def _identity(condition: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    config_bytes = PROVIDER_CONFIG.read_bytes()
    here = Path(__file__).parent
    code = {str((here / n).relative_to(ROOT)): _sha((here / n).read_bytes()) for n in ("common.py", "headroom.py", "render_pages.py")}
    return {
        "run_id": RUN_IDS[condition], "condition": condition, "kind": "development_pilot",
        "diagnostic_reference": True,
        "baseline_run": BASELINE_DIR.name,
        "runtime_manifest_sha256": _sha((PILOT_DIR / "runtime_manifest.json").read_bytes()),
        "evaluation_manifest_sha256": _sha((PILOT_DIR / "evaluation_manifest.json").read_bytes()),
        "provider_config_sha256": _sha(config_bytes),
        "template_id": TEMPLATE_ID, "template_sha256": TEMPLATE_SHA256,
        "image_note": IMAGE_NOTE, "no_text_user_template": NO_TEXT_USER_TEMPLATE,
        "image_detail": IMAGE_DETAIL, "image_token_bound": IMAGE_TOKEN_BOUND,
        "render_record_sha256": _sha((OUT_DIR / "render_record.json").read_bytes()),
        "requests_sha256": _canonical(records),
        "code": code,
    }


def _write_requests(condition: str) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    records = prepare(condition)
    identity = _identity(condition, records)
    rdir = run_dir(condition)
    rdir.mkdir(parents=True, exist_ok=True)
    config_path = rdir / "run_config.json"
    if config_path.exists():
        old = read_json(config_path)
        if old != identity:
            diff = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
            raise SystemExit(f"{config_path} exists with a different identity ({', '.join(diff)}); a changed run needs a new run_id")
    else:
        config_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (rdir / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return rdir, records, identity


def _events(rdir: Path) -> list[dict[str, Any]]:
    path = rdir / "attempts.jsonl"
    return read_jsonl(path) if path.exists() else []


def _append(rdir: Path, event: dict[str, Any]) -> None:
    with (rdir / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _ledger(events: list[dict[str, Any]]) -> tuple[float, float, set[str], set[str]]:
    measured, reserved, saved, blocked = 0.0, 0.0, set(), set()
    open_attempts: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["event"] == "dispatch":
            open_attempts[e["attempt_id"]] = e
        elif e["event"] in ("saved", "failed"):
            start = open_attempts.pop(e["attempt_id"], None)
            if e["event"] == "saved":
                saved.add(e["question_id"])
                if e.get("measured_cost") is None:
                    reserved += start["cost_upper_bound"]
                else:
                    measured += e["measured_cost"]
            elif e.get("outcome") != "not_sent":
                reserved += start["cost_upper_bound"]
                blocked.add(e["question_id"])
    for start in open_attempts.values():
        reserved += start["cost_upper_bound"]
        blocked.add(start["question_id"])
    return measured, reserved, saved, blocked


def run(condition: str, cap: float) -> None:
    if os.environ.get(LIVE_ENV_NAME) != LIVE_ENV_VALUE:
        raise SystemExit(f"live requests need {LIVE_ENV_NAME}={LIVE_ENV_VALUE}; nothing was sent")
    refuse_redirecting_environment(os.environ)
    rdir, records, identity = _write_requests(condition)
    config = load_provider_config(PROVIDER_CONFIG)
    provider = build_live_provider(config, os.environ)
    invocation = uuid.uuid4().hex
    _append(rdir, {"event": "invocation_started", "invocation": invocation, "cap": cap, "run_id": identity["run_id"]})
    failures = 0
    end_reason = "complete"
    for record in records:
        if record["action"] != "send":
            continue
        measured, reserved, saved, blocked = _ledger(_events(rdir))
        qid = record["question_id"]
        if qid in saved or qid in blocked:
            continue
        if measured + reserved + record["cost_upper_bound"] > cap:
            end_reason = "cap_reached"
            break
        attempt_id = f"{qid[:8]}-{uuid.uuid4().hex[:8]}"
        request = ProviderRequest(qid, attempt_id, _messages(record), dict(config.params), config.max_output_tokens, config.timeout_seconds)
        _append(rdir, {"event": "dispatch", "question_id": qid, "attempt_id": attempt_id, "cost_upper_bound": record["cost_upper_bound"]})
        started = time.monotonic()
        try:
            response = provider.send(request)
        except ProviderError as exc:
            # outcome is not_sent (nothing left the machine: no cost, may be resent by a later invocation),
            # rejected or unknown (cost reserved at the upper bound, question blocked for a person to look at).
            _append(rdir, {"event": "failed", "question_id": qid, "attempt_id": attempt_id, "kind": exc.kind, "http_status": exc.http_status,
                           "outcome": exc.outcome, "message": str(exc)[:500]})
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
            "question_id": qid, "attempt_id": attempt_id, "condition": condition, "returned_model": response.returned_model,
            "returned_service_tier": response.returned_service_tier, "response_id": response.response_id,
            "finish_reason": response.finish_reason, "usage": response.usage.as_dict(), "measured_cost": cost,
            "latency_ms": latency, **parsed, "raw": response.raw,
        }
        out = rdir / "responses" / f"{attempt_id}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _append(rdir, {"event": "saved", "question_id": qid, "attempt_id": attempt_id, "measured_cost": cost,
                       "response_sha256": _sha(out.read_bytes()), "returned_model": response.returned_model,
                       "returned_service_tier": response.returned_service_tier})
        if response.returned_model != config.model or response.returned_service_tier != "default":
            end_reason = "safety_stop"
            break
    measured, reserved, saved, blocked = _ledger(_events(rdir))
    _append(rdir, {"event": "invocation_ended", "invocation": invocation, "end_reason": end_reason,
                   "measured_cost": round(measured, 6), "reserved_cost": round(reserved, 6), "saved": len(saved), "blocked": sorted(blocked)})
    print(json.dumps({"end_reason": end_reason, "measured": round(measured, 6), "reserved": round(reserved, 6), "saved": len(saved), "blocked": len(blocked)}))


# --------------------------------------------------------------------------- export and score


def export(condition: str) -> dict[str, Any]:
    rdir = run_dir(condition)
    records = read_jsonl(rdir / "requests.jsonl")
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    events = _events(rdir)
    measured, reserved, saved, blocked = _ledger(events)
    saved_by_q = {e["question_id"]: e for e in events if e["event"] == "saved"}
    rows, unserved = [], 0
    for r in records:
        qid = r["question_id"]
        gold = evaluation[qid]["answers"]
        if r["action"] != "send":
            status, answer, abstained = "no_evidence", "", True
        elif qid in saved_by_q:
            resp = read_json(rdir / "responses" / f"{saved_by_q[qid]['attempt_id']}.json")
            status, answer, abstained = "answered", resp["answer"], resp["abstained"]
        else:
            status, answer, abstained = "execution_failed", "", False
            unserved += 1
        rows.append({"question_id": qid, "status": status, "answer": answer, "abstained": abstained,
                     "em": exact_match_score(answer, gold), "f1": f1_score(answer, gold)})
    (rdir / "scores.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    n = len(rows)
    summary = {
        "run_id": RUN_IDS[condition], "condition": condition, "questions": n,
        "answered": sum(x["status"] == "answered" for x in rows), "no_evidence": sum(x["status"] == "no_evidence" for x in rows),
        "execution_failed": unserved, "abstained": sum(x["abstained"] for x in rows),
        "em": sum(x["em"] for x in rows) / n, "f1": sum(x["f1"] for x in rows) / n,
        "measured_cost_usd": round(measured, 6), "reserved_cost_usd": round(reserved, 6),
        "returned_models": sorted({e.get("returned_model") for e in events if e["event"] == "saved"}),
        "complete": unserved == 0 and not blocked,
    }
    (rdir / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


RUN_ORDER = ("gold_page_image", "clean_text", "gold_page_text", "gold_page_image_only")


def run_all(total_cap: float) -> None:
    """Run every condition in RUN_ORDER; each run's cap is what the earlier runs left of ``total_cap``."""
    for condition in RUN_ORDER:
        spent = 0.0
        for other in CONDITIONS:
            if (run_dir(other) / "attempts.jsonl").exists():
                measured, reserved, _, _ = _ledger(_events(run_dir(other)))
                spent += measured + reserved if other != condition else 0.0
        own = _ledger(_events(run_dir(condition)))[:2] if (run_dir(condition) / "attempts.jsonl").exists() else (0.0, 0.0)
        cap = total_cap - spent
        print(f"{condition}: other runs spent {spent:.6f} USD, this run has {sum(own):.6f}, cap {cap:.6f}")
        run(condition, cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_all = sub.add_parser("run-all")
    p_all.add_argument("--total-cap", type=float, required=True, help="ceiling in USD over all four runs together")
    for name in ("prepare", "run", "export"):
        p = sub.add_parser(name)
        p.add_argument("--condition", choices=CONDITIONS, required=True)
        if name == "run":
            p.add_argument("--cap", type=float, required=True, help="hard ceiling in USD for this run, counting reserved cost")
    args = parser.parse_args()
    if args.command == "run-all":
        run_all(args.total_cap)
    elif args.command == "prepare":
        rdir, records, identity = _write_requests(args.condition)
        sent = [r for r in records if r["action"] == "send"]
        print(json.dumps({"run_dir": str(rdir.relative_to(ROOT)), "send": len(sent), "skip": len(records) - len(sent),
                          "cost_upper_bound_usd": round(sum(r["cost_upper_bound"] for r in sent), 4),
                          "skip_reasons": sorted({r["skip_reason"] for r in records if r["skip_reason"]})}, indent=2))
    elif args.command == "run":
        run(args.condition, args.cap)
    else:
        export(args.condition)


if __name__ == "__main__":
    sys.exit(main())
