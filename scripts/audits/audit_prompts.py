"""Runtime-only integrity audit of the prompts a dry run prepared.

The audit reads ``prepared_requests.jsonl`` and ``dry_run_summary.json`` from a dry-run directory, the runtime
manifest, and the MinerU files the manifest names. It never opens ``evaluation_manifest.json``, a gold answer, a
label, the clean reference text or an annotation. The evaluation-based leakage search is a separate script,
``audit_prompt_leakage_diagnostic.py``, and its findings must never feed runtime preparation.

The script does not build prompts. It confirms the saved messages and hashes two ways: it parses each saved user
message with patterns derived from the constants of ``faar.answer_prompt``, and it calls the repository's own
``build_prompt`` on the saved evidence and compares the result with the saved fields.

Checks (each one reports pass or fail; the exit code is 1 when any fails):

    association     each question equals its manifest text; each block belongs to the question's document; the
                    saved user message parses into the question and the blocks with the right number, page,
                    label, fence and text; evidence and prompt hashes equal the repository builder's; each chunk
                    text is a substring of the whitespace-collapsed MinerU text of its page
    blocks          no empty, markup-only or duplicated block; no fence collision, header-like line or control
                    character inside a block
    identity        the document id, its basename and its folder prefix appear in no text outside the evidence
                    and the question; other basename fragments outside them are template words
    bounds          the input bound recomputed with ``faar.request_budget``, the cost bound from the price table
                    in the dry-run summary, and the totals in the summary
    skipped         each skip reason is true of the MinerU text; no skipped request carries a prompt
    instructions    a scan of the evidence for text that addresses a model; strong patterns fail, weak ones are counted

The output holds question ids, ranks, counts and character offsets. It holds no question, prompt or document text.

Usage:
    python scripts/audits/audit_prompts.py --dry-run-dir .local/work/dry-run --out .local/work/audits
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EXIT_FAILED_CHECK,
    HAN,
    REPO_ROOT,
    fail,
    norm_ws,
    prepare_out,
    read_json,
    read_jsonl,
    require_dir,
    require_file,
    resolve,
    sha256_file,
    sha256_hex,
    write_json,
)

from faar import answer_prompt as ap
from faar.live_contract import EvidenceBlock, PriceTable
from faar.request_budget import MICRO, input_token_upper_bound, request_cost_upper_bound_micro

DEFAULT_DRY_RUN = ".local/work/dry-run"
DEFAULT_OUT = ".local/work/audits"
DEFAULT_MANIFEST = "results/pilots/ohr_dev_v1/runtime_manifest.json"
CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}

Findings = dict[str, list[Any]]


def _template_pattern(template: str, groups: dict[str, str]) -> re.Pattern[str]:
    escaped = re.escape(template)
    for name, pattern in groups.items():
        escaped = escaped.replace(re.escape("{" + name + "}"), f"(?P<{name}>{pattern})")
    return re.compile(escaped, re.S)


_FENCE_RUN = re.escape(ap.FENCE_CHAR) + "+"
USER_PATTERN = _template_pattern(ap.USER_TEMPLATE, {"question": ".*?", "count": r"\d+", "blocks": ".*"})
_HEAD_TEMPLATE, _TAIL_TEMPLATE = ap.BLOCK_TEMPLATE.split("{text}")
HEAD_PATTERN = _template_pattern(
    _HEAD_TEMPLATE, {"number": r"\d+", "page": r"\d+", "label": r"[^\n]+", "fence": _FENCE_RUN}
)
TEMPLATE_WORDS = (ap.SYSTEM_TEMPLATE + ap.USER_TEMPLATE + ap.BLOCK_TEMPLATE).casefold()


def parse_user_message(user: str, evidence: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse a saved user message. Return its question, block count, block spans and problems, or ``None``.

    Spans are ``(kind, start, end)`` offsets in ``user`` for the kinds ``header``, ``fence``, ``evidence`` and ``sep``.
    A malformed message gives a ``problems`` list instead of spans past the failure.
    """
    match = USER_PATTERN.fullmatch(user)
    if not match:
        return None
    problems: list[str] = []
    spans: list[tuple[str, int, int]] = []
    base = match.start("blocks")
    rest = match["blocks"]
    pos = 0
    if int(match["count"]) != len(evidence):
        problems.append("block count in the text differs from the saved evidence")
    for index, block in enumerate(evidence, start=1):
        head = HEAD_PATTERN.match(rest, pos)
        if not head:
            problems.append(f"block {index}: header does not match the block template")
            break
        text = block["text"]
        prefix = block["doc_id"] + "-"
        label = block["chunk_id"][len(prefix) :] if block["chunk_id"].startswith(prefix) else None
        if int(head["number"]) != index:
            problems.append(f"block {index}: number {head['number']} in the header")
        if int(head["page"]) != block["page_idx"] + 1:
            problems.append(f"block {index}: page {head['page']} in the header, page_idx {block['page_idx']}")
        if head["label"] != label:
            problems.append(f"block {index}: label in the header differs from the chunk id")
        fence = head["fence"]
        spans.append(("header", base + head.start(), base + head.start("fence")))
        spans.append(("fence", base + head.start("fence"), base + head.end()))
        if fence != ap.fence_for(text):
            problems.append(f"block {index}: fence differs from the repository's fence_for")
        if len(fence) < ap.MIN_FENCE_LENGTH or len(fence) <= _longest_run(text):
            problems.append(f"block {index}: fence is not longer than every tilde run in the text")
        pos = head.end()
        if not rest.startswith(text, pos):
            problems.append(f"block {index}: the saved user message does not hold the saved evidence text")
            break
        spans.append(("evidence", base + pos, base + pos + len(text)))
        pos += len(text)
        tail = _TAIL_TEMPLATE.replace("{fence}", fence)
        if not rest.startswith(tail, pos):
            problems.append(f"block {index}: closing fence is wrong")
            break
        spans.append(("fence", base + pos, base + pos + len(tail)))
        pos += len(tail)
        if index < len(evidence):
            if not rest.startswith(ap.BLOCK_SEPARATOR, pos):
                problems.append(f"block {index}: separator is wrong")
                break
            spans.append(("sep", base + pos, base + pos + len(ap.BLOCK_SEPARATOR)))
            pos += len(ap.BLOCK_SEPARATOR)
    else:
        if pos != len(rest):
            problems.append("text follows the last block")
    return {"question": match["question"], "question_span": match.span("question"), "spans": spans, "problems": problems}


def outside_evidence_text(record: dict[str, Any], message: dict[str, Any]) -> tuple[str, str]:
    """Split a request into the text outside the evidence (question removed) and the question text."""
    user = record["user"]
    q_start, q_end = message["question_span"]
    cut = sorted([(q_start, q_end)] + [(s, e) for kind, s, e in message["spans"] if kind == "evidence"])
    pieces = [record["system"]]
    cursor = 0
    for start, end in cut:
        pieces.append(user[cursor:start])
        cursor = end
    pieces.append(user[cursor:])
    return "\n".join(pieces), user[q_start:q_end]


def load_mineru_pages(manifest: dict[str, Any], project_root: Path) -> tuple[dict[str, dict[int, str]], list[str]]:
    """Read the MinerU file of every document that declares one. Return page text by document, and problems."""
    pages: dict[str, dict[int, str]] = {}
    problems: list[str] = []
    missing: list[str] = []
    for doc in manifest["documents"]:
        entry = doc["noisy_text"]
        by_page: dict[int, str] = {}
        if entry["status"] == "present":
            path = project_root / entry["path"]
            if not path.is_file():
                missing.append(entry["path"])
                continue
            if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
                problems.append(f"{doc['doc_id']}: MinerU file hash differs from the manifest")
            for row in read_json(path, "MinerU file"):
                if isinstance(row, dict) and isinstance(row.get("page_idx"), int) and row["page_idx"] >= 0:
                    text = row.get("text") if isinstance(row.get("text"), str) else ""
                    by_page[row["page_idx"]] = by_page[row["page_idx"]] + "\n" + text if row["page_idx"] in by_page else text
        pages[doc["doc_id"]] = by_page
    if missing:
        fail(
            f"{len(missing)} MinerU file(s) named by the manifest are missing under {project_root} "
            f"(first: {missing[0]}). Use --project-root for the checkout that holds OHR-Bench/.",
        )
    return pages, problems


def _has_letter_or_digit(text: str) -> bool:
    return any(unicodedata.category(c)[0] in "LN" for c in text)


def _report(passed: bool, **details: Any) -> dict[str, Any]:
    return {"pass": passed, **details}


def check_association(
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    pages: dict[str, dict[int, str]],
    summary: dict[str, Any],
    manifest_problems: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    questions = {q["question_id"]: q for q in manifest["questions"]}
    documents = {d["doc_id"]: d for d in manifest["documents"]}
    found: Findings = defaultdict(list)
    found["mineru_file_hash_differs_from_manifest"].extend(manifest_problems)
    if [r["question_id"] for r in records] != list(questions):
        found["records_not_in_manifest_order"].append("ids or order differ")
    if ap.TEMPLATE_SHA256 != summary["template"]["template_sha256"]:
        found["repository_template_differs_from_dry_run"].append(
            "the current answer_prompt template hash is not the dry run's; the builder comparison below would fail"
        )
    sends = [r for r in records if r["action"] == "send"]
    if len({r["system"] for r in sends}) > 1:
        found["more_than_one_system_message"].append("")
    if sends and sends[0]["system"] != ap.SYSTEM_TEMPLATE:
        found["system_differs_from_module_template"].append("")
    parsed: dict[str, dict[str, Any]] = {}
    for record in sends:
        qid = record["question_id"]
        question = questions.get(qid)
        if question is None:
            found["question_not_in_manifest"].append(qid)
            continue
        if record["question"] != question["question"] or record["doc_id"] != question["doc_id"]:
            found["record_differs_from_manifest"].append(qid)
        evidence = record["evidence"]
        message = parse_user_message(record["user"], evidence)
        if message is None:
            found["user_message_does_not_match_template"].append(qid)
            continue
        parsed[qid] = message
        if message["question"] != question["question"]:
            found["question_in_user_message_not_verbatim"].append(qid)
        for problem in message["problems"]:
            found["user_message_parse"].append((qid, problem))
        if [e["rank"] for e in evidence] != list(range(1, len(evidence) + 1)):
            found["ranks_not_1_to_n"].append(qid)
        declared_pages = {p["page_idx"] for p in documents[question["doc_id"]]["pages"]}
        for index, e in enumerate(evidence, start=1):
            if e["doc_id"] != question["doc_id"]:
                found["evidence_doc_id_differs_from_question"].append((qid, index))
            if not e["chunk_id"].startswith(e["doc_id"] + "-"):
                found["chunk_id_lacks_doc_prefix"].append((qid, index))
            if e["page_idx"] + 1 != e["pdf_page_number"]:
                found["pdf_page_number_not_page_idx_plus_1"].append((qid, index))
            if e["page_idx"] not in declared_pages:
                found["page_not_declared_for_document"].append((qid, index))
            if sha256_hex(e["text"].encode("utf-8")) != e["text_sha256"]:
                found["text_sha256_mismatch"].append((qid, index))
            if e["chars"] != len(e["text"]):
                found["chars_mismatch"].append((qid, index))
            page = pages.get(question["doc_id"], {}).get(e["page_idx"])
            if page is None or e["text"] not in norm_ws(page):
                found["chunk_text_not_in_mineru_page"].append((qid, index))
        blocks = [EvidenceBlock(e["rank"], e["chunk_id"], e["doc_id"], e["page_idx"], e["text"]) for e in evidence]
        try:
            built = ap.build_prompt(question["question"], blocks)
        except (ValueError, TypeError) as exc:
            found["repository_builder_raises"].append((qid, type(exc).__name__))
            continue
        expected = (built.system, built.user, built.prompt_sha256, built.evidence_sha256, built.template_sha256)
        saved = (
            record["system"],
            record["user"],
            record["prompt_sha256"],
            record["evidence_sha256"],
            summary["template"]["template_sha256"],
        )
        if expected != saved:
            found["repository_builder_disagrees"].append(qid)
    failures = {k: v for k, v in found.items() if v}
    return _report(not failures, records_checked=len(sends), failures=failures), parsed


def check_blocks(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hard: Findings = defaultdict(list)
    control_character_questions: list[str] = []
    counts: Counter[str] = Counter()
    low_content: list[tuple[str, int, int]] = []
    chars: list[int] = []
    markup_blocks = 0
    for record in records:
        if record["action"] != "send":
            continue
        qid = record["question_id"]
        evidence = record["evidence"]
        expected_blocks = min(5, record["ocr_condition"]["chunks"])
        if len(evidence) < 5:
            counts["fewer_than_5_blocks"] += 1
            if len(evidence) != expected_blocks:
                hard["fewer_than_5_blocks_unexplained"].append((qid, len(evidence)))
        if len({e["chunk_id"] for e in evidence}) != len(evidence):
            hard["duplicate_chunk_id"].append(qid)
        if len({e["text_sha256"] for e in evidence}) != len(evidence):
            hard["duplicate_text"].append(qid)
        if not record["question"].strip():
            hard["empty_question"].append(qid)
        elif any(unicodedata.category(c) in CONTROL_CATEGORIES for c in record["question"]):
            counts["questions_with_control_character"] += 1
            control_character_questions.append(qid)
        for index, e in enumerate(evidence, start=1):
            text = e["text"]
            chars.append(len(text))
            if not text.strip():
                hard["empty_or_whitespace_block"].append((qid, index))
            elif not _has_letter_or_digit(text):
                hard["block_without_letter_or_digit"].append((qid, index))
            if "~" in text:
                counts["blocks_with_tilde"] += 1
                if re.search(r"^~{4,}$", text, re.M):
                    hard["line_equal_to_a_minimal_fence"].append((qid, index))
            if re.search(r"\[Evidence \d+\]", text) or "Question:" in text or "Evidence, best match" in text:
                hard["header_like_or_template_text_in_block"].append((qid, index))
            if any(unicodedata.category(c) in CONTROL_CATEGORIES or c == "\ufffd" for c in text):
                hard["control_or_replacement_character_in_block"].append((qid, index))
            if any(c in text for c in "\n\r\t"):
                hard["newline_or_tab_in_block"].append((qid, index))
            alnum = sum(ch.isalnum() for ch in text)
            if len(text) < 80 or alnum / max(1, len(text)) < 0.5 or alnum < 30:
                low_content.append((qid, index, len(text)))
            if re.search(r"\\[a-zA-Z]+\{|<table|</?td>|\$\$", text):
                markup_blocks += 1
    warnings = {"question_with_control_character": control_character_questions} if control_character_questions else {}
    failures = {k: v for k, v in hard.items() if v}
    return _report(
        not failures,
        blocks=len(chars),
        counts=dict(counts),
        block_chars={"min": min(chars), "median": statistics.median(chars), "max": max(chars)} if chars else None,
        blocks_with_latex_or_table_markup=markup_blocks,
        short_or_symbol_heavy_blocks=len(low_content),
        short_or_symbol_heavy_block_ids=low_content,
        warnings=warnings,
        failures=failures,
    )


def _longest_run(text: str) -> int:
    return max((len(run) for run in re.findall(_FENCE_RUN, text)), default=0)


def _identity_fragments(doc_id: str) -> dict[str, str]:
    basename = doc_id.split("/")[-1]
    return {
        "doc_id": doc_id,
        "doc_id_casefold": doc_id.casefold(),
        "basename": basename,
        "basename_casefold": basename.casefold(),
        "basename_spaced": re.sub(r"[_\-]+", " ", basename),
        "folder_prefix": doc_id.split("/")[0] + "/",
    }


def _basename_tokens(doc_id: str) -> list[str]:
    basename = doc_id.split("/")[-1]
    return [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", basename) if len(t) >= 4 or (HAN.search(t) and len(t) >= 2)]


def check_identity(records: Sequence[dict[str, Any]], parsed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Look for document identity in every part of a request that is not evidence text or the question."""
    full_hits: list[tuple[str, str, str]] = []
    unexplained: list[tuple[str, str]] = []
    coincidental: Counter[str] = Counter()
    in_question: Counter[str] = Counter()
    in_evidence: Counter[str] = Counter()
    for record in records:
        if record["action"] != "send" or record["question_id"] not in parsed:
            continue
        qid = record["question_id"]
        outside_text, question_text = (text.casefold() for text in outside_evidence_text(record, parsed[qid]))
        evidence_text = "\n".join(e["text"] for e in record["evidence"]).casefold()
        for name, fragment in _identity_fragments(record["doc_id"]).items():
            needle = fragment.casefold()
            if needle in outside_text:
                if needle in TEMPLATE_WORDS:  # a name that is itself template wording cannot be told apart
                    coincidental[fragment] += 1
                else:
                    full_hits.append((qid, name, "template_or_header"))
            if needle in question_text:
                in_question[name] += 1
            if needle in evidence_text:
                in_evidence[name] += 1
        for token in _basename_tokens(record["doc_id"]):
            needle = token.casefold()
            if needle in outside_text:
                if needle in TEMPLATE_WORDS:
                    coincidental[token] += 1
                else:
                    unexplained.append((qid, "basename_token"))
    return _report(
        not full_hits and not unexplained,
        full_identity_hits_outside_evidence_and_question=full_hits,
        unexplained_basename_token_hits_outside_evidence_and_question=unexplained,
        identity_fragments_and_basename_tokens_that_are_template_wording=dict(coincidental),
        fragment_hits_inside_question_text=dict(in_question),
        fragment_hits_inside_evidence_text=dict(in_evidence),
        note="hits inside the question or the evidence are runtime text; only hits elsewhere fail",
    )


def check_bounds(records: Sequence[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    sends = [r for r in records if r["action"] == "send"]
    prices = PriceTable(**summary["provider_config"]["prices"])
    mismatched_bound: list[str] = []
    mismatched_cost: list[str] = []
    mismatched_heuristic: list[str] = []
    output_limits: Counter[int] = Counter()
    bounds: list[int] = []
    per_request: list[int] = []
    for r in sends:
        bound = input_token_upper_bound([{"content": r["system"]}, {"content": r["user"]}])
        bounds.append(bound)
        if bound != r["input_token_upper_bound"]:
            mismatched_bound.append(r["question_id"])
        output_limits[r["max_output_tokens"]] += 1
        micro = request_cost_upper_bound_micro(bound, r["max_output_tokens"], prices)
        per_request.append(micro)
        if abs(micro / MICRO - r["cost_upper_bound"]) > 5e-7:
            mismatched_cost.append(r["question_id"])
        estimate = ap.estimate_tokens(r["system"] + "\n\n" + r["user"])["estimate"]
        if estimate != r["token_estimate"]["estimate"]:
            mismatched_heuristic.append(r["question_id"])
    limit = summary["input_token_upper_bound"]["limit"]
    totals = summary["cost_upper_bound"]
    total_cost = sum(per_request) / MICRO
    conditions = {
        "input_bounds_equal_saved": not mismatched_bound,
        "cost_bounds_equal_saved": not mismatched_cost,
        "heuristic_estimates_equal_saved": not mismatched_heuristic,
        "no_bound_above_limit": bool(bounds) and max(bounds) <= limit,
        "bound_total_equals_summary": sum(bounds) == summary["input_token_upper_bound"]["total"],
        "bound_max_equals_summary": bool(bounds) and max(bounds) == summary["input_token_upper_bound"]["max"],
        "one_attempt_cost_equals_summary": abs(total_cost - totals["per_attempt_total"]) < 5e-7,
        "worst_case_cost_equals_summary": abs(total_cost * totals["max_attempts"] - totals["worst_case_all_attempts"]) < 5e-7,
        "sent_count_equals_summary": len(sends) == summary["counts"]["send"],
    }
    return _report(
        all(conditions.values()),
        conditions=conditions,
        input_bound={
            "min": min(bounds, default=None),
            "median": statistics.median(bounds) if bounds else None,
            "max": max(bounds, default=None),
            "total": sum(bounds),
            "limit": limit,
        },
        cost_bound_usd={"one_attempt": total_cost, f"{totals['max_attempts']}_attempts": total_cost * totals["max_attempts"]},
        max_output_tokens_on_sent_requests=dict(output_limits),
        mismatched_bound=mismatched_bound,
        mismatched_cost=mismatched_cost,
        mismatched_heuristic=mismatched_heuristic,
        formula="16 + sum over the two messages of (UTF-8 bytes + 4), from faar.request_budget.input_token_upper_bound",
    )


def check_skipped(
    records: Sequence[dict[str, Any]], manifest: dict[str, Any], pages: dict[str, dict[int, str]]
) -> dict[str, Any]:
    documents = {d["doc_id"]: d for d in manifest["documents"]}
    skipped = [r for r in records if r["action"] != "send"]
    details: list[dict[str, Any]] = []
    all_true = True
    for r in skipped:
        doc_pages = documents[r["doc_id"]]["pages"]
        texts = [pages.get(r["doc_id"], {}).get(p["page_idx"]) for p in doc_pages]
        if r["skip_reason"] == "no_text_chunks":
            true = all(t is None or not t.strip() for t in texts)
        elif r["skip_reason"] == "no_text_content":
            true = any(t and t.strip() for t in texts) and not any(t and _has_letter_or_digit(t) for t in texts)
        else:
            true = False
        all_true &= true
        details.append(
            {
                "question_id": r["question_id"],
                "skip_reason": r["skip_reason"],
                "reason_true_per_mineru": true,
                "page_text_lengths": [None if t is None else len(t) for t in texts],
                "no_prompt_or_evidence": r["prompt_sha256"] is None and not r["evidence"] and r["system"] is None,
            }
        )
    none_dropped = [r["question_id"] for r in records] == [q["question_id"] for q in manifest["questions"]]
    no_prompts = all(d["no_prompt_or_evidence"] for d in details)
    return _report(
        all_true and none_dropped and no_prompts,
        skipped=len(skipped),
        reason_counts=dict(Counter(r["skip_reason"] for r in skipped)),
        every_manifest_question_has_a_record=none_dropped,
        details=details,
    )


STRONG_PATTERNS = {
    "ignore_previous": r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|above|prior|earlier|instruction|prompt|rule|direction)s?\b",
    "please_do": r"\b(please|kindly)\s+(answer|reply|respond|provide|write|explain|summari[sz]e|translate|say|output|print|return|ignore|state|list)\b",
    "role_marker": r"(?:^|\s)(system|assistant|user|human|ai|model)\s*:\s",
    "chat_tokens": r"<\|[^|>]{1,20}\|>|\[/?INST\]|<<SYS>>|###\s*(instruction|system|response|input)",
    "model_reference": r"\b(as an ai|language model|chatgpt|openai|gpt-?[34]\w*|large language)\b",
    "template_tokens": r"\bNO_ANSWER\b|\[Evidence \d+\]|^Question:|\bAnswer:",
    "zh_ignore": "忽略|无视|不要理会",
    "zh_instruction": "指令|指示|提示词",
    "zh_you_are": "你是|你的任务",
}
WEAK_PATTERNS = {
    "you_must_should_are": r"\byou\s+(must|should|shall|are to|will need|need to|are an?|are the)\b",
    "answer_reply_with": r"\b(answer|reply|respond)\s+(with|only|in|as|by|the question|briefly|yes|no)\b",
    "instruction_word": r"\binstructions?\b",
    "prompt_word": r"\bprompts?\b",
    "zh_please": "请",
    "zh_answer": "回答",
    "zh_must": "必须",
    "zh_output_translate_reply": "输出|翻译|回复",
}


def check_instructions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strong: list[tuple[str, int, str, int]] = []
    weak: list[tuple[str, int, str, int]] = []
    for r in records:
        if r["action"] != "send":
            continue
        for e in r["evidence"]:
            for table, sink in ((STRONG_PATTERNS, strong), (WEAK_PATTERNS, weak)):
                for name, pattern in table.items():
                    for m in re.finditer(pattern, e["text"], re.I | re.M):
                        sink.append((r["question_id"], e["rank"], name, m.start()))
    return _report(
        not strong,
        strong_hits=len(strong),
        strong_hit_locations=[list(s) for s in strong],
        weak_hits=len(weak),
        weak_hits_by_pattern=dict(Counter(w[2] for w in weak)),
        questions_with_weak_hits=len({w[0] for w in weak}),
        note="locations are (question_id, rank, pattern, character offset in the block). Weak hits are ordinary "
        "document wording, such as 'must' or 'instructions'. The system rule covers text inside the fences.",
    )


def check_reproducibility(records_path: Path, other: Path) -> dict[str, Any]:
    other_file = require_file(other / "prepared_requests.jsonl", "comparison prepared_requests.jsonl")
    same = records_path.read_bytes() == other_file.read_bytes()
    return _report(
        same,
        compared_with=str(other),
        note="byte equality of prepared_requests.jsonl; a different provider config changes request ids, so compare "
        "dry runs made with the same config",
    )


def trace_dry_run(project_root: Path, provider_config: Path | None, out_dir: Path) -> dict[str, Any]:
    """Run ``run_pilot_live.py dry-run`` in this process under an audit hook and list the data files it opens.

    The repository modules load first, in their own phase, because importing ``faar.settings`` reads the model
    revision lock. That read belongs to the import, not to the dry run, so the report lists it separately.
    """
    phases: dict[str, list[str]] = {"import": [], "dry_run": []}
    current: dict[str, str | None] = {"phase": None}

    def hook(event: str, args: tuple[Any, ...]) -> None:
        phase = current["phase"]
        if phase and event == "open" and isinstance(args[0], (str, os.PathLike)):
            phases[phase].append(os.fspath(args[0]))

    sys.addaudithook(hook)
    current["phase"] = "import"
    try:
        import faar.answer_providers  # noqa: F401
        import faar.live_runner  # noqa: F401
        import faar.pilot_runner  # noqa: F401
    finally:
        current["phase"] = None
    script = REPO_ROOT / "scripts/experiments/run_pilot_live.py"
    argv = ["run_pilot_live.py", "dry-run", "--project-root", str(project_root), "--out", str(out_dir)]
    if provider_config is not None:
        argv += ["--provider-config", str(provider_config)]
    saved_argv = sys.argv
    sys.argv = argv
    exit_code: int | str = 0
    current["phase"] = "dry_run"
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if exc.code is not None else 0
    finally:
        current["phase"] = None
        sys.argv = saved_argv

    root = str(project_root.resolve()) + os.sep

    def data_files(opened: Sequence[str]) -> list[str]:
        resolved = {str(Path(p).resolve()) for p in opened if not p.startswith("/dev/")}
        inside = sorted(p[len(root) :] for p in resolved if p.startswith(root) and "/.local/venv" not in p)
        return [p for p in inside if not p.endswith((".py", ".pyc", ".so", ".pth", ".cfg"))]

    during = data_files(phases["dry_run"])
    at_import = data_files(phases["import"])
    forbidden_markers = ("evaluation_manifest", "/gt/", "annotation", "selection_record", "inspection")
    forbidden = [p for p in during + at_import if any(m in "/" + p for m in forbidden_markers)]
    return _report(
        exit_code == 0 and not forbidden,
        dry_run_exit_code=exit_code,
        mineru_files_opened=sum(1 for p in during if "/MinerU/" in "/" + p),
        other_data_files_opened_during_dry_run=[p for p in during if "/MinerU/" not in "/" + p],
        data_files_opened_while_importing_modules=at_import,
        forbidden_files_opened=forbidden,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run-dir", default=DEFAULT_DRY_RUN, help=f"dry-run directory (default: {DEFAULT_DRY_RUN})")
    parser.add_argument(
        "--runtime-manifest", default=DEFAULT_MANIFEST, help=f"runtime manifest (default: {DEFAULT_MANIFEST})"
    )
    parser.add_argument(
        "--project-root", default=str(REPO_ROOT), help="checkout that holds OHR-Bench/ (default: this repository)"
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--compare-dry-run", metavar="DIR", help="second dry-run directory that must be byte-identical")
    parser.add_argument(
        "--trace-dry-run",
        action="store_true",
        help="also run a dry run under an audit hook and list the data files it opens (writes under --out)",
    )
    parser.add_argument("--provider-config", help="provider config for --trace-dry-run")
    args = parser.parse_args(argv)

    project_root = resolve(args.project_root)
    dry_run = require_dir(resolve(args.dry_run_dir), "dry-run directory", "run_pilot_live.py dry-run writes it")
    records_path = require_file(dry_run / "prepared_requests.jsonl", "prepared_requests.jsonl")
    records = read_jsonl(records_path, "prepared_requests.jsonl")
    summary = read_json(dry_run / "dry_run_summary.json", "dry_run_summary.json")
    manifest_path = resolve(args.runtime_manifest, project_root)
    manifest = read_json(manifest_path, "runtime manifest")
    if summary.get("runtime_manifest_sha256") and summary["runtime_manifest_sha256"] != sha256_file(manifest_path):
        fail("the dry run was made from a different runtime manifest than --runtime-manifest", code=EXIT_FAILED_CHECK)
    pages, manifest_problems = load_mineru_pages(manifest, project_root)
    out = prepare_out(resolve(args.out))

    association, parsed = check_association(records, manifest, pages, summary, manifest_problems)
    checks: dict[str, dict[str, Any]] = {
        "association": association,
        "blocks": check_blocks(records),
        "identity": check_identity(records, parsed),
        "bounds": check_bounds(records, summary),
        "skipped": check_skipped(records, manifest, pages),
        "instructions": check_instructions(records),
    }
    if args.compare_dry_run:
        checks["reproducibility"] = check_reproducibility(records_path, resolve(args.compare_dry_run))
    if args.trace_dry_run:
        trace_dir = out / "trace-dry-run"
        prepare_out(trace_dir)
        provider_config = resolve(args.provider_config) if args.provider_config else None
        checks["dry_run_file_access"] = trace_dry_run(project_root, provider_config, trace_dir)

    result = {
        "kind": "runtime-only prompt audit; reads no evaluation data",
        "dry_run_sha256": {"prepared_requests.jsonl": sha256_file(records_path)},
        "template_sha256": summary["template"]["template_sha256"],
        "records": len(records),
        "sent": sum(1 for r in records if r["action"] == "send"),
        "skipped": sum(1 for r in records if r["action"] != "send"),
        "evidence_blocks": sum(len(r["evidence"]) for r in records if r["action"] == "send"),
        "checks": checks,
    }
    write_json(out / "audit_prompts.json", result)

    print(f"records {result['records']} (sent {result['sent']}, skipped {result['skipped']}), "
          f"evidence blocks {result['evidence_blocks']}")
    for name, check in checks.items():
        print(f"{'PASS' if check['pass'] else 'FAIL'}  {name}")
        if not check["pass"]:
            print(f"      {check.get('failures') or check.get('conditions') or check}")
    print(f"weak instruction-like hits: {checks['instructions']['weak_hits']} in "
          f"{checks['instructions']['questions_with_weak_hits']} questions "
          f"({checks['instructions']['weak_hits_by_pattern']})")
    print(f"wrote {out / 'audit_prompts.json'}")
    return 0 if all(c["pass"] for c in checks.values()) else EXIT_FAILED_CHECK


if __name__ == "__main__":
    raise SystemExit(main())
