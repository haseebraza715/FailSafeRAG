"""Readable Markdown preview of prepared answer requests.

The lead reads this file before approving prompt wording (study brief section
15.7 and the review step in section 15.12). It shows, for every prepared
request, the exact system and user messages that would be sent, and it lists
every question, including the ones that will not be sent.

Input: prepared-request records as written by the dry run (contract section
"Prepared-request record"), with ``evidence[i].text`` present when the export
carries it. The function reads no files, opens no connection and needs no
provider. Missing keys render as ``n/a`` instead of failing.

Layout of the output
--------------------
1. Title and a note that the wording is a draft and that token estimates are
   heuristic.
2. A summary table with one row per record, in input order: question id, action,
   skip reason, evidence blocks, evidence characters, token estimate, prompt
   SHA-256 prefix. Each id links to its section.
3. A case-type index (below).
4. One section per record with the question, action, evidence ids and pages,
   the exact system and user messages, and the hashes.

Case-type index
---------------
The index classifies from the question text and the evidence text only. It never
reads gold data. Han characters and Latin letters are counted separately in the
question and in the evidence. A text's Han share is Han / (Han + Latin letters).

- English: every text with letters has a Han share of at most 0.1.
- Chinese (Han): every text with letters has a Han share of at least 0.5.
- Mixed-language: any other case with letters, for example an English question
  over Chinese evidence, or a Chinese document with many Latin terms.
- Empty evidence (skip): the record has no evidence blocks.
- Skipped for another reason: ``action`` is ``skip`` and there is evidence.
- Longest evidence: the five records with the most evidence characters.

English, Chinese (Han) and Mixed-language are exclusive. A record with no
letters in either text appears in none of the three. The other groups overlap
with them.

Safe fences
-----------
Message and question text is shown in fenced code blocks. The fence is a run of
backticks longer than any run of backticks in the content (minimum three), so no
line of the content can close the block early. Table cells and headings escape
pipes and line breaks, and show ids as code spans with a matching delimiter.

Ways this module could fail (each is covered in tests/test_prompt_preview.py)
-----------------------------------------------------------------------------
  V1. A record is missing from the summary or has no section, especially a skipped one.
  V2. Message text that contains backticks or a fence-like line breaks out of its block.
  V3. A question id or text with a pipe, newline or backtick breaks the table or a heading.
  V4. The preview alters the message text, so a reviewer reads something other than what would be sent.
  V5. Case-type classification uses a gold field, or puts a Chinese case under English.
  V6. The longest-evidence entry is wrong, or a skipped question crashes the index.
  V7. Token estimates read as tokenizer counts.
  V8. The output depends on dict order in a way that changes between runs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

HAN_PATTERN = re.compile("[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ebef]")
LATIN_PATTERN = re.compile("[A-Za-zÀ-ɏ]")

ENGLISH_MAX_HAN_SHARE = 0.1
CHINESE_MIN_HAN_SHARE = 0.5
LONGEST_COUNT = 5
SHA_PREFIX_LENGTH = 12

CASE_ENGLISH = "English"
CASE_CHINESE = "Chinese (Han)"
CASE_MIXED = "Mixed-language"
CASE_EMPTY = "Empty evidence (skip)"
CASE_OTHER_SKIP = "Skipped for another reason"
CASE_LONGEST = f"Longest evidence (top {LONGEST_COUNT})"
CASE_ORDER = (CASE_ENGLISH, CASE_CHINESE, CASE_MIXED, CASE_EMPTY, CASE_OTHER_SKIP, CASE_LONGEST)

MISSING = "n/a"


def fence_for(content: str, minimum: int = 3) -> str:
    """Return a backtick fence longer than any backtick run in ``content``."""
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    return "`" * max(minimum, longest + 1)


def fenced(content: str) -> str:
    fence = fence_for(content)
    return f"{fence}text\n{content}\n{fence}"


def code_span(value: Any) -> str:
    """Inline code for an id, safe for any content. Line breaks become spaces."""
    text = re.sub(r"\s+", " ", str(value)) if re.search(r"[\r\n  ]", str(value)) else str(value)
    if not text:
        return MISSING
    delimiter = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") or text.startswith(" ") or text.endswith(" ") else ""
    return f"{delimiter}{pad}{text}{pad}{delimiter}"


def cell(value: Any) -> str:
    """Table cell text: pipes escaped, line breaks flattened."""
    if value is None or value == "":
        return MISSING
    text = re.sub(r"[\r\n  ]+", " ", str(value))
    return text.replace("\\", "\\\\").replace("|", "\\|")


def id_cell(value: Any) -> str:
    """A table cell that holds an id, shown as a code span with pipes escaped."""
    if value is None or value == "":
        return MISSING
    return code_span(value).replace("|", "\\|")


def _han_share(text: str) -> float | None:
    han = len(HAN_PATTERN.findall(text))
    latin = len(LATIN_PATTERN.findall(text))
    if han + latin == 0:
        return None
    return han / (han + latin)


def _evidence_list(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evidence = record.get("evidence")
    return list(evidence) if evidence else []


def _evidence_text(record: Mapping[str, Any]) -> str:
    return "\n".join(str(item.get("text")) for item in _evidence_list(record) if item.get("text") is not None)


def _evidence_chars(record: Mapping[str, Any]) -> int:
    total = 0
    for item in _evidence_list(record):
        chars = item.get("chars")
        if isinstance(chars, int) and not isinstance(chars, bool):
            total += chars
        elif item.get("text") is not None:
            total += len(str(item["text"]))
    return total


def classify_language(question: str, evidence_text: str) -> str | None:
    """Return CASE_ENGLISH, CASE_CHINESE or CASE_MIXED from the two texts, or None when neither has letters."""
    shares = [share for share in (_han_share(question), _han_share(evidence_text)) if share is not None]
    if not shares:
        return None
    if all(share <= ENGLISH_MAX_HAN_SHARE for share in shares):
        return CASE_ENGLISH
    if all(share >= CHINESE_MIN_HAN_SHARE for share in shares):
        return CASE_CHINESE
    return CASE_MIXED


def case_index(prepared: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Map each case type to the zero-based positions of the records in it, in input order."""
    groups: dict[str, list[int]] = {name: [] for name in CASE_ORDER}
    for position, record in enumerate(prepared):
        language = classify_language(str(record.get("question") or ""), _evidence_text(record))
        if language is not None:
            groups[language].append(position)
        if not _evidence_list(record):
            groups[CASE_EMPTY].append(position)
        elif record.get("action") == "skip":
            groups[CASE_OTHER_SKIP].append(position)
    ranked = sorted(
        (position for position, record in enumerate(prepared) if _evidence_chars(record) > 0),
        key=lambda position: (-_evidence_chars(prepared[position]), position),
    )
    groups[CASE_LONGEST] = ranked[:LONGEST_COUNT]
    return groups


def _anchor(position: int) -> str:
    return f"q-{position + 1:03d}"


def _link(position: int, record: Mapping[str, Any]) -> str:
    return f"[{code_span(record.get('question_id'))}](#{_anchor(position)})"


def _estimate(record: Mapping[str, Any]) -> str:
    estimate = record.get("token_estimate")
    if isinstance(estimate, Mapping) and estimate.get("estimate") is not None:
        return str(estimate["estimate"])
    return MISSING


def _sha_prefix(value: Any) -> str:
    return str(value)[:SHA_PREFIX_LENGTH] if value else MISSING


def _summary_table(prepared: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| # | Question id | Action | Skip reason | Evidence blocks | Evidence chars | Token estimate (heuristic) "
        "| Prompt SHA-256 prefix |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for position, record in enumerate(prepared):
        blocks = len(_evidence_list(record))
        row = [
            str(position + 1),
            _link(position, record).replace("|", "\\|"),
            cell(record.get("action")),
            cell(record.get("skip_reason")),
            str(blocks),
            str(_evidence_chars(record)),
            _estimate(record),
            id_cell(_sha_prefix(record.get("prompt_sha256"))) if record.get("prompt_sha256") else MISSING,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _index_section(prepared: Sequence[Mapping[str, Any]]) -> list[str]:
    groups = case_index(prepared)
    lines = [
        "## Case-type index",
        "",
        "Each question is classified from its question text and evidence text only. English, Chinese (Han) and "
        "Mixed-language are exclusive, and a question with no letters in either text appears in none of them. The "
        "other groups overlap with them. The rules are in the docstring of `faar.prompt_preview`.",
        "",
    ]
    for name in CASE_ORDER:
        positions = groups[name]
        lines.append(f"- **{name}** ({len(positions)})")
        if not positions:
            lines.append("  - none")
            continue
        for position in positions:
            record = prepared[position]
            note = f" ({_evidence_chars(record)} evidence chars)" if name == CASE_LONGEST else ""
            if name in (CASE_EMPTY, CASE_OTHER_SKIP) and record.get("skip_reason"):
                note = f" (reason {code_span(record['skip_reason'])})"
            lines.append(f"  - {_link(position, record)}{note}")
    return lines


def _evidence_table(record: Mapping[str, Any]) -> list[str]:
    evidence = _evidence_list(record)
    if not evidence:
        return ["No evidence blocks."]
    lines = ["| Rank | Chunk id | PDF page | Chars | Text SHA-256 prefix |", "| --- | --- | --- | --- | --- |"]
    for item in evidence:
        page = item.get("pdf_page_number")
        if page is None and isinstance(item.get("page_idx"), int):
            page = item["page_idx"] + 1
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(item.get("rank")),
                    id_cell(item.get("chunk_id")),
                    cell(page),
                    cell(item.get("chars") if item.get("chars") is not None else _len_or_none(item.get("text"))),
                    id_cell(_sha_prefix(item.get("text_sha256"))) if item.get("text_sha256") else MISSING,
                ]
            )
            + " |"
        )
    return lines


def _len_or_none(text: Any) -> int | None:
    return None if text is None else len(str(text))


def _json_line(value: Any) -> str:
    if value is None:
        return MISSING
    return code_span(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _record_section(position: int, record: Mapping[str, Any]) -> list[str]:
    action = record.get("action")
    reason = record.get("skip_reason")
    lines = [
        f'<a id="{_anchor(position)}"></a>',
        f"## {position + 1}. {code_span(record.get('question_id'))}",
        "",
        f"- Document: {code_span(record.get('doc_id')) if record.get('doc_id') else MISSING}",
        f"- Action: {code_span(action) if action else MISSING}"
        + (f", skip reason {code_span(reason)}" if reason else ""),
        f"- Evidence: {len(_evidence_list(record))} blocks, {_evidence_chars(record)} characters",
        f"- Token estimate (heuristic, not a tokenizer count): {_estimate(record)}",
        f"- Input token upper bound: {_json_line(record.get('input_token_upper_bound'))}",
        f"- Max output tokens: {_json_line(record.get('max_output_tokens'))}",
        f"- Cost upper bound per attempt: {_json_line(record.get('cost_upper_bound'))}",
        "",
        "Question:",
        "",
        fenced(str(record.get("question") if record.get("question") is not None else "")),
        "",
        "Evidence:",
        "",
        *_evidence_table(record),
        "",
    ]
    system, user = record.get("system"), record.get("user")
    if system is None or user is None:
        lines += ["No prompt was built for this question.", ""]
    else:
        lines += ["System message:", "", fenced(str(system)), "", "User message:", "", fenced(str(user)), ""]
    lines += [
        "Hashes:",
        "",
        f"- Template: {code_span(record.get('template_id')) if record.get('template_id') else MISSING}",
        f"- Prompt SHA-256: {code_span(record['prompt_sha256']) if record.get('prompt_sha256') else MISSING}",
        f"- Evidence SHA-256: {code_span(record['evidence_sha256']) if record.get('evidence_sha256') else MISSING}",
        f"- Request id: {code_span(record['request_id']) if record.get('request_id') else MISSING}",
        "",
    ]
    return lines


def render_preview(prepared: Sequence[Mapping[str, Any]], *, title: str) -> str:
    """Render prepared-request records as one Markdown document.

    Every record gets a row and a section, whatever its action. The message
    text appears unchanged inside fenced blocks.
    """
    sends = sum(1 for record in prepared if record.get("action") == "send")
    skips = sum(1 for record in prepared if record.get("action") == "skip")
    templates = sorted({str(record["template_id"]) for record in prepared if record.get("template_id")})
    one_line_title = re.sub(r"[\r\n]+", " ", title)
    header = [
        f"# {one_line_title}",
        "",
        "Draft prompt wording for review. It is not an approved prompt (study brief section 15.7 leaves the exact "
        "text to the lead). This file shows what would be sent. Nothing in it was sent.",
        "",
        "Token estimates are a heuristic: about one token per four characters, plus one per CJK character. They are "
        "not tokenizer counts and no budget uses them.",
        "",
        f"- Questions: {len(prepared)} ({sends} send, {skips} skip"
        + (f", {len(prepared) - sends - skips} other" if len(prepared) - sends - skips else "")
        + ")",
        f"- Template: {', '.join(code_span(name) for name in templates) if templates else MISSING}",
        "",
        "## Summary",
        "",
    ]
    lines = header + _summary_table(prepared) + [""] + _index_section(prepared) + [""]
    for position, record in enumerate(prepared):
        lines += _record_section(position, record)
    return "\n".join(lines).rstrip("\n") + "\n"
