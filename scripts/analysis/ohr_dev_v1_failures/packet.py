"""Blind inspection packet for a second labeller of the 55 baseline failures.

The packet shows what the first coder saw and nothing the first coder decided: page images, MinerU text,
the question, the gold answer, the prompt evidence and the model reply. It never reads coding_agent_v1.csv.
Cases appear in a fixed order sorted by a hash of the question id, so the order carries no hint from the
coding or the scores. Written to ``packet/`` (git-ignored, because it holds full document text).
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from typing import Any

from common import OUT_DIR, read_json

FORM_COLUMNS = ["case", "question_id", "text_defect", "evidence_impact", "passage_in_prompt", "answer", "gold_flags",
                "primary_cause", "repair_hint", "note", "annotator"]

STYLE = """
body{font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px;color:#1b1b1b;background:#fff}
h1{font-size:22px} h2{font-size:18px;border-top:2px solid #444;padding-top:12px;margin-top:40px}
img{max-width:100%;border:1px solid #bbb} pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;font-size:13px;max-height:420px;overflow:auto}
details{margin:10px 0;padding:6px 10px;border:1px solid #ccc;border-radius:6px} summary{cursor:pointer;font-weight:600}
.step{color:#555;font-size:13px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:3px 6px;font-size:13px}
"""


def _order_key(qid: str) -> str:
    return hashlib.sha256(("packet-order-v1:" + qid).encode()).hexdigest()


def build(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    failures = {r["question_id"] for r in rows if r["baseline_em"] == 0}
    chosen = sorted((c for c in cases if c["question_id"] in failures), key=lambda c: _order_key(c["question_id"]))
    render = {(r["doc_id"], r["page_idx"]): r for r in read_json(OUT_DIR / "render_record.json")["pages"]}
    out = OUT_DIR / "packet"
    out.mkdir(exist_ok=True)
    e = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Failure Labelling Packet</title><style>{STYLE}</style></head><body>",
        "<h1>ohr_dev_v1 baseline failures: labelling packet</h1>",
        "<p>55 questions that the no-recovery baseline (gpt-6.1-terra) did not answer with an exact match. "
        "Label each case with the codebook (<code>../codebook.md</code>) in <code>labeller_form.csv</code>. "
        "Work top to bottom and open each section only when its step comes:</p><ol>"
        "<li>Compare the page image with the MinerU text. Record <b>text_defect</b>. Do not open the later sections yet.</li>"
        "<li>Open <i>Question and gold answer</i>. Record <b>evidence_impact</b> and <b>gold_flags</b>.</li>"
        "<li>Open <i>What the model saw and said</i>. Record <b>passage_in_prompt</b>, <b>answer</b>, <b>primary_cause</b> and <b>repair_hint</b>.</li>"
        "</ol><p>Write your name or initials in <code>annotator</code>. Do not use a model's suggestions, and do not open "
        "<code>coding_agent_v1.csv</code> or <code>failure_table.csv</code> before you finish. The clean reference text is "
        "OHR-Bench's own transcription and has errors of its own.</p>",
    ]
    form = []
    for number, c in enumerate(chosen, start=1):
        qid = c["question_id"]
        form.append({"case": number, "question_id": qid, **{k: "" for k in FORM_COLUMNS[2:]}})
        parts.append(f"<h2 id='case-{number}'>Case {number} <span class='step'>({e(qid)}, {e(c['doc_type'])}, "
                     f"{c['page_count']}-page document, evidence page(s) {', '.join(str(p + 1) for p in c['evidence_pages'])})</span></h2>")
        parts.append("<p class='step'>Step 1: page image and MinerU text</p>")
        for p in c["evidence_pages"]:
            img = render[(c["doc_id"], p)]
            noisy = c["noisy_pages"].get(p)
            noisy_html = "<p><i>No MinerU entry for this page.</i></p>" if noisy is None else (
                "<p><i>MinerU text is empty.</i></p>" if not noisy.strip() else f"<pre>{e(noisy)}</pre>")
            parts.append(f"<h3>Page {p + 1}</h3><img loading='lazy' src='../{e(img['png'])}' alt='page {p + 1}' "
                         f"width='{img['png_size'][0]}' height='{img['png_size'][1]}'><p><b>MinerU text</b></p>{noisy_html}"
                         f"<details><summary>Clean reference text of page {p + 1} (only after step 1)</summary>"
                         f"<pre>{e(c['clean_pages'].get(p, '(none)'))}</pre></details>")
        context = c["evidence_context"] if isinstance(c["evidence_context"], list) else [c["evidence_context"]]
        parts.append("<details><summary>Step 2: question and gold answer</summary>"
                     f"<p><b>Question:</b> {e(c['question'])}</p><p><b>Gold answer:</b> {e(c['gold'])}</p>"
                     f"<p><b>Gold evidence context</b> (from the clean reference):</p><pre>{e(chr(10).join(context))}</pre></details>")
        if c["user_prompt"]:
            prompt_html = f"<pre>{e(c['user_prompt'])}</pre>"
        else:
            prompt_html = f"<p><i>No request was sent: {e(c['no_evidence_reason'] or 'no evidence')}.</i></p>"
        reply = "(no reply: nothing was sent)" if c["raw_reply"] is None else c["raw_reply"]
        parts.append("<details><summary>Step 3: what the model saw and said</summary>"
                     f"<p><b>Model reply:</b> {e(reply)}</p><p><b>Prompt sent</b> (user message; the system message is the "
                     f"same for every case):</p>{prompt_html}</details>")
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    form_path = out / "labeller_form.csv"
    if not form_path.exists():  # never overwrite a form someone may have started
        with form_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FORM_COLUMNS)
            writer.writeheader()
            writer.writerows(form)
    (out / "packet_record.json").write_text(json.dumps({
        "cases": len(chosen), "order": "sha256('packet-order-v1:' + question_id)",
        "question_ids_in_order": [c["question_id"] for c in chosen],
        "contains_agent_codes": False,
    }, indent=2) + "\n", encoding="utf-8")
