"""Build every offline output of the ohr_dev_v1 terra failure analysis from saved files.

    .local/venv-prebaseline/bin/python scripts/analysis/ohr_dev_v1_failures/analyse.py

Reads the frozen pilot manifests, the baseline run, the agent coding file and any completed headroom runs.
Sends nothing. Writes, under results/analysis/2026-10-31-ohr-dev-v1-terra-failures/:

    case_measures.csv     all 70 questions: retrieval, CER, scores per condition
    failure_table.csv     the 55 coded failures: codes next to measurements
    results.json          aggregates, paired statistics and power analysis
    report.md             the written report, numbers filled from results.json
    packet/               the blind inspection packet for a second labeller (git-ignored; holds document text)

Pages are rendered first if ``pages/`` is missing.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import packet
import report
from common import OUT_DIR, ROOT, load_cases, read_json, read_jsonl
from headroom import CONDITIONS, RUN_IDS, run_dir
from measures import anls, page_cer, page_ned, page_recall_at_k, retrieval_lcs, retrieval_lcs_chars
from stats import cluster_bootstrap_diff, icc_oneway, mcnemar_exact, mcnemar_sample_size

CODING = OUT_DIR / "coding_agent_v1.csv"
CAUSES = ("scoring_artifact", "gold_or_question", "ocr_lost", "ocr_damaged", "retrieval_miss", "model_error")
LCS_PASSAGE_THRESHOLD = 0.95  # OHR-Bench uses LCS > 0.95 to call evidence "included" (paper section 4.3)


def read_coding() -> dict[str, dict[str, str]]:
    with CODING.open(encoding="utf-8", newline="") as f:
        rows = {r["question_id"]: r for r in csv.DictReader(f)}
    return rows


def evidence_text(case: dict[str, Any]) -> list[str]:
    ctx = case["evidence_context"]
    return ctx if isinstance(ctx, list) else [ctx]


def prompt_evidence(user_prompt: str | None) -> str:
    """The evidence blocks of a baseline user message (question and instructions removed)."""
    if not user_prompt:
        return ""
    start = user_prompt.find("[Evidence 1]")
    end = user_prompt.rfind("\n\nAnswer the question using only the evidence.")
    return user_prompt[start:end] if start >= 0 else ""


def condition_scores() -> dict[str, dict[str, dict[str, Any]]]:
    """Per-condition rows keyed by question id. Baseline always; a headroom condition only when complete."""
    out = {"baseline": {}}
    for row in read_jsonl(ROOT / "results" / "development" / "2026-10-31-ohr-dev-v1-baseline-terra-r1" / "scores.jsonl"):
        out["baseline"][row["question_id"]] = row
    for condition in CONDITIONS:
        summary_path = run_dir(condition) / "score_summary.json"
        if summary_path.exists() and read_json(summary_path).get("complete"):
            out[condition] = {r["question_id"]: r for r in read_jsonl(run_dir(condition) / "scores.jsonl")}
    return out


def build_case_rows(cases: list[dict[str, Any]], coding: dict[str, dict[str, str]], scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for c in cases:
        qid = c["question_id"]
        ev_pages = c["evidence_pages"]
        ret_pages = [r["page_idx"] for r in c["retrieved"]]
        evidence = prompt_evidence(c["user_prompt"])
        ctx = evidence_text(c)
        cers = [page_cer(c["noisy_pages"].get(p, ""), c["clean_pages"].get(p, "")) for p in ev_pages]
        neds = [page_ned(c["noisy_pages"].get(p, ""), c["clean_pages"].get(p, "")) for p in ev_pages]
        lcs = sum(retrieval_lcs(evidence, x) for x in ctx) / len(ctx) if evidence else 0.0
        lcs_top2 = 0.0
        if evidence:
            second = evidence.find("[Evidence 3]")
            top2 = evidence if second < 0 else evidence[:second]
            lcs_top2 = sum(retrieval_lcs(top2, x) for x in ctx) / len(ctx)
        lcs_chars = [retrieval_lcs_chars(evidence, x) for x in ctx]
        lcs_chars = [v for v in lcs_chars if v is not None]
        row: dict[str, Any] = {
            "question_id": qid, "doc_id": c["doc_id"], "doc_type": c["doc_type"], "page_count": c["page_count"],
            "evidence_pages": ";".join(map(str, ev_pages)), "evidence_source": c["evidence_source"], "answer_form": c["answer_form"],
            "han_question": "han" in c["question_scripts"], "evidence_page_ocr_status": c["evidence_page_ocr_status"],
            "baseline_status": c["status"], "no_evidence_reason": c["no_evidence_reason"] or "",
            "retrieved_pages": ";".join(map(str, ret_pages)),
            "page_recall_at_1": page_recall_at_k(ret_pages, ev_pages, 1) if ret_pages else 0.0,
            "page_recall_at_3": page_recall_at_k(ret_pages, ev_pages, 3) if ret_pages else 0.0,
            "page_recall_at_5": page_recall_at_k(ret_pages, ev_pages, 5) if ret_pages else 0.0,
            "single_page_document": c["page_count"] == 1,
            "retrieval_lcs_top5": round(lcs, 4), "retrieval_lcs_top2": round(lcs_top2, 4),
            "retrieval_lcs_chars_top5": round(sum(lcs_chars) / len(lcs_chars), 4) if lcs_chars else None,
            "passage_included_chars_lcs95": bool(lcs_chars) and sum(lcs_chars) / len(lcs_chars) > LCS_PASSAGE_THRESHOLD,
            "evidence_page_cer": round(sum(cers) / len(cers), 4) if all(v is not None for v in cers) else None,
            "evidence_page_ned": round(sum(neds) / len(neds), 4) if all(v is not None for v in neds) else None,
            "gold": c["gold"], "baseline_answer": c["answer"], "baseline_abstained": c["abstained"],
        }
        for name, table in scores.items():
            r = table[qid]
            answer = c["answer"] if name == "baseline" else r["answer"]
            row[f"{name}_em"] = r["em"]
            row[f"{name}_f1"] = round(r["f1"], 4)
            row[f"{name}_anls"] = round(anls(answer, c["gold"]), 4)
            if name != "baseline":
                row[f"{name}_answer"] = answer
        code = coding.get(qid)
        for field in ("text_defect", "evidence_impact", "passage_in_prompt", "answer", "gold_flags", "primary_cause", "repair_hint", "note"):
            row[f"code_{field}"] = code[field] if code else ""
        rows.append(row)
    if set(coding) - {r["question_id"] for r in rows if r["baseline_em"] == 0}:
        raise SystemExit("coding file holds a question that is not a baseline EM-0 case")
    if {r["question_id"] for r in rows if r["baseline_em"] == 0} - set(coding):
        raise SystemExit("an EM-0 case has no code in coding_agent_v1.csv")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    columns = columns or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict[str, Any]], scores: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    failures = [r for r in rows if r["baseline_em"] == 0]
    docs = [r["doc_id"] for r in rows]
    result: dict[str, Any] = {"questions": n, "documents": len(set(docs)), "failures": len(failures)}
    result["baseline_partition"] = Counter("exact" if r["baseline_em"] else ("partial" if r["baseline_f1"] > 0 else "zero") for r in rows)
    result["primary_cause"] = {c: sum(r["code_primary_cause"] == c for r in failures) for c in CAUSES}
    result["primary_cause_by_f1"] = {
        c: {"partial": sum(r["code_primary_cause"] == c and r["baseline_f1"] > 0 for r in failures),
            "zero": sum(r["code_primary_cause"] == c and r["baseline_f1"] == 0 for r in failures)} for c in CAUSES}
    result["evidence_impact"] = Counter(r["code_evidence_impact"] for r in failures)
    defect_counts: Counter[str] = Counter()
    for r in failures:
        for d in r["code_text_defect"].split(";"):
            defect_counts[d] += 1
    result["text_defect_labels"] = defect_counts
    flag_counts: Counter[str] = Counter()
    for r in failures:
        for f in filter(None, r["code_gold_flags"].split(";")):
            flag_counts[f] += 1
    result["gold_flags"] = flag_counts
    result["anls_by_cause"] = {c: mean([r["baseline_anls"] for r in failures if r["code_primary_cause"] == c]) for c in CAUSES}
    result["f1_by_cause"] = {c: mean([r["baseline_f1"] for r in failures if r["code_primary_cause"] == c]) for c in CAUSES}
    result["cause_by_impact"] = {c: Counter(r["code_evidence_impact"] for r in failures if r["code_primary_cause"] == c) for c in CAUSES}
    sent = [r for r in rows if r["baseline_status"] == "answered"]
    multi = [r for r in rows if not r["single_page_document"]]
    result["retrieval"] = {
        "page_recall_at_k_all": {k: mean([r[f"page_recall_at_{k}"] for r in rows]) for k in (1, 3, 5)},
        "page_recall_at_k_multipage_docs": {k: mean([r[f"page_recall_at_{k}"] for r in multi]) for k in (1, 3, 5)},
        "multipage_questions": len(multi),
        "retrieval_lcs_top5_sent": mean([r["retrieval_lcs_top5"] for r in sent]),
        "retrieval_lcs_top2_sent": mean([r["retrieval_lcs_top2"] for r in sent]),
        "retrieval_lcs_top5_all": mean([r["retrieval_lcs_top5"] for r in rows]),
        "retrieval_lcs_top2_all": mean([r["retrieval_lcs_top2"] for r in rows]),
        "retrieval_lcs_chars_top5_sent": mean([r["retrieval_lcs_chars_top5"] for r in sent if r["retrieval_lcs_chars_top5"] is not None]),
        "passage_included_chars_lcs95": sum(r["passage_included_chars_lcs95"] for r in rows),
        "sent": len(sent),
    }
    page_rows = {}
    cases = load_cases()
    for c in cases:
        for p in c["evidence_pages"]:
            page_rows[(c["doc_id"], p)] = (page_cer(c["noisy_pages"].get(p, ""), c["clean_pages"].get(p, "")),
                                           page_ned(c["noisy_pages"].get(p, ""), c["clean_pages"].get(p, "")),
                                           "han" in c["question_scripts"] or any("一" <= ch <= "鿿" for ch in c["clean_pages"].get(p, "")[:400]),
                                           c["noisy_pages"].get(p, "").strip() == "")
    vals = list(page_rows.values())
    result["ocr"] = {
        "evidence_pages": len(vals),
        "mean_cer": mean([v[0] for v in vals if v[0] is not None]),
        "mean_ned": mean([v[1] for v in vals if v[1] is not None]),
        "mean_ned_nonempty": mean([v[1] for v in vals if v[1] is not None and not v[3]]),
        "median_ned_nonempty": sorted(v[1] for v in vals if v[1] is not None and not v[3])[len([v for v in vals if v[1] is not None and not v[3]]) // 2],
        "empty_pages": sum(v[3] for v in vals),
        "mean_ned_han_pages": mean([v[1] for v in vals if v[1] is not None and v[2]]),
        "mean_ned_other_pages": mean([v[1] for v in vals if v[1] is not None and not v[2]]),
        "ned_by_doc_type": {t: mean([v[1] for k, v in page_rows.items() if v[1] is not None and k[0].split("/")[0] == t])
                            for t in sorted({k[0].split("/")[0] for k in page_rows})},
        "cer_by_impact": {imp: mean([r["evidence_page_cer"] for r in failures if r["code_evidence_impact"] == imp and r["evidence_page_cer"] is not None])
                          for imp in ("survives", "damaged", "lost")},
    }
    conditions = {}
    base_em = [r["baseline_em"] for r in rows]
    for name in scores:
        em, f1, an = [r[f"{name}_em"] for r in rows], [r[f"{name}_f1"] for r in rows], [r[f"{name}_anls"] for r in rows]
        entry: dict[str, Any] = {"em": mean(em), "f1": mean(f1), "anls": mean(an)}
        if name != "baseline":
            entry["paired"] = {}
            for metric, values in (("em", em), ("f1", f1), ("anls", an)):
                base_values = [r[f"baseline_{metric}"] for r in rows]
                entry["paired"][metric] = {
                    "by_document": cluster_bootstrap_diff(docs, base_values, values),
                    "by_question": cluster_bootstrap_diff(docs, base_values, values, by_cluster=False),
                }
            entry["mcnemar_em"] = mcnemar_exact(base_em, em)
            entry["fixed_by_cause"] = {c: sum(r[f"{name}_em"] == 1 for r in failures if r["code_primary_cause"] == c) for c in CAUSES}
            entry["broke_baseline_exact"] = sum(r["baseline_em"] == 1 and r[f"{name}_em"] == 0 for r in rows)
            summary = read_json(run_dir(name) / "score_summary.json")
            entry["run_id"] = RUN_IDS[name]
            entry["cost_usd"] = summary["measured_cost_usd"]
            entry["counts"] = {k: summary[k] for k in ("answered", "no_evidence", "execution_failed", "abstained")}
        conditions[name] = entry
    result["conditions"] = conditions
    headroom = [c for c in scores if c != "baseline"]
    if headroom:
        best = [max(r[f"{c}_em"] for c in ["baseline", *headroom]) for r in rows]
        result["best_in_hindsight_em"] = mean(best)
    icc = icc_oneway(docs, base_em)
    result["icc_baseline_em"] = icc
    discordant = {name: conditions[name]["mcnemar_em"]["discordant"] / n for name in headroom}
    rates = sorted({0.06, 0.10, 0.15, 0.20, *[round(v, 3) for v in discordant.values()]})
    result["power"] = {
        "delta": 0.05, "alpha": 0.05, "power": 0.80,
        "observed_discordant_rates": discordant,
        "table": [{"discordant_rate": r, "questions": mcnemar_sample_size(0.05, r),
                   "questions_with_design_effect": mcnemar_sample_size(0.05, r, design_effect=icc["design_effect"])} for r in rates],
    }
    return result


def main() -> None:
    if not (OUT_DIR / "render_record.json").exists() or not (OUT_DIR / "pages").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("render_pages.py"))], check=True)
    cases = load_cases()
    coding = read_coding()
    scores = condition_scores()
    rows = build_case_rows(cases, coding, scores)
    write_csv(OUT_DIR / "case_measures.csv", rows)
    failure_columns = [
        "question_id", "doc_id", "doc_type", "evidence_source", "answer_form", "han_question", "evidence_pages",
        "retrieved_pages", "page_recall_at_5", "retrieval_lcs_top5", "evidence_page_cer", "evidence_page_ned",
        "baseline_status", "baseline_f1", "baseline_anls", "gold", "baseline_answer",
        "code_text_defect", "code_evidence_impact", "code_passage_in_prompt", "code_answer", "code_gold_flags",
        "code_primary_cause", "code_repair_hint", "code_note",
    ] + [f"{c}_{m}" for c in scores if c != "baseline" for m in ("em", "f1")]
    write_csv(OUT_DIR / "failure_table.csv", [r for r in rows if r["baseline_em"] == 0], failure_columns)
    results = json.loads(json.dumps(aggregate(rows, scores), ensure_ascii=False, default=dict))
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=dict) + "\n", encoding="utf-8")
    packet.build(cases, rows)
    report.write(results, rows)
    print(json.dumps({k: results[k] for k in ("primary_cause", "retrieval", "ocr")}, indent=2, default=dict))
    print(json.dumps({k: {m: v.get(m) for m in ("em", "f1", "anls")} for k, v in results["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
