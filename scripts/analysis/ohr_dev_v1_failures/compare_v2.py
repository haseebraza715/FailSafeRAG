"""Compare the faar-answer-v2 runs with the faar-answer-draft-v1 runs on ohr_dev_v1.

    PYTHONPATH=src .local/venv-prebaseline/bin/python scripts/analysis/ohr_dev_v1_failures/compare_v2.py

Reads the two no-recovery baselines (v1 and v2 templates), the four headroom runs of each template, the frozen
evaluation manifest and the codebook v2 coding of the v1 failures. Sends nothing. Writes, under
results/analysis/2026-10-31-ohr-dev-v1-terra-answer-v2/:

    case_scores.csv    one row per question: status, answer and scores for every run
    results.json       per-run scores, paired contrasts, McNemar tests, per-cause changes, power table
    report.md          the written comparison, numbers filled from results.json

Scores. Official OHR-Bench EM is the primary score (study brief section 8). F1, ANLS and NFKC EM are secondary.
NFKC EM is FAAR-defined: official EM after Unicode NFKC normalisation of both strings, so fullwidth forms such as
``～`` match their ASCII forms. It is reported, not adopted; changing the primary metric is the lead's decision.

Unserved questions. A question whose request ended with an unknown outcome and was not resent is
``execution_failed`` and scores 0 in the all-question view (study brief section 7). Each contrast that involves
such a run is also reported on the questions both runs answered.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from typing import Any

from common import PILOT_DIR, ROOT, read_json, read_jsonl
from headroom import CONDITIONS, V1, V2, run_dir
from measures import anls
from stats import cluster_bootstrap_diff, icc_oneway, mcnemar_exact, mcnemar_sample_size

from faar.ohr_scoring import exact_match_score

OUT = ROOT / "results" / "analysis" / "2026-10-31-ohr-dev-v1-terra-answer-v2"
CODING = ROOT / "results" / "analysis" / "2026-10-31-ohr-dev-v1-terra-failures" / "coding_agent_v2.csv"
METRICS = ("em", "f1", "anls", "nfkc_em")
VARIANTS = {"v1": V1, "v2": V2}
LABELS = {
    "baseline": "No recovery (retrieved MinerU text)",
    "clean_text": "(a) Clean ground-truth text, same retrieval",
    "gold_page_text": "(c) All MinerU text of the gold page(s)",
    "gold_page_image": "(b) Baseline text plus gold page image(s)",
    "gold_page_image_only": "(b') Gold page image(s) only",
}

# Interpretation written after reading results.json; shown only while the hash matches.
INTERPRETATION_FOR_SHA256 = "ac77cb55d40bfadc459eadd714c165f37095c34c710c75e00e40d13d12302c2e"
INTERPRETATION = """\
**The format change helped, but less than the failure coding suggested.** v2 raised official EM from 0.214 to 0.314
on identical evidence. All of the gain came from cases coded `scoring_artifact`: 8 of 27 became exact. v2 also lost
one v1 exact match, where the unit rule cut "apiece" from "$25 to $30 apiece". v2 was written after reading these
cases, so +10 points is an optimistic estimate of what it does on new questions.

**Most format failures are span disagreements that no prompt can fix.** 19 of the 27 `scoring_artifact` cases stay at
EM 0 under v2. Their replies differ from the gold in where the span starts or ends, in list wording ("NPS, BLM"
against "NPS and BLM"), in LaTeX notation, or in a fullwidth tilde. The model cannot know the gold's span convention
without the gold. These cases limit what official EM can show on this pilot, whatever the repair. NFKC EM recovers
one of them. The coding said 42 of 70 questions were answered correctly; v2 reaches 22 exact matches.

**With the format fixed, the page image is the only evidence repair that clearly moves EM.** Against the v2
baseline, text plus gold page image gained 7.1 points over all 70 questions (interval 0.0 to +17.5, McNemar 7 gained
and 2 lost, p = 0.18) and 9.0 points on the 67 questions both runs answered (+1.4 to +19.6, p = 0.07). Clean text and
the full gold-page MinerU text each gained 1.4 points. The image made 4 of the 10 `ocr_lost` cases exact; clean text
made 3. F1 and ANLS rose by about 12 points with the image, so the image also improved answers that EM still scores 0.
No contrast passes McNemar's test at 0.05, and 70 questions cannot settle any of them.

**Clean text is a weak upper bound for text repair here.** Its EM gain fell from 5.7 points under v1 to 1.4 under v2,
because 2 of its 6 v1 gains were format cases that the v2 baseline now answers itself. In both templates it broke
2 exact matches: OHR-Bench's clean reference text has errors of its own on some pages.

**What to do next follows from this.** The visual repair is the repair worth building first. Its effect is likely
6 to 9 EM points at best, with page selection by the gold labels; a runtime page policy can only lose some of that.
Detecting a gain of that size needs a few hundred questions, so the repair comparison should run on a larger
sample (`docs/research/ohr-dev-v2-proposal.md`)."""


def nfkc_em(prediction: str, gold: str) -> int:
    return exact_match_score(unicodedata.normalize("NFKC", prediction), unicodedata.normalize("NFKC", gold))


def load_runs() -> dict[str, dict[str, Any]]:
    """Every run keyed ``<template>:<condition>``, with per-question status, answer and official scores."""
    runs: dict[str, dict[str, Any]] = {}
    for tag, variant in VARIANTS.items():
        base = variant.baseline_dir
        answers = {p["question_id"]: p for p in read_jsonl(base / "predictions.jsonl")}
        rows = {}
        for s in read_jsonl(base / "scores.jsonl"):
            p = answers[s["question_id"]]
            rows[s["question_id"]] = {"status": s["status"], "answer": p["answer"] or "", "em": s["em"], "f1": s["f1"]}
        summary = read_json(base / "run_summary.json")
        runs[f"{tag}:baseline"] = {"run_id": base.name, "rows": rows, "cost_usd": summary["cost"]["measured"],
                                   "complete": summary["run_state"] == "complete"}
        for condition in CONDITIONS:
            rdir = run_dir(condition, variant)
            if not (rdir / "score_summary.json").exists():
                continue
            summary = read_json(rdir / "score_summary.json")
            rows = {r["question_id"]: {k: r[k] for k in ("status", "answer", "em", "f1")} for r in read_jsonl(rdir / "scores.jsonl")}
            runs[f"{tag}:{condition}"] = {"run_id": rdir.name, "rows": rows, "cost_usd": summary["measured_cost_usd"],
                                          "reserved_usd": summary["reserved_cost_usd"], "complete": summary["complete"]}
    return runs


def case_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    runtime = read_json(PILOT_DIR / "runtime_manifest.json")["questions"]
    with CODING.open(encoding="utf-8", newline="") as f:
        cause = {r["question_id"]: r["primary_cause"] for r in csv.DictReader(f)}
    rows = []
    for q in runtime:
        qid, gold = q["question_id"], evaluation[q["question_id"]]["answers"]
        row: dict[str, Any] = {"question_id": qid, "doc_id": q["doc_id"], "gold": gold,
                               "v1_primary_cause": cause.get(qid, "exact_in_v1")}
        for key, run in runs.items():
            r = run["rows"][qid]
            row[f"{key}:status"] = r["status"]
            row[f"{key}:answer"] = r["answer"]
            row[f"{key}:em"] = r["em"]
            row[f"{key}:f1"] = r["f1"]
            row[f"{key}:anls"] = anls(r["answer"], gold)
            row[f"{key}:nfkc_em"] = nfkc_em(r["answer"], gold)
        rows.append(row)
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def contrast(rows: list[dict[str, Any]], base: str, cond: str) -> dict[str, Any]:
    out: dict[str, Any] = {"base": base, "condition": cond}
    views = {"all_questions": rows}
    served = [r for r in rows if r[f"{base}:status"] != "execution_failed" and r[f"{cond}:status"] != "execution_failed"]
    if len(served) < len(rows):
        views["served_by_both"] = served
    for view, subset in views.items():
        docs = [r["doc_id"] for r in subset]
        entry: dict[str, Any] = {"n": len(subset)}
        for m in METRICS:
            b, c = [r[f"{base}:{m}"] for r in subset], [r[f"{cond}:{m}"] for r in subset]
            entry[m] = {"base": mean(b), "condition": mean(c),
                        "by_document": cluster_bootstrap_diff(docs, b, c),
                        "by_question": cluster_bootstrap_diff(docs, b, c, by_cluster=False)}
        entry["mcnemar_em"] = mcnemar_exact([r[f"{base}:em"] for r in subset], [r[f"{cond}:em"] for r in subset])
        out[view] = entry
    return out


def analyse(runs: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    result: dict[str, Any] = {"questions": n, "documents": len({r["doc_id"] for r in rows}), "runs": {}}
    for key, run in runs.items():
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[f"{key}:status"]] = counts.get(r[f"{key}:status"], 0) + 1
        result["runs"][key] = {"run_id": run["run_id"], "complete": run["complete"], "cost_usd": run["cost_usd"],
                               "reserved_usd": run.get("reserved_usd", 0.0), "status_counts": counts,
                               **{m: mean([r[f"{key}:{m}"] for r in rows]) for m in METRICS}}
    pairs = [("v1:baseline", "v2:baseline")]
    for tag in VARIANTS:
        pairs += [(f"{tag}:baseline", f"{tag}:{c}") for c in CONDITIONS if f"{tag}:{c}" in runs]
    result["contrasts"] = {f"{b} -> {c}": contrast(rows, b, c) for b, c in pairs}
    causes = sorted({r["v1_primary_cause"] for r in rows})
    result["v2_baseline_by_v1_cause"] = {
        cause: {"questions": sum(r["v1_primary_cause"] == cause for r in rows),
                "v1_exact": sum(r["v1:baseline:em"] for r in rows if r["v1_primary_cause"] == cause),
                "v2_exact": sum(r["v2:baseline:em"] for r in rows if r["v1_primary_cause"] == cause),
                "v2_nfkc_exact": sum(r["v2:baseline:nfkc_em"] for r in rows if r["v1_primary_cause"] == cause)}
        for cause in causes}
    result["exact_by_v1_cause"] = {
        cause: {key: sum(r[f"{key}:em"] for r in rows if r["v1_primary_cause"] == cause) for key in runs} for cause in causes}
    for tag in VARIANTS:
        headroom = [c for c in CONDITIONS if f"{tag}:{c}" in runs]
        if headroom:
            best = [max(r[f"{tag}:{c}:em"] for c in ["baseline", *headroom]) for r in rows]
            result[f"{tag}_best_in_hindsight_em"] = mean(best)
    icc = icc_oneway([r["doc_id"] for r in rows], [r["v2:baseline:em"] for r in rows])
    result["icc_v2_baseline_em"] = icc
    discordant = {name: c["all_questions"]["mcnemar_em"]["discordant"] / n for name, c in result["contrasts"].items()}
    rates = sorted({0.06, 0.10, 0.15, 0.20, *[round(v, 3) for v in discordant.values() if v > 0.05]})
    result["power"] = {
        "delta": 0.05, "alpha": 0.05, "power": 0.80, "observed_discordant_rates": discordant,
        "table": [{"discordant_rate": rate, "questions": mcnemar_sample_size(0.05, rate),
                   "questions_with_design_effect": mcnemar_sample_size(0.05, rate, design_effect=icc["design_effect"])}
                  for rate in rates],
    }
    result["total_cost_v2_runs_usd"] = sum(r["cost_usd"] + r["reserved_usd"] for k, r in result["runs"].items() if k.startswith("v2:"))
    return result


def num(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def signed(x: float, digits: int = 1) -> str:
    return f"{100 * x:+.{digits}f}"


def interval(d: dict[str, float]) -> str:
    return f"{signed(d['diff'])} [{signed(d['ci_low'])}, {signed(d['ci_high'])}]"


def report(results: dict[str, Any]) -> str:
    sha = hashlib.sha256((OUT / "results.json").read_bytes()).hexdigest()
    runs, con = results["runs"], results["contrasts"]
    L: list[str] = []
    a = L.append
    a("# Answer format: faar-answer-v2 against faar-answer-draft-v1 on ohr_dev_v1")
    a("")
    a(f"Generated by `scripts/analysis/ohr_dev_v1_failures/compare_v2.py` from `results.json` (sha256 `{sha[:12]}`). "
      f"Development data only: {results['questions']} questions from {results['documents']} documents. Nothing here is a "
      "scientific evaluation, and the v2 template was written after reading these questions' v1 failures.")
    a("")
    a("## What is measured, what is interpretation, what is unverified")
    a("")
    a("- **Measured**: every score, difference, interval, McNemar count and p-value, cost, and the per-cause counts below. "
      "The per-cause table groups questions by the agent's v1 failure codes (codebook v2), which are themselves interpretation.")
    a("- **Interpretation**: the section headed so, and any reading of why a score moved.")
    a("- **Unverified**: whether v2's gain holds on questions it was not written against (it was written after reading "
      "these 70); whether the image runs' unknown-outcome requests were billed; how the undated `gpt-6.1-terra` alias "
      "behaves on a rerun.")
    a("")
    a("## 1. Runs")
    a("")
    a("| Run | Template | Condition | Complete | Answered | No evidence | Execution failed | EM | F1 | ANLS | NFKC EM | Cost USD |")
    a("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, r in runs.items():
        tag, cond = key.split(":")
        sc = r["status_counts"]
        cost = num(r["cost_usd"], 4) + (f" (+{num(r['reserved_usd'], 4)} reserved)" if r["reserved_usd"] else "")
        a(f"| `{r['run_id']}` | {tag} | {LABELS[cond]} | {'yes' if r['complete'] else 'no'} | {sc.get('answered', 0)} | "
          f"{sc.get('no_evidence', 0)} | {sc.get('execution_failed', 0)} | {num(r['em'])} | {num(r['f1'])} | {num(r['anls'])} | "
          f"{num(r['nfkc_em'])} | {cost} |")
    a("")
    a(f"Cost of the five v2 runs, counting unknown outcomes at their upper bound: {num(results['total_cost_v2_runs_usd'], 4)} USD.")
    a("")
    a("## 2. Paired differences")
    a("")
    a("Differences are in percentage points, condition minus base, with 95% intervals from a paired bootstrap that "
      "resamples documents (10,000 resamples, seed 20261031). McNemar's exact test is on official EM. Execution failures "
      "score 0 in the all-question rows.")
    a("")
    a("| Contrast | View | n | EM | F1 | ANLS | NFKC EM | McNemar gained / lost | p |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, c in con.items():
        for view in ("all_questions", "served_by_both"):
            if view not in c:
                continue
            v = c[view]
            mc = v["mcnemar_em"]
            a(f"| {name} | {view.replace('_', ' ')} | {v['n']} | " + " | ".join(interval(v[m]["by_document"]) for m in METRICS)
              + f" | {mc['gained']} / {mc['lost']} | {num(mc['p_value'])} |")
    a("")
    a("The question-level bootstrap is in `results.json` next to each document-level interval.")
    a("")
    a("## 3. Where the v2 prompt changed the baseline")
    a("")
    a("Questions grouped by the primary cause the agent coded for the v1 baseline (codebook v2; `exact_in_v1` are the "
      "questions v1 already answered exactly).")
    a("")
    a("| v1 primary cause | Questions | Exact under v1 | Exact under v2 | NFKC-exact under v2 |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for cause, d in results["v2_baseline_by_v1_cause"].items():
        a(f"| `{cause}` | {d['questions']} | {d['v1_exact']} | {d['v2_exact']} | {d['v2_nfkc_exact']} |")
    a("")
    a("Exact matches per run, by the same v1 cause:")
    a("")
    keys = list(runs)
    a("| v1 primary cause | Questions | " + " | ".join(f"`{k}`" for k in keys) + " |")
    a("| --- | ---: | " + " | ".join("---:" for _ in keys) + " |")
    for cause, counts in results["exact_by_v1_cause"].items():
        a(f"| `{cause}` | {results['v2_baseline_by_v1_cause'][cause]['questions']} | " + " | ".join(str(counts[k]) for k in keys) + " |")
    a("")
    a("## 4. Sample size")
    a("")
    icc = results["icc_v2_baseline_em"]
    a(f"Questions needed to detect a 5-point paired EM gain with McNemar's test (alpha 0.05, power 0.80). The design "
      f"effect {num(icc['design_effect'], 2)} comes from the v2 baseline (ICC {num(icc['icc'], 2)}, "
      f"{num(icc['mean_cluster_size'], 2)} questions per document).")
    a("")
    a("| Discordant rate | Questions | With design effect |")
    a("| ---: | ---: | ---: |")
    for row in results["power"]["table"]:
        a(f"| {row['discordant_rate']:.3f} | {row['questions'] or 'n/a'} | {row['questions_with_design_effect'] or 'n/a'} |")
    a("")
    a("Observed discordant rates (all questions): " + ", ".join(f"{k} {v:.3f}" for k, v in results["power"]["observed_discordant_rates"].items()) + ".")
    a("")
    a("## 5. Interpretation")
    a("")
    if INTERPRETATION and INTERPRETATION_FOR_SHA256 == sha:
        a(INTERPRETATION)
    else:
        a("Not written for this version of `results.json`.")
    a("")
    a("## Files")
    a("")
    a("- `case_scores.csv`: every question with status, answer and scores per run.")
    a("- `results.json`: every number above.")
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    rows = case_rows(runs)
    columns = list(rows[0])
    with (OUT / "case_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:  # round for display only
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    results = analyse(runs, rows)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(report(results), encoding="utf-8")
    print(json.dumps({k: {m: v[m] for m in ("em", "f1", "anls", "nfkc_em", "complete")} for k, v in results["runs"].items()}, indent=1))


if __name__ == "__main__":
    main()
