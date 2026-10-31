"""Pre-registered comparisons on the ohr_dev_v2 confirmation sample (repair analysis plan, sections 10 and 12).

    PYTHONPATH=src .local/venv-prebaseline/bin/python scripts/analysis/ohr_dev_v2_confirmation/compare.py

Committed before the first run on ohr_dev_v2. It reads the runs named below, the frozen runtime and evaluation
manifests, and nothing else. It sends nothing. A run that does not exist yet is left out, and so is every contrast
that needs it. Writes, under results/analysis/2026-10-31-ohr-dev-v2-confirmation/:

    case_scores.csv    one row per question: subgroup flags, gate flag, and status, answer and scores per policy
    results.json       every number in the report
    report.md          the comparisons, numbers only; interpretation is added separately after reading them

Contrasts
---------
prompt   faar-answer-v2 minus faar-answer-draft-v1, both without recovery (amendment 1, item 2). Own decision rule.
repair   visual repair on every question minus no recovery (v2). Primary contrast of the first experiment.
gated    gated visual repair (abstain-or-ocr-gap-v1) minus no recovery (v2). Secondary, no decision rule.

Each contrast: official EM (primary), F1, ANLS and NFKC EM; paired difference with a 95% bootstrap interval that
resamples documents (10,000 resamples, seed 20261031) and one that resamples questions; McNemar's exact two-sided
test on EM. Decision rule for prompt and repair: "raised EM" only if McNemar's p < 0.05 and the document-level
interval lies above 0. Execution failures score 0; contrasts are also reported on the questions both policies
answered. Subgroups are descriptive only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "analysis" / "ohr_dev_v1_failures"))

from measures import anls  # noqa: E402
from scipy.stats import beta  # noqa: E402
from stats import cluster_bootstrap_diff, icc_oneway, mcnemar_exact, mcnemar_sample_size  # noqa: E402

from faar.ohr_scoring import exact_match_score, f1_score  # noqa: E402
from faar.visual_repair import gate_flags  # noqa: E402

PILOT_DIR = ROOT / "results" / "pilots" / "ohr_dev_v2"
DEV = ROOT / "results" / "development"
RUNS = {
    "v1": DEV / "2026-10-31-ohr-dev-v2-baseline-answer-v1-terra-r1",
    "v2": DEV / "2026-10-31-ohr-dev-v2-baseline-answer-v2-terra-r1",
    "repair": DEV / "2026-10-31-ohr-dev-v2-visual-repair-answer-v2-terra-r1",
}
OUT = ROOT / "results" / "analysis" / "2026-10-31-ohr-dev-v2-confirmation"
METRICS = ("em", "f1", "anls", "nfkc_em")
CONTRASTS = {"prompt": ("v1", "v2"), "repair": ("v2", "repair"), "gated": ("v2", "gated")}
DECISION_RULE = {"prompt", "repair"}
LABELS = {
    "v1": "No recovery, faar-answer-draft-v1",
    "v2": "No recovery, faar-answer-v2",
    "repair": "Visual repair on every question",
    "gated": "Gated visual repair (abstain-or-ocr-gap-v1)",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nfkc_em(prediction: str, gold: str) -> int:
    """FAAR-defined secondary score: official EM after NFKC normalisation of both strings (as in compare_v2.py)."""
    return exact_match_score(unicodedata.normalize("NFKC", prediction), unicodedata.normalize("NFKC", gold))


# --------------------------------------------------------------------------- loading


def baseline_events(run_dir: Path) -> dict[str, Any]:
    """Unknown outcomes, reconciliations (resends under amendment 1) and attempts of a run_pilot_live run."""
    events = read_jsonl(run_dir / "attempts.jsonl")
    unknown = sorted({e["question_id"] for e in events if e["event"] == "outcome_unknown"})
    resent = sorted({e.get("question_id") for e in events if e["event"] == "reconciled" and e.get("resolution") == "allow_new_attempt"})
    return {"questions_with_unknown_outcome": unknown, "resent": resent,
            "attempts": sum(e["event"] == "dispatch_started" for e in events)}


def load_baseline(run_dir: Path) -> dict[str, Any]:
    predictions = {p["question_id"]: p for p in read_jsonl(run_dir / "predictions.jsonl")}
    summary = read_json(run_dir / "run_summary.json")
    rows = {}
    for qid, p in predictions.items():
        rows[qid] = {"status": p["status"], "answer": p["answer"] or "", "abstained": bool(p["abstained"]),
                     "cost": p.get("measured_cost") or 0.0, "images": 0}
    return {"run_id": run_dir.name, "rows": rows, "measured_usd": summary["cost"]["measured"],
            "reserved_usd": summary["cost"]["reserved"], "run_state": summary["run_state"],
            "valid_baseline": summary["valid_baseline"], "execution": baseline_events(run_dir)}


def load_repair(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "run_summary.json")
    identity = read_json(run_dir / "run_config.json")
    rows = {}
    for p in read_jsonl(run_dir / "predictions.jsonl"):
        rows[p["question_id"]] = {"status": p["status"], "answer": p["answer"] or "", "abstained": bool(p["abstained"]),
                                  "cost": p.get("measured_cost") or 0.0, "images": p["images"]}
    return {"run_id": run_dir.name, "rows": rows, "measured_usd": summary["measured_cost_usd"],
            "reserved_usd": summary["reserved_cost_usd"], "complete": summary["complete"],
            "baseline_run": identity["baseline_run"],
            "execution": {"questions_with_unknown_outcome": summary["questions_with_unknown_outcomes"],
                          "resent": summary["resent"], "blocked": summary["blocked"]}}


def load_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for key in ("v1", "v2"):
        if (RUNS[key] / "run_summary.json").exists():
            runs[key] = load_baseline(RUNS[key])
    if (RUNS["repair"] / "run_summary.json").exists():
        runs["repair"] = load_repair(RUNS["repair"])
        if runs["repair"]["baseline_run"] != RUNS["v2"].name:
            raise SystemExit(f"the repair run used {runs['repair']['baseline_run']}, not {RUNS['v2'].name}")
    return runs


def case_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    runtime = read_json(PILOT_DIR / "runtime_manifest.json")
    evaluation = read_json(PILOT_DIR / "evaluation_manifest.json")["questions"]
    docs = {d["doc_id"]: d for d in runtime["documents"]}
    rows = []
    for q in runtime["questions"]:
        qid, doc = q["question_id"], docs[q["doc_id"]]
        ev = evaluation[qid]
        gold = ev["answers"]
        row: dict[str, Any] = {
            "question_id": qid, "doc_id": q["doc_id"], "doc_type": ev["doc_type"], "gold": gold,
            "evidence_page_ocr_gap": ev["evidence_page_ocr_status"] != "ok", "han_script": "han" in ev["question_scripts"],
            "multi_page_document": len(doc["pages"]) > 1, "multi_page_evidence": ev["multi_page_evidence"],
        }
        if "v2" in runs:
            initial = runs["v2"]["rows"][qid]
            row["gate_flagged"] = gate_flags(initial_status=initial["status"], abstained=initial["abstained"], pages=doc["pages"])
        policies = dict(runs)
        if "repair" in runs and "v2" in runs:
            policies["gated"] = {"rows": {qid: runs["repair"]["rows"][qid] if row["gate_flagged"] else runs["v2"]["rows"][qid]}}
        for key, run in policies.items():
            r = run["rows"][qid]
            row[f"{key}:status"] = r["status"]
            row[f"{key}:answer"] = r["answer"]
            row[f"{key}:em"] = exact_match_score(r["answer"], gold)
            row[f"{key}:f1"] = f1_score(r["answer"], gold)
            row[f"{key}:anls"] = anls(r["answer"], gold)
            row[f"{key}:nfkc_em"] = nfkc_em(r["answer"], gold)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- analysis


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def clopper_pearson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def contrast(rows: list[dict[str, Any]], base: str, cond: str, decision: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"base": base, "condition": cond}
    served = [r for r in rows if r[f"{base}:status"] != "execution_failed" and r[f"{cond}:status"] != "execution_failed"]
    for view, subset in (("all_questions", rows), ("served_by_both", served)):
        docs = [r["doc_id"] for r in subset]
        entry: dict[str, Any] = {"n": len(subset)}
        for m in METRICS:
            b, c = [r[f"{base}:{m}"] for r in subset], [r[f"{cond}:{m}"] for r in subset]
            entry[m] = {"base": mean(b), "condition": mean(c),
                        "by_document": cluster_bootstrap_diff(docs, b, c),
                        "by_question": cluster_bootstrap_diff(docs, b, c, by_cluster=False)}
        entry["mcnemar_em"] = mcnemar_exact([r[f"{base}:em"] for r in subset], [r[f"{cond}:em"] for r in subset])
        out[view] = entry
    if decision:
        primary = out["all_questions"]
        out["decision"] = {
            "rule": "raised EM only if McNemar p < 0.05 and the document-level 95% interval lies above 0 (all questions)",
            "mcnemar_p": primary["mcnemar_em"]["p_value"], "interval": [primary["em"]["by_document"]["ci_low"], primary["em"]["by_document"]["ci_high"]],
            "raised_em": primary["mcnemar_em"]["p_value"] < 0.05 and primary["em"]["by_document"]["ci_low"] > 0,
        }
    return out


def repair_damage(rows: list[dict[str, Any]], base: str, cond: str) -> dict[str, Any]:
    """Brief section 8: repair rate over flagged questions wrong at first; damage rate over those right at first."""
    flagged = [r for r in rows if r["gate_flagged"]]
    wrong = [r for r in flagged if not r[f"{base}:em"]]
    right = [r for r in flagged if r[f"{base}:em"]]
    repaired = sum(r[f"{cond}:em"] for r in wrong)
    damaged = sum(1 - r[f"{cond}:em"] for r in right)
    return {"flagged": len(flagged), "wrong_at_first": len(wrong), "repaired": repaired,
            "repair_rate": repaired / len(wrong) if wrong else None, "repair_rate_ci": clopper_pearson(repaired, len(wrong)),
            "right_at_first": len(right), "damaged": damaged,
            "damage_rate": damaged / len(right) if right else None, "damage_rate_ci": clopper_pearson(damaged, len(right))}


SUBGROUPS = ("evidence_page_ocr_gap", "han_script", "multi_page_document", "multi_page_evidence")


def subgroups(rows: list[dict[str, Any]], policies: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for flag in SUBGROUPS:
        for value in (True, False):
            subset = [r for r in rows if r[flag] is value]
            out[f"{flag}={str(value).lower()}"] = {"n": len(subset), **{p: mean([r[f"{p}:em"] for r in subset]) for p in policies}}
    for doc_type in sorted({r["doc_type"] for r in rows}):
        subset = [r for r in rows if r["doc_type"] == doc_type]
        out[f"doc_type={doc_type}"] = {"n": len(subset), **{p: mean([r[f"{p}:em"] for r in subset]) for p in policies}}
    return out


def analyse(runs: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    policies = [p for p in ("v1", "v2", "repair", "gated") if f"{p}:em" in rows[0]]
    result: dict[str, Any] = {"pilot_id": PILOT_DIR.name, "questions": n, "documents": len({r["doc_id"] for r in rows}),
                              "policies": {}, "contrasts": {}}
    for p in policies:
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[f"{p}:status"]] = counts.get(r[f"{p}:status"], 0) + 1
        entry: dict[str, Any] = {"label": LABELS[p], "status_counts": counts, **{m: mean([r[f"{p}:{m}"] for r in rows]) for m in METRICS}}
        if p in runs:
            run = runs[p]
            entry.update(run_id=run["run_id"], measured_usd=run["measured_usd"], reserved_usd=run["reserved_usd"], execution=run["execution"])
            entry["cost_per_question_usd"] = (run["measured_usd"] + run["reserved_usd"]) / n
        if p in ("repair",):
            entry["images_per_request"] = mean([runs["repair"]["rows"][r["question_id"]]["images"] for r in rows])
        result["policies"][p] = entry
    if "gated" in policies:
        flagged = [r["question_id"] for r in rows if r["gate_flagged"]]
        repair_cost = sum(runs["repair"]["rows"][q]["cost"] for q in flagged)
        result["policies"]["gated"]["flagged"] = len(flagged)
        result["policies"]["gated"]["cost_note"] = (
            "a deployed gated policy sends repair requests only for flagged questions; its cost is the v2 no-recovery "
            "cost plus the measured repair cost of the flagged questions")
        result["policies"]["gated"]["cost_per_question_usd"] = (runs["v2"]["measured_usd"] + runs["v2"]["reserved_usd"] + repair_cost) / n
    for name, (base, cond) in CONTRASTS.items():
        if base in policies and cond in policies:
            result["contrasts"][name] = contrast(rows, base, cond, name in DECISION_RULE)
    if "repair" in policies:
        result["repair_and_damage"] = {p: repair_damage(rows, "v2", p) for p in ("repair", "gated")}
    result["subgroups_em"] = subgroups(rows, policies)
    if "v2" in policies:
        icc = icc_oneway([r["doc_id"] for r in rows], [r["v2:em"] for r in rows])
        result["icc_v2_em"] = icc
        result["detectable"] = {
            name: {"discordant_rate": c["all_questions"]["mcnemar_em"]["discordant"] / n,
                   "questions_for_5_points": mcnemar_sample_size(0.05, c["all_questions"]["mcnemar_em"]["discordant"] / n),
                   "questions_for_5_points_with_design_effect": mcnemar_sample_size(
                       0.05, c["all_questions"]["mcnemar_em"]["discordant"] / n, design_effect=icc["design_effect"])}
            for name, c in result["contrasts"].items()}
    return result


# --------------------------------------------------------------------------- report


def num(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# {results['pilot_id']} comparisons: no recovery and visual repair")
    a("")
    a("Generated by `scripts/analysis/ohr_dev_v2_confirmation/compare.py` from `results.json`. Numbers only: the plan is")
    a("`docs/research/repair-analysis-plan.md` (sections 10 and 12). EM is official OHR-Bench EM, the primary score.")
    a(f"{results['questions']} questions on {results['documents']} documents. Execution failures score 0.")
    a("")
    a("## 1. Policies")
    a("")
    a("| Policy | Run | EM | F1 | ANLS | NFKC EM | Status counts | Cost per question (USD) |")
    a("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for p, e in results["policies"].items():
        counts = ", ".join(f"{k} {v}" for k, v in sorted(e["status_counts"].items()))
        a(f"| {e['label']} | `{e.get('run_id', 'from repair and v2 runs')}` | {num(e['em'])} | {num(e['f1'])} | {num(e['anls'])} | "
          f"{num(e['nfkc_em'])} | {counts} | {num(e.get('cost_per_question_usd'), 5)} |")
    a("")
    a("## 2. Execution: unknown outcomes, resends and blocked questions")
    a("")
    a("| Policy | Measured USD | Reserved USD | Questions with an unknown outcome | Resent (amendment 1) | Still blocked or failed |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for p, e in results["policies"].items():
        if "execution" not in e:
            continue
        x = e["execution"]
        unknown = x["questions_with_unknown_outcome"]
        unknown = len(unknown) if isinstance(unknown, list) else unknown
        resent = len(x["resent"]) if isinstance(x["resent"], list) else x["resent"]
        failed = e["status_counts"].get("execution_failed", 0)
        a(f"| {e['label']} | {num(e['measured_usd'], 6)} | {num(e['reserved_usd'], 6)} | {unknown} | {resent} | {failed} |")
    if "repair" in results["policies"]:
        a("")
        a(f"Images per repair request: {num(results['policies']['repair']['images_per_request'], 2)}.")
    a("")
    a("## 3. Contrasts")
    a("")
    a("Paired by question. Intervals are 95% bootstrap intervals in EM points (10,000 resamples, seed 20261031).")
    a("")
    a("| Contrast | View | n | EM difference, by document | by question | McNemar gained / lost, p | F1 diff | ANLS diff | NFKC EM diff |")
    a("| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |")
    for name, c in results["contrasts"].items():
        for view in ("all_questions", "served_by_both"):
            e = c[view]
            d, q = e["em"]["by_document"], e["em"]["by_question"]
            mc = e["mcnemar_em"]
            a(f"| {name}: {LABELS[c['condition']]} minus {LABELS[c['base']]} | {view.replace('_', ' ')} | {e['n']} | "
              f"{pts(d['diff'])} [{pts(d['ci_low'])}, {pts(d['ci_high'])}] | {pts(q['diff'])} [{pts(q['ci_low'])}, {pts(q['ci_high'])}] | "
              f"{mc['gained']} / {mc['lost']}, p = {mc['p_value']:.3g} | {pts(e['f1']['by_document']['diff'])} | "
              f"{pts(e['anls']['by_document']['diff'])} | {pts(e['nfkc_em']['by_document']['diff'])} |")
    a("")
    for name, c in results["contrasts"].items():
        if "decision" in c:
            d = c["decision"]
            verdict = "met" if d["raised_em"] else "not met"
            a(f"- **{name}**: decision rule {verdict} (McNemar p = {d['mcnemar_p']:.3g}; document interval "
              f"[{pts(d['interval'][0])}, {pts(d['interval'][1])}] points).")
    if "repair_and_damage" in results:
        a("")
        a("## 4. Repair and damage on the gate's flagged set")
        a("")
        a("Against the v2 no-recovery answer. Exact 95% intervals (Clopper-Pearson).")
        a("")
        a("| Policy | Flagged | Repaired / wrong at first | Repair rate | Damaged / right at first | Damage rate |")
        a("| --- | ---: | --- | --- | --- | --- |")
        for p, e in results["repair_and_damage"].items():
            rr = f"{num(e['repair_rate'])} [{num(e['repair_rate_ci'][0])}, {num(e['repair_rate_ci'][1])}]" if e["repair_rate_ci"] else "n/a"
            dr = f"{num(e['damage_rate'])} [{num(e['damage_rate_ci'][0])}, {num(e['damage_rate_ci'][1])}]" if e["damage_rate_ci"] else "n/a"
            a(f"| {LABELS[p]} | {e['flagged']} | {e['repaired']} / {e['wrong_at_first']} | {rr} | {e['damaged']} / {e['right_at_first']} | {dr} |")
    a("")
    a("## 5. Subgroups (descriptive, no tests)")
    a("")
    policies = [p for p in ("v1", "v2", "repair", "gated") if p in results["policies"]]
    a("| Subgroup | n | " + " | ".join(f"EM {p}" for p in policies) + " |")
    a("| --- | ---: | " + " | ".join("---:" for _ in policies) + " |")
    for name, e in results["subgroups_em"].items():
        a(f"| {name} | {e['n']} | " + " | ".join(num(e[p]) for p in policies) + " |")
    if "detectable" in results:
        a("")
        a("## 6. Sample size check")
        a("")
        a(f"Intraclass correlation of v2 EM within documents: {num(results['icc_v2_em']['icc'])} "
          f"(design effect {num(results['icc_v2_em']['design_effect'], 2)}).")
        a("")
        a("| Contrast | Discordant rate | Questions for a 5-point gain | With the design effect |")
        a("| --- | ---: | ---: | ---: |")
        for name, e in results["detectable"].items():
            a(f"| {name} | {num(e['discordant_rate'])} | {e['questions_for_5_points'] or 'n/a'} | {e['questions_for_5_points_with_design_effect'] or 'n/a'} |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    with (OUT / "case_scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no run on ohr_dev_v2 exists yet")
    rows = case_rows(runs)
    results = analyse(runs, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(report(results), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
