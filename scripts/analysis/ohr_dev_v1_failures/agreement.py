"""Cohen's kappa between the agent coding and a second labeller's filled form.

    .local/venv-prebaseline/bin/python scripts/analysis/ohr_dev_v1_failures/agreement.py \
        results/analysis/2026-10-31-ohr-dev-v1-terra-failures/packet/labeller_form.csv

Single-label fields (evidence_impact, passage_in_prompt, answer, primary_cause) get one kappa each.
Multi-label fields (text_defect, gold_flags, repair_hint) get one kappa per label, treating each label as
present or absent, as study brief section 3 asks for per-signal agreement. Rows the labeller left blank in a
field are left out of that field and counted. Writes agreement.json next to the form.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from common import OUT_DIR
from stats import cohens_kappa

SINGLE = ("evidence_impact", "passage_in_prompt", "answer", "primary_cause")
MULTI = ("text_defect", "gold_flags", "repair_hint")


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["question_id"]: r for r in csv.DictReader(f)}


def labels(value: str) -> set[str]:
    return {x.strip() for x in value.replace(",", ";").split(";") if x.strip()}


def main() -> None:
    form_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "packet" / "labeller_form.csv"
    agent, human = read(OUT_DIR / "coding_agent_v1.csv"), read(form_path)
    if set(agent) != set(human):
        raise SystemExit("the form and the agent coding cover different questions")
    out: dict[str, object] = {"form": str(form_path), "annotators": sorted({r.get("annotator", "") for r in human.values()})}
    for field in SINGLE:
        ids = [q for q in agent if human[q][field].strip()]
        out[field] = {"blank": len(agent) - len(ids), **(cohens_kappa([agent[q][field] for q in ids], [human[q][field].strip() for q in ids]) if ids else {})}
    for field in MULTI:
        ids = [q for q in agent if human[q][field].strip()]
        values = sorted(set().union(*(labels(agent[q][field]) | labels(human[q][field]) for q in ids))) if ids else []
        out[field] = {"blank": len(agent) - len(ids), "per_label": {
            v: cohens_kappa([str(v in labels(agent[q][field])) for q in ids], [str(v in labels(human[q][field])) for q in ids]) for v in values}}
    target = form_path.with_name("agreement.json")
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
