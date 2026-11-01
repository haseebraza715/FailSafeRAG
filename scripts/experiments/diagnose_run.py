"""Gate and diagnose every question of a no-recovery run, offline. Sends nothing and reads no evaluation file.

    PYTHONPATH=src .local/venv-prebaseline/bin/python scripts/experiments/diagnose_run.py \\
        --pilot-id ohr_long_dev_v1 --baseline-run-dir results/development/<no-recovery run> \\
        --out-dir results/development/<diagnosis id> [--gate-feature query_token_coverage --gate-threshold 0.5]

Writes ``run_config.json`` and ``diagnosis.jsonl`` with a row per question, including gate decisions, reasons,
confidence features, diagnoses, routes and signals. It also writes ``flagged_questions.txt`` in manifest order
and ``flagged_questions_meta.json`` with its hash and count. The retrieval is replayed offline and checked hit for
hit against the evidence the no-recovery run sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import run_visual_repair as rvr

from faar.diagnosis import (
    CONFIDENCE_FEATURES,
    DEFAULT_STRUCTURAL_THRESHOLD,
    DEFAULT_WEIRD_CHAR_THRESHOLD,
    DIAGNOSIS_ID,
    GateRule,
    RetrievedChunk,
    confidence_features,
    diagnose,
    gate,
)
from faar.text_repair import replay_retrieval
from faar.visual_repair import DEFAULT_BUDGET

ROOT = Path(__file__).resolve().parents[2]


def diagnose_run(
    *,
    root: Path,
    pilot_id: str,
    baseline_run_dir: Path,
    out_dir: Path,
    rule: GateRule,
    budget: int = DEFAULT_BUDGET,
    structural_threshold: int = DEFAULT_STRUCTURAL_THRESHOLD,
    weird_char_threshold: float = DEFAULT_WEIRD_CHAR_THRESHOLD,
) -> list[dict[str, Any]]:
    loaded = rvr.load_baseline(root, pilot_id, baseline_run_dir)
    predictions = {p["question_id"]: p for p in rvr._read_jsonl(baseline_run_dir / "predictions.jsonl")}
    replay = replay_retrieval(root, pilot_id, loaded["requests"])
    docs = {d["doc_id"]: d for d in loaded["runtime"]["documents"]}
    rows = []
    for q in loaded["runtime"]["questions"]:
        qid = q["question_id"]
        initial = predictions[qid]
        base = loaded["requests"][qid]
        sent = base["action"] == "send"
        chunks = [RetrievedChunk(rank, h.chunk.page_id, h.fused_score, h.chunk.text) for rank, h in enumerate(replay["hits"][qid], start=1)] if sent else []
        features = confidence_features(q["question"], chunks, replay["tokenizer"])
        pages = docs[q["doc_id"]]["pages"]
        flagged, reasons = gate(initial_status=initial["status"], abstained=bool(initial["abstained"]), pages=pages, features=features, rule=rule)
        result = diagnose(initial_sent=sent, chunks=chunks, pages=pages, budget=budget,
                          structural_threshold=structural_threshold, weird_char_threshold=weird_char_threshold)
        rows.append({"question_id": qid, "doc_id": q["doc_id"], "initial_status": initial["status"],
                     "initial_abstained": bool(initial["abstained"]), "flagged": flagged, "gate_reasons": reasons,
                     "features": features, **result})
    config = {
        "pilot_id": pilot_id, "baseline_run": baseline_run_dir.name, "baseline_requests_sha256": loaded["requests_sha256"],
        "runtime_manifest_sha256": loaded["runtime_sha256"], "gate": rule.gate_id,
        "gate_rule": {"feature": rule.feature, "threshold": rule.threshold}, "diagnosis": DIAGNOSIS_ID,
        "structural_threshold": structural_threshold, "weird_char_threshold": weird_char_threshold, "image_budget": budget,
        "confidence_features": list(CONFIDENCE_FEATURES), "retrieval": replay["retrieval"],
        "code": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                 for rel in ("scripts/experiments/diagnose_run.py", "src/faar/diagnosis.py", "src/faar/quality.py", "src/faar/text_repair.py")},
    }
    flagged = "".join(r["question_id"] + "\n" for r in rows if r["flagged"]).encode("utf-8")
    rvr.write_identity(out_dir, config, None)
    (out_dir / "diagnosis.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "flagged_questions.txt").write_bytes(flagged)
    metadata = {"flagged_questions_sha256": hashlib.sha256(flagged).hexdigest(), "flagged_count": sum(r["flagged"] for r in rows)}
    (out_dir / "flagged_questions_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pilot-id", required=True)
    p.add_argument("--baseline-run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gate-feature", choices=CONFIDENCE_FEATURES)
    p.add_argument("--gate-threshold", type=float)
    p.add_argument("--structural-threshold", type=int, default=DEFAULT_STRUCTURAL_THRESHOLD)
    p.add_argument("--weird-char-threshold", type=float, default=DEFAULT_WEIRD_CHAR_THRESHOLD)
    args = p.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path  # noqa: E731
    rows = diagnose_run(root=ROOT, pilot_id=args.pilot_id, baseline_run_dir=resolve(args.baseline_run_dir), out_dir=resolve(args.out_dir),
                        rule=GateRule(args.gate_feature, args.gate_threshold), structural_threshold=args.structural_threshold,
                        weird_char_threshold=args.weird_char_threshold)
    flagged = [r for r in rows if r["flagged"]]
    routes: dict[str, int] = {}
    for r in flagged:
        routes[r["diagnosis"]] = routes.get(r["diagnosis"], 0) + 1
    print(json.dumps({"questions": len(rows), "flagged": len(flagged), "diagnoses_of_flagged": routes}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
