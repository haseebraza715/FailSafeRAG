#!/usr/bin/env python3
"""Phase 0 one-document B0 end-to-end smoke CLI.

Uses prepared assets under data/benchmark_prep/smoke/ only.
Does not modify config/datasets/ohr_split.json, does not run the full dataset, and does not
count as a paper result. Default retrieval is a local lexical mock (no HF
downloads, no paid VLM calls). Pass --use-pinned-models to load the locked
NV-Embed-v2 + bge-reranker-v2-m3 stack when available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The default smoke uses the lexical retriever but still imports the shared
# graph module. Keep that import path stable in constrained subprocesses where
# OpenMP shared-memory initialization can abort before the smoke starts.
if "--use-pinned-models" not in sys.argv:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_INIT_AT_FORK"] = "TRUE"

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.settings import AppSettings
from faar.smoke_b0 import (
    DEFAULT_SMOKE_DOC,
    DEFAULT_SMOKE_SPLIT,
    REQUIRED_SUMMARY_FIELDS,
    run_one_document_b0_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clearly marked one-document B0 smoke after OHR asset preparation. "
            "Not a paper result."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/smoke/b0_one_doc.json"),
        help="Output JSON path (default: results/smoke/b0_one_doc.json).",
    )
    parser.add_argument(
        "--smoke-root",
        type=Path,
        default=Path("data/benchmark_prep/smoke"),
        help="Prepared one-document asset root.",
    )
    parser.add_argument(
        "--doc",
        default=DEFAULT_SMOKE_DOC,
        help=f"Smoke document id (default: {DEFAULT_SMOKE_DOC}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SMOKE_SPLIT,
        help="Immutable split used to select real QA rows (default: val).",
    )
    parser.add_argument(
        "--use-pinned-models",
        action="store_true",
        help=(
            "Load locked NV-Embed-v2 and bge-reranker-v2-m3 instead of the default "
            "lexical mock. Still not a paper result; may download models from HF."
        ),
    )
    parser.add_argument(
        "--no-routing-probe",
        action="store_true",
        help="Skip the secondary diagnosis/recovery routing probe.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    args = parser.parse_args()

    settings = AppSettings(project_root=args.project_root.resolve())
    # Smoke keeps pinned snapshot metadata for provenance without requiring
    # runtime path phase0 fixtures beyond prepared smoke assets.
    try:
        settings.validate_openai_snapshot()
        settings.validate_model_revisions()
    except ValueError as exc:
        raise SystemExit(f"Pinned model contract failed: {exc}") from exc

    payload = run_one_document_b0_smoke(
        settings,
        out_path=args.out,
        doc_name=args.doc,
        split=args.split,
        smoke_root=args.smoke_root,
        mock_retrieval=not args.use_pinned_models,
        probe_routing=not args.no_routing_probe,
    )
    summary = payload["summary"]
    if payload.get("paper_result") is not False or not payload.get("smoke"):
        raise SystemExit("Smoke payload must set smoke=true and paper_result=false.")

    print(
        json.dumps(
            {
                "mode": "b0_one_doc_smoke",
                "paper_result": False,
                "smoke": True,
                "out": str(args.out),
                "smoke_doc": args.doc,
                "split": args.split,
                "mock_retrieval": not args.use_pinned_models,
                "summary": {field: summary[field] for field in REQUIRED_SUMMARY_FIELDS},
                "pipeline_stages": payload.get("pipeline_stages"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
