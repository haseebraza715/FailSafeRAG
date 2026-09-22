#!/usr/bin/env python3
"""Validate and merge per-shard OHR asset-preparation manifests.

Consumes the shard manifests written by cluster/prepare_assets.py
(shard_manifest_shardNofM.json), verifies that the shard set is complete and
non-overlapping, that every shard index is in range, that the combined
documents exactly match the locked split, and that no shard declares
uncompleted documents. The merged record is published atomically.

Usage:

    .venv-aaai/bin/python cluster/merge_prep_shards.py \
      --project-root "$PWD" \
      --split val \
      --out results/validation/merged_assets.json \
      results/validation/faar-ohr-val/shard_manifest_shard*.json

Exit codes: 0 on a validated merge; 1 when the split checksum lock fails or a
blocking validation error occurs (SystemExit with the reason).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_assets import (
    SPLIT_CHOICES,
    load_split_documents,
    merge_shard_manifests,
    verify_split_checksums,
)

from faar.run_io import atomic_write_text

SCHEMA_VERSION = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="val",
        help="Split whose locked document set the merged manifests must match exactly.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Merged JSON output path (written atomically)")
    parser.add_argument("manifests", nargs="+", type=Path, help="Shard manifest JSON files")
    args = parser.parse_args(argv)

    project_root = args.project_root.expanduser().resolve()
    verify_split_checksums(project_root)
    doc_names, _example_count = load_split_documents(project_root, args.split)
    merged = merge_shard_manifests([path.expanduser().resolve() for path in args.manifests], expected_documents=doc_names)
    if merged["split"] != args.split:
        raise SystemExit(
            f"manifests declare split {merged['split']!r} but --split {args.split!r} was requested; "
            "refusing to publish a mismatched merge."
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "merged-asset-preparation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": merged["dataset"],
        "split": merged["split"],
        "num_shards": merged["num_shards"],
        "documents": merged["documents"],
        "documents_failed": merged["documents_failed"],
        "pages_completed": merged["pages_completed"],
        "runtime_sec": merged["runtime_sec"],
        "bounds_hit": merged["bounds_hit"],
        "merged_from": [str(path) for path in args.manifests],
    }
    atomic_write_text(args.out, json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"[merge-prep-shards] merged validation record written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
