from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.gate_tuning import (
    THETA_GRID,
    load_gate_examples,
    require_validation_payload,
    search_threshold,
    write_locked_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the FAAR quality-gate threshold on validation results only.")
    parser.add_argument("--baseline", type=Path, required=True, help="Validation B0 result JSON with per-example reranker scores.")
    parser.add_argument("--out", type=Path, default=Path("results/gate_threshold_search.json"))
    parser.add_argument("--lock", type=Path, default=Path("config/gate_threshold.json"))
    args = parser.parse_args()

    require_validation_payload(args.baseline)
    search = search_threshold(load_gate_examples(args.baseline), THETA_GRID)
    search["created_at_utc"] = datetime.now(UTC).isoformat()
    search["source_results"] = str(args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(search, indent=2) + "\n")
    write_locked_threshold(args.lock, search, source=args.baseline)
    print(json.dumps(search, indent=2))


if __name__ == "__main__":
    main()
