from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.annotation import sample_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample incorrect B0 predictions for the FAAR double-annotation study.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = sample_failures(args.baseline, args.n, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "baseline": str(args.baseline),
        "seed": args.seed,
        "count": len(samples),
        "samples": samples,
    }
    destination = args.out / "samples.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(destination), "count": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
