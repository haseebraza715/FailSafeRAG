from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.final_analysis import pareto_points, summarize_analysis, write_pareto_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the fixed Phase 6 FAAR analysis from real result JSON files.")
    parser.add_argument("--results", type=Path, required=True, nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    analysis = summarize_analysis(args.results, args.baseline)
    analysis["created_at_utc"] = datetime.now(UTC).isoformat()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase6_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    write_pareto_pdf(pareto_points(analysis), args.out / "cost_accuracy_pareto.pdf")
    print(json.dumps({"analysis": str(args.out / 'phase6_analysis.json'), "pareto": str(args.out / 'cost_accuracy_pareto.pdf')}, indent=2))


if __name__ == "__main__":
    main()
