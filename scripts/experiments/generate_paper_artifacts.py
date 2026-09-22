from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.paper_artifacts import write_paper_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed FAAR AAAI tables and Figure 1 from real artifacts.")
    parser.add_argument("--main", type=Path, required=True, nargs=3, metavar=("OHR", "MPDOCVQA", "ARXIVQA"))
    parser.add_argument("--ablations", type=Path, required=True, nargs=4, metavar=("A1", "A2", "A3", "A4"))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--result-label", required=True, help="Label of the FAAR result in the Phase 6 analysis JSON.")
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("paper_artifacts"))
    args = parser.parse_args()

    outputs = write_paper_artifacts(args.main, args.ablations, args.analysis, args.pareto, args.result_label, args.out)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
