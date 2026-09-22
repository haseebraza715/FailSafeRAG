from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.annotation import cohens_kappa, load_label_studio_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa from two independent Label Studio exports.")
    parser.add_argument("--annotator1", type=Path, required=True)
    parser.add_argument("--annotator2", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("annotation/kappa.json"))
    args = parser.parse_args()

    result = cohens_kappa(load_label_studio_labels(args.annotator1), load_label_studio_labels(args.annotator2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Cohen's kappa is below the required 0.65 threshold.")


if __name__ == "__main__":
    main()
