"""Compare one frozen final test sample without sending requests."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/analysis/faar_confirmation"))
import compare_confirmation as confirmation  # noqa: E402


def compare(root: Path, sample: str) -> dict[str, Any]:
    if sample not in ("primary", "secondary"):
        raise ValueError("unknown sample")
    primary = sample == "primary"
    kind = "multipage" if primary else "singlepage"
    pilot = f"ohr_test_{kind}_v1"
    prefix = f"2026-11-01-ohr-test-{kind}-v1-"
    suffixes = {"baseline": "baseline-answer-v2-terra", "diagnosis": "diagnosis",
                "invoke_vlm": "visual-terra", "retry_retrieval": "retry-keep-top3-terra",
                "correct_text": "correct-symspell-terra"}
    label = ("Final test result (pre-registered, section 15; primary)" if primary else
             "Secondary result (pre-registered, section 15): outside the primary Holm family; cannot change the primary decision")
    return confirmation.compare(root, pilot=pilot, runs={k: prefix + v for k, v in suffixes.items()},
                                output=f"results/analysis/{prefix}final-test", allowed_split="test",
                                title=f"{pilot} final test policy comparison ({sample})", label=label,
                                heading="Final test decision" if primary else "Secondary result",
                                script_path=Path(__file__))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("primary", "secondary"), required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        compare(args.project_root.resolve(), args.sample)
    except (confirmation.shared.ComparisonError, OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"comparison refused: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
