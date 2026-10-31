"""The same comparisons as compare.py, on ohr_dev_v1, for the debug run of the visual repair (development data).

    PYTHONPATH=src .local/venv-prebaseline/bin/python scripts/analysis/ohr_dev_v2_confirmation/debug_v1.py

Writes results/analysis/2026-10-31-ohr-dev-v1-visual-repair-debug/. These numbers are development evidence: the v2
prompt was written on these questions, and nothing here bears on the ohr_dev_v2 confirmation.
"""

from __future__ import annotations

import compare

compare.PILOT_DIR = compare.ROOT / "results" / "pilots" / "ohr_dev_v1"
compare.RUNS = {
    "v1": compare.DEV / "2026-10-31-ohr-dev-v1-baseline-terra-r1",
    "v2": compare.DEV / "2026-10-31-ohr-dev-v1-baseline-answer-v2-terra-r1",
    "repair": compare.DEV / "2026-10-31-ohr-dev-v1-visual-repair-answer-v2-terra-r1",
}
compare.OUT = compare.ROOT / "results" / "analysis" / "2026-10-31-ohr-dev-v1-visual-repair-debug"

if __name__ == "__main__":
    compare.main()
