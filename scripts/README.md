# Commands

Run these commands from the repository root with the pinned virtual environment.

| Directory | Purpose |
| --- | --- |
| `experiments/` | Answer-model runs and repairs (`run_pilot_live.py`, `run_text_repair.py`, `run_visual_repair.py`), the gate and diagnosis (`diagnose_run.py`, `choose_confidence_gate.py`), the prototype baselines, evaluation, gate tuning and paper artifacts, and the run registry (`registry.py`) |
| `analysis/` | Comparisons that score saved runs and write the study reports: `ohr_dev_v1_failures/`, `ohr_dev_v2_confirmation/`, `faar_long_dev/` (practice rounds), `faar_confirmation/` and `faar_test/` (final test). Only `ohr_dev_v1_failures/headroom.py` sends paid requests |
| `data/` | Dataset splits, preparation, remapping, and asset registration |
| `diagnostics/` | `openmp_check.py`, which checks the macOS OpenMP conflict between `faiss` and `torch` |
| `audits/` | Reproducible offline audits of a dry run's prompts, cost bounds, the live-check selection and the agent review counts (see [`audits/README.md`](audits/README.md)) |
| `annotation/` | Failure sampling, OCR extraction, and agreement measurement |
| `smoke/` | Bounded local smoke checks that do not count as paper results |
| `release/` | Pre-submission repository and paper checks |
| `demo/` | Historical offline demonstration |

Shared-cluster jobs use `cluster/`. The exact supervisor procedure is in
[`docs/operations/runbook.md`](../docs/operations/runbook.md).
