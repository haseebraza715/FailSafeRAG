# Commands

Run these commands from the repository root with the pinned virtual environment.

| Directory | Purpose |
| --- | --- |
| `experiments/` | Baselines, evaluation, gate tuning, analysis, and paper artifacts |
| `data/` | Dataset splits, preparation, remapping, and asset registration |
| `annotation/` | Failure sampling, OCR extraction, and agreement measurement |
| `smoke/` | Bounded local smoke checks that do not count as paper results |
| `release/` | Pre-submission repository and paper checks |
| `demo/` | Historical offline demonstration |

Shared-cluster jobs use `cluster/`. The exact supervisor procedure is in
[`docs/operations/runbook.md`](../docs/operations/runbook.md).
