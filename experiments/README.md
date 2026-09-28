# Experiment registry

`registry.jsonl` records every run that produced or will produce evidence:
data preparation, engineering checks, development pilots and scientific
evaluations. It stores identity, state and output checksums, not the outputs
themselves. Result payloads stay where their tools write them (`results/`,
`artifacts/`, `logs/`, cluster scratch).

Validate it with:

```bash
.venv-aaai/bin/python scripts/experiments/registry.py check
```

`check` rejects malformed records, identity changes under an existing run_id,
and recorded outputs whose bytes changed. A tracked output that is missing is an
error; an untracked one (ignored data, cluster outputs) is a warning.

## Current runs

| run_id | Kind | Status |
| --- | --- | --- |
| `proto-phase3-mock-40` | engineering_check | completed (mock backend; not a baseline) |
| `2026-08-03-b0-one-doc-smoke` | engineering_check | completed (smoke; not a baseline) |
| `2026-08-08-arxivqa-full-paper-prep` | data_preparation | completed per run log; bulk outputs absent locally |
| `2026-09-28-ohr-pdf-archive` | data_preparation | completed |
| `2026-09-28-ohr-asset-audit` | data_preparation | completed |
| `2026-09-28-ohr-dev-v1-selection` | data_preparation | completed (pilot sample only; no model run) |
| `faar-ohr-108-calibration` | engineering_check | planned |
| `2026-09-28-ohr-dev-v1-offline-engineering` | engineering_check | completed; outputs kept only in the ignored `.local/work/runs/` because its `run_config.json` records home-directory paths |
| `2026-09-28-ohr-dev-v1-offline-engineering-r2` | engineering_check | completed; superseded by r3 after the review fixes |
| `2026-09-28-ohr-dev-v1-offline-engineering-r3` | engineering_check | completed (rule-based extractor, no repair; not a baseline) |

No scientific evaluation has run. This table is a convenience; the registry is
authoritative.

## Record format (schema_version 1)

Each line is one JSON object: a complete snapshot of one run. The last line for
a run_id is the current state; earlier lines are kept as its history. Never edit
or delete a line; append a new snapshot instead.

Every key is required. Record an unknown value as `null`; never estimate it.

| Key | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `run_id` | Stable ID, lowercase `[a-z0-9._-]`. Dated IDs for new runs, e.g. `2026-10-02-ohr-dev-v1-b0` |
| `recorded_at` | When this snapshot was written (not when the run happened) |
| `kind` | `data_preparation`, `engineering_check`, `development_pilot`, `scientific_evaluation` |
| `status` | `planned`, `running`, `interrupted`, `failed`, `completed`, `abandoned`; equals the latest attempt's status once an attempt exists |
| `purpose` | Hypothesis or reason for the run |
| `dataset` | `{name, split, pilot_id, ...}`: dataset, split or pilot identity |
| `identity` | `{files: {path: sha256}, settings: {...}}`: configuration and source-data hashes and settings that define the run |
| `command` | Exact command, or `null` if not recorded |
| `environment` | Environment reference, e.g. `.venv-aaai` with `config/environment/constraints-aaai.txt` |
| `seed` | Seed, or `null` |
| `models` | `{role: model@revision}`, or `null` |
| `attempts` | List of `{attempt, status, code_commit, dirty, started_at, ended_at, note}`, numbered from 1 |
| `outputs` | List of `{path, sha256, in_git, backup}`. `backup` says where a copy other than this checkout exists; `none` means none |
| `checkpoint` | Resume/checkpoint file, or `null` |
| `related_runs` | `[{run_id, relation}]`, e.g. `uses`, `supersedes`, `repeats` |
| `limitations` | What the result cannot show |
| `evidence` | Documents that support historical values in the record |

`kind` follows what the run measured, not what it was meant for. Mock-backend
or smoke runs are `engineering_check` even when they compute EM/F1. Only a run
on the locked protocol with real models is `scientific_evaluation`.

### Same run or new run

- **Resume** an interrupted run whose `kind`, `dataset`, `identity`, `command`, `seed` and `models` are unchanged: keep the run_id and add an attempt. Record the new commit in that attempt.
- **Register a new run** when data, experimental settings or measurement code change. Link it with `related_runs`. The existing run fingerprint (`faar.run_io.run_fingerprint`) already binds measurement code identity for B0-B4 runs. A changed fingerprint means a new run.
- Failed, interrupted and abandoned runs stay in the registry.

## Registering a run

The registry is not wired into the launcher or cluster templates; registration
is a manual step around each run.

1. Before starting, write a record with `status: planned` (or `running` with one attempt whose `code_commit` is `"auto"`) and add it:
   ```bash
   .venv-aaai/bin/python scripts/experiments/registry.py add path/to/record.json
   ```
   `add` fills `"auto"` commits with `git rev-parse HEAD` and the dirty flag, and stamps `recorded_at`.
2. When the job starts, or on each resume, add an attempt. It refuses (exit 3) if any `identity.files` hash differs from disk:
   ```bash
   .venv-aaai/bin/python scripts/experiments/registry.py attempt RUN_ID --started-at 2026-10-02T09:00:00Z
   ```
   Use the scheduler's start time when you know it. For cluster runs, add the attempt locally from the returned files and `sacct` times.
3. When it ends, close the attempt and hash small outputs:
   ```bash
   .venv-aaai/bin/python scripts/experiments/registry.py finish RUN_ID --status completed --output results/b0.json
   ```
   `--backup` states where untracked outputs are copied (default `none`).
4. Commit `registry.jsonl` with the run's compact outputs.

Large logs, rendered pages and bulk outputs stay outside Git. Record their
location in `outputs` with `in_git: false` and an honest `backup` value. A hash
or manifest is not a backup.

## Known gaps

- Historical records were written on 2026-09-28 from existing payloads, run logs and reports. Where those did not record a commit, command or start time, the field is `null`.
- `annotation/` studies and paper-table generation have no registry integration yet.
