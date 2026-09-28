# FAAR

FAAR is a failure-aware OCR-RAG pipeline for document question answering. It
retrieves from OCR text first, then applies a typed recovery only when a quality
gate indicates that the evidence is likely to fail. Recoveries are `semantic`
(retry retrieval), `word_level` (correct OCR noise), and `structural` (selective
visual fallback).

## Status

| Stage | State |
| --- | --- |
| Local implementation and regression tests | Ready |
| OHR data audit and locked PDF archive | Done ([report](docs/reports/data_audit.md)) |
| Development pilot `ohr_dev_v1` selection and inspection packet | Frozen; no model run ([report](docs/reports/pilot_readiness.md)) |
| Bounded 108-page CUDA calibration on a shared cluster | Ready to run |
| Full OHR validation preparation and B0-B4 paper runs | Not done |

Real GPU calibration measurements and full validation results are still pending.
Older 40-example mock-backend numbers in `docs/history/reports/` and
`artifacts/phase3/` are prototype evidence only. They are not AAAI baselines.

## Where things are

| Need | Location |
| --- | --- |
| Research question, first-study scope, open decisions | [docs/research/study-brief.md](docs/research/study-brief.md) |
| Fixed protocol and B0-B4 order | [docs/research/aaai-plan.md](docs/research/aaai-plan.md) |
| Code | `src/faar/` (package), `scripts/` (CLIs), `cluster/` (launcher and scheduler templates), `tests/` |
| Configuration and locks | `config/`: split, checksums, OHR PDF source lock, model revisions, pilot configs |
| Benchmark inputs | `OHR-Bench/` (vendored upstream QA and text; tracked), `data/ohr_bench_raw/pdfs.zip` (locked archive; ignored) |
| Derived data | `data/benchmark_prep/`, `data/external/` (ignored); `data/phase0/` (prototype, tracked) |
| Result payloads | `results/` (audit, pilot, smoke, environment); prototype outputs in `artifacts/` and `logs/` |
| Current reports | [docs/reports/](docs/reports/index.md) |
| History and the 2026-09-28 path map | [docs/history/](docs/history/README.md) |

## Repository

- GitHub: <https://github.com/haseebraza715/FailSafeRAG>
- Branch: `faar-aaai-experiments`

Do not run the cluster workflow from `main` until this branch has been reviewed
and merged. Do not edit `config/datasets/ohr_split.json` or the locked OHR QA file. Their SHA-256
checksums must remain:

- `config/datasets/ohr_split.json`: `64583a532c5db5aa31e4cbb5cd9c7d894c7a2d5e8aa49f1a7f6041f54e714f53`
- `OHR-Bench/data/qas_v2.json`: `2446db28741fa9f392067ee7aae7f3b05e0d85c584069a50ddd5b1b5bc783f58`

## Setup

Use CPython 3.12 and the pinned AAAI extra. A paper run is invalid if `pip check`
fails. The same command works on macOS for local checks and on Linux for Slurm.

```bash
git clone https://github.com/haseebraza715/FailSafeRAG.git faar
cd faar
git checkout faar-aaai-experiments
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
```

Copy `.env.example` to `.env` and fill paths and resource names only. Never
commit `.env`. Local `pytest` is a code check, not a paper result. On macOS,
run the bounded-memory tests in a separate process if the full suite hits the
known OpenMP segfault.

## First cluster commands

Login-node preflight, no CUDA:

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Allocated-GPU preflight, then the bounded 108-page calibration. Submit nothing
else until the calibration report is approved.

```bash
sbatch cluster/templates/slurm_preflight.sbatch
sbatch cluster/templates/slurm_calibration_108.sbatch
```

Edit partition, account, QOS, and `FAAR_GPU_BUDGET_GB` in those templates before
submission. Exact procedure, resume, shard merge, and stop conditions are in
[SUPERVISOR_HANDOFF.md](SUPERVISOR_HANDOFF.md) and
[docs/operations/runbook.md](docs/operations/runbook.md).
The next cluster-only work is that preflight and calibration. Full validation
stays blocked until those measurements are approved.

## Documentation

- [Supervisor handoff](SUPERVISOR_HANDOFF.md)
- [Shared-cluster runbook](docs/operations/runbook.md)
- [Architecture](docs/architecture/overview.md)
- [Experimental plan](docs/research/aaai-plan.md)
- [Documentation index](docs/README.md)
