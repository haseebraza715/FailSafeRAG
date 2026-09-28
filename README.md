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
| Development pilot `ohr_dev_v1` selection and inspection packet | Frozen; no real-model run ([report](docs/reports/pilot_readiness.md)) |
| Offline engineering path: `src/faar/pilot_runner.py`, official scorer `src/faar/ohr_scoring.py` | Exists. Current run: [`results/engineering/2026-09-28-ohr-dev-v1-offline-engineering-r3/`](results/engineering/2026-09-28-ohr-dev-v1-offline-engineering-r3/). It uses a rule-based extractor, no repair and no model call. It is an engineering check, not a baseline ([report](docs/reports/prebaseline-engineering.md)) |
| First real baseline (no recovery, real answer model, `ohr_dev_v1`) | Proposed in [study brief section 15](docs/research/study-brief.md#15-first-real-baseline-proposed-protocol). Nothing is approved, and no model call has run |
| Bounded 108-page CUDA calibration on a shared cluster | Commands are ready. On hold until the lead approves it. It is not the automatic next step |
| Full OHR validation preparation and B0-B4 paper runs | Not done |

Real-model results, GPU calibration measurements and full validation results
are still pending. The study brief is the only current plan.
Older 40-example mock-backend numbers in `docs/history/reports/` and
`artifacts/phase3/` are prototype evidence only. They are not AAAI baselines.

## Where things are

| Need | Location |
| --- | --- |
| Current study design: question, comparisons, scoring, decisions | [docs/research/study-brief.md](docs/research/study-brief.md) |
| Earlier AAAI plan, kept as evidence; the study brief says which parts still apply | [docs/research/aaai-plan.md](docs/research/aaai-plan.md) |
| Code | `src/faar/` (package), `scripts/` (CLIs), `cluster/` (launcher and scheduler templates), `tests/` |
| Configuration and locks | `config/`: split, checksums, OHR PDF source lock, model revisions, pilot configs |
| Benchmark inputs | `OHR-Bench/` (vendored upstream QA and text; tracked), `data/ohr_bench_raw/pdfs.zip` (locked archive; ignored) |
| Derived data | `data/benchmark_prep/`, `data/external/` (ignored); `data/phase0/` (prototype, tracked) |
| Experiment runs and their state | [experiments/](experiments/README.md): `registry.jsonl` plus the run-record format |
| Result payloads | `results/` (audit, pilot, smoke, environment); prototype outputs in `artifacts/` and `logs/` |
| Current reports | [docs/reports/](docs/reports/index.md) |
| Paper | [paper/main.tex](paper/main.tex) ([build and provenance](paper/README.md)) |
| History and the 2026-09-28 path map | [docs/history/](docs/history/README.md) |

## Repository

- GitHub: <https://github.com/haseebraza715/FailSafeRAG>
- Branch: `main`

Record the commit SHA of the checkout with every returned result. Do not edit `config/datasets/ohr_split.json` or the locked OHR QA file. Their SHA-256
checksums must remain:

- `config/datasets/ohr_split.json`: `64583a532c5db5aa31e4cbb5cd9c7d894c7a2d5e8aa49f1a7f6041f54e714f53`
- `OHR-Bench/data/qas_v2.json`: `2446db28741fa9f392067ee7aae7f3b05e0d85c584069a50ddd5b1b5bc783f58`

## Setup

Use CPython 3.12 and the pinned AAAI extra. A paper run is invalid if `pip check`
fails. The same command works on macOS for local checks and on Linux for Slurm.

```bash
git clone https://github.com/haseebraza715/FailSafeRAG.git faar
cd faar
git checkout main
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
```

Copy `.env.example` to `.env` and fill paths and resource names only. Never
commit `.env`. Local `pytest` is a code check, not a paper result.

### Local checks and known issues

Run project scripts with a Python 3.12 environment built from the declared
dependencies. The system `python3` may be older than 3.12. To build a fresh
environment the way CI does, without the CPU-only PyTorch index that CI adds:

```bash
uv venv --python 3.12 --seed .local/venv-prebaseline
.local/venv-prebaseline/bin/python -m pip install -c config/environment/constraints-aaai.txt -e ".[test,lint]"
.local/venv-prebaseline/bin/python -m pytest -q -ra -p no:cacheprovider
.local/venv-prebaseline/bin/ruff check .
uv lock --check
```

An older `.venv-aaai` built before these pins has `click` 8.4.2 and lacks
`jieba`. There `faar-demo --help` fails with `TypeError: Secondary flag is not
valid for non-boolean flag`, and `tests/test_cli_help.py` and
`tests/test_ohr_scoring.py` fail. Install the pinned packages with
`.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt click jieba regex pypdfium2`.

The test setup has these safeguards:

- `tests/conftest.py` removes provider keys, blocks non-loopback network connections and sets `HF_HUB_OFFLINE=1`.
- It sends API-call logs aimed at the repository's `logs/` to a temporary directory, and fails the run if any file under `logs/` changes.
- On macOS it limits `faiss` and `torch` to one thread, because their wheels each bundle a `libomp` and a process that runs both at full thread count segfaults or hangs. The pytest header says when the limit is active.
- `tests/test_b0_one_doc_smoke.py` skips one test when the prepared one-document smoke assets are absent, as in CI.

**OpenMP on macOS.** The `faiss-cpu` and `torch` wheels each bundle their own
`libomp.dylib`, and every macOS `faiss-cpu` wheel from 1.9.0 to 1.15.1 does so,
so no pin avoids a second runtime. With the pinned versions, a process that
imports `faiss` and then runs a parallel torch operation, or the reverse,
segfaults or hangs. `KMP_DUPLICATE_LIB_OK` does not prevent this, and a thread
limit does. Run any macOS command that does real (sentence-transformers)
retrieval with `OMP_NUM_THREADS=1`, for example
`OMP_NUM_THREADS=1 faar-demo run-example ...`. That path was checked only with
stand-in models, not the real ones. Local-hash runs such as
`run_pilot_offline.py generate`, and `faar-demo --help`, need nothing. Linux and
the cluster were not tested. To re-check after a dependency change, run
`python scripts/diagnostics/openmp_check.py`. It prints a JSON report and takes
about 6 minutes with the default 5 trials.

## First cluster commands

Login-node preflight, no CUDA:

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Allocated-GPU preflight, then the bounded 108-page calibration. These commands
are available. Do not submit them until the lead approves the calibration, and
submit nothing else until its report is approved.

```bash
sbatch cluster/templates/slurm_preflight.sbatch
sbatch cluster/templates/slurm_calibration_108.sbatch
```

Edit partition, account, QOS, and `FAAR_GPU_BUDGET_GB` in those templates before
submission. Exact procedure, resume, shard merge, and stop conditions are in
[SUPERVISOR_HANDOFF.md](SUPERVISOR_HANDOFF.md) and
[docs/operations/runbook.md](docs/operations/runbook.md).
The calibration is the only cluster job prepared so far. It waits for the
lead's approval, and full validation stays blocked until its measurements are
approved.

## Documentation

- [Supervisor handoff](SUPERVISOR_HANDOFF.md)
- [Shared-cluster runbook](docs/operations/runbook.md)
- [Architecture](docs/architecture/overview.md)
- [Study design](docs/research/study-brief.md)
- [Documentation index](docs/README.md)
