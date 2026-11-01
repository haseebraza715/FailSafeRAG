# FAAR

FAAR (Failure-Aware Agentic Retrieval) is a document question-answering pipeline over OCR text. It answers from the
retrieved OCR text first. When a gate flags the answer as likely wrong, FAAR diagnoses why and picks a matching
repair. Empty OCR pages and broken layout go to an image repair that shows the page images to the model. Word-level
OCR noise goes to SymSpell text correction. Everything else goes to a second retrieval.

## Result

The study is finished, and FAAR does not beat a simple image repair. On 946 held-out multi-page OHR-Bench test
questions, the answer model (`gpt-6.1-terra`) got these shares of questions exactly right:

| Policy | Exact match | Cost per question (USD) |
| --- | ---: | ---: |
| No recovery | 0.236 | 0.00171 |
| FAAR | 0.266 | 0.00320 |
| Image repair on every flagged question | 0.315 | 0.00831 |

FAAR is 4.9 points below the image repair (95% document interval [-7.3, -2.6], Holm p 6.6e-6) and 3.1 points above
no recovery. It costs 0.39 times as much as the image repair. Its diagnosis does no better than picking a repair at
random. The validation confirmation and the development rounds show the same pattern.

Read the [FAAR study report](docs/reports/faar-study-report.md) for the method, every stage's results, why FAAR lost
and the limitations.

## Status

| Stage | State |
| --- | --- |
| Data audit and locked PDF archive | Done ([report](docs/reports/data_audit.md)) |
| Development pilots `ohr_dev_v1` and `ohr_dev_v2` | Done. The image repair gained 8.5 points over no recovery and became the bar |
| Practice rounds on `ohr_long_dev_v1` (423 train questions) | Done, method frozen at `b075dbd` ([log](docs/research/faar-practice-rounds.md)) |
| Confirmation on `ohr_val_multipage_v1` (835 validation questions) | Done, run once. FAAR 4.4 points below the bar ([report](docs/research/faar-confirmation.md)) |
| Final test on `ohr_test_multipage_v1` (946) and `ohr_test_singlepage_v1` (330) | Done, run once ([report](docs/research/faar-final-test.md)) |
| Cluster calibration and B0-B4 paper runs from the earlier plan | Not run. The FAAR study used the OpenAI API only |

Every run is recorded in [experiments/registry.jsonl](experiments/registry.jsonl). Older 40-example mock-backend
numbers in `docs/history/reports/` and `artifacts/phase3/` are prototype evidence only.

## Where things are

| Need | Location |
| --- | --- |
| Study report with all results | [docs/reports/faar-study-report.md](docs/reports/faar-study-report.md) |
| Pre-registration, frozen method and test plan | [docs/research/faar-vs-gated-preregistration.md](docs/research/faar-vs-gated-preregistration.md) |
| Study design: question, comparisons, scoring, decisions | [docs/research/study-brief.md](docs/research/study-brief.md) |
| Earlier AAAI plan, kept as evidence; the study brief says which parts still apply | [docs/research/aaai-plan.md](docs/research/aaai-plan.md) |
| Code | `src/faar/` (package), `scripts/` (CLIs), `cluster/` (launcher and scheduler templates), `tests/` |
| Configuration and locks | `config/`: split, checksums, OHR PDF source lock, model revisions, pilot configs |
| Benchmark inputs | `OHR-Bench/` (vendored upstream QA and text; tracked), `data/ohr_bench_raw/pdfs.zip` (locked archive; ignored) |
| Derived data | `data/benchmark_prep/`, `data/external/` (ignored); `data/phase0/` (prototype, tracked) |
| Experiment runs and their state | [experiments/](experiments/README.md): `registry.jsonl` plus the run-record format |
| Result payloads | `results/development/` (raw model replies and run records), `results/analysis/` (comparison reports and per-question scores), `results/pilots/` (frozen samples); prototype outputs in `artifacts/` and `logs/` |
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
`libomp.dylib`. Every macOS `faiss-cpu` wheel checked (1.9.0.post1, 1.10.0,
1.11.0, 1.12.0, 1.13.2, 1.14.3 and 1.15.1) bundles one, so a faiss pin does not
avoid a second runtime. With the pinned versions, a process that
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

## Cluster commands

The FAAR study did not use the cluster. These commands are from the earlier plan, were never submitted and have no
planned use. Before anyone submits them, the research lead has to approve the calibration.

Login-node preflight, no CUDA:

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Allocated-GPU preflight, then the bounded 108-page calibration. These commands
are available.

```bash
sbatch cluster/templates/slurm_preflight.sbatch
sbatch cluster/templates/slurm_calibration_108.sbatch
```

Edit partition, account, QOS, and `FAAR_GPU_BUDGET_GB` in those templates before
submission. Exact procedure, resume, shard merge, and stop conditions are in
[SUPERVISOR_HANDOFF.md](SUPERVISOR_HANDOFF.md) and
[docs/operations/runbook.md](docs/operations/runbook.md).
The calibration is the only cluster job prepared.

## Documentation

- [Supervisor handoff](SUPERVISOR_HANDOFF.md)
- [Shared-cluster runbook](docs/operations/runbook.md)
- [Architecture](docs/architecture/overview.md)
- [Study design](docs/research/study-brief.md)
- [Documentation index](docs/README.md)
