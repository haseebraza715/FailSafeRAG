# Supervisor handoff

FAAR is ready for a bounded 108-page CUDA calibration on a shared cluster.
Full validation and B0-B4 paper runs have not started. The complete command and
recovery reference is [docs/operations/runbook.md](docs/operations/runbook.md).

## Current gate

| Stage | Status |
| --- | --- |
| Local implementation and regression tests | Ready |
| Allocated-GPU preflight and 108-page calibration | Ready to run |
| Full validation preparation | Waiting for calibration approval |
| B0-B4 and paper runs | Not run |

The repository contains 40-example mock-backend results from prototype work.
They are not paper baselines.

## Checkout and environment

```bash
git clone https://github.com/haseebraza715/FailSafeRAG.git faar
cd faar
git checkout faar-aaai-experiments
git rev-parse HEAD

python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
cp .env.example .env
```

Record the commit SHA with every returned artifact. The supplied templates use
Slurm. Confirm the scheduler, partition, account, QOS, wall-time limit, CPU and
RAM allocation, scratch path, and quota before submission.

## Required data and settings

Calibration needs the tracked split and checksum files, OHR QA metadata, page
inventory, and the OHR PDF archive. Supply the archive at
`data/ohr_bench_raw/pdfs.zip` or set `FAAR_PDF_ROOT` / `FAAR_PDF_ZIP`. The
official archive is locked in `config/ohr_pdf_source_lock.json`: Hugging Face
`opendatalab/OHR-Bench` revision `7f833e3e`, 1,516,951,813 bytes compressed.
The earlier 200 GB expanded-corpus figure is unmeasured. Measure generated
storage on a small sample before allocating space.

Set these values in the ignored `.env` file or scheduler environment.
Slurm/PBS templates source `.env` after `cd`, and cluster entry points load it
without overriding exported values:

| Variable | Purpose |
| --- | --- |
| `FAAR_PROJECT_ROOT` | Repository checkout |
| `FAAR_SCRATCH` | Fast scratch for outputs and caches |
| `FAAR_OUT_ROOT` | Asset output root |
| `HF_HOME` | Hugging Face cache on scratch |
| `FAAR_PDF_ROOT` / `FAAR_PDF_ZIP` | PDF source |
| `FAAR_DOCUMENT_INVENTORY` | Optional inventory override |
| `HF_TOKEN` | Only when model access requires it |
| `FAAR_GPU_BUDGET_GB` | Positive process VRAM budget chosen after GPU preflight |

Calibration does not require `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Paid VLM
credentials are needed later for B1, B2, and B4, according to the selected
backend.

## Shared-server limits

The launcher refuses GPU work without `FAAR_GPU_BUDGET_GB`. Free VRAM must fit
that budget plus `FAAR_MIN_GPU_FREE_GB`, which defaults to 20% of visible VRAM.
When unset, `FAAR_MAX_RSS_GB` becomes 90% of the scheduler or cgroup RAM limit,
and OpenMP/MKL threads become half the allocated CPUs. The calibration template
requests one GPU, eight CPUs, and 32 GB RAM. Do not raise limits in code.

## Run only the bounded calibration

First run the login-node-safe check:

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Exit 0 means ready, exit 1 is blocking, and exit 2 contains warnings only. Edit
the placeholders in the Slurm templates, then submit the allocated-GPU check:

```bash
sbatch cluster/templates/slurm_preflight.sbatch
```

Choose `FAAR_GPU_BUDGET_GB` from that report. Then submit only:

```bash
sbatch cluster/templates/slurm_calibration_108.sbatch
```

This processes one complete 108-page document
(`manual/User_Manual_1500S_Classic_EN`, pages 0–107) through PDF extraction,
Docling, page rendering, and pinned GOT-OCR. The job refuses to start if that
document or page set does not match. It is calibration evidence, not a paper
result. Do not submit `slurm_one_gpu.sbatch`, validation preparation, or
baseline jobs concurrently.

## Resume and return files

Slurm sends SIGTERM 120 seconds before wall time. Re-submit the same script
after exit 143; completed work resumes from the checkpoint. For SIGKILL or node
failure, obtain elapsed time from `sacct` or `qstat` and follow
[runbook section 8](docs/operations/runbook.md#8-resume-a-preempted-job)
before projecting resources. Never estimate missing wall time.

Return these files, without `.env`, keys, PDFs, or model caches:

1. `cluster/preflight_<jobid>.json`
2. `results/environment/preflight_calibration.json`
3. `results/calibration/faar-ohr-108/prepare_checkpoint.json`
4. `results/calibration/calibration_summary.json`
5. `results/calibration/preparation_projection.json`

[Runbook section 9](docs/operations/runbook.md#9-build-the-calibration-report-and-send-it-back)
contains the summary and projection commands. The report must
include commit SHA, GPU model, peak RSS, stage and total runtime, OCR throughput,
generated storage, cache growth, resume state, timing completeness, and locked
hashes.

## Stop conditions

Stop and return logs if:

- preflight exits 1;
- the commit or split checksum differs from the recorded run;
- the VRAM budget plus co-tenant reserve exceeds free VRAM;
- CUDA OOM or the RSS limit aborts the process;
- timing is incomplete and scheduler elapsed time is unavailable; or
- measured or projected storage exceeds the lab quota.

Wait for approval of the calibration summary and projection before preparing
validation shards, running a pilot, or starting B0-B4.
