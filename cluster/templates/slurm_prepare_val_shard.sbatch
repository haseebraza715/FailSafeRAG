#!/usr/bin/env bash
#SBATCH --job-name=faar-prepare-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=<EDIT: partition that offers the intended GPU, e.g. gpu>
#SBATCH --qos=<EDIT: qos, e.g. normal>
#SBATCH --account=<EDIT: account only if the cluster requires one>
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120
#SBATCH --array=0-3
#SBATCH --output=logs/prepare_val_%A_%a.out
#SBATCH --error=logs/prepare_val_%A_%a.err

# Sharded validation asset preparation (1,274 questions / 549 documents /
# 7,037 pages). Deterministic contiguous shards: keep --array and SHARDS in
# sync (--array=0-3 with SHARDS=4 launches shards 0..3).
#
# Only after the calibration report is approved and the shard count is sized
# (see docs/operations/runbook.md "Sizing and shard count"). Each shard writes its own
# checkpoint and a shard manifest; shards are resumed by re-submitting the job.
#
# After ALL shards complete, build the locked benchmark manifest with
# scripts/data/register_benchmark_assets.py (see docs/operations/runbook.md) and validate the merge before
# any baseline run.
#
# Requirements: FAAR_GPU_BUDGET_GB set from the preflight report; OHR archive
# present (data/ohr_bench_raw/pdfs.zip) or FAAR_PDF_ROOT exported; HF access to
# the locked GOT-OCR model.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export SHARDS="${SHARDS:-4}"
export FAAR_PYTHON="${FAAR_PYTHON:-$PWD/.venv-aaai/bin/python}"
export FAAR_SCRATCH="${FAAR_SCRATCH:-$PWD/results/validation}"
export HF_HOME="${HF_HOME:-$FAAR_SCRATCH/huggingface}"
# <EDIT: positive GiB process VRAM budget chosen from the preflight report>
export FAAR_GPU_BUDGET_GB="${FAAR_GPU_BUDGET_GB:-}"
mkdir -p "$FAAR_SCRATCH" "$HF_HOME"

if (( SLURM_ARRAY_TASK_ID >= SHARDS )); then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for SHARDS=$SHARDS" >&2
  exit 1
fi

"${FAAR_PYTHON}" cluster/launcher.py \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --entrypoint cluster/prepare_assets.py \
  --preflight-out "results/environment/preflight_val_shard${SLURM_ARRAY_TASK_ID}.json" \
  --split val \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${SHARDS}" \
  --out-root "$FAAR_SCRATCH/faar-ohr-val" \
  --resume
