#!/usr/bin/env bash
#SBATCH --job-name=faar-calibration-108
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=<EDIT: partition that offers the intended GPU, e.g. gpu>
#SBATCH --qos=<EDIT: qos, e.g. normal>
#SBATCH --account=<EDIT: account only if the cluster requires one>
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=logs/calibration_%j.out
#SBATCH --error=logs/calibration_%j.err

# Bounded 108-page single-document calibration (manual/User_Manual_1500S_Classic_EN).
# Calibration evidence only, never a paper result. Submit this job FIRST and
# only after the preflight report is approved; do not submit anything else yet.
#
# Requirements before submission:
#   1. Preflight approved (see docs/operations/runbook.md).
#   2. FAAR_GPU_BUDGET_GB set below to a budget chosen from the preflight VRAM report.
#   3. OHR archive present (data/ohr_bench_raw/pdfs.zip) or FAAR_PDF_ROOT exported.
#
# Preemption: Slurm sends SIGTERM 120 s before walltime; the pipeline stops at
# the next page/document boundary and every completed stage is checkpointed.
# Re-submit the same job to resume; completed stages are skipped.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export FAAR_PYTHON="${FAAR_PYTHON:-$PWD/.venv-aaai/bin/python}"
export FAAR_SCRATCH="${FAAR_SCRATCH:-$PWD/results/calibration}"
export HF_HOME="${HF_HOME:-$FAAR_SCRATCH/huggingface}"
# <EDIT: positive GiB process VRAM budget chosen from the preflight report>
export FAAR_GPU_BUDGET_GB="${FAAR_GPU_BUDGET_GB:-}"
mkdir -p "$FAAR_SCRATCH" "$HF_HOME"

"${FAAR_PYTHON}" cluster/launcher.py \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --entrypoint scripts/data/prepare_benchmark_assets.py \
  --preflight-out "results/environment/preflight_calibration.json" \
  --dataset ohrbench \
  --out-root "$FAAR_SCRATCH/faar-ohr-108" \
  --smoke-doc manual/User_Manual_1500S_Classic_EN \
  --require-calibration-108 \
  --execute
