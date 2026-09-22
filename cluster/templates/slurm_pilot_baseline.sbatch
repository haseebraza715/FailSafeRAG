#!/usr/bin/env bash
#SBATCH --job-name=faar-pilot-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=<EDIT: partition that offers the intended GPU, e.g. gpu>
#SBATCH --qos=<EDIT: qos, e.g. normal>
#SBATCH --account=<EDIT: account only if the cluster requires one>
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err

# Small 50-100 question baseline pilot (engineering check only, never a paper
# result). Do NOT submit until the calibration report is approved.
#
# Prerequisites:
#   - Prepared validation assets + locked manifest for the split in use
#     (scripts/data/register_benchmark_assets.py output; see docs/operations/runbook.md).
#   - config/gate_threshold.json if the pilot runs B2 or later
#     (scripts/experiments/tune_gate.py on validation B0; run --stop-after b0 otherwise).
#   - OPENAI_API_KEY (or ANTHROPIC_API_KEY for Claude) read from the submit
#     environment or $PWD/.env. NEVER paste keys into this file.
#
# Resume: --resume skips completed stages and re-validates them.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# <EDIT: 50-100 example pilot size>
export PILOT_EXAMPLES="${PILOT_EXAMPLES:-100}"
export FAAR_PYTHON="${FAAR_PYTHON:-$PWD/.venv-aaai/bin/python}"
export FAAR_SPLIT="${FAAR_SPLIT:-val}"
export VLM_BACKEND="${VLM_BACKEND:-openai}"
# <EDIT: positive GiB process VRAM budget chosen from the preflight report>
export FAAR_GPU_BUDGET_GB="${FAAR_GPU_BUDGET_GB:-}"

"${FAAR_PYTHON}" cluster/run_baselines.py \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --dataset ohrbench \
  --split "${FAAR_SPLIT}" \
  --max-examples "${PILOT_EXAMPLES}" \
  --resume
