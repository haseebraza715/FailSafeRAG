#!/usr/bin/env bash
#SBATCH --job-name=faar-one-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=<EDIT_PARTITION>
#SBATCH --qos=<EDIT_QOS>
#SBATCH --account=<EDIT_ACCOUNT>
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120

# Generic full B0-B4 runner. This is NOT the bounded 108-page calibration.
# Do not submit this file for supervisor preflight or calibration.
# Do not submit until the calibration report is approved and the locked
# validation manifest exists. Replace every <EDIT_...> placeholder first.
# If the cluster does not use accounts, comment out the --account line.
# B0--B4 checkpoint each example and resume with --resume.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if grep -E '^#SBATCH --(partition|qos|account)=<EDIT_' "$0" >/dev/null; then
  echo "slurm_one_gpu.sbatch: replace <EDIT_PARTITION>, <EDIT_QOS>, and <EDIT_ACCOUNT> before submitting." >&2
  echo "If the cluster does not require an account, comment out the --account line." >&2
  echo "This template is not the bounded 108-page calibration." >&2
  echo "Use cluster/templates/slurm_preflight.sbatch and slurm_calibration_108.sbatch instead." >&2
  exit 2
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export VLM_BACKEND="${VLM_BACKEND:-openai}"
export FAAR_PYTHON="${FAAR_PYTHON:-$PWD/.venv-aaai/bin/python}"
export FAAR_DATASET="${FAAR_DATASET:-ohrbench}"
export FAAR_SPLIT="${FAAR_SPLIT:-val}"
# Required after preflight: pick a positive GPU process budget (GiB) from the
# reported GPU model and free VRAM. The launcher fails closed without it.
export FAAR_GPU_BUDGET_GB="${FAAR_GPU_BUDGET_GB:-}"

"${FAAR_PYTHON}" cluster/run_baselines.py \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --dataset "${FAAR_DATASET}" \
  --split "${FAAR_SPLIT}" \
  --resume
