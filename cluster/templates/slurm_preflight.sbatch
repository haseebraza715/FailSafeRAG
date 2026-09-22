#!/usr/bin/env bash
#SBATCH --job-name=faar-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=<EDIT: partition that offers the intended GPU, e.g. gpu>
#SBATCH --qos=<EDIT: qos, e.g. normal>
#SBATCH --account=<EDIT: account only if the cluster requires one>
#SBATCH --time=00:30:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

# Allocated-GPU preflight: proves the scheduler allocation matches the GPU
# budget, RAM/cgroup limits, dataset paths, checksums, cache/output writability,
# HF model access, API-key presence, and external executables.
#
# A login-node-safe preflight without CUDA checks runs directly (no sbatch):
#   .venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
#
# The job exit code mirrors the preflight exit code:
#   0 = ready; 1 = blocking failure; 2 = warnings only.
# Fix blocking failures before submitting any compute job.

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

"${FAAR_PYTHON}" cluster/preflight.py \
  --check \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --out "cluster/preflight_${SLURM_JOB_ID}.json"
