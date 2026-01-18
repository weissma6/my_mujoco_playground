#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "${REPO_DIR}"
source cluster/env_bootstrap.sh
# Make per-job W&B directory to avoid collisions
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export WANDB_DIR="/scratch/wandb/${USER}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
  mkdir -p "${WANDB_DIR}"
fi
# Your actual training command goes here.
# Keep "$@" so W&B agent/Hydra can inject parameters.
# Example A: Hydra module
python -m src.train "$@"
# Example B: If you use a script instead
# python train.py "$@"
