#!/bin/bash
# ===== Slurm job metadata =====
#SBATCH --job-name=UR10_sweep
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=1-2%2

set -euo pipefail


set -x
echo "=== SBATCH START ==="
date
pwd
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
ls -la
echo "=== BEFORE PODMAN ==="

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
cd "$REPO_DIR"

mkdir -p logs results

# Unique tmp per task
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

IMAGE_TAR="$REPO_DIR/mujoco_env_ffmpeg.tar"
IMAGE_TAG="mujoco:env-ffmpeg"

# Load image safely (avoid parallel podman load)
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load.lock"
(
  flock -x 200
  if [ -z "$(podman images -q "$IMAGE_TAG")" ]; then
    echo "Image $IMAGE_TAG not found; loading from $IMAGE_TAR"
    podman load -i "$IMAGE_TAR"
    podman images | head -20
  fi
) 200>"$LOCKFILE"


# Load W&B API key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_key")"
fi

# IMPORTANT: no -it in batch jobs
podman run --rm \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e MUJOCO_GL=osmesa \
  -e PYTHONPATH=/workspace \
  -v "$REPO_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_TAG" \
  python batch_runs/scripts/run_one_ur10.py \
    --jsonl "batch_runs/sweeps/UR10_sweep.jsonl" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --out_root "results"