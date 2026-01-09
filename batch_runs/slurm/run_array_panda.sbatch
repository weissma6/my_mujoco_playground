#!/bin/bash
#SBATCH --job-name=panda_sweep
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=1-4%4

set -euo pipefail

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
  if ! podman image exists "$IMAGE_TAG"; then
    echo "Image $IMAGE_TAG not found; loading from $IMAGE_TAR"
    podman load -i "$IMAGE_TAR"
  fi
) 200>"$LOCKFILE"

# Load W&B API key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_key")"
fi

podman run --rm \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e MUJOCO_GL=osmesa \
  -e PYTHONPATH=/workspace \
  -v "$REPO_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_TAG" \
  python batch_runs/scripts/run_one_panda.py \
    --jsonl "batch_runs/sweeps/panda_sweep.jsonl" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --out_root "results"
