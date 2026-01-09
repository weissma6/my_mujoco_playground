#!/bin/bash
# ===== Slurm job metadata =====
#SBATCH --job-name=UR10_sweep
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

# Use tmp directory in your home for podman instead of node /tmp
export TMPDIR="$HOME/tmp_podman"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"   # optional debug print

IMAGE_TAR="$REPO_DIR/mujoco_env.tar"
IMAGE_TAG="mujoco:env"

mkdir -p logs results

# Load image if needed (now using TMPDIR in home)
if ! podman image exists "$IMAGE_TAG"; then
  echo "Image $IMAGE_TAG not found; loading from $IMAGE_TAR"
  podman load -i "$IMAGE_TAR"
fi

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
  mujoco:env \
  python batch_runs/scripts/run_one_ur10.py \
    --jsonl "batch_runs/sweeps/UR10_sweep.jsonl" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --out_root "results"