#!/bin/bash
#SBATCH --job-name=panda_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=1-<2>%<2>

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAG="mujoco:env"

cd "$REPO_DIR"
mkdir -p logs results

# Load the image tar if needed
if [[ -f "$IMAGE_TAR" ]]; then
  podman load -i "$IMAGE_TAR" || true
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
  python batch_runs/scripts/run_one.py \
    --jsonl "batch_runs/sweeps/panda_sweep.jsonl" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --out_root "results"

