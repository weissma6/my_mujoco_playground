#!/bin/bash
#SBATCH --job-name=UR10_sweep_EGL
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%2

set -euo pipefail
umask 0022

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"

# ALWAYS load from node-local persistent scratch tar
IMAGE_TAR="/raid/persistent_scratch/$USER/mujoco_env_EGL.tar"

IMAGE_TAG="mujoco:env_EGL"
NODE="$(hostname)"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

# Ensure tar exists on THIS node
[ -f "$IMAGE_TAR" ] || { echo "ERROR: tar not found on this node: $IMAGE_TAR"; exit 2; }

# Load once per node (lock is per-node + on node-local scratch)
LOCKDIR="/raid/persistent_scratch/$USER/.locks"
mkdir -p "$LOCKDIR"
LOCKFILE="$LOCKDIR/podman_load_${NODE}.lock"

(
  flock -x 200
  podman image exists "$IMAGE_TAG" 2>/dev/null || podman load -i "$IMAGE_TAR"
) 200>"$LOCKFILE"

echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID gpu=${CUDA_VISIBLE_DEVICES:-unset}"

# Optional: W&B key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_key")"
fi

cd "$REPO_DIR"

podman run --rm \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e EGL_PLATFORM=surfaceless \
  -e PYTHONPATH=/workspace \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  -v "$REPO_DIR:/workspace:Z" \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -lc "
    python -c 'import jax; print(\"jax devices:\", jax.devices())' || true
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "


