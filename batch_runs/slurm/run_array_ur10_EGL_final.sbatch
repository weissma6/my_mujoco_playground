#!/bin/bash
#SBATCH --job-name=UR10_parallel
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-4%4

set -euo pipefail

# IMPORTANT:
# If you don't want to use srun, then you CANNOT run "one job on multiple nodes" in one sbatch allocation.
# The Slurm-correct way without srun is a JOB ARRAY: each array task is its own job and Slurm places them on different nodes.

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }
[ -f "$IMAGE_TAR" ] || { echo "ERROR: image tar not found: $IMAGE_TAR"; exit 2; }

# Scratch (prefer node-local if provided)
BASE_SCRATCH="${SLURM_TMPDIR:-$HOME}"
JOB_SCRATCH="$BASE_SCRATCH/podman_${SLURM_JOB_ID}"
TASK_SCRATCH="$JOB_SCRATCH/task_${SLURM_ARRAY_TASK_ID}"

PODMAN_ROOT="$JOB_SCRATCH/root"
PODMAN_RUNROOT="$TASK_SCRATCH/runroot"
PODMAN_TMP="$TASK_SCRATCH/tmp"
JAX_CACHE="$TASK_SCRATCH/jax_cache"

mkdir -p "$PODMAN_ROOT" "$PODMAN_RUNROOT" "$PODMAN_TMP" "$JAX_CACHE"

export XDG_RUNTIME_DIR="$PODMAN_RUNROOT/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR" || true

# Optional: W&B key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_key")"
fi

cd "$REPO_DIR"

# Load image tar once per job (lock protects concurrent steps if Slurm co-schedules on same node)
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load_${SLURM_JOB_ID}.lock"
(
  flock -x 200
  if [ -z "$(podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" images -q "$IMAGE_TAG" 2>/dev/null || true)" ]; then
    podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" load -i "$IMAGE_TAR"
  fi
  if [ -z "$(podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" images -q "$IMAGE_TAG" 2>/dev/null || true)" ] && \
     [ -n "$(podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" images -q "$LOCALHOST_TAG" 2>/dev/null || true)" ]; then
    podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" tag "$LOCALHOST_TAG" "$IMAGE_TAG"
  fi
) 200>"$LOCKFILE"

# Map array task -> sweep index (your original behavior)
INDEX="$SLURM_ARRAY_TASK_ID"

podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" run --rm \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --security-opt=label=disable \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e EGL_PLATFORM=surfaceless \
  -e PYTHONPATH=/workspace \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  -e TMPDIR=/tmp \
  -e JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache" \
  -v "$REPO_DIR:/workspace:Z" \
  -v "$PODMAN_TMP:/tmp:Z" \
  -v "$JAX_CACHE:/tmp/jax_cache:Z" \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -lc "
    set -euo pipefail
    echo node=\$(hostname) array_task=$SLURM_ARRAY_TASK_ID index=$INDEX
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $INDEX \
      --out_root results
  "
