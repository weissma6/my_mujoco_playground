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
#SBATCH --array=1-10%1

set -euo pipefail
umask 0022

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"
cd "$REPO_DIR"

IMAGE_TAR="/raid/persistent_scratch/$USER/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
[ -f "$IMAGE_TAR" ] || { echo "ERROR: tar not found on this node: $IMAGE_TAR"; exit 2; }

command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }

NODE="$(hostname)"
NODE_PS="/raid/persistent_scratch/$USER"

# IMPORTANT: runroot must be stable for the DB
PODMAN_ROOT="$NODE_PS/podman_root"
PODMAN_RUNROOT="$NODE_PS/podman_runroot"
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUNROOT"

export XDG_RUNTIME_DIR="$PODMAN_RUNROOT/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR" || true

# Task-local temp dirs mounted into container
HOST_TMP="${SLURM_TMPDIR:-/tmp}/ctmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
JAX_CACHE="${SLURM_TMPDIR:-/tmp}/jaxcache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HOST_TMP" "$JAX_CACHE"

# Load once per node (node-local lock)
LOCKDIR="$NODE_PS/.locks"
mkdir -p "$LOCKDIR"
LOCKFILE="$LOCKDIR/podman_load_${NODE}.lock"
(
  flock -x 200
  podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" image exists "$IMAGE_TAG" 2>/dev/null \
    || podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" load -i "$IMAGE_TAR"
) 200>"$LOCKFILE"

echo "node=$NODE task=$SLURM_ARRAY_TASK_ID gpu=${CUDA_VISIBLE_DEVICES:-unset} podman_root=$PODMAN_ROOT runroot=$PODMAN_RUNROOT"

if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_key")"
fi

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
  -e TMPDIR=/tmp \
  -e JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  -v "$REPO_DIR:/workspace:Z" \
  -v "$HOST_TMP:/tmp:Z" \
  -v "$JAX_CACHE:/tmp/jax_cache:Z" \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -lc "
    set -euo pipefail
    python -c 'import jax; print(\"jax devices:\", jax.devices())' || true
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "
