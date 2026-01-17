#!/bin/bash
#SBATCH --job-name=UR10_multi_nodes
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }
[ -f "$IMAGE_TAR" ] || { echo "ERROR: image tar not found: $IMAGE_TAR"; exit 2; }

# Shared per-job storage (image + overlay), per-task runtime dirs
BASE_SCRATCH="${SLURM_TMPDIR:-$HOME}"
JOB_SCRATCH="$BASE_SCRATCH/podman_${SLURM_JOB_ID}"

PODMAN_ROOT="$JOB_SCRATCH/root"
mkdir -p "$PODMAN_ROOT"

# Load image tar once (lock protects concurrent ranks if they start near-simultaneously)
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load_${SLURM_JOB_ID}.lock"
(
  flock -x 200
  if [ -z "$(podman --root "$PODMAN_ROOT" images -q "$IMAGE_TAG" 2>/dev/null || true)" ]; then
    podman --root "$PODMAN_ROOT" load -i "$IMAGE_TAR"
  fi
  if [ -z "$(podman --root "$PODMAN_ROOT" images -q "$IMAGE_TAG" 2>/dev/null || true)" ] && \
     [ -n "$(podman --root "$PODMAN_ROOT" images -q "$LOCALHOST_TAG" 2>/dev/null || true)" ]; then
    podman --root "$PODMAN_ROOT" tag "$LOCALHOST_TAG" "$IMAGE_TAG"
  fi
) 200>"$LOCKFILE"

cd "$REPO_DIR"

# Rank list: one per node (ntasks-per-node=1)
# SLURM_PROCID is 0..(nodes-1)
RANK="${SLURM_PROCID:-0}"
INDEX="$((RANK + 1))"   # map ranks to sweep indices 1..N (adjust if you want)

# Per-rank runtime dirs (avoid "readonly db" + collisions)
TASK_SCRATCH="$JOB_SCRATCH/rank_${RANK}"
PODMAN_RUNROOT="$TASK_SCRATCH/runroot"
PODMAN_TMP="$TASK_SCRATCH/tmp"
JAX_CACHE="$TASK_SCRATCH/jax_cache"

mkdir -p "$PODMAN_RUNROOT" "$PODMAN_TMP" "$JAX_CACHE"

export XDG_RUNTIME_DIR="$PODMAN_RUNROOT/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR" || true

# Optional: W&B key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_key")"
fi

# Run one container per node (per rank) WITHOUT srun launching multiple steps
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
    echo node=\$(hostname) rank=$RANK index=$INDEX
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $INDEX \
      --out_root results
  "
