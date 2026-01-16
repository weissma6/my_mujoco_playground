#!/bin/bash
#SBATCH --job-name=UR10_sweep_EGL
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-16%4

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

# --------------------------
# Robust rootless podman dirs (per job/task) to avoid /scratch issues
# --------------------------
JOB_ID="${SLURM_JOB_ID:-manual}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

export XDG_RUNTIME_DIR="$HOME/.run/${JOB_ID}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

PODMAN_ROOT="$HOME/.podman/${JOB_ID}_${TASK_ID}"
mkdir -p "$PODMAN_ROOT"/{runroot,graphroot,tmp,cache,config}

export TMPDIR="$PODMAN_ROOT/tmp"
export XDG_CACHE_HOME="$PODMAN_ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

cat > "$PODMAN_ROOT/config/storage.conf" <<EOF
[storage]
driver = "vfs"
graphroot = "$PODMAN_ROOT/graphroot"
runroot  = "$PODMAN_ROOT/runroot"
EOF

export CONTAINERS_STORAGE_CONF="$PODMAN_ROOT/config/storage.conf"

# Ensure podman exists
command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }

# --------------------------
# Load image tar once per job (protect with a lock)
# --------------------------
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load_${JOB_ID}.lock"
(
  flock -x 200

  # Show where podman will store its DB (debug)
  podman info | sed -n '1,120p' || true

  if [ -z "$(podman images -q "$IMAGE_TAG" 2>/dev/null || true)" ]; then
    podman load -i "$IMAGE_TAR"
  fi

  if [ -z "$(podman images -q "$IMAGE_TAG" 2>/dev/null || true)" ] && \
     [ -n "$(podman images -q "$LOCALHOST_TAG" 2>/dev/null || true)" ]; then
    podman tag "$LOCALHOST_TAG" "$IMAGE_TAG"
  fi
) 200>"$LOCKFILE"

# Optional: W&B key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_key")"
fi

cd "$REPO_DIR"

# --------------------------
# Run inside container
# --------------------------
podman run --rm \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e EGL_PLATFORM=surfaceless \
  -e PYTHONPATH=/workspace \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  -e TMPDIR="$TMPDIR" \
  -e XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  -v "$REPO_DIR:/workspace:Z" \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -lc "
    set -e
    python -c 'import jax; print(\"jax devices:\", jax.devices())'
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $TASK_ID \
      --out_root results
  "
