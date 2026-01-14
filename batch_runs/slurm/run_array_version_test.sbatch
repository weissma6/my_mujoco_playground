#!/bin/bash
# ===== Slurm job metadata =====
#SBATCH --job-name=UR10_sweep_EGL
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-4%2

set -euo pipefail
set -x

echo "=== SBATCH START ==="
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
cd "$REPO_DIR"

mkdir -p logs results

# Unique tmp per task (important for rootless podman)
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

# ----------------------------------------------------------------------
# Podman availability (DO NOT require a module; cluster has no podman module)
# ----------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module -t avail 2>&1 | grep -i podman || true
fi

command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }
podman --version

# ----------------------------------------------------------------------
# Image handling (load tar on each node + retag away from localhost/)
# ----------------------------------------------------------------------
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

echo "=== BEFORE PODMAN ==="
echo "IMAGE_TAR=$IMAGE_TAR"
echo "IMAGE_TAG=$IMAGE_TAG"

mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load.lock"
(
  flock -x 200

  # Load tar if target tag not present on this node
  if [ -z "$(podman images -q "$IMAGE_TAG" 2>/dev/null || true)" ]; then
    echo "Image $IMAGE_TAG not found; loading from $IMAGE_TAR"
    podman load -i "$IMAGE_TAR"
  fi

  # podman load commonly restores as localhost/mujoco:env_EGL; ensure $IMAGE_TAG exists
  if [ -z "$(podman images -q "$IMAGE_TAG" 2>/dev/null || true)" ] && \
     [ -n "$(podman images -q "$LOCALHOST_TAG" 2>/dev/null || true)" ]; then
    echo "Retagging $LOCALHOST_TAG -> $IMAGE_TAG"
    podman tag "$LOCALHOST_TAG" "$IMAGE_TAG"
  fi
) 200>"$LOCKFILE"

echo "=== PODMAN IMAGES (mujoco/env_EGL) ==="
podman images | grep -E "mujoco|env_EGL" || true

# ----------------------------------------------------------------------
# Load W&B API key (optional)
# ----------------------------------------------------------------------
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_key")"
fi

# ===============================
# Run training inside container
# ===============================
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
    set -euo pipefail

    echo '=== CONTAINER START ==='
    echo 'CUDA_VISIBLE_DEVICES='\"\$CUDA_VISIBLE_DEVICES\"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

    echo 'MUJOCO_GL='\"\$MUJOCO_GL\"
    echo 'PYOPENGL_PLATFORM='\"\$PYOPENGL_PLATFORM\"
    echo 'EGL_PLATFORM='\"\$EGL_PLATFORM\"

    echo '=== PYTHON/JAX/MJX DIAGNOSTICS ==='
    python - << 'EOF'
import sys, jax, jaxlib, importlib.util
print('exe:', sys.executable)
print('python:', sys.version)
print('jax:', jax.__version__, jax.__file__)
print('jaxlib:', jaxlib.__version__, jaxlib.__file__)
print('mhlo:', importlib.util.find_spec('jaxlib.mlir.dialects.mhlo'))
print('devices:', jax.devices())
EOF

    echo '=== RUNNING EXPERIMENT ==='
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "

echo "=== SBATCH END ==="
date
