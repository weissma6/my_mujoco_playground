#!/bin/bash
# ===== Slurm job metadata =====
#SBATCH --account=cai_iai
#SBATCH --job-name=ur10_version_test
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=1-1%1

set -euo pipefail
set -x

echo "=== SBATCH START ==="
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# ----------------------------------------------------------------------
# Paths & dirs
# ----------------------------------------------------------------------
REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
cd "$REPO_DIR"

mkdir -p logs results

# Unique tmp per task (important for rootless podman)
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

# ----------------------------------------------------------------------
# Podman + image handling
# ----------------------------------------------------------------------
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

# Ensure module command works in batch
source /etc/profile.d/modules.sh || true
module --ignore_cache load podman

echo "=== BEFORE PODMAN ==="

# Load image safely (avoid race conditions across array tasks)
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load.lock"

(
  flock -x 200

  # Load tar if target tag not present
  if [ -z "$(podman images -q "$IMAGE_TAG" 2>/dev/null || true)" ]; then
    echo "Image $IMAGE_TAG not found; loading from $IMAGE_TAR"
    podman load -i "$IMAGE_TAR"
  fi

  # Retag if podman restored it as localhost/...
  if [ -z "$(podman images -q "$IMAGE_TAG" 2>/dev/null || true)" ] && \
     [ -n "$(podman images -q "$LOCALHOST_TAG" 2>/dev/null || true)" ]; then
    echo "Retagging $LOCALHOST_TAG -> $IMAGE_TAG"
    podman tag "$LOCALHOST_TAG" "$IMAGE_TAG"
  fi
) 200>"$LOCKFILE"

echo "=== PODMAN IMAGES ==="
podman images | grep -E "mujoco|env_EGL" || true

# ----------------------------------------------------------------------
# Host diagnostics (proves we are NOT using host python)
# ----------------------------------------------------------------------
which python || true
python -c "import sys; print(sys.executable)" || true

# ----------------------------------------------------------------------
# Run INSIDE container only
# ----------------------------------------------------------------------
podman run --rm \
  -v "$HOME/VT1_MBRL:/workspace/VT1_MBRL:Z" \
  -w /workspace/VT1_MBRL/my_mujoco_playground \
  "$IMAGE_TAG" \
  bash -lc '
    echo "=== CONTAINER DIAGNOSTICS ==="
    python - << EOF
import sys, jax, jaxlib, importlib.util
print("exe:", sys.executable)
print("python:", sys.version)
print("jax:", jax.__version__, jax.__file__)
print("jaxlib:", jaxlib.__version__, jaxlib.__file__)
print("mhlo:", importlib.util.find_spec("jaxlib.mlir.dialects.mhlo"))
print("devices:", jax.devices())
EOF

    echo "=== RUNNING EXPERIMENT ==="
    python batch_runs/scripts/run_one_ur10.py
  '

echo "=== SBATCH END ==="
date
