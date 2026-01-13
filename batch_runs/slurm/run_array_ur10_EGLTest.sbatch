#!/bin/bash
# ===== Slurm job metadata =====
#SBATCH --job-name=UR10_sweep_EGL_v1
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=1-1%1

set -euo pipefail
set -x

echo "=== SBATCH START ==="
date
pwd
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
ls -la

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
cd "$REPO_DIR"

mkdir -p logs results

# Unique tmp per task
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"

echo "=== BEFORE PODMAN ==="

# Load image safely (avoid parallel podman load)
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load.lock"
(
  flock -x 200
  if [ -z "$(podman images -q "$IMAGE_TAG")" ]; then
    echo "Image $IMAGE_TAG not found; loading from $IMAGE_TAR"
    podman load -i "$IMAGE_TAR"
    podman images | head -20
  fi
) 200>"$LOCKFILE"

# Load W&B API key
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_key")"
fi

# Run container
podman run --rm \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e EGL_PLATFORM=surfaceless \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
  -e PYTHONPATH=/workspace \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  -v "$REPO_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_TAG" \
  bash -lc "
    set -euo pipefail

    echo '=== CONTAINER PREFLIGHT ==='
    echo 'CUDA_VISIBLE_DEVICES='\"\$CUDA_VISIBLE_DEVICES\"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
    echo 'MUJOCO_GL='\"\$MUJOCO_GL\"
    echo 'PYOPENGL_PLATFORM='\"\$PYOPENGL_PLATFORM\"
    echo 'EGL_PLATFORM='\"\$EGL_PLATFORM\"

    echo '=== EGL LIBS (container) ==='
    ldconfig -p | grep -i egl || true

    echo '=== VERSIONS ==='
    python - << 'EOF'
import mujoco, OpenGL
print('mujoco:', mujoco.__version__)
print('PyOpenGL:', OpenGL.__version__)
EOF

    echo '=== EGL SMOKE TEST ==='
    python - << 'EOF'
import os
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
r = mujoco.Renderer(m)
img = r.render()
print('EGL render OK:', img.shape, img.dtype)
EOF

    echo '=== START TRAINING ==='
    python batch_runs/scripts/run_one_ur10.py \
      --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "
