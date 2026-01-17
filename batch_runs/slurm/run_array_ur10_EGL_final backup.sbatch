cat > "$HOME/VT1_MBRL/my_mujoco_playground/run_array_ur10_EGL_final.sbatch" <<'SBATCH'
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
#SBATCH --array=1-2%2

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

# Per-task tmp (helps rootless podman + XLA temp files)
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---- FIX 1: podman runtime dir (avoid /scratch/container + lock issues) ----
export XDG_RUNTIME_DIR="$TMPDIR/xdg_runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# ---- FIX 2: force podman storage into writable locations (NOT /scratch) ----
export CONTAINERS_STORAGE_CONF="$TMPDIR/storage.conf"
mkdir -p "$HOME/.local/share/containers/storage"
mkdir -p "$TMPDIR/containers-runroot"
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "overlay"
graphroot = "$HOME/.local/share/containers/storage"
runroot  = "$TMPDIR/containers-runroot"
EOF

# Helps XLA/JAX/TF caches avoid weird /tmp behavior
export XDG_CACHE_HOME="$TMPDIR/.cache"
mkdir -p "$XDG_CACHE_HOME"

# If you do NOT need XLA JIT, uncomment this workaround:
# export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# Ensure podman exists (no module needed on this cluster)
command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }

# Load image tar once per node (protect with a lock for arrays)
mkdir -p "$HOME/.locks"
LOCKFILE="$HOME/.locks/podman_load.lock"
(
  flock -x 200
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

echo "---- DEBUG ----"
echo "TMPDIR=$TMPDIR"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "CONTAINERS_STORAGE_CONF=$CONTAINERS_STORAGE_CONF"
ec
