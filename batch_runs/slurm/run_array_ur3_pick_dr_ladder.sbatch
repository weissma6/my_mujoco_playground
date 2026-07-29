#!/bin/bash
#SBATCH --job-name=UR3Pick_DR_ladder
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Same per-run cost as run_array_ur3_pick.sbatch: this sweep does NOT touch
# num_timesteps/num_envs/batch_size/LR/episode_length (Pass-1 shared-HP
# strategy, see "Plan - Sim-to-Real Gap Protocol") -- only domain_rand.*
# .enable flags and the deterministic-position literals vary per line. 12h
# is the same safe upper bound the sibling script uses.
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-15%4

set -euo pipefail

# Provenance banner -- FIRST thing written, so a log is never empty. Empty logs
# were what made the 2026-07-29 flock deadlock undiagnosable for hours.
echo "[$(date +%T)] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

# Per-task tmp (helps rootless podman)
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# Ensure podman exists (no module needed on this cluster)
command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }

# Load image tar once per node (protect with a lock for arrays).
#
# 2026-07-29 -- THE LOCK MUST BE NODE-LOCAL, NOT ON $HOME.
# It used to live at $HOME/.locks/podman_load.lock, i.e. on NFS. But the store
# it protects (podman GraphRoot) is /scratch/container, which is PER-NODE. So a
# global lock was serialising independent per-node loads, and -- worse -- flock
# over NFS leaks: when a holder dies uncleanly the lock stays held server-side
# with no live holder, and every later `flock -x` (no timeout) blocks FOREVER.
# Observed 2026-07-29: 3 blocked `flock -x 200` processes at 0:00 CPU, no
# podman load running, jobs "RUNNING" with completely empty .out/.err. The
# stale $HOME/.locks/podman_load_819*.lock files date back to January.
# Fix: put the lock next to the store it guards, and always use a timeout.
LOCKDIR="${PODMAN_LOCK_DIR:-/scratch}"
mkdir -p "$LOCKDIR" 2>/dev/null || LOCKDIR="/tmp"
LOCKFILE="$LOCKDIR/.podman_load_${USER}.lock"
echo "[$(date +%T)] image-load lock: $LOCKFILE"
(
  # -w 900: fail loudly after 15 min instead of hanging the full 12 h walltime.
  flock -w 900 200 || {
    echo "ERROR: timed out (900s) waiting for $LOCKFILE." >&2
    echo "  A previous holder probably died. Check: ps aux | grep flock" >&2
    echo "  Then remove the lock file and resubmit." >&2
    exit 1
  }
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

# GPU preflight -- exit 75 (requeueable) so a bad node fails LOUDLY instead of
# silently consuming an array slot. 6bb19b2's message claimed the ladder sbatch
# got this; it did not -- only the pilot/grasp_retry/realdeploy scripts had it.
echo "[$(date +%T)] GPU preflight ..."
if ! podman run --rm "$IMAGE_TAG" nvidia-smi -L; then
  echo "ERROR: no usable GPU in the container on $(hostname) -- exiting 75." >&2
  exit 75
fi

# Run inside container
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
    python batch_runs/scripts/run_one_ur3.py \
      --jsonl batch_runs/sweeps/UR3Pick_dr_ladder.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "
