#!/bin/bash
#SBATCH --job-name=UR3Pick_reward_fix_pilot
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Reward-fix pilots (2026-07-26). Two L1_pos_vel runs, now that `success` no
# longer terminates the episode (episodes always run to episode_length):
#   index 1 = L1_pos_vel_pilotA_s0  hold_target=6.0 (default)
#   index 2 = L1_pos_vel_pilotB_s0  hold_target=0.0
# success_bonus defaults to 0.0 in ur3_pick.py and is left at that default in
# both lines -- it's edge-triggered dead weight now that there's nothing to
# forfeit by succeeding; kept in the code only in case termination ever comes
# back. The 2026-07-24 ladder ran ~30 min/run, so 2h is a generous cap.
#
# NOTE (2026-07-26): this deliberately passes NO `--device` to podman. The
# cluster wraps podman (`podman_original version 4.9.3`) and INJECTS
# `--device nvidia.com/gpu=$CUDA_VISIBLE_DEVICES` itself; passing it manually
# is rejected outright ("Direct use of --device with podman is not allowed").
# The manual `--device` line in run_array_ur3_pick_dr_ladder_velocity.sbatch is
# what killed 24 of job 38501's 30 tasks -- MUST be deleted there too.
#
# Partition/time are exactly what the DR ladders have always submitted with
# (partition=gpu, time=12:00:00) -- see b170af5, which ran 30/30 on 2026-07-17.
# Do not change these without checking `sinfo`: `sanfrancisco` is a NODE name
# (what givemeagpushell allocates), not a partition, and `gpu_ia` rejects a 12h
# limit. Both were tried on 2026-07-26 and rejected at submission.
#
# CDI health is per-node, not per-partition. If a task fails with
# "unresolvable CDI devices", get an interactive shell, run `hostname`, and
# check that node's /etc/cdi/nvidia.yaml -- on the sanfrancisco node it
# correctly declares devices "0".."7" (+UUIDs, +all).
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-2

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results"

# Per-task tmp (helps rootless podman)
export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

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

# Preflight: fail LOUDLY and requeueably if the GPU never reaches the
# container. Job 38501 died exactly here on 24/30 tasks but produced no W&B
# run, so the failures were invisible until the .err files were read.
if ! podman run --rm "$IMAGE_TAG" nvidia-smi -L; then
  echo "ERROR: no GPU inside container on $(hostname) -- not starting training"
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
      --jsonl batch_runs/sweeps/UR3Pick_reward_fix_pilot.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "
