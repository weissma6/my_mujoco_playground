#!/bin/bash
#SBATCH --job-name=UR3Pick_grasp_retry
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# GRASP-RETRY: reproduce the first hardware success (052d79f, Align_logic) +
# user deltas + the cube-mass DR that transferred, now on the REAL 95 mm lab
# table (3e67921) with the table height randomized. 2 runs, init pose A/B,
# everything else identical (seed 1):
#   index 1 = Tablelift_light   init_start_random="light"
#   index 2 = Tablelift_mid     init_start_random="mid"
# Both: lifter_height_max=0.02 -> table top ~ U(0.075, 0.115) every episode
# (95 mm +-20 mm), level (lifter_tilt_max=0). The draw is in the env's reset(),
# so it applies to the eval/video rollouts too, and the lift target is shifted
# by the same per-episode deviation -- the cube always spawns resting ON the
# realized table top and the lift stays a fixed 0.18..0.21 m ABOVE it.
# results/ dirs are named after run_id, so keep run_ids space-free.
# See batch_runs/sweeps/UR3Pick_grasp_retry.jsonl for full provenance/config.
# ~14.4M steps/run (num_minibatches=64, num_evals=12); the 07-24/26 velocity
# ladders ran ~30 min/run at ~2x this step count, so 12h is a generous cap.
#
# NOTE: this deliberately passes NO `--device` to podman. The cluster wraps
# podman (`podman_original` 4.9.3) and INJECTS `--device
# nvidia.com/gpu=$CUDA_VISIBLE_DEVICES` itself; passing it manually is
# rejected outright ("Direct use of --device with podman is not allowed").
# Manual `--device` is what killed 24 of job 38501's 30 tasks.
#
# Partition/time are exactly what the DR ladders have always submitted with
# (partition=gpu, time=12:00:00) -- see 5e610d3, "the header that ran 30/30
# on 07-17".
# Do not change these without checking `sinfo` first: `sanfrancisco` and
# `losangeles` are NODE names, not partitions, and `gpu_ia` rejects a 12h
# limit -- both were tried on 2026-07-26 and rejected at submission.
#
# CDI/launch health is per-node, not per-partition. Job 39029 (2026-07-26)
# lost 24/30 tasks, ALL on node `losangeles`, ALL exit code 126 ("found but
# cannot execute") -- distinct from the earlier CDI exit-125 failure mode,
# and NOT caught by the exit-75 preflight below (that preflight never got
# far enough to report). Root cause not yet confirmed -- read a failed
# task's .err before resubmitting anything at scale on that node. If this
# job lands on `losangeles`, do not trust a silent success; check
# `sacct -j <jobid> --format=JobID,NodeList,State,ExitCode` before assuming
# both tasks trained.
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
      --jsonl batch_runs/sweeps/UR3Pick_grasp_retry.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "
