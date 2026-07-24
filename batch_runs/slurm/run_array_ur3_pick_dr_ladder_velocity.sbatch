#!/bin/bash
#SBATCH --job-name=UR3Pick_DR_ladder_velocity
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Velocity-obs variant of the DR ladder (see the vault's "Plan - Add
# Velocity"): same 10 configs x 3 seeds = 30 runs as run_array_ur3_pick_dr_
# ladder.sbatch, this sweep just adds obs_include_velocity=true + explicit
# action_scale=0.04 to every line (see gen_dr_ladder_velocity.py). Same
# per-run cost (num_timesteps/num_envs/batch_size/LR/episode_length
# untouched) -- 12h is the same safe upper bound the sibling scripts use,
# even though the one pilot run so far converged much earlier (~10M steps);
# left at the standard budget until the whole ladder confirms that holds.
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=1-30%4

set -euo pipefail

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
      --jsonl batch_runs/sweeps/UR3Pick_dr_ladder_velocity.jsonl \
      --index $SLURM_ARRAY_TASK_ID \
      --out_root results
  "
