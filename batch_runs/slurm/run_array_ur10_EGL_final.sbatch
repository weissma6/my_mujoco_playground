#!/bin/bash
#SBATCH --job-name=UR10_array
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-100%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
cd "$REPO_DIR"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Each array task gets its own node slot when the scheduler places it.
# With %10 you run up to 10 in parallel (on up to 10 nodes, if available).
srun python -u batch_runs/scripts/run_one_ur10.py \
  --jsonl batch_runs/sweeps/UR10_sweep.jsonl \
  --index "$SLURM_ARRAY_TASK_ID" \
  --out_root results
