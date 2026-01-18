#!/usr/bin/env bash
set -euo pipefail

# Pick a venv name; allow override from environment
VENV="${VENV:-mujoco_env_egl}"
export VENV

module purge || true
module load python/3.12.4

# Create/activate uv venv (same name you used before)
VENV="$VENV" module load uv/0.6.12

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not on PATH after 'module load uv/0.6.12'." >&2
  exit 1
fi

# Create venv if missing (cluster module convention uses /scratch/venvs/$VENV)
uv venv
source "/scratch/venvs/${VENV}/bin/activate"

python --version
which uv || true

# Caches
export HF_HOME="/scratch/hf_home/${USER}"
export WANDB_DIR="/scratch/wandb/${USER}"
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

# Make repo-root imports work (you do NOT have src/ in your tree)
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Install your pinned deps
uv pip install -r requirements.txt

# JAX is needed by UR10_ppo.py; your requirements.txt does not include it
uv pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

uv pip freeze > /tmp/pip_freeze_env_bootstrap.txt || true
