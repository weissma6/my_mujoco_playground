#!/bin/bash
#SBATCH --job-name=UR3Pick_gap_seedsel
#SBATCH --account=cai_iai
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# D10 median-seed selection for the gap protocol (see "Plan - Sim-to-Real Gap
# Protocol", D10 + Commit 7). Runs run_gap_protocol_sim.py --protocol_only for
# all 4 surviving DR-ladder configs x 3 seeds = 12 short MJX inference jobs
# (10 protocol episodes x 400 steps each), then select_median_seed.py to write
# robots/UR3e/gap_protocol_policy_map.json. NO training here -- inference only,
# so a few hours is plenty. Single job (not an array): the 12 rollouts are
# cheap; the cost is jax/env build + jit compile per invocation.
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"
IMAGE_TAR="$REPO_DIR/mujoco_env_EGL.tar"
IMAGE_TAG="mujoco:env_EGL"
LOCALHOST_TAG="localhost/mujoco:env_EGL"

mkdir -p "$REPO_DIR/logs"

export TMPDIR="$HOME/tmp_podman/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

command -v podman >/dev/null 2>&1 || { echo "ERROR: podman not found on PATH"; exit 2; }

# Load the image tar if not already present
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

# W&B key (needed: the sim script downloads best_params.msgpack from W&B)
if [ -f "$HOME/.secrets/wandb_key" ]; then
  export WANDB_API_KEY="$(< "$HOME/.secrets/wandb_key")"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: no WANDB_API_KEY -- put your key in \$HOME/.secrets/wandb_key" >&2
  exit 3
fi

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# The 4 surviving DR-ladder configs (D14) x 3 seeds = 12 (config, seed, run_id).
# Run ids are the W&B run names in project UR3_pick_ppo (name == id here).
# Update these only if you retrain the ladder.
# ---------------------------------------------------------------------------
RUNS=(
  "L0_none:0:L0_none_s0_20260717_121358_6311"
  "L0_none:1:L0_none_s1_20260717_121551_2201"
  "L0_none:2:L0_none_s2_20260717_121754_926"
  "L1_pos:0:L1_pos_s0_20260717_121957_6311"
  "L1_pos:1:L1_pos_s1_20260717_124401_2201"
  "L1_pos:2:L1_pos_s2_20260717_124604_926"
  "L2_pos_cube:0:L2_pos_cube_s0_20260717_124807_6311"
  "L2_pos_cube:1:L2_pos_cube_s1_20260717_125159_2201"
  "L2_pos_cube:2:L2_pos_cube_s2_20260717_131601_926"
  "L3_pos_cube_robot:0:L3_pos_cube_robot_s0_20260717_131805_6311"
  "L3_pos_cube_robot:1:L3_pos_cube_robot_s1_20260717_132102_2201"
  "L3_pos_cube_robot:2:L3_pos_cube_robot_s2_20260717_132430_926"
)

SIM_ROOT="robots/UR3e/sim_results/seed_selection"

run_in_container() {
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
    bash -lc "$1"
}

# ---------------------------------------------------------------------------
# Step 1: 12 sim rollouts (--protocol_only, one --out_root per seed).
# Resumable: run_gap_protocol_sim.py skips episode folders that already exist,
# so re-submitting after a timeout continues where it stopped.
# ---------------------------------------------------------------------------
echo "=== jax device check ==="
run_in_container "python -c 'import jax; print(\"jax devices:\", jax.devices())' || true"

for entry in "${RUNS[@]}"; do
  IFS=":" read -r cfg seed run_id <<< "$entry"
  echo "=== SIM  config=$cfg  seed=$seed  run_id=$run_id ==="
  run_in_container "python evaluation/run_gap_protocol_sim.py \
    --config_id $cfg \
    --policy_run_id $run_id \
    --protocol_only \
    --out_root $SIM_ROOT/s$seed"
done

# ---------------------------------------------------------------------------
# Step 2: pick the median seed per config -> write gap_protocol_policy_map.json.
# Pure Python/file-IO (no MJX), but run in-container for a consistent env.
# ---------------------------------------------------------------------------
echo "=== SELECT median seed -> gap_protocol_policy_map.json ==="
run_in_container "python evaluation/select_median_seed.py --config_id L0_none \
  --seed_run_id_map '{\"0\":\"L0_none_s0_20260717_121358_6311\",\"1\":\"L0_none_s1_20260717_121551_2201\",\"2\":\"L0_none_s2_20260717_121754_926\"}' \
  --sim_root $SIM_ROOT"
run_in_container "python evaluation/select_median_seed.py --config_id L1_pos \
  --seed_run_id_map '{\"0\":\"L1_pos_s0_20260717_121957_6311\",\"1\":\"L1_pos_s1_20260717_124401_2201\",\"2\":\"L1_pos_s2_20260717_124604_926\"}' \
  --sim_root $SIM_ROOT"
run_in_container "python evaluation/select_median_seed.py --config_id L2_pos_cube \
  --seed_run_id_map '{\"0\":\"L2_pos_cube_s0_20260717_124807_6311\",\"1\":\"L2_pos_cube_s1_20260717_125159_2201\",\"2\":\"L2_pos_cube_s2_20260717_131601_926\"}' \
  --sim_root $SIM_ROOT"
run_in_container "python evaluation/select_median_seed.py --config_id L3_pos_cube_robot \
  --seed_run_id_map '{\"0\":\"L3_pos_cube_robot_s0_20260717_131805_6311\",\"1\":\"L3_pos_cube_robot_s1_20260717_132102_2201\",\"2\":\"L3_pos_cube_robot_s2_20260717_132430_926\"}' \
  --sim_root $SIM_ROOT"

echo "=== DONE. Policy map: ==="
cat robots/UR3e/gap_protocol_policy_map.json || echo "MISSING -- check the select step output above"
