# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of Google DeepMind's [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — a GPU-accelerated RL environment suite built on MuJoCo MJX. This fork adds a **custom UR10e robot arm** (`my_ur10`) for reach/pick tasks, with training, evaluation, and **real robot deployment via RTDE**.

Current active branch: `simple_reach` — 6-DOF arm-only reaching task (no gripper).

## Setup

```bash
# Install from source (requires Python 3.10+, uv recommended)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -U "jax[cuda12]"   # GPU; or set JAX_PLATFORM_NAME=cpu for macOS
uv pip install -e ".[all]"

# macOS backend setup (required before any MuJoCo/JAX usage)
export MUJOCO_GL=glfw
export JAX_PLATFORM_NAME=cpu

# Pre-commit hooks (pyink formatting + isort)
pre-commit install
pre-commit run --all-files
```

## Key Commands

```bash
# Training (from repo root)
python learning/train_jax_ppo.py --env_name UR10SimpleReach
# Common training flags:
#   --num_timesteps 1000000  --num_envs 1024  --learning_rate 5e-4
#   --policy_hidden_layer_sizes 64 64 64  --use_wandb  --play_only

# Real robot deployment (from learning/notebooks/)
python UR10_RealRobot_Reach_ONE.py

# Linting & formatting
ruff check mujoco_playground/
pyink --check mujoco_playground/    # 2-space indent, 80-char lines

# Tests
pytest mujoco_playground/_src/
pytest mujoco_playground/_src/registry_test.py  # single test
```

## Architecture

### Environment Layer (`mujoco_playground/_src/`)
- `registry.py` — Central env registry. `registry.load("UR10SimpleReach")` instantiates any env.
- `mjx_env.py` — Base class for all MJX environments.
- `manipulation/my_ur10/` — Custom UR10e environments:
  - `ur10_base.py` — Base class with UR10e model loading and shared utils.
  - `ur10_reach.py` — `UR10SimpleReach` env. 18D obs `[q(6), qd(6), tcp_pos(3), target_pos(3)]`, 6D action (delta joint positions). `ctrl_dt=0.02` (50Hz), `action_scale=0.04`.
  - `ur10pick.py` — Pick task with gripper.
  - `xmls/mjx_reach.xml` — Primary MuJoCo model for reach task.
- `manipulation/` also contains upstream Panda environments.
- Config defaults per domain: `mujoco_playground/config/{manipulation,locomotion,dm_control_suite}_params.py`

### Training (`learning/`)
- `train_jax_ppo.py` — Main PPO training script using Brax. Supports `--env_name`, `--num_timesteps`, wandb logging.
- `notebooks/UR10_ppo.py` — UR10-specific training script with custom hyperparameters.

### Trained Policies (`evaluation/downloaded_policies/`)
- `simple_reach_policy_50hz/` — Production policy. Contains `params.msgpack` + `metadata.json`.
- `simple_reach_policy_200hz/` — Higher frequency variant.
- Each policy dir has `metadata.json` with env_name, obs/action dims, network architecture, and training config.

### Real Robot Deployment (`learning/notebooks/`)
- `URSim_RTDE_dependencies.py` — **Core dependency class** `URSimRTDESimpleReach`. Handles:
  - RTDE connection (`rtde_receive` library) for joint state feedback
  - URScript commands via socket (port 30002 for servoj/movej, port 29999 for dashboard)
  - Policy loading from saved Brax PPO artifacts
  - `policy_step_ctrl_update()` — computes servoj target: `q_measured + action_scale * action`, with actuator range clipping, optional alpha blending, and velocity clamping
  - `run_policy_loop()` — 50Hz control loop: RTDE receive → build obs → policy inference → servoj send
  - MuJoCo video rendering from logged trajectories
  - Diagnostic plots and run metadata export
- `UR10_RealRobot_Reach_ONE.py` — Single-target reach script. Config at top, outputs video + plots + JSON to `results/`.
- `run_multi_target.py` — Multi-target reach with 5mm convergence tolerance.
- `URSim_RTDE_SimpleReach.ipynb` — Comprehensive testing notebook (frequency sweeps, robustness tests, policy comparison).

### Sim-to-Real Control Flow
```
Policy (JAX/Brax) → action ∈ [-1,1]^6
  → policy_step_ctrl_update(q_measured, action)
    → ctrl = q_measured + action_scale * action  (anchored to real joint positions)
    → clip to actuator_ctrlrange
    → optional alpha blending with q_measured
    → optional velocity clamp (max_joint_speed * dt per joint)
  → servoj(ctrl) via URScript socket
```

### Robot Connection
- Real robot IP: `192.168.1.2`
- URSim IP: `127.0.0.1`
- Ports: 30004 (RTDE), 30002 (URScript), 29999 (Dashboard)
- Requires `rtde_receive` Python package (from `ur_rtde` library)

## Important Conventions

- Policy observations must match the 18D layout: `[q(6), qd(6), tcp_pos(3), target_pos(3)]`
- RTDE TCP pose has X/Y axes negated to match MuJoCo base frame orientation
- The `action_scale` in deployment must match training (0.04 for 50Hz policy)
- servoj parameters affect smoothness: `gain` (stiffness 100-2000), `lookahead_time` (smoothing 0.03-0.2s)
- MuJoCo model XML: `mujoco_playground/_src/manipulation/my_ur10/xmls/mjx_reach.xml`
- Code style: pyink with 2-space indentation, 80-char line length, majority quotes. Imports sorted by isort (single-line, 120-char).
