# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of Google DeepMind's [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — a GPU-accelerated RL environment suite built on MuJoCo MJX. This fork adds a **custom UR10e robot arm** (`my_ur10`) for reach/pick tasks, with training, evaluation, and **real robot deployment via RTDE**. The active task is `UR10SimpleReach` — 6-DOF arm-only reaching, no gripper.

## Setup

```bash
# Install from source (requires Python 3.10+, uv recommended)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -U "jax[cuda12]"   # GPU; or set JAX_PLATFORM_NAME=cpu for macOS
uv pip install -e ".[all]"

# macOS backend setup (required before any MuJoCo/JAX usage)
export MUJOCO_GL=glfw
export JAX_PLATFORM_NAME=cpu

# Linux backend (lab machine): MUJOCO_GL=egl works out of the box;
# osmesa is NOT installed (avoid MUJOCO_GL=osmesa unless you apt install libosmesa6).
export MUJOCO_GL=egl

# Pre-commit hooks (pyink formatting + isort only; ruff is NOT a hook)
pre-commit install
pre-commit run --all-files

# First import auto-clones MuJoCo Menagerie into external_deps/
python -c "import mujoco_playground"

# Real-robot deployment dep — NOT pulled by .[all]; install once per venv,
# otherwise save_video() crashes with "Could not find a backend ... .mp4".
uv pip install imageio-ffmpeg
```

## Key Commands

```bash
# Training (from repo root)
python learning/train_jax_ppo.py --env_name UR10SimpleReach
# Common training flags:
#   --num_timesteps 10000000  --num_envs 2048  --learning_rate 1e-3
#   --policy_hidden_layer_sizes 64 64 64  --use_wandb  --play_only
# Note: env-specific defaults in config/manipulation_params.py override script defaults.
# UR10SimpleReach uses lr=1e-3, num_envs=2048, entropy_cost=2e-2, discounting=0.97.

# Real robot deployment (from learning/notebooks/)
python UR10_RealRobot_Reach_ONE.py

# Linting & formatting
ruff check mujoco_playground/
pyink --check mujoco_playground/    # 2-space indent, 80-char lines

# Tests
pytest mujoco_playground/_src/
pytest mujoco_playground/_src/registry_test.py  # single test
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main` across Python 3.10–3.12. Installs with `uv pip install -e ".[test]"`, auto-clones MuJoCo Menagerie, runs `pytest -n auto mujoco_playground/_src/`. Note: pre-commit hooks (pyink + isort) are not enforced in CI — run them locally before pushing.

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
  - RTDE connection: `rtde_receive.RTDEReceiveInterface` (joint state feedback) **and** `rtde_control.RTDEControlInterface` (motion commands). Both ship in the `ur_rtde` PyPI package.
  - `send_movej` / `send_servoj` are thin wrappers around `rtde_control.moveJ` / `rtde_control.servoJ`. **Parameter-order footgun:** the wrapper signature keeps URScript-style `(q, a, v, ...)` on the outside, but `rtde_control.servoJ` is `(q, speed, acceleration, ...)` — the swap happens inside the wrapper. Do not flip it at call sites.
  - `disconnect()` calls `servoStop()` + `stopScript()` before tearing down the control interface; `run_policy_loop` wraps its main loop in `try / finally` with a `servoStop()` so a KeyboardInterrupt or early break doesn't leave the controller holding the last servoj target.
  - On the real UR10e, PolyScope must be in **Remote Control** mode or `RTDEControlInterface()` connects but `servoJ` silently fails. URSim doesn't require this.
  - Policy loading from saved Brax PPO artifacts
  - `policy_step_ctrl_update()` — computes servoj target: `q_measured + action_scale * action`, with actuator range clipping, optional alpha blending, and velocity clamping
  - `run_policy_loop()` — 50Hz control loop: RTDE receive → build obs → policy inference → servoj send
  - MuJoCo video rendering from logged trajectories
  - Diagnostic plots and run metadata export
- `UR10_RealRobot_Reach_ONE.py` — Single-target reach script. Config at top, outputs video + plots + JSON to `results/`.
- `run_multi_target.py` — Multi-target reach with 5mm convergence tolerance.
- `URSim_RTDE_SimpleReach.ipynb` — Comprehensive testing notebook (frequency sweeps, robustness tests, policy comparison).

### Motion Capture (`motion_capture/`)
- Nokov XING Python SDK (vendored from `XING_Python_SDK-4.1.0.5873/`) for streaming marker/body poses from a Nokov mocap system. Imported from Melvin's setup as the basis for a closed-loop "real TCP from mocap, not from RTDE forward-kinematics" pipeline.
- The SDK ships a `.whl` for Python 3.9 and example clients (`Nokov_SDK_Client*.py`). It is **Windows-only at the moment** — Linux integration is pending; on this Linux dev machine the wheel is not installed and the scripts are reference material only.

### Sim-to-Real Control Flow
```
Policy (JAX/Brax) → action ∈ [-1,1]^6
  → policy_step_ctrl_update(q_measured, action)
    → ctrl = q_measured + action_scale * action  (anchored to real joint positions)
    → clip to actuator_ctrlrange
    → optional alpha blending with q_measured
    → optional velocity clamp (max_joint_speed * dt per joint)
  → rtde_control.servoJ(ctrl, speed, accel, t, lookahead, gain)
```

### Robot Connection
- Real robot IP: `192.168.1.2` (lab machine via Lindy USB→Ethernet adapter, host typically at `192.168.1.89/24`)
- URSim IP: `127.0.0.1`
- Port 30004 (RTDE) is the only socket the code touches now. URScript (30002) and Dashboard (29999) ports are no longer used since the rewrite to `rtde_control`.
- Requires `ur_rtde` package, providing both `rtde_receive` and `rtde_control`. If only one imports, the install is partial — reinstall `ur_rtde`.

## Important Conventions

- Policy observations must match the 18D layout: `[q(6), qd(6), tcp_pos(3), target_pos(3)]`
- RTDE TCP pose has X/Y axes negated to match MuJoCo base frame orientation
- The `action_scale` in deployment must match training (0.04 for 50Hz policy)
- servoj parameters affect smoothness: `gain` (stiffness 100-2000), `lookahead_time` (smoothing 0.03-0.2s). `rtde_control.servoJ` **hard-enforces** the `lookahead_time ∈ [0.03, 0.2]` range — a value outside it raises `ValueError("The value is not within [0.03;0.2]")` on the first servoj call inside the policy loop.
- MuJoCo model XML: `mujoco_playground/_src/manipulation/my_ur10/xmls/mjx_reach.xml`
- Code style: pyink with 2-space indentation, 80-char line length, majority quotes. Imports sorted by isort (single-line, 120-char). Ruff config is in `pyproject.toml` (no standalone `ruff.toml`). `__init__.py` files are excluded from pre-commit formatting.
- `mujoco_menagerie/` and `external_deps/` are excluded from all linting/formatting tools.
