# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

Fork of Google DeepMind's [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — a GPU-accelerated RL environment suite built on MuJoCo MJX. This fork adds **custom Universal Robots arms** for reach/pick tasks, with training, evaluation, and **real robot deployment via RTDE**:
- **UR10e** (`my_ur10`) — `UR10SimpleReach` (6-DOF arm-only reaching, no gripper) and `UR10PickCube` (pick with a Hand-E gripper).
- **UR3e + Robotiq Hand-E** (`my_ur3`) — `UR3Pick`, actively developed on the `ur3_pick` branch: a 7-DOF (6 arm + 1 Hand-E tendon) pick task whose box target is fed from Nokov motion capture at deploy time, trained on the ZHAW SLURM cluster.

**`ur3_pick` branch policy:** the UR3 work is a *structural sibling* of the UR10 work — each UR3 artifact is a separate copy of a UR10 one (XMLs, env classes, sweep/sbatch, deployment scripts), not a shared/refactored abstraction. Keep all UR10 / URSim / reach / `simple_reach_*` files intact.

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

# UR3 pick — local smoke run (verifies env loads + trains without NaN)
python learning/train_jax_ppo.py --env_name UR3Pick \
  --num_timesteps 200000 --num_envs 256 --num_evals 2 --episode_length 150
# UR3Pick defaults: lr=1e-3, num_envs=2048, entropy_cost=2e-2, discounting=0.97,
# policy_hidden_layer_sizes=(32,32,32,32), num_timesteps=20M.

# Cluster training (ZHAW SLURM, rootless Podman + EGL). Submit from repo root:
sbatch --array=1 batch_runs/slurm/run_array_ur3.sbatch   # ur3pick_smoke (line 1 of sweep)
sbatch --array=2 batch_runs/slurm/run_array_ur3.sbatch   # ur3pick_base baseline (line 2)
sbatch --array=1-10 batch_runs/slurm/run_array_ur3.sbatch  # full hyperparameter sweep
# Sweep config: batch_runs/sweeps/UR3Pick_sweep.jsonl (1-indexed by SLURM array task).
# Lines: 1 smoke, 2 base, 3 base_seed1, 4 lr_low, 5 lr_high, 6 entropy_high,
#        7 gamma_high, 8 unroll_long, 9 envs_large, 10 DR_MFR.

# Real robot deployment (from learning/notebooks/)
python UR10_RealRobot_Reach_ONE.py                       # UR10 reach
python ur3_realrobot_pickloop.py                         # UR3 pick (mocap box target)

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
- `manipulation/my_ur3/` — Custom UR3e + Robotiq Hand-E pick environment (`ur3_pick` branch), mirroring `my_ur10/`:
  - `ur3_base.py` — `UR3Base`. `_ARM_JOINTS` are the 6 standard UR joint names; `_FINGER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]`. `get_assets()` reads from `my_ur3/xmls` + `my_ur3/universal_robots_ur3e` (+ its `assets/`). `_post_init` resolves `self._left_finger_touch` / `self._right_finger_touch` (site IDs for `left_finger_touch_site` / `right_finger_touch_site` on the inner faces of the Hand-E fingers) — used by `ur3_pick._get_reward` for `finger_touch_dist`.
  - `ur3_pick.py` — `UR3Pick` env. **Pick + lift task** (drop/place stage deferred). **20D obs** `[q(8 arm+finger), qd(6 arm), (box−tcp)(3), (target_pos−box)(3)]`, **7D action** (6 arm delta + 1 Hand-E tendon). `ctrl_dt=0.02`, `sim_dt=0.005`, `action_scale=0.04`, `episode_length=150`, `init_keyframe="low_home"`. Reward scales `{box_target:8.0, gripper_box:4.0, finger_touch:1.0, no_floor_collision:0.25, robot_target_qpos:0.3}` — the `_get_reward` body mirrors `ur10pick.py`'s commented scaffolding (sections: positions / distances / rotation [commented] / reward terms / `reached_box` gate) so future re-enables are comment-flips. `target_pos` is sampled at reset as a **lift point in the air above the box** (`init_obj_pos + uniform([-0.1,-0.1,0.10], [0.1,0.1,0.25])`), written to `mocap_pos` for viz, and stored in `info["target_pos"]`. `info["reached_box"]` is a sticky gate that latches at `gripper_box_dist < 0.02`. Success: `||box − target_pos|| < 0.03` for 3 consecutive steps. `out_of_bounds`: `|tcp.xy| > 0.6` (UR3 reach ≈ 0.5 m).
  - `universal_robots_ur3e/ur3e_position.xml` — UR3e arm with Hand-E **inlined** at the wrist_3 attachment (`pos="0 0.09215 0"`, vs UR10's 0.1): mount body, two coupled slide-joint fingers via `<tendon><fixed name="split">` (coef 0.5 each), one `position` actuator `ctrlrange="0 0.05"` → `nu=7`, `nq=8` standalone. **meshdir footgun:** set to `../universal_robots_ur3e/assets` so it resolves from the top-level loader (`xmls/`), not the included file's dir. Hand-E meshes (`hande.stl`, `coupler.stl`, `finger_*.stl`) live in `universal_robots_ur3e/assets/` — there is **no** separate `robotiq_hande/` dir (unlike `my_ur10/`).
  - `xmls/mjx_single_cube_position_ur3.xml` — pick scene: includes the robot + scene, box freejoint, `mocap_target` body, `ur3_pick_sensor.xml`. Loads to `nu=7, nq=15, nkey=5, nsensor=8`. Keyframes `task_home`/`low_home`/`tucked` (15 qpos = 6 arm + 2 finger + 7 box; 7 ctrl).
  - `xmls/ur3_pick_sensor.xml` — 3 floor-contact + 3 finger/box-contact + 2 framepos sensors.
- `manipulation/` also contains upstream Panda environments.
- Config defaults per domain: `mujoco_playground/config/{manipulation,locomotion,dm_control_suite}_params.py`

### Training (`learning/`)
- `train_jax_ppo.py` — Main PPO training script using Brax. Supports `--env_name`, `--num_timesteps`, wandb logging.
- `notebooks/UR10_ppo.py` — **env-agnostic** training entry point: `run_experiment(cfg)` dispatches on `cfg["env_name"]` via `registry.load()` and logs to W&B. Reused as-is for UR3 (no `UR3_ppo.py` needed).

### Cluster Training (`batch_runs/`)
ZHAW SLURM cluster, rootless Podman container (`mujoco_env_EGL.tar`, tag `mujoco:env_EGL`) with `MUJOCO_GL=egl`. A sweep is a JSONL file (one config per line, **1-indexed** by `$SLURM_ARRAY_TASK_ID`); each array task trains one config and uploads a `policy_parameters_*` artifact to W&B.
- `slurm/run_array_ur{10,3}.sbatch` — SLURM array job. Loads the image tar under a `flock`, mounts the repo at `/workspace`, runs `scripts/run_one_ur{10,3}.py`. `REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"`; W&B key read from `$HOME/.secrets/wandb_key`.
- `scripts/run_one_ur{10,3}.py` — thin per-task runner: parses one JSONL line (`--jsonl … --index … --out_root …`) and calls `UR10_ppo.run_experiment`. The UR3 copy differs only in defaults; the import target is identical (env-agnostic).
- `sweeps/UR3Pick_sweep.jsonl` — 10 lines, mirrors the UR10 sweep's hyperparameter-exploration pattern. `run_id`s name the variation: `ur3pick_smoke` (300k/512 envs), `ur3pick_base` (20M baseline), `ur3pick_base_seed1`, `ur3pick_lr_low_5e-4`, `ur3pick_lr_high_2e-3`, `ur3pick_entropy_high_5e-2`, `ur3pick_gamma_high_0.99`, `ur3pick_unroll_long_50`, `ur3pick_envs_large_8k`, `ur3pick_DR_MFR`. Each line carries a unique `seed` (0–8). Variation directions are *vs UR3 defaults* (lr=1e-3, entropy=0.02, γ=0.97, unroll=10) — γ and unroll vary *up* because UR3 defaults are lower than UR10's. All `env_name="UR3Pick"`, `wandb_project="UR3_pick_ppo"`.

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
- `ur3_realrobot_dependencies.py` — `UR3RealRobotPick`, the UR3 pick analog of `URSim_RTDE_dependencies.py`. Key differences:
  - `build_obs_from_feedback(fb, box_pos, drop_target, tcp_pos)` currently still assembles the old 21D obs. **STALE**: the sim env was rewritten to 20D `[q(8), qd_arm(6), (box−tcp)(3), (target−box)(3)]`. This function needs updating before the next real-robot run; the deploy script's `drop_target` arg becomes the lift target `target_pos`.
  - `policy_step_ctrl_update` returns `(arm_ctrl[6], gripper_norm)` — the 7th action drives `self._gripper_ctrl` clipped to `[0, 0.05]`, normalized to `[0, 1]`. **`servoJ` accepts only 6 joints**, so the gripper goes on a separate channel.
  - `send_gripper(norm_cmd)` is a **STUB** (`raise NotImplementedError`) — it must be wired to the lab's specific Hand-E setup (Robotiq URCap RTDE register vs. tool I/O) before a real-robot run.
  - `run_policy_loop(drop_target, mocap_reader, …, gripper_fn, use_fk_tcp)` reads `mocap_reader.get_rigid_body_xyz()` every tick, falling back to the last good value when a frame is missing.
- `ur3_realrobot_pickloop.py` — single-run UR3 pick entry (mirror of `UR10_RealRobot_Reach_ONE.py`). Config at top: `ROBOT_IP`, `DROP_TARGET`, `Q_START` (UR3 `low_home`), `MOCAP_SERVER_IP`/`MOCAP_RIGID_BODY_ID`, `ENABLE_GRIPPER` (False = no-op gripper for URSim dry-runs). For a dry-run set `ROBOT_IP=127.0.0.1` and leave `ENABLE_GRIPPER=False`.

### Motion Capture (`motion_capture/`)
- Nokov (XINGYING) mocap streaming for closed-loop "real target from mocap, not hardcoded" reaching. The SDK is now **pip-installed** from `motion_capture/nokov_python_sdk-master/dist/nokovpy-3.0.1-py3-none-any.whl` (`uv pip install …whl` — gives `from nokov.nokovsdk import PySDKClient` with no sys.path hack). The wheel bundles `libnokov_sdk.so` (x86_64 + aarch64) plus the Windows DLL. The older extracted copy at `motion_capture/XING_Python_SDK-4.1.0.5873/dist/nokovpy-3.0.1-py3.9/` is still on disk and is what `motion_capture_reader.py` / `mocap_publisher.py` reach into via `sys.path`; only `test_mocap_connection.py` has migrated to the pip-installed package.
- Active Linux integration on `linux_motion_capture` branch. Layout:
  - `motion_capture/mymocap/test_mocap_connection.py` — **scene-inventory** streamer. At startup it calls `PyGetDataDescriptions()` to dump every skeleton / markerset / rigid-body definition the server publishes (with names + IDs), then in the data callback prints a full per-frame list of every entity in each frame (skeleton segments, markerset marker counts, rigid-body pose, unidentified-marker count). CLI: `--server-ip` (default `10.1.1.198`), `--duration` (default 10 s). Run this first to verify end-to-end before integrating into the robot loop.
  - `motion_capture/mymocap/motion_capture_reader.py` — `XINYINGReader` class. Background thread connects to the Nokov server, receives marker frames (~60Hz), stores latest poses in a thread-safe dict. Public API: `get_first_marker_position()`, `get_marker_centroid([...])`, `get_markers_in_region(center, radius)`, `is_connected()`. **Markers-only** (no rigid-body API yet).
  - `motion_capture/mymocap/mocap_publisher.py` / `mocap_receiver.py` — split-process variant (publisher on the machine that can talk to the mocap server, receiver in the control loop) for when the SDK can't run in-process alongside RTDE. **Markers-only.**
  - `motion_capture/nokov_python_sdk-master/` — unzipped upstream SDK from the advisor (`nokov_python_sdk-master.zip`). `dist/` ships the wheel that is now pip-installed; `examples/` (including `Nokov_SDK_Client.py` and a `Utility.py` with `Point` / `SlideFrameArray` / `CalculateVelocity` velocity helpers) is the read-only reference. Not on `sys.path` — examples are reference, not import targets.
  - `motion_capture/INTEGRATION_PLAN.md` — design doc for the planned `URSimRTDESimpleReach.initialize_mocap()` / `get_mocap_target()` hooks and the 50Hz-loop ↔ 60Hz-mocap frequency-independent read pattern (background thread writes latest pose; control loop reads whenever it needs).
  - `motion_capture/mocap_dependencies.py` — `NokovRigidBodyReader` (used by the UR3 pick loop). Unlike the markers-only readers above, this reads a **rigid body** from `frame.RigidBodies[i]` (matching `body.ID == rigid_body_id`, or the first body if `None`) and exposes only its XYZ center via `get_rigid_body_xyz() -> np.ndarray(3,) | None` (mm→m). Background daemon thread + `Lock`. Prefers the pip-installed wheel, falls back to the vendored SDK path. Does **not** call `Uninitialize()`.
- **Credentials**: none. The SDK authenticates with only the server IP via `client.Initialize(bytes(ip, "utf8"))`. There are no secrets in the code.
- **Server-side prerequisites** (on the Windows mocap PC, not in this repo): XINGYING must have **"SDK Enabled"** toggled ON, and at least one **rigid body / markerset / skeleton** must be defined and visible to the cameras. Without "SDK Enabled" the client connects but zero frames arrive. With "SDK Enabled" but an empty scene, frames arrive at 60 Hz but every frame has zero entities — `test_mocap_connection.py` distinguishes these two cases in its watchdog output.
- **`PySDKClient` API footgun**: there is **no** `Uninitialize()` method (verified via `dir(PySDKClient())`). The advisor's examples don't clean up at all — just let the script exit. `motion_capture_reader.py:137` and `mocap_publisher.py:131` still call `self.client.Uninitialize()` and will `AttributeError` on shutdown if exercised; `test_mocap_connection.py` has had this removed.
- **Network**: USB Ethernet adapter to mocap subnet `10.1.1.0/24`, server at `10.1.1.198`. Lab wifi is offline (no internet) — all SDK deps must already be vendored.

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

- UR10 reach policy observations must match the 18D layout: `[q(6), qd(6), tcp_pos(3), target_pos(3)]`. UR3 pick is **20D**: `[q(8 arm+finger), qd(6 arm), (box−tcp)(3), (target_pos−box)(3)]` (relative vectors, no raw world positions) with a **7D** action (the 7th element is the Hand-E tendon command). Obs/action dims and ordering must match between sim training and deployment.
- RTDE TCP pose has X/Y axes negated to match MuJoCo base frame orientation
- For UR3 pick, the gripper is **not** part of the `servoJ` `q`-vector (servoJ is 6-joint only) — it travels on a separate channel via `send_gripper()`, which is an unwired stub until the lab Hand-E I/O is confirmed.
- The `action_scale` in deployment must match training (0.04 for 50Hz policy)
- servoj parameters affect smoothness: `gain` (stiffness 100-2000), `lookahead_time` (smoothing 0.03-0.2s). `rtde_control.servoJ` **hard-enforces** the `lookahead_time ∈ [0.03, 0.2]` range — a value outside it raises `ValueError("The value is not within [0.03;0.2]")` on the first servoj call inside the policy loop.
- MuJoCo model XML: UR10 reach `mujoco_playground/_src/manipulation/my_ur10/xmls/mjx_reach.xml`; UR3 pick `mujoco_playground/_src/manipulation/my_ur3/xmls/mjx_single_cube_position_ur3.xml`
- Code style: pyink with 2-space indentation, 80-char line length, majority quotes. Imports sorted by isort (single-line, 120-char). Ruff config is in `pyproject.toml` (no standalone `ruff.toml`). `__init__.py` files are excluded from pre-commit formatting.
- `mujoco_menagerie/` and `external_deps/` are excluded from all linting/formatting tools.
