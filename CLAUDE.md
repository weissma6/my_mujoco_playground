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

**Keep scripts and notebooks simple.** Hardware bring-up notebooks (e.g. `robots/hande/`) must be dead-simple and step-by-step: one concern per cell, minimal boilerplate, no clever path-finding or guard scaffolding. A cell that hangs with no output is worse than a cell that fails loudly — prefer short, observable steps.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- No unnecessary prints — keep only what aids safety/observability or reports the result.
- Occam's razor for functions: write one only if it aids reuse or understanding. If it exists purely for completeness, don't write it.
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
# policy_hidden_layer_sizes=(32,32,32,32), num_timesteps=20M, episode_length=250.

# Cluster training (ZHAW SLURM, rootless Podman + EGL). Submit from repo root:
sbatch --array=1 batch_runs/slurm/run_array_ur3.sbatch    # config 1 (env2048)
sbatch --array=1-2 batch_runs/slurm/run_array_ur3.sbatch  # full sweep (2 configs)
# Sweep config: batch_runs/sweeps/UR3Pick_sweep.jsonl (1-indexed by SLURM array task).
# 2 lines (PicknDrop_un20_env{2048,1024}): fixed lr=8e-4, unroll=20, num_timesteps=20M,
# seed 0, init_keyframe="tucked" (overrides the env's low_home default), num_envs 2048/1024,
# wandb_project=UR3_PicknDrop.

# Real robot deployment (run from the script's own folder so its ../../ paths resolve)
cd robots/UR10e && python UR10_RealRobot_Reach_ONE.py    # UR10 reach
cd robots/UR3e  && python ur3_realrobot_pickloop.py      # UR3 pick (mocap box target)

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
- `registry.py` — Central env registry. `registry.load("UR10SimpleReach")` instantiates any env. **Registration happens in `manipulation/__init__.py`**, not here: to add an env, add it to the `_envs` (class) and `_cfgs` (`default_config`) dicts there — the UR envs are wired at `__init__.py:54-56` / `:71-73`.
- `mjx_env.py` — Base class for all MJX environments.
- `manipulation/my_ur10/` — Custom UR10e environments:
  - `ur10_base.py` — Base class with UR10e model loading and shared utils.
  - `ur10_reach.py` — `UR10SimpleReach` env. 18D obs `[q(6), qd(6), tcp_pos(3), target_pos(3)]`, 6D action (delta joint positions). `ctrl_dt=0.02` (50Hz), `action_scale=0.04`.
  - `ur10pick.py` — Pick task with gripper.
  - `xmls/mjx_reach.xml` — Primary MuJoCo model for reach task.
- `manipulation/my_ur3/` — Custom UR3e + Robotiq Hand-E pick environment (`ur3_pick` branch), mirroring `my_ur10/`:
  - `ur3_base.py` — `UR3Base`. `_ARM_JOINTS` are the 6 standard UR joint names; `_FINGER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]`. `get_assets()` reads from `my_ur3/xmls` + `my_ur3/universal_robots_ur3e` (+ its `assets/`). `_post_init` resolves `self._left_finger_touch` / `self._right_finger_touch` (site IDs for `left_finger_touch_site` / `right_finger_touch_site` on the inner faces of the Hand-E fingers) — used by `ur3_pick._get_reward` for `finger_touch_dist`.
  - `ur3_pick.py` — `UR3Pick` env. **Pick-and-place task** (box spawns on the **+Y** side ~20 cm out with a random Y-tilt; carried to a drop zone on the **−Y** side ~20 cm away **at box-resting height** so a released box settles *inside* the zone). **26D obs** `[q(8 arm+finger), qd(6 arm), (box−tcp)(3), (target_pos−box)(3), box_xmat.ravel()[:6](6)]`, **7D action** (6 arm delta + 1 Hand-E tendon) — obs/action layout unchanged. `ctrl_dt=0.02`, `sim_dt=0.005`, `action_scale=0.04`, `episode_length=250`, `init_keyframe="low_home"`. **Staged reward (9 terms) sequenced by sticky `info` latches** so phases don't fight; scales `{gripper_box:2.0, grasp:3.0, lift:5.0, box_target:8.0, box_inside:6.0, release:3.0, retract:4.0, no_floor_collision:0.25, robot_target_qpos:0.3}`.
    - **Sticky latches** (monotone via `jp.maximum`, re-zeroed each episode): `reached_box` (`gripper_box_dist < 0.02`), `grasped` (`reached_box` ∧ **both** `left/right_finger_box_contact` sensors fire), `lifted` (`grasped` ∧ `box_z > box_rest_z + lift_eps` — the **anti-push** latch), `delivered` (`lifted` ∧ `containment_violation < 5e-3`).
    - **Anti-push design** (why pickup was lost before this rewrite): `box_target = (1 − tanh(5·box_target_dist)) · lifted` is **gated by `lifted`**, so sliding the box along the floor earns nothing — grasp→lift is the *only* path to the dominant transport reward. (Previously `box_target` was ungated and the target sat ~5 cm above the floor on the −Y side, so the policy pushed the box instead of lifting it.)
    - Terms (with `near_target = 1 − tanh(10·box_target_dist)`, `finger_open = tanh(finger_touch_dist/0.05)`, `finger_closed = 1 − finger_open`): `gripper_box = 1 − tanh(5·gripper_box_dist)` (approach, always on); `grasp = finger_closed · reached_box · (1 − near_target)` (close on box); `lift = tanh(clip(box_z − rest,0,0.12)/0.06) · reached_box · (1 − near_target)`; `box_inside = (1 − tanh(15·containment_violation)) · grasped` (8-corner containment of the 4 cm box in the 5 cm target box, `target_half=0.025`); `release = finger_open · near_target · lifted` (open at the drop); `retract = (0.5·tanh(Δz/0.05) + 0.5·tanh(Δxy/0.05)) · delivered` (TCP up/away so the box ends in the zone **without** the gripper); `robot_target_qpos = 1 − tanh(||arm_qpos − init||)`. `grasp`/`lift` fade via `(1 − near_target)` (not via a latch) so latching a stage never causes a reward cliff. **Dropped** `box_orient` (redundant — `box_inside` already requires upright; `rot_err` kept as a logged diagnostic) and `finger_touch`.
    - **`curriculum` sub-config** (knobs read by `reset`/`step`): `tilt_deg` (spawn ±tilt about Y, **default 15°**, ramp toward 45), `lift_eps` (0.03), `spawn_xy_jitter` (0.05, **shared** pick/drop XY radius), `success_tol` (1e-3).
    - Reset: box spawn `init_obj_pos + [0,0.20,0] + uniform(±[xy,xy,0])`; `target_pos` `init_obj_pos + [0,-0.20,0] + uniform(±[xy,xy,0])` (**z = box rest ≈ 0.02**, written to `mocap_pos` for viz, stored in `info["target_pos"]`); random Y-tilt `θ ∈ (−tilt, +tilt)` baked into the box quat. Success: `max(corner_overshoot) ≤ success_tol` for 3 consecutive steps. `out_of_bounds`: `|tcp.xy| > 0.6` (UR3 reach ≈ 0.5 m). Grasp detection uses the `left/right_finger_box_contact` sensors already present in `ur3_pick_sensor.xml` (**no XML change**).
  - `universal_robots_ur3e/ur3e_position.xml` — UR3e arm with Hand-E **inlined** at the wrist_3 attachment (`pos="0 0.09215 0"`, vs UR10's 0.1): mount body, two coupled slide-joint fingers via `<tendon><fixed name="split">` (coef 0.5 each), one `position` actuator `ctrlrange="0 0.05"` → `nu=7`, `nq=8` standalone. **meshdir footgun:** set to `../universal_robots_ur3e/assets` so it resolves from the top-level loader (`xmls/`), not the included file's dir. Hand-E meshes (`hande.stl`, `coupler.stl`, `finger_*.stl`) live in `universal_robots_ur3e/assets/` — there is **no** separate `robotiq_hande/` dir (unlike `my_ur10/`).
  - `xmls/mjx_single_cube_position_ur3.xml` — pick scene: includes the robot + scene, **4×4×4 cm box** freejoint (geom half-extents `0.02`, resting center z=0.02), `mocap_target` body (geom `mocap_target_geom`, **5 cm** half-extent `0.025` — the box must be positioned inside it), `ur3_pick_sensor.xml`. Loads to `nu=7, nq=15, nkey=5, nsensor=8`. Keyframes `task_home`/`low_home`/`tucked` (15 qpos = 6 arm + 2 finger + 7 box; 7 ctrl); box pose `(0.3, 0, 0.02)` identity orientation.
  - `xmls/ur3_pick_sensor.xml` — 3 floor-contact + 3 finger/box-contact + 2 framepos sensors.
- `manipulation/` also contains upstream Panda environments.
- Config defaults per domain: `mujoco_playground/config/{manipulation,locomotion,dm_control_suite}_params.py`

### Training (`learning/`)
- `train_jax_ppo.py` — Main PPO training script using Brax. Supports `--env_name`, `--num_timesteps`, wandb logging.
- `robots/UR10e/UR10_ppo.py` — **env-agnostic** training entry point: `run_experiment(cfg)` dispatches on `cfg["env_name"]` via `registry.load()` and logs to W&B. Reused as-is for UR3 (no `UR3_ppo.py` needed). (Moved out of `learning/notebooks/` into `robots/UR10e/`.)

### Cluster Training (`batch_runs/`)
ZHAW SLURM cluster, rootless Podman container (`mujoco_env_EGL.tar`, tag `mujoco:env_EGL`) with `MUJOCO_GL=egl`. A sweep is a JSONL file (one config per line, **1-indexed** by `$SLURM_ARRAY_TASK_ID`); each array task trains one config and uploads a `policy_parameters_*` artifact to W&B.
- `slurm/run_array_ur{10,3}.sbatch` — SLURM array job. Loads the image tar under a `flock`, mounts the repo at `/workspace`, runs `scripts/run_one_ur{10,3}.py`. `REPO_DIR="$HOME/VT1_MBRL/my_mujoco_playground"`; W&B key read from `$HOME/.secrets/wandb_key`.
- `scripts/run_one_ur{10,3}.py` — thin per-task runner: parses one JSONL line (`--jsonl … --index … --out_root …`) and calls `run_experiment` imported as `from robots.UR10e.UR10_ppo import run_experiment` (namespace package; repo root is on `PYTHONPATH=/workspace` in the sbatch). The UR3 copy differs only in defaults; the import target is identical (env-agnostic).
- `sweeps/UR3Pick_sweep.jsonl` — **2 lines** (`PicknDrop_un20_env2048`, `PicknDrop_un20_env1024`): a small `num_envs` comparison, not a hyperparameter sweep — fixed `learning_rate=8e-4`, `unroll_length=20`, `num_timesteps=20M`, seed 0, `num_evals=15`, `init_keyframe="tucked"` (overrides the env's `low_home` default); varies `num_envs` **2048/1024**. All `env_name="UR3Pick"`, `wandb_project="UR3_PicknDrop"` (case-sensitive), `camera_kwargs={"camera":"box_detail"}`, tags `cube4cm`/`place`/`rand_orient`/`un20`/`env<N>`/`20M` (footgun: line 2's tag reads `env4096`, a stale copy-paste — `run_id`/`num_envs` are the source of truth). (Older pick policies still live in the separate `UR3_pick_ppo` project; the deployment `download_policy_from_wandb` default still points there.)

### Trained Policies (`evaluation/downloaded_policies/`)
- `simple_reach_policy_50hz/` — Production policy. Contains `params.msgpack` + `metadata.json`.
- `simple_reach_policy_200hz/` — Higher frequency variant.
- Each policy dir has `metadata.json` with env_name, obs/action dims, network architecture, and training config.

### Real Robot Deployment (`robots/{URSim,UR10e,UR3e}/`)
The UR deployment/RTDE/training-entry scripts were moved out of `learning/notebooks/` into a
top-level `robots/` folder grouped by target: `URSim/` (RTDE backbone + bring-up), `UR10e/`,
`UR3e/`. Cross-folder imports use a `sys.path` shim (e.g. UR10e scripts add `../URSim`). Misc
scripts not named with the `URSim/UR10/UR3` prefix were collected under
`robots/random_sample_code/` (`robotiq_gripper.py`, `SHIVA_rtde_trajectory_motion.py`,
`run_multi_target.py`, `collection_2D_sequences.py.py`); `manipulation UR10.ipynb` stayed in
`learning/notebooks/`.
- `robots/URSim/URSim_RTDE_dependencies.py` — **Core dependency class** `URSimRTDESimpleReach`. Handles:
  - RTDE connection: `rtde_receive.RTDEReceiveInterface` (joint state feedback) **and** `rtde_control.RTDEControlInterface` (motion commands). Both ship in the `ur_rtde` PyPI package.
  - `send_movej` / `send_servoj` are thin wrappers around `rtde_control.moveJ` / `rtde_control.servoJ`. **Parameter-order footgun:** the wrapper signature keeps URScript-style `(q, a, v, ...)` on the outside, but `rtde_control.servoJ` is `(q, speed, acceleration, ...)` — the swap happens inside the wrapper. Do not flip it at call sites.
  - `disconnect()` calls `servoStop()` + `stopScript()` before tearing down the control interface; `run_policy_loop` wraps its main loop in `try / finally` with a `servoStop()` so a KeyboardInterrupt or early break doesn't leave the controller holding the last servoj target.
  - On the real UR10e, PolyScope must be in **Remote Control** mode or `RTDEControlInterface()` connects but `servoJ` silently fails. URSim doesn't require this.
  - Policy loading from saved Brax PPO artifacts
  - `policy_step_ctrl_update()` — computes servoj target: `q_measured + action_scale * action`, with actuator range clipping, optional alpha blending, and velocity clamping
  - `run_policy_loop()` — 50Hz control loop: RTDE receive → build obs → policy inference → servoj send
  - MuJoCo video rendering from logged trajectories
  - Diagnostic plots and run metadata export
- `robots/UR10e/UR10_RealRobot_Reach_ONE.py` — Single-target reach script (imports the URSim backbone via a `../URSim` shim). Config at top, outputs video + plots + JSON to `results/`.
- `robots/random_sample_code/run_multi_target.py` — Multi-target reach with 5mm convergence tolerance (imports the URSim backbone via a `../URSim` shim).
- `robots/URSim/URSim_RTDE_SimpleReach.ipynb` — Comprehensive testing notebook (frequency sweeps, robustness tests, policy comparison).
- `robots/UR3e/ur3_realrobot_dependencies.py` — `UR3RealRobotPick`, the UR3 pick analog of `URSim_RTDE_dependencies.py`. Key differences:
  - `build_obs_from_feedback(fb, box_pos, target_pos, tcp_pos, box_quat)` assembles the **26D** obs `[q(8), qd_arm(6), (box−tcp)(3), (target−box)(3), box_xmat[:6](6)]` to match `UR3Pick._get_obs`. `q(8)` = 6 arm joints (RTDE) + 2 finger positions from the internal `self._gripper_ctrl` tracker (no finger encoder on the robot). `box_xmat[:6]` is derived from `box_quat` (mocap quaternion) via the `quat_to_xmat_flat()` static helper, identity rows when `box_quat` is None.
  - `download_policy_from_wandb(run_id, out_dir, entity, project, force=False)` — static helper that fetches the run's `model` artifact (`params.msgpack`) and writes a `load_policy_fn`-ready `metadata.json` alongside it (`env_name` + `network_factory` from `run.config`; `obs_dim`/`action_dim` from `registry.load(env_name)`). No-ops if `out_dir` already has both files unless `force=True`. Default entity/project `weissma6-zhaw-school-of-engineering` / `UR3_pick_ppo`.
  - `policy_step_ctrl_update` returns `(arm_ctrl[6], gripper_norm)` — the 7th action drives `self._gripper_ctrl` clipped to `[0, 0.05]`, normalized to `[0, 1]`. **`servoJ` accepts only 6 joints**, so the gripper goes on a separate channel.
  - **Arm control path (`use_ext_urcap`):** PolyScope X (10.x) does **not** run the headless control script `ur_rtde` normally uploads. Construct with `use_ext_urcap=True` (+ `ur_cap_port`, default 50002) to use the **External Control URCapX** instead — `RTDEControlInterface(host, -1.0, FLAG_USE_EXT_UR_CAP, ur_cap_port)`; `connect()` then **blocks until an External Control program (Host IP = this PC, port = ur_cap_port) is PLAYING** on the pendant with the robot in Remote Control. Default (`use_ext_urcap=False`) keeps the script-upload path for PolyScope 5 / CB-series.
  - **Gripper (now wired, not a stub):** on PolyScope X the Hand-E is driven via the **Robotiq URCapX XML-RPC server** at `http://<host>:49999/` (NOT the legacy 63352 socket), this Hand-E being **slaveId 9**; units are **percent**. `connect_gripper(slave_id, speed, force, reset)` connects + activates (`activateIfRequired`, or `activate` if `reset`); `send_gripper(norm_cmd)` maps norm `[0,1]` → native percent via `_gripper_open_pct`/`_gripper_closed_pct` (defaults 0%=open, 100%=closed; **confirm + flip via the open/close test**) and calls `move()`; `open_gripper()`/`close_gripper()` use the direction-independent URCapX calls (correct even before the mapping is confirmed); `read_gripper_state()` returns raw native values (`pos_pct`, `obj_flag`, `fault`, `activated`, `connected`) plus a derived `sim_finger ∈ [0, 0.025]` for feeding real gripper feedback into the 26D obs. **Footgun:** do **not** command the XML-RPC server above **~10 Hz** — `run_policy_loop` rate-limits `gripper_fn` to ≤10 Hz (`gripper_min_dt=0.1`) and only re-sends on meaningful change, even though the arm loop runs at 50 Hz.
  - `run_policy_loop(drop_target, mocap_reader, …, gripper_fn, use_fk_tcp)` reads `mocap_reader.get_rigid_body_xyz()` **and** `get_rigid_body_quat()` every tick (orientation feeds the 26D obs internally; the user only places the box), each falling back to the last good value when a frame is missing. **Every mocap reading is mapped into the policy/base frame** by `mocap_pos_to_base()` / `mocap_quat_to_base()` (calibration `world→raw base` via `R0ᵀ(p−p0)` **then** the same X/Y negation `receive_feedback` applies to the TCP, so box and TCP share the MuJoCo/sim frame). The transform is **identity if no calibration is loaded**. `load_base_calibration()` auto-runs in `__init__` from `calibration/base_frame_calibration.json` (override path or pass a different file). **Footgun:** the calibration JSON maps mocap→**raw** base (its probes use the raw TCP), hence the extra X/Y negation here — flip `_XY_NEG` if box/TCP frames disagree at deploy.
- `robots/UR3e/ur3_realrobot_pickloop.py` — single-run UR3 pick entry (mirror of `UR10_RealRobot_Reach_ONE.py`; imports `ur3_realrobot_dependencies` from the same folder). Config at top: `ROBOT_IP`, `DROP_TARGET` (lift target), `Q_START` (UR3 `low_home`), `MOCAP_SERVER_IP`/`MOCAP_RIGID_BODY_ID`, `WANDB_RUN_ID`/`WANDB_ENTITY`/`WANDB_PROJECT` (auto-downloads the policy into `POLICY_PATH` if absent), `ENABLE_GRIPPER`, `USE_EXT_URCAP`. For a dry-run set `ROBOT_IP=127.0.0.1`, `ENABLE_GRIPPER=False`, `USE_EXT_URCAP=False`. **Gripper wiring:** `ENABLE_GRIPPER=True` builds a `robots/hande/HandE_dependency.HandEGripper` and passes `gripper_fn = lambda norm: gripper.command(norm * 0.025)` to `run_policy_loop` (the loop's `gripper_norm ∈ [0,1]` maps to per-finger meters `[0,0.025]`); `False` keeps the no-op so the arm-only dry-run is unchanged. This routes the gripper through the new wrapper, **not** the inline (inverted) `send_gripper` in `ur3_realrobot_dependencies.py`. **Go-gate:** `USE_EXT_URCAP=True` constructs `UR3RealRobotPick(use_ext_urcap=True, ur_cap_port=UR_CAP_PORT)` so `robot.connect()` **blocks until the External Control program is PLAYING** on the pendant — pressing **Play** is the robot-side "go" (the same mechanism exercised in `robots/UR3e/UR3_RTDE_Tests.ipynb`). `False` keeps the URSim script-upload path with no gate.
- `robots/random_sample_code/robotiq_gripper.py` — vendored SDU Robotics / `ur_rtde` `RobotiqGripper` socket client (stdlib `socket`, default port **63352**). This is the **PolyScope 5 / CB-series fallback** path (native position 0=open, 255=closed); the lab UR3e on **PolyScope X uses the XML-RPC `:49999` path** in `ur3_realrobot_dependencies.py` instead. Reference/fallback only — not imported by the pick loop.
- Bring-up sanity notebooks for the lab **UR3e on PolyScope X 10.12** (`192.168.1.4`), split by channel (the former combined `robots/URSim/UR_RTDE_Tests.ipynb` was taken apart into these two):
  - `robots/UR3e/UR3_RTDE_Tests.ipynb` — **arm** bring-up. Self-locates `ur3_realrobot_dependencies.py` (cwd-independent). (1) raw `rtde_receive` reachability + state (no pendant program needed), (2) live receive stream, (3) `robot.connect_arm()` (External Control URCap, blocks until **Play**) + `move_to_start()` to the `tucked` keyframe. Constructs `UR3RealRobotPick(host, ur_cap_port=50002)` (no `use_ext_urcap` — the arm is always ext-urcap).
  - `robots/hande/HandE_Tests.ipynb` — **gripper** bring-up, using `HandEGripper` directly (self-locating import). (1) `g.connect()` + `read_state()`, (2) `open_gripper()`/`close_gripper()` to lock the percent direction (flip `NATIVE_OPEN_PCT`/`NATIVE_CLOSED_PCT` if reversed), (3) `command(sim)` sweep `0→0.025`. Run both before a full pick loop.
- `robots/hande/` — standalone Hand-E gripper module (used by the UR3 pick loop and a mapping notebook):
  - `HandE_dependency.py` — `HandEGripper`, a reusable wrapper around the **Robotiq URCapX XML-RPC server** (PolyScope X, `http://<host>:49999/`, slaveId 9, native units **percent 0–100**) — **not** the 63352 socket. API: `connect(reset, speed, force)` / `disconnect()` / `is_connected()`; `command(sim_value, speed, force)` (move to a sim finger position); `open_gripper()`/`close_gripper()` (direction-independent URCapX calls); `read_state()` → `{pos_pct, obj_flag, grasped, sim_finger, fault, activated, connected}` (grasp = `obj_flag ∈ {1,2}`); and the **classmethod** mapping `sim_to_native()`/`native_to_sim()`. **Direction (verified from `ur3_pick.py`, opposite to the inline `send_gripper` docstrings):** sim per-finger meters `[0, 0.025]` with **`0 = OPEN`, `0.025 = CLOSED`**; native percent `[0, 100]` with default **`0 % = OPEN`, `100 % = CLOSED`** → same direction, **pure scale, no sign flip**. The percent direction is the one bit not provable from code — confirm/flip the `NATIVE_OPEN_PCT`/`NATIVE_CLOSED_PCT` constants via the notebook's open/close test (the only line that encodes direction).
  - `hande_mapping_playground.ipynb` — Phase-1 notebook: a no-hardware mapping cell (prints `0/50/100 %`, asserts endpoints/clamp/round-trip, plots the curve) and a `RUN_HARDWARE`-guarded section (connect → open/close test to lock the percent direction → sweep sim values, printing `read_state()`). The import cell self-locates `HandE_dependency.py` and adds its dir to `sys.path`, so it runs from any kernel cwd (VSCode runs notebooks from the workspace root).
- `robots/UR3e/calibration/` — UR3e **base-frame calibration**: recover the robot base frame {B} (origin `p0` + rotation `R0`) in the **mocap world frame** so mocap box poses can be mapped into the robot/policy frame. Method is **fully closed-form, no iterative regression**: sweep joint 1 and joint 2 each alone while a tracked rigid body (`calibrig4`) traces a circle → fit each circle (SVD plane + Kasa) → intersect the two joint axes at the shoulder → `p0 = shoulder − d1·z0` (`d1 = UR3E_D1 = 0.15185 m`, nominal UR3e shoulder height); two orientation-fixed base-frame probe moves (+X, +Y) give the base axes directly → Gram-Schmidt → `R0`. Mapping convention: `p_W = p0 + R0 @ p_B` (R0 columns are the base axes in {W}).
  - `base_calibration_dependencies.py` — **pure-numpy** solver (no RTDE/mocap/MuJoCo imports): `fit_circle_3d`, `circumcenter_3pt` (3-point cross-check), `closest_point_two_lines` (axis intersection; `gap_h` mm is the fit-quality metric), `frame_from_probe`, `rotvec_to_matrix` (RTDE rotvec → R), `forward_tcp` (see check 3 below), `average_rotations` (SVD/SO(3) mean) + `rotation_geodesic_deg` (used by the repeatability harness), `solve(...)` → `(p0, R0, residuals)`, and `build_output_dict`/`rotation_matrix_to_quat_wxyz` → the `base_frame_calibration.json` payload. Constants: `UR3E_D1 = 0.15185` (shoulder height) and `UR3E_D6 = 0.0921` (wrist-center→flange along tool Z; sourced from UR's published DH parameters, cited in-comment). **All inputs/outputs in METERS.**
  - `base_calibration.py` — live calibration, refactored into a callable `run_calibration(sweep_v/move_a/start_v/probe_v/probe_a, robot=, reader=, write_json=, append_hist=, plot=, verbose=, run_index=) -> result dict` (pass an already-connected `robot`/`reader` to reuse one connection across runs — only self-created connections are torn down; flags decouple compute from file I/O); `main()` is a thin wrapper that runs it with defaults. Connects via External Control URCapX (`use_ext_urcap=True`, blocks until **Play** on the pendant), runs both sweeps + both probes against `VRPNRigidBodyReader`, feeds points to `cal.solve`, writes `robots/UR3e/calibration/base_frame_calibration.json` (consumed by `ur3_realrobot_dependencies.load_base_calibration`) and `base_frame_calibration.png`. **Probes use the RAW `getActualTCPPose()` base frame, not the negated obs frame.** Sweeps are absolute-target (base −1.5→+1.5, then shoulder 0→−3.0 with base held at +1.5), accel reduced after a robot warning. Path shim adds `robots/UR3e`, `robots/UR3e/calibration`, and `motion_capture/mymocap`. Parks at the `task_home` keyframe. Runs **three** live checks (all leave the arm at `Q_TASK_HOME`):
    1. **TCP cross-check (×2)** — map the RTDE TCP into mocap world via the fresh `p0`/`R0`, compare to the tracked body; the residual is the fixed `calibrig4`→TCP mount offset (not zero) and should be small + repeatable. **Not independent of the calibration** (uses `p0`/`R0`).
    2. **Wrist-3 orientation check** (`wrist_orientation_check`) — sweep the last joint (wrist 3, `WRIST_JOINT=5`), fit the circle; its axis IS the tool-flange Z. Compare the world-measured flange-Z to `R0 @ (flange-Z read from the RTDE flange orientation)` → an orientation error (deg) the probes never saw. Warns past `ORIENT_WARN_DEG=3°`. Skips if the arc radius < `MIN_WRIST_RADIUS_MM`.
    3. **Forward-TCP sanity check** (`forward_tcp_check` + `cal.forward_tcp`) — additionally sweep wrist 2 (`WRIST2_JOINT=4`), fit its axis, intersect with the wrist-3 axis → **wrist center** (DH `a5=0` ⇒ axes meet at frame-5 origin), then push `L_NOMINAL = UR3E_D6 + TCP_Z_OFFSET` along the measured tool Z → `tcp_forward`. This carries **NO `p0`/`R0`** (pure mocap circle fits), so `|tcp_world − tcp_forward|` is a true **outside-check** on the base calibration. Reports three distances (mm): RTDE-TCP↔body, forward-TCP↔body, RTDE↔forward (the sanity residual; a residual *along tool Z* ⇒ wrong `L_NOMINAL`/`p0` height, a *rotational* residual ⇒ `R0` error). **Footgun:** `TCP_Z_OFFSET` MUST match the active PolyScope pendant TCP (flange TCP here → 0.0, so `L_NOMINAL = 0.0921`), else the residual is biased along tool Z. `WRIST2_SWEEP_TO` is a proposed default — verify collision-free before trusting it. Results land in JSON as `wrist_validation` / `tcp_forward_check` blocks, and every computed position is appended per run (via `append_run_record`, expanded 22-column schema) to **`calibration_accuracy/calibration_history.csv`** (was the 8-column `tcp_forward_history.csv`). The PNG (`plot_calibration`) shows the base+shoulder sweeps (points + axes + intersection + `p0`) and, for the wrist sweeps, ONLY their two fitted axes + the wrist-center intersection.
  - `calibration_accuracy/` — repeatability tooling + its generated reports (the **only** accuracy-meta files; the core `base_frame_calibration.json` and `.png` stay in `calibration/`).
    - `test_accuracy.py` — runs `run_calibration()` **N times** (`--runs`, default 10) over **one** robot+mocap connection (fail-soft: a bad run is logged + skipped, never aborts the batch). Aggregates `p0`/`tcp_world`/`tcp_forward` (per-axis mean/std/min/max + 3D RMS spread), the TCP↔TCP distance, and rotation dispersion; writes `accuracy_report_<ts>.csv` + `accuracy_summary_<ts>.md` + `accuracy_histograms_<ts>.png` (via `plot_accuracy`) here; and overwrites the parent `base_frame_calibration.json` with the **mean** calibration — `p0` averaged linearly, `R0` via `cal.average_rotations` (SVD on SO(3); rotations don't average element-wise) — plus an `averaging` block (N, per-axis std, spread, rotation dispersion). Existing JSON keys preserved → deployment loader unaffected. Run from this folder; ~15–30 min of live motion for N=10 (smoke `--runs 2` first).
    - `calibration_history.csv` — append-only audit log, one row per calibration run (single-run `main()` **and** every batch run).
    - `plot_accuracy.py` — **hardware-free** histogram plotter (matplotlib + numpy + stdlib `csv`; never calls `plt.show()` → headless-safe). Reads any report CSV (or `calibration_history.csv` — same schema) → a 2×3 figure: per-axis X/Y/Z TCP histograms with `tcp_world` (RTDE) vs `tcp_forward` overlaid on **shared bins** (shows spread + the world↔forward offset, e.g. ~1.6 mm on Z), plus `|tcp_world − tcp_forward|` and `|tcp_world − mocap|` (≈47 mm mount offset) histograms and a stats panel. `python plot_accuracy.py [--csv … --out … --show]` (default: latest `accuracy_report_*.csv`). NaN-safe (forward-skipped runs drop out, noted on the panel). N≈10 → coarse bins; point it at `calibration_history.csv` to pool all runs.
  - `base_calibration_interactive.ipynb` — step-by-step / dry-run version of the same flow (was `base_calibration.ipynb`). **Note:** still defines its own pre-refactor `sweep_joint`/`probe_axis`/`wrist_orientation_check` copies (motion speeds read from notebook globals), not the parameterized `base_calibration.py` versions.
  - `_phase1_synthetic_test.py` — offline checks (no hardware): `main()` builds two circles sharing a known shoulder, adds noise, asserts `solve()` recovers `p0` < 1 mm and the circumcenter agrees with the SVD+Kasa fit; `test_forward_tcp()` builds two perpendicular axes intersecting at a known wrist center with a FLIPPED tool-Z sign and asserts `forward_tcp` recovers the wrist center + TCP < 1 mm (verifying the `ref_dir` sign correction); `test_average_rotations()` builds N noisy rotations about a known `R` and asserts `cal.average_rotations` recovers it (orthonormal, det +1, < 0.5°). Stays in `calibration/` (it tests the solver, not accuracy reports). Run `cd robots/UR3e/calibration && python _phase1_synthetic_test.py` to validate the solver/helpers after any edit.

### Motion Capture (`motion_capture/`)
- Nokov (XINGYING) mocap streaming for closed-loop "real target from mocap, not hardcoded" reaching. The SDK is now **pip-installed** from `motion_capture/nokov_python_sdk-master/dist/nokovpy-3.0.1-py3-none-any.whl` (`uv pip install …whl` — gives `from nokov.nokovsdk import PySDKClient` with no sys.path hack). The wheel bundles `libnokov_sdk.so` (x86_64 + aarch64) plus the Windows DLL. The older extracted copy at `motion_capture/XING_Python_SDK-4.1.0.5873/dist/nokovpy-3.0.1-py3.9/` is still on disk and is what `motion_capture_reader.py` / `mocap_publisher.py` reach into via `sys.path`. **Note:** the active deployment path is now **VRPN** (see the VRPN pipeline subsection below), not the SDK; `test_mocap_connection.py` was rewritten onto VRPN.
- Active Linux integration on `linux_motion_capture` branch. Layout:
  - `motion_capture/mymocap/test_mocap_connection.py` — **VRPN scene-inventory streamer** (rewritten off the Nokov SDK onto the VRPN pipeline). Uses `vrpn_dependencies.discover_senders()` to list every tracker, then subscribes to ALL of them via `VRPNRigidBodyReader(names=None)` and prints each reporting body's xyz + quat(wxyz) per tick. CLI: `--server-ip` (default `10.1.1.198`), `--port` (default 3883), `--duration` (default 10 s), `--list` (names then exit). Run this first to verify the VRPN pipeline end-to-end before wiring it into the robot loop.
  - `motion_capture/mymocap/motion_capture_reader.py` — `XINYINGReader` class. Background thread connects to the Nokov server, receives marker frames (~60Hz), stores latest poses in a thread-safe dict. Public API: `get_first_marker_position()`, `get_marker_centroid([...])`, `get_markers_in_region(center, radius)`, `is_connected()`. **Markers-only** (no rigid-body API yet).
  - `motion_capture/mymocap/mocap_publisher.py` / `mocap_receiver.py` — split-process variant (publisher on the machine that can talk to the mocap server, receiver in the control loop) for when the SDK can't run in-process alongside RTDE. **Markers-only.**
  - `motion_capture/nokov_python_sdk-master/` — unzipped upstream SDK from the advisor (`nokov_python_sdk-master.zip`). `dist/` ships the wheel that is now pip-installed; `examples/` (including `Nokov_SDK_Client.py` and a `Utility.py` with `Point` / `SlideFrameArray` / `CalculateVelocity` velocity helpers) is the read-only reference. Not on `sys.path` — examples are reference, not import targets.
  - `motion_capture/INTEGRATION_PLAN.md` — design doc for the planned `URSimRTDESimpleReach.initialize_mocap()` / `get_mocap_target()` hooks and the 50Hz-loop ↔ 60Hz-mocap frequency-independent read pattern (background thread writes latest pose; control loop reads whenever it needs).
  - `motion_capture/mymocap/mocap_dependencies.py` — `NokovRigidBodyReader` (the Nokov-SDK rigid-body reader). Unlike the markers-only readers above, this reads a **rigid body** from `frame.RigidBodies[i]` (matching `body.ID == rigid_body_id`, or the first body if `None`) and exposes only its XYZ center via `get_rigid_body_xyz() -> np.ndarray(3,) | None` (mm→m). Background daemon thread + `Lock`. Prefers the pip-installed wheel, falls back to the vendored SDK path. Does **not** call `Uninitialize()`. (The UR3 pick loop now prefers the VRPN reader below; this is the SDK-path analog with the same public API.)

#### VRPN pipeline (preferred for the pick loop)
The Nokov server **also** broadcasts every rigid body and marker as a **VRPN tracker** on port **3883** (no auth, server IP only — same as the SDK). This path avoids running the Nokov SDK in-process alongside RTDE.
  - `motion_capture/mymocap/vrpn_dependencies.py` — `VRPNRigidBodyReader`, the VRPN analog of `NokovRigidBodyReader` with the **same public API** (`get_rigid_body_xyz/quat/pose`, `is_connected`, background thread), so it is a **drop-in** for the pick loop. Two modes: `names=[...]` (specific trackers) or `names=None` (auto-discover + subscribe to **all**). **Data conventions:** VRPN positions are already in **meters** (no mm→m); VRPN quaternions arrive `(x,y,z,w)` and are stored **`(w,x,y,z)`** to match the SDK reader + 26D obs builder. Extra API beyond the SDK reader: `get_body(name=None)` (per-tracker dict / one body / all), and a **body-frame offset** — `set_offset()`/`get_offset()` plus `get_offset_point()`/`get_offset_pose()` return `center + R(quat)·offset` (offset rotates with the body; default zeros, settable at runtime). `discover_senders(ip, port)` is a module function (shells out to the `discover_senders` binary; the internal `"VRPN Control"` sender is filtered out). **Import resolution:** prefers an installed `vrpn` module, else adds the vendored `nokov_python_vrpn-master/dist` to `sys.path`. **Binary resolution:** `discover_senders` is found via `shutil.which` (PATH) first, else the vendored `nokov_python_vrpn-master/bin/`. Run `python motion_capture/mymocap/vrpn_dependencies.py --list` / `--name <body>` / `--offset x,y,z` as a demo. **Footgun:** a tracker the server knows but isn't currently streaming (body not visible to cameras) yields no reports — pose stays `None` and VRPN logs "No response from server"; that is normal.
  - `motion_capture/nokov_python_vrpn-master/` — vendored VRPN client built from upstream VRPN suite 07.38 against this venv's Python 3.10 (see its `README.md` for rebuild provenance). Ships `dist/vrpn.so` (hand-coded VRPN 3.x binding — **not** the SWIG one), `bin/discover_senders` (+ `src/discover_senders.cpp`), and `examples/vrpn_tracker_client.py` (raw-module reference, no project wrappers). **Now installed into the venv** (`vrpn.so`→site-packages, `discover_senders`→`.venv/bin/`), so this folder is **not required at runtime** — the fallbacks above exist only for fresh checkouts / the cluster.
  - `motion_capture/mymocap/vrpn_mocap_streaming_exploreation.ipynb` — step-by-step VRPN exploration notebook (discover senders → raw dump → sensor/skeleton probe → velocity/accel → stream all → stream one body with offset → **realtime 3D view**). The 3D view subscribes to `RIGID_BODY_NAME` (**`CubInCube2`** — note the server spells it with no "e") and redraws at ~10 Hz a 4 cm cube + body-frame XYZ triad (rotated by the live quaternion) around the streamed centerpoint, via `%matplotlib inline` + `clear_output`/`display` (no ipympl backend needed).
- **Credentials**: none. The SDK authenticates with only the server IP via `client.Initialize(bytes(ip, "utf8"))`. There are no secrets in the code.
- **Server-side prerequisites** (on the Windows mocap PC, not in this repo): XINGYING must have **"SDK Enabled"** toggled ON, and at least one **rigid body / markerset / skeleton** must be defined and visible to the cameras. Without "SDK Enabled" the client connects but zero frames arrive. With "SDK Enabled" but an empty scene, frames arrive but every frame has zero entities. (The same toggle gates the VRPN broadcast: no toggle → client connects but no trackers; empty scene → only the internal `"VRPN Control"` sender.)
- **`PySDKClient` API footgun**: there is **no** `Uninitialize()` method (verified via `dir(PySDKClient())`). The advisor's examples don't clean up at all — just let the script exit. `motion_capture_reader.py:137` and `mocap_publisher.py:131` still call `self.client.Uninitialize()` and will `AttributeError` on shutdown if exercised. (Moot for the VRPN path — `VRPNRigidBodyReader.stop()` just joins the thread.)
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
- UR10 reach (`192.168.1.2`): port 30004 (RTDE) is the only socket the code touches — URScript (30002) and Dashboard (29999) are unused since the `rtde_control` rewrite.
- UR3e pick on **PolyScope X** (`192.168.1.4`): RTDE 30004 (receive) + **External Control URCapX on 50002** (arm, since the script-upload path doesn't run on PolyScope X) + **Robotiq URCapX XML-RPC on 49999** (Hand-E gripper).
- Requires `ur_rtde` package, providing both `rtde_receive` and `rtde_control`. If only one imports, the install is partial — reinstall `ur_rtde`.

## Important Conventions

- UR10 reach policy observations must match the 18D layout: `[q(6), qd(6), tcp_pos(3), target_pos(3)]`. UR3 pick is **26D**: `[q(8 arm+finger), qd(6 arm), (box−tcp)(3), (target_pos−box)(3), box_xmat.ravel()[:6](6)]` (relative position vectors + first two rows of the box rotation matrix; no raw world positions) with a **7D** action (the 7th element is the Hand-E tendon command). Obs/action dims and ordering must match between sim training and deployment.
- RTDE TCP pose has X/Y axes negated to match MuJoCo base frame orientation
- For UR3 pick, the gripper is **not** part of the `servoJ` `q`-vector (servoJ is 6-joint only) — it travels on a separate channel, wired on PolyScope X to the Robotiq URCapX XML-RPC server (`:49999`, slaveId 9), rate-limited to ≤10 Hz. The pick loop drives it through `robots/hande/HandE_dependency.HandEGripper` (correct direction); the inline `send_gripper()` in `ur3_realrobot_dependencies.py` has an **inverted** norm→percent mapping (a latent bug, kept for the bring-up notebook). The `robotiq_gripper.py` socket client (port 63352) is the PolyScope 5 / CB-series fallback.
- **Gripper direction (verified, not the docstrings):** in sim, the Hand-E per-finger value is `[0, 0.025]` m with **`0 = OPEN`, `0.025 = CLOSED`** (`ur3_pick.py`: `finger_open = tanh(finger_touch_dist/0.05)`; fingers close as qpos rises; `low_home`/`tucked` start open). `run_policy_loop` passes `gripper_norm ∈ [0,1]` (`0 = open`, `1 = closed`, `norm == finger_meters/0.025`). Native URCapX percent defaults to `0 % = open`, `100 % = closed`. Several docstrings in `ur3_realrobot_dependencies.py` claim "0 = closed" — they are wrong.
- The `action_scale` in deployment must match training (0.04 for 50Hz policy)
- servoj parameters affect smoothness: `gain` (stiffness 100-2000), `lookahead_time` (smoothing 0.03-0.2s). `rtde_control.servoJ` **hard-enforces** the `lookahead_time ∈ [0.03, 0.2]` range — a value outside it raises `ValueError("The value is not within [0.03;0.2]")` on the first servoj call inside the policy loop.
- MuJoCo model XML: UR10 reach `mujoco_playground/_src/manipulation/my_ur10/xmls/mjx_reach.xml`; UR3 pick `mujoco_playground/_src/manipulation/my_ur3/xmls/mjx_single_cube_position_ur3.xml`
- Code style: pyink with 2-space indentation, 80-char line length, majority quotes. Imports sorted by isort (single-line, 120-char). Ruff config is in `pyproject.toml` (no standalone `ruff.toml`). `__init__.py` files are excluded from pre-commit formatting.
- `mujoco_menagerie/` and `external_deps/` are excluded from all linting/formatting tools.
