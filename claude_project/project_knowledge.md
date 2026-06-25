# Project Knowledge — Pick n' Place Robot MuJoCo

A distilled summary of what was built in this project across ~9 prior
Claude Code working sessions on my laptop, plus the current state of the
codebase. Use this as background context when I ask questions in chat.

## Project in one paragraph

Fork of Google DeepMind's [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
that adds **custom Universal Robots arms** for reach and pick tasks. PPO
training in JAX/Brax on MuJoCo MJX, deployed on real robots via `ur_rtde`.
Three robots / tasks live in the repo:

- **UR10e — `UR10SimpleReach`**: 6-DoF arm-only reaching, no gripper.
  *Working end-to-end on the real UR10e.*
- **UR10e — `UR10PickCube`**: pick task with Robotiq Hand-E gripper.
- **UR3e + Hand-E — `UR3Pick`** (current focus, `ur3_pick` branch): 7-DoF
  pick task whose box target is fed from Nokov motion capture at deploy
  time. *Training on ZHAW SLURM cluster; real-robot loop scaffolded but
  Hand-E I/O still stubbed.*

## Branch map

| Branch | Purpose | Status |
|---|---|---|
| `main` | UR10 reach baseline | stable |
| `simple_reach` | UR10 reach development | merged into main |
| `linux_motion_capture` | Linux-ready RTDE + Nokov mocap | stable |
| `ur3_pick` (current) | UR3 + Hand-E pick task | active dev |

**Important policy:** UR3 is a *structural sibling* of UR10 — every UR3
artifact (XMLs, env classes, sbatch, deployment scripts) is a separate
copy, not a shared abstraction. UR10 / `simple_reach_*` files must stay
intact.

---

## What was built, chronologically

### Phase 1 — UR10 reach in sim
- Built `UR10SimpleReach` env in `mujoco_playground/_src/manipulation/my_ur10/`
  (18D obs, 6D delta-joint action, `ctrl_dt=0.02`, `action_scale=0.04`).
- Trained PPO policies at 50 Hz / 200 Hz / 540 Hz via Brax.
- Multiple starting positions and TCP targets validated in notebook
  rollouts.
- Saved policies to `evaluation/downloaded_policies/simple_reach_policy_*/`.

### Phase 2 — UR10 reach on the real robot (URSim → physical UR10e)
- Started with raw URScript over TCP sockets (macOS limitation —
  `RTDEControlInterface` wouldn't build cleanly there).
- Rewrote everything around `ur_rtde`'s `RTDEReceiveInterface` +
  `RTDEControlInterface`. Receive gives joint state at the loop rate;
  control runs `servoJ` at 50 Hz.
- **Fixes that mattered:**
  - X/Y axis flip between MuJoCo base frame and RTDE TCP pose.
  - Joint-velocity clamping inside `policy_step_ctrl_update` to kill jitter
    (`max_joint_speed * dt` per joint).
  - `servoJ` lookahead parameter is hard-clamped to `[0.03, 0.2]` — outside
    that range it raises on the first call inside the policy loop.
  - PolyScope must be in **Remote Control** mode for the real UR10 or
    `servoJ` silently does nothing (URSim doesn't need this).
  - `disconnect()` must call `servoStop()` + `stopScript()` before tearing
    down, and `run_policy_loop` wraps in `try/finally` with a `servoStop()`
    so a Ctrl-C doesn't leave the controller chasing the last target.
- Output: `robots/UR10e/UR10_RealRobot_Reach_ONE.py`,
  `learning/notebooks/run_multi_target.py`, plus a comprehensive testing notebook.

### Phase 3 — Linux-ready stack
- Moved off macOS sockets. `ur_rtde` builds cleanly on Linux.
- Set `MUJOCO_GL=egl` for headless rendering. `osmesa` is *not* installed
  — avoid `MUJOCO_GL=osmesa` unless `libosmesa6` is `apt install`-ed.
- `imageio-ffmpeg` is **not** pulled by `.[all]` — install once per venv,
  otherwise `save_video()` crashes on `.mp4` backend lookup.

### Phase 4 — Nokov motion capture (XINGYING SDK)
- Got `nokov_python_sdk-master.zip` from advisor; built the wheel into the
  venv (`uv pip install …whl`). No more `sys.path` hacks for new code; only
  `motion_capture_reader.py` / `mocap_publisher.py` still reach into the
  unzipped vendored copy.
- Built four readers:
  - `XINYINGReader` — markers only, background thread, ~60 Hz.
  - `mocap_publisher` / `mocap_receiver` — split-process variant for when
    SDK can't co-exist with RTDE in the same Python.
  - `NokovRigidBodyReader` — rigid-body XYZ centre, the one used by the
    UR3 pick loop.
- `test_mocap_connection.py` enumerates skeletons / markersets / rigid
  bodies on connect, then dumps per-frame inventory — run this first
  before touching the robot loop.
- **SDK quirks documented (real footguns):**
  - `PySDKClient` has **no** `Uninitialize()` method. Calling it
    `AttributeError`s on shutdown. Two old files still call it.
  - Server-side: XINGYING must have **"SDK Enabled"** toggled ON,
    *and* at least one rigid body / markerset defined, or you get
    connected-but-empty 60 Hz frames.
  - Network: USB-Ethernet to mocap subnet `10.1.1.0/24`, server at
    `10.1.1.198`. Lab wifi is offline — all deps must already be vendored.

### Phase 5 — UR3 + Hand-E pick task (current work)
- Created `manipulation/my_ur3/` as a **copy** of `my_ur10/` (per the
  sibling-not-refactor policy).
- Built `ur3e_position.xml` with Hand-E **inlined** at wrist_3
  (`pos="0 0.09215 0"`, vs UR10's 0.1):
  - Two slide-joint fingers, tendon-coupled (`<fixed name="split">`,
    coef 0.5 each).
  - One `position` actuator, `ctrlrange="0 0.05"` → `nu=7`, `nq=8`.
  - **meshdir footgun:** must be `../universal_robots_ur3e/assets` so it
    resolves from the top-level loader in `xmls/`, not the included file's
    dir.
  - Hand-E meshes (`hande.stl`, `coupler.stl`, `finger_*.stl`) live in
    `universal_robots_ur3e/assets/` — there is **no** separate
    `robotiq_hande/` dir (unlike `my_ur10/`).
- `UR3Pick` env (`ur3_pick.py`):
  - **21D obs** `[q(6), qd(6), tcp(3), box(3), drop_target(3)]`.
  - **7D action** (6 arm delta + 1 Hand-E tendon).
  - `ctrl_dt=0.02`, `sim_dt=0.005`, `action_scale=0.04`,
    `episode_length=150`, `init_keyframe="low_home"`.
  - Reward scales `{box_target:8.0, reach_box:4.0,
    no_floor_collision:0.25, robot_target_qpos:0.3}` (matched to
    `ur10pick` reward shape in commit `8481513`).
  - Success: `||box - drop_target|| < 0.03` for 3 consecutive steps.
- Pick scene `xmls/mjx_single_cube_position_ur3.xml`: robot + scene +
  box freejoint + `mocap_target` body + sensors. Loads to
  `nu=7, nq=15, nkey=5, nsensor=8`.
- Sensors: 3 floor-contact + 3 finger/box-contact + 2 framepos.
- Keyframes: `task_home` / `low_home` / `tucked`.

### Phase 6 — Cluster training pipeline for UR3
- ZHAW SLURM cluster, rootless Podman with image `mujoco:env_EGL` built
  from `mujoco_env_EGL.tar`. Loaded under a `flock` so concurrent array
  tasks don't race.
- `batch_runs/slurm/run_array_ur3_pick.sbatch` — array job, mounts repo at
  `/workspace`, calls `scripts/run_one_ur3.py`. W&B key read from
  `~/.secrets/wandb_key`.
- `batch_runs/sweeps/UR3Pick_sweep.jsonl` — **1-indexed by SLURM array
  task ID**:
  1. `ur3pick_smoke` (300k steps, 512 envs)
  2. `ur3pick_baseline` (20M steps, 2048 envs)
  3. `ur3pick_DR_MFR` (domain randomization)
- All write `policy_parameters_*` artifacts to W&B project
  `UR3_pick_ppo`.
- Pinned `jax[cuda]==0.6.2` (commit `e89fc24`) for cluster GPU compat.

### Phase 7 — UR3 real-robot scaffolding
- `robots/UR3e/ur3_realrobot_dependencies.py` — `UR3RealRobotPick`
  class, mirror of `URSim_RTDE_dependencies.py`. Differences:
  - `build_obs_from_feedback(fb, box_pos, drop_target, tcp_pos)` builds
    the 21D obs; `box_pos` comes from `NokovRigidBodyReader` each tick
    (last-good fallback when a frame is missing).
  - `policy_step_ctrl_update` returns `(arm_ctrl[6], gripper_norm)`. The
    7th action drives `self._gripper_ctrl` clipped to `[0, 0.05]`,
    normalised to `[0, 1]`. **`servoJ` only takes 6 joints**, so the
    gripper goes on a separate channel via `send_gripper(norm_cmd)`.
  - `send_gripper()` is a **STUB** — `raise NotImplementedError`. Must be
    wired to the lab's Hand-E setup (Robotiq URCap RTDE register vs. tool
    I/O) before a real run. **Don't fake an implementation.**
- `ur3_realrobot_pickloop.py` — single-run entry, mirror of
  `UR10_RealRobot_Reach_ONE.py`. Config at top:
  `ROBOT_IP`, `DROP_TARGET`, `Q_START` (UR3 `low_home`),
  `MOCAP_SERVER_IP`, `MOCAP_RIGID_BODY_ID`, `ENABLE_GRIPPER`
  (False = no-op for URSim dry-runs).
- For URSim dry-runs: `ROBOT_IP=127.0.0.1`, `ENABLE_GRIPPER=False`.

---

## Key files (current state on `ur3_pick`)

```
mujoco_playground/_src/manipulation/
├── my_ur10/              # UR10 reach + pick (do not touch unless asked)
│   ├── ur10_base.py
│   ├── ur10_reach.py     # UR10SimpleReach env
│   ├── ur10pick.py
│   └── xmls/mjx_reach.xml
└── my_ur3/               # UR3 pick (active)
    ├── ur3_base.py
    ├── ur3_pick.py       # UR3Pick env (21D obs, 7D action)
    ├── universal_robots_ur3e/
    │   ├── ur3e_position.xml      # UR3e + Hand-E inlined
    │   └── assets/                # Hand-E meshes live here
    └── xmls/
        ├── mjx_single_cube_position_ur3.xml
        └── ur3_pick_sensor.xml

learning/
├── train_jax_ppo.py
└── notebooks/
    ├── UR10_ppo.py                     # env-agnostic, reused for UR3
    ├── UR10_RealRobot_Reach_ONE.py
    ├── URSim_RTDE_dependencies.py
    ├── ur3_realrobot_dependencies.py   # UR3RealRobotPick
    └── ur3_realrobot_pickloop.py

batch_runs/
├── slurm/run_array_ur3_pick.sbatch
├── scripts/run_one_ur3.py
└── sweeps/UR3Pick_sweep.jsonl          # 3 configs

motion_capture/
├── mocap_dependencies.py               # NokovRigidBodyReader (used by UR3 loop)
├── mymocap/
│   ├── test_mocap_connection.py
│   ├── motion_capture_reader.py
│   ├── mocap_publisher.py
│   └── mocap_receiver.py
└── nokov_python_sdk-master/            # vendored SDK wheel + examples
```

## Robot connection cheat-sheet

- Real UR10e: `192.168.1.2` (lab USB→Ethernet adapter, host `192.168.1.89/24`)
- URSim: `127.0.0.1`
- Mocap server (Nokov): `10.1.1.198` on subnet `10.1.1.0/24`
- Only port `30004` (RTDE) is used. URScript (30002) and Dashboard (29999)
  are no longer touched since the `rtde_control` rewrite.

## Where things still hurt

- **`send_gripper()` is stubbed.** Real Hand-E I/O on the lab UR3 has to be
  decided (Robotiq URCap RTDE register vs. tool digital out) before any
  real-robot pick run can complete.
- **Two old mocap files still call the non-existent
  `PySDKClient.Uninitialize()`** — `motion_capture_reader.py:137` and
  `mocap_publisher.py:131`. Will `AttributeError` on shutdown if exercised.
- **UR3 baseline training is just starting on the cluster.** Latest commit
  (`8481513`) reshaped the reward to match `ur10pick`. Whether that learns
  is the next open question.
