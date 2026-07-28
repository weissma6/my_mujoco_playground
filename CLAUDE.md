# CLAUDE.md

Fork of MuJoCo Playground: custom UR arms (UR10 reach/pick, UR3e+Hand-E pick) + RTDE deploy + ZHAW SLURM training. Architecture is in the code — below are only the non-obvious, behavior-changing rules. Prefer minimum, surgical edits.

## Workflow
- **Never train or smoke-test locally — it breaks this machine. Use SLURM** (`sbatch batch_runs/slurm/*.sbatch` or uv `batch_runs/cluster/*.submit`).
- UR3 is a **structural sibling** of UR10 — each UR3 file is a separate copy, not a shared abstraction. Don't refactor them together; keep UR10/reach files intact.
- Bring-up notebooks (`robots/hande/`, `robots/UR3e/*_Tests.ipynb`): dead-simple, one concern per cell, no clever path-finding — a silently-hanging cell is worse than a loud failure.

## Sweeps (UR3Pick / UR3PicknPlace)
- Follow `batch_runs/sweeps/jax_ppo_paramcalculation.md`, not memory. Always `num_resets_per_eval=1`, `num_evals>=15`.
- **Bump the SLURM `--array` to the JSONL data-line count** — too small silently skips trailing configs; too large → IndexError.
- A new sweep-driven env knob needs **3 edits**: env `default_config()`, the `env_overrides` forward in `learning/notebooks/run_experiment.py`, and that file's `_extract_ppo_overrides` reserved set.
- Contact-physics XMLs are shared by both envs; changing them needs SLURM retraining.

## Real-robot deploy (RTDE)
- Run deploy scripts from their own folder (`../../` paths depend on it).
- Leave `ACTION_SCALE`/`GRIPPER_ACTION_SCALE = None` — resolved from the checkpoint `metadata.json`; **deploy scale must equal trained scale.**
- Gripper (docstrings lie): sim per-finger `0 = OPEN`, `0.025 = CLOSED`; native percent same direction.
- RTDE TCP X/Y are negated to match the MuJoCo base frame. `servoJ`/`moveJ` args are `(q, speed, accel)` — speed before accel.
- Arm needs PolyScope **Remote Control** + External Control URCap **PLAYING** (press Play); `connect()` blocks until then. Hand-E = Robotiq URCapX XML-RPC `:49999`, ≤10 Hz.

## Env
- Linux `MUJOCO_GL=egl` (osmesa absent); macOS `MUJOCO_GL=glfw` + `JAX_PLATFORM_NAME=cpu`. `imageio-ffmpeg` isn't in `.[all]` — install once or `save_video()` crashes.
