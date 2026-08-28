"""Robot-independent PPO training entry point.

`run_experiment(cfg, out_dir)` dispatches on `cfg["env_name"]` via
`registry.load()` and trains with Brax PPO, logging to W&B. It is the single
shared runner for every robot/task (UR10, UR3, Panda, ...): the env-specific
defaults live in `mujoco_playground.config.manipulation_params`, and the sweep
JSONL only overrides hyperparameters + run metadata. `cfg["env_name"]` is
required (no robot-specific hardcoded default).

This consolidates the former per-robot copies (robots/UR10e/UR10_ppo.py,
learning/notebooks/Panda_ppo.py, learning/notebooks/panda_test.py), which now
re-export `run_experiment` from here for back-compat.
"""

import os
import platform
import subprocess
import json
import functools
import random
from datetime import datetime
from typing import Any, Dict, Optional
from mujoco_playground.config import manipulation_params
from mujoco_playground import registry, wrapper
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from jax.tree_util import tree_leaves
import wandb
import mediapy
import numpy as np
import jax.numpy as jnp
from datetime import datetime
from flax import serialization
import mujoco
from learning.curriculum.best_params import BestParamsRecorder



def is_nvidia_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _setup_mujoco_backend() -> None:
    system = platform.system()
    if "MUJOCO_GL" in os.environ:
        print(
            f"[INFO] MUJOCO_GL already set to {os.environ['MUJOCO_GL']!r}, not overriding.",
            flush=True,
        )
        return
    if system == "Linux" and is_nvidia_available():
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] Detected NVIDIA GPU on Linux – using MUJOCO_GL=egl", flush=True)
    elif system == "Darwin":
        os.environ["MUJOCO_GL"] = "glfw"
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        print("[INFO] Running on macOS – forcing CPU backend (glfw)", flush=True)
    else:
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        print("[INFO] No GPU / unknown system – using MUJOCO_GL=osmesa", flush=True)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def apply_validated_overrides(base: dict, overrides: dict, *, strict: bool = True) -> dict:
    unknown = []
    for k in overrides.keys():
        if k not in base and k != "network_factory":
            unknown.append(k)
    if unknown:
        msg = f"Unknown override keys: {unknown}"
        if strict:
            raise ValueError(msg)
        print("[WARN]", msg)
    return _deep_update(base, overrides)

def cast_to_schema(values: dict, schema: dict) -> dict:
    """
    Cast values to the types given by schema.
    Extra keys (not in schema) are left untouched.
    """
    out = {}
    for k, v in values.items():
        if k in schema and v is not None:
            try:
                out[k] = type(schema[k])(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out



def _extract_ppo_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {
        "run_id",
        "wandb_project",
        "wandb_mode",
        "wandb_group",
        "wandb_tags",
        "out_dir",
        "video_every_evals",
        "render_every",
        "video_tag",
        "camera_kwargs",
        "deterministic",
        "env_name",
        "algo",
        "notes",
        "init_keyframe",
        "init_qpos_noise",
        "init_start_random",
        "lifter_height_nom",
        # Kept reserved so the loud migration error below (not a confusing
        # "Unknown override keys" from the PPO validator) is what a stale JSONL
        # carrying this removed key actually hits.
        "lifter_height_max",
        # D18/D24: absolute table-top range, replaces lifter_height_max.
        "lifter_height_abs_min",
        "lifter_height_abs_max",
        "lifter_tilt_max",
        "box_z_rot_range",
        "box_y_center_offset",
        "target_y_center_offset",
        "action_scale",
        "gripper_action_scale",
        # reward_config.scales.* keys -- forwarded to the env below, never PPO
        # params. Must be reserved so apply_validated_overrides(strict=True)
        # doesn't reject them as "Unknown override keys".
        "action_rate",
        "box_target",
        "gripper_align",
        "hold_target",
        "success_bonus",
        # 2026-08 (v3 reward fix): the two scales behind the "moving is net
        # negative at long range" diagnostic. Exposed so a sweep line can dial
        # them without a code edit -- their DEFAULTS deliberately stay at the
        # legacy -0.10 / 0.3, because evaluation/ur3_reward_replay.py reads
        # ur3_pick.default_config() at import time to score ARCHIVED real-robot
        # runs, and moving a default would retroactively rescore the D23
        # campaign with scales those policies never trained under.
        "robot_target_qpos",
        "gripper_box",
        "success_tol",
        "num_eval_envs",
        "seed",
        "domain_randomization",
        "sticky_latches",
        # 2026-08 post-lift annuity gating. TOP-LEVEL env flags (not
        # reward_config.scales keys), forwarded to the env below -- reserved so
        # apply_validated_overrides(strict=True) does not reject them as
        # "Unknown override keys". That rejection would raise only AFTER SLURM
        # has already allocated the GPU, burning the whole array allocation.
        "gate_gripper_box_on_lift",
        "gate_gripper_align_on_lift",
        "target_mode",
        "target_r_min",
        "target_r_max",
        "target_azim_min",
        "target_azim_max",
        # DR-ladder position knobs ("Plan - Sim-to-Real Gap Protocol", C1):
        # were hardcoded literals in ur3_pick.reset(), now config fields so a
        # sweep can zero them for a deterministic L0 baseline.
        "box_xy_jitter",
        "target_z_jitter",
        "finger_random_init",
        "obs_include_velocity",
        # 2026-08 (v3 reward fix): grasp-frame alignment knobs. align_mode is a
        # STRING ("axis_free" | "axis_aware"); grasp_align_thresh's UNITS depend
        # on it (a 39 deg misaligned grasp scores 0.307 under axis_free but
        # 0.091 under axis_aware), so a sweep line must always set the two
        # TOGETHER or the number is silently reinterpreted.
        "align_mode",
        "align_pref_floor",
        "grasp_align_thresh",
    }
    overrides: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in reserved or k.startswith("domain_rand."):
            # domain_rand.* are nested env-config overrides (Part 2 physics DR),
            # forwarded to registry.load below -- never PPO params.
            continue
        overrides[k] = v
    if "network_factory" in cfg and isinstance(cfg["network_factory"], dict):
        overrides["network_factory"] = cfg["network_factory"]
    return overrides


def _make_progress_wandb(prog_state=None, on_eval=None):
    """Build brax's progress_fn.

    `on_eval(step, metrics)` is invoked AFTER the wandb.log so that the eval
    which triggers a curriculum early stop is still recorded before any
    exception unwinds the training loop.

    `prog_state` is retained only for backwards compatibility with external
    callers; the best-checkpoint pairing no longer uses it. It used to carry
    "last_eval_reward" into policy_params_fn, which is precisely the off-by-one
    that BestParamsRecorder fixes -- brax fires policy_params_fn(s_k) BEFORE
    progress_fn(s_k) (ppo/train.py:727 vs :748), so that read returned eval
    k-1's reward. See learning/curriculum/best_params.py.
    """
    def progress_wandb(num_steps, metrics):
        log_dict = {"training/num_steps": int(num_steps)}
        for k, v in metrics.items():
            try:
                log_dict[k] = float(v)
            except Exception:
                pass
        if prog_state is not None and "eval/episode_reward" in metrics:
            try:
                prog_state["last_eval_reward"] = float(metrics["eval/episode_reward"])
                prog_state["last_step"] = int(num_steps)
            except Exception:
                pass
        wandb.log(log_dict, step=int(num_steps))
        if on_eval is not None:
            on_eval(int(num_steps), metrics)

    return progress_wandb

def _extract_body_positions(state):
    """Extract all body Cartesian positions from a Brax/MJX state.
    
    Tries multiple access paths since State structure varies across
    Brax versions and env wrappers.
    """
    # Possible locations of the physics pipeline state
    candidates = [
        getattr(state, "pipeline_state", None),
        getattr(state, "data", None),
        getattr(state, "qp", None),       # older Brax
        getattr(state, "mjx_data", None),
        state,                              # state itself might have x/xpos
    ]
    for ps in candidates:
        if ps is None:
            continue
        # Brax MJX style: ps.x.pos → (n_bodies, 3)
        x = getattr(ps, "x", None)
        if x is not None:
            pos = getattr(x, "pos", None)
            if pos is not None:
                return np.asarray(pos)
        # Classic MuJoCo / dm_control style
        xpos = getattr(ps, "xpos", None)
        if xpos is not None:
            return np.asarray(xpos)
    return None

def _extract_torques(state):
    """
    Extract actuator torques / joint-level forces from the state.
    Returns a 1-D numpy array of shape (n_actuators,) or None.
    """
    candidates = [
        getattr(state, "pipeline_state", None),
        getattr(state, "data", None),
        getattr(state, "qp", None),
        getattr(state, "mjx_data", None),
        state,
    ]
    for ps in candidates:
        if ps is None:
            continue
        for attr in ("actuator_force", "qfrc_actuator", "qfrc_applied"):
            val = getattr(ps, attr, None)
            if val is not None:
                return np.asarray(val).flatten()
    return None

def _extract_velocities(state):
    """
    Extract joint/generalised velocities from the state.
    Returns a 1-D numpy array of shape (nv,) or None.

    Tries qvel (generalised velocities) first, then xd.vel (body velocities).
    """
    candidates = [
        getattr(state, "pipeline_state", None),
        getattr(state, "data", None),
        getattr(state, "qp", None),
        getattr(state, "mjx_data", None),
        state,
    ]
    for ps in candidates:
        if ps is None:
            continue
        # Generalised velocities (nv,) — most useful for joint-level analysis
        qvel = getattr(ps, "qvel", None)
        if qvel is not None:
            return np.asarray(qvel).flatten()
        # Brax MJX body velocities: ps.xd.vel → (n_bodies, 3)
        xd = getattr(ps, "xd", None)
        if xd is not None:
            vel = getattr(xd, "vel", None)
            if vel is not None:
                return np.asarray(vel).flatten()
        # Older Brax: qp.vel
        vel = getattr(ps, "vel", None)
        if vel is not None:
            return np.asarray(vel).flatten()
    return None

def _get_body_names(env):
    """Try to read body names from the MuJoCo model."""
    try:
        for attr in ("mj_model", "model", "sys"):
            obj = getattr(env, attr, None)
            if obj is None:
                continue
            mj = getattr(obj, "mj_model", obj)
            if hasattr(mj, "body") and hasattr(mj, "nbody"):
                return [mj.body(i).name for i in range(mj.nbody)]
    except Exception:
        pass
    return None
    
def _get_actuator_names(env):
    """Try to read actuator names from the MuJoCo model."""
    try:
        for attr in ("mj_model", "model", "sys"):
            obj = getattr(env, attr, None)
            if obj is None:
                continue
            mj = getattr(obj, "mj_model", obj)
            if hasattr(mj, "actuator") and hasattr(mj, "nu"):
                return [mj.actuator(i).name for i in range(mj.nu)]
    except Exception:
        pass
    return None
def _extract_qpos(state):
    """
    Extract generalised joint positions (qpos) from the state.
    Returns a 1-D numpy array of shape (nq,) or None.
    """
    candidates = [
        getattr(state, "pipeline_state", None),
        getattr(state, "data", None),
        getattr(state, "qp", None),
        getattr(state, "mjx_data", None),
        state,
    ]
    for ps in candidates:
        if ps is None:
            continue
        qpos = getattr(ps, "qpos", None)
        if qpos is not None:
            return np.asarray(qpos).flatten()
        # Older Brax: qp.pos
        pos = getattr(ps, "pos", None)
        if pos is not None and not hasattr(ps, "x"):
            return np.asarray(pos).flatten()
    return None


def _get_joint_names(env):
    """Try to read joint names from the MuJoCo model."""
    try:
        for attr in ("mj_model", "model", "sys"):
            obj = getattr(env, attr, None)
            if obj is None:
                continue
            mj = getattr(obj, "mj_model", obj)
            if hasattr(mj, "joint") and hasattr(mj, "njnt"):
                return [mj.joint(i).name for i in range(mj.njnt)]
    except Exception:
        pass
    return None

def _rollout_and_log_video_from_make_policy(
    *,
    num_steps: int,
    make_policy,
    params,
    env_name: str,
    run_id: str,
    seed: int,
    episode_length: int,
    render_every: int,
    camera_kwargs: Dict[str, Any],
    env_overrides: Optional[Dict[str, Any]] = None,
):
    import jax.numpy as jnp

    eval_env = registry.load(env_name, config_overrides=env_overrides or {})
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    policy = jax.jit(make_policy(params, deterministic=True))
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    rollout = [state]
    ep_reward = 0.0

    # ------------one time debug print------------
    # One-time debug: show what the State object looks like
    state_attrs = [a for a in dir(state) if not a.startswith("_")]
    print(f"[INFO] State attributes: {state_attrs}", flush=True)
    for attr in ("pipeline_state", "data", "qp", "mjx_data"):
        sub = getattr(state, attr, None)
        if sub is not None:
            sub_attrs = [a for a in dir(sub) if not a.startswith("_")]
            print(f"[INFO]   state.{attr} attrs: {sub_attrs[:20]}...", flush=True)
    # -----------------------------------------

    # ── Per-step data collection ──
    step_data = {
        "obs":             [],   # (T, obs_dim)
        "ctrl":            [],   # (T, act_dim)
        "reward":          [],   # (T,)
        "body_pos":        [],   # (T, n_bodies, 3)
        "torques":         [],   # (T, n_actuators) — may stay empty
        "velocities":      [],   # (T, nv) — may stay empty
        "qpos":            [],   # (T, nq) — joint positions
    }


    # Record initial state (before any action)
    pos0 = _extract_body_positions(state)
    if pos0 is not None:
        step_data["body_pos"].append(pos0.copy())
        print(f"[INFO] Body positions found: shape={pos0.shape}", flush=True)
    else:
        print("[INFO] No body positions found in state — coords will be empty", flush=True)

    qpos0 = _extract_qpos(state)
    if qpos0 is not None:
        step_data["qpos"].append(qpos0.copy())
        print(f"[INFO] Qpos found: shape={qpos0.shape}", flush=True)
    else:
        print("[INFO] No qpos found in state — joint angle columns will be empty", flush=True)

    step_data["obs"].append(np.asarray(state.obs).flatten())
    step_data["ctrl"].append(np.zeros_like(np.asarray(state.obs).flatten()[:0]))  # placeholder
    step_data["reward"].append(0.0)

    torque0 = _extract_torques(state)
    if torque0 is not None:
        step_data["torques"].append(torque0.copy())
        print(f"[INFO] Torques found: shape={torque0.shape}", flush=True)
    else:
        print("[INFO] No torques found in state — torque columns will be empty", flush=True)

    vel0 = _extract_velocities(state)
    if vel0 is not None:
        step_data["velocities"].append(vel0.copy())
        print(f"[INFO] Velocities found: shape={vel0.shape}", flush=True)
    else:
        print("[INFO] No velocities found in state — velocity columns will be empty", flush=True)
    
    # ────────────────start rollout───────────────
    for _ in range(int(episode_length)):
        rng, act_rng = jax.random.split(rng)
        out = policy(state.obs, act_rng)
        if isinstance(out, tuple):
            ctrl = out[0]
        elif isinstance(out, dict):
            ctrl = out.get("action", out.get("ctrl", out))
        else:
            ctrl = out
        ctrl = jnp.asarray(ctrl)

        state = jit_step(state, ctrl)
        rollout.append(state)

        r = float(jnp.asarray(state.reward))
        ep_reward += r

        # Collect everything
        step_data["obs"].append(np.asarray(state.obs).flatten())
        step_data["ctrl"].append(np.asarray(ctrl).flatten())
        step_data["reward"].append(r)

        pos = _extract_body_positions(state)
        if pos is not None:
            step_data["body_pos"].append(pos.copy())

        torque = _extract_torques(state)
        if torque is not None:
            step_data["torques"].append(torque.copy())
        vel = _extract_velocities(state)
        if vel is not None:
            step_data["velocities"].append(vel.copy())
        qp = _extract_qpos(state)
        if qp is not None:
            step_data["qpos"].append(qp.copy())

    # Convert lists → numpy arrays
    step_data["obs"]    = np.array(step_data["obs"])       # (T+1, obs_dim)
    step_data["reward"] = np.array(step_data["reward"])    # (T+1,)
    if step_data["body_pos"]:
        step_data["body_pos"] = np.array(step_data["body_pos"])  # (T+1, n_bodies, 3)
    if step_data["torques"]:
        step_data["torques"] = np.array(step_data["torques"])    # (T+1, n_actuators)
    if step_data["velocities"]:
        step_data["velocities"] = np.array(step_data["velocities"])  # (T+1, nv)
    if step_data["qpos"]:
        step_data["qpos"] = np.array(step_data["qpos"])  # (T+1, nq)

    # ctrl: first entry was placeholder, replace with zeros matching action dim
    act_dim = step_data["ctrl"][1].shape[0] if len(step_data["ctrl"]) > 1 else 0
    step_data["ctrl"][0] = np.zeros(act_dim)
    step_data["ctrl"] = np.array(step_data["ctrl"])        # (T+1, act_dim)

    # Attach metadata
    step_data["body_names"]     = _get_body_names(eval_env)
    step_data["actuator_names"] = _get_actuator_names(eval_env)
    step_data["joint_names"]    = _get_joint_names(eval_env)
    step_data["ep_reward"]      = ep_reward
    step_data["num_steps"]      = num_steps
    step_data["seed"]           = seed

    # Build tag using episode reward + steps
    step_tag = f"{run_id}_rew{ep_reward:.1f}_steps{int(num_steps)}"

    trajectory = rollout[:: int(render_every)]

    cam = camera_kwargs or {"camera": "side_130"}
    cam["width"] = 800
    cam["height"] = 600 

    frames = eval_env.render(trajectory, **cam)
    # frames might be list-of-arrays; make it (T,H,W,C) uint8 on CPU
    frames = np.asarray(frames)

    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    fps = float(1.0 / eval_env.dt) / float(render_every)
    video_path = os.path.join(wandb.run.dir, f"{env_name}_{step_tag}.mp4")
    try:
        mediapy.write_video(video_path, frames, fps=fps)
        # Make W&B track the file explicitly
        wandb.save(video_path)
        # Use a stable key that changes per eval/step
        wandb.log(
            {f"eval/video": wandb.Video(video_path, fps=int(fps), format="mp4")},
            step=int(num_steps),
        )
    except Exception as e:
        print(f"[WARN] Video skipped: {e}", flush=True)

    # ── Save rollout data as CSV + W&B artifact ──
    try:
        import csv as csv_mod

        T = len(step_data["reward"])
        body_names = step_data["body_names"] or []
        act_names  = step_data["actuator_names"] or []
        act_dim    = step_data["ctrl"].shape[1] if step_data["ctrl"].ndim == 2 else 0
        has_pos    = isinstance(step_data["body_pos"], np.ndarray) and step_data["body_pos"].ndim == 3
        has_torque = isinstance(step_data["torques"], np.ndarray) and step_data["torques"].ndim == 2
        has_vel    = isinstance(step_data["velocities"], np.ndarray) and step_data["velocities"].ndim == 2
        has_qpos   = isinstance(step_data["qpos"], np.ndarray) and step_data["qpos"].ndim == 2


        # ── Build column headers ──
        columns = ["timestep", "reward"]
        ctrl_cols = [f"ctrl_{i}" for i in range(act_dim)]
        columns += ctrl_cols

        body_pos_cols = []
        if has_pos:
            n_bodies = step_data["body_pos"].shape[1]
            if not body_names or len(body_names) != n_bodies:
                body_names = [f"body_{i}" for i in range(n_bodies)]
            for b_name in body_names:
                body_pos_cols += [f"{b_name}_x", f"{b_name}_y", f"{b_name}_z"]
            columns += body_pos_cols

        torque_cols = []
        if has_torque:
            n_act = step_data["torques"].shape[1]
            if not act_names or len(act_names) != n_act:
                act_names = [f"actuator_{i}" for i in range(n_act)]
            torque_cols = [f"torque_{a}" for a in act_names]
            columns += torque_cols
        
        vel_cols = []
        if has_vel:
            n_vel = step_data["velocities"].shape[1]
            vel_cols = [f"vel_{i}" for i in range(n_vel)]
            columns += vel_cols

        qpos_cols = []
        if has_qpos:
            jnt_names = step_data.get("joint_names") or []
            n_qpos = step_data["qpos"].shape[1]
            if not jnt_names or len(jnt_names) != n_qpos:
                jnt_names = [f"joint_{i}" for i in range(n_qpos)]
            qpos_cols = [f"qpos_{j}" for j in jnt_names]
            columns += qpos_cols

        # ── Build rows ──
        rows = []
        for t in range(T):
            row = [t, float(step_data["reward"][t])]
            row += step_data["ctrl"][t].tolist() if t < step_data["ctrl"].shape[0] else [0.0] * act_dim
            if has_pos:
                row += step_data["body_pos"][t].flatten().tolist()
            if has_torque:
                row += step_data["torques"][t].flatten().tolist()
            if has_vel:
                row += step_data["velocities"][t].flatten().tolist()
            if has_qpos:
                row += step_data["qpos"][t].flatten().tolist()

            rows.append(row)

        # Save CSV
        csv_path = os.path.join(wandb.run.dir, f"{env_name}_{step_tag}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        wandb.save(csv_path)

        # Log as W&B Table (capped at reasonable size for the UI)
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({"eval/rollout_data": table}, step=int(num_steps))

        n_bodies_str = f"{n_bodies} bodies" if has_pos else "no body pos"
        n_qpos_str   = f"{step_data['qpos'].shape[1]} qpos dims" if has_qpos else "no qpos"
        n_torque_str = f"{step_data['torques'].shape[1]} actuators" if has_torque else "no torques"
        n_vel_str    = f"{step_data['velocities'].shape[1]} vel dims" if has_vel else "no velocities"
        print(
            f"[INFO] Rollout logged: {T} steps, {act_dim} ctrl dims, "
            f"{n_bodies_str}, {n_torque_str}, {n_vel_str}, {n_qpos_str}",
        )

    except Exception as e:
        print(f"[WARN] Rollout data logging skipped: {e}", flush=True)

    return ep_reward, step_data

def _wb_jsonify(x):
    # Make values W&B-config friendly
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _wb_jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_wb_jsonify(v) for v in x]
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)

def _wb_log_final_train_config(*, ppo_training_params: dict, nf_cfg: dict | None, env_name: str, seed: int):
    # Nested blobs for readability
    wandb.config.update(
        {
            "final": {
                "env_name": env_name,
                "seed": seed,
                "ppo": _wb_jsonify(ppo_training_params),
                "net": _wb_jsonify(nf_cfg) if isinstance(nf_cfg, dict) else _wb_jsonify({"network_factory": nf_cfg}),
            }
        },
        allow_val_change=True,
    )
    # Flattened keys for easy filtering/columns
    wandb.config.update(
        {f"ppo.{k}": _wb_jsonify(v) for k, v in ppo_training_params.items()},
        allow_val_change=True,
    )
    if isinstance(nf_cfg, dict):
        wandb.config.update(
            {f"net.{k}": _wb_jsonify(v) for k, v in nf_cfg.items()},
            allow_val_change=True,
        )
# def _dbg_print_ppo_and_eval_calcs(cfg: dict, env=None) -> None:
#     # ---- TRAINING CALCS ----
#     ne = int(cfg.get("num_envs"))
#     ul = int(cfg.get("unroll_length"))
#     nt = int(cfg.get("num_timesteps"))
#     num_evals = int(cfg.get("num_evals", 0))

#     # Your "batch_size_steps" (collector throughput per iteration)
#     batch_size_steps = ne * ul

#     # PPO mini-batch params (must exist in cfg if you want deterministic prints)
#     batch_size = cfg.get("batch_size", None)
#     num_minibatches = cfg.get("num_minibatches", None)

#     print(f"[DBG CALC] num_envs={ne}", flush=True)
#     print(f"[DBG CALC] unroll_length={ul}", flush=True)
#     print(f"[DBG CALC] num_timesteps={nt}", flush=True)
#     print(f"[DBG CALC] batch_size_steps = num_envs*unroll_length = {ne}*{ul} = {batch_size_steps}", flush=True)

#     expected_updates_floor = max(1, nt // batch_size_steps)
#     print(f"[DBG CALC] expected_updates_floor = max(1, num_timesteps//batch_size_steps) = max(1, {nt}//{batch_size_steps}) = {expected_updates_floor}", flush=True)

#     # Print the assertion-related values (this is the one that bit you)
#     if batch_size is not None and num_minibatches is not None:
#         batch_size = int(batch_size)
#         num_minibatches = int(num_minibatches)
#         prod = batch_size * num_minibatches
#         mod = prod % ne
#         print(f"[DBG CALC] batch_size={batch_size}", flush=True)
#         print(f"[DBG CALC] num_minibatches={num_minibatches}", flush=True)
#         print(f"[DBG CALC] assert check: (batch_size*num_minibatches) % num_envs = ({batch_size}*{num_minibatches}) % {ne} = {mod}", flush=True)
#         print(f"[DBG CALC] batch_size % num_minibatches = {batch_size} % {num_minibatches} = {batch_size % num_minibatches}", flush=True)
#     else:
#         print(f"[DBG CALC] batch_size/num_minibatches not provided in cfg; cannot print PPO assert check.", flush=True)

#     # ---- EVAL CALCS ----
#     num_eval_envs = int(cfg.get("num_eval_envs", 0))
#     eval_every_steps = None
#     if num_evals and num_evals > 0:
#         eval_every_steps = nt / float(num_evals)
#         # also show as int floor for log cadence intuition
#         print(f"[DBG EVAL] num_evals={num_evals}", flush=True)
#         print(f"[DBG EVAL] eval cadence (steps_between_evals) = num_timesteps/num_evals = {nt}/{num_evals} = {eval_every_steps:.3f}", flush=True)
#     else:
#         print(f"[DBG EVAL] num_evals not set (or 0); no eval cadence computed.", flush=True)

#     print(f"[DBG EVAL] num_eval_envs={num_eval_envs}", flush=True)

#     # If we can, use env.episode_length to estimate env-steps per eval.
#     # In Brax, episode_length is usually available on the env.
#     episode_length = None
#     if env is not None:
#         episode_length = getattr(env, "episode_length", None)

#     if episode_length is not None and num_eval_envs > 0:
#         episode_length = int(episode_length)
#         eval_env_steps_per_eval = num_eval_envs * episode_length
#         print(f"[DBG EVAL] env.episode_length={episode_length}", flush=True)
#         print(f"[DBG EVAL] approx eval env-steps per eval = num_eval_envs*episode_length = {num_eval_envs}*{episode_length} = {eval_env_steps_per_eval}", flush=True)
#         if eval_every_steps is not None:
#             # ratio: how much eval “cost” per training steps between evals
#             print(f"[DBG EVAL] approx eval-cost ratio = eval_env_steps_per_eval / steps_between_evals = {eval_env_steps_per_eval}/{eval_every_steps:.3f} = {eval_env_steps_per_eval / eval_every_steps:.6f}", flush=True)
#     else:
#         # Fallback: we can still print a lower-bound notion:
#         # “at least num_eval_envs steps occur on the first eval step”
#         if num_eval_envs > 0:
#             print(f"[DBG EVAL] env.episode_length unavailable; cannot estimate full eval env-steps.", flush=True)
#             print(f"[DBG EVAL] lower bound: eval will step at least once across num_eval_envs => >= {num_eval_envs} env-steps per eval step.", flush=True)
#         else:
#             print(f"[DBG EVAL] num_eval_envs not set (or 0); cannot estimate eval env-steps.", flush=True)
## ═══════════════════════════════════════════════════════════
##  DOMAIN RANDOMIZATION
## ═══════════════════════════════════════════════════════════
#
#  Progression plan (controlled via cfg["domain_randomization"]):
#
#    Phase 1  {"enabled": true}
#             → mass ×[0.5, 2.0]  +  sliding friction ×[0.5, 2.0]
#
#    Phase 2  {"enabled": true, "size_range": [0.7, 1.3]}
#             → adds geom size scaling
#
#    Phase 3  (future) different geom_type — needs separate compiled
#             models per shape; not wired yet.
#
#  The function signature Brax PPO expects:
#      domain_randomize(sys, rng)  →  (sys_batched, in_axes)
#
# ═══════════════════════════════════════════════════════════


def _get_mj_model(env):
    """Extract the raw mujoco.MjModel from a Playground env."""
    for attr in ("mj_model", "_mj_model", "model"):
        obj = getattr(env, attr, None)
        if obj is not None and hasattr(obj, "ngeom"):
            return obj
    sys = getattr(env, "sys", None)
    if sys is not None:
        mj = getattr(sys, "mj_model", None)
        if mj is not None:
            return mj
    raise RuntimeError(
        "[DR] Cannot find mujoco.MjModel on env. "
        f"Tried attrs: mj_model, _mj_model, model, sys.mj_model. "
        f"Env type: {type(env)}"
    )


def _find_box_ids(mj_model, box_body_name="box", box_geom_name="box"):
    """Look up body/geom IDs + nominal values. Called ONCE before training."""
    box_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, box_body_name
    )
    box_geom_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, box_geom_name
    )

    if box_body_id < 0:
        for alt in ("cube", "object", "target_object", "pick_object"):
            box_body_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, alt
            )
            if box_body_id >= 0:
                box_body_name = alt
                break
    if box_geom_id < 0:
        for alt in ("cube", "object", "target_object", "pick_object"):
            box_geom_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, alt
            )
            if box_geom_id >= 0:
                box_geom_name = alt
                break

    if box_body_id < 0:
        raise ValueError(
            f"[DR] Body '{box_body_name}' not found in model. "
            f"Available bodies: "
            + str([mj_model.body(i).name for i in range(mj_model.nbody)])
        )
    if box_geom_id < 0:
        raise ValueError(
            f"[DR] Geom '{box_geom_name}' not found in model. "
            f"Available geoms: "
            + str([mj_model.geom(i).name for i in range(mj_model.ngeom)])
        )

    info = {
        "box_body_id":      box_body_id,
        "box_geom_id":      box_geom_id,
        "box_body_name":    box_body_name,
        "box_geom_name":    box_geom_name,
        "nominal_mass":     float(mj_model.body_mass[box_body_id]),
        "nominal_friction": mj_model.geom_friction[box_geom_id].copy(),
        "nominal_size":     mj_model.geom_size[box_geom_id].copy(),
    }
    print(f"[DR] Found body '{box_body_name}' id={box_body_id}, "
          f"geom '{box_geom_name}' id={box_geom_id}", flush=True)
    print(f"[DR]   nominal mass     = {info['nominal_mass']:.4f} kg", flush=True)
    print(f"[DR]   nominal friction = {info['nominal_friction']}", flush=True)
    print(f"[DR]   nominal size     = {info['nominal_size']}", flush=True)
    return info


def make_domain_randomize_fn(
    box_info: dict,
    *,
    mass_range=(0.5, 2.0),
    friction_range=(0.5, 2.0),
    size_range=None,
):
    """Build a domain_randomize(sys, rng) function for Brax PPO."""
    box_body_id = box_info["box_body_id"]
    box_geom_id = box_info["box_geom_id"]
    nom_mass    = box_info["nominal_mass"]
    nom_fric    = jnp.array(box_info["nominal_friction"])
    nom_size    = jnp.array(box_info["nominal_size"])

    do_size = size_range is not None

    print(f"[DR] Building randomizer:", flush=True)
    print(f"[DR]   mass_range     = {mass_range}", flush=True)
    print(f"[DR]   friction_range = {friction_range}", flush=True)
    print(f"[DR]   size_range     = {size_range}", flush=True)

    def domain_randomize(sys, rng):
        @jax.vmap
        def rand(rng):
            k1, k2, k3 = jax.random.split(rng, 3)

            mass_scale = jax.random.uniform(
                k1, minval=mass_range[0], maxval=mass_range[1]
            )
            new_mass = sys.body_mass.at[box_body_id].set(
                nom_mass * mass_scale
            )

            fric_scale = jax.random.uniform(
                k2, minval=friction_range[0], maxval=friction_range[1]
            )
            new_fric_row = nom_fric.at[0].set(nom_fric[0] * fric_scale)
            new_friction = sys.geom_friction.at[box_geom_id].set(new_fric_row)

            if do_size:
                size_scale = jax.random.uniform(
                    k3, minval=size_range[0], maxval=size_range[1]
                )
                new_size_row = nom_size * size_scale
                new_geom_size = sys.geom_size.at[box_geom_id].set(new_size_row)
            else:
                new_geom_size = sys.geom_size

            return new_mass, new_friction, new_geom_size

        new_mass, new_friction, new_geom_size = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, sys)

        replace_axes = {"body_mass": 0, "geom_friction": 0}
        replace_vals = {"body_mass": new_mass, "geom_friction": new_friction}

        if do_size:
            replace_axes["geom_size"] = 0
            replace_vals["geom_size"] = new_geom_size

        in_axes = in_axes.tree_replace(replace_axes)
        sys = sys.tree_replace(replace_vals)

        return sys, in_axes

    return domain_randomize


def print_dict_pairs(d, prefix=""):
    for k, v in d.items():
        print(f"{prefix}{k}: {v}")


def run_experiment(cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    _setup_mujoco_backend()

    if not cfg.get("env_name"):
        raise ValueError(
            "cfg['env_name'] is required (robot-independent runner has no default). "
            "Set it in the sweep JSONL, e.g. \"env_name\": \"UR3PicknPlace\"."
        )
    env_name = str(cfg["env_name"])
    print("JAX devices:", jax.devices(), flush=True)
    base_run_id = str(cfg.get("run_id", "run"))
    camera_kwargs = cfg.get("camera_kwargs") or {"camera": "side_130"}

    # Seed
    cfg_seed = cfg.get("seed", None)
    seed = int(cfg_seed) if cfg_seed is not None else random.getrandbits(32)
    random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    # W&B run identity (unique)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = random.randint(0, 9999)
    run_id_tag = f"{base_run_id}_{timestamp}_{short_uid}"

    wandb_mode = str(cfg.get("wandb_mode", "online"))
    run = wandb.init(
        project=str(cfg.get("wandb_project", "mjx_ppo")),
        name=run_id_tag,
        id=run_id_tag,
        group=str(cfg.get("wandb_group", "")) or None,
        tags=cfg.get("wandb_tags", None),
        resume="never",
        config=cfg,
        mode=wandb_mode,
        dir=out_dir,
    )
    # Track the PEAK reward/success in the run summary (default is last value),
    # so the W&B runs table / parallel-coords / param-importance panels rank by
    # best-over-training rather than the final eval (which can be a post-peak dip).
    wandb.define_metric("eval/episode_reward", summary="max")
    wandb.define_metric("eval/episode_success", summary="max")
    print(f"W&B run: {run_id_tag} (mode={wandb_mode})", flush=True)
    try:
        # -----------------------------
        # Env overrides + env creation
        # -----------------------------
        env_overrides = {}
        if "init_keyframe" in cfg:
            env_overrides["init_keyframe"] = cfg["init_keyframe"]
        if "init_qpos_noise" in cfg:
            env_overrides["init_qpos_noise"] = tuple(
                float(x) for x in cfg["init_qpos_noise"]
            )
        if "init_start_random" in cfg:
            env_overrides["init_start_random"] = str(cfg["init_start_random"])
        if "lifter_height_nom" in cfg:
            env_overrides["lifter_height_nom"] = float(cfg["lifter_height_nom"])
        # D18/D24 (2026-07-29): the table top is now drawn from an ABSOLUTE
        # range instead of a +- band about lifter_height_nom, so the old
        # `lifter_height_max` key no longer exists in ur3_pick.default_config().
        # Fail LOUD on it rather than dropping it silently: its meaning has now
        # changed twice (absolute max of uniform(0.003, v) -> symmetric +- band
        # -> gone), and a stale JSONL that still carries it would otherwise
        # train against a table-height distribution it never asked for. Every
        # pre-2026-07-29 sweep that set it (UR3Pick_grasp_retry,
        # UR3Pick_sweep_stickyoff, the old DR-ladder JSONLs) must be updated to
        # the new keys before it can be re-run.
        if "lifter_height_max" in cfg:
            raise ValueError(
                "cfg['lifter_height_max'] is no longer a valid key (D18/D24, "
                "2026-07-29). The table top is now sampled as an ABSOLUTE height "
                "uniform(lifter_height_abs_min, lifter_height_abs_max) rather "
                "than lifter_height_nom +- lifter_height_max. Translate it: a "
                "flat table at the nominal height is "
                "lifter_height_abs_min = lifter_height_abs_max = "
                "lifter_height_nom (0.095); the old +-v band is "
                "abs_min = nom - v, abs_max = nom + v. See "
                "ur3_pick.default_config() and batch_runs/sweeps/gen_dr_ladder.py's "
                "_DETERMINISTIC_POSITION."
            )
        if "lifter_height_abs_min" in cfg:
            env_overrides["lifter_height_abs_min"] = float(
                cfg["lifter_height_abs_min"]
            )
        if "lifter_height_abs_max" in cfg:
            env_overrides["lifter_height_abs_max"] = float(
                cfg["lifter_height_abs_max"]
            )
        if "lifter_tilt_max" in cfg:
            env_overrides["lifter_tilt_max"] = float(cfg["lifter_tilt_max"])
        if "box_z_rot_range" in cfg:
            env_overrides["box_z_rot_range"] = float(cfg["box_z_rot_range"])
        if "box_y_center_offset" in cfg:
            env_overrides["box_y_center_offset"] = float(cfg["box_y_center_offset"])
        if "target_y_center_offset" in cfg:
            env_overrides["target_y_center_offset"] = float(
                cfg["target_y_center_offset"]
            )
        if "action_scale" in cfg:
            env_overrides["action_scale"] = float(cfg["action_scale"])
        if "gripper_action_scale" in cfg:
            env_overrides["gripper_action_scale"] = float(
                cfg["gripper_action_scale"]
            )
        if "obs_include_velocity" in cfg:
            # addvelocity: appends arm qvel(6)+finger qvel(1) to the obs
            # (26D -> 33D). Default False reproduces the 26D obs exactly.
            env_overrides["obs_include_velocity"] = bool(
                cfg["obs_include_velocity"]
            )
        if "action_rate" in cfg:
            # action_rate is nested under reward_config.scales; ConfigDict
            # update_from_flattened_dict handles the dotted key.
            env_overrides["reward_config.scales.action_rate"] = float(
                cfg["action_rate"]
            )
        if "box_target" in cfg:
            # reward_config.scales.box_target. Exposed so a sweep can restore the
            # pre-8bb664a budget (8.0, was raised to 20.0) -- the "first success
            # vs now" note traced the grasp/align dilution to this bump.
            env_overrides["reward_config.scales.box_target"] = float(
                cfg["box_target"]
            )
        if "gripper_align" in cfg:
            # reward_config.scales.gripper_align. Exposed so a sweep can raise the
            # grasp-frame rotation weight -- the wrist must align earlier when the
            # arm approaches faster (higher action_scale).
            env_overrides["reward_config.scales.gripper_align"] = float(
                cfg["gripper_align"]
            )
        if "hold_target" in cfg:
            # Same nesting as action_rate. Exposed so the reward-fix pilots can
            # switch the dwell bonus off (0.0) without a code edit.
            env_overrides["reward_config.scales.hold_target"] = float(
                cfg["hold_target"]
            )
        if "success_bonus" in cfg:
            # Terminal bonus paid when the success gate fires; see
            # ur3_pick.default_config()'s reward_config.scales.success_bonus.
            # 0.0 restores the pre-2026-07-26 (hover-farming) behaviour.
            env_overrides["reward_config.scales.success_bonus"] = float(
                cfg["success_bonus"]
            )
        # --- 2026-08 v3 reward fix: anti-motion pressure -------------------
        # Measured per-tick economics at d = 0.40 m: gripper_box pays +0.023 for
        # closing distance flat out, while action_rate costs -0.175 (|dA| = 0.5
        # over 7 dims) and robot_target_qpos costs -0.163 SUSTAINED for having
        # left the start pose. Standing still was optimal until d ~= 2 cm --
        # the reward-side half of the "slow approach when far away" finding.
        # Recommended v3: robot_target_qpos 0.05, action_rate -0.02 (moves the
        # break-even out to ~15 cm). Defaults stay legacy on purpose; see the
        # reserved-set comment above.
        if "robot_target_qpos" in cfg:
            env_overrides["reward_config.scales.robot_target_qpos"] = float(
                cfg["robot_target_qpos"]
            )
        if "gripper_box" in cfg:
            # The approach-pull weight itself. Raising it is the only in-term
            # lever left on long-range pull: the cascade's coarse scale (1.5) is
            # already within ~15% of the best any single tanh can do at 0.40 m
            # (k*sech^2(k*d) peaks at k ~= 1.9 there), so a tanh cannot be
            # retuned into a strong long-range gradient -- only reweighted.
            env_overrides["reward_config.scales.gripper_box"] = float(
                cfg["gripper_box"]
            )
        # --- 2026-08 v3 reward fix: grasp-frame alignment ------------------
        if "align_mode" in cfg:
            # STRING knob, NOT a float. ur3_pick.__init__ validates it and
            # resolves the branch at trace time. "axis_free" reproduces the
            # pre-v3 env byte-for-byte; "axis_aware" additionally prefers a
            # top-down approach and a jaw span with real finger clearance, so a
            # side grasp across the box's 4 cm axis (9.9 mm clearance) stops
            # scoring identically to a 3 cm face grasp (19.9 mm).
            env_overrides["align_mode"] = str(cfg["align_mode"])
        if "align_pref_floor" in cfg:
            env_overrides["align_pref_floor"] = float(cfg["align_pref_floor"])
        if "grasp_align_thresh" in cfg:
            # UNITS DEPEND ON align_mode -- always set the pair together. This
            # gate gutted the D23 grasp stage: at 0.3 under axis_free a 39.3 deg
            # misaligned grasp still latched, unlocking lift(5) + box_target(20)
            # + hold_target(6) = 31 raw/step for the rest of the episode, and
            # measured grasp-stage sim->real retention was 0.06-0.21.
            env_overrides["grasp_align_thresh"] = float(cfg["grasp_align_thresh"])
        if "success_tol" in cfg:
            env_overrides["success_tol"] = float(cfg["success_tol"])
        if "sticky_latches" in cfg:
            env_overrides["sticky_latches"] = bool(cfg["sticky_latches"])
        if "gate_gripper_box_on_lift" in cfg:
            env_overrides["gate_gripper_box_on_lift"] = bool(
                cfg["gate_gripper_box_on_lift"]
            )
        if "gate_gripper_align_on_lift" in cfg:
            env_overrides["gate_gripper_align_on_lift"] = bool(
                cfg["gate_gripper_align_on_lift"]
            )
        if "target_mode" in cfg:
            env_overrides["target_mode"] = str(cfg["target_mode"])
        if "target_r_min" in cfg:
            env_overrides["target_r_min"] = float(cfg["target_r_min"])
        if "target_r_max" in cfg:
            env_overrides["target_r_max"] = float(cfg["target_r_max"])
        if "target_azim_min" in cfg:
            env_overrides["target_azim_min"] = float(cfg["target_azim_min"])
        if "target_azim_max" in cfg:
            env_overrides["target_azim_max"] = float(cfg["target_azim_max"])
        if "box_xy_jitter" in cfg:
            env_overrides["box_xy_jitter"] = tuple(
                float(x) for x in cfg["box_xy_jitter"]
            )
        if "target_z_jitter" in cfg:
            env_overrides["target_z_jitter"] = tuple(
                float(x) for x in cfg["target_z_jitter"]
            )
        if "finger_random_init" in cfg:
            env_overrides["finger_random_init"] = bool(cfg["finger_random_init"])
        if "episode_length" in cfg:
            # Reach the env too, not just ppo.train's EpisodeWrapper: the env's
            # own at_horizon check uses self._config.episode_length, so without
            # this a JSONL override cannot lengthen the episode (the env still
            # terminates at its default). episode_length stays non-reserved, so
            # it also flows to ppo.train -- both horizons stay consistent.
            env_overrides["episode_length"] = int(cfg["episode_length"])
        # Physics domain randomization (Part 2): forward every "domain_rand.*"
        # dotted key straight through as a nested env-config override. ConfigDict
        # update_from_flattened_dict resolves the dotted path, so a sweep can set
        # e.g. "domain_rand.enable"/"domain_rand.cube_mass.enable"/"...min" and it
        # lands on config.domain_rand.*. Values pass through verbatim (bool/float).
        for k, v in cfg.items():
            if k.startswith("domain_rand."):
                env_overrides[k] = v

        env = registry.load(env_name, config_overrides=env_overrides)
        env_cfg = registry.get_default_config(env_name)
        episode_length = int(
            cfg.get("episode_length", getattr(env_cfg, "episode_length", 1000))
        )
        
        
        # ─────────────────────────────────────────────
        # Domain Randomization (optional)
        # ─────────────────────────────────────────────
        dr_cfg = cfg.get("domain_randomization", {})
        dr_enabled = dr_cfg.get("enabled", False)
        domain_randomize_fn = None

        if dr_enabled:
            print("\n[DR] ═══ Domain Randomization ENABLED ═══", flush=True)
            try:
                mj_model = _get_mj_model(env)
                box_info = _find_box_ids(
                    mj_model,
                    box_body_name=dr_cfg.get("box_body_name", "box"),
                    box_geom_name=dr_cfg.get("box_geom_name", "box"),
                )

                dr_mass_range = tuple(dr_cfg.get("mass_range", [0.5, 2.0]))
                dr_fric_range = tuple(dr_cfg.get("friction_range", [0.5, 2.0]))
                dr_size_range = dr_cfg.get("size_range", None)
                if dr_size_range is not None:
                    dr_size_range = tuple(dr_size_range)

                domain_randomize_fn = make_domain_randomize_fn(
                    box_info,
                    mass_range=dr_mass_range,
                    friction_range=dr_fric_range,
                    size_range=dr_size_range,
                )

                wandb.config.update(
                    {"dr": {
                        "enabled": True,
                        "box_body": box_info["box_body_name"],
                        "box_geom": box_info["box_geom_name"],
                        "nominal_mass": box_info["nominal_mass"],
                        "mass_range": list(dr_mass_range),
                        "friction_range": list(dr_fric_range),
                        "size_range": list(dr_size_range) if dr_size_range else None,
                    }},
                    allow_val_change=True,
                )
            except Exception as e:
                print(f"[DR] ⚠ Setup failed, training WITHOUT randomization: {e}",
                      flush=True)
                domain_randomize_fn = None
        else:
            print("\n[DR] Domain Randomization DISABLED", flush=True)

        # -----------------------------
        # PPO defaults -> dict
        # -----------------------------
        base = manipulation_params.brax_ppo_config(env_name)
        try:
            base_dict = base.to_dict()
        except AttributeError:
            base_dict = dict(base)

        ppo_params = dict(base_dict)

        print("\n Original ppo_params:", flush=True)
        print_dict_pairs(ppo_params)

        # -----------------------------
        # Apply cfg overrides ONCE
        # -----------------------------
        overrides = _extract_ppo_overrides(cfg)
        ppo_params = apply_validated_overrides(ppo_params, overrides, strict=True)

        # Cast types to schema (optional, but keep)
        schema = dict(base_dict)
        schema.pop("network_factory", None)
        ppo_params_overwrite = cast_to_schema(ppo_params, schema)

        print("\nppo_params_overwrite after overrides:", flush=True)
        print_dict_pairs(ppo_params_overwrite)
        print("type of ppo_param_overwrite: ", type(ppo_params_overwrite), flush=True)

        # Remove keys not accepted by ppo.train or passed explicitly
        for k in ["network_factory", "seed", "init_keyframe"]:
            ppo_params_overwrite.pop(k, None)


        # -----------------------------
        # Video / policy_params_fn
        # -----------------------------
        video_every_evals = int(cfg.get("video_every_evals", 10))
        render_every = int(cfg.get("render_every", 1))
        video_state = {"eval_idx": 0}
        total_timesteps = ppo_params_overwrite.get("num_timesteps", None)

        # Per-eval checkpointing. The final params train_fn returns can be a
        # post-peak DIP: UR3Pick runs learned to reach/grasp by ~5M then the
        # policy collapsed (reached_box -> 0) by ~6M, and only the final
        # (collapsed) params were saved. Snapshot every eval to disk (an
        # order-independent safety net) and keep the best-by-eval-reward params
        # in memory to publish alongside the final policy.
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        # The recorder pairs each eval's params with the reward measured on
        # THOSE params, matching on step rather than on callback arrival order.
        # The previous code read progress_fn's stashed reward from inside
        # policy_params_fn, but brax calls policy_params_fn FIRST within an
        # iteration (ppo/train.py:727 before :748), so every params tree was
        # scored with its predecessor's number and best_params.msgpack held the
        # params one eval PAST the peak.
        recorder = BestParamsRecorder(
            ckpt_dir=ckpt_dir,
            serialize=serialization.to_bytes,
        )

        def policy_params_fn(num_steps, make_policy, params):
            video_state["eval_idx"] += 1
            eval_idx = video_state["eval_idx"]

            # Snapshot this eval to disk and register the params for pairing.
            # The reward half arrives from progress_fn for the same step.
            recorder.on_policy_params(int(num_steps), params)

            is_last_eval = (total_timesteps is not None) and (int(num_steps) >= int(total_timesteps))
            should_record = (eval_idx % video_every_evals == 0) or is_last_eval

            if should_record:
                _rollout_and_log_video_from_make_policy(
                    num_steps=int(num_steps),
                    make_policy=make_policy,
                    params=params,
                    env_name=env_name,
                    run_id=base_run_id,
                    seed=seed,
                    episode_length=episode_length,
                    render_every=render_every,
                    camera_kwargs=camera_kwargs,
                    env_overrides=env_overrides,
                )

        # -----------------------------
        # Network factory resolution
        # -----------------------------
        nf_params = dict(ppo_params.get("network_factory") or {})
        if isinstance(nf_params.get("policy_hidden_layer_sizes"), list):
            nf_params["policy_hidden_layer_sizes"] = tuple(nf_params["policy_hidden_layer_sizes"])
        if isinstance(nf_params.get("value_hidden_layer_sizes"), list):
            nf_params["value_hidden_layer_sizes"] = tuple(nf_params["value_hidden_layer_sizes"])
        network_factory = ppo_networks.make_ppo_networks
        if isinstance(nf_params, dict) and len(nf_params) > 0:
            network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_params)
        
        print("\n ppo_params_overwrite before training", flush=True)
        print_dict_pairs(nf_params)
        print_dict_pairs(ppo_params_overwrite)
        print("========== Training ppo.train ==============")
        # -----------------------------
        # Train (PASS ONLY FINAL KWARGS)
        # -----------------------------
        dr_kwargs = {}
        if domain_randomize_fn is not None:
            dr_kwargs["randomization_fn"] = domain_randomize_fn
            print("[DR] randomization_fn passed to ppo.train ✓", flush=True)

        train_fn = functools.partial(
            ppo.train,
            **ppo_params_overwrite,
            **dr_kwargs,
            network_factory=network_factory,
            progress_fn=_make_progress_wandb(
                on_eval=lambda _s, _m: recorder.on_progress(
                    _s, _m.get("eval/episode_reward")
                )
            ),
            policy_params_fn=policy_params_fn,
            seed=seed,
        )


        make_inference_fn, params, final_metrics = train_fn(
            environment=env, wrap_env_fn=wrapper.wrap_for_brax_training
        )
        # Log final config once
        print("\n ppo_params_overwrite before training", flush=True)
        print_dict_pairs(ppo_params_overwrite)
        print_dict_pairs(dr_kwargs, prefix="[DR] ")

        # Get git hash and build the inference config (everything needed to
        # reconstruct the policy at deploy time).
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]  # First 8 chars
        except Exception:
            git_hash = "unknown"

        wrapped_env = wrapper.wrap_for_brax_training(env)
        inference_cfg = {
            "env_name": env_name,
            "episode_length": episode_length,
            "observation_size": int(wrapped_env.observation_size),
            "action_size": int(wrapped_env.action_size),
            "network_factory": _wb_jsonify(nf_params),
            "seed": seed,
            "env_overrides": env_overrides,           # init_keyframe, custom params
            "domain_randomization": cfg.get("domain_randomization", {}),
            "git_commit": git_hash,     # Which code version?
            "training_timestamp": str(datetime.now()),
            "full_cfg": _wb_jsonify(cfg),             # All training config
        }
        if "eval/episode_reward" in final_metrics:
            wandb.run.summary["final_eval_return"] = float(final_metrics["eval/episode_reward"])

        # Publish the policy under trained_policy/ in the run's Files tab —
        # browsable in the W&B UI and read by the policy downloader. Written
        # directly from the in-memory variables (params, final_metrics,
        # inference_cfg); no W&B artifact.
        tp_dir = os.path.join(out_dir, "trained_policy")
        os.makedirs(tp_dir, exist_ok=True)

        with open(os.path.join(tp_dir, "params.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(params))
        with open(os.path.join(tp_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
                f,
                indent=2,
            )
        with open(os.path.join(tp_dir, "inference_config.json"), "w", encoding="utf-8") as f:
            json.dump(inference_cfg, f, indent=2)

        for name in ("params.msgpack", "final_metrics.json", "inference_config.json"):
            wandb.save(os.path.join(tp_dir, name), base_path=out_dir)

        # Also publish the BEST-by-eval-reward checkpoint next to the final one.
        # Guards against a post-peak collapse (the final params.msgpack can be
        # worse than a mid-run policy). The deploy loader still defaults to
        # params.msgpack; point it at best_params.msgpack to deploy the peak.
        if recorder.best.get("params") is not None:
            with open(os.path.join(tp_dir, "best_params.msgpack"), "wb") as f:
                f.write(serialization.to_bytes(recorder.best["params"]))
            with open(os.path.join(tp_dir, "best_info.json"), "w", encoding="utf-8") as f:
                json.dump(recorder.best_info(), f, indent=2)
            for name in ("best_params.msgpack", "best_info.json"):
                wandb.save(os.path.join(tp_dir, name), base_path=out_dir)
            print(
                f"[ckpt] best policy: step={recorder.best['step']} "
                f"eval_reward={recorder.best['reward']:.1f} -> trained_policy/best_params.msgpack",
                flush=True,
            )

        result = {
            "params": params,
            "best_params": recorder.best.get("params"),
            "best_reward": recorder.best.get("reward"),
            "best_step": recorder.best.get("step"),
            "final_metrics": final_metrics,
            "best_info": recorder.best_info(),
            "steps_completed": int(total_timesteps) if total_timesteps else None,
            "stopped_early": False,
            "wandb_run_id": run_id_tag,
            "out_dir": out_dir,
            "observation_size": int(inference_cfg["observation_size"]),
            "action_size": int(inference_cfg["action_size"]),
            "network_factory": inference_cfg["network_factory"],
        }

    finally:
        run.finish()

    # Returned AFTER the finally block so the wandb run is closed first.
    # Additive: every existing caller (batch_runs/scripts/run_one_*.py,
    # robots/UR10e/UR10_ppo.py) ignores the return value.
    return result
