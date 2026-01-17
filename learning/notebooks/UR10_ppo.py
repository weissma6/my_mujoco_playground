"""
This file loads all the mujoco and checks for cuda of appe
"""

import os
import platform
import subprocess
import json
import functools
import random
from datetime import datetime
from typing import Any, Dict
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
import time


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
        "num_eval_envs"

    }
    overrides: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in reserved:
            continue
        overrides[k] = v
    if "network_factory" in cfg and isinstance(cfg["network_factory"], dict):
        overrides["network_factory"] = cfg["network_factory"]
    return overrides


def _make_progress_wandb():
    def progress_wandb(num_steps, metrics):
        log_dict = {"training/num_steps": int(num_steps)}
        for k, v in metrics.items():
            try:
                log_dict[k] = float(v)
            except Exception:
                pass
        wandb.log(log_dict, step=int(num_steps))

    return progress_wandb


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
):
    import jax.numpy as jnp

    eval_env = registry.load(env_name)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    policy = jax.jit(make_policy(params, deterministic=True))
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    rollout = [state]
    ep_reward = 0.0

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

        ep_reward += float(jnp.asarray(state.reward))

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

    return ep_reward

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


def run_experiment(cfg: Dict[str, Any], out_dir: str) -> None:
    _setup_mujoco_backend()

    env_name = str(cfg.get("env_name", "UR10PickCube"))
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
        project=str(cfg.get("wandb_project", "UR10_pick_ppo")),
        name=run_id_tag,
        id=run_id_tag,
        group=str(cfg.get("wandb_group", "")) or None,
        tags=cfg.get("wandb_tags", None),
        resume="never",
        config=cfg,
        mode=wandb_mode,
        dir=out_dir,
    )

    try:
        # -----------------------------
        # Env overrides + env creation
        # -----------------------------
        env_overrides = {}
        if "init_keyframe" in cfg:
            env_overrides["init_keyframe"] = cfg["init_keyframe"]

        env = registry.load(env_name, config_overrides=env_overrides)
        env_cfg = registry.get_default_config(env_name)
        episode_length = int(getattr(env_cfg, "episode_length", 1000))

        # -----------------------------
        # PPO defaults -> dict
        # -----------------------------
        base = manipulation_params.brax_ppo_config(env_name)
        try:
            base_dict = base.to_dict()
        except AttributeError:
            base_dict = dict(base)

        ppo_params = dict(base_dict)

        print("\n[DBG PPO DEFAULTS] key training controls:", flush=True)
        for k in ["num_timesteps", "num_envs", "unroll_length", "num_evals", "num_updates_per_batch"]:
            if k in ppo_params:
                print(f"[DBG PPO DEFAULTS] {k:>22} = {ppo_params[k]!r}", flush=True)

        # -----------------------------
        # Apply cfg overrides ONCE
        # -----------------------------
        overrides = _extract_ppo_overrides(cfg)
        ppo_params = apply_validated_overrides(ppo_params, overrides, strict=True)

        # Cast types to schema (optional, but keep)
        schema = dict(base_dict)
        schema.pop("network_factory", None)
        ppo_params = cast_to_schema(ppo_params, schema)

        # -----------------------------
        # FINAL kwargs to ppo.train
        # -----------------------------
        ppo_train_kwargs = dict(ppo_params)

        # Remove keys not accepted by ppo.train or passed explicitly
        for k in ["network_factory", "seed", "init_keyframe"]:
            ppo_train_kwargs.pop(k, None)

        print("\n[FINAL->ppo.train] key training controls:", flush=True)
        for k in ["num_timesteps", "num_envs", "unroll_length", "num_evals", "num_updates_per_batch"]:
            if k in ppo_train_kwargs:
                print(f"[FINAL->ppo.train] {k:>22} = {ppo_train_kwargs[k]!r}", flush=True)

        # Derived
        nt = int(ppo_train_kwargs.get("num_timesteps", -1))
        ne = int(ppo_train_kwargs.get("num_envs", -1))
        ul = int(ppo_train_kwargs.get("unroll_length", -1))
        bs = ne * ul if ne > 0 and ul > 0 else -1
        if bs > 0 and nt > 0:
            print(f"[DBG DERIVED] batch_size_steps = {ne}*{ul} = {bs}", flush=True)
            print(f"[DBG DERIVED] expected_updates_floor = max(1, {nt}//{bs}) = {max(1, nt // bs)}", flush=True)

        # DEBUG SAFETY: if asking for fewer steps than one batch, make eval sane
        # (This avoids training being driven by eval scheduling weirdness in some implementations)
        if bs > 0 and nt > 0 and nt < bs:
            print("[DBG] num_timesteps < one batch; forcing num_evals=1 for debug.", flush=True)
            ppo_train_kwargs["num_evals"] = 1

        # -----------------------------
        # Video / policy_params_fn
        # -----------------------------
        video_every_evals = int(cfg.get("video_every_evals", 10))
        render_every = int(cfg.get("render_every", 1))
        video_state = {"eval_idx": 0}
        total_timesteps = ppo_train_kwargs.get("num_timesteps", None)

        def policy_params_fn(num_steps, make_policy, params):
            video_state["eval_idx"] += 1
            eval_idx = video_state["eval_idx"]

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
                )

        # -----------------------------
        # Network factory resolution
        # -----------------------------
        nf_cfg = dict(ppo_params.get("network_factory") or {})
        if isinstance(nf_cfg.get("policy_hidden_layer_sizes"), list):
            nf_cfg["policy_hidden_layer_sizes"] = tuple(nf_cfg["policy_hidden_layer_sizes"])
        if isinstance(nf_cfg.get("value_hidden_layer_sizes"), list):
            nf_cfg["value_hidden_layer_sizes"] = tuple(nf_cfg["value_hidden_layer_sizes"])

        network_factory = ppo_networks.make_ppo_networks
        if isinstance(nf_cfg, dict) and len(nf_cfg) > 0:
            network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_cfg)

        # Best-effort param count (do not crash)
        try:
            networks = network_factory(observation_size=env.observation_size, action_size=env.action_size)
            rng = jax.random.PRNGKey(seed)
            dummy_obs = jnp.zeros((env.observation_size,), dtype=jnp.float32)

            # Depending on brax version, init signatures differ. Keep guarded.
            pi_params = networks.policy_network.init(rng, dummy_obs)  # may fail; ok
            v_params = networks.value_network.init(rng, dummy_obs)

            def _count_params(pytree) -> int:
                return sum(x.size for x in tree_leaves(pytree))

            wandb.run.summary["net/policy_num_params"] = int(_count_params(pi_params))
            wandb.run.summary["net/value_num_params"]  = int(_count_params(v_params))
            wandb.run.summary["net/total_num_params"]  = int(_count_params(pi_params) + _count_params(v_params))
        except Exception as e:
            print(f"[WARN] Could not compute network param counts: {e}", flush=True)

        # Log final config once
        _wb_log_final_train_config(
            ppo_training_params=ppo_train_kwargs,
            nf_cfg=nf_cfg if isinstance(nf_cfg, dict) else None,
            env_name=env_name,
            seed=seed,
        )

        # -----------------------------
        # Train (PASS ONLY FINAL KWARGS)
        # -----------------------------
        train_fn = functools.partial(
            ppo.train,
            **ppo_train_kwargs,
            network_factory=network_factory,
            progress_fn=_make_progress_wandb(),
            policy_params_fn=policy_params_fn,
            seed=seed,
        )

        make_inference_fn, params, final_metrics = train_fn(
            environment=env, wrap_env_fn=wrapper.wrap_for_brax_training
        )
        # Prepare output paths once
        params_path = os.path.join(out_dir, "params.msgpack")
        metrics_path = os.path.join(out_dir, "final_metrics.json")

        # Save params (JAX/Flax-safe)
        with open(params_path, "wb") as f:
            f.write(serialization.to_bytes(params))

        # Save metrics
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
                f,
                indent=2,
            )

        if "eval/episode_reward" in final_metrics:
            wandb.run.summary["final_eval_return"] = float(final_metrics["eval/episode_reward"])

        # Log everything as a single artifact
        artifact = wandb.Artifact(f"policy_checkpoint_{wandb.run.id}", type="model")
        artifact.add_file(params_path)
        artifact.add_file(metrics_path)
        wandb.log_artifact(artifact)

    finally:
        run.finish()
