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
    run_id = str(cfg.get("run_id", "run"))
    camera_kwargs = cfg.get("camera_kwargs") or {"camera": "side_130"}
    # Resolve seed: sweep overrides, otherwise random-by-default
    cfg_seed = cfg.get("seed", None)
    if cfg_seed is None:
        seed = random.getrandbits(32)   # random default, good entropy range
    else:
        seed = int(cfg_seed)
    random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    wandb_mode = str(cfg.get("wandb_mode", "online"))
    run = wandb.init(
        project=str(cfg.get("wandb_project", "UR10_pick_ppo")),
        name=str(cfg.get("run_id", "")) or None,
        group=str(cfg.get("wandb_group", "")) or None,
        tags=cfg.get("wandb_tags", None),
        config=cfg,
        mode=wandb_mode,
        dir=out_dir,
    )
    try:
        #print("JAX devices:", jax.devices(), flush=True)
        # Build env overrides from cfg (env-only params)
        env_overrides = {}
        if "init_keyframe" in cfg:
            env_overrides["init_keyframe"] = cfg["init_keyframe"]

        # Create env with overrides
        env = registry.load(env_name, config_overrides=env_overrides)
        env_cfg = registry.get_default_config(env_name)
        
        base = manipulation_params.brax_ppo_config(env_name)
        try:
            ppo_params = base.to_dict()
            base_dict = base.to_dict()
        except AttributeError:
            ppo_params = dict(base)
            base_dict = dict(base)
        # ====================================================================
        # Apply overrides from cfg to ppo_params
        # ====================================================================
        overrides = _extract_ppo_overrides(cfg)
        ppo_params = apply_validated_overrides(ppo_params, overrides, strict=False)
        # cast the original types of base on ppo_params
        schema = dict(base_dict)
        schema.pop("network_factory", None)
        ppo_params = cast_to_schema(ppo_params, schema)
        # ====================================================================

        video_every_evals = int(cfg.get("video_every_evals", 10))           # Log video every N evals
        render_every = int(cfg.get("render_every", 1))
        video_tag = str(cfg.get("video_tag", "eval"))

        episode_length = int(env_cfg.episode_length)
        video_state = {"eval_idx": 0}

        def policy_params_fn(num_steps, make_policy, params):
            video_state["eval_idx"] += 1
            eval_idx = video_state["eval_idx"]
            total_timesteps = ppo_params.get("num_timesteps", None)
            is_last_eval = total_timesteps is not None and int(num_steps) >= int(
                total_timesteps
            )
            should_record = (eval_idx % video_every_evals == 0) or is_last_eval
            if should_record:
                _rollout_and_log_video_from_make_policy(
                    num_steps=int(num_steps),
                    make_policy=make_policy,
                    params=params,
                    env_name=env_name,
                    run_id=run_id,
                    seed=seed,
                    episode_length=episode_length,
                    render_every=render_every,
                    camera_kwargs=camera_kwargs,
                )

        # Build training kwargs for ppo.train
        ppo_training_params = dict(ppo_params)

        # Only remove keys that truly are not kwargs for ppo.train OR that you pass explicitly
        for k in ["network_factory", "seed", "init_keyframe"]:
            ppo_training_params.pop(k, None)

        # Resolve network config
        nf_cfg = dict(ppo_params.get("network_factory") or {})

        network_factory = ppo_networks.make_ppo_networks
        if isinstance(nf_cfg, dict):
            if isinstance(nf_cfg.get("policy_hidden_layer_sizes"), list):
                nf_cfg["policy_hidden_layer_sizes"] = tuple(nf_cfg["policy_hidden_layer_sizes"])
            if isinstance(nf_cfg.get("value_hidden_layer_sizes"), list):
                nf_cfg["value_hidden_layer_sizes"] = tuple(nf_cfg["value_hidden_layer_sizes"])
            network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_cfg)

        def _count_params(pytree) -> int:
            return sum(x.size for x in tree_leaves(pytree))

        # Try to instantiate & count params (best-effort; do not crash training if API differs)
        try:
            # Brax make_ppo_networks usually takes sizes as keyword args
            networks = network_factory(
                observation_size=env.observation_size,
                action_size=env.action_size,
            )

            # Common Brax pattern: networks has policy_network and value_network
            rng = jax.random.PRNGKey(seed)
            rng_pi, rng_v = jax.random.split(rng)

            # Dummy observation is required for init
            dummy_obs = jnp.zeros((env.observation_size,), dtype=jnp.float32)

            pi_params = networks.policy_network.init(rng_pi, dummy_obs)
            v_params  = networks.value_network.init(rng_v, dummy_obs)

            wandb.run.summary["net/policy_num_params"] = int(_count_params(pi_params))
            wandb.run.summary["net/value_num_params"]  = int(_count_params(v_params))
            wandb.run.summary["net/total_num_params"]  = int(_count_params(pi_params) + _count_params(v_params))
        except Exception as e:
            print(f"[WARN] Could not compute network param counts: {e}", flush=True)

        # Log FINAL training config + FINAL network spec to W&B (once)
        _wb_log_final_train_config(
            ppo_training_params=ppo_training_params,
            nf_cfg=nf_cfg if isinstance(nf_cfg, dict) else None,
            env_name=env_name,
            seed=seed,
        )
        train_fn = functools.partial(
            ppo.train,
            **ppo_training_params,
            network_factory=network_factory,
            progress_fn=_make_progress_wandb(),
            policy_params_fn=policy_params_fn,
            seed=seed,
        )

        make_inference_fn, params, final_metrics = train_fn(
            environment=env, wrap_env_fn=wrapper.wrap_for_brax_training
        )
        with open(
            os.path.join(out_dir, "final_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    k: float(v)
                    for k, v in final_metrics.items()
                    if isinstance(v, (int, float))
                },
                f,
                indent=2,
            )
        if "eval/episode_reward" in final_metrics:
            wandb.run.summary["final_eval_return"] = float(
                final_metrics["eval/episode_reward"]
            )
    finally:
        run.finish()

