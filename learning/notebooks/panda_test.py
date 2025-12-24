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
import wandb
import mediapy


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
    seed: int,
    episode_length: int,
    render_every: int,
    camera_kwargs: Dict[str, Any],
    step_tag: str,
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
    trajectory = rollout[:: int(render_every)]
    frames = eval_env.render(trajectory, **(camera_kwargs or {}))
    fps = float(1.0 / eval_env.dt) / float(render_every)
    video_path = os.path.join(wandb.run.dir, f"{env_name}_{step_tag}.mp4")
    mediapy.write_video(video_path, frames, fps=fps)
    wandb.log(
        {"eval/video": wandb.Video(video_path, fps=int(fps), format="mp4")},
        step=int(num_steps),
    )


def run_experiment(cfg: Dict[str, Any], out_dir: str) -> None:
    _setup_mujoco_backend()
    env_name = str(cfg.get("env_name", "PandaPickCube"))
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    wandb_mode = str(cfg.get("wandb_mode", "online"))
    run = wandb.init(
        project=str(cfg.get("wandb_project", "panda_pick_ppo")),
        name=str(cfg.get("run_id", "")) or None,
        group=str(cfg.get("wandb_group", "")) or None,
        tags=cfg.get("wandb_tags", None),
        config=cfg,
        mode=wandb_mode,
        dir=out_dir,
    )
    try:
        print("JAX devices:", jax.devices(), flush=True)
        env = registry.load(env_name)
        env_cfg = registry.get_default_config(env_name)
        base = manipulation_params.brax_ppo_config(env_name)
        try:
            ppo_params = base.to_dict()
        except AttributeError:
            ppo_params = dict(base)
        overrides = _extract_ppo_overrides(cfg)
        _deep_update(ppo_params, overrides)
        if "learning_rate" in ppo_params:
            ppo_params["learning_rate"] = float(ppo_params["learning_rate"])
        if "batch_size" in ppo_params:
            ppo_params["batch_size"] = int(ppo_params["batch_size"])
        if "num_envs" in ppo_params:
            ppo_params["num_envs"] = int(ppo_params["num_envs"])
        if "num_timesteps" in ppo_params:
            ppo_params["num_timesteps"] = int(ppo_params["num_timesteps"])
        if "unroll_length" in ppo_params:
            ppo_params["unroll_length"] = int(ppo_params["unroll_length"])
        if "num_minibatches" in ppo_params:
            ppo_params["num_minibatches"] = int(ppo_params["num_minibatches"])
        if "num_updates_per_batch" in ppo_params:
            ppo_params["num_updates_per_batch"] = int(
                ppo_params["num_updates_per_batch"]
            )
        if "episode_length" in ppo_params:
            ppo_params["episode_length"] = int(ppo_params["episode_length"])
        if "discounting" in ppo_params:
            ppo_params["discounting"] = float(ppo_params["discounting"])
        if "entropy_cost" in ppo_params:
            ppo_params["entropy_cost"] = float(ppo_params["entropy_cost"])
        video_every_evals = int(cfg.get("video_every_evals", 5))
        render_every = int(cfg.get("render_every", 1))
        video_tag = str(cfg.get("video_tag", "eval"))
        camera_kwargs = cfg.get("camera_kwargs", {}) or {}
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
                    seed=seed,
                    episode_length=episode_length,
                    render_every=render_every,
                    camera_kwargs=camera_kwargs,
                    step_tag=f"{video_tag}{eval_idx}_steps{int(num_steps)}",
                )

        ppo_training_params = dict(ppo_params)
        # REMOVE parameters we pass explicitly below
        for k in ["seed", "progress_fn", "policy_params_fn", "network_factory"]:
            ppo_training_params.pop(k, None)

        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in ppo_params and isinstance(
            ppo_params["network_factory"], dict
        ):
            nf_cfg = dict(ppo_params["network_factory"])
            if "policy_hidden_layer_sizes" in nf_cfg and isinstance(
                nf_cfg["policy_hidden_layer_sizes"], list
            ):
                nf_cfg["policy_hidden_layer_sizes"] = tuple(
                    nf_cfg["policy_hidden_layer_sizes"]
                )
            if "value_hidden_layer_sizes" in nf_cfg and isinstance(
                nf_cfg["value_hidden_layer_sizes"], list
            ):
                nf_cfg["value_hidden_layer_sizes"] = tuple(
                    nf_cfg["value_hidden_layer_sizes"]
                )
            del ppo_training_params["network_factory"]
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **nf_cfg
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
