"""
This file loads all the mujoco and checks for cuda of appe
"""

import os
import platform
import subprocess
import sys
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union
from mujoco_playground.config import manipulation_params


# -----------------------------------------------------------------------------
# Environment / backend selection
# -----------------------------------------------------------------------------
def is_nvidia_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


system = platform.system()

if "MUJOCO_GL" in os.environ:
    # Respect what the user / container set
    print(
        f"[INFO] MUJOCO_GL already set to {os.environ['MUJOCO_GL']!r}, not overriding."
    )
else:
    if system == "Linux" and is_nvidia_available():
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] Detected NVIDIA GPU on Linux – using MUJOCO_GL=egl")
    elif system == "Darwin":
        os.environ["MUJOCO_GL"] = "glfw"
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        print("[INFO] Running on macOS – forcing CPU backend (glfw)")
    else:
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        print(f"[INFO] No GPU / unknown system – using MUJOCO_GL=osmesa")

# -----------------------------------------------------------------------------
# (rest of your imports remain as they were)
# -----------------------------------------------------------------------------
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax

print("JAX devices:", jax.devices(), flush=True)
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import pickle
from mujoco_playground import wrapper
from mujoco_playground import registry
from copy import deepcopy
import wandb
import imageio
import mediapy


# -----------------------------------------------------------------------------
# 1) Basic config
# -----------------------------------------------------------------------------
env_name = "PandaPickCube"

# Base PPO config from Mujoco Playground
base_ppo_params = manipulation_params.brax_ppo_config(env_name)

# Convert to plain dict for easier manipulation / W&B
try:
    ppo_params = base_ppo_params.to_dict()
except AttributeError:
    ppo_params = dict(base_ppo_params)

# Optional: override something for quick local runs:
# ppo_params["num_timesteps"] = int(100_000)

# Seed (store in W&B for reproducibility
seed = 1  # = int(np.random.randint(0, 2**31 - 1))

# -----------------------------------------------------------------------------
# 2) Init env and W&B run
# -----------------------------------------------------------------------------
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

# Flatten config for W&B
wandb_cfg = {
    "env_name": env_name,
    "algo": "PPO",
    "seed": seed,
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # e.g. 20251211_1037
run_name = f"PandaPickCube_{wandb_cfg['algo']}_{timestamp}"

for k, v in ppo_params.items():
    if isinstance(v, (int, float, str, bool)):
        wandb_cfg[f"ppo/{k}"] = v

run = wandb.init(
    project="panda_pick_ppo",  # <-- change to your project name
    name=run_name,
    config=wandb_cfg,
)


# -----------------------------------------------------------------------------
# 3) Progress function: log metrics to W&B (no plotting)
# -----------------------------------------------------------------------------
def progress_wandb(num_steps, metrics):
    """
    Called periodically during PPO training.
    Logs scalar metrics to W&B.
    """
    print(
        "WANDB CALLBACK:", num_steps, "Episode Reward", metrics["eval/episode_reward"]
    )
    log_dict = {"training/num_steps": int(num_steps)}
    for k, v in metrics.items():
        try:
            log_dict[k] = float(v)
        except Exception:
            # Skip non-scalars / weird values
            pass

    wandb.log(log_dict, step=int(num_steps))


# -----------------------------------------------------------------------------
# 4) Rollout + video logging helper
# -----------------------------------------------------------------------------
def rollout_and_log_video(
    num_steps,
    make_policy,
    params,
    env_name: str,
    seed: int,
    video_length: int,
    fps: int,
    camera_kwargs: dict,
    step_tag: str,
):
    """
    Runs a rollout with the current policy, renders it via MuJoCo Playground's
    MjxEnv.render(trajectory, ...) and logs the video to W&B.
    """
    eval_env = registry.load(env_name)

    # Main RNG -> split into reset + policy sampling streams
    rng = jax.random.PRNGKey(seed)
    rng, reset_key, policy_key = jax.random.split(rng, 3)

    # IMPORTANT: reset the *eval_env* (not the training env)
    state = eval_env.reset(reset_key)
    policy = make_policy(params)
    trajectory = []

    for t in range(video_length):
        trajectory.append(state)
        obs = state.obs
        # Add batch dim (policy expects batched inputs)
        if isinstance(obs, dict):
            batched_obs = jax.tree_util.tree_map(lambda x: x[jp.newaxis, ...], obs)
        else:
            batched_obs = obs[jp.newaxis, ...]

        # Sample action with RNG key
        policy_key, key_sample = jax.random.split(policy_key)
        action = policy(batched_obs, key_sample)

        # Remove batch dim
        action = action[0]
        state = eval_env.step(state, action)

    # Render the full trajectory in one call (required by MjxEnv.render)
    frames = eval_env.render(trajectory, **camera_kwargs)

    os.makedirs("wandb_videos", exist_ok=True)
    video_path = os.path.join("wandb_videos", f"{env_name}_{step_tag}.mp4")

    mediapy.write_video(video_path, frames, fps=fps)
    wandb.log(
        {"eval/video": wandb.Video(video_path, fps=fps, format="mp4")},
        step=int(num_steps),
    )

    print(f"[VIDEO] Logged evaluation video at step {num_steps} as {video_path}")


# -----------------------------------------------------------------------------
# 5) Hook into PPO internals: policy_params_fn for videos every N evals
# -----------------------------------------------------------------------------
VIDEO_EVERY_EVALS = 1  # <-- "every n episodes": every  eval callbacks

video_state = {
    "eval_idx": 0,
    "video_every_evals": VIDEO_EVERY_EVALS,
    "env_name": env_name,
    "seed": seed,
}


def policy_params_wandb(num_steps, make_policy, params):
    """
    Called by PPO with latest policy params at evaluation checkpoints.
    We use this to periodically log rollout videos.
    """
    # Track how many times we've been called (i.e., how many evals have happened)
    video_state["eval_idx"] += 1
    eval_idx = video_state["eval_idx"]

    # Store latest training step in W&B summary
    wandb.run.summary["latest_num_steps"] = int(num_steps)
    wandb.run.summary["latest_eval_idx"] = int(eval_idx)

    total_timesteps = ppo_params.get("num_timesteps", None)
    is_last_eval = total_timesteps is not None and int(num_steps) >= int(
        total_timesteps
    )
    should_record = (eval_idx % video_state["video_every_evals"] == 0) or is_last_eval

    if should_record:
        rollout_and_log_video(
            num_steps=num_steps,
            make_policy=make_policy,
            params=params,
            env_name=video_state["env_name"],
            seed=video_state["seed"],
            video_length=env_cfg.episode_length,
            fps=int(1.0 / env.dt),
            camera_kwargs={},  # e.g. {"camera_id": 0} if you want a specific camera
            step_tag=f"eval{eval_idx}",
        )


# -----------------------------------------------------------------------------
# 6) Build network_factory and call PPO train
# -----------------------------------------------------------------------------
# --- build params + network_factory exactly like Panda ---
ppo_training_params = dict(ppo_params)

# ppo_params["num_timesteps"] = int(200_000)  # instead of millions
# ppo_params["num_envs"] = 16  # smaller parallelism
# ppo_params["unroll_length"] = 20  # shorter unrolls

network_factory = ppo_networks.make_ppo_networks
# Handle the optional network_factory config
if "network_factory" in ppo_params:
    # Remove it from the params dict so we don't pass it twice
    del ppo_training_params["network_factory"]

    # Build the partial network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params["network_factory"],
    )

# Build train_fn exactly like the Panda example
train_fn = functools.partial(
    ppo.train,
    **ppo_training_params,
    network_factory=network_factory,
    progress_fn=progress_wandb,  # your custom W&B logger
    policy_params_fn=policy_params_wandb,  # your video logger
    seed=seed,  # reproducibility
)

make_inference_fn, params, final_metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)

# -----------------------------------------------------------------------------
# 7) Save final model (seed, hyperparams, params) to W&B
# -----------------------------------------------------------------------------
import pickle

model_path = os.path.join(wandb.run.dir, "panda_test_final.pkl")
to_save = {
    "params": params,
    "seed": seed,
    "env_name": env_name,
    "ppo_config": ppo_params,
    "timestamp": datetime.now().isoformat(),
}
with open(model_path, "wb") as f:
    pickle.dump(to_save, f)

artifact = wandb.Artifact("panda_test_policy", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

# Log final eval return in W&B summary
if "eval/episode_reward" in final_metrics:
    wandb.run.summary["final_eval_return"] = float(final_metrics["eval/episode_reward"])

wandb.finish()
