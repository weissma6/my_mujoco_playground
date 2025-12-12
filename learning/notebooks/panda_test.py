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
def final_video_rollout(
    env_name: str,
    env_cfg,
    make_inference_fn,
    params,
    seed: int,
    tag: str = "final",
    render_every: int = 2,
    camera_kwargs: dict | None = None,
    deterministic: bool = True,
):
    """
    Final evaluation rollout + render + W&B video logging.
    Video is ALWAYS written into the active W&B run directory:
      wandb/run-*/files/
    Matches the manipulation notebook rollout pattern.
    """

    if camera_kwargs is None:
        camera_kwargs = {}

    assert (
        wandb.run is not None
    ), "wandb.init() must be called before final_video_rollout()"

    # ------------------------------------------------------------------
    # Environment + JIT
    # ------------------------------------------------------------------
    env = registry.load(env_name)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    inference_fn = jax.jit(make_inference_fn(params, deterministic=deterministic))

    # ------------------------------------------------------------------
    # Rollout (same as manipulation notebook)
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    rollout = [state]

    for _ in range(int(env_cfg.episode_length)):
        rng, act_rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    # ------------------------------------------------------------------
    # Render trajectory
    # ------------------------------------------------------------------
    trajectory = rollout[:: int(render_every)]
    frames = env.render(trajectory, **camera_kwargs)

    # ------------------------------------------------------------------
    # Save video INTO W&B RUN DIRECTORY
    # ------------------------------------------------------------------
    run_dir = wandb.run.dir  # <-- this is the key difference
    video_path = os.path.join(run_dir, f"{env_name}_{tag}_seed{seed}.mp4")

    fps = float(1.0 / env.dt) / float(render_every)
    mediapy.write_video(video_path, frames, fps=fps)

    # ------------------------------------------------------------------
    # Log to W&B (media)
    # ------------------------------------------------------------------
    wandb.log(
        {
            "eval/final_video": wandb.Video(
                video_path,
                fps=int(fps),
                format="mp4",
            )
        },
        step=int(env_cfg.episode_length),
    )

    print(f"[FINAL VIDEO] saved & logged: {video_path}")

    return video_path


# -----------------------------------------------------------------------------
# 5) Hook into PPO internals: policy_params_fn for videos every N evals
# -----------------------------------------------------------------------------
VIDEO_EVERY_EVALS = 5  # every N eval callbacks
RENDER_EVERY = 1  # keep every Nth state for rendering (smaller/faster)
VIDEO_TAG = "eval"

video_state = {
    "eval_idx": 0,
    "video_every_evals": VIDEO_EVERY_EVALS,
    "env_name": env_name,
    "seed": seed,
}


def rollout_and_log_video_from_make_policy(
    num_steps: int,
    make_policy,
    params,
    env_name: str,
    seed: int,
    episode_length: int,
    render_every: int = 2,
    camera_kwargs: dict | None = None,
    step_tag: str = "eval",
):
    """
    SAFE MJX video rollout using make_policy(params, deterministic=True).

    - builds a fresh eval env
    - uses jit reset/step
    - builds a trajectory list of State
    - renders via env.render(trajectory, ...)
    - saves into wandb.run.dir (files/)
    - logs wandb.Video
    """
    import os
    import jax
    import jax.numpy as jnp
    import mediapy
    import wandb

    if camera_kwargs is None:
        camera_kwargs = {}

    assert wandb.run is not None, "wandb.init() must be called before logging video."

    eval_env = registry.load(env_name)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # IMPORTANT: deterministic=True to avoid needing key_sample / stochastic sampling
    policy = jax.jit(make_policy(params, deterministic=True))

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    # Debug once (now state exists)
    test_out = make_policy(params, deterministic=True)(
        state.obs, jax.random.PRNGKey(seed + 999)
    )
    print("[VIDEO DEBUG] policy output type:", type(test_out))

    rollout = [state]
    for _ in range(int(episode_length)):
        rng, act_rng = jax.random.split(rng)

        out = policy(state.obs, act_rng)

        # policy outputs can be action OR (action, extras) OR dict
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
    frames = eval_env.render(trajectory, **camera_kwargs)

    fps = float(1.0 / eval_env.dt) / float(render_every)

    # Save INTO W&B run dir so it ends up under run/files/
    video_path = os.path.join(wandb.run.dir, f"{env_name}_{step_tag}.mp4")
    mediapy.write_video(video_path, frames, fps=fps)

    wandb.log(
        {"eval/video": wandb.Video(video_path, fps=int(fps), format="mp4")},
        step=int(num_steps),
    )

    print(f"[VIDEO] Logged {video_path} at step={int(num_steps)}")
    return video_path


def policy_params_wandb(num_steps, make_policy, params):
    """
    Called by PPO at evaluation checkpoints.
    Logs rollout videos every N eval callbacks AND always at the last eval.
    """
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
        rollout_and_log_video_from_make_policy(
            num_steps=int(num_steps),
            make_policy=make_policy,
            params=params,
            env_name=video_state["env_name"],
            seed=int(video_state["seed"]),
            episode_length=int(env_cfg.episode_length),
            render_every=int(RENDER_EVERY),
            camera_kwargs={},  # optionally: {"height": 480, "width": 640, "camera": "front"}
            step_tag=f"{VIDEO_TAG}{eval_idx}_steps{int(num_steps)}",
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
