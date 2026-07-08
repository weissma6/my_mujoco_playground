"""
Download a trained UR3 PPO policy from W&B, rebuild it, roll it out in the
matching MuJoCo Playground sim env, and render an mp4 of the rollout.

This is an OFFLINE check of a downloaded policy (no robot, no mocap): it reuses
policy_downloader.download_policy to fetch + unpack the policy into
evaluation/downloaded_policies/{run_id}/, then drives the registry env with the
loaded inference function and renders with env.render().

Usage:
    python evaluation/render_ur3_policy_rollout.py  
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import imageio.v2 as imageio
from flax import serialization

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "downloaded_policies"))
from policy_downloader import download_policy  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────
POLICY_REGISTRY = {
    "EP200_AS0.025_78fd8855a6eeb72d8ef1a0395cd6e4860db1439d": "EP200_AS0.025_d0.99_20260703_132735_2201",
    "NoVelocity_ea1ffd26f79c25db5c62af8e68022f6677b5aff6": "cur_light_lift8rot_20260701_163113_3867",
    "base90_lr4e-10": "base90_j10_fin25_20M_lr4e-4_20260626_101918_6311",
    "reasonable starting positions": "Reso_Pos_lr6e-4_20260626_110022_7585",
    "pick_12M_rand_base85_fin25": "Pick_12M_rand_base85_fin25_20260626_094554_6311",
    "pick_un20_env2048":          "Pick_un20_env2048_20260624_160023_6311",
    "pick_dr_medium":             "ur3pick_DR_MFR_medium_20260603_092056_5305",
}

POLICY_NAME = "EP200_AS0.025_78fd8855a6eeb72d8ef1a0395cd6e4860db1439d"  # <- select which policy to run

WANDB_ENTITY = "weissma6-zhaw-school-of-engineering"
WANDB_PROJECT = "UR3_pick_ppo"
WANDB_RUN_ID = POLICY_REGISTRY[POLICY_NAME]
CAMERA = "box_detail"
N_EPISODES = 1
# Random per-run seed (never 0 — seed 0 gave the same "far" target every render).
#SEED = int.from_bytes(os.urandom(4), "little") or 1
SEED = 9
HEIGHT, WIDTH = 480, 640

# ── 1. Resolve the policy (cache-aware: reuses downloaded_policies/{run_id}/
#       if present, only hits W&B on a cache miss) ──────────────────────────
print(f"Resolving policy {WANDB_RUN_ID}")
POLICY_DIR = download_policy(
    WANDB_RUN_ID,
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
)
VIDEO_OUT = os.path.join(POLICY_DIR, "rollout.mp4")

# ── 2. Rebuild the policy from the saved artifacts ────────────────────────
import json

with open(os.path.join(POLICY_DIR, "metadata.json")) as f:
    meta = json.load(f)
print(f"Loaded metadata: env={meta['env_name']} obs={meta['obs_dim']} "
      f"act={meta['action_dim']} nf={meta['network_factory']}")

nf_kwargs = {
    k: (tuple(v) if isinstance(v, list) else v)
    for k, v in meta["network_factory"].items()
}
ppo_net = ppo_networks.make_ppo_networks(
    observation_size=meta["obs_dim"],
    action_size=meta["action_dim"],
    preprocess_observations_fn=running_statistics.normalize,
    **nf_kwargs,
)
template = {
    "0": running_statistics.init_state(
        jax.ShapeDtypeStruct((meta["obs_dim"],), jnp.float32)
    ),
    "1": ppo_net.policy_network.init(jax.random.PRNGKey(0)),
    "2": ppo_net.value_network.init(jax.random.PRNGKey(0)),
}
with open(os.path.join(POLICY_DIR, "params.msgpack"), "rb") as f:
    params = serialization.from_bytes(template, f.read())
raw_policy = ppo_networks.make_inference_fn(ppo_net)(
    (params["0"], params["1"]), deterministic=True
)
inference_fn = jax.jit(raw_policy)

# ── 3. Roll out in the sim env ────────────────────────────────────────────
env_overrides = meta.get("env_overrides", {})
env = registry.load(meta["env_name"], config_overrides=env_overrides)
print(f"env_overrides applied: {env_overrides}")
episode_length = int(env._config.episode_length)
print(f"Rolling out {N_EPISODES} episode(s) x {episode_length} steps ...")

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

print(f"Using seed {SEED}")
rng = jax.random.PRNGKey(SEED)
rollout = []
rewards = []
for ep in range(N_EPISODES):
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    rollout.append(state)
    ep_reward = 0.0
    for _ in range(episode_length):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)
        ep_reward += float(state.reward)
    rewards.append(ep_reward)
    print(f"  episode {ep}: return={ep_reward:.2f}")

# ── 4. Render ─────────────────────────────────────────────────────────────
print(f"Rendering {len(rollout)} frames (camera={CAMERA}) ...")
frames = env.render(rollout, height=HEIGHT, width=WIDTH, camera=CAMERA)

fps = round(1.0 / env.dt)
writer = imageio.get_writer(VIDEO_OUT, fps=fps)
for frame in frames:
    writer.append_data(frame)
writer.close()
print(f"Saved {len(frames)} frames @ {fps} fps -> {VIDEO_OUT}")
