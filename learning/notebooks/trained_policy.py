# trained_policy.py
# Notebook-friendly utilities to load and run a Brax / JAX policy checkpoint locally.
# /Users/matthiasweiss/Desktop/ZHAW/MSE/3_VT1_Model Based RL/My_Mujoco/my_mujoco_playground/evaluation/trained_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class LoadedPolicy:
    """Callable policy + optional metadata."""
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray]  # (params, obs)-> action
    params: Any
    obs_normalizer: Optional[Any] = None  # e.g., running mean/var, if you have one
    metadata: Optional[Dict[str, Any]] = None


# -----------------------
# Loading helpers
# -----------------------

def load_orbax_checkpoint(ckpt_dir: str) -> Dict[str, Any]:
    """
    Loads a PyTree checkpoint saved with Orbax.
    Expects a directory containing Orbax checkpoint files.
    Returns the restored pytree (often includes params, normalizer state, etc.).
    """
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_dir)
    return restored


def load_flax_bytes(path: str) -> Any:
    """
    Loads a Flax-serialized pytree saved via flax.serialization.to_bytes(...).
    You must know the target pytree structure to deserialize properly.
    For many simple cases, you saved a dict and can deserialize into an empty dict.
    """
    from flax import serialization

    with open(path, "rb") as f:
        raw = f.read()

    # If you know the structure, replace {} with that structure.
    restored = serialization.from_bytes({}, raw)
    return restored


def load_pickle(path: str) -> Any:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------
# Policy construction
# -----------------------

def make_policy_from_apply_fn(
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    params: Any,
    obs_normalizer: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> LoadedPolicy:
    return LoadedPolicy(
        apply_fn=apply_fn,
        params=params,
        obs_normalizer=obs_normalizer,
        metadata=metadata or {},
    )


def _maybe_normalize_obs(policy: LoadedPolicy, obs: jnp.ndarray) -> jnp.ndarray:
    """
    If you used observation normalization during training, you MUST apply the same
    transform at inference. This function is a placeholder.

    Adapt this to your normalizer object:
    - If you used Brax running statistics, you might have (mean, var) and epsilon.
    """
    if policy.obs_normalizer is None:
        return obs

    norm = policy.obs_normalizer

    # Example patterns (edit to match your saved structure):
    # 1) dict with mean/var:
    if isinstance(norm, dict) and "mean" in norm and "var" in norm:
        mean = jnp.asarray(norm["mean"])
        var = jnp.asarray(norm["var"])
        eps = float(norm.get("eps", 1e-8))
        return (obs - mean) / jnp.sqrt(var + eps)

    # 2) callable:
    if callable(norm):
        return norm(obs)

    # Fallback: return as-is
    return obs


def jit_policy(policy: LoadedPolicy) -> LoadedPolicy:
    """
    JIT compile the policy apply step for fast inference.
    """
    @jax.jit
    def act(params: Any, obs: jnp.ndarray) -> jnp.ndarray:
        obs_n = _maybe_normalize_obs(policy, obs)
        return policy.apply_fn(params, obs_n)

    return LoadedPolicy(
        apply_fn=act,
        params=policy.params,
        obs_normalizer=policy.obs_normalizer,
        metadata=policy.metadata,
    )


# -----------------------
# Rollout runner
# -----------------------

def rollout(
    env,                         # Brax environment (or compatible)
    policy: LoadedPolicy,
    seed: int = 0,
    horizon: int = 1000,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Rollout a policy in a Brax env locally.
    Works with env.reset(rng) and env.step(state, action).
    Returns a dict with trajectory data.

    NOTE:
    - If your policy outputs distribution params (mean, log_std) instead of actions,
      you must adapt the action sampling below.
    """

    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)

    state = env.reset(reset_key)

    def step_fn(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)

        obs = state.obs  # Brax State typically has .obs
        act_out = policy.apply_fn(policy.params, obs)

        # If your network returns both mean/log_std, adapt here:
        # Example:
        # mean, log_std = act_out
        # if deterministic:
        #     action = mean
        # else:
        #     std = jnp.exp(log_std)
        #     action = mean + std * jax.random.normal(act_key, mean.shape)

        action = act_out
        if not deterministic:
            # Add small exploration noise if desired (optional)
            action = action + 0.0 * jax.random.normal(act_key, action.shape)

        next_state = env.step(state, action)
        transition = (state, action, next_state.reward, next_state.done)
        return (next_state, rng), transition

    (final_state, _), traj = jax.lax.scan(step_fn, (state, rng), xs=None, length=horizon)

    states, actions, rewards, dones = traj

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "final_state": final_state,
        "return": jnp.sum(rewards),
    }

