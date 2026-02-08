#!/usr/bin/env python3
"""
UR10 Robustness Evaluation — GPU-parallel with vmap
Usage: python run_robustness_eval.py
Runs 4 policies × 4 conditions, logs histogram grid + raw JSON to W&B.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from learning.notebooks.apple_mujoco_setup import *
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import wandb
import os
import json
import contextlib
import io
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from flax import serialization
from mujoco_playground import registry

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════
USER = "weissma6-zhaw-school-of-engineering"
PROJECT = "UR10_pick_ppo"
ENV_NAME = "UR10PickCube"

POLICY_RUNS = {
    "No DR":    "Damp=2_kp400kv1_DR_off_20260208_174610_5184",
    "DR Fric":  "Damp=2_kp400kv1_DR_FR_20260208_174619_4881",
    "DR Mass":  "Damp=2_kp400kv1_DR_M_20260208_174609_5582",
    "DR M+F":   "Damp=2_kp400kv1_DR_MFR_20260208_174610_1425",
}

TEST_CONDITIONS = {
    "Default":                               {"mass_range": None,       "friction_range": None},
    "Mass [1.9, 2.0]":                       {"mass_range": (1.9, 2.0), "friction_range": None},
    "Friction [0.5, 0.6]":                   {"mass_range": None,       "friction_range": (0.5, 0.6)},
    "Mass [1.9, 2.0] / Friction [0.5, 0.6]": {"mass_range": (1.9, 2.0), "friction_range": (0.5, 0.6)},
}

NUM_ROLLOUTS = 100
EPISODE_LENGTH = 100
SEED_BASE = 10
N_BINS = 5

# ═══════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════
api = wandb.Api()


def get_box_info(env):
    mj_model = None
    for attr in ("mj_model", "_mj_model", "model"):
        obj = getattr(env, attr, None)
        if obj is not None and hasattr(obj, "ngeom"):
            mj_model = obj
            break
    if mj_model is None:
        sys_obj = getattr(env, "sys", None)
        if sys_obj is not None:
            mj_model = getattr(sys_obj, "mj_model", None)
    if mj_model is None:
        raise RuntimeError("Cannot find mj_model on env")

    box_body_id, box_geom_id = -1, -1
    for name in ("box", "cube", "object", "target_object", "pick_object"):
        if box_body_id < 0:
            box_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if box_geom_id < 0:
            box_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)

    assert box_body_id >= 0, "Box body not found"
    assert box_geom_id >= 0, "Box geom not found"

    return {
        "box_body_id": box_body_id,
        "box_geom_id": box_geom_id,
        "nominal_mass": float(mj_model.body_mass[box_body_id]),
        "nominal_friction": mj_model.geom_friction[box_geom_id].copy(),
    }


def load_env_with_shift(env_name, box_info, mass_scale=1.0, friction_scale=1.0):
    with contextlib.redirect_stdout(io.StringIO()):
        env = registry.load(env_name)

    if mass_scale == 1.0 and friction_scale == 1.0:
        return env

    mj_model = None
    for attr in ("mj_model", "_mj_model", "model"):
        obj = getattr(env, attr, None)
        if obj is not None and hasattr(obj, "ngeom"):
            mj_model = obj
            break
    if mj_model is None:
        raise RuntimeError(f"Cannot find mj_model on {type(env).__name__}")

    bid = box_info["box_body_id"]
    gid = box_info["box_geom_id"]

    if mass_scale != 1.0:
        mj_model.body_mass[bid] = box_info["nominal_mass"] * mass_scale
    if friction_scale != 1.0:
        mj_model.geom_friction[gid][0] = box_info["nominal_friction"][0] * friction_scale

    new_mjx = mjx.put_model(mj_model)
    for attr in ("_mjx_model", "mjx_model", "sys", "_sys"):
        if hasattr(env, attr):
            setattr(env, attr, new_mjx)
            break

    return env


def load_policy_from_run(run_id, env):
    run = api.run(f"{USER}/{PROJECT}/{run_id}")
    nf_params = run.config.get("network_factory", {}) or {}

    policy_art = next((a for a in run.logged_artifacts() if a.type == "model"), None)
    if policy_art is None:
        raise ValueError(f"No model artifact for {run_id}")
    art_dir = policy_art.download(root="downloaded_policies")

    with open(os.path.join(art_dir, "params.msgpack"), "rb") as f:
        params_bytes = f.read()

    nf_kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in nf_params.items()}
    ppo_net = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **nf_kwargs,
    )

    rng = jax.random.PRNGKey(0)
    template = {
        "0": running_statistics.init_state(jax.ShapeDtypeStruct((env.observation_size,), jnp.float32)),
        "1": ppo_net.policy_network.init(rng),
        "2": ppo_net.value_network.init(rng),
    }
    params = serialization.from_bytes(template, params_bytes)
    make_policy = ppo_networks.make_inference_fn(ppo_net)
    return jax.jit(make_policy((params["0"], params["1"]), deterministic=True))


def run_rollouts_parallel(policy_fn, env, n_rollouts, episode_length, seed_base):
    """Run N rollouts in parallel using jax.vmap — all on GPU simultaneously."""

    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))
    v_policy = jax.jit(jax.vmap(policy_fn))

    # N different reset keys
    reset_rngs = jax.random.split(jax.random.PRNGKey(seed_base), n_rollouts)

    # Batched reset — N envs at once
    states = v_reset(reset_rngs)
    total_rewards = jnp.zeros(n_rollouts)

    # Running RNG for policy actions
    step_rng = jax.random.PRNGKey(seed_base + 10_000)

    for t in range(episode_length):
        # Split into N action rngs for this timestep
        step_rng, sub_rng = jax.random.split(step_rng)
        act_rngs = jax.random.split(sub_rng, n_rollouts)

        # Batched policy + step
        act_out = v_policy(states.obs, act_rngs)
        actions = act_out[0] if isinstance(act_out, tuple) else act_out
        states = v_step(states, actions)
        total_rewards = total_rewards + states.reward

    return np.array(jax.device_get(total_rewards))


def run_rollouts_binned(policy_fn, env_name, box_info, num_rollouts, episode_length, seed_base,
                        mass_range=None, friction_range=None, n_bins=N_BINS):
    """Bin physics params, then run each bin in parallel with vmap."""
    all_rewards = []

    if mass_range is None and friction_range is None:
        bins = [{"mass_scale": 1.0, "friction_scale": 1.0, "n": num_rollouts}]
    else:
        rolls_per_bin = num_rollouts // n_bins
        rng_np = np.random.RandomState(seed_base)
        bins = []
        for b in range(n_bins):
            bins.append({
                "mass_scale": rng_np.uniform(*mass_range) if mass_range else 1.0,
                "friction_scale": rng_np.uniform(*friction_range) if friction_range else 1.0,
                "n": rolls_per_bin,
            })

    for i, b in enumerate(bins):
        with contextlib.redirect_stdout(io.StringIO()):
            test_env = load_env_with_shift(env_name, box_info,
                                           mass_scale=b["mass_scale"],
                                           friction_scale=b["friction_scale"])

        rewards = run_rollouts_parallel(
            policy_fn, test_env, b["n"], episode_length,
            seed_base=seed_base + i * b["n"],
        )
        all_rewards.extend(rewards.tolist())

    return np.array(all_rewards)


def run_rollouts_sequential(policy_fn, env_name, box_info, num_rollouts, episode_length, seed_base,
                            mass_range=None, friction_range=None, n_bins=N_BINS):
    """Fallback: sequential rollouts if vmap fails."""
    rewards = []

    if mass_range is None and friction_range is None:
        bins = [{"mass_scale": 1.0, "friction_scale": 1.0, "n": num_rollouts, "seed_offset": 0}]
    else:
        rolls_per_bin = num_rollouts // n_bins
        rng_np = np.random.RandomState(seed_base)
        bins = []
        for b in range(n_bins):
            bins.append({
                "mass_scale": rng_np.uniform(*mass_range) if mass_range else 1.0,
                "friction_scale": rng_np.uniform(*friction_range) if friction_range else 1.0,
                "n": rolls_per_bin, "seed_offset": b * rolls_per_bin,
            })

    for b in bins:
        with contextlib.redirect_stdout(io.StringIO()):
            test_env = load_env_with_shift(env_name, box_info,
                                           mass_scale=b["mass_scale"],
                                           friction_scale=b["friction_scale"])
        jit_reset = jax.jit(test_env.reset)
        jit_step = jax.jit(test_env.step)

        for i in range(b["n"]):
            rng = jax.random.PRNGKey(seed_base + b["seed_offset"] + i)
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            ep_reward = 0.0
            for _ in range(episode_length):
                rng, act_rng = jax.random.split(rng)
                act_out = policy_fn(state.obs, act_rng)
                action = act_out[0] if isinstance(act_out, tuple) else act_out
                state = jit_step(state, jnp.asarray(action))
                ep_reward += float(state.reward)
            rewards.append(ep_reward)

    return np.array(rewards)


def make_histogram_grid(results, num_rollouts, save_path):
    policy_labels = list(results.keys())
    cond_labels = list(TEST_CONDITIONS.keys())
    n_rows, n_cols = len(policy_labels), len(cond_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    all_rewards = [np.array(r) for p in results.values() for r in p.values()]
    global_min = min(r.min() for r in all_rewards)
    global_max = max(r.max() for r in all_rewards)
    pad = (global_max - global_min) * 0.1
    hist_bins = np.linspace(global_min - pad, global_max + pad, 20)

    for row, plabel in enumerate(policy_labels):
        for col, clabel in enumerate(cond_labels):
            ax = axes[row][col]
            rews = np.array(results[plabel][clabel])

            ax.hist(rews, bins=hist_bins, color="#2E86AB", edgecolor="white", alpha=0.8)
            ax.axvline(rews.mean(), color="red", linestyle="--", linewidth=2,
                       label=f"μ={rews.mean():.1f}")
            ax.axvline(rews.mean() - rews.std(), color="gray", linestyle=":", linewidth=1.2)
            ax.axvline(rews.mean() + rews.std(), color="gray", linestyle=":", linewidth=1.2,
                       label=f"σ={rews.std():.1f}")
            ax.legend(fontsize=8)

            if row == 0:
                ax.set_title(clabel, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(plabel, fontsize=12, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("Episode Reward", fontsize=10)

    fig.suptitle(f"Robustness Comparison ({num_rollouts} rollouts each)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {save_path}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    print(f"JAX devices: {jax.devices()}")
    print(f"Policies: {len(POLICY_RUNS)}, Conditions: {len(TEST_CONDITIONS)}")
    print(f"Rollouts per cell: {NUM_ROLLOUTS}, Episode length: {EPISODE_LENGTH}\n")

    # Load reference env + box info
    ref_env = registry.load(ENV_NAME)
    box_info = get_box_info(ref_env)
    print(f"Box body id: {box_info['box_body_id']}, "
          f"nominal mass: {box_info['nominal_mass']:.4f}, "
          f"nominal friction: {box_info['nominal_friction']}")

    # ── Decide: try vmap, fall back to sequential ──
    print("\nTesting vmap rollouts...", end=" ", flush=True)
    try:
        test_policy = load_policy_from_run(list(POLICY_RUNS.values())[0], ref_env)
        test_rewards = run_rollouts_binned(test_policy, ENV_NAME, box_info, 5, 10, 0)
        use_vmap = True
        print(f"✓ vmap works (test mean={test_rewards.mean():.1f})")
    except Exception as e:
        use_vmap = False
        print(f"✗ vmap failed ({e}), falling back to sequential")

    run_fn = run_rollouts_binned if use_vmap else run_rollouts_sequential
    print(f"Mode: {'PARALLEL (vmap)' if use_vmap else 'SEQUENTIAL'}\n")

    # ── Run full grid ──
    results = {}
    for policy_label, rid in POLICY_RUNS.items():
        print(f"\n── {policy_label} ──")
        results[policy_label] = {}
        policy_fn = load_policy_from_run(rid, ref_env)

        for cond_label, cond in TEST_CONDITIONS.items():
            t_cell = time.time()
            print(f"  {cond_label} ... ", end="", flush=True)
            rewards = run_fn(
                policy_fn, ENV_NAME, box_info, NUM_ROLLOUTS, EPISODE_LENGTH, SEED_BASE,
                mass_range=cond.get("mass_range"),
                friction_range=cond.get("friction_range"),
            )
            results[policy_label][cond_label] = rewards.tolist()
            dt = time.time() - t_cell
            print(f"mean={np.mean(rewards):.1f} ± {np.std(rewards):.1f}  ({dt:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\n✓ Grid complete in {elapsed/60:.1f} min")

    # ── Save histogram ──
    os.makedirs("eval_outputs", exist_ok=True)
    hist_path = "eval_outputs/robustness_histogram_grid.png"
    make_histogram_grid(results, NUM_ROLLOUTS, hist_path)

    # ── Save raw JSON ──
    json_path = "eval_outputs/robustness_results.json"
    summary = {}
    for plabel in results:
        summary[plabel] = {}
        for clabel in results[plabel]:
            rews = np.array(results[plabel][clabel])
            summary[plabel][clabel] = {
                "mean": float(rews.mean()),
                "std": float(rews.std()),
                "min": float(rews.min()),
                "max": float(rews.max()),
                "n": len(rews),
                "rewards": rews.tolist(),
            }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved {json_path}")

    # ── Log everything to W&B ──
    run = wandb.init(
        project=PROJECT,
        entity=USER,
        name=f"robustness_eval_{NUM_ROLLOUTS}r",
        group="eval",
        config={
            "policy_runs": POLICY_RUNS,
            "test_conditions": {k: str(v) for k, v in TEST_CONDITIONS.items()},
            "num_rollouts": NUM_ROLLOUTS,
            "episode_length": EPISODE_LENGTH,
            "seed_base": SEED_BASE,
            "n_bins": N_BINS,
            "parallel": use_vmap,
            "elapsed_min": elapsed / 60,
        },
    )

    run.log({"robustness_grid": wandb.Image(hist_path)})

    table_data = []
    for plabel in summary:
        for clabel in summary[plabel]:
            s = summary[plabel][clabel]
            table_data.append([plabel, clabel, s["mean"], s["std"], s["min"], s["max"]])

    table = wandb.Table(
        columns=["Policy", "Condition", "Mean", "Std", "Min", "Max"],
        data=table_data,
    )
    run.log({"results_table": table})

    artifact = wandb.Artifact(
        name=f"robustness_results_{NUM_ROLLOUTS}r",
        type="eval_results",
    )
    artifact.add_file(json_path)
    artifact.add_file(hist_path)
    run.log_artifact(artifact)

    run.finish()
    print(f"\n✓ All done — logged to W&B ({elapsed/60:.1f} min total)")
