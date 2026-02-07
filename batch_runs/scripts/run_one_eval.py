"""
Cluster evaluation runner.
Each SLURM array task runs ONE line from the eval JSONL:
  one policy × one condition → N rollouts → results JSON.

JSONL fields:
  eval_id          : str   – unique name for this eval cell
  policy_run_id    : str   – wandb run id to pull the trained policy from
  condition        : str   – human label ("Default", "Mass", "Friction", …)
  mass_range       : [lo, hi] or null  – if set, sample mass scale per bin
  friction_range   : [lo, hi] or null  – if set, sample friction scale per bin
  mass_scale       : float or null     – if set, use fixed mass scale (overrides mass_range)
  friction_scale   : float or null     – if set, use fixed friction scale (overrides friction_range)
  n_bins           : int   – number of discrete physics bins (default 5, ignored if no range)
  num_rollouts     : int   – total rollouts (default 50)
  episode_length   : int   – steps per episode (default 150)
  seed_base        : int   – base seed (default 200)
  env_name         : str   – environment name (default "UR10PickCube")
  wandb_project    : str   – wandb project (default "UR10_pick_ppo")
  wandb_entity     : str   – wandb entity (default "weissma6-zhaw-school-of-engineering")
"""

import argparse
import contextlib
import io
import json
import os
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from flax import serialization
from mujoco_playground import registry
import wandb


# ──────────────────────────────────────────────
# Helpers (same as notebook, self-contained)
# ──────────────────────────────────────────────

def get_box_info(env):
    """Find box body/geom IDs and nominal physics values."""
    mj_model = None
    for attr in ("mj_model", "_mj_model", "model"):
        obj = getattr(env, attr, None)
        if obj is not None and hasattr(obj, "ngeom"):
            mj_model = obj
            break
    if mj_model is None:
        sys = getattr(env, "sys", None)
        if sys is not None:
            mj_model = getattr(sys, "mj_model", None)
    if mj_model is None:
        raise RuntimeError(f"Cannot find mj_model on env (type={type(env)})")

    box_body_id, box_geom_id = -1, -1
    for name in ("box", "cube", "object", "target_object", "pick_object"):
        if box_body_id < 0:
            box_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if box_geom_id < 0:
            box_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert box_body_id >= 0, "Box body not found in model"
    assert box_geom_id >= 0, "Box geom not found in model"

    return {
        "box_body_id": box_body_id,
        "box_geom_id": box_geom_id,
        "nominal_mass": float(mj_model.body_mass[box_body_id]),
        "nominal_friction": mj_model.geom_friction[box_geom_id].copy(),
    }


def load_env_with_shift(env_name, box_info, mass_scale=1.0, friction_scale=1.0):
    """Load env, patch raw MuJoCo model, then recompile MJX model."""
    from mujoco import mjx

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
        sys = getattr(env, "sys", None)
        if sys is not None:
            mj_model = getattr(sys, "mj_model", None)
    if mj_model is None:
        raise RuntimeError(f"Cannot find mj_model on env (type={type(env)})")

    bid = box_info["box_body_id"]
    gid = box_info["box_geom_id"]
    if mass_scale != 1.0:
        mj_model.body_mass[bid] = box_info["nominal_mass"] * mass_scale
    if friction_scale != 1.0:
        mj_model.geom_friction[gid][0] = box_info["nominal_friction"][0] * friction_scale

    # Re-compile the MJX model so jit_reset/jit_step see the changes
    env._mjx_model = mjx.put_model(mj_model)

    return env


def load_policy_from_run(run_id, env, entity, project):
    """Download policy from wandb and build the inference function."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    train_cfg = dict(run.config)
    nf_params = train_cfg.get("network_factory", {}) or {}

    artifacts = run.logged_artifacts()
    policy_art = next((a for a in artifacts if a.type == "model"), None)
    if policy_art is None:
        raise ValueError(f"No model artifact for run {run_id}")
    art_dir = policy_art.download(root="downloaded_policies")

    with open(os.path.join(art_dir, "params.msgpack"), "rb") as f:
        params_bytes = f.read()

    nf_kwargs = {}
    for k, v in nf_params.items():
        nf_kwargs[k] = tuple(v) if isinstance(v, list) else v

    obs_size = env.observation_size
    action_size = env.action_size

    ppo_net = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **nf_kwargs,
    )

    rng = jax.random.PRNGKey(0)
    template = {
        "0": running_statistics.init_state(jax.ShapeDtypeStruct((obs_size,), jnp.float32)),
        "1": ppo_net.policy_network.init(rng),
        "2": ppo_net.value_network.init(rng),
    }
    params = serialization.from_bytes(template, params_bytes)

    make_policy = ppo_networks.make_inference_fn(ppo_net)
    inference_params = (params["0"], params["1"])
    policy_fn = jax.jit(make_policy(inference_params, deterministic=True))
    return policy_fn


def run_rollouts(policy_fn, env_name, box_info, num_rollouts, episode_length,
                 seed_base, mass_range=None, friction_range=None,
                 mass_scale=None, friction_scale=None, n_bins=5):
    """
    Run N rollouts with optional physics perturbation.

    Priority: fixed scale > range > nominal.
    When using ranges, rollouts are grouped into n_bins discrete settings to
    minimize JIT recompilations.
    """
    rewards = []

    # ── Determine bins ──
    if mass_scale is not None or friction_scale is not None:
        # Fixed scale: single bin
        bins = [{
            "mass_scale": mass_scale or 1.0,
            "friction_scale": friction_scale or 1.0,
            "n": num_rollouts,
            "seed_offset": 0,
        }]
    elif mass_range is None and friction_range is None:
        # Default: single bin, nominal physics
        bins = [{
            "mass_scale": 1.0, "friction_scale": 1.0,
            "n": num_rollouts, "seed_offset": 0,
        }]
    else:
        # Range: spread across bins
        rolls_per_bin = num_rollouts // n_bins
        rng_np = np.random.RandomState(seed_base)
        bins = []
        for b in range(n_bins):
            ms = rng_np.uniform(*mass_range) if mass_range else 1.0
            fs = rng_np.uniform(*friction_range) if friction_range else 1.0
            bins.append({
                "mass_scale": ms, "friction_scale": fs,
                "n": rolls_per_bin, "seed_offset": b * rolls_per_bin,
            })

    # ── Run ──
    for b in bins:
        test_env = load_env_with_shift(
            env_name, box_info,
            mass_scale=b["mass_scale"],
            friction_scale=b["friction_scale"],
        )
        jit_reset = jax.jit(test_env.reset)
        jit_step = jax.jit(test_env.step)

        for i in range(b["n"]):
            seed_i = seed_base + b["seed_offset"] + i
            rng = jax.random.PRNGKey(seed_i)
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


# ──────────────────────────────────────────────
# JSONL loader (same pattern as run_one_ur10.py)
# ──────────────────────────────────────────────

def load_config(jsonl_path: str, index_1based: int) -> dict:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            i += 1
            if i == index_1based:
                return json.loads(line)
    raise IndexError(f"Index {index_1based} out of range for {jsonl_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate a single policy × condition cell")
    ap.add_argument("--jsonl", required=True, help="Path to eval sweep JSONL")
    ap.add_argument("--index", type=int, required=True, help="1-based line index")
    ap.add_argument("--out_root", default="eval_results", help="Output directory root")
    args = ap.parse_args()

    cfg = load_config(args.jsonl, args.index)

    eval_id = cfg.get("eval_id", f"eval_{args.index:04d}")
    env_name = cfg.get("env_name", "UR10PickCube")
    entity = cfg.get("wandb_entity", "weissma6-zhaw-school-of-engineering")
    project = cfg.get("wandb_project", "UR10_pick_ppo")
    policy_run_id = cfg["policy_run_id"]
    condition = cfg.get("condition", "Default")
    num_rollouts = cfg.get("num_rollouts", 50)
    episode_length = cfg.get("episode_length", 150)
    seed_base = cfg.get("seed_base", 200)
    n_bins = cfg.get("n_bins", 5)

    # Physics perturbation (fixed scale takes priority over range)
    mass_scale = cfg.get("mass_scale", None)
    friction_scale = cfg.get("friction_scale", None)
    mass_range = cfg.get("mass_range", None)
    friction_range = cfg.get("friction_range", None)
    if mass_range is not None:
        mass_range = tuple(mass_range)
    if friction_range is not None:
        friction_range = tuple(friction_range)

    out_dir = os.path.join(args.out_root, eval_id)
    os.makedirs(out_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(out_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"{'='*60}", flush=True)
    print(f"Eval: {eval_id}", flush=True)
    print(f"  Policy:    {policy_run_id}", flush=True)
    print(f"  Condition: {condition}", flush=True)
    print(f"  Mass:      scale={mass_scale} range={mass_range}", flush=True)
    print(f"  Friction:  scale={friction_scale} range={friction_range}", flush=True)
    print(f"  Rollouts:  {num_rollouts}  (bins={n_bins})", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()

    # Load reference env (for obs/action sizes and box info)
    with contextlib.redirect_stdout(io.StringIO()):
        ref_env = registry.load(env_name)
    box_info = get_box_info(ref_env)
    print(f"Box: body_id={box_info['box_body_id']}, "
          f"nominal_mass={box_info['nominal_mass']:.4f}, "
          f"nominal_friction={box_info['nominal_friction']}", flush=True)

    # Load policy
    policy_fn = load_policy_from_run(policy_run_id, ref_env, entity, project)
    print("✓ Policy loaded", flush=True)

    # Run rollouts
    rewards = run_rollouts(
        policy_fn, env_name, box_info,
        num_rollouts=num_rollouts,
        episode_length=episode_length,
        seed_base=seed_base,
        mass_range=mass_range,
        friction_range=friction_range,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
        n_bins=n_bins,
    )

    elapsed = time.time() - t0

    # ── Save results locally ──
    result = {
        "eval_id": eval_id,
        "policy_run_id": policy_run_id,
        "condition": condition,
        "mass_scale": mass_scale,
        "mass_range": list(mass_range) if mass_range else None,
        "friction_scale": friction_scale,
        "friction_range": list(friction_range) if friction_range else None,
        "num_rollouts": len(rewards),
        "episode_length": episode_length,
        "seed_base": seed_base,
        "n_bins": n_bins,
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "rewards": rewards.tolist(),
        "elapsed_s": round(elapsed, 1),
    }

    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Log to W&B ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wandb_project = cfg.get("wandb_project", "UR10_pick_ppo")
    run = wandb.init(
        project=wandb_project,
        name=eval_id,
        group="eval",
        tags=["eval", condition, policy_run_id.split("_20")[0]],
        config=cfg,
        job_type="eval",
    )

    # Log summary metrics
    run.summary["mean_reward"] = float(rewards.mean())
    run.summary["std_reward"] = float(rewards.std())
    run.summary["min_reward"] = float(rewards.min())
    run.summary["max_reward"] = float(rewards.max())
    run.summary["elapsed_s"] = round(elapsed, 1)

    # Log histogram plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rewards, bins=15, color="#2E86AB", edgecolor="white", alpha=0.8)
    ax.axvline(rewards.mean(), color="red", linestyle="--", linewidth=2,
               label=f"μ={rewards.mean():.1f}")
    ax.axvline(rewards.mean() - rewards.std(), color="gray", linestyle=":", linewidth=1.2)
    ax.axvline(rewards.mean() + rewards.std(), color="gray", linestyle=":", linewidth=1.2,
               label=f"σ={rewards.std():.1f}")
    ax.legend(fontsize=10)
    ax.set_xlabel("Episode Reward", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{eval_id}  (n={len(rewards)})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    wandb.log({"reward_histogram": wandb.Image(fig)})
    plt.close(fig)

    # Log raw rewards as a table
    table = wandb.Table(columns=["rollout", "reward"])
    for i, r in enumerate(rewards.tolist()):
        table.add_data(i, r)
    wandb.log({"rewards_table": table})

    # Log results JSON as artifact
    artifact = wandb.Artifact(f"eval_{eval_id}", type="eval")
    artifact.add_file(results_path)
    wandb.log_artifact(artifact)

    run.finish()

    print(f"\n✓ Done in {elapsed:.0f}s", flush=True)
    print(f"  mean={rewards.mean():.1f} ± {rewards.std():.1f}  "
          f"[{rewards.min():.1f}, {rewards.max():.1f}]", flush=True)
    print(f"  Logged to W&B: {eval_id}", flush=True)

if __name__ == "__main__":
    main()
