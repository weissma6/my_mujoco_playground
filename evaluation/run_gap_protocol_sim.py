"""Gap-protocol sim mirror (Commit 7 of "Plan - Sim-to-Real Gap Protocol").

MJX -- WRITE-ONLY / verified by py_compile + static reasoning here. Per
CLAUDE.md, never run env.step()/MJX locally on this machine (no GPU; CPU MJX
OOM-kills it). Every claim below about behaviour is a reading of the code, not
an observed run -- run this on the ZHAW HPC (see the bottom of this docstring
for the exact commands).

What this does
--------------------------------------------------------------------------
For each real-robot gap-protocol run folder (the ep{ID}_rep{K}/ layout
robots/UR3e/run_gap_protocol.py produces), read its measured_init.json and
reset the MJX UR3Pick env to EXACTLY that measured state (arm pose + box
pose) via the new eval-only UR3Pick.reset_to_state() (see
mujoco_playground/_src/manipulation/my_ur3/ur3_pick.py -- an ADDITIVE method;
the training reset() above it is untouched, byte-for-byte). Then roll out the
config's best-checkpoint policy DETERMINISTICALLY (mean action, no
exploration noise) for exactly H = protocol.horizon steps -- never fewer,
never more, matching run_gap_protocol.py's own "D6: sim truncates to H"
convention (this script does not stop early on `state.metrics["success"]` or
`out_of_bounds`; see the "Fixed horizon" note below). Per-step scaled reward
terms + geometry are written to sim_results/{config_id}/{protocol_id}/
ep{ID}_rep{K}/, mirroring the real output's directory layout so a later
merge into evaluation/gap_metrics.csv can walk both trees symmetrically.

Target computation -- same rule as run_gap_protocol.py
--------------------------------------------------------------------------
The lift target is NEVER re-derived here: `evaluation.gap_target.
target_for_episode(protocol_id, episode_id, config_id, box_xy)` is the one
call site both the real runner and this sim mirror use, so they can never
disagree about which target they were scored against (D2). box_xy comes from
the SAME measured_init.json the real run wrote (its own box_xy was fed to
the identical function when that run happened), so re-calling it here
reproduces the exact same draw -- this script also cross-checks the
recomputed target against the one already stored in measured_init.json and
warns loudly on any mismatch (which would mean gap_target.py or
gen_dr_ladder._DETERMINISTIC_POSITION drifted since that real run).

Checkpoint choice -- best only, never final (locked decision)
--------------------------------------------------------------------------
Always loads `trained_policy/best_params.msgpack` from the run's W&B Files
tab -- never `params.msgpack` (the final-step params, which can be past a
post-peak training collapse; see learning/notebooks/run_experiment.py's
best_ckpt tracking). There is deliberately NO --checkpoint flag to override
this. evaluation/downloaded_policies/policy_downloader.py's download_policy()
only fetches params.msgpack, so it is NOT reused for the params file itself
(its Files-tab lookup pattern IS mirrored, in `_fetch_best_params` below, to
avoid re-deriving how to reach a run's Files tab); metadata.json (network
factory, obs/action dims, env_overrides) still comes from download_policy()
since that part is unaffected by which params file gets loaded.

Fixed horizon (D6) -- mirrors run_gap_protocol.py's real-side convention
--------------------------------------------------------------------------
H is read from `protocol.horizon` (the frozen protocol JSON) and asserted
equal to `int(env._config.episode_length)` (itself resolved from the
policy's own `env_overrides`/config, NEVER hardcoded to 400) -- a mismatch
raises immediately rather than silently truncating/padding. The rollout
always executes exactly H `step()` calls, even past `state.metrics["success"]`
or `state.metrics["out_of_bounds"]` going high -- MJX's raw step() (unlike a
brax-wrapped env) does not auto-reset or halt on `done`, so "run for exactly
H steps" is simply "call step() H times without checking done", matching
run_gap_protocol.py's own choice to disable early termination (reach_tol=None)
and truncate/pad to H post-hoc.

Determinism check
--------------------------------------------------------------------------
`_check_determinism` runs the identical (init, target, seed) through
`rollout_one` TWICE and asserts the two summed returns match exactly (no
tolerance beyond float round-off) -- reset_to_state + a deterministic
`inference_fn` + a fixed rng make the whole rollout a pure function of its
inputs, so any mismatch means something in the pipeline (a stray
non-deterministic op, or an uncontrolled global) needs investigating before
trusting any result from this run. Runs once per invocation of `main()`
(on the FIRST processed folder) -- not on every folder, to keep the added
compute small on a run that may already be O(100) episodes.

Bootstrapping the D10 seed-selection dependency (an addition beyond the
literal task spec -- flagged, not silently assumed)
--------------------------------------------------------------------------
D10 (evaluation/select_median_seed.py) needs sim returns on the protocol's
10 episodes PER SEED to pick which seed's checkpoint ever reaches the real
robot -- but that selection must happen BEFORE robots/UR3e/run_gap_protocol.py
can run at all (it refuses to run without a resolved policy_run_id, and the
policy map is what D10 writes). At that point no real run folder -- and
therefore no MEASURED box pose -- exists yet for the config being selected.
The literal task text for this script is written entirely in terms of real
run folders' measured_init.json, which cannot supply that. `--protocol_only`
below is the closing move: it iterates the protocol's OWN frozen episodes
directly (arm_qpos + finger from the protocol, exactly as commanded), with
the box at its NOMINAL keyframe pose (identity orientation) instead of a
measured one -- fine for seed SELECTION (all 3 seeds are compared against
the identical nominal box placement, so the comparison stays fair/unbiased)
but NOT a substitute for the measured-init mirror once real data exists.
Output goes to ep{ID}_rep0/ (repeat 0 is reserved for this nominal-box mode;
real repeats are 1..k) so it can never collide with a measured-init run.

Local usage (dry run only -- lists what WOULD run, no MJX/jax import at all):
    python evaluation/run_gap_protocol_sim.py --config_id L1_pos \
        --policy_run_id <wandb_run_id> --dry_run

HPC usage (after the local py_compile / dry-run checks pass):
    python evaluation/run_gap_protocol_sim.py --config_id L1_pos \
        --policy_run_id <wandb_run_id>
    python evaluation/run_gap_protocol_sim.py --config_id L0_none \
        --policy_run_id <wandb_run_id> --protocol_only   # D10 bootstrap
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (REPO_ROOT, _THIS_DIR, os.path.join(REPO_ROOT, "evaluation", "downloaded_policies")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.protocols import load_protocol  # noqa: E402
from evaluation import gap_target  # noqa: E402
# Reuse the term/scale taxonomy -- do not re-type it. sim_rollout_reward
# itself is NOT reused unmodified: it hardcodes params.msgpack, and this
# script must load best_params.msgpack (see module docstring).
from evaluation.ur3_reward_replay import TERMS, SCALES  # noqa: E402
from policy_downloader import default_policy_dir, download_policy  # noqa: E402

WANDB_ENTITY = "weissma6-zhaw-school-of-engineering"
WANDB_PROJECT = "UR3_pick_ppo"

MODEL_PATH = os.path.join(
    REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "xmls",
    "mjx_single_cube_position_ur3.xml",
)
DEFAULT_PROTOCOL = os.path.join(REPO_ROOT, "evaluation", "protocols", "gap_protocol_v1.json")
DEFAULT_REAL_ROOT = os.path.join(REPO_ROOT, "robots", "UR3e", "real_robot_results", "gap_protocol")
DEFAULT_SIM_ROOT = os.path.join(REPO_ROOT, "robots", "UR3e", "sim_results", "gap_protocol")


# ===========================================================================
# Checkpoint fetch -- best_params.msgpack only (see module docstring).
# ===========================================================================


def _fetch_best_params(run_id: str, policy_dir: str, entity: str, project: str) -> str:
    """Download trained_policy/best_params.msgpack for `run_id` into
    `policy_dir`. Mirrors policy_downloader._fetch_from_wandb's Files-tab
    lookup pattern (not reused directly -- that function is hardwired to
    params.msgpack). Raises loudly (never falls back to params.msgpack) if
    the run published no best checkpoint -- see run_experiment.py's
    `if best_ckpt.get("params") is not None:` guard; not every run has one.
    """
    best_path = os.path.join(policy_dir, "best_params.msgpack")
    if os.path.exists(best_path):
        return best_path

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    files = {f.name: f for f in run.files()}
    name = "trained_policy/best_params.msgpack"
    if name not in files:
        raise SystemExit(
            f"run {run_id!r} has no {name!r} in its W&B Files tab -- this run "
            f"never published a best-by-eval-reward checkpoint. Per the locked "
            f"design decision (best checkpoint only, never final -- final can "
            f"be past a training collapse), this script refuses to silently "
            f"fall back to trained_policy/params.msgpack. Pick a different "
            f"seed/run, or re-check why this run has no best_ckpt."
        )
    stage = os.path.join(policy_dir, "_wandb_best")
    os.makedirs(stage, exist_ok=True)
    files[name].download(root=stage, replace=True)
    shutil.copyfile(os.path.join(stage, name), best_path)
    return best_path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_best_policy(policy_dir: str):
    """Build a jitted DETERMINISTIC inference_fn from best_params.msgpack.

    Network construction mirrors evaluation/ur3_reward_replay.
    sim_rollout_reward's (keep the two in sync if either changes) -- the only
    difference is the params FILE loaded (best, not final). MJX/jax imports
    are local to this function so `--dry_run` never touches them.
    """
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from flax import serialization

    with open(os.path.join(policy_dir, "metadata.json")) as f:
        meta = json.load(f)

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
    best_path = os.path.join(policy_dir, "best_params.msgpack")
    if not os.path.exists(best_path):
        raise SystemExit(
            f"{best_path} missing -- call _fetch_best_params() first."
        )
    with open(best_path, "rb") as f:
        params = serialization.from_bytes(template, f.read())
    raw_policy = ppo_networks.make_inference_fn(ppo_net)(
        (params["0"], params["1"]), deterministic=True
    )
    inference_fn = jax.jit(raw_policy)
    return inference_fn, meta


def build_env(meta: dict, horizon: int):
    """registry.load the training env (env_overrides from the policy's own
    metadata, NEVER hardcoded) and assert its episode_length == protocol
    horizon -- fail loudly on mismatch rather than silently truncate/pad.
    """
    from mujoco_playground import registry

    env_overrides = meta.get("env_overrides", {})
    env = registry.load(meta["env_name"], config_overrides=env_overrides)
    ep_len = int(env._config.episode_length)
    if ep_len != horizon:
        raise SystemExit(
            f"episode_length mismatch: env config (from this policy's "
            f"env_overrides) says {ep_len}, protocol says horizon={horizon}. "
            f"Refusing to silently truncate/pad -- these are not the same "
            f"policy/protocol pairing the plan intends. Fix the sweep's "
            f"episode_length or confirm the policy_run_id."
        )
    return env


# ===========================================================================
# One rollout: reset_to_state -> H deterministic steps -> per-step DataFrame.
# ===========================================================================


def rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos, box_quat,
                 target_pos, horizon: int) -> pd.DataFrame:
    """H deterministic step() calls from an EXACT reset_to_state() init.

    Collects state.metrics[term] * SCALES[term] for every reward term (same
    pattern as ur3_reward_replay.sim_rollout_reward), plus box_target_dist/
    success/reward_total. Never checks `done` -- see the module docstring's
    "Fixed horizon" note.
    """
    import jax

    jit_step = jax.jit(env.step)
    jit_reset_to_state = jax.jit(env.reset_to_state)

    state = jit_reset_to_state(rng, arm_qpos, finger, box_pos, box_quat, target_pos)
    rows = []
    for t in range(horizon):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)  # deterministic -> act_rng unused
        state = jit_step(state, action)
        raw = {k: float(state.metrics[k]) for k in TERMS}
        scaled = {k: raw[k] * SCALES[k] for k in TERMS}
        row = dict(scaled)
        row["step"] = t
        row["reward_total"] = float(state.reward)
        row["box_target_dist"] = float(state.metrics["box_target_dist"])
        row["success"] = float(state.metrics["success"])
        row["out_of_bounds"] = float(state.metrics["out_of_bounds"])
        rows.append(row)
    return pd.DataFrame(rows)


def _check_determinism(env, inference_fn, rng, arm_qpos, finger, box_pos,
                        box_quat, target_pos, horizon: int):
    """Same init run twice -> must produce IDENTICAL summed return. Fires
    once per `main()` invocation, on the first processed folder.
    """
    df1 = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                       box_quat, target_pos, horizon)
    df2 = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                       box_quat, target_pos, horizon)
    r1, r2 = float(df1["reward_total"].sum()), float(df2["reward_total"].sum())
    if not np.isclose(r1, r2, rtol=0.0, atol=1e-9):
        raise AssertionError(
            f"determinism check FAILED: identical (init, target, seed) produced "
            f"different returns ({r1} vs {r2}). This should be impossible for a "
            f"deterministic policy from a fixed reset_to_state() -- investigate "
            f"before trusting ANY result from this invocation (a stray "
            f"non-deterministic op or uncontrolled global RNG is the likely cause)."
        )
    print(f"  [determinism check] OK: return={r1:.4f} (identical init+seed, run twice)")


# ===========================================================================
# Real-run-folder driven mode.
# ===========================================================================


def _find_real_run_dirs(real_root: str, config_id: str, protocol_id: str,
                         episodes=None):
    pattern = os.path.join(real_root, config_id, protocol_id, "ep*_rep*")
    dirs = sorted(glob.glob(pattern))
    if episodes is not None:
        wanted = {f"ep{e}_" for e in episodes}
        dirs = [d for d in dirs if any(os.path.basename(d).startswith(w) for w in wanted)]
    return dirs


def _parse_ep_rep(dirname: str):
    base = os.path.basename(dirname)  # "ep{ID}_rep{K}"
    ep_str, rep_str = base.split("_rep")
    return int(ep_str[2:]), int(rep_str)


def process_real_run(run_dir, env, inference_fn, meta, protocol, config_id,
                      policy_run_id, checkpoint_hash, horizon, rng,
                      run_determinism_check=False):
    with open(os.path.join(run_dir, "measured_init.json")) as f:
        m = json.load(f)

    episode_id, repeat = m["episode_id"], m["repeat"]
    arm_qpos = np.asarray(m["measured_arm_qpos"], dtype=float)
    finger = float(m["commanded_finger"])
    box_pos = np.asarray(m["box_pos"], dtype=float)
    box_quat = np.asarray(m["box_quat_wxyz"], dtype=float)

    target_pos, _components = gap_target.target_for_episode(
        protocol.protocol_id, episode_id, config_id,
        box_xy=box_pos[:2], xml_path=MODEL_PATH,
    )
    stored_target = np.asarray(m["target_pos"], dtype=float)
    if not np.allclose(target_pos, stored_target, atol=1e-6):
        print(
            f"  WARNING: recomputed target {target_pos} != measured_init.json's "
            f"stored target {stored_target} for {run_dir} -- gap_target.py or "
            f"gen_dr_ladder._DETERMINISTIC_POSITION may have drifted since this "
            f"real run. Using the RECOMPUTED value (single source of truth)."
        )

    if run_determinism_check:
        _check_determinism(env, inference_fn, rng, arm_qpos, finger, box_pos,
                            box_quat, target_pos, horizon)

    df = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                      box_quat, target_pos, horizon)
    return episode_id, repeat, df, {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "config_id": config_id,
        "episode_id": episode_id,
        "repeat": repeat,
        "policy_run_id": policy_run_id,
        "checkpoint": "best_params.msgpack",
        "checkpoint_hash_sha256": checkpoint_hash,
        "source": "real_measured_init",
        "measured_init_path": os.path.abspath(
            os.path.join(run_dir, "measured_init.json")
        ),
        "expected_steps": horizon,
        "actual_steps": len(df),
        "achieved_hz": float("nan"),  # not applicable to a scripted MJX rollout
        "overrun_count": 0,
        "stop_reason": "horizon_complete",
        "episode_return": float(df["reward_total"].sum()),
        "final_box_target_dist": float(df["box_target_dist"].iloc[-1]),
        "success": bool(df["success"].iloc[-1] > 0.5),
    }


# ===========================================================================
# --protocol_only mode (D10 bootstrap -- see module docstring).
# ===========================================================================


def _nominal_box_pose(xml_path: str = None):
    """Box pose at the scene's task_home keyframe (identity orientation) --
    used only by --protocol_only, where no real measurement exists yet.
    """
    import mujoco

    xml_path = xml_path or MODEL_PATH
    model = mujoco.MjModel.from_xml_path(xml_path)
    box_jntadr = model.body("box").jntadr[0]
    box_qposadr = int(model.jnt_qposadr[box_jntadr])
    qpos = model.key("task_home").qpos
    box_pos = np.asarray(qpos[box_qposadr : box_qposadr + 3], dtype=float)
    box_quat = np.array([1.0, 0.0, 0.0, 0.0])
    return box_pos, box_quat


def process_protocol_episode(episode, env, inference_fn, protocol, config_id,
                              policy_run_id, checkpoint_hash, horizon, rng,
                              run_determinism_check=False):
    box_pos, box_quat = _nominal_box_pose()
    target_pos, _components = gap_target.target_for_episode(
        protocol.protocol_id, episode.episode_id, config_id,
        box_xy=box_pos[:2], xml_path=MODEL_PATH,
    )
    arm_qpos = np.asarray(episode.arm_qpos, dtype=float)
    finger = float(episode.finger)

    if run_determinism_check:
        _check_determinism(env, inference_fn, rng, arm_qpos, finger, box_pos,
                            box_quat, target_pos, horizon)

    df = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                      box_quat, target_pos, horizon)
    repeat = 0  # reserved for nominal-box mode; real repeats are 1..k
    return episode.episode_id, repeat, df, {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "config_id": config_id,
        "episode_id": episode.episode_id,
        "repeat": repeat,
        "policy_run_id": policy_run_id,
        "checkpoint": "best_params.msgpack",
        "checkpoint_hash_sha256": checkpoint_hash,
        "source": "protocol_nominal_box",
        "expected_steps": horizon,
        "actual_steps": len(df),
        "achieved_hz": float("nan"),
        "overrun_count": 0,
        "stop_reason": "horizon_complete",
        "episode_return": float(df["reward_total"].sum()),
        "final_box_target_dist": float(df["box_target_dist"].iloc[-1]),
        "success": bool(df["success"].iloc[-1] > 0.5),
    }


# ===========================================================================
# CLI
# ===========================================================================


def run_dir_out(out_root, config_id, protocol_id, episode_id, repeat):
    return os.path.join(out_root, config_id, protocol_id, f"ep{episode_id}_rep{repeat}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--config_id", required=True)
    ap.add_argument("--policy_run_id", required=True,
                     help="W&B run id whose best_params.msgpack to load")
    ap.add_argument("--real_root", default=DEFAULT_REAL_ROOT,
                     help="real gap-protocol run tree to mirror (ignored with --protocol_only)")
    ap.add_argument("--out_root", default=DEFAULT_SIM_ROOT)
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                     help="subset of episode_ids; default = all found/in protocol")
    ap.add_argument("--protocol_only", action="store_true",
                     help="D10 bootstrap mode: run the protocol's own frozen "
                          "episodes with a NOMINAL (unmeasured) box pose, "
                          "instead of mirroring real run folders -- see the "
                          "module docstring")
    ap.add_argument("--force", action="store_true",
                     help="re-run folders that already exist (resume is the default)")
    ap.add_argument("--seed", type=int, default=0,
                     help="base PRNG seed threaded into reset_to_state/rollout")
    ap.add_argument("--dry_run", action="store_true",
                     help="list what would run (folders + output paths) and exit -- "
                          "no jax/mjx import, safe to run on this machine")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol, dry_run=True)
    print(f"Protocol {protocol.protocol_id}: {len(protocol)} episodes, "
          f"horizon={protocol.horizon}, hash={protocol.poses_sha256[:16]}...")

    if args.protocol_only:
        episodes = [e for e in protocol.episodes
                    if args.episodes is None or e.episode_id in args.episodes]
        pending = []
        for ep in episodes:
            out_dir = run_dir_out(args.out_root, args.config_id, protocol.protocol_id,
                                   ep.episode_id, 0)
            if os.path.exists(out_dir) and not args.force:
                continue
            pending.append((ep, out_dir))
        print(f"{len(pending)}/{len(episodes)} protocol episodes pending "
              f"(--protocol_only, nominal box).")
    else:
        run_dirs = _find_real_run_dirs(args.real_root, args.config_id,
                                        protocol.protocol_id, args.episodes)
        pending = []
        for rd in run_dirs:
            episode_id, repeat = _parse_ep_rep(rd)
            out_dir = run_dir_out(args.out_root, args.config_id, protocol.protocol_id,
                                   episode_id, repeat)
            if os.path.exists(out_dir) and not args.force:
                continue
            pending.append((rd, out_dir))
        print(f"{len(pending)}/{len(run_dirs)} real run folders pending "
              f"under {args.real_root}/{args.config_id}/{protocol.protocol_id}/.")

    if args.dry_run:
        for item, out_dir in pending:
            src = item if isinstance(item, str) else f"protocol ep{item.episode_id}"
            print(f"  {src}  ->  {out_dir}")
        print("(--dry_run: stopping before any jax/mjx import.)")
        return
    if not pending:
        print("Nothing to do.")
        return

    policy_dir = default_policy_dir(args.policy_run_id)
    download_policy(args.policy_run_id, out_dir=policy_dir,
                     entity=WANDB_ENTITY, project=WANDB_PROJECT)  # metadata.json etc.
    _fetch_best_params(args.policy_run_id, policy_dir, WANDB_ENTITY, WANDB_PROJECT)
    checkpoint_hash = sha256_file(os.path.join(policy_dir, "best_params.msgpack"))

    inference_fn, meta = load_best_policy(policy_dir)
    env = build_env(meta, protocol.horizon)

    import jax
    base_rng = jax.random.PRNGKey(args.seed)

    completed = 0
    for i, (item, out_dir) in enumerate(pending):
        rng = jax.random.fold_in(base_rng, i)  # distinct-but-deterministic per folder
        run_determinism_check = (i == 0)
        if args.protocol_only:
            episode_id, repeat, df, meta_out = process_protocol_episode(
                item, env, inference_fn, protocol, args.config_id,
                args.policy_run_id, checkpoint_hash, protocol.horizon, rng,
                run_determinism_check=run_determinism_check,
            )
        else:
            episode_id, repeat, df, meta_out = process_real_run(
                item, env, inference_fn, meta, protocol, args.config_id,
                args.policy_run_id, checkpoint_hash, protocol.horizon, rng,
                run_determinism_check=run_determinism_check,
            )
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "sim_states.csv"), index=False)
        with open(os.path.join(out_dir, "sim_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, default=str)
        print(f"  ep{episode_id}_rep{repeat}: return={meta_out['episode_return']:.2f} "
              f"success={meta_out['success']}  ->  {out_dir}")
        completed += 1

    print(f"\nDone: {completed}/{len(pending)} pending runs completed.")


if __name__ == "__main__":
    main()
