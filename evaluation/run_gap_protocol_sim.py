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
import re
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

# D17 geometry (confirmed 2026-07-22), RETIRED as an OOD probe by D19
# (2026-07-29): cube size is no longer a single scale factor with a hard
# graspability clamp the 4cm cube sat outside of. domain_rand.cube_size_xy /
# cube_size_z now draw INDEPENDENT ABSOLUTE half-extents spanning 2x2x3 cm to
# 4x4x4 cm (and _dr_max_box_half_xy was raised 0.018 -> 0.020 to admit the top
# of that range -- see ur3_pick.py), so "4cm" below is IN-distribution DR now,
# not an extrapolation test. It reappears as D22/D23 Block E: a same-
# distribution real-robot check.
# "3cm" is the sim's nominal box (Cube A, 3x3x4 cm) -- passing it through
# reset_to_state's cube_half_extents override is a NO-OP vs. not passing an
# override at all, since it equals the model's baked nominal size. Both
# physical cubes are 4 cm tall and share a mocap-rig centre (D17), so z stays
# 0.02 for both -- only x,y differ. This dict already stored independent xy/z
# half-extents (it never was a single scalar), so D19 needed no structural
# change here beyond this note.
# NOTE the grouping discipline is UNCHANGED by D19: 3cm and 4cm remain
# different physical conditions and must never be averaged together -- see
# evaluation/gap_metrics.py's _filter_cube_size.
CUBE_HALF_EXTENTS = {
    "3cm": (0.015, 0.015, 0.02),
    "4cm": (0.020, 0.020, 0.020),
}

MODEL_PATH = os.path.join(
    REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "xmls",
    "mjx_single_cube_position_ur3.xml",
)
DEFAULT_PROTOCOL = os.path.join(REPO_ROOT, "evaluation", "protocols", "gap_protocol_v1.json")
DEFAULT_REAL_ROOT = os.path.join(REPO_ROOT, "robots", "UR3e", "real_robot_results", "gap_protocol")
DEFAULT_SIM_ROOT = os.path.join(REPO_ROOT, "robots", "UR3e", "sim_results", "gap_protocol")
DEFAULT_BASE_CALIBRATION = os.path.join(
    REPO_ROOT, "robots", "UR3e", "calibration", "base_frame_calibration.json"
)


# ===========================================================================
# D25 (2026-07-29): mocap-world -> robot-base frame calibration.
#
# FOUND this session: robots/UR3e/run_gap_protocol.py's ONE-SHOT
# measured_init.json logging reads mocap.get_rigid_body_pose() and writes
# box_pos/box_quat_wxyz DIRECTLY, with no calibration applied -- unlike the
# LIVE control loop (ur3_realrobot_dependencies.py's run_policy_loop, lines
# ~1333/1353), which calls self.mocap_pos_to_base / self.mocap_quat_to_base
# on every streamed box observation the deployed policy actually sees. So
# the real robot's behaviour during collection is trustworthy (the policy
# always saw calibrated positions) -- only the LOGGED measured_init.json is
# in the wrong frame, which is exactly what feeds reset_to_state() here.
#
# Confirmed against physical fact (Matthias, 2026-07-29): robot base sits
# 77cm above the floor, and the mocap XYZ calibration was done AT the floor
# -- consistent with the empirical Z offset measured across 75 flat-table
# samples (mean 0.7827m, std 5.2mm) before this fix, and with
# translation_m[2]=0.7787 in base_frame_calibration.json.
#
# The transform below is a DELIBERATE, comment-linked duplicate of
# ur3_realrobot_dependencies.py's UR3RealRobotPick.load_base_calibration /
# .mocap_pos_to_base / .mocap_quat_to_base -- not a re-derivation. Reusing
# that class directly is not viable here: it imports rtde_receive/
# rtde_control (real-hardware libraries not installed where this MJX script
# runs, on HPC or in --dry_run locally). If either implementation changes,
# update both and verify they still agree (see selftest usage above these
# functions -- there is no formal selftest here yet since this module has no
# --selftest flag; verified manually against the same JSON this session).
# ===========================================================================


def load_base_calibration(json_path: str = None):
    """Load base_frame_calibration.json. Returns a dict of the precomputed
    pieces `mocap_pos_to_base`/`mocap_quat_to_base` need, or None if the file
    is absent (mocap used raw -- matches
    UR3RealRobotPick.load_base_calibration's own fallback).
    """
    json_path = json_path or DEFAULT_BASE_CALIBRATION
    if not os.path.exists(json_path):
        print(f"  WARNING: no base-frame calibration at {json_path} -- "
              f"box_pos/box_quat will be used RAW (mocap-world frame, NOT "
              f"the robot-base/sim frame). See this module's calibration "
              f"note above load_base_calibration.")
        return None
    with open(json_path) as f:
        cal = json.load(f)
    p0 = np.array(cal["translation_m"], dtype=float)
    r0t = np.array(cal["rotation_matrix"], dtype=float).T  # world -> raw base
    qw, qx, qy, qz = (float(v) for v in cal["rotation_quat_wxyz"])
    q_r0_conj = np.array([qw, -qx, -qy, -qz])
    q_d = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about Z: raw base -> sim base
    q_w2s = _quat_mult(q_d, q_r0_conj)
    return {"p0": p0, "r0t": r0t, "q_w2s": q_w2s, "path": json_path}


def _quat_mult(a, b) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions -- identical to
    UR3RealRobotPick._quat_mult."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


# Raw UR base -> MuJoCo/sim base: the sim base is 180 deg about Z from the
# real UR base (identical rationale/constant to UR3RealRobotPick._XY_NEG).
_XY_NEG = np.array([-1.0, -1.0, 1.0])


def mocap_pos_to_base(p_world, cal):
    """Map a mocap-world position into the sim/policy base frame. `cal` is
    `load_base_calibration()`'s return value; None passes `p_world` through
    unchanged (matches UR3RealRobotPick's identity fallback)."""
    if cal is None or p_world is None:
        return p_world
    p_raw = cal["r0t"] @ (np.asarray(p_world, dtype=float) - cal["p0"])
    return p_raw * _XY_NEG


def mocap_quat_to_base(quat_world, cal):
    """Map a mocap-world (w, x, y, z) quaternion into the sim/policy base
    frame. `cal=None` passes `quat_world` through unchanged."""
    if cal is None or quat_world is None:
        return quat_world
    return _quat_mult(cal["q_w2s"], np.asarray(quat_world, dtype=float))


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
                 target_pos, horizon: int, cube_half_extents=None,
                 lifter_top_height=None, lifter_tilt_rp=None) -> pd.DataFrame:
    """H deterministic step() calls from an EXACT reset_to_state() init.

    Collects state.metrics[term] * SCALES[term] for every reward term (same
    pattern as ur3_reward_replay.sim_rollout_reward), plus box_target_dist/
    success/reward_total. Never checks `done` -- see the module docstring's
    "Fixed horizon" note.

    D22 (2026-07-29) -- NO drop and NO landed-pose logging here, BY DESIGN.
    After its H-step loop the REAL runner (robots/UR3e/run_gap_protocol.py)
    opens the gripper, waits --drop_settle_s, reads the landed box pose off
    mocap and scores it with evaluation.gap_metrics.place_error. This sim
    mirror deliberately does none of that: it runs the fixed H steps and
    stops. That is an expected real<->sim ASYMMETRY, not an omission --
    place_error measures where a physically dropped cube came to rest, which
    only exists on real hardware; there is no sim-side equivalent to compute,
    and this function does not fabricate one. (place_error is also kept
    strictly out of R_real / retention / noise_floor / sim_selection_regret in
    gap_metrics.py, so nothing downstream expects a sim counterpart either.)

    `cube_half_extents` (D17): optional (3,) box geom half-extents override,
    forwarded verbatim to `env.reset_to_state` -- None (the default, and
    every --protocol_only call) reproduces the pre-D17 nominal-box rollout
    byte-for-byte; see ur3_pick.UR3Pick.reset_to_state's docstring for why
    this is unclamped and additive.

    `lifter_top_height`/`lifter_tilt_rp` (D23/D25, 2026-07-29): optional
    table-geometry override, forwarded verbatim to `env.reset_to_state` --
    both None (the default) reproduces the pre-D25 nominal-flat-table
    rollout byte-for-byte. See ur3_pick.UR3Pick.reset_to_state's docstring
    for the exact semantics; see `process_real_run` below for where these
    come from (currently: nowhere in existing real-run logs -- see that
    function's own note).
    """
    import jax

    jit_step = jax.jit(env.step)
    jit_reset_to_state = jax.jit(env.reset_to_state)

    state = jit_reset_to_state(
        rng, arm_qpos, finger, box_pos, box_quat, target_pos, cube_half_extents,
        lifter_top_height, lifter_tilt_rp,
    )
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
                        box_quat, target_pos, horizon: int, cube_half_extents=None,
                        lifter_top_height=None, lifter_tilt_rp=None):
    """Same init run twice -> must produce IDENTICAL summed return. Fires
    once per `main()` invocation, on the first processed folder.
    """
    df1 = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                       box_quat, target_pos, horizon, cube_half_extents,
                       lifter_top_height, lifter_tilt_rp)
    df2 = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                       box_quat, target_pos, horizon, cube_half_extents,
                       lifter_top_height, lifter_tilt_rp)
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
    """D23 (2026-07-29): folder names now optionally carry a leading
    `{condition_id}_` prefix (robots/UR3e/run_gap_protocol.py's `run_dir`,
    e.g. `A1_ep0_rep1_3cm`) ahead of the pre-D23 `ep{ID}_rep{K}[_{cube}]`
    shape. The old glob (`ep*_rep*`) silently matched ZERO D23 folders --
    found 2026-07-29 when the D23 board's first 135-run campaign produced no
    hits at all. `*_rep*` catches both shapes (condition-prefixed or not);
    `_parse_ep_rep` below does the real validation via regex, so a stray
    unrelated directory under this path still raises loudly there rather
    than being silently swallowed by a looser glob.
    """
    pattern = os.path.join(real_root, config_id, protocol_id, "*_rep*")
    dirs = sorted(glob.glob(pattern))
    if episodes is not None:
        wanted = set(episodes)
        dirs = [d for d in dirs if _parse_ep_rep(d)[0] in wanted]
    return dirs


_EP_REP_RE = re.compile(r"^(?:(?P<condition>[A-E]\d+)_)?ep(?P<episode>\d+)_rep(?P<repeat>\d+)(?:_(?P<cube>.+))?$")


def _parse_ep_rep(dirname: str):
    """Parse the pre-D17 "ep{ID}_rep{K}", the D17 "ep{ID}_rep{K}_{cube_size}",
    and the D23 "{condition_id}_ep{ID}_rep{K}_{cube_size}" real-run folder
    names (robots/UR3e/run_gap_protocol.py's `run_dir`). Returns
    (episode_id, repeat, cube_size, condition_id) -- cube_size is None for a
    pre-D17 folder (no suffix), condition_id is None for anything before D23
    (legacy campaign / --config_id single-config runs, which never carried
    a condition tag).
    """
    base = os.path.basename(dirname)
    m = _EP_REP_RE.match(base)
    if not m:
        raise ValueError(
            f"real run folder name {base!r} doesn't match "
            f"[<condition_id>_]ep<ID>_rep<K>[_<cube_size>] -- unexpected "
            f"directory under {os.path.dirname(dirname)}."
        )
    return (int(m.group("episode")), int(m.group("repeat")),
            m.group("cube"), m.group("condition"))


def _real_run_stop_reason(run_dir: str):
    """Read the D16 stop_reason out of the real runner's ur3_pick_meta.json.

    Returns None if the meta file is absent or predates D16 (no "stop_reason"
    key) -- old-format logs are NOT skipped (there was no abort concept to
    check), matching the pre-D16 behaviour exactly.
    """
    meta_path = os.path.join(run_dir, "ur3_pick_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("stop_reason")


# D23 (2026-07-29): conditions whose physical setup changes the TABLE itself
# (not just where the box sits on it). `reset_to_state()` gained a
# lifter_top_height/lifter_tilt_rp override (D25, 2026-07-29 -- see
# ur3_pick.py) to mirror this, but robots/UR3e/run_gap_protocol.py never
# logged the physical tilt angle or lifter height as its own field --
# D23_BOARD only carries descriptive strings ("~5 deg wedge", "top ~150mm").
# RECOVERED instead (2026-07-29) from data already in every run's
# measured_init.json, using the fact that D23_BOARD commands yaw_deg=0 for
# BOTH C1 and D1, and each varies exactly one axis of table geometry:
#   - D1 (lifter height; tilt is flat by spec): the box rests flush and
#     level, so its measured centre height alone gives the table top --
#     lifter_top_height = box_pos[2] - cube_half_extents[2].
#   - C1 (tilt wedge; height is nominal by spec): the box rests flush on the
#     tilted plate with no commanded yaw, so its measured orientation IS the
#     plate tilt (box_quat = quat_mul(q_pitch, q_roll) exactly, no yaw
#     factor to strip) -- inverted in closed form by `_decompose_tilt_quat`.
# Neither is derived for the other condition (deriving height from a TILTED
# box's z would conflate height with the tilt-induced lever-arm shift, and
# vice versa) -- each condition supplies only the one quantity D23_BOARD
# says it actually varies. Every OTHER condition (A/B/E) does not touch
# table geometry and both stay None (nominal flat table), unchanged from
# before this addition.
_TABLE_GEOMETRY_MISMATCH_CONDITIONS = {"C1", "D1"}


def _decompose_tilt_quat(box_quat):
    """Invert ur3_pick.py reset()'s tilt composition
    `q_tilt = _quat_mul(q_pitch, q_roll)` (roll about x, then pitch about y)
    to recover (roll, pitch) from a MEASURED box quaternion. Valid ONLY when
    the box's commanded yaw is exactly 0 (true for D23_BOARD's C1), so
    `box_quat` IS `q_tilt` with no separate yaw factor to strip.

    Closed form: q_tilt = (cp*cr, cp*sr, sp*cr, -sp*sr) where
    c*/s* = cos/sin of the half-angle, so roll = 2*atan2(x, w) and
    pitch = 2*atan2(y, w) (robust for D23's small tilt angles, where
    cos(angle/2) > 0 throughout).

    Returns ((roll, pitch), residual_rad). `residual_rad` is the geodesic
    angle between the measured quat and the roll/pitch-only recomposition of
    it -- ~0 for a pure roll+pitch tilt with no yaw, growing if the real
    placement had a non-negligible yaw component this decomposition cannot
    separate from tilt. The caller decides what residual is worth a warning.
    """
    q = np.asarray(box_quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, _z = q
    roll = 2.0 * np.arctan2(x, w)
    pitch = 2.0 * np.arctan2(y, w)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    q_recomposed = np.array([cp * cr, cp * sr, sp * cr, -sp * sr])
    dot = np.clip(abs(np.dot(q, q_recomposed)), -1.0, 1.0)
    residual_rad = 2.0 * np.arccos(dot)
    return np.array([roll, pitch]), float(residual_rad)


def process_real_run(run_dir, env, inference_fn, meta, protocol, config_id,
                      policy_run_id, checkpoint_hash, horizon, rng,
                      run_determinism_check=False, condition_id=None, cal=None):
    """Returns None (SKIPPED -- caller must not write output) if the real run
    folder's ur3_pick_meta.json says stop_reason == "abort" (D16): an aborted
    real episode has no full-length pair to mirror against, per D16/D17.

    `cal`: `load_base_calibration()`'s return value (or None -- box_pos/
    box_quat used raw, the pre-D25 behaviour). See the module's calibration
    note above `load_base_calibration` for why this exists: every
    measured_init.json on disk stores RAW mocap-world box_pos/box_quat, not
    the robot-base/sim frame `reset_to_state` needs.
    """
    stop_reason = _real_run_stop_reason(run_dir)
    if stop_reason == "abort":
        print(f"  SKIPPING {run_dir}: real run aborted (stop_reason='abort', "
              f"see its ur3_pick_meta.json) -- no full-length real pair to "
              f"mirror. Not written to sim_results/ (D16).")
        return None

    with open(os.path.join(run_dir, "measured_init.json")) as f:
        m = json.load(f)

    episode_id, repeat = m["episode_id"], m["repeat"]
    arm_qpos = np.asarray(m["measured_arm_qpos"], dtype=float)
    finger = float(m["commanded_finger"])
    # D25 (2026-07-29): measured_init.json stores RAW mocap-world box_pos/
    # box_quat -- see the module's calibration note. Calibrate here, once,
    # before anything downstream (target diagnostic, cube geometry is frame-
    # independent so unaffected, C1/D1 lifter derivation, reset_to_state).
    box_pos_raw = np.asarray(m["box_pos"], dtype=float)
    box_quat_raw = np.asarray(m["box_quat_wxyz"], dtype=float)
    box_pos = mocap_pos_to_base(box_pos_raw, cal)
    box_quat = mocap_quat_to_base(box_quat_raw, cal)
    if cal is None:
        print(f"  WARNING: {run_dir}: no base-frame calibration loaded -- "
              f"box_pos/box_quat used RAW (mocap-world), NOT the robot-base "
              f"frame reset_to_state expects. This mirror will place the box "
              f"in the wrong location.")

    # D17: cube_size drives the box geom override -- see CUBE_HALF_EXTENTS.
    # Missing key (pre-D17 measured_init.json) defaults to "3cm" (the sim
    # nominal, a no-op) -- flagged loudly rather than silently assumed, since
    # a genuinely-4cm run with a missing/stale cube_size field would silently
    # mirror the wrong geometry otherwise.
    if "cube_size" not in m:
        print(f"  WARNING: {run_dir}'s measured_init.json has no 'cube_size' "
              f"field (pre-D17 log) -- defaulting to '3cm' (nominal, a "
              f"no-op override). If this run actually used the 4cm cube, "
              f"this mirror will be WRONG.")
    cube_size = m.get("cube_size", "3cm")
    if cube_size not in CUBE_HALF_EXTENTS:
        raise SystemExit(
            f"{run_dir}'s measured_init.json has cube_size={cube_size!r}, not "
            f"one of {sorted(CUBE_HALF_EXTENTS)} -- refusing to guess a geometry."
        )
    cube_half_extents = np.asarray(CUBE_HALF_EXTENTS[cube_size], dtype=float)

    # D25 (2026-07-29): recover the table geometry override for C1/D1 from
    # the measured box pose -- see the module note above
    # _TABLE_GEOMETRY_MISMATCH_CONDITIONS and `_decompose_tilt_quat`. Every
    # other condition stays None/None (nominal flat table, unchanged).
    lifter_top_height = None
    lifter_tilt_rp = None
    if condition_id == "D1":
        lifter_top_height = float(box_pos[2] - cube_half_extents[2])
        print(f"  {run_dir} (condition D1): derived lifter_top_height="
              f"{lifter_top_height:.4f} m from measured box_pos[2]="
              f"{box_pos[2]:.4f} - cube_half_z={cube_half_extents[2]:.4f}.")
    elif condition_id == "C1":
        lifter_tilt_rp, residual_rad = _decompose_tilt_quat(box_quat)
        print(f"  {run_dir} (condition C1): derived lifter_tilt_rp="
              f"[{lifter_tilt_rp[0]:.4f}, {lifter_tilt_rp[1]:.4f}] rad "
              f"({np.degrees(lifter_tilt_rp)} deg) from measured box_quat, "
              f"decomposition residual={np.degrees(residual_rad):.3f} deg.")
        if residual_rad > 0.02:  # ~1.1 deg -- generous but non-trivial
            print(f"  WARNING: {run_dir} tilt-decomposition residual "
                  f"{np.degrees(residual_rad):.2f} deg is non-negligible -- "
                  f"the measured box orientation may include a real yaw "
                  f"component this roll/pitch-only decomposition can't "
                  f"separate from tilt. Using the decomposed roll/pitch "
                  f"anyway (best available reconstruction).")

    # D25 (2026-07-29): USE THE STORED target, do not recompute-and-override.
    # Before this session, recomputing from box_xy and trusting the
    # recomputed value was correct -- it only existed to catch gap_target.py/
    # gen_dr_ladder drift, under the assumption box_xy was already correct.
    # That assumption is now known false: run_gap_protocol.py computed the
    # STORED target from RAW (uncalibrated) box_xy, and that stored value is
    # what the real robot was ACTUALLY judged against during the episode.
    # Recomputing from the newly-calibrated box_xy here would silently score
    # the sim rollout against a DIFFERENT goal than the real one used --
    # breaking the matched-init comparison Commit 7 exists for. Matthias's
    # call (2026-07-29): keep the stored target as ground truth; the
    # recompute below is now a DIAGNOSTIC ONLY (uses the calibrated box_xy,
    # so a large discrepancy here is a real signal that run_gap_protocol.py's
    # target computation was meaningfully affected by the frame bug -- see
    # the module's calibration note -- not evidence of gap_target.py drift).
    target_pos = np.asarray(m["target_pos"], dtype=float)
    diagnostic_target, _components = gap_target.target_for_episode(
        protocol.protocol_id, episode_id, config_id,
        box_xy=box_pos[:2], xml_path=MODEL_PATH,
    )
    target_discrepancy_m = float(np.linalg.norm(diagnostic_target - target_pos))
    if target_discrepancy_m > 1e-6:
        print(
            f"  {'WARNING' if target_discrepancy_m > 0.005 else 'note'}: "
            f"{run_dir}: target computed from CALIBRATED box_xy "
            f"({diagnostic_target}) differs from the STORED target "
            f"({target_pos}) by {target_discrepancy_m*1000:.1f} mm. Using "
            f"the STORED value (what the real robot was actually judged "
            f"against) -- this discrepancy is diagnostic evidence of how "
            f"much run_gap_protocol.py's target computation was affected by "
            f"the pre-calibration-fix box_xy bug, not a sign gap_target.py "
            f"has drifted."
        )

    if run_determinism_check:
        _check_determinism(env, inference_fn, rng, arm_qpos, finger, box_pos,
                            box_quat, target_pos, horizon, cube_half_extents,
                            lifter_top_height, lifter_tilt_rp)

    df = rollout_one(env, inference_fn, rng, arm_qpos, finger, box_pos,
                      box_quat, target_pos, horizon, cube_half_extents,
                      lifter_top_height, lifter_tilt_rp)
    return episode_id, repeat, df, {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "config_id": config_id,
        "episode_id": episode_id,
        "repeat": repeat,
        "cube_size": cube_size,  # D17 -- NEW
        "condition_id": condition_id,  # D23 -- NEW, None for pre-D23 real runs
        # D25 -- NEW. None/None for every run collected so far (see the
        # module note above _TABLE_GEOMETRY_MISMATCH_CONDITIONS) -- recorded
        # so a later reader can tell a nominal-table mirror from a matched one
        # without re-deriving it from condition_id.
        "lifter_top_height_m": lifter_top_height,
        "lifter_tilt_rp_rad": (
            None if lifter_tilt_rp is None else lifter_tilt_rp.tolist()
        ),
        # D25 -- NEW. Whether the box_pos/box_quat fed to reset_to_state were
        # calibrated (mocap-world -> robot-base frame) before use, and by how
        # much the stored target diverges from what calibrated box_xy would
        # give -- see the module's calibration note and the target-handling
        # comment above. cal_applied=False means this mirror used RAW mocap
        # box_pos/box_quat (wrong frame) -- flag any such result as unusable.
        "cal_applied": cal is not None,
        "target_discrepancy_mm": round(target_discrepancy_m * 1000, 3),
        "box_pos_raw": box_pos_raw.tolist(),
        "box_quat_raw_wxyz": box_quat_raw.tolist(),
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
        # D17: --protocol_only has no real measured_init.json to read a
        # cube_size from (no real run exists yet -- see the module
        # docstring's D10-bootstrap note); it always uses the nominal 3cm
        # box (cube_half_extents=None -> reset_to_state's pre-D17 default),
        # so this is recorded as "3cm" for schema consistency with
        # process_real_run's output, not read from anywhere.
        "cube_size": "3cm",
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


def run_dir_out(out_root, config_id, protocol_id, episode_id, repeat, cube_size=None,
                 condition_id=None):
    """D17/D23: mirrors the real runner's folder key exactly (`cube_size`
    suffix when given, `condition_id` prefix when given) so a 3cm/4cm or
    A1/B1 sim mirror of the same (config, episode, repeat) can never collide
    -- same rationale as robots/UR3e/run_gap_protocol.py's `run_dir`.
    `cube_size=None`/`condition_id=None` (the --protocol_only path, and any
    pre-D17/D23 caller) keeps the old "ep{ID}_rep{K}" layout.
    """
    base = f"ep{episode_id}_rep{repeat}"
    if cube_size is not None:
        base = f"{base}_{cube_size}"
    if condition_id is not None:
        base = f"{condition_id}_{base}"
    return os.path.join(out_root, config_id, protocol_id, base)


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
    ap.add_argument("--base_calibration", default=DEFAULT_BASE_CALIBRATION,
                     help="D25: mocap-world -> robot-base transform applied to every "
                          "real run's box_pos/box_quat before use (see the module's "
                          "calibration note). Pass an empty string to force RAW mocap "
                          "(pre-D25 behaviour) -- not recommended, see that note.")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol, dry_run=True)
    print(f"Protocol {protocol.protocol_id}: {len(protocol)} episodes, "
          f"horizon={protocol.horizon}, hash={protocol.poses_sha256[:16]}...")

    cal = load_base_calibration(args.base_calibration) if args.base_calibration else None
    if cal is not None:
        print(f"[cal] loaded base-frame calibration from {cal['path']}")
    elif not args.protocol_only:
        print("[cal] running WITHOUT base-frame calibration -- every real-run "
              "mirror below will use RAW mocap-world box_pos/box_quat. See the "
              "module's calibration note above load_base_calibration.")

    if args.protocol_only:
        episodes = [e for e in protocol.episodes
                    if args.episodes is None or e.episode_id in args.episodes]
        pending = []
        for ep in episodes:
            out_dir = run_dir_out(args.out_root, args.config_id, protocol.protocol_id,
                                   ep.episode_id, 0)
            if os.path.exists(out_dir) and not args.force:
                continue
            pending.append((ep, out_dir, None))  # no condition_id in this mode
        print(f"{len(pending)}/{len(episodes)} protocol episodes pending "
              f"(--protocol_only, nominal box).")
    else:
        run_dirs = _find_real_run_dirs(args.real_root, args.config_id,
                                        protocol.protocol_id, args.episodes)
        pending = []
        for rd in run_dirs:
            episode_id, repeat, cube_size, condition_id = _parse_ep_rep(rd)
            out_dir = run_dir_out(args.out_root, args.config_id, protocol.protocol_id,
                                   episode_id, repeat, cube_size, condition_id)
            if os.path.exists(out_dir) and not args.force:
                continue
            pending.append((rd, out_dir, condition_id))
        print(f"{len(pending)}/{len(run_dirs)} real run folders pending "
              f"under {args.real_root}/{args.config_id}/{protocol.protocol_id}/.")

    if args.dry_run:
        n_abort = 0
        for item, out_dir, condition_id in pending:
            src = item if isinstance(item, str) else f"protocol ep{item.episode_id}"
            tag = ""
            if isinstance(item, str):
                # Pure JSON read (no jax/mjx import) -- safe under --dry_run.
                sr = _real_run_stop_reason(item)
                if sr == "abort":
                    tag = "  [will be SKIPPED: real run aborted (D16)]"
                    n_abort += 1
                elif condition_id in _TABLE_GEOMETRY_MISMATCH_CONDITIONS:
                    # D25: the override is DERIVED at process time from the
                    # measured box pose (see _decompose_tilt_quat / the
                    # module note), not read from a stored field -- so this
                    # is unconditional for C1/D1, unlike the abort check
                    # above which genuinely depends on file contents.
                    tag = (
                        f"  [condition {condition_id}: table geometry will be "
                        f"DERIVED from the measured box pose -- lifter height "
                        f"for D1, tilt for C1, see _decompose_tilt_quat]"
                    )
            print(f"  {src}  ->  {out_dir}{tag}")
        if n_abort:
            print(f"({n_abort}/{len(pending)} pending folders are aborted real "
                  f"runs -- D16, no full-length pair to mirror.)")
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
    skipped_aborted = 0
    for i, (item, out_dir, condition_id) in enumerate(pending):
        rng = jax.random.fold_in(base_rng, i)  # distinct-but-deterministic per folder
        run_determinism_check = (i == 0)
        if args.protocol_only:
            result = process_protocol_episode(
                item, env, inference_fn, protocol, args.config_id,
                args.policy_run_id, checkpoint_hash, protocol.horizon, rng,
                run_determinism_check=run_determinism_check,
            )
        else:
            # D16: None means the real run folder was aborted -- no
            # full-length pair to mirror, nothing written for it.
            result = process_real_run(
                item, env, inference_fn, meta, protocol, args.config_id,
                args.policy_run_id, checkpoint_hash, protocol.horizon, rng,
                run_determinism_check=run_determinism_check,
                condition_id=condition_id, cal=cal,
            )
        if result is None:
            skipped_aborted += 1
            continue
        episode_id, repeat, df, meta_out = result
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "sim_states.csv"), index=False)
        with open(os.path.join(out_dir, "sim_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, default=str)
        cond_tag = f"{condition_id}_" if condition_id else ""
        print(f"  {cond_tag}ep{episode_id}_rep{repeat}: return={meta_out['episode_return']:.2f} "
              f"success={meta_out['success']}  cube_size={meta_out.get('cube_size')} "
              f"->  {out_dir}")
        completed += 1

    print(f"\nDone: {completed}/{len(pending)} pending runs completed "
          f"({skipped_aborted} skipped: real run aborted, D16).")


if __name__ == "__main__":
    main()
