"""Generate the DR-ladder sweep JSONL for the sim-to-real gap study.

See the vault plan "Plan - Sim-to-Real Gap Protocol" (VT2-SimToReal-Robotics),
decision D21 (2026-07-29), for the full research design. D21 SUPERSEDES the
prior 10-config/30-run ladder and its leave-one-out attribution block. This
script now emits exactly 5 configs x 3 seeds = 15 lines:

  Ladder (monotone dose-response):
    L0_none              -- everything off: deterministic position, no physics DR
    L1_pos                + position DR (== the current baked default behaviour)
    L2_pos_cube            + cube physics (mass/friction/size)
    L3_pos_cube_robot      + robot physics (arm stiffness/damping) + action delay
    L4_full                + environment (gravity + burst cube force + burst
                             joint torque) + observation noise -- this ABSORBS
                             what used to be the separate L5_full_obs rung.

  The LOO block (LOO_no_pos / no_cube / no_robot / no_env) is DROPPED entirely
  (D21): 5 configs is too few to support a leave-one-out attribution study on
  top of the ladder itself.

Cluster -> env_overrides / domain_rand.* key mapping (already implemented in
ur3_pick.py):
  position : box_xy_jitter, target_z_jitter, finger_random_init,
             box_z_rot_range, lifter_height_abs_min/max, lifter_tilt_max,
             init_qpos_noise, init_start_random, target_r_min/max,
             target_azim_min/max
             -- these are NOT under domain_rand; L1's "on" state is simply
             the baked default_config() values (no keys need to be set at
             all). L0 collapses them to a single deterministic point.
             (D18/D24: lifter_height_max was REPLACED by the absolute
             lifter_height_abs_min/lifter_height_abs_max pair.)
  cube     : domain_rand.cube_mass / cube_friction / cube_size_xy / cube_size_z
             (D19: cube size decoupled into independent xy/z half-extent draws,
             replacing the single cube_size scale factor)
  robot    : domain_rand.arm_stiffness / arm_damping / action_delay
  env      : domain_rand.gravity / cube_force / joint_torque
             (D20: cube_force respecced from continuous to bursts;
             joint_torque is a NEW axis on the same burst schedule)
  obs      : domain_rand.obs_noise

domain_rand.enable is the STATIC master switch for ONLY the physics axes
applied in UR3Pick._randomize_physics (cube_mass / cube_friction /
cube_size_xy / cube_size_z / gravity / arm_stiffness / arm_damping).
action_delay, cube_force, joint_torque and obs_noise are gated INDEPENDENTLY
of it (see ur3_pick.py step()/_get_obs). So every line touching ANY of the
_randomize_physics axes must ALSO set domain_rand.enable: true.

Every range/value used below is a BAKED DEFAULT already present in
ur3_pick.default_config() -- this script only sets .enable flags (plus the
deterministic-position literals for L0). No range is re-invented here; see
default_config() for the single source of truth.

episode_length=400 IS set explicitly on every line, and MUST stay that way.
This is the one winner knob that was never baked into default_config():

  b0731d7 set the env default to 250 back when action_scale was 0.01.
  eaabb6b's slowdown sweep -- the one that CHOSE action_scale=0.015
  (Smooth_slow_mid_as015_ar10) -- ran at ep400. 1636c98 then baked that
  winner's action_scale/action_rate (and init_start_random="mid" was
  already default) but NOT its episode_length, and its DR ablation sweep
  kept carrying ep400 per-line instead. So EVERY validated run at the
  current defaults -- DR_baseline_ep400, and DR_cube_mass_light, the
  policy that lifted the cube on the real robot -- trained at 400, while
  ur3_pick.py's default sat stale at 250.

The first version of this script omitted episode_length on purpose, on the
(wrong) DRY premise quoted above that every line "inherits the baked
winner". It inherited the stale 250 instead: 5.0 s of episode at
ctrl_dt=0.02 instead of 8.0 s. gripper_align is approach-only (a fixed
number of steps) while lift/box_target/hold are per-step and gated behind
box_off_rest, so a horizon cut comes almost entirely out of the POST-GRASP
budget -- the policies farmed the dense approach reward and never
bootstrapped the lift. All 30 runs were dead. Do not "clean this up" back
into an inherited default unless ur3_pick.py:episode_length is ALSO moved
to 400 and re-validated.

Deliberately UNCHANGED (Training strategy: shared HPs, pass 1 -- see the
vault plan): num_envs/batch_size/num_minibatches/LR are not overridden
anywhere in this sweep -- every line inherits the current
manipulation_params.py / ur3_pick.default_config() values, matching the
existing UR3Pick_sweep.jsonl convention (DRY: only per-run metadata + the
knob(s) that vary + the ep400 correction above).

Usage:
    .venv/bin/python batch_runs/sweeps/gen_dr_ladder.py \
        [--out batch_runs/sweeps/UR3Pick_dr_ladder.jsonl] [--seeds 0 1 2] \
        [--force]
"""

import argparse
import json
import math
import os

WANDB_PROJECT = "UR3_pick_ppo"

# Episode horizon for EVERY line -- see the module docstring. This matches
# DR_baseline_ep400 / the whole UR3Pick_sweep.jsonl DR ablation exactly, so the
# ladder's L1_pos is a bit-for-bit replicate of that baseline (same seed s1) and
# the two sweeps stay directly comparable in W&B. It is NOT inherited from
# ur3_pick.default_config(), whose 250 is stale. run_experiment.py forwards
# episode_length to BOTH the env config (the at_horizon metric gate) and
# ppo.train's EpisodeWrapper (the real truncation), so this one key keeps both
# horizons consistent.
EPISODE_LENGTH = 400

# The three literal values that collapse "position" to a single deterministic
# point -- SAME numbers that used to be hardcoded in ur3_pick.reset() before
# Commit 2 exposed them as config fields. target_z/r/azim use the MIDPOINT of
# their baked jitter range (zero-width interval -> deterministic draw).
#
# NOTE: evaluation/gap_target.py imports this dict directly (its
# "L0_deterministic" target profile) rather than re-typing target_r_min/max/
# target_azim_min/max/target_z_jitter -- keep this name and its keys stable,
# or update that import alongside any change here.
_DETERMINISTIC_POSITION = {
    "box_xy_jitter": [0.0, 0.0],
    # D24 (2026-07-29): L0's single fixed target IS the D22 eval drop point, so
    # L0 can actually perform the eval task. Drop point = base-frame
    # (0.212, 0.212, 0.165): r = sqrt(0.212^2 + 0.212^2) = 0.2998 ~= 0.30 at
    # azim 45 deg, world z 0.165.
    #   target_z = draw + _init_obj_pos[2](0.115) + (table_top - 0.095)
    # With the table flat at nominal (deviation 0), draw = 0.165 - 0.115 = 0.05.
    # NOT the 0.145 used on the novelocitymodel branch -- there the cube rests
    # on the bare FLOOR so the anchor is 0.02, not 0.115. Porting that number
    # here would put the drop 95 mm too high.
    "target_z_jitter": [0.05, 0.05],
    "finger_random_init": False,
    "box_z_rot_range": 0.0,
    # D18/D24: `lifter_height_max` NO LONGER EXISTS (it was a +- band about the
    # nominal; the table top is now an ABSOLUTE draw). A zero-width absolute
    # range pinned at the nominal == "flat table at the real 95 mm height",
    # which is what a deterministic L0 wants. Passing the old key now fails
    # loud in ur3_pick.py rather than silently meaning something else.
    "lifter_height_abs_min": 0.095,
    "lifter_height_abs_max": 0.095,
    "lifter_tilt_max": 0.0,
    "init_qpos_noise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "init_start_random": "none",
    "target_r_min": 0.30,                     # == D22 eval drop radius
    "target_r_max": 0.30,
    "target_azim_min": 0.7854,                # 45 deg == D22 eval drop azimuth
    "target_azim_max": 0.7854,
}

# ---------------------------------------------------------------------------
# D18/D24 reach-envelope validation, run at GENERATION time so a bad geometry
# is caught before the sweep is emitted, not after burning HPC hours.
#
# These four constants MUST mirror the scene XML / ur3_pick.default_config().
# They are literals rather than an `import ur3_pick` on purpose: this generator
# has to run in a bare Python with no jax/mjx installed (and gap_target.py
# imports THIS module, so pulling MJX in here would drag it into every
# gap-protocol process too). The check below is what keeps them honest -- if
# the env moves and these do not, the emitted envelope stops matching what is
# actually trained, so re-derive both together.
_BOX_CENTER_X = 0.32          # scene XML <body name="box"> / task_home keyframe
_BOX_XY_JITTER = (0.17, 0.24)  # ur3_pick.default_config().box_xy_jitter
_BOX_XY_R_MIN = 0.18          # ur3_pick._BOX_XY_R_MIN (base-column clearance)
_UR3_WORKING_RADIUS_M = 0.49  # ur3_pick._UR3_WORKING_RADIUS_M
# Measured lab table (D24), the surface every spawn has to land on.
_TABLE_X_RANGE = (0.10, 0.70)
_TABLE_Y_RANGE = (-0.50, 0.50)
# D22 eval drop square, and the geometry L0's fixed target must reproduce.
_DROP_SQUARE_CENTER = (0.212, 0.212)
_BOX_Z_ANCHOR = 0.115         # task_home keyframe box Z (cube on the 95 mm table)
_TABLE_TOP_NOM = 0.095        # ur3_pick.default_config().lifter_height_nom


def _validate_reach_envelope():
    """D18/D24: check the emitted geometry is reachable AND on the table.

    Unlike the novelocitymodel version of this check, the box-spawn RECTANGLE
    is deliberately allowed to stick out past the reach envelope: ur3_pick.py's
    reset() projects every draw onto the annulus r in [_BOX_XY_R_MIN,
    _UR3_WORKING_RADIUS_M] (the D24 radial clip). So the thing worth asserting
    is not "is every rectangle corner reachable" -- it is not, by design -- but
    "does the CLIPPED envelope stay on the table, and is the clip actually
    doing something". Hard-fails (not warnings): every one of these is
    arithmetic this script can settle on its own, with no robot needed.
    """
    cx, (jx, jy) = _BOX_CENTER_X, _BOX_XY_JITTER
    corners = [
        (cx - jx, jy), (cx - jx, -jy), (cx + jx, jy), (cx + jx, -jy),
    ]

    def _clip(x, y):
        """Mirror of reset()'s radial clip (base at the world origin)."""
        r = (x * x + y * y) ** 0.5
        if r <= 0.0:
            return x, y, r
        s = min(max(r, _BOX_XY_R_MIN), _UR3_WORKING_RADIUS_M) / r
        return x * s, y * s, r

    clipped_any = False
    for x, y in corners:
        cxp, cyp, r_raw = _clip(x, y)
        r_new = (cxp * cxp + cyp * cyp) ** 0.5
        if abs(r_new - r_raw) > 1e-12:
            clipped_any = True
        if not (_BOX_XY_R_MIN - 1e-9 <= r_new <= _UR3_WORKING_RADIUS_M + 1e-9):
            raise SystemExit(
                f"D24 reach check FAILED: spawn corner ({x:.3f}, {y:.3f}) "
                f"clips to r={r_new:.4f} m, outside the reachable annulus "
                f"[{_BOX_XY_R_MIN}, {_UR3_WORKING_RADIUS_M}]."
            )
        if not (_TABLE_X_RANGE[0] - 1e-9 <= cxp <= _TABLE_X_RANGE[1] + 1e-9):
            raise SystemExit(
                f"D24 table check FAILED: spawn corner ({x:.3f}, {y:.3f}) "
                f"clips to x={cxp:.4f} m, off the table {_TABLE_X_RANGE}."
            )
        if not (_TABLE_Y_RANGE[0] - 1e-9 <= cyp <= _TABLE_Y_RANGE[1] + 1e-9):
            raise SystemExit(
                f"D24 table check FAILED: spawn corner ({x:.3f}, {y:.3f}) "
                f"clips to y={cyp:.4f} m, off the table {_TABLE_Y_RANGE}."
            )
        print(
            f"  spawn corner ({x:+.3f}, {y:+.3f})  r={r_raw:.4f} -> "
            f"({cxp:+.4f}, {cyp:+.4f}) r={r_new:.4f}  OK"
        )
    if not clipped_any:
        raise SystemExit(
            "D24 reach check FAILED: the radial clip never fires on any spawn "
            "corner, so box_xy_jitter/_BOX_CENTER_X here no longer match the "
            "widened envelope the clip exists for. Re-derive both together."
        )

    # L0's fixed target must BE the D22 eval drop point. Same formula
    # ur3_pick.reset()/gap_target.compute_target_pos use, with the table flat at
    # nominal (L0 pins lifter_height_abs_min == abs_max == nominal, deviation 0):
    #   target_z = target_z_draw + box_z_anchor + (table_top - table_top_nom)
    r0 = _DETERMINISTIC_POSITION["target_r_min"]
    azim0 = _DETERMINISTIC_POSITION["target_azim_min"]
    dev = (
        _DETERMINISTIC_POSITION["lifter_height_abs_min"] - _TABLE_TOP_NOM
    )
    tz = _DETERMINISTIC_POSITION["target_z_jitter"][0] + _BOX_Z_ANCHOR + dev
    # Box nominal bearing from the base is 0 (nominal box is on +x, y=0) and L0
    # zeroes box_xy_jitter, so phi == azim0 and side is frozen to +1 in
    # evaluation/gap_target.py.
    tx = r0 * math.cos(azim0)
    ty = r0 * math.sin(azim0)
    want_r = (
        _DROP_SQUARE_CENTER[0] ** 2 + _DROP_SQUARE_CENTER[1] ** 2
    ) ** 0.5
    if abs(tx - _DROP_SQUARE_CENTER[0]) > 5e-4 or abs(ty - _DROP_SQUARE_CENTER[1]) > 5e-4:
        raise SystemExit(
            f"D22 target check FAILED: L0's fixed target XY resolves to "
            f"({tx:.4f}, {ty:.4f}) but the taped drop square is centred at "
            f"{_DROP_SQUARE_CENTER} (r={want_r:.4f}). Fix target_r_min/max or "
            f"target_azim_min/max in _DETERMINISTIC_POSITION."
        )
    want_z = _TABLE_TOP_NOM + 0.070  # D24: 70 mm drop height above the table top
    if abs(tz - want_z) > 1e-9:
        raise SystemExit(
            f"D22 target check FAILED: L0's fixed target world z resolves to "
            f"{tz:.4f} m but the D24 drop point is {want_z:.4f} m "
            f"(table top {_TABLE_TOP_NOM} + 70 mm). target_z_jitter must be "
            f"{want_z - _BOX_Z_ANCHOR - dev:.4f}, i.e. the lift above the "
            f"cube's RESTING height, not above the floor."
        )
    print(
        f"  L0 fixed target -> ({tx:.4f}, {ty:.4f}, {tz:.4f})  "
        f"== D22 drop square {(*_DROP_SQUARE_CENTER, want_z)}  OK"
    )
    # Air gap the drop actually produces, for the record.
    gap = tz - 0.02 - _TABLE_TOP_NOM
    print(f"  drop: cube centre {tz - _TABLE_TOP_NOM:.3f} m above the table "
          f"top, air gap under the 40 mm cube {gap:.3f} m")


_CUBE = {
    "domain_rand.cube_mass.enable": True,
    "domain_rand.cube_friction.enable": True,
    # D19 (2026-07-29): the single cube_size scale factor was split into two
    # independent absolute half-extent draws (2x2x3 cm .. 4x4x4 cm).
    "domain_rand.cube_size_xy.enable": True,
    "domain_rand.cube_size_z.enable": True,
}
_ROBOT = {
    "domain_rand.arm_stiffness.enable": True,
    "domain_rand.arm_damping.enable": True,
    "domain_rand.action_delay.enable": True,
}
_ENV = {
    "domain_rand.gravity.enable": True,
    "domain_rand.cube_force.enable": True,
    # D20 (2026-07-29): new arm joint-torque burst axis, same cluster and same
    # burst schedule as the respecced cube_force.
    "domain_rand.joint_torque.enable": True,
}
_OBS = {
    "domain_rand.obs_noise.enable": True,
}


def _off(d):
    """Same keys as `d`, all forced False.

    D21 (2026-07-29): no longer used by this module -- its only consumers were
    the dropped LOO configs. Kept because gen_dr_ladder_velocity.py imports
    from here and may still want it; delete once that is confirmed unused.
    """
    return {k: False for k in d}


# Ordered so the JSONL reads top-to-bottom as the ladder (L0 -> L4).
# Each entry: (config_id, overrides_dict, wandb_tags).
_CONFIGS = []


def _add(config_id, overrides, tags):
    _CONFIGS.append((config_id, overrides, tags))


# --- Ladder --------------------------------------------------------------
_add("L0_none", dict(_DETERMINISTIC_POSITION), ["DR_ladder", "L0", "none"])
_add("L1_pos", {}, ["DR_ladder", "L1", "position"])  # baked defaults == position DR on
_add(
    "L2_pos_cube",
    {"domain_rand.enable": True, **_CUBE},
    ["DR_ladder", "L2", "position", "cube"],
)
_add(
    "L3_pos_cube_robot",
    {"domain_rand.enable": True, **_CUBE, **_ROBOT},
    ["DR_ladder", "L3", "position", "cube", "robot"],
)
# L4_full (D21, 2026-07-29): the TOP rung now carries env AND obs together,
# absorbing what used to be the separate L5_full_obs. The whole leave-one-out
# block (LOO_no_pos/cube/robot/env) is DROPPED: 5 configs is too few to
# support a leave-one-out attribution study on top of the ladder itself, and
# cutting it is where the 30-run -> 15-run saving comes from.
_add(
    "L4_full",
    {"domain_rand.enable": True, **_CUBE, **_ROBOT, **_ENV, **_OBS},
    ["DR_ladder", "L4", "position", "cube", "robot", "env", "obs"],
)

_HEADER = f"""\
# DR-ladder sweep -- "Plan - Sim-to-Real Gap Protocol" (VT2-SimToReal-Robotics),
# D21/D24 (2026-07-29): 5 configs, LOO block dropped, geometry re-derived for
# the real 95 mm table on the addvelocity branch.
# Generated by batch_runs/sweeps/gen_dr_ladder.py -- DO NOT hand-edit; regenerate
# instead so the config table stays the single source of truth (see that script's
# docstring for the full cluster -> key mapping and the domain_rand.enable caveat).
#
# DRY: base config lives in manipulation_params.py (PPO) + ur3_pick.py
# default_config() (env). Every line below carries ONLY per-run metadata plus the
# DR-cluster .enable flags (or the deterministic-position literals for L0);
# every omitted key inherits the current baked defaults. Training
# strategy is Pass 1 (shared HPs, no per-level tuning) -- num_envs/batch_size/
# num_minibatches/LR are intentionally NOT touched anywhere here.
#
# EXCEPTION: episode_length=400 is set on every line and is NOT inherited.
# ur3_pick.default_config()'s 250 is STALE (predates action_scale 0.015); every
# validated run -- incl. DR_baseline_ep400 and the real-robot-proven
# DR_cube_mass_light -- trained at 400. Omitting it silently trained all 30 runs
# at 5.0 s instead of 8.0 s and killed the lift phase. See gen_dr_ladder.py's
# docstring before touching this.
#
# {len(_CONFIGS)} configs x N seeds. Comment/blank lines are skipped by the
# runner and do NOT count toward the 1-based SLURM_ARRAY_TASK_ID.
"""


def build_lines(seeds):
    lines = [_HEADER]
    for config_id, overrides, tags in _CONFIGS:
        for seed in seeds:
            entry = {
                "env_name": "UR3Pick",
                "seed": int(seed),
                "run_id": f"{config_id}_s{seed}",
                "wandb_project": WANDB_PROJECT,
                "video_every_evals": 6,
                "render_every": 1,
                "episode_length": EPISODE_LENGTH,
                **overrides,
                "wandb_tags": [*tags, f"s{seed}"],
            }
            lines.append(json.dumps(entry))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "UR3Pick_dr_ladder.jsonl"
        ),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing file"
    )
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite "
            f"(the file should be regenerated from this script, not hand-edited)"
        )

    # D18/D24: validate the geometry BEFORE writing anything, so a bad envelope
    # can never reach HPC. Hard-fails; see _validate_reach_envelope's docstring.
    print("D18/D24 reach + table + drop-target validation:")
    _validate_reach_envelope()

    lines = build_lines(args.seeds)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_runs = len(_CONFIGS) * len(args.seeds)
    print(f"Wrote {n_runs} runs ({len(_CONFIGS)} configs x {len(args.seeds)} "
          f"seeds) to {args.out}")
    assert {c for c, _, _ in _CONFIGS} == {
        "L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full",
    }, "config set drifted from the D21 5-config ladder -- update this assertion"


if __name__ == "__main__":
    main()
