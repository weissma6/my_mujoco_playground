"""Generate the DR-ladder + LOO sweep JSONL for the sim-to-real gap study.

See the vault plan "Plan - Sim-to-Real Gap Protocol" (VT2-SimToReal-Robotics)
for the full research design. This script emits exactly the 10 configs x 3
seeds = 30 lines specified there:

  Ladder (monotone dose-response):
    L0_none              -- everything off: deterministic position, no physics DR
    L1_pos                + position DR (== the current baked default behaviour)
    L2_pos_cube            + cube physics (mass/friction/size)
    L3_pos_cube_robot      + robot physics (arm stiffness/damping) + action delay
    L4_full                + environment (gravity + cube perturbation force)
    L5_full_obs             + observation noise (arm q/finger/box pos/orientation)

  Leave-one-out from L5_full_obs (the ladder is nested and cannot attribute
  which cluster carries the transfer effect on its own):
    LOO_no_pos, LOO_no_cube, LOO_no_robot, LOO_no_env
    (LOO_no_obs is NOT emitted -- it is identical to L4_full)

Cluster -> env_overrides / domain_rand.* key mapping (Commit 2, already
implemented in ur3_pick.py):
  position : box_xy_jitter, target_z_jitter, finger_random_init,
             box_z_rot_range, lifter_height_max, lifter_tilt_max,
             init_qpos_noise, init_start_random, target_r_min/max,
             target_azim_min/max
             -- these are NOT under domain_rand; L1's "on" state is simply
             the baked default_config() values (no keys need to be set at
             all). L0/LOO_no_pos collapse them to a single deterministic
             point using the SAME literal values that used to be hardcoded.
  cube     : domain_rand.cube_mass / cube_friction / cube_size
  robot    : domain_rand.arm_stiffness / arm_damping / action_delay
  env      : domain_rand.gravity / cube_force
  obs      : domain_rand.obs_noise

domain_rand.enable is the STATIC master switch for ONLY the original 6
physics axes (cube_mass/cube_friction/cube_size/gravity/arm_stiffness/
arm_damping) -- see UR3Pick._randomize_physics. action_delay, cube_force,
and obs_noise are gated independently of it (see ur3_pick.py step()/
_get_obs). So every line touching ANY of the 6 original axes must ALSO set
domain_rand.enable: true, even when some of those 6 are left individually
disabled (e.g. LOO_no_cube still needs domain_rand.enable=true for gravity/
arm_stiffness to fire, with cube_mass/friction/size explicitly false).

Every range/value used below is a BAKED DEFAULT already present in
ur3_pick.default_config() -- this script only sets .enable flags (plus the
deterministic-position literals for L0/LOO_no_pos). No range is
re-invented here; see default_config() for the single source of truth.

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
    "target_z_jitter": [0.195, 0.195],       # midpoint of baked [0.18, 0.21]
    "finger_random_init": False,
    "box_z_rot_range": 0.0,
    "lifter_height_max": 0.0,
    "lifter_tilt_max": 0.0,
    "init_qpos_noise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "init_start_random": "none",
    "target_r_min": 0.335,                   # midpoint of baked [0.25, 0.42]
    "target_r_max": 0.335,
    "target_azim_min": 0.7854,                # midpoint of baked [0.5236, 1.0472]
    "target_azim_max": 0.7854,
}

_CUBE = {
    "domain_rand.cube_mass.enable": True,
    "domain_rand.cube_friction.enable": True,
    "domain_rand.cube_size.enable": True,
}
_ROBOT = {
    "domain_rand.arm_stiffness.enable": True,
    "domain_rand.arm_damping.enable": True,
    "domain_rand.action_delay.enable": True,
}
_ENV = {
    "domain_rand.gravity.enable": True,
    "domain_rand.cube_force.enable": True,
}
_OBS = {
    "domain_rand.obs_noise.enable": True,
}


def _off(d):
    """Same keys as `d`, all forced False -- for explicit LOO exclusion."""
    return {k: False for k in d}


# Ordered so the JSONL reads top-to-bottom as the ladder, then the LOO block.
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
_add(
    "L4_full",
    {"domain_rand.enable": True, **_CUBE, **_ROBOT, **_ENV},
    ["DR_ladder", "L4", "position", "cube", "robot", "env"],
)
_add(
    "L5_full_obs",
    {"domain_rand.enable": True, **_CUBE, **_ROBOT, **_ENV, **_OBS},
    ["DR_ladder", "L5", "position", "cube", "robot", "env", "obs"],
)

# --- Leave-one-out from L5_full_obs (attribution) -------------------------
# LOO_no_obs is NOT emitted: it is byte-identical to L4_full (see docstring).
_add(
    "LOO_no_pos",
    {
        **_DETERMINISTIC_POSITION,
        "domain_rand.enable": True,
        **_CUBE,
        **_ROBOT,
        **_ENV,
        **_OBS,
    },
    ["DR_LOO", "no_pos", "cube", "robot", "env", "obs"],
)
_add(
    "LOO_no_cube",
    {
        "domain_rand.enable": True,
        **_off(_CUBE),
        **_ROBOT,
        **_ENV,
        **_OBS,
    },
    ["DR_LOO", "position", "no_cube", "robot", "env", "obs"],
)
_add(
    "LOO_no_robot",
    {
        "domain_rand.enable": True,
        **_CUBE,
        **_off(_ROBOT),
        **_ENV,
        **_OBS,
    },
    ["DR_LOO", "position", "cube", "no_robot", "env", "obs"],
)
_add(
    "LOO_no_env",
    {
        "domain_rand.enable": True,
        **_CUBE,
        **_ROBOT,
        **_off(_ENV),
        **_OBS,
    },
    ["DR_LOO", "position", "cube", "robot", "no_env", "obs"],
)

_HEADER = f"""\
# DR-ladder + LOO sweep -- "Plan - Sim-to-Real Gap Protocol" (VT2-SimToReal-Robotics).
# Generated by batch_runs/sweeps/gen_dr_ladder.py -- DO NOT hand-edit; regenerate
# instead so the config table stays the single source of truth (see that script's
# docstring for the full cluster -> key mapping and the domain_rand.enable caveat).
#
# DRY: base config lives in manipulation_params.py (PPO) + ur3_pick.py
# default_config() (env). Every line below carries ONLY per-run metadata plus the
# DR-cluster .enable flags (or the deterministic-position literals for L0/
# LOO_no_pos); every omitted key inherits the current baked defaults. Training
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

    lines = build_lines(args.seeds)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_runs = len(_CONFIGS) * len(args.seeds)
    print(f"Wrote {n_runs} runs ({len(_CONFIGS)} configs x {len(args.seeds)} "
          f"seeds) to {args.out}")
    assert {c for c, _, _ in _CONFIGS} == {
        "L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full",
        "L5_full_obs", "LOO_no_pos", "LOO_no_cube", "LOO_no_robot",
        "LOO_no_env",
    }, "config set drifted from the plan's 10 configs -- update this assertion"


if __name__ == "__main__":
    main()
