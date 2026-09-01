"""Generate the L0 -> L0.5 -> L1 -> L4 curriculum rung spec (6 rungs).

The ladder is NOT restated here. `_CONFIGS`, `EPISODE_LENGTH` and
`WANDB_PROJECT` are imported from `batch_runs/sweeps/gen_dr_ladder.py`, which is
the single source of truth for what each rung randomises. If the two ever drift,
`test_curriculum_spec.py` fails on exact dict equality rather than training a
ladder that no longer matches the DR study it is supposed to warm-start.

This script must run in a bare Python with no jax/mjx installed -- same
constraint gen_dr_ladder.py carries, and the reason that module holds literals
instead of importing ur3_pick.

Run:
    python batch_runs/curriculum/gen_curriculum.py --seed 0
    python batch_runs/curriculum/gen_curriculum.py --seed 0 --force
"""

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from batch_runs.sweeps.gen_dr_ladder import (  # noqa: E402
    _CONFIGS,
    EPISODE_LENGTH,
    WANDB_PROJECT,
)

# The axes UR3Pick._randomize_physics actually applies. `domain_rand.enable` is
# the static master switch for THESE ONLY -- action_delay, cube_force,
# joint_torque and obs_noise are gated independently (gen_dr_ladder.py:41-46).
# Any rung touching one of these must therefore ALSO set enable: true, or the
# randomisation silently does nothing and the rung trains at L1.
# NOTE the `.enable` suffix: the ladder sets `domain_rand.cube_mass.enable`,
# not `domain_rand.cube_mass`. Writing the bare names here produced a guard that
# matched nothing and silently passed on every rung -- caught by mutation-testing
# it, not by reading it.
PHYSICS_AXES = (
    "domain_rand.cube_mass.enable",
    "domain_rand.cube_friction.enable",
    "domain_rand.cube_size_xy.enable",
    "domain_rand.cube_size_z.enable",
    "domain_rand.gravity.enable",
    "domain_rand.arm_stiffness.enable",
    "domain_rand.arm_damping.enable",
)

# _LADDER_EXPECTED_ORDER guards gen_dr_ladder._CONFIGS's shape ONLY (5 entries,
# unmodified -- see the module docstring and CLAUDE.md: _CONFIGS stays at
# exactly 5, three external consumers hardcode that count). It is NOT the
# curriculum's own rung order any more -- that is the public EXPECTED_ORDER
# below, which has 6 entries once L0.5 is inserted.
_LADDER_EXPECTED_ORDER = (
    "L0_none",
    "L1_pos",
    "L2_pos_cube",
    "L3_pos_cube_robot",
    "L4_full",
)

# The curriculum's own 6-rung order. L0_5_light sits between L0_none and
# L1_pos: L1_pos now warm-starts from L0_5_light, not from L0_none.
EXPECTED_ORDER = (
    "L0_none",
    "L0_5_light",
    "L1_pos",
    "L2_pos_cube",
    "L3_pos_cube_robot",
    "L4_full",
)

# --- L0.5 "light" rung -------------------------------------------------------
# *** THE HIGHEST-CONSEQUENCE TRAP IN THIS WHOLE CHANGE ***
# run_experiment.py:~1005-1188 forwards each env key behind `if key in cfg`.
# An OMITTED key therefore falls back to the BAKED ENV DEFAULT, not to L0. A
# sparse two-key overrides dict (just init_start_random + box_xy_jitter) would
# silently train L0.5 WIDER than L1 on every key it left out -- e.g.
# lifter_tilt_max would fall back to the env default (0.12 rad) instead of
# L0's pinned 0.0 -- with no error raised anywhere. It would look fine and be
# scientifically void.
#
# So L0.5's dict is built from ALL 13 of L0_none's keys, found by ID LOOKUP
# (never by index -- _CONFIGS[0] would silently break if the ladder's own
# order ever changed -- and never retyped by hand), with exactly two keys
# overridden.
_L0 = next(o for cid, o, _ in _CONFIGS if cid == "L0_none")
L0_5_LIGHT_OVERRIDES = {
    **_L0,
    "init_start_random": "light",
    "box_xy_jitter": [0.085, 0.12],
}

SPEC_PATH = os.path.join(REPO, "batch_runs", "curriculum", "UR3Pick_curriculum.json")

# 40M is the per-rung CAP, not a target: the windowed-trend tracker is expected
# to cut most rungs well short of it. num_evals=42 is not a round number picked
# for readability -- it is chosen so brax's quantized eval cadence lands on
# EXACTLY the interval the eval noise was characterized at:
#
#   env_step_per_training_step = batch_size*unroll_length*num_minibatches
#                                 *action_repeat = 512*10*32*1 = 163_840
#   num_training_steps_per_epoch = ceil(num_timesteps /
#       ((num_evals-1)*env_step_per_training_step*max(num_resets_per_eval,1)))
#   interval = num_training_steps_per_epoch * env_step_per_training_step
#
# At 40_000_000/42: ceil(40_000_000/(41*163_840)) = 6 -> interval EXACTLY
# 983_040, total 41*983_040 = 40_304_640 (0.76% overshoot of the 40M cap).
# 983_040 is the SAME cadence the v1 ladder's eval-to-eval noise was measured
# at, which is why 42 -- not some other num_evals -- was chosen here.
#
# NOTE the old 24M/30 comment this replaced claimed "~800k steps per eval" --
# that was WRONG for the same quantization reason: ceil(24_000_000/(29*163_840))
# = 6 too, so the real v1 interval was also 983_040, not 24_000_000//29
# = 827_586.
NUM_TIMESTEPS = 40_000_000
NUM_EVALS = 42

# --- Defence-best training hyperparameters -----------------------------------
# WHY THIS BLOCK EXISTS AT ALL: the v1 ladder inherited gen_dr_ladder's
# deliberate silence on action_scale (see that module's docstring: it only
# sets .enable flags plus the L0 position literals, on purpose) and therefore
# trained every rung at the env default action_scale=0.015 instead of 0.04.
# L0 peaked at eval/episode_success 0.016 against the archived
# L0_none_vel_s1's 1.000 -- the whole ladder was roughly 4x slower than it
# should have been for want of one number. These constants pin the
# defence-best values directly into `defaults` so that omission cannot
# recur. All sourced from W&B run Snappy2_as04_ar70_g01_s1.
#
# Deliberately NOT imported from gen_dr_ladder_velocity.py: that module sets
# gripper_action_scale=0.01, which conflicts with the defence value 0.02 below
# and would trade one wrong number for another.
_DEFENCE_ACTION_SCALE = 0.04            # Snappy2_as04_ar70_g01_s1
_DEFENCE_GRIPPER_ACTION_SCALE = 0.02    # Snappy2_as04_ar70_g01_s1
_DEFENCE_ACTION_RATE = -0.7             # Snappy2_as04_ar70_g01_s1
_DEFENCE_ENTROPY_COST = 0.02            # Snappy2_as04_ar70_g01_s1
_DEFENCE_LEARNING_RATE = 3e-4           # Snappy2_as04_ar70_g01_s1
_DEFENCE_REWARD_SCALING = 0.03          # Snappy2_as04_ar70_g01_s1
_DEFENCE_NETWORK_FACTORY = {            # Snappy2_as04_ar70_g01_s1
    "policy_hidden_layer_sizes": [256, 256, 256],
    "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
}
_DEFENCE_GATE_GRIPPER_ALIGN_ON_LIFT = True   # Snappy2_as04_ar70_g01_s1
_DEFENCE_GATE_GRIPPER_BOX_ON_LIFT = False    # Snappy2_as04_ar70_g01_s1
_DEFENCE_NUM_EVAL_ENVS = 256                 # Snappy2_as04_ar70_g01_s1


def build_spec(seed: int = 0, group: str = None) -> dict:
    ids = [c[0] for c in _CONFIGS]
    if tuple(ids) != _LADDER_EXPECTED_ORDER:
        raise SystemExit(
            f"gen_dr_ladder._CONFIGS changed shape.\n  expected "
            f"{_LADDER_EXPECTED_ORDER}\n  got      {tuple(ids)}\n"
            f"Re-derive the curriculum before generating."
        )

    # {config_id: overrides} for every ladder rung, plus L0.5 injected. L0.5's
    # overrides are the module constant above -- not recomputed here -- so the
    # spec test can assert on-disk equality against the very same object.
    by_id = {cid: overrides for cid, overrides, _tags in _CONFIGS}
    by_id["L0_5_light"] = L0_5_LIGHT_OVERRIDES

    rungs = []
    for i, config_id in enumerate(EXPECTED_ORDER):
        rungs.append({
            "config_id": config_id,
            "run_id": f"Curr_{config_id}_s{seed}",
            # Each rung warm-starts from its IMMEDIATE predecessor. No rung
            # skips, none self-references -- asserted in the spec tests.
            "warm_start_from": None if i == 0 else EXPECTED_ORDER[i - 1],
            # Exactly the ladder entry, byte for byte, for every rung except
            # L0.5 (which has no ladder entry -- it is the module constant
            # above). Not a copy with edits.
            "overrides": dict(by_id[config_id]),
        })

    return {
        # null => the driver stamps a unique group at launch. Baking a timestamp
        # into the file here would make regeneration non-reproducible, which the
        # spec tests forbid; two ladder jobs on one day must still not collide.
        "wandb_group": group,
        "wandb_group_prefix": "curriculum",
        "wandb_project": WANDB_PROJECT,
        "seed": int(seed),
        "defaults": {
            "env_name": "UR3Pick",
            # NEVER inherited: ur3_pick.default_config()'s 250 is stale and
            # omitting this once trained 30 runs at the wrong horizon.
            "episode_length": EPISODE_LENGTH,
            "obs_include_velocity": True,
            "num_timesteps": NUM_TIMESTEPS,
            "num_evals": NUM_EVALS,
            "num_resets_per_eval": 1,
            # Defence-best hyperparameters -- see the module-level comment
            # above _DEFENCE_ACTION_SCALE for why this block exists at all.
            "action_scale": _DEFENCE_ACTION_SCALE,
            "gripper_action_scale": _DEFENCE_GRIPPER_ACTION_SCALE,
            "action_rate": _DEFENCE_ACTION_RATE,
            "entropy_cost": _DEFENCE_ENTROPY_COST,
            "learning_rate": _DEFENCE_LEARNING_RATE,
            "reward_scaling": _DEFENCE_REWARD_SCALING,
            "network_factory": dict(_DEFENCE_NETWORK_FACTORY),
            "gate_gripper_align_on_lift": _DEFENCE_GATE_GRIPPER_ALIGN_ON_LIFT,
            "gate_gripper_box_on_lift": _DEFENCE_GATE_GRIPPER_BOX_ON_LIFT,
            "num_eval_envs": _DEFENCE_NUM_EVAL_ENVS,
            # Windowed-trend convergence stop (see
            # learning/curriculum/early_stop.py:WindowedTrendTracker /
            # build_tracker). Replaces the v1 running-max ratchet, which fired
            # on eval noise (6.5% median on L3, 4.7% on L4) rather than actual
            # convergence.
            "early_stop": {
                "strategy": "windowed",
                "window": 4,
                "patience": 3,
                "min_delta": 0.02,
                "min_steps": 6_000_000,
            },
            "normalizer_count_reset": None,
        },
        "rungs": rungs,
    }


def dumps(spec: dict) -> str:
    """One canonical rendering, so regeneration is byte-identical."""
    return json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", default=None,
                    help="pin the W&B group; default null => driver stamps it")
    ap.add_argument("--out", default=SPEC_PATH)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing spec")
    args = ap.parse_args(argv)

    text = dumps(build_spec(seed=args.seed, group=args.group))

    if os.path.exists(args.out) and not args.force:
        if open(args.out, encoding="utf-8").read() == text:
            print(f"unchanged: {args.out}")
            return 0
        raise SystemExit(
            f"refusing to overwrite {args.out} (content differs).\n"
            f"Re-run with --force if that is intended."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(build_spec(args.seed, args.group)['rungs'])} rungs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
