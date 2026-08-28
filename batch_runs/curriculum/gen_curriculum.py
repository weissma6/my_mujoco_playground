"""Generate the L0 -> L4 curriculum rung spec.

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

EXPECTED_ORDER = (
    "L0_none",
    "L1_pos",
    "L2_pos_cube",
    "L3_pos_cube_robot",
    "L4_full",
)

SPEC_PATH = os.path.join(REPO, "batch_runs", "curriculum", "UR3Pick_curriculum.json")

# 24M is the per-rung CAP, not a target: the patience tracker is expected to cut
# most rungs well short of it. num_evals=30 puts an eval every ~800k steps, which
# is the resolution the tracker's patience=5 is tuned against.
NUM_TIMESTEPS = 24_000_000
NUM_EVALS = 30


def build_spec(seed: int = 0, group: str = None) -> dict:
    ids = [c[0] for c in _CONFIGS]
    if tuple(ids) != EXPECTED_ORDER:
        raise SystemExit(
            f"gen_dr_ladder._CONFIGS changed shape.\n  expected {EXPECTED_ORDER}\n"
            f"  got      {tuple(ids)}\nRe-derive the curriculum before generating."
        )

    rungs = []
    for i, (config_id, overrides, _tags) in enumerate(_CONFIGS):
        rungs.append({
            "config_id": config_id,
            "run_id": f"Curr_{config_id}_s{seed}",
            # Each rung warm-starts from its IMMEDIATE predecessor. No rung
            # skips, none self-references -- asserted in the spec tests.
            "warm_start_from": None if i == 0 else _CONFIGS[i - 1][0],
            # Exactly the ladder entry, byte for byte. Not a copy with edits.
            "overrides": dict(overrides),
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
            "early_stop": {
                "patience": 5,
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
