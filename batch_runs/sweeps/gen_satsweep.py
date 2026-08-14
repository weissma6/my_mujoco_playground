"""Generate the saturating-reward PPO sweep JSONL (Tier A screen / Tier B deepen).

Plan: "20260814_Saturating reward - sweep exploration" (VT2-SimToReal-Robotics).

WHAT IS NEW HERE: this is the first sweep in this repo that varies **PPO
hyperparameters**. Every one of the 293 prior runs in `UR3_pick_ppo` took its
PPO config unmodified from `mujoco_playground/config/manipulation_params.py`;
the earlier "sweeps" (gen_dr_ladder*.py, gen_reward_fix_v3_pilot.py) varied
env/DR keys only. The target is the 0.930 -> 0.99 gap between the best
DR-carrying run and the already-solved no-DR rung.

DESIGN -- a full 2^4 factorial, not OFAT. Four binary factors, level 0 always
being the current `manipulation_params.py` baseline, so cell a0b0c0d0 *is* the
reference run and no separate control is needed:

    A  policy_hidden_layer_sizes   (32,32,32,32)  ->  (256,256,256)
    B  entropy_cost                2e-2           ->  5e-3
    C  learning_rate               6e-4           ->  3e-4
    D  reward_scaling              0.05           ->  0.03

A full factorial is deliberate: A x C is expected to interact (a wider actor
plausibly only helps at the lower LR), and one-factor-at-a-time cannot see
that.

THE SHARED BLOCK is not written here. It is loaded from
`target_config_satsweep.json`, which was extracted from the recorded W&B
config of `RealDR_vel_as04_gas02_s0` -- the best DR-carrying run. Hand-writing
it would have meant guessing the keys that define "mid difficulty", and this
repo has been bitten by exactly that twice (`box_xy_jitter`,
`lifter_height_abs_min/max` both silently dropped by an override whitelist).
Keys prefixed `_` in that file are provenance, not config, and are stripped.

WHAT IS DELIBERATELY ABSENT: any `reward_config` / reward-term key. The value
function is given; only PPO knobs move. A single stray reward key would make
two cells incomparable, which is why `check_lines()` below rejects them rather
than trusting the author.

The four fixed anchors are also absent, i.e. inherited from
manipulation_params.py unchanged: `discounting` 0.99 (0.995 blew v_loss to
1e10), `max_grad_norm` 1.0, `num_minibatches` 32, and
num_envs/batch_size/unroll_length/num_updates_per_batch -- those four set
`env_step_per_training_step` and changing them would break budget
comparability across cells.
"""
import argparse
import itertools
import json
import math
import os

# Resolved relative to this file -- never an absolute path. The repo lives at a
# different path on each of the three machines (Mac dev, Linux robotics PC,
# ZHAW HPC) and those trees have drifted before. Stdlib only, no repo imports,
# so this generator runs under any interpreter on any of the three.
_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_CONFIG = os.path.join(_HERE, "target_config_satsweep.json")

# ---------------------------------------------------------------------------
# The swept factors. Level 0 == the manipulation_params.py UR3Pick baseline.
# ---------------------------------------------------------------------------
FACTORS = {
    "a": ("policy_hidden_layer_sizes", [(32, 32, 32, 32), (256, 256, 256)]),
    "b": ("entropy_cost", [2e-2, 5e-3]),
    "c": ("learning_rate", [6e-4, 3e-4]),
    "d": ("reward_scaling", [0.05, 0.03]),
}

# Not swept, but written on EVERY line. run_experiment.py merges network_factory
# as a nested dict, so a line that sets only policy_hidden_layer_sizes risks
# leaving the value net to whatever the merge yields. Both keys, always,
# including on the a0 baseline.
VALUE_HIDDEN_LAYER_SIZES = (256, 256, 256, 256, 256)

# Budget, from batch_runs/sweeps/jax_ppo_paramcalculation.md. These four are the
# fixed anchors that make env_step_per_training_step constant across cells.
BATCH_SIZE = 512
UNROLL_LENGTH = 10
NUM_MINIBATCHES = 32
NUM_RESETS_PER_EVAL = 1

TIERS = {
    # tier: (num_timesteps target, num_evals, video_every_evals)
    "A": (24_000_000, 20, 5),
    # Tier B's budget is NOT a constant: it is sized from measured Tier A
    # throughput in WP5 and must be passed with --num-timesteps. Refusing a
    # default here is the point -- see the plan's "Tier B budget is NOT yet
    # sized" warning.
    "B": (None, 40, 5),
}

# Reward-term keys. None of these may appear on a generated line; see the module
# docstring. Kept next to the generator rather than only in the plan so the
# guard travels with the code.
REWARD_KEYS = {
    "reward_config", "gripper_box", "approach_open", "grasp", "lift",
    "box_target", "gripper_align", "no_floor_collision", "robot_target_qpos",
    "action_rate", "hold_target", "success_bonus",
}


def env_step_per_training_step():
    return BATCH_SIZE * UNROLL_LENGTH * NUM_MINIBATCHES * NUM_RESETS_PER_EVAL


def solve_budget(num_timesteps_target, num_evals):
    """nts and the ACTUAL total steps, per jax_ppo_paramcalculation.md.

    brax does not train for exactly `num_timesteps`: it trains
    num_evals_after_init * num_resets_per_eval * nts * env_step_per_training_step,
    where nts is a ceiling. Returning the real number is what lets the plan
    quote a step budget W&B can actually be checked against.
    """
    e = env_step_per_training_step()
    num_evals_after_init = num_evals - 1
    nts = math.ceil(num_timesteps_target / (num_evals_after_init * e
                                            * NUM_RESETS_PER_EVAL))
    total = num_evals_after_init * NUM_RESETS_PER_EVAL * nts * e
    return nts, total


def load_target_config():
    with open(TARGET_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # `_`-prefixed keys are provenance (where each value came from, and which
    # env defaults the reference run inherited). They document the block; they
    # are not config and must never reach a sweep line.
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def cell_id(levels):
    return "".join(f"{k}{levels[k]}" for k in "abcd")


def build_entry(levels, seed, tier, num_timesteps, num_evals,
                video_every_evals, shared):
    policy = FACTORS["a"][1][levels["a"]]
    entry = dict(shared)
    entry.update({
        "seed": int(seed),
        "run_id": f"SatSweep_{cell_id(levels)}_s{seed}",
        "video_every_evals": video_every_evals,
        "num_timesteps": num_timesteps,
        "num_evals": num_evals,
        "num_resets_per_eval": NUM_RESETS_PER_EVAL,
        "entropy_cost": FACTORS["b"][1][levels["b"]],
        "learning_rate": FACTORS["c"][1][levels["c"]],
        "reward_scaling": FACTORS["d"][1][levels["d"]],
        "network_factory": {
            "policy_hidden_layer_sizes": list(policy),
            "value_hidden_layer_sizes": list(VALUE_HIDDEN_LAYER_SIZES),
        },
        "wandb_tags": [
            "satsweep", f"tier{tier}", cell_id(levels),
            f"pol{'x'.join(str(n) for n in policy)}",
            f"ent{FACTORS['b'][1][levels['b']]}",
            f"lr{FACTORS['c'][1][levels['c']]}",
            f"rs{FACTORS['d'][1][levels['d']]}",
            f"s{seed}",
        ],
    })
    return entry


def build_lines(tier, seeds, cells=None, num_timesteps=None):
    target_default, num_evals, video_every_evals = TIERS[tier]
    if num_timesteps is None:
        num_timesteps = target_default
    if num_timesteps is None:
        raise SystemExit(
            f"tier {tier} has no default num_timesteps: it must be sized from "
            f"MEASURED throughput (plan WP5) and passed via --num-timesteps"
        )

    nts, total = solve_budget(num_timesteps, num_evals)
    shared = load_target_config()

    all_cells = ["".join(f"{k}{v}" for k, v in zip("abcd", combo))
                 for combo in itertools.product([0, 1], repeat=4)]
    if cells:
        unknown = [c for c in cells if c not in all_cells]
        if unknown:
            raise SystemExit(f"unknown cell id(s): {unknown}")
        wanted = list(cells)
    else:
        wanted = all_cells

    header = (
        f"# Saturating-reward PPO sweep, tier {tier}. Generated by\n"
        f"# batch_runs/sweeps/gen_satsweep.py -- DO NOT hand-edit; regenerate.\n"
        f"# Plan: '20260814_Saturating reward - sweep exploration'.\n"
        f"#\n"
        f"# Full 2^4 factorial over the four PPO factors below; level 0 is the\n"
        f"# manipulation_params.py UR3Pick baseline, so cell a0b0c0d0 IS the\n"
        f"# reference run. Shared env block comes from\n"
        f"# target_config_satsweep.json (extracted from W&B run\n"
        f"# RealDR_vel_as04_gas02_s0), NOT hand-written.\n"
        f"#   a policy_hidden_layer_sizes (32,32,32,32) -> (256,256,256)\n"
        f"#   b entropy_cost              2e-2          -> 5e-3\n"
        f"#   c learning_rate             6e-4          -> 3e-4\n"
        f"#   d reward_scaling            0.05          -> 0.03\n"
        f"#\n"
        f"# Budget: num_timesteps={num_timesteps}, num_evals={num_evals},\n"
        f"# env_step_per_training_step={env_step_per_training_step()},\n"
        f"# nts={nts} -> ACTUAL TOTAL_STEPS={total}\n"
        f"# ({100.0 * total / num_timesteps:.1f}% of the requested budget --\n"
        f"# brax ceilings nts, so this number, not num_timesteps, is what W&B\n"
        f"# will show as training/num_steps).\n"
        f"#\n"
        f"# {len(wanted)} cells x {len(seeds)} seed(s) = "
        f"{len(wanted) * len(seeds)} runs. Comment/blank lines are skipped by\n"
        f"# run_one_ur3.py::load_config and do NOT consume a SLURM array index:\n"
        f"# set --array=1-{len(wanted) * len(seeds)}%4."
    )

    lines = [header]
    for c in wanted:
        levels = {k: int(v) for k, v in zip("abcd", c[1::2])}
        for seed in seeds:
            lines.append(json.dumps(build_entry(
                levels, seed, tier, num_timesteps, num_evals,
                video_every_evals, shared)))
    return lines, total


def check_lines(path, expected_n):
    """Re-read the written file and assert the invariants the plan names.

    Deliberately reads the FILE, not the in-memory list: what SLURM consumes is
    the file, and a serialisation bug (a tuple that json-encodes oddly, a
    duplicated run_id) is invisible to a check on the objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f
                if l.strip() and not l.startswith("#")]
    assert len(rows) == expected_n, f"{len(rows)} data lines, want {expected_n}"

    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate run_id"

    for r in rows:
        rid = r["run_id"]
        assert r["box_z_rot_range"] == 6.283185307179586, rid
        assert r["episode_length"] == 400, rid
        assert r["obs_include_velocity"] is True, rid
        assert r["num_resets_per_eval"] == NUM_RESETS_PER_EVAL, rid
        stray = REWARD_KEYS & set(r)
        assert not stray, f"{rid}: reward key(s) leaked: {sorted(stray)}"
        nf = r["network_factory"]
        assert "policy_hidden_layer_sizes" in nf, rid
        assert "value_hidden_layer_sizes" in nf, rid
        assert tuple(nf["value_hidden_layer_sizes"]) == \
            VALUE_HIDDEN_LAYER_SIZES, rid

    # Reward weights identical across cells is not provable by absence alone if
    # the shared block ever grew one, so compare the whole shared env block
    # pairwise: every key that is not swept must be byte-identical everywhere.
    swept = {"seed", "run_id", "wandb_tags", "entropy_cost", "learning_rate",
             "reward_scaling", "network_factory"}
    base_shared = {k: v for k, v in rows[0].items() if k not in swept}
    for r in rows[1:]:
        other = {k: v for k, v in r.items() if k not in swept}
        assert other == base_shared, (
            f"{r['run_id']} differs from {rows[0]['run_id']} outside the swept "
            f"factors: "
            f"{ {k: (base_shared.get(k), other.get(k)) for k in set(base_shared) | set(other) if base_shared.get(k) != other.get(k)} }"
        )
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--cells", nargs="+", default=None,
        help="cell ids to emit, e.g. a1b0c1d0 (default: all 16). Tier B uses "
             "the top 4 named by the WP5 analysis.")
    ap.add_argument(
        "--num-timesteps", type=int, default=None,
        help="override the tier's target budget. REQUIRED for tier B, which "
             "has no default on purpose -- its budget is sized from measured "
             "Tier A throughput.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite (this file "
            f"is generated, not hand-edited)")

    lines, total = build_lines(args.tier, args.seeds, cells=args.cells,
                               num_timesteps=args.num_timesteps)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n = len(lines) - 1  # minus the header block
    check_lines(args.out, n)
    print(f"Wrote {n} runs to {args.out}")
    print(f"  ACTUAL TOTAL_STEPS per run: {total}")
    print(f"  set --array=1-{n}%4")


if __name__ == "__main__":
    main()
