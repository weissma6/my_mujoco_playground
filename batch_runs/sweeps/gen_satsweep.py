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
import hashlib
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
    # Tier A-ext: the entropy_cost extension (plan WP5b). Same budget and
    # num_evals as Tier A ON PURPOSE -- it is only comparable to Tier A if the
    # step budget and eval cadence are identical. It exists as its own tier
    # solely so its runs carry `tierAext` rather than `tierA`: the WP5 analyzer
    # selects on that tag, and reusing `tierA` would silently grow the 2^4
    # factorial from 16 cells to 20 and corrupt every main effect it computes.
    "Aext": (24_000_000, 20, 5),
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


def init_pose_library_fingerprint():
    """(level, n_poses, sha256[:12]) of the init-pose library the runs will use.

    `init_start_random` is NOT a difficulty scalar: it is a LEVEL NAME selecting
    a set of starting positions. ur3_pick.py does load_init_poses(level, "train")
    -> init_poses/train/<level>.json, and reset() draws one pose per episode out
    of that file. The poses are hand-collected real UR3e poses (RTDE getActualQ).

    The library is therefore a DATA dependency of this sweep, and it has already
    moved once under a running experiment: train/mid.json went 30 -> 60 poses on
    2026-07-29 (bb35de8), after the reference run trained on 2026-07-27. The two
    versions carry byte-identical metadata (same `created`, same seed), so
    nothing inside the file distinguishes them -- only the pose count and the
    content hash do.

    Tier A and Tier B are submitted weeks apart. Stamping this into each
    generated JSONL header is what makes "the winner reproduced by 3 seeds"
    checkable: if the fingerprints differ between tiers, the seeds did not train
    on the same starting positions and the reproduction claim is void.

    The level is read from the recorded provenance rather than assumed, because
    the reference run inherits it from the env default instead of pinning it.
    """
    with open(TARGET_CONFIG, "r", encoding="utf-8") as f:
        prov = json.load(f).get("_provenance", {})
    level = prov.get("inherited_env_defaults", {}).get("init_start_random")
    if not level or level == "none":
        return level, None, None
    path = os.path.join(
        _HERE, os.pardir, os.pardir, "mujoco_playground", "_src",
        "manipulation", "my_ur3", "init_poses", "train", f"{level}.json")
    if not os.path.exists(path):
        return level, None, None
    with open(path, "rb") as f:
        raw = f.read()
    return (level, len(json.loads(raw)["poses"]),
            hashlib.sha256(raw).hexdigest()[:12])


def cell_id(levels):
    return "".join(f"{k}{levels[k]}" for k in "abcd")


def build_entry(levels, seed, tier, num_timesteps, num_evals,
                video_every_evals, shared, entropy=None):
    """One sweep line.

    `entropy` overrides the b-factor's value (plan WP5b). Tier A measured
    entropy_cost as by far the largest main effect (-0.297 on
    eval/episode_success for 2e-2 -> 5e-3) -- i.e. the optimum lies ABOVE the
    swept range, which the 2^4 design never reached. The override keeps the
    cell id (so the other three factors stay readable) but makes the entropy
    value explicit in the run_id, because it is no longer recoverable from the
    `b` digit.
    """
    policy = FACTORS["a"][1][levels["a"]]
    ent = FACTORS["b"][1][levels["b"]] if entropy is None else float(entropy)
    suffix = "" if entropy is None else f"_ent{ent:g}"
    entry = dict(shared)
    entry.update({
        "seed": int(seed),
        "run_id": f"SatSweep_{cell_id(levels)}{suffix}_s{seed}",
        "video_every_evals": video_every_evals,
        "num_timesteps": num_timesteps,
        "num_evals": num_evals,
        "num_resets_per_eval": NUM_RESETS_PER_EVAL,
        "entropy_cost": ent,
        "learning_rate": FACTORS["c"][1][levels["c"]],
        "reward_scaling": FACTORS["d"][1][levels["d"]],
        "network_factory": {
            "policy_hidden_layer_sizes": list(policy),
            "value_hidden_layer_sizes": list(VALUE_HIDDEN_LAYER_SIZES),
        },
        "wandb_tags": [
            "satsweep", f"tier{tier}", cell_id(levels),
            f"pol{'x'.join(str(n) for n in policy)}",
            f"ent{ent:g}",
            f"lr{FACTORS['c'][1][levels['c']]}",
            f"rs{FACTORS['d'][1][levels['d']]}",
            f"s{seed}",
        ] + ([] if entropy is None else ["entscan"]),
    })
    return entry


def build_lines(tier, seeds, cells=None, num_timesteps=None, entropy=None):
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
    pose_level, pose_n, pose_sha = init_pose_library_fingerprint()

    all_cells = ["".join(f"{k}{v}" for k, v in zip("abcd", combo))
                 for combo in itertools.product([0, 1], repeat=4)]
    if cells:
        unknown = [c for c in cells if c not in all_cells]
        if unknown:
            raise SystemExit(f"unknown cell id(s): {unknown}")
        wanted = list(cells)
    else:
        wanted = all_cells

    if entropy is None:
        design = (
            f"# Full 2^4 factorial over the four PPO factors below; level 0 is the\n"
            f"# manipulation_params.py UR3Pick baseline, so cell a0b0c0d0 IS the\n"
            f"# reference run. Shared env block comes from\n"
            f"# target_config_satsweep.json (extracted from W&B run\n"
            f"# RealDR_vel_as04_gas02_s0), NOT hand-written.\n"
            f"#   a policy_hidden_layer_sizes (32,32,32,32) -> (256,256,256)\n"
            f"#   b entropy_cost              2e-2          -> 5e-3\n"
            f"#   c learning_rate             6e-4          -> 3e-4\n"
            f"#   d reward_scaling            0.05          -> 0.03\n"
        )
    else:
        design = (
            f"# ENTROPY EXTENSION (plan WP5b) -- NOT a factorial. The b digit in\n"
            f"# each cell id is OVERRIDDEN; read entropy_cost off the run_id.\n"
            f"# Scanned entropy_cost: {', '.join(f'{e:g}' for e in entropy)}\n"
            f"#\n"
            f"# Why: Tier A measured entropy_cost as by far the largest main\n"
            f"# effect, -0.297 on eval/episode_success for 2e-2 -> 5e-3. That is\n"
            f"# the OPPOSITE of the plan's hypothesis B, and it means the optimum\n"
            f"# lies ABOVE 2e-2 -- outside the range the 2^4 design ever tested,\n"
            f"# since every one of its 16 cells sat at 2e-2 or below. This scan\n"
            f"# bounds the other side. Budget and num_evals are identical to Tier\n"
            f"# A so the results are directly comparable.\n"
            f"#\n"
            f"# The other three factors are held at the Tier A winner unless\n"
            f"# --cells says otherwise. Shared env block comes from\n"
            f"# target_config_satsweep.json, NOT hand-written.\n"
        )

    header = (
        f"# Saturating-reward PPO sweep, tier {tier}. Generated by\n"
        f"# batch_runs/sweeps/gen_satsweep.py -- DO NOT hand-edit; regenerate.\n"
        f"# Plan: '20260814_Saturating reward - sweep exploration'.\n"
        f"#\n"
        + design +
        f"#\n"
        f"# Budget: num_timesteps={num_timesteps}, num_evals={num_evals},\n"
        f"# env_step_per_training_step={env_step_per_training_step()},\n"
        f"# nts={nts} -> ACTUAL TOTAL_STEPS={total}\n"
        f"# ({100.0 * total / num_timesteps:.1f}% of the requested budget --\n"
        f"# brax ceilings nts, so this number, not num_timesteps, is what W&B\n"
        f"# will show as training/num_steps).\n"
        f"#\n"
        # Phrased WITHOUT the entropy term when there is no entropy scan, so a
        # plain `--tier A` regeneration still reproduces
        # UR3Pick_satsweep_tierA.jsonl byte-for-byte -- that file is the record
        # of what the 16 landed Tier A runs actually trained under, and a
        # regenerate-diff is how a future session checks it was not tampered
        # with. Do not "simplify" these two branches into one.
        + (f"# {len(wanted)} cells x {len(seeds)} seed(s) = "
           f"{len(wanted) * len(seeds)} runs. "
           if entropy is None else
           f"# {len(wanted)} cells x {len(entropy)} entropy value(s) x "
           f"{len(seeds)} seed(s) = "
           f"{len(wanted) * len(entropy) * len(seeds)} runs. ") +
        f"Comment/blank lines are skipped by\n"
        f"# run_one_ur3.py::load_config and do NOT consume a SLURM array index:\n"
        f"# set --array=1-"
        f"{len(wanted) * (len(entropy) if entropy else 1) * len(seeds)}%4.\n"
        f"#\n"
        f"# INIT-POSE LIBRARY: level={pose_level!r} n_poses={pose_n} "
        f"sha256={pose_sha}\n"
        f"# 'mid' is a level name selecting a set of hand-collected real UR3e\n"
        f"# starting positions (init_poses/train/<level>.json), NOT a difficulty\n"
        f"# scalar -- reset() draws one pose per episode from that file. It is a\n"
        f"# DATA dependency and it has moved under a running experiment before\n"
        f"# (train/mid.json: 30 -> 60 poses on 2026-07-29, bb35de8, with\n"
        f"# byte-identical metadata). Tier A and Tier B must show the SAME\n"
        f"# fingerprint here, or the 3-seed reproduction claim is void."
    )

    lines = [header]
    for c in wanted:
        levels = {k: int(v) for k, v in zip("abcd", c[1::2])}
        for ent in (entropy if entropy else [None]):
            for seed in seeds:
                lines.append(json.dumps(build_entry(
                    levels, seed, tier, num_timesteps, num_evals,
                    video_every_evals, shared, entropy=ent)))
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
    ap.add_argument(
        "--entropy", type=float, nargs="+", default=None,
        help="override entropy_cost with these values, one run per value "
             "(plan WP5b). Requires --tier Aext so the runs carry `tierAext` "
             "and cannot contaminate the 16-cell Tier A factorial.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file")
    args = ap.parse_args()

    # Keep the tier tag and the design honest about each other. Tier A's
    # analysis selects runs by the `tierA` tag and assumes exactly 16 cells; an
    # entropy scan emitted under that tag would silently enlarge the factorial
    # and corrupt every main effect computed from it.
    if args.entropy and args.tier != "Aext":
        raise SystemExit(
            f"--entropy requires --tier Aext (got {args.tier!r}): an entropy "
            f"scan is not part of the 2^4 factorial and must not be tagged as "
            f"tier{args.tier}")
    if args.tier == "Aext" and not args.entropy:
        raise SystemExit(
            "--tier Aext is the entropy extension and needs --entropy VALUES")

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite (this file "
            f"is generated, not hand-edited)")

    lines, total = build_lines(args.tier, args.seeds, cells=args.cells,
                               num_timesteps=args.num_timesteps,
                               entropy=args.entropy)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n = len(lines) - 1  # minus the header block
    check_lines(args.out, n)
    print(f"Wrote {n} runs to {args.out}")
    print(f"  ACTUAL TOTAL_STEPS per run: {total}")
    print(f"  set --array=1-{n}%4")


if __name__ == "__main__":
    main()
