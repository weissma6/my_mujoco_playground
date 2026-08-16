"""Generate the post-lift annuity 2x2 gating JSONL (8 runs).

Plan: "20260816_Post-lift annuity - reward gating" (VT2-SimToReal-Robotics).
Direct successor to "20260814_Saturating reward - sweep exploration".

WHAT THIS SWEEP VARIES: exactly two TOP-LEVEL env booleans, nothing else.

    gate_gripper_box_on_lift     False -> True
    gate_gripper_align_on_lift   False -> True

Both multiply their reward term by (1 - lifted) in ur3_pick._get_reward, the
same treatment robot_target_qpos already gets. WHY: the predecessor plan's WP5
decomposition measured ~41% of post-lift return as an annuity carrying no
gradient toward the target -- gripper_box at 4.00/step and gripper_align at
3.75/step, the two largest terms with no (1 - lifted) gate, of which 88% of
gripper_box's raw episode sum is banked AFTER the lift. The pick itself is
already solved (reached/grasped/lifted all complete ~100%); only placement
fails, at ~46-55%.

THE 2x2, and why it is a factorial rather than one combined on/off switch:

                     | box ungated | box gated
    align ungated    |     g00     |    g10
    align gated      |     g01     |    g11

g00 is the MATCHED BASELINE and needs no separate control run: with both flags
False the code is a proven bit-identical no-op (plan WP1's guard, verified by
rollout diff), so g00 re-runs the Tier A winner a1b0c1d1 under the new code
path -- which doubles as a determinism check. The two terms are 13.5% and 16.3%
of return and there is no reason to assume they act alike: gripper_box is a
pure distance term that is trivially maximal while carrying, whereas
gripper_align still varies with wrist orientation during transport and may be
doing real work. One combined switch could not tell those apart.

WHAT IS DELIBERATELY ABSENT: any reward_config / reward-term SCALE key. This
plan changes GATING, not magnitudes -- check_lines() rejects them rather than
trusting the author, exactly as gen_satsweep.py does.

THE PPO BLOCK IS FROZEN at the Tier A winner a1b0c1d1 -- policy (256,256,256),
entropy_cost 2e-2, learning_rate 3e-4, reward_scaling 0.03 -- so the reward
change is not confounded with a PPO change. Budget and num_evals are identical
to Tier A so these 8 runs are directly comparable to its 16 cells.

The shared env block and the init-pose fingerprint are IMPORTED from
gen_satsweep.py rather than re-implemented: the whole comparison to Tier A is
void if either differs, and a copy could drift silently.
"""
import argparse
import itertools
import json
import os
import sys

# Same directory, so a plain script invocation resolves this. Imported (not
# copied) on purpose -- see the module docstring: an independent re-implement-
# ation of the budget math or the pose fingerprint could drift away from the
# Tier A runs these 8 are compared against, and that drift would be invisible.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_satsweep as gs  # noqa: E402

# The Tier A winner, a1b0c1d1 (predecessor WP5: 0.555 success / 8727 reward --
# the best of all 16 cells). Frozen, not swept. Written out literally rather
# than indexed out of gs.FACTORS so a future edit to that dict cannot silently
# move this plan's anchor.
ANCHOR_CELL = "a1b0c1d1"
ANCHOR = {
    "policy_hidden_layer_sizes": (256, 256, 256),
    "entropy_cost": 2e-2,
    "learning_rate": 3e-4,
    "reward_scaling": 0.03,
}

# Tier A's budget, reused verbatim so the 8 runs sit on the same axis as the 16.
NUM_TIMESTEPS = 24_000_000
NUM_EVALS = 20
VIDEO_EVERY_EVALS = 5

# The two knobs under test. Order is (gate_box, gate_align) -> cell id gNM.
GATE_BOX_KEY = "gate_gripper_box_on_lift"
GATE_ALIGN_KEY = "gate_gripper_align_on_lift"


def cell_id(gate_box, gate_align):
    return f"g{int(gate_box)}{int(gate_align)}"


def build_entry(gate_box, gate_align, seed, shared, nts_budget, num_evals):
    cid = cell_id(gate_box, gate_align)
    entry = dict(shared)
    entry.update({
        "seed": int(seed),
        "run_id": f"RewardGate_{cid}_s{seed}",
        "video_every_evals": VIDEO_EVERY_EVALS,
        "num_timesteps": nts_budget,
        "num_evals": num_evals,
        "num_resets_per_eval": gs.NUM_RESETS_PER_EVAL,
        # --- frozen PPO anchor (Tier A winner) ---
        "entropy_cost": ANCHOR["entropy_cost"],
        "learning_rate": ANCHOR["learning_rate"],
        "reward_scaling": ANCHOR["reward_scaling"],
        "network_factory": {
            "policy_hidden_layer_sizes": list(ANCHOR["policy_hidden_layer_sizes"]),
            "value_hidden_layer_sizes": list(gs.VALUE_HIDDEN_LAYER_SIZES),
        },
        # --- the only thing this sweep varies ---
        GATE_BOX_KEY: bool(gate_box),
        GATE_ALIGN_KEY: bool(gate_align),
        "wandb_tags": [
            "rewardgate", cid,
            f"gatebox{int(gate_box)}", f"gatealign{int(gate_align)}",
            f"anchor{ANCHOR_CELL}", f"s{seed}",
        ],
    })
    return entry


def build_lines(seeds, num_timesteps=None, num_evals=None):
    num_timesteps = NUM_TIMESTEPS if num_timesteps is None else num_timesteps
    num_evals = NUM_EVALS if num_evals is None else num_evals

    nts, total = gs.solve_budget(num_timesteps, num_evals)
    shared = gs.load_target_config()
    pose_level, pose_n, pose_sha = gs.init_pose_library_fingerprint()

    combos = list(itertools.product([0, 1], repeat=2))  # (box, align)
    n_runs = len(combos) * len(seeds)

    header = (
        f"# Post-lift annuity gating, 2x2 factorial. Generated by\n"
        f"# batch_runs/sweeps/gen_rewardgate.py -- DO NOT hand-edit; regenerate.\n"
        f"# Plan: '20260816_Post-lift annuity - reward gating'.\n"
        f"#\n"
        f"# Two TOP-LEVEL env booleans, nothing else:\n"
        f"#   {GATE_BOX_KEY}    False -> True\n"
        f"#   {GATE_ALIGN_KEY}  False -> True\n"
        f"# Each multiplies its reward term by (1 - lifted) in _get_reward, the\n"
        f"# same treatment robot_target_qpos already gets. Cell id is gNM where\n"
        f"# N = gate_box, M = gate_align:\n"
        f"#\n"
        f"#                  | box ungated | box gated\n"
        f"#   align ungated  |     g00     |    g10\n"
        f"#   align gated    |     g01     |    g11\n"
        f"#\n"
        f"# g00 is the MATCHED BASELINE, not a separate control: with both flags\n"
        f"# False the code is a proven BIT-IDENTICAL no-op (plan WP1 guard), so\n"
        f"# g00 re-runs Tier A winner {ANCHOR_CELL} under the new code path and\n"
        f"# doubles as a determinism check against its 0.555.\n"
        f"#\n"
        f"# NO reward_config / reward-term SCALE key appears on any line: this\n"
        f"# plan changes GATING, not magnitudes.\n"
        f"#\n"
        f"# PPO BLOCK FROZEN at Tier A winner {ANCHOR_CELL}:\n"
        f"#   policy_hidden_layer_sizes="
        f"{list(ANCHOR['policy_hidden_layer_sizes'])}\n"
        f"#   entropy_cost={ANCHOR['entropy_cost']:g} "
        f"learning_rate={ANCHOR['learning_rate']:g} "
        f"reward_scaling={ANCHOR['reward_scaling']:g}\n"
        f"# so the reward change is not confounded with a PPO change.\n"
        f"#\n"
        f"# Budget: num_timesteps={num_timesteps}, num_evals={num_evals},\n"
        f"# env_step_per_training_step={gs.env_step_per_training_step()},\n"
        f"# nts={nts} -> ACTUAL TOTAL_STEPS={total}\n"
        f"# ({100.0 * total / num_timesteps:.1f}% of the requested budget --\n"
        f"# brax ceilings nts, so this number, not num_timesteps, is what W&B\n"
        f"# will show as training/num_steps). IDENTICAL to Tier A on purpose.\n"
        f"#\n"
        f"# {len(combos)} cells x {len(seeds)} seed(s) = {n_runs} runs.\n"
        f"# Comment/blank lines are skipped by run_one_ur3.py::load_config and\n"
        f"# do NOT consume a SLURM array index: set --array=1-{n_runs}%4.\n"
        f"#\n"
        f"# INIT-POSE LIBRARY: level={pose_level!r} n_poses={pose_n} "
        f"sha256={pose_sha}\n"
        f"# MUST match the satsweep Tier A runs these are compared against, or\n"
        f"# the comparison is void -- 'mid' is a level name selecting a set of\n"
        f"# hand-collected real UR3e starting positions, a DATA dependency that\n"
        f"# has moved under a running experiment before (train/mid.json: 30 ->\n"
        f"# 60 poses on 2026-07-29, bb35de8, with byte-identical metadata)."
    )

    lines = [header]
    for gate_box, gate_align in combos:
        for seed in seeds:
            lines.append(json.dumps(
                build_entry(gate_box, gate_align, seed, shared, num_timesteps,
                            num_evals)))
    return lines, total


def check_lines(path, expected_n, seeds):
    """Re-read the WRITTEN FILE and assert every invariant the plan names.

    Reads the file rather than the in-memory list on purpose (same reason as
    gen_satsweep.check_lines): what SLURM consumes is the file, and a
    serialisation bug is invisible to a check on the objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f
                if l.strip() and not l.startswith("#")]
    assert len(rows) == expected_n, f"{len(rows)} data lines, want {expected_n}"

    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate run_id: {ids}"

    # Each of the 4 gating combinations appears exactly len(seeds) times.
    seen = {}
    for r in rows:
        seen.setdefault((r[GATE_BOX_KEY], r[GATE_ALIGN_KEY]), []).append(
            r["run_id"])
    assert len(seen) == 4, f"want 4 gating combos, got {sorted(seen)}"
    for combo, members in seen.items():
        assert len(members) == len(seeds), \
            f"combo {combo} appears {len(members)}x, want {len(seeds)}: {members}"

    for r in rows:
        rid = r["run_id"]
        # Inherited anchors that must never move (plan "Fixed anchors").
        assert r["box_z_rot_range"] == 6.283185307179586, rid
        assert r["episode_length"] == 400, rid
        assert r["obs_include_velocity"] is True, rid
        assert r["num_resets_per_eval"] == gs.NUM_RESETS_PER_EVAL, rid
        # This plan changes gating, NOT magnitudes.
        stray = gs.REWARD_KEYS & set(r)
        assert not stray, f"{rid}: reward key(s) leaked: {sorted(stray)}"
        # Both flags must be present and boolean on EVERY line, including the
        # g00 baseline -- an absent key would silently inherit the env default
        # and make the cell id a lie.
        for k in (GATE_BOX_KEY, GATE_ALIGN_KEY):
            assert k in r, f"{rid}: missing {k}"
            assert isinstance(r[k], bool), f"{rid}: {k} is {type(r[k])}, want bool"
        # The cell id must agree with the flags it claims to encode.
        assert rid.startswith(
            f"RewardGate_{cell_id(r[GATE_BOX_KEY], r[GATE_ALIGN_KEY])}_"), \
            f"{rid}: run_id disagrees with its own gating flags"
        # PPO block frozen at the Tier A winner on every line.
        assert r["entropy_cost"] == ANCHOR["entropy_cost"], rid
        assert r["learning_rate"] == ANCHOR["learning_rate"], rid
        assert r["reward_scaling"] == ANCHOR["reward_scaling"], rid
        nf = r["network_factory"]
        assert tuple(nf["policy_hidden_layer_sizes"]) == \
            ANCHOR["policy_hidden_layer_sizes"], rid
        assert tuple(nf["value_hidden_layer_sizes"]) == \
            gs.VALUE_HIDDEN_LAYER_SIZES, rid

    # Everything outside the two gating flags and the per-run identity must be
    # byte-identical across all 8 lines. This is the guard that actually makes
    # the 2x2 a 2x2 -- absence of a stray key is not provable term by term.
    varying = {"seed", "run_id", "wandb_tags", GATE_BOX_KEY, GATE_ALIGN_KEY}
    base = {k: v for k, v in rows[0].items() if k not in varying}
    for r in rows[1:]:
        other = {k: v for k, v in r.items() if k not in varying}
        assert other == base, (
            f"{r['run_id']} differs from {rows[0]['run_id']} outside the two "
            f"gating flags: "
            f"{ {k: (base.get(k), other.get(k)) for k in set(base) | set(other) if base.get(k) != other.get(k)} }"
        )
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--num-timesteps", type=int, default=None,
                    help="override the Tier A budget (default %d). Changing it "
                         "breaks comparability with Tier A." % NUM_TIMESTEPS)
    ap.add_argument("--num-evals", type=int, default=None,
                    help="override num_evals (default %d)." % NUM_EVALS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite (this file "
            f"is generated, not hand-edited)")

    lines, total = build_lines(args.seeds, num_timesteps=args.num_timesteps,
                               num_evals=args.num_evals)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n = len(lines) - 1  # minus the header block
    check_lines(args.out, n, args.seeds)
    print(f"Wrote {n} runs to {args.out}")
    print(f"  ACTUAL TOTAL_STEPS per run: {total}")
    print(f"  set --array=1-{n}%4")


if __name__ == "__main__":
    main()
