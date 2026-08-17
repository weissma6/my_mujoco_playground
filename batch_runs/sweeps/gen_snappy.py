#!/usr/bin/env python3
"""One-knob action_scale arm, bolted onto the reward-gating winner.

ONE knob: `action_scale`. Everything else -- PPO block, gating flags, gripper
scale, budget, env -- is frozen at RewardGate cell g01, so any difference is
attributable to the arm's per-step control delta and nothing else.

WHY action_scale. The runs so far use 0.04, inherited unexamined from the
RealDR reference run via target_config_satsweep.json (satsweep WP1). The env
DEFAULT is 0.015, and that is not an arbitrary default -- it is the winner of a
dedicated slowdown sweep (eaabb6b, `Smooth_slow_mid_as015_ar10`). So every run
in this lineage has been training at 2.7x the value a previous sweep chose for
smoothness. Measured across all 8 rewardgate runs, rms|delta a| ~= 0.70 per
dim, so the commanded joint target moves 0.70 * 0.04 = 0.028 rad ~= 1.6 deg of
step-to-step jitter; at 0.015 the same policy noise would produce ~0.6 deg.
That is the "snappy" look in the eval videos, quantified.

NOT swept here, deliberately:
  * `action_rate` (-0.10) is ALREADY at the value that same slowdown sweep
    settled on, so the smoothness PENALTY is not the thing that drifted.
  * `gripper_action_scale` stays 0.02. It has its own documented problem -- the
    comment above it in ur3_pick.py justifies 0.01 ("full open->close travel is
    0.025, which needs >=2.5 steps at 0.01") while the value is 0.02, which
    takes 1.25 steps, i.e. the hand CAN effectively snap shut in one step. Real,
    but it is a SECOND knob and this arm is deliberately one.

NO BASELINE RUN IS GENERATED. `RewardGate_g01_s{0,1}` (0.602 / 0.664) already
IS the action_scale=0.04 arm at this exact configuration, so re-running it
would burn GPU time to reproduce a number already in W&B. The comparison is
against those two runs by name.

THE TRADEOFF IS THE POINT, not smoothness alone. Lowering action_scale lowers
max joint speed, and t_lift is already 121-140 of 400 steps -- the predecessor
ceiling analysis says a later lift LOWERS the achievable return. So this arm is
judged on eval/episode_success and t_lift together with the smoothness
diagnostics, never on the videos alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_satsweep as gs        # noqa: E402
import gen_rewardgate as rg      # noqa: E402

# The reward-gating winner: gripper_align gated, gripper_box NOT gated.
# From the 2x2 (2 seeds/cell): g01 = 0.633 mean, the best of the four, and the
# negative interaction means gating BOTH (g11 = 0.602) is worse than align
# alone. Frozen here, not swept.
GATE_BOX = False
GATE_ALIGN = True
GATE_CELL = "g01"

# The one knob. 0.04 is omitted ON PURPOSE -- RewardGate_g01_s{0,1} already is
# that arm; see the module docstring.
ACTION_SCALES = [0.025, 0.015]
BASELINE_ACTION_SCALE = 0.04
BASELINE_RUNS = "RewardGate_g01_s0 (0.602), RewardGate_g01_s1 (0.664)"

# Held fixed -- the second smoothness knob, deliberately not touched.
GRIPPER_ACTION_SCALE = 0.02


def scale_tag(v):
    """0.04 -> 'as04', 0.025 -> 'as025', 0.015 -> 'as015'.

    Matches the existing repo convention (RealDR_vel_as04_gas02_s0).
    """
    return "as" + f"{v:g}".replace("0.", "", 1)


def build_entry(action_scale, seed, shared, num_timesteps, num_evals):
    tag = scale_tag(action_scale)
    entry = dict(shared)
    entry.update({
        "seed": int(seed),
        "run_id": f"Snappy_{tag}_{GATE_CELL}_s{seed}",
        "video_every_evals": rg.VIDEO_EVERY_EVALS,
        "num_timesteps": num_timesteps,
        "num_evals": num_evals,
        "num_resets_per_eval": gs.NUM_RESETS_PER_EVAL,
        # --- frozen PPO anchor (Tier A winner), same as the gating sweep ---
        "entropy_cost": rg.ANCHOR["entropy_cost"],
        "learning_rate": rg.ANCHOR["learning_rate"],
        "reward_scaling": rg.ANCHOR["reward_scaling"],
        "network_factory": {
            "policy_hidden_layer_sizes":
                list(rg.ANCHOR["policy_hidden_layer_sizes"]),
            "value_hidden_layer_sizes": list(gs.VALUE_HIDDEN_LAYER_SIZES),
        },
        # --- frozen at the gating winner ---
        rg.GATE_BOX_KEY: GATE_BOX,
        rg.GATE_ALIGN_KEY: GATE_ALIGN,
        # --- the ONE knob under test; the shared block carries 0.04, override ---
        "action_scale": float(action_scale),
        # --- explicitly pinned so the second knob provably did NOT move ---
        "gripper_action_scale": GRIPPER_ACTION_SCALE,
        "wandb_tags": [
            "snappy", tag, GATE_CELL,
            f"gas{f'{GRIPPER_ACTION_SCALE:g}'.replace('0.', '', 1)}",
            f"anchor{rg.ANCHOR_CELL}", f"s{seed}",
        ],
    })
    return entry


def build_lines(seeds, num_timesteps=None, num_evals=None):
    num_timesteps = rg.NUM_TIMESTEPS if num_timesteps is None else num_timesteps
    num_evals = rg.NUM_EVALS if num_evals is None else num_evals
    nts, total = gs.solve_budget(num_timesteps, num_evals)
    shared = gs.load_target_config()
    pose_level, pose_n, pose_sha = gs.init_pose_library_fingerprint()

    # The shared block carries action_scale=0.04; assert that, so that if the
    # frozen target config is ever regenerated with a different value this
    # generator fails loudly instead of silently changing what "baseline" means.
    assert shared.get("action_scale") == BASELINE_ACTION_SCALE, (
        f"target_config_satsweep.json has action_scale="
        f"{shared.get('action_scale')}, expected {BASELINE_ACTION_SCALE}; the "
        f"baseline runs {BASELINE_RUNS} no longer describe the omitted arm")
    assert shared.get("gripper_action_scale") == GRIPPER_ACTION_SCALE, (
        f"gripper_action_scale drifted to {shared.get('gripper_action_scale')}")

    n_runs = len(ACTION_SCALES) * len(seeds)
    header = (
        f"# Snappiness: ONE-KNOB action_scale arm. Generated by\n"
        f"# batch_runs/sweeps/gen_snappy.py -- DO NOT hand-edit; regenerate.\n"
        f"# Plan: '20260816_Post-lift annuity - reward gating' (parallel arm).\n"
        f"#\n"
        f"# The ONLY key that varies across these lines is action_scale:\n"
        f"#   {' -> '.join(f'{v:g}' for v in ACTION_SCALES)}"
        f"   (baseline {BASELINE_ACTION_SCALE:g} is NOT regenerated)\n"
        f"#\n"
        f"# NO BASELINE RUN HERE. {BASELINE_RUNS}\n"
        f"# already IS the action_scale={BASELINE_ACTION_SCALE:g} arm at this exact\n"
        f"# configuration -- gating {GATE_CELL}, same PPO anchor, same budget, same\n"
        f"# env. Compare against those two by name; do not re-run them.\n"
        f"#\n"
        f"# WHY: 0.04 was inherited unexamined from the RealDR reference run via\n"
        f"# target_config_satsweep.json. The env DEFAULT is 0.015, which is the\n"
        f"# winner of a dedicated slowdown sweep (eaabb6b,\n"
        f"# Smooth_slow_mid_as015_ar10) -- so this lineage has trained at 2.7x a\n"
        f"# value previously chosen for smoothness. Measured over the 8 rewardgate\n"
        f"# runs: rms|delta a| ~= 0.70/dim, so the commanded joint target jitters\n"
        f"# 0.70*0.04 = 0.028 rad ~= 1.6 deg per step (~0.6 deg at 0.015).\n"
        f"#\n"
        f"# FROZEN, so the knob is the only explanation for any difference:\n"
        f"#   {rg.GATE_BOX_KEY}={GATE_BOX}\n"
        f"#   {rg.GATE_ALIGN_KEY}={GATE_ALIGN}   (gating winner {GATE_CELL})\n"
        f"#   gripper_action_scale={GRIPPER_ACTION_SCALE:g}  <- the SECOND\n"
        f"#     smoothness knob, deliberately NOT moved. Its own comment in\n"
        f"#     ur3_pick.py justifies 0.01 (\"needs >=2.5 steps\" for the 0.025\n"
        f"#     full travel) while the value is 0.02 = 1.25 steps. Real issue,\n"
        f"#     separate knob, separate arm.\n"
        f"#   action_rate=-0.10 (env default) <- ALREADY the slowdown sweep's\n"
        f"#     value, so the smoothness PENALTY is not what drifted.\n"
        f"#   PPO at Tier A winner {rg.ANCHOR_CELL}: "
        f"policy={list(rg.ANCHOR['policy_hidden_layer_sizes'])}, "
        f"ent={rg.ANCHOR['entropy_cost']:g}, "
        f"lr={rg.ANCHOR['learning_rate']:g}, "
        f"rs={rg.ANCHOR['reward_scaling']:g}\n"
        f"#\n"
        f"# JUDGE ON THE TRADEOFF, not on smoothness alone: lower action_scale\n"
        f"# means lower max joint speed, and t_lift is already 121-140 of 400\n"
        f"# steps. A smoother policy that lifts later has a LOWER reward ceiling\n"
        f"# (predecessor plan's ceiling table). Report eval/episode_success and\n"
        f"# t_lift alongside eval/episode_action_rate and tcp_speed.\n"
        f"#\n"
        f"# Budget: num_timesteps={num_timesteps}, num_evals={num_evals},\n"
        f"# env_step_per_training_step={gs.env_step_per_training_step()},\n"
        f"# nts={nts} -> ACTUAL TOTAL_STEPS={total}\n"
        f"# IDENTICAL to Tier A and to the rewardgate 2x2, on purpose.\n"
        f"#\n"
        f"# {len(ACTION_SCALES)} action_scale value(s) x {len(seeds)} seed(s) = "
        f"{n_runs} runs. Comment/blank lines are skipped by\n"
        f"# run_one_ur3.py::load_config and do NOT consume a SLURM array index:\n"
        f"# set --array=1-{n_runs}%4.\n"
        f"#\n"
        f"# INIT-POSE LIBRARY: level={pose_level!r} n_poses={pose_n} "
        f"sha256={pose_sha}\n"
        f"# MUST match the rewardgate runs these are compared against."
    )

    lines = [header]
    for a in ACTION_SCALES:
        for seed in seeds:
            lines.append(json.dumps(
                build_entry(a, seed, shared, num_timesteps, num_evals)))
    return lines, total


def check_lines(path, expected_n, seeds):
    """Re-read the WRITTEN file and assert every invariant this arm depends on."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip() and not l.lstrip().startswith("#")]
    assert len(rows) == expected_n, f"{len(rows)} data lines, expected {expected_n}"

    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate run_id: {ids}"

    seen = set()
    for r in rows:
        rid = r["run_id"]
        # no reward-magnitude key may appear -- this arm changes control, not reward
        assert not (gs.REWARD_KEYS & set(r)), (rid, gs.REWARD_KEYS & set(r))
        # the knob itself
        a = r["action_scale"]
        assert a in ACTION_SCALES, f"{rid}: unexpected action_scale {a}"
        assert a != BASELINE_ACTION_SCALE, (
            f"{rid}: regenerates the baseline arm, which already exists as "
            f"{BASELINE_RUNS}")
        assert scale_tag(a) in rid, f"{rid}: run_id disagrees with action_scale {a}"
        seen.add((a, r["seed"]))
        # the second knob provably did not move
        assert r["gripper_action_scale"] == GRIPPER_ACTION_SCALE, rid
        # gating frozen at the winner
        assert r[rg.GATE_BOX_KEY] is GATE_BOX, rid
        assert r[rg.GATE_ALIGN_KEY] is GATE_ALIGN, rid
        # PPO frozen at the Tier A winner
        assert r["entropy_cost"] == rg.ANCHOR["entropy_cost"], rid
        assert r["learning_rate"] == rg.ANCHOR["learning_rate"], rid
        assert r["reward_scaling"] == rg.ANCHOR["reward_scaling"], rid
        nf = r["network_factory"]
        assert tuple(nf["policy_hidden_layer_sizes"]) == \
            rg.ANCHOR["policy_hidden_layer_sizes"], rid
        assert tuple(nf["value_hidden_layer_sizes"]) == \
            gs.VALUE_HIDDEN_LAYER_SIZES, rid
        # env invariants inherited from the frozen target config
        assert r["box_z_rot_range"] == 6.283185307179586, rid
        assert r["episode_length"] == 400, rid
        assert r["obs_include_velocity"] is True, rid

    assert seen == {(a, s) for a in ACTION_SCALES for s in seeds}, \
        f"missing or extra (action_scale, seed) combinations: {sorted(seen)}"

    # Every key except the knob, the seed and the identity fields must be
    # byte-identical across all lines -- otherwise "one knob" is a lie.
    varying = {"seed", "run_id", "wandb_tags", "action_scale"}
    base = {k: v for k, v in rows[0].items() if k not in varying}
    for r in rows[1:]:
        other = {k: v for k, v in r.items() if k not in varying}
        assert other == base, (
            f"{r['run_id']} differs from {rows[0]['run_id']} outside the one "
            f"swept knob: "
            f"{ {k: (base.get(k), other.get(k)) for k in set(base) | set(other) if base.get(k) != other.get(k)} }")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--num-evals", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite (this file "
            f"is generated, not hand-edited)")

    lines, total = build_lines(args.seeds, args.num_timesteps, args.num_evals)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n = len(lines) - 1
    check_lines(args.out, n, args.seeds)
    print(f"Wrote {n} runs to {args.out}")
    print(f"  action_scale values: {[f'{a:g}' for a in ACTION_SCALES]} "
          f"(baseline {BASELINE_ACTION_SCALE:g} NOT regenerated)")
    print(f"  baseline for comparison: {BASELINE_RUNS}")
    print(f"  ACTUAL TOTAL_STEPS per run: {total}")
    print(f"  set --array=1-{n}%4")


if __name__ == "__main__":
    main()
