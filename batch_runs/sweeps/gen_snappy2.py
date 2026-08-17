#!/usr/bin/env python3
"""action_scale x action_rate arm -- restoring the pairing the lineage broke.

WHAT THE FIRST SNAPPY ARM FOUND (Snappy_as{025,015}_g01_s{0,1}, 2026-08-17):

  action_scale | mean succ | t_lift | action_rate/step | rms|da| | jitter deg/step
      0.040    |   0.633   |  129   |      3.284       |  0.685  |     1.57
      0.025    |   0.508   |  155   |      3.231       |  0.679  |     0.97
      0.015    |   0.199   |  193   |      2.842       |  0.637  |     0.55

The rms|da| column is the point: the policy's NORMALIZED action noise barely
moves (0.685 -> 0.637) across a 2.7x change in action_scale. Shrinking
action_scale does not teach the policy to be smooth -- it scales every command
down, useful ones included. Hence t_lift 129 -> 193 and success 0.633 -> 0.199.
Smoothness bought this way costs roughly 0.005 success per 0.01 deg/step.

WHY action_rate IS THE RIGHT KNOB. In ur3_pick.py:3143 the term is

    action_rate_Reward = jp.sum(jp.square(action - last_action))

computed on the RAW normalized action, so the penalty is SCALE-INDEPENDENT
while the physical jitter is `da * action_scale`. The slowdown sweep that
settled action_rate=-0.10 (eaabb6b, Smooth_slow_mid_as015_ar10) tuned it AT
action_scale=0.015. This lineage kept ar10 but moved the scale to 0.04, so the
penalty now permits 2.7x more physical jitter than where it was tuned. That
broken pairing -- not the scale alone -- is the snappiness.

Raising action_rate attacks rms|da| directly and leaves the arm's authority
(and therefore t_lift) intact, which is exactly what the scale knob could not do.

THE DESIGN: action_scale x action_rate, minus the cell that already exists.

              | ar -0.10          | ar -0.30 | ar -0.70
    as 0.040  | RewardGate_g01 *  |   run    |   run
    as 0.030  |      run          |   run    |   run

  * NOT REGENERATED. RewardGate_g01_s{0,1} (0.602 / 0.664) already is
    as=0.04 / ar=-0.10 at this exact configuration. 5 cells x 2 seeds = 10 runs.

as=0.030 is Matthias's hypothesis ("a bit slower than 0.04, but not as far back
as 0.025"). It is carried as a full factor rather than dropped, so his reading
and the action_rate diagnosis are tested against each other rather than one
being assumed.

SCOPE NOTE -- this arm deliberately breaks an invariant the two predecessor
sweeps enforced. gen_satsweep and gen_rewardgate both REJECT any reward-scale
key ("the value function is given"; "this plan changes GATING, not magnitudes").
action_rate is such a key -- run_experiment.py maps the bare JSONL key
"action_rate" onto reward_config.scales.action_rate. Changing it is a reward
MAGNITUDE change and is recorded as an explicit scope extension in the Planfile,
not smuggled in. Every OTHER reward scale stays frozen and is still guarded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_satsweep as gs        # noqa: E402
import gen_rewardgate as rg      # noqa: E402
import gen_snappy as sn          # noqa: E402

GATE_BOX = sn.GATE_BOX
GATE_ALIGN = sn.GATE_ALIGN
GATE_CELL = sn.GATE_CELL
GRIPPER_ACTION_SCALE = sn.GRIPPER_ACTION_SCALE

ACTION_SCALES = [0.04, 0.03]
ACTION_RATES = [-0.10, -0.30, -0.70]

# The cell that already exists in W&B and must NOT be regenerated.
BASELINE_CELL = (0.04, -0.10)
BASELINE_RUNS = "RewardGate_g01_s0 (0.602), RewardGate_g01_s1 (0.664)"

# The env default, asserted so a silent drift in the frozen config or in the env
# cannot change what "baseline" means without failing loudly here.
DEFAULT_ACTION_RATE = -0.10


def ar_tag(v):
    """-0.10 -> 'ar10', -0.30 -> 'ar30', -0.70 -> 'ar70'.

    Matches the existing convention in Smooth_slow_mid_as015_ar10.
    """
    return "ar" + f"{abs(v):g}".replace("0.", "", 1).ljust(2, "0")


def cells():
    return [(a, r) for a in ACTION_SCALES for r in ACTION_RATES
            if (a, r) != BASELINE_CELL]


def build_entry(action_scale, action_rate, seed, shared, num_timesteps, num_evals):
    tag = f"{sn.scale_tag(action_scale)}_{ar_tag(action_rate)}"
    entry = dict(shared)
    entry.update({
        "seed": int(seed),
        "run_id": f"Snappy2_{tag}_{GATE_CELL}_s{seed}",
        "video_every_evals": rg.VIDEO_EVERY_EVALS,
        "num_timesteps": num_timesteps,
        "num_evals": num_evals,
        "num_resets_per_eval": gs.NUM_RESETS_PER_EVAL,
        # --- frozen PPO anchor (Tier A winner) ---
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
        # --- the two knobs under test ---
        "action_scale": float(action_scale),
        # bare key on purpose: run_experiment.py:1030-1035 maps it onto
        # reward_config.scales.action_rate via update_from_flattened_dict.
        "action_rate": float(action_rate),
        # --- pinned so the third smoothness knob provably did NOT move ---
        "gripper_action_scale": GRIPPER_ACTION_SCALE,
        "wandb_tags": [
            "snappy2", sn.scale_tag(action_scale), ar_tag(action_rate), GATE_CELL,
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

    assert shared.get("action_scale") == BASELINE_CELL[0], (
        f"target_config_satsweep.json has action_scale="
        f"{shared.get('action_scale')}, expected {BASELINE_CELL[0]}; the "
        f"baseline runs {BASELINE_RUNS} no longer describe the omitted cell")
    assert "action_rate" not in shared, (
        "the frozen target config now pins action_rate; the baseline runs were "
        "trained at the env default and the omitted cell is no longer valid")

    cs = cells()
    n_runs = len(cs) * len(seeds)
    header = (
        f"# Snappiness arm 2: action_scale x action_rate. Generated by\n"
        f"# batch_runs/sweeps/gen_snappy2.py -- DO NOT hand-edit; regenerate.\n"
        f"# Plan: '20260816_Post-lift annuity - reward gating' (parallel arm).\n"
        f"#\n"
        f"# FINDING FROM ARM 1 that motivates this one: across action_scale\n"
        f"# 0.04 -> 0.025 -> 0.015 the policy's NORMALIZED action noise barely\n"
        f"# moved (rms|da| 0.685 -> 0.679 -> 0.637) while success collapsed\n"
        f"# 0.633 -> 0.508 -> 0.199 and t_lift rose 129 -> 155 -> 193. Shrinking\n"
        f"# action_scale does not make the policy smooth; it shrinks every\n"
        f"# command, useful ones included.\n"
        f"#\n"
        f"# ur3_pick.py:3143 computes action_rate on the RAW normalized action,\n"
        f"# so the penalty is SCALE-INDEPENDENT while physical jitter is\n"
        f"# da*action_scale. The sweep that settled action_rate=-0.10 (eaabb6b,\n"
        f"# Smooth_slow_mid_as015_ar10) tuned it AT action_scale=0.015; this\n"
        f"# lineage kept ar10 and moved to 0.04, so the penalty now permits 2.7x\n"
        f"# more physical jitter than where it was tuned. Raising action_rate\n"
        f"# attacks rms|da| directly and leaves arm authority (t_lift) intact.\n"
        f"#\n"
        f"# GRID (baseline cell omitted, not regenerated):\n"
        f"#               | ar -0.10        | ar -0.30 | ar -0.70\n"
        f"#     as 0.040  | RewardGate_g01  |   run    |   run\n"
        f"#     as 0.030  |      run        |   run    |   run\n"
        f"#\n"
        f"# {BASELINE_RUNS}\n"
        f"# already IS as=0.04 / ar=-0.10 at this exact configuration. Compare\n"
        f"# against those two by name; do NOT re-run them.\n"
        f"#\n"
        f"# as=0.030 is Matthias's hypothesis ('a bit slower than 0.04, but not\n"
        f"# as far back as 0.025'), carried as a full factor so his reading and\n"
        f"# the action_rate diagnosis are tested against each other.\n"
        f"#\n"
        f"# SCOPE NOTE: action_rate is a reward MAGNITUDE key -- the bare JSONL\n"
        f"# key maps onto reward_config.scales.action_rate. gen_satsweep and\n"
        f"# gen_rewardgate both REJECT reward-scale keys; this arm deliberately\n"
        f"# allows exactly this one, recorded as a scope extension in the\n"
        f"# Planfile. Every OTHER reward scale stays frozen and is still guarded.\n"
        f"#\n"
        f"# FROZEN: {rg.GATE_BOX_KEY}={GATE_BOX}, "
        f"{rg.GATE_ALIGN_KEY}={GATE_ALIGN} (gating winner {GATE_CELL});\n"
        f"#   gripper_action_scale={GRIPPER_ACTION_SCALE:g} (third smoothness knob,\n"
        f"#   still untouched -- its comment justifies 0.01 while the value is\n"
        f"#   0.02 = 1.25 steps for the 0.025 full travel);\n"
        f"#   PPO at Tier A winner {rg.ANCHOR_CELL}.\n"
        f"#\n"
        f"# JUDGE ON THE TRADEOFF: report eval/episode_success AND t_lift next to\n"
        f"# eval/episode_action_rate and tcp_speed. The quantity that matters is\n"
        f"# rms|da| = sqrt((action_rate/episode_length)/7) -- if raising the\n"
        f"# penalty does NOT move rms|da| below arm 1's ~0.68, the knob is not\n"
        f"# working and the snappiness is coming from somewhere else.\n"
        f"#\n"
        f"# Budget: num_timesteps={num_timesteps}, num_evals={num_evals},\n"
        f"# env_step_per_training_step={gs.env_step_per_training_step()},\n"
        f"# nts={nts} -> ACTUAL TOTAL_STEPS={total}. Identical to every prior arm.\n"
        f"#\n"
        f"# {len(cs)} cells x {len(seeds)} seed(s) = {n_runs} runs. Comment/blank\n"
        f"# lines are skipped by run_one_ur3.py::load_config and do NOT consume a\n"
        f"# SLURM array index: set --array=1-{n_runs}%4.\n"
        f"#\n"
        f"# INIT-POSE LIBRARY: level={pose_level!r} n_poses={pose_n} "
        f"sha256={pose_sha}\n"
        f"# MUST match the runs these are compared against."
    )

    lines = [header]
    for a, r in cs:
        for seed in seeds:
            lines.append(json.dumps(
                build_entry(a, r, seed, shared, num_timesteps, num_evals)))
    return lines, total


def check_lines(path, expected_n, seeds):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip() and not l.lstrip().startswith("#")]
    assert len(rows) == expected_n, f"{len(rows)} data lines, expected {expected_n}"

    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate run_id: {ids}"

    # Every reward key EXCEPT action_rate is still forbidden. The scope
    # extension is exactly one key wide, and this is what proves it.
    allowed = gs.REWARD_KEYS - {"action_rate"}

    seen = set()
    for r in rows:
        rid = r["run_id"]
        assert not (allowed & set(r)), (rid, allowed & set(r))
        a, ar = r["action_scale"], r["action_rate"]
        assert a in ACTION_SCALES, f"{rid}: unexpected action_scale {a}"
        assert ar in ACTION_RATES, f"{rid}: unexpected action_rate {ar}"
        assert (a, ar) != BASELINE_CELL, (
            f"{rid}: regenerates the baseline cell, which already exists as "
            f"{BASELINE_RUNS}")
        assert sn.scale_tag(a) in rid and ar_tag(ar) in rid, \
            f"{rid}: run_id disagrees with its own ({a}, {ar})"
        seen.add((a, ar, r["seed"]))
        assert r["gripper_action_scale"] == GRIPPER_ACTION_SCALE, rid
        assert r[rg.GATE_BOX_KEY] is GATE_BOX, rid
        assert r[rg.GATE_ALIGN_KEY] is GATE_ALIGN, rid
        assert r["entropy_cost"] == rg.ANCHOR["entropy_cost"], rid
        assert r["learning_rate"] == rg.ANCHOR["learning_rate"], rid
        assert r["reward_scaling"] == rg.ANCHOR["reward_scaling"], rid
        nf = r["network_factory"]
        assert tuple(nf["policy_hidden_layer_sizes"]) == \
            rg.ANCHOR["policy_hidden_layer_sizes"], rid
        assert tuple(nf["value_hidden_layer_sizes"]) == \
            gs.VALUE_HIDDEN_LAYER_SIZES, rid
        assert r["box_z_rot_range"] == 6.283185307179586, rid
        assert r["episode_length"] == 400, rid
        assert r["obs_include_velocity"] is True, rid

    assert seen == {(a, r, s) for a, r in cells() for s in seeds}, \
        f"missing or extra (action_scale, action_rate, seed) cells: {sorted(seen)}"

    varying = {"seed", "run_id", "wandb_tags", "action_scale", "action_rate"}
    base = {k: v for k, v in rows[0].items() if k not in varying}
    for r in rows[1:]:
        other = {k: v for k, v in r.items() if k not in varying}
        assert other == base, (
            f"{r['run_id']} differs from {rows[0]['run_id']} outside the two "
            f"swept knobs: "
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
    print(f"  cells: {[(f'{a:g}', f'{r:g}') for a, r in cells()]}")
    print(f"  omitted baseline cell {BASELINE_CELL} -> {BASELINE_RUNS}")
    print(f"  ACTUAL TOTAL_STEPS per run: {total}")
    print(f"  set --array=1-{n}%4")


if __name__ == "__main__":
    main()
