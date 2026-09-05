#!/usr/bin/env python3
"""Sweep C: position envelope -- init_start_random x box_xy_jitter (8 runs).

Plan: "20260904_Sweep C - Position envelope" (campaign "Sweeps From What
Worked", Wave 1 row 3).

THE QUESTION. The DR-on ceiling fell from 0.953 (RealDR_vel_as04_gas01_s0,
2026-07-27, commit 2109541, narrow pre-D18 envelope, 30-pose mid library) to
roughly 0.55 (SatSweep_a0b0c0d0_s0 0.484 with the same PPO bundle;
Snappy2/RewardGate 0.59-0.68) after the 2026-07-29 envelope widening D18-D24:
box_xy_jitter -> [0.17, 0.24], 60-pose mid library, lifter height 0-0.22 m.
Two axes moved together and cost nothing to isolate. This sweep asks which of
them carries the drop, and whether a narrower envelope saturates from scratch
under cube-mass DR rather than merely reproducing an old, DR-free number.

THE 2x2x2 DESIGN: init_start_random {light 30 poses, mid 60 poses} x
box_xy_jitter {narrow [0.085, 0.12], wide [0.17, 0.24]} x seed {0, 1}.

narrow is the curriculum L0.5 value, not the pre-D18 (0.15, 0.20) rectangle --
L0.5 is the envelope the ladder actually solved to 1.000, so a narrow-envelope
result here lands on ground the ladder already validated rather than on an
untested guess.

mid_wide is the CONTROL: it equals the defence line
Snappy2_as04_ar70_g01_s{0,1} (0.586 / 0.656) plus lifter_tilt_max=0.12 and the
ladder eval geometry (32 evals / 256 eval envs / 30M steps). Every other cell
is read as a delta against it.

Lifter height is the untested THIRD axis and deliberately stays at the env
default U(0, 0.22) -- that is Wave 2, not this sweep.

Both pose libraries are fingerprinted (not just "mid") because
init_poses/train/mid.json moved 30 -> 60 poses under a running experiment
(bb35de8, byte-identical metadata) -- a sweep that varies the library itself
must be able to prove which library each run trained on, and one
provenance-derived fingerprint cannot describe two different libraries at
once. See gen_satsweep.init_pose_library_fingerprint's `level` argument.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_satsweep as gs        # noqa: E402
import gen_rewardgate as rg      # noqa: E402
import gen_snappy2 as s2         # noqa: E402
import gen_dr_ladder as dl       # noqa: E402

POSE_LEVELS = ["light", "mid"]
XY_JITTER = {
    # wide == ur3_pick.default_config().box_xy_jitter; pinned explicitly here
    # so it lands in the run config / metadata.json rather than staying an
    # implicit env default.
    "narrow": [0.085, 0.12],
    "wide": [0.17, 0.24],
}
DEFENCE_ACTION_SCALE = 0.04
DEFENCE_ACTION_RATE = -0.70
# Imported so every row can be compared to the ladder runs; the defence line
# itself trained at 0.05.
LIFTER_TILT_MAX = dl.LIFTER_TILT_MAX
NUM_TIMESTEPS = 30_000_000
NUM_EVALS = 32
# The ladder eval geometry. A PPO param forwarded by run_experiment, so it
# must not be reserved.
NUM_EVAL_ENVS = 256
VARYING = {"seed", "run_id", "wandb_tags", "init_start_random", "box_xy_jitter"}


def cells():
    return [(lv, xy) for lv in POSE_LEVELS for xy in XY_JITTER]


def build_entry(pose_level, xy_name, seed, shared):
    # s2.build_entry IS the defence line: video_every_evals 5,
    # num_resets_per_eval 1, anchor PPO, gates g01, gripper_action_scale 0.02,
    # action_rate -0.7.
    entry = s2.build_entry(
        DEFENCE_ACTION_SCALE, DEFENCE_ACTION_RATE, seed, shared,
        NUM_TIMESTEPS, NUM_EVALS)
    entry.update({
        "run_id": f"SweepC_{pose_level}_{xy_name}_s{seed}",
        "num_eval_envs": NUM_EVAL_ENVS,
        "lifter_tilt_max": LIFTER_TILT_MAX,
        "init_start_random": pose_level,
        "box_xy_jitter": list(XY_JITTER[xy_name]),
        "wandb_tags": [
            "sweepC", f"pose{pose_level}", f"xy{xy_name}", f"s{seed}",
            f"anchor{rg.ANCHOR_CELL}",
        ],
    })
    return entry


def build_lines(seeds):
    nts, total = gs.solve_budget(NUM_TIMESTEPS, NUM_EVALS)
    shared = gs.load_target_config()

    assert (DEFENCE_ACTION_SCALE, DEFENCE_ACTION_RATE) in s2.cells(), (
        f"({DEFENCE_ACTION_SCALE}, {DEFENCE_ACTION_RATE}) is not one of "
        f"gen_snappy2's cells; the base cell must be one snappy2 actually ran")
    assert XY_JITTER["wide"] == list(dl._BOX_XY_JITTER), (
        f"XY_JITTER['wide'] {XY_JITTER['wide']} no longer mirrors "
        f"gen_dr_ladder._BOX_XY_JITTER {list(dl._BOX_XY_JITTER)}")
    for key in ("init_start_random", "box_xy_jitter", "lifter_tilt_max",
                "lifter_height_abs_min", "lifter_height_abs_max",
                "init_qpos_noise"):
        assert key not in shared, (
            f"target_config_satsweep.json now carries {key!r}; this sweep "
            f"sets/sweeps it itself and a duplicate source would silently "
            f"win or lose depending on dict merge order")

    with open(gs.TARGET_CONFIG, encoding="utf-8") as f:
        prov = json.load(f)["_provenance"]["inherited_env_defaults"]
    assert prov["init_start_random"] == "mid", (
        f"reference run's inherited init_start_random is "
        f"{prov['init_start_random']!r}, expected 'mid'; the control cell "
        f"SweepC_mid_wide would no longer BE the defence line's env")
    assert prov["box_xy_jitter"] == [0.17, 0.24], (
        f"reference run's inherited box_xy_jitter is {prov['box_xy_jitter']}, "
        f"expected [0.17, 0.24]; XY_JITTER['wide'] would silently stop being "
        f"the env default")

    fps = {lv: gs.init_pose_library_fingerprint(level=lv) for lv in POSE_LEVELS}

    cs = cells()
    n_runs = len(cs) * len(seeds)

    header = (
        f"# Sweep C: position envelope -- init_start_random x box_xy_jitter. "
        f"Generated by\n"
        f"# batch_runs/sweeps/gen_sweep_env_envelope.py -- DO NOT hand-edit; "
        f"regenerate.\n"
        f"# Plan: '20260904_Sweep C - Position envelope' (campaign 'Sweeps "
        f"From What Worked', Wave 1 row 3).\n"
        f"#\n"
        f"# QUESTION: the DR-on ceiling fell from 0.953 "
        f"(RealDR_vel_as04_gas01_s0, 2026-07-27, commit 2109541, narrow\n"
        f"# pre-D18 envelope, 30-pose mid library) to ~0.55 "
        f"(SatSweep_a0b0c0d0_s0 0.484 with the same PPO bundle;\n"
        f"# Snappy2/RewardGate 0.59-0.68) after the 2026-07-29 envelope "
        f"widening D18-D24: box_xy_jitter -> [0.17, 0.24],\n"
        f"# 60-pose mid library, lifter height 0-0.22 m. Which of the two "
        f"cheap axes carries the drop, and does a narrower\n"
        f"# envelope saturate from scratch under cube-mass DR?\n"
        f"#\n"
        f"# GRID (2x2x2 -- init_start_random x box_xy_jitter x seed):\n"
        f"#              | narrow xy              | wide xy\n"
        f"#   light pose | SweepC_light_narrow_s* | SweepC_light_wide_s*\n"
        f"#   mid pose   | SweepC_mid_narrow_s*   | SweepC_mid_wide_s*   "
        f"<- CONTROL\n"
        f"# eight runs: SweepC_light_narrow_s0, SweepC_light_narrow_s1,\n"
        f"# SweepC_light_wide_s0, SweepC_light_wide_s1, SweepC_mid_narrow_s0,\n"
        f"# SweepC_mid_narrow_s1, SweepC_mid_wide_s0, SweepC_mid_wide_s1.\n"
        f"#\n"
        f"# narrow={XY_JITTER['narrow']} is gen_curriculum's L0.5 value -- the "
        f"envelope the ladder actually\n"
        f"# solved to 1.000 -- NOT the pre-D18 (0.15, 0.20) rectangle.\n"
        f"#\n"
        f"# CONTROL: SweepC_mid_wide_s{{0,1}} is the defence line\n"
        f"# Snappy2_as04_ar70_g01_s{{0,1}} (0.586 / 0.656) plus "
        f"lifter_tilt_max=0.12 and the ladder eval geometry\n"
        f"# (32 evals / 256 eval envs / 30M steps).\n"
        f"#\n"
        f"# BASE BUNDLE on every line (via gen_snappy2.build_entry): the "
        f"Snappy2_as04_ar70_g01 bundle -- action_scale,\n"
        f"# action_rate, gripper_action_scale, gates, PPO anchor a1b0c1d1.\n"
        f"#\n"
        f"# PINNED on every line: lifter_tilt_max=0.12 (the ladder value); "
        f"num_eval_envs=256 (the ladder eval geometry).\n"
        f"#\n"
        f"# UNTESTED AXIS: lifter height stays at the env default "
        f"U(0, 0.22) -- that is Wave 2, not this sweep.\n"
        f"#\n"
        f"# Budget: num_timesteps={NUM_TIMESTEPS}, num_evals={NUM_EVALS},\n"
        f"# env_step_per_training_step={gs.env_step_per_training_step()},\n"
        f"# nts={nts} -> ACTUAL TOTAL_STEPS={total}.\n"
        f"#\n"
        f"# {len(cs)} cells x {len(seeds)} seed(s) = {n_runs} runs. "
        f"Comment/blank\n"
        f"# lines are skipped by run_one_ur3.py::load_config and do NOT "
        f"consume a SLURM array index: set --array=1-{n_runs}%4.\n"
        f"#\n"
        f"# INIT-POSE LIBRARY: level={fps['light'][0]!r} "
        f"n_poses={fps['light'][1]} sha256={fps['light'][2]}\n"
        f"# MUST match the July 'light' runs, e.g. "
        f"Grasp_light_as03_vel_dr_14M_s1 -- init_poses/train/light.json is a\n"
        f"# DATA dependency, not config.\n"
        f"# INIT-POSE LIBRARY: level={fps['mid'][0]!r} n_poses={fps['mid'][1]} "
        f"sha256={fps['mid'][2]}\n"
        f"# MUST match the snappy2/rewardgate/satsweep runs this sweep "
        f"compares against -- both pose libraries are DATA\n"
        f"# dependencies, not config, and train/mid.json moved 30 -> 60 "
        f"poses under a running experiment (bb35de8), with\n"
        f"# byte-identical metadata."
    )

    lines = [header]
    for lv, xy in cs:
        for seed in seeds:
            lines.append(json.dumps(build_entry(lv, xy, seed, shared)))
    return lines, total


def check_lines(path, expected_n, seeds):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f
                if l.strip() and not l.startswith("#")]
    assert len(rows) == expected_n, f"{len(rows)} data lines, expected {expected_n}"

    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate run_id: {ids}"

    seen = set()
    for r in rows:
        rid = r["run_id"]
        lv = r["init_start_random"]
        assert lv in POSE_LEVELS, f"{rid}: unexpected init_start_random {lv}"
        xy_val = r["box_xy_jitter"]
        assert xy_val in XY_JITTER.values(), f"{rid}: unexpected box_xy_jitter {xy_val}"
        xy = next(k for k, v in XY_JITTER.items() if v == xy_val)
        seed = r["seed"]
        assert rid == f"SweepC_{lv}_{xy}_s{seed}", \
            f"{rid}: run_id disagrees with its own (init_start_random, box_xy_jitter, seed)"
        assert r["wandb_tags"] == [
            "sweepC", f"pose{lv}", f"xy{xy}", f"s{seed}", f"anchor{rg.ANCHOR_CELL}",
        ], f"{rid}: wandb_tags disagree with its own cell"

        stray = (gs.REWARD_KEYS - {"action_rate"}) & set(r)
        assert not stray, f"{rid}: reward key(s) leaked: {sorted(stray)}"

        assert r["action_scale"] == DEFENCE_ACTION_SCALE, rid
        assert r["action_rate"] == DEFENCE_ACTION_RATE, rid
        assert r["gripper_action_scale"] == s2.GRIPPER_ACTION_SCALE, rid
        assert r[rg.GATE_BOX_KEY] is s2.GATE_BOX, rid
        assert r[rg.GATE_ALIGN_KEY] is s2.GATE_ALIGN, rid

        assert r["entropy_cost"] == rg.ANCHOR["entropy_cost"], rid
        assert r["learning_rate"] == rg.ANCHOR["learning_rate"], rid
        assert r["reward_scaling"] == rg.ANCHOR["reward_scaling"], rid
        nf = r["network_factory"]
        assert tuple(nf["policy_hidden_layer_sizes"]) == \
            rg.ANCHOR["policy_hidden_layer_sizes"], rid
        assert tuple(nf["value_hidden_layer_sizes"]) == \
            gs.VALUE_HIDDEN_LAYER_SIZES, rid

        assert r["episode_length"] == 400, rid
        assert r["box_z_rot_range"] == 6.283185307179586, rid
        assert r["obs_include_velocity"] is True, rid
        assert r["num_resets_per_eval"] == gs.NUM_RESETS_PER_EVAL, rid
        assert r["lifter_tilt_max"] == LIFTER_TILT_MAX, rid
        assert r["num_timesteps"] == NUM_TIMESTEPS, rid
        assert r["num_evals"] == NUM_EVALS, rid
        assert r["num_eval_envs"] == NUM_EVAL_ENVS, rid
        for k in ("lifter_height_abs_min", "lifter_height_abs_max", "init_qpos_noise"):
            assert k not in r, f"{rid}: {k} leaked"

        seen.add((lv, xy, seed))

    assert seen == {(lv, xy, s) for lv, xy in cells() for s in seeds}, \
        f"missing or extra (level, xy, seed) cells: {sorted(seen)}"

    base = {k: v for k, v in rows[0].items() if k not in VARYING}
    for r in rows[1:]:
        other = {k: v for k, v in r.items() if k not in VARYING}
        assert other == base, (
            f"{r['run_id']} differs from {rows[0]['run_id']} outside the "
            f"swept keys: "
            f"{ {k: (base.get(k), other.get(k)) for k in set(base) | set(other) if base.get(k) != other.get(k)} }")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite (this file "
            f"is generated, not hand-edited)")

    lines, total = build_lines(args.seeds)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n = len(lines) - 1
    check_lines(args.out, n, args.seeds)
    print(f"Wrote {n} runs to {args.out}")
    print(f"  cells: {cells()}")
    print(f"  ACTUAL TOTAL_STEPS per run: {total}")
    print(f"  set --array=1-{n}%4")


if __name__ == "__main__":
    main()
