#!/usr/bin/env python3
"""Generate the v3 reward-fix PILOT sweep (UR3Pick_reward_fix_v3_pilot.jsonl).

WHY A PILOT AT ALL
------------------
The v3 ladder is 15 runs x 12 h. Before spending that, we need to know that the
reward changes help, because the starting point is weak: on the held-out D23
protocol the v2 policies succeeded in SIM only 0/21, 2/18, 5/23, 2/15, 0/15 of
episodes. A ladder trained on a worse reward would be 45 h of wall-clock spent
proving nothing.

Every line is L1_pos (the cheapest informative rung: position DR only, i.e. the
baked default_config, no `domain_rand.*`), seed 0, so the lines differ ONLY in
the factor under test.

WHAT THE PILOT IS TESTING (the three D23 diagnoses)
--------------------------------------------------
1. ALIGNMENT / GRASP GATE. Measured D23 stage retention collapses at the grasp:
   approach 0.52-0.81, grasp 0.06-0.21, transport 0.004-0.34. Cause: the box is
   a 3x3x4 cm PRISM and the legacy score takes max|axis . box_axis| over ALL
   THREE box axes, so a jaw spanning the 4 cm axis (9.9 mm total finger
   clearance against the 49.9 mm opening) scores exactly as well as a 3 cm face
   grasp (19.9 mm), and a horizontal approach scores as well as top-down. On top
   of that grasp_align_thresh=0.3 let a 39.3-deg-misaligned grasp latch
   `grasped`, which unlocks lift(5)+box_target(20)+hold_target(6) = 31 raw/step
   for the rest of the episode. align_mode="axis_aware" fixes the scoring;
   lines 2-4 bracket the threshold.

2. ANTI-MOTION PRESSURE ("slow when far"). Measured per-tick economics at
   d = 0.40 m: gripper_box pays +0.023 for closing distance flat out, while
   action_rate costs -0.175 (|dA| = 0.5 across 7 dims) and robot_target_qpos
   costs -0.163 SUSTAINED for having left the start pose. Standing still was
   optimal until d ~= 2 cm. Lines 5-7 cut that opposition.
   HONEST LIMIT: cutting the opposition moves the break-even out to ~15 cm, not
   to 40 cm. A tanh cascade structurally cannot pull hard at long range --
   k*sech^2(k*d) peaks at k ~= 1.9 for d = 0.40 m, so the existing coarse scale
   of 1.5 is already within ~15% of the best any single tanh can do there. Line
   8 tries the only in-term lever left (reweight gripper_box). If t_reach /
   tcp_speed still look bad after this pilot, the remaining option is
   potential-based progress shaping, which is deliberately NOT in this pilot.

3. REAL-ROBOT SPEED. Not a training issue at all -- the deploy control law
   rebases the servoJ target on the measured joint position every tick
   (robots/UR3e/ur3_realrobot_dependencies.py) instead of integrating it as sim
   does, which measured 0.137 achieved/commanded across all 5 policies and 127
   runs. No sweep line can address it; see that file's control_law parameter.

HOW TO READ THE RESULT (the metrics added alongside this change)
---------------------------------------------------------------
Do NOT compare eval/episode_reward across lines -- axis_aware changes the
dynamic range of gripper_align (5.0) and grasp (3.0), and two scales move, so
returns are not commensurable. Compare:
  * eval/episode_success            -- the outcome that matters
  * eval/episode_align_at_grasp     -- alignment at the moment `grasped` latched
  * eval/episode_jaw_span_at_grasp  -- expect ~0.030 (3 cm face), NOT 0.040
  * eval/episode_grasp_gate_blocked -- large WHILE grasped ~ 0 => gate too tight
  * eval/episode_t_reach / _t_grasp -- steps to stage (400 == never happened)
  * eval/episode_tcp_speed          -- summed; divide by episode_length for m/s

GATES BEFORE LAUNCHING THE LADDER
---------------------------------
  * pick the grasp_align_thresh where align_at_grasp is well above 0.30 in the
    new units and jaw_span_at_grasp ~ 0.030, WITHOUT `grasped` collapsing;
  * pick the anti-motion dose that lowers t_reach / raises tcp_speed at
    equal-or-better success.
Feed both into gen_dr_ladder_velocity.py's v3 constants.

Usage:
    .venv/bin/python batch_runs/sweeps/gen_reward_fix_v3_pilot.py --force
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from batch_runs.sweeps.gen_dr_ladder import (  # noqa: E402
    EPISODE_LENGTH,
    WANDB_PROJECT,
)
from batch_runs.sweeps.gen_dr_ladder_velocity import (  # noqa: E402
    ACTION_SCALE,
    GRIPPER_ACTION_SCALE,
)

DEFAULT_OUT = os.path.join(
    REPO_ROOT, "batch_runs", "sweeps", "UR3Pick_reward_fix_v3_pilot.jsonl"
)

# Legacy (v2) values, stated explicitly on EVERY line -- including the control.
# Two reasons this is not redundant with default_config():
#   1. only a value that reaches env_overrides -> metadata.json is a real
#      trained value that a deploy-time or replay-time check can trust (this is
#      the exact distinction that broke the first real campaign, when policies
#      inheriting action_scale=0.015 were deployed at 0.04);
#   2. grasp_align_thresh's UNITS depend on align_mode -- 39 deg of
#      misalignment scores 0.307 under axis_free but 0.091 under axis_aware --
#      so the pair must never be split across a default and an override.
_V2 = {
    "align_mode": "axis_free",
    "grasp_align_thresh": 0.30,
    "align_pref_floor": 0.15,
    "action_rate": -0.10,
    "robot_target_qpos": 0.3,
}

# (run_id, overrides-on-top-of-_V2, extra wandb tags)
_LINES = [
    # 1. Control. Sets ONLY the v2 values, so it must reproduce
    #    L1_pos_vel_v2_s0 within seed noise. If it does not, something in the
    #    v3 code changed the axis_free path and everything below is unreadable.
    ("P0_ctrl", {}, ["control", "axis_free"]),

    # 2-4. Alignment + gate bracket. axis_aware alone changes WHICH grasps
    #      score well; the threshold changes which ones may latch. Bracketed
    #      because too tight starves the lift/box_target/hold chain of any
    #      bootstrap signal and the run flatlines -- watch grasp_gate_blocked.
    #      Calibration (both axes off by theta, nominal box):
    #        15 deg -> 0.582   20 deg -> 0.443   25 deg -> 0.321   30 deg -> 0.220
    ("P1_align45", {"align_mode": "axis_aware", "grasp_align_thresh": 0.45},
     ["axis_aware", "thresh0.45"]),
    ("P2_align35", {"align_mode": "axis_aware", "grasp_align_thresh": 0.35},
     ["axis_aware", "thresh0.35"]),
    ("P3_align55", {"align_mode": "axis_aware", "grasp_align_thresh": 0.55},
     ["axis_aware", "thresh0.55"]),

    # 5-7. Anti-motion dose-response, on top of the mid alignment setting so the
    #      two factors are not confounded with each other.
    ("P4_motion", {"align_mode": "axis_aware", "grasp_align_thresh": 0.45,
                   "action_rate": -0.02, "robot_target_qpos": 0.05},
     ["axis_aware", "thresh0.45", "motion_full"]),
    ("P5_motion_soft", {"align_mode": "axis_aware", "grasp_align_thresh": 0.45,
                        "action_rate": -0.05, "robot_target_qpos": 0.15},
     ["axis_aware", "thresh0.45", "motion_half"]),
    # Is the stay-home bonus needed at all once it stops fighting the approach?
    # It was introduced to stop the arm wandering during approach; if success
    # holds at 0.0 it is pure drag.
    ("P6_rtq0", {"align_mode": "axis_aware", "grasp_align_thresh": 0.45,
                 "action_rate": -0.02, "robot_target_qpos": 0.0},
     ["axis_aware", "thresh0.45", "rtq0"]),

    # 8. The only remaining in-term lever on long-range pull (see the module
    #    docstring's k*sech^2 argument for why retuning the cascade's scales
    #    cannot substitute for this).
    ("P7_gripbox6", {"align_mode": "axis_aware", "grasp_align_thresh": 0.45,
                     "action_rate": -0.02, "robot_target_qpos": 0.05,
                     "gripper_box": 6.0},
     ["axis_aware", "thresh0.45", "motion_full", "gripper_box6"]),
]

_HEADER = f"""\
# UR3Pick v3 REWARD-FIX PILOT -- generated by
# batch_runs/sweeps/gen_reward_fix_v3_pilot.py. DO NOT hand-edit; regenerate.
#
# {len(_LINES)} lines, all L1_pos (position DR only) x seed 0, one factor at a
# time. Gates the 15-run v3 ladder; see the generator docstring for what each
# line tests and how to read the result.
#
# Every line states align_mode + grasp_align_thresh + action_rate +
# robot_target_qpos EXPLICITLY, including the control -- grasp_align_thresh's
# units depend on align_mode, and only values that reach env_overrides ->
# metadata.json are trustworthy at replay/deploy time.
#
# COMPARE ON: eval/episode_success, _align_at_grasp, _jaw_span_at_grasp,
# _grasp_gate_blocked, _t_reach, _t_grasp, _tcp_speed.
# NOT on eval/episode_reward -- axis_aware changes the reward's dynamic range,
# so returns are not commensurable across these lines.
#
# Comment/blank lines are skipped by the runner and do NOT count toward the
# 1-based SLURM_ARRAY_TASK_ID. --array MUST equal the data-line count ({len(_LINES)}).
"""


def build_lines():
    lines = [_HEADER]
    for run_id, overrides, tags in _LINES:
        entry = {
            "env_name": "UR3Pick",
            "seed": 0,
            "run_id": f"{run_id}_v3pilot_s0",
            "wandb_project": WANDB_PROJECT,
            "video_every_evals": 6,
            "render_every": 1,
            # NEVER inherit episode_length: default_config()'s 250 is stale
            # (it predates action_scale=0.015) and omitting it once killed all
            # 30 runs of an earlier ladder -- at a 5.0 s horizon the policies
            # farmed approach reward and never bootstrapped the lift.
            "episode_length": EPISODE_LENGTH,
            "action_scale": ACTION_SCALE,
            "gripper_action_scale": GRIPPER_ACTION_SCALE,
            "obs_include_velocity": True,
            **_V2,
            **overrides,
            "wandb_tags": ["v3_pilot", "reward_fix", "L1_pos", "velocity",
                           "s0", *tags],
        }
        lines.append(json.dumps(entry))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="required to overwrite an existing file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite.")

    # Guard the same class of mistake the ladder generator guards: a threshold
    # that is out of range, or an align_mode typo, would train silently wrong
    # and only surface as a flat result hours later. ur3_pick.__init__ raises on
    # a bad align_mode too, but failing here is cheaper than failing on a GPU.
    for run_id, ov, _ in _LINES:
        merged = {**_V2, **ov}
        assert merged["align_mode"] in ("axis_free", "axis_aware"), run_id
        assert 0.0 <= merged["grasp_align_thresh"] <= 1.0, run_id
        assert 0.0 <= merged["align_pref_floor"] < 1.0, run_id
        assert merged["action_rate"] <= 0.0, f"{run_id}: action_rate must be <= 0"
        assert merged["robot_target_qpos"] >= 0.0, run_id
    assert len({r for r, _, _ in _LINES}) == len(_LINES), "duplicate run_id"

    lines = build_lines()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    n = len(lines) - 1
    print(f"wrote {args.out}")
    print(f"  {n} data lines -> SLURM --array=1-{n}%4")
    for run_id, ov, _ in _LINES:
        merged = {**_V2, **ov}
        print(f"    {run_id:16s} align={merged['align_mode']:10s} "
              f"thresh={merged['grasp_align_thresh']:.2f} "
              f"action_rate={merged['action_rate']:+.2f} "
              f"rtq={merged['robot_target_qpos']:.2f}"
              + (f" gripper_box={ov['gripper_box']}" if "gripper_box" in ov else ""))


if __name__ == "__main__":
    main()
