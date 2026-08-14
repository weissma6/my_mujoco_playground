# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""UR3 pick task: 6-DOF arm + Hand-E gripper, lift a box to a target point.

The mocap target is used as the lift goal (a point in the air above the box).
The 4x4x4 cm box spawns with a random Z-axis yaw (range set by box_z_rot_range).
The box spawns on the "lifter" body -- the lab TABLE (measured 0.6 m in X x 1.0 m
in Y, x in [0.10, 0.70] and y in [-0.50, 0.50], top surface nominally 95 mm above
the base origin), whose height and level are randomized per episode (an ABSOLUTE
top-surface draw U(lifter_height_abs_min, lifter_height_abs_max) plus
lifter_tilt_max) so the policy learns to grasp off a table it can't assume the
exact pose of. The table itself stays FIXED at its XML XY -- it is a fixture, not
a riser that follows the cube. The arm start pose can be drawn from a library of
hand-collected real-robot poses (init_start_random).
Mirrors the commented-out reward scaffolding of ur10pick.py.
"""

import warnings
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.my_ur3 import ur3_base
from mujoco_playground._src.manipulation.my_ur3.init_poses import load_init_poses
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

# Lifter geometry. The "lifter" body is the lab TABLE the cube is picked from
# (0.6 x 1.0 m plate, see xmls/mjx_single_cube_position_ur3.xml): a kinematic
# mocap body whose top surface sets the box's starting height. reset() places it
# at an ABSOLUTE sampled top height (D18/D24: U(lifter_height_abs_min,
# lifter_height_abs_max), no longer a +- band about the nominal) plus a slight
# tilt, so the policy learns to grasp off a table whose height/level it can't
# assume. Its XY is pinned at the XML body pos every episode -- the table is a
# fixed fixture of the cell, NOT a riser that follows the cube.
# _LIFTER_HEIGHT_MIN is a GUARD, not the nominal: it clamps the sampled height so
# a wide jitter draw can never sink the plate bottom into the floor plane (z=0),
# which the collision masks alone can't separate.
_LIFTER_HALF_THICKNESS = 0.0025  # 5 mm plate -> half-extent
_LIFTER_HEIGHT_MIN = 0.003
_BOX_HALF_EXTENT = 0.02  # half the box HEIGHT (3x3x4 cm box, 4 cm tall) -> rest offset
# D19: nominal box WIDTH (xy) half-extent -- the box is 3x3x4 cm, so xy=0.015,
# z=_BOX_HALF_EXTENT=0.02. Identity value for the cube_size_xy DR axis (see
# default_config().domain_rand.cube_size_xy).
_BOX_HALF_EXTENT_XY = 0.015
# Tolerance (m) for the table/box-anchor consistency guards in __init__. See the
# comment at the lifter check for why this is 1 um and not 0.
_ANCHOR_TOL = 1e-6
# Grasp-frame alignment cone bound, cos(60 deg). Shared by the face scores and
# by the axis-aware top-down score, so widening the cone widens both together.
# Was a local `_cos_bound` inside _get_reward; hoisted so the alignment smoke
# tests can import the exact value the env uses instead of re-typing 0.5.
_COS_BOUND = 0.5

# --- Reach envelope (D18 open risk, re-derived for the 95 mm table in D24) ---
# The UR3e's usable WORKING radius with a cube in the jaws is ~0.49 m. That
# number is not invented here: it is what this file's own target_r_max comment
# already derives -- sqrt(target_r_max^2 + max_target_z^2) = sqrt(0.35^2 +
# 0.345^2) ~= 0.49 -- i.e. the far corner of the target band that was validated
# to still be reachable. _UR3_APPROX_MAX_REACH_M (0.54) is the DATASHEET reach
# with nothing held; keep both, they answer different questions.
_UR3_APPROX_MAX_REACH_M = 0.54
_UR3_WORKING_RADIUS_M = 0.49
# Base-column clearance: the box must never spawn closer than this to the base
# origin, or the arm cannot get around it (and the column is in the way).
_BOX_XY_R_MIN = 0.18
# Hard cap on the ABSOLUTE world-Z of any sampled lift target.
#   at target_r_max = 0.35, staying inside the 0.49 m working radius allows
#   z <= sqrt(0.49^2 - 0.35^2) = sqrt(0.2401 - 0.1225) = sqrt(0.1176) ~= 0.343
# rounded to the value that ALSO equals this branch's pre-D24 maximum target z
# (target_z_jitter max 0.21 + box anchor 0.115 + the old +-0.02 table jitter
# = 0.345), so the cap changes nothing for the previously validated range and
# only clips the NEW high-table draws (D18 raised the table top to 0.220, which
# stacked on a 0.21 lift draw would demand sqrt(0.35^2 + 0.45^2) ~= 0.57 m).
# Physically sensible: you cannot lift as far above an already-high table.
# NOTE: this is 0.345, NOT the 0.40 used on the (floor-based) novelocitymodel
# branch -- 0.40 was too permissive once the box anchor carries the table.
_TARGET_WORLD_Z_CAP = 0.345

# D17 cube-size probe: the info-dict key reset_to_state() stashes an OPTIONAL
# eval-only box geom half-extents override under, and step() reads back to
# apply it UNCLAMPED (bypassing _dr_max_box_half_xy) every substep. See
# reset_to_state()'s docstring. Absent (the default) -> no D17 override, step()
# behaves exactly as before this axis existed.
_DR17_EVAL_CUBE_HALF_EXTENTS_KEY = "eval_cube_half_extents"

# Per-episode domain-randomization factors logged to W&B (terminal-gated, see
# step()). Realized scale per axis + the gravity z-component.
_DR_METRIC_KEYS = (
    "cube_mass",
    "cube_friction",
    # D19: cube size decoupled into two independent absolute half-extent draws.
    "cube_size_xy",
    "cube_size_z",
    "gravity_z",
    "arm_stiffness",
    "arm_damping",
)

# 2026-08 approach/grasp diagnostics. Before these, the approach phase was
# effectively invisible in W&B: grip_box_dist, alignment and finger_touch_dist
# were computed into _get_reward's raw_signals but never copied into `metrics`,
# and nothing about speed or time-to-stage existed at all -- so "the approach is
# slow" and "the grasp is misaligned" were both unmeasurable from a training run.
#
# Declared once and consumed by BOTH metrics dicts (reset and reset_to_state) so
# they cannot drift apart -- brax's EvalWrapper requires the reset and step
# metric key sets to match exactly, and a key seeded in only one of the two
# resets fails at eval time, not at trace time.
#
# PER-STEP (brax SUMS these into eval/episode_*; divide by episode_length for a
# mean).
_STEP_METRIC_KEYS = (
    "grip_box_dist",        # TCP->box distance, m
    "finger_touch_dist",    # pad separation, m
    "alignment",            # the value that actually gates the `grasped` latch
    "jaw_span",             # box support width along the jaw axis, m
    "tcp_speed",            # m/s
    "grasp_gate_blocked",   # steps where only the alignment gate blocked a grasp
)
# TERMINAL-GATED (nonzero on the single terminal step, so the W&B episode-sum
# equals that one value and the cross-env mean is the true mean -- same pattern
# as box_target_dist_final).
_TERMINAL_METRIC_KEYS = (
    "grip_box_dist_min",    # closest TCP->box approach, m
    "align_at_grasp",       # alignment at the rising edge of `grasped`
    "jaw_span_at_grasp",    # jaw span at that same edge, m (expect ~0.030)
    "t_reach",              # steps to first reach  (== episode_length if never)
    "t_grasp",              # steps to first grasp  (== episode_length if never)
    "t_lift",               # steps to first lift   (== episode_length if never)
)


def _skew(v: jax.Array) -> jax.Array:
    """3x3 skew-symmetric (cross-product) matrix of a 3-vector.

    Used for the small-angle box-orientation obs-noise perturbation
    (domain_rand.obs_noise): for a small rotation vector `d` (rad),
    `R_noisy_axes ~= axes + skew(d) @ axes` is the standard first-order SO(3)
    approximation (`R(d) ~= I + skew(d)` for |d| << 1). Valid here because
    the configured biases/jitters are ~1-3 deg (~0.02-0.05 rad); the
    resulting matrix is not re-orthonormalized since only its dot products
    with jaw/approach axes are ever used (see UR3Pick._get_obs), not its
    validity as a rotation.
    """
    return jp.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product of two [w, x, y, z] quaternions (MuJoCo convention)."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def default_config() -> config_dict.ConfigDict:
    """Default config for the UR3 pick task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        # 250 steps = 5s (was 300 = 6s, before that 200 = 4s). 4s left no time
        # to transport a lifted box across a shifted target
        # (box_y_center_offset/target_y_center_offset) or to recover from a
        # drop -- FIXVERIFY hard-pose episodes ended ~190 steps having never
        # held the box at the target, hence the bump to 300. But 300 walked
        # back: gripper_align is approach-only (a fixed number of steps per
        # episode) while lift/box_target/hold pay every step, so a longer
        # episode dilutes the align share of the return and inflates the
        # return magnitude that reward_scaling was tuned for (see
        # manipulation_params.py). 250 keeps most of the transport/recovery
        # headroom 300 added while cutting both costs. num_timesteps/num_evals
        # are unaffected -- episode horizon doesn't change total env steps or
        # eval cadence. run_experiment.py reads this from the env default
        # config and forwards it to ppo.train, so env/EpisodeWrapper/eval stay
        # consistent automatically.
        #
        # !! STALE -- DO NOT INHERIT THIS IN A SWEEP. 250 was tuned at b0731d7,
        # when action_scale was 0.01. eaabb6b's slowdown sweep then chose
        # action_scale=0.015 at ep400 (Smooth_slow_mid_as015_ar10), and 1636c98
        # baked that winner's action_scale/action_rate but NOT its
        # episode_length -- its DR ablation kept carrying ep400 per-line
        # instead. So every run validated at the CURRENT defaults trained at
        # 400, incl. DR_baseline_ep400 and DR_cube_mass_light (the policy that
        # lifted the cube on the real robot); ep250 at action_scale=0.015 has
        # never been shown to learn the lift. The DR-ladder sweep omitted
        # episode_length on the assumption that the default was the winner and
        # trained all 30 runs at 5.0 s instead of 8.0 s: they farmed the dense
        # approach reward and never bootstrapped lift/box_target (both per-step
        # and gated behind box_off_rest, so a horizon cut comes almost entirely
        # out of the post-grasp budget). Left at 250 rather than moved to 400
        # because reward_scaling=0.05 in manipulation_params.py is explicitly
        # tuned against the ep250 return scale -- changing this default silently
        # re-scopes that tuning and every other consumer, so it needs its own
        # validated decision. Until then: set episode_length=400 EXPLICITLY in
        # the sweep JSONL (see batch_runs/sweeps/gen_dr_ladder.py).
        episode_length=250,
        action_repeat=1,
        # Arm per-step ctrl delta = action * action_scale (swept per run). The
        # gripper is DECOUPLED via gripper_action_scale below so it can be kept
        # slow (stays open on approach) while the arm runs faster.
        action_scale=0.015,
        # Separate per-step scale for the gripper actuator (last ctrl dim). Small
        # + fixed so the hand can't snap shut in one step — full open->close
        # travel is 0.025, which needs >=2.5 steps at 0.01. This is what keeps
        # the hand open on approach independent of the swept arm action_scale.
        gripper_action_scale=0.02,
        # addvelocity (2026-07-22): append arm joint velocity (6D) + finger
        # velocity (1D) to the obs, AFTER the existing 26D layout (base_obs +
        # jaw_proj + app_proj + last_action) -> 33D total when enabled. STATIC
        # gate: default False reproduces the 26D obs byte-for-byte (see the
        # regression test in ur3_pick_test.py). Motivation: the policy only
        # ever sees POSITION state, so it cannot anticipate its own momentum --
        # hypothesis is this caps how fast+accurate the approach/grasp can be,
        # independent of action_scale. Velocity is read from data.qvel (sim,
        # exact) / RTDE getActualQd + a finger finite-difference (real, see
        # robots/UR3e/ur3_realrobot_dependencies.py's build_obs_from_feedback).
        # No noise is added to these channels in this first run (obs_noise
        # above is not extended to velocity yet) -- isolate the effect of
        # having velocity at all before adding robustness DR on top of it.
        obs_include_velocity=False,
        reward_config=config_dict.create(
            scales=config_dict.create(
                ## Staged reward scaling factors (sequenced by sticky latches).
                # Gripper (TCP) approaches the box (always on).
                gripper_box=4.0,
                # Keep the gripper OPEN while approaching, until the box is
                # reached (complement of `grasp`, which rewards closing after
                # reached). One-sided finger_open*(1-reached), V1 shape. Kept
                # SMALL (1.0, < grasp=3.0): the v2 two-sided @6.0 experiment
                # reward-hacked — W&B showed reached_box collapsing to ~0 while
                # the policy hovered open just outside the box to farm this bonus
                # (861/1197 of the return). The open bonus must never out-pay
                # actually reaching and grasping.
                approach_open=1.0,
                # Close the fingers on the box once the gripper has reached it.
                grasp=3.0,
                # Raise the box off its resting height — anti-push lever that
                # gates box_target behind a real lift.
                lift=5.0,
                # Box goes to the mocap target (lift point in the air); gated by
                # the sticky "lifted" latch so a sliding box earns nothing.
                # RAISED 8.0->20.0: with base_polar targets the box was picked/
                # lifted reliably but stalled ~13cm short and drifted back --
                # the "grab-and-hold" stack (gripper_box 4.0 + gripper_align 5.0
                # + lift 5.0 + grasp 3.0, ~3100 of return) dwarfed transport
                # (~390, ~11%), and none of it requires MOVING the box, so PPO
                # had no incentive to finish the carry. At 20.0 transport is ~25%
                # of return, on par with the largest hold terms and the dominant
                # POST-LIFT incentive. Safe to raise (no reward-hacking): gated
                # on the sticky `lifted` latch, so it only pays after a real
                # grasp+lift and cannot short-circuit the pick chain.
                box_target=20.0,
                # Grasp-frame alignment: the jaw axis AND the approach axis each
                # line up with a box face-normal (2 of 3 gripper axes -> the 3rd
                # is forced). Shapes the final approach so a parallel-jaw hand can
                # actually close on the (rotated + tilted) cube. RAISED 2.0->4.0:
                # DIAGHOLD300 (W&B) showed the pick chain unlocking at 1.5cm+2-pad
                # contact with no alignment requirement, so misaligned grabs still
                # earned the full sticky lift/box_target/hold chain (~51% of
                # return) while this term was only ~11% and approach-only ->
                # alignment got unlearned, worst on hard start poses. Now
                # intentionally the dominant approach-shaping term: safe to raise
                # because `grasped` (below) is now GATED on alignment
                # (grasp_align_thresh) so this can no longer be shortcut by a
                # lucky grab, and it stays proximity-gated so it can't be farmed
                # in free space (the reward-hacking lesson from earlier open
                # bonuses). RAISED 4.0->5.0 to give orientation control a touch
                # more weight than the approach term (gripper_box=4.0), now that
                # training exposes the full +/-2pi cube-yaw + wrist range.
                gripper_align=5.0,
                # Do not collide the gripper with the floor OR THE TABLE. Name
                # kept for sweep/JSONL compatibility; the sensor set behind it
                # covers both surfaces (see _floor_hand_found_sensor).
                no_floor_collision=0.25,
                # Arm stays close to initial pose. Gated by (1-lifted) in
                # _get_reward so it only regularizes the pre-lift approach —
                # once the box is lifted this stopped fighting the transport
                # stage, which needs the arm to move AWAY from its init pose
                # to reach the (raised) target. Was un-gated and always-on;
                # box_target_dist floored at ~30mm / success ~3% on the
                # 20260709 speedtest runs, partly because this term penalized
                # exactly the motion transport requires.
                #
                # 2026-08 "moving is net negative" finding. Despite the name
                # this is a BONUS (positive scale, raw signal maximal at zero
                # deviation), so it pays the arm to stay near where it started
                # for the entire pre-lift phase. Measured cost of travelling:
                # -0.163/tick SUSTAINED, against a gripper_box pull of only
                # +0.023/tick at d = 0.40 m. Together with action_rate below
                # (-0.175/tick at |dA| = 0.5) that makes standing still the
                # optimal action until d ~= 2 cm -- the reward-side half of the
                # "slow approach when far away" diagnostic.
                # RECOMMENDED v3 value: 0.05 (moves the break-even from ~2 cm
                # out to ~15 cm). Set it PER SWEEP LINE, not here: this default
                # is what evaluation/ur3_reward_replay.py reads at import time
                # to score ARCHIVED real-robot runs, so changing it would
                # retroactively rescore the D23 campaign with scales those
                # policies never trained under.
                robot_target_qpos=0.3,
                # Penalize large action deltas between consecutive steps
                # (DeXtreme-style smoothness term) to stop the visible
                # shaking/jitter in rollout videos that knocks the box loose.
                # Raw signal is a squared L2 norm (unbounded, unlike the other
                # tanh-saturated terms here) so keep this scale small relative
                # to the others; NEGATIVE scale turns the raw magnitude into a
                # penalty (same sign convention `franka_emika_panda_robotiq`
                # and `leap_hand` use for their action_rate terms).
                #
                # 2026-08: this is the larger half of the "moving is net
                # negative" problem (see robot_target_qpos above). Being an
                # UNBOUNDED squared L2 over 7 dims, a modest |dA| = 0.5 per dim
                # already costs 7 * 0.25 * 0.10 = -0.175/tick, roughly 7.6x the
                # +0.023/tick that closing distance pays at d = 0.40 m.
                # RECOMMENDED v3 value: -0.02. Same warning as above -- set it
                # per sweep line, not here, or the archived-campaign replay
                # gets rescored.
                action_rate=-0.10,
                # Sustained-proximity bonus: pay for KEEPING a lifted box inside
                # the target sphere, ramping with dwell time so the policy
                # settles and holds instead of tapping the point and drifting
                # off. Resets to 0 the instant the box leaves the radius (a drop
                # kills it), so it doubles as a drop penalty. Gated by `lifted`
                # and tanh-capped so it cannot out-pay box_target(20)/grasp(3)
                # -- the reward-hacking guard: it must never be farmable without
                # an actual lifted box held at the goal. Added because
                # box_target only rewards distance-to-point, nothing rewarded
                # HOLDING it there (FIXVERIFY: box visibly enters the 4cm target
                # sphere but drifts/drops with no recovery pressure). RAISED
                # 3.0->6.0 to fight the observed drift-back (box reached ~13cm
                # then retreated): reward dwelling inside hold_radius, still
                # < box_target so it cannot out-pay the transport gradient.
                hold_target=6.0,
                # One-shot bonus on the rising edge of the 5mm/3-step success
                # gate. DEFAULT 0.0 -- it is deliberately OFF and should stay
                # off under the current done= policy.
                #
                # It exists for the case where success TERMINATES the episode.
                # Under that policy, succeeding forfeited every remaining step
                # of box_target(20)+hold_target(6)+lift(5)+grasp(3) ~= 31
                # raw/step -- ~3.1k of discounted future reward (gamma=0.99 caps
                # the geometric sum at 31/0.01). That made hovering in the
                # 5mm..30mm ring strictly better than succeeding, and the
                # 2026-07-24 ladder learned exactly that: eval/episode_success
                # peaked 0.67-0.69 at 60-70% of training then collapsed below
                # 0.03 while episode reward climbed to its maximum. A value
                # around 5000 would clear that hover value with margin.
                #
                # Since 2026-07-26 episodes always run to episode_length (see
                # done= in step), so nothing is forfeited and no compensation is
                # needed -- box_target is strictly maximised at d=0, which is
                # inside success_tol, so the shaping already points at success.
                # The term is edge-triggered in step(), so it stays safe (one
                # payout per episode) if you ever re-enable termination.
                # Keep any non-zero value under the +-1e4 per-step reward clip.
                success_bonus=0.0,
            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="task_home",
        # Per-joint per-direction amplitude (rad) for reset randomization: 6 arm
        # + 1 finger. Applied symmetrically as uniform(-v, +v) on top of the init
        # keyframe. Default 0.05 reproduces the legacy uniform(-0.05, 0.05) arm noise.
        # D18 (2026-07-29): widened 0.05 -> 0.10 rad per arm joint (the wrist
        # stays at 6.28319 = full 2*pi travel, the finger at 0.0). See the vault
        # Plan - Sim-to-Real Gap Protocol, D18.
        init_qpos_noise=(0.10, 0.10, 0.10, 0.10, 0.10, 6.28319, 0.0),
        # Arm/finger start-pose source. "none" = literal keyframe start (then +
        # init_qpos_noise jitter); "light"/"mid"/"hard" = randomly pick one
        # hand-collected pose from init_poses/train/<level>.json each reset.
        init_start_random="mid",
        # Nominal TOP-SURFACE height (m) of the table the cube is picked from,
        # measured in the lab: 95 mm above the UR3e base origin. The table is a
        # permanent fixture -- there is no "off" any more (see the absolute
        # lifter_height_abs_min/max range below).
        lifter_height_nom=0.095,
        # Per-episode table TOP-SURFACE height (m), drawn ABSOLUTELY:
        #   top ~ uniform(lifter_height_abs_min, lifter_height_abs_max)
        # then clamped so the plate bottom stays off the floor plane
        # (_LIFTER_HEIGHT_MIN). The lift target is shifted by the DEVIATION of
        # that draw from lifter_height_nom (deviation = top - 0.095), because the
        # box anchor (_init_obj_pos[2] = 0.115) already carries the nominal.
        #
        # !! SEMANTICS CHANGED TWICE -- read this before reusing an old JSONL.
        #   pre-2026-07-28: `lifter_height_max` was the ABSOLUTE max of a
        #     uniform(0.003, v) draw; 0.0 meant "no plate, cube on the floor".
        #   2026-07-28:     `lifter_height_max` became a SYMMETRIC +- band about
        #     lifter_height_nom; 0.0 meant "flat table at the nominal 95 mm".
        #   2026-07-29 (D18/D24, HERE): the band is replaced by an explicit
        #     ABSOLUTE range and `lifter_height_max` is GONE. Default
        #     U(0.0, 0.220) spans bare floor to a 220 mm table.
        # The old key was deliberately REMOVED rather than silently reinterpreted
        # a third time: any sweep JSONL still passing lifter_height_max now fails
        # loud in learning/notebooks/run_experiment.py with a migration message,
        # instead of quietly training under a range it never asked for.
        # For a deterministic baseline (DR-ladder L0), set BOTH to 0.095 -- that
        # is "flat table at the nominal height", deviation 0.
        lifter_height_abs_min=0.0,
        lifter_height_abs_max=0.220,
        # Per-episode SLIGHT plate tilt (rad). roll AND pitch are sampled
        # INDEPENDENTLY, each ~ uniform(-t, +t) about world X and world Y, then
        # composed — so both axes tilt at once and the worst-case surface-normal
        # tilt is ~sqrt(2)*t when both hit their extremes. The box rests FLUSH on
        # the tilted plate, so it starts both at a variable height and slightly
        # tilted. This is what makes the out-of-plane (approach-axis) component of
        # the 2-of-3-axis grasp alignment matter. 0.0 => perfectly level table.
        # Pivots about the TABLE body origin (the fixed XML pos, x=0.40, y=0), so
        # the tilt lever arm is |box_xy - table_xy| (<=~0.32 m), NOT the plate's
        # full half-length. At 0.05 rad that is <=16 mm of height change across
        # the spawn envelope.
        # Reasonable values (per axis; remember the ~1.41x combined worst case):
        #   0.00  => flat (baked default here)
        #   0.05  => mild,     ~2.9 deg  (~4.0 deg combined)  -- gentle start
        #   0.08  => moderate, ~4.6 deg  (~6.5 deg combined)  -- prior default
        #   0.10  => strong,   ~5.7 deg  (~8.1 deg combined)  -- stickyoff "hard"
        # Keep < ~0.12 rad (~6.9 deg / ~9.7 deg combined) so the cube can't
        # slide/tip off the plate before the grasp.
        # D18 (2026-07-29): 0.08 -> 0.05 rad ("light tilt", Matthias).
        #
        # 2026-08: 0.05 -> 0.12 rad. The D23 protocol's C1 condition is a
        # physical wedge whose REALIZED tilt measured 5.47 deg on one axis,
        # i.e. the eval sat almost 2x OUTSIDE the trained 2.86 deg/axis band --
        # every ladder policy was tested on a tilt none of them had ever seen,
        # which made C1 uninformative about DR dose-response (it was OOD for
        # L0..L4 alike). 0.12 rad = 6.9 deg/axis covers 5.47 deg with ~26%
        # margin and is exactly the ceiling this comment already documents.
        #
        # MEASURED COST (200 resets x 60 settle steps, zero action) -- the cube
        # is NOT perfectly stable at this tilt, so budget for it:
        #     tilt   slide p95   tip p95   episodes disturbed
        #     0.00    0.00 mm     0.00 deg      1.0%
        #     0.05    0.32 mm     3.43 deg      0.5%
        #     0.12    2.90 mm     8.34 deg      3.0%
        # ("disturbed" = slid >2 cm or tipped >30 deg.) So ~3% of episodes now
        # start with the cube having shifted or toppled during the spawn
        # transient, up from ~0.5%. It is NOT the arm knocking it: 0% of the
        # disturbed episodes had the TCP within 10 cm of the cube at reset. Nor
        # is it a statics failure -- at 6.9 deg, tan(theta)=0.12 against a
        # friction coefficient of 1.0, and the 3x3x4 cm cube only tips past
        # atan(0.015/0.020) = 36.9 deg. It is the drop-and-settle transient,
        # which a steeper plate turns into a longer slide.
        # Judged acceptable: the cube settles to a valid pose on the plate, its
        # pose is fully observed, and axis_aware alignment reads the REALIZED
        # box axes, so a toppled cube is scored correctly rather than silently
        # mis-graded. Drop to 0.10 rad (5.7 deg/axis) if 3% proves too noisy --
        # that still covers C1's 5.47 deg, but with only ~4% margin.
        # NOTE the table HEIGHT range is deliberately left at its full
        # 0.0..0.220 m span (Matthias, 2026-08) even though the real lab table
        # is fixed at 0.095 m and the only other tested condition (D1) is
        # 0.150 m.
        lifter_tilt_max=0.12,  # ~6.9 deg/axis (~9.7 deg combined); covers D23 C1
        # Box spawn yaw about world Z (rad); yaw ~ uniform(-r, +r). pi/4 covers
        # all yaw thanks to the cube's 4-fold symmetry, so the policy must learn
        # to match the jaw axis to a face rather than getting a free alignment.
        # 2*pi, full-rotation yaw coverage. NOT a no-op and NOT reducible to
        # pi/2 via the cube's symmetry: the alignment REWARD is symmetry-
        # invariant (max|axis . box_axis|) but the OBSERVATION is not -- it
        # carries raw signed projections onto the box axes, so a 180 deg yaw
        # is a different input vector. Narrowing it broke the real policy on a
        # 180 deg-rotated cube. Full reasoning at the sampling site in
        # _reset_box_pose; decision D26 in the vault plan. DO NOT NARROW.
        box_z_rot_range=6.28319,
        # Y-axis center offset (m) for the box spawn / lift target, applied
        # BEFORE their existing jitter ranges (box jitter Y +-0.2, target
        # jitter Y +-0.03) — see reset(). Both currently jitter around the
        # SAME anchor (self._init_obj_pos), so nominal box->target lateral
        # distance is ~0. Setting these to opposite signs pushes the box spawn
        # to one side and the target to the other, adding real transport
        # distance on top of the jitter, to stress-test the box_target /
        # robot_target_qpos transport stage. 0.0 = legacy (both centered).
        box_y_center_offset=0.0,
        target_y_center_offset=0.0,
        # Box XY spawn jitter half-range (m), per axis: box_xy ~ uniform(-v, +v)
        # + the XML keyframe box XY (+ box_y_center_offset). Was a hardcoded
        # [0.15, 0.20] literal in reset(); exposed here with the SAME default
        # so a sweep can zero it for a deterministic baseline (the "position"
        # cluster of "Plan - Sim-to-Real Gap Protocol", DR-ladder L0) without
        # touching this file again. 0.0 => box always at the nominal XY.
        #
        # D24 (2026-07-29): widened (0.15, 0.20) -> (0.17, 0.24) about the new
        # nominal centre x=0.32 (scene XML), i.e. the RECTANGLE sampled is
        # x in [0.15, 0.49], y in [-0.24, 0.24]. That rectangle's far corners sit
        # at r = sqrt(0.49^2 + 0.24^2) = 0.546 m, ~11% past the UR3e's ~0.49 m
        # WORKING radius, and its near edge (0.15, 0) is inside the base column.
        # Neither is fixed by shrinking the rectangle -- reset() instead projects
        # every draw onto the reachable annulus
        #   r in [_BOX_XY_R_MIN, _UR3_WORKING_RADIUS_M] = [0.18, 0.49].
        # Net effect vs the old (0.15, 0.20) rectangle: strictly MORE reachable
        # table covered, with ZERO unreachable draws. The clip piles a little
        # probability mass onto the two annulus boundaries -- accepted and
        # documented, see the radial-clip block in reset(). Every post-clip draw
        # also lands on the real table (x in [0.10, 0.70], y in [-0.50, 0.50]).
        box_xy_jitter=(0.17, 0.24),
        # Lift-target Z-band jitter (m): target_z ~ uniform(*target_z_jitter) +
        # _init_obj_pos[2] (+ the table's deviation from its nominal height).
        # NOTE the anchor _init_obj_pos[2] is now the cube resting ON the nominal
        # table (0.115), not on the floor, so this band is measured from the
        # table top -- an 0.18‥0.21 draw is a 18‥21 cm lift off the table, the
        # same lift it always was. Was a hardcoded [0.18, 0.21] literal,
        # shared verbatim by BOTH target_mode paths ("box" and "base_polar");
        # exposed here for the same reason as box_xy_jitter. A zero-width
        # tuple (e.g. (0.195, 0.195)) makes the goal height deterministic.
        #
        # D24 (2026-07-29): LOWER bound 0.18 -> 0.05 so the D22 eval drop point
        # sits INSIDE the trained distribution. This band is the LIFT ABOVE THE
        # CUBE'S RESTING HEIGHT, so:
        #   drop height 70 mm above the table top
        #     -> cube centre world z = 0.095 + 0.070 = 0.165
        #     -> draw                = 0.165 - 0.115  = 0.05
        #     -> air gap under the 40 mm cube = 0.165 - 0.02 - 0.095 = 0.050 m
        # (70 mm centre height / 50 mm air gap are the same spec stated twice.)
        #
        # 2026-07-29, follow-up (Matthias): the D24 band (0.05, 0.21) put the
        # eval height at its extreme LOW edge -- training spent most of its
        # mass on lifts up to 21 cm that eval never actually tests. RENARROWED
        # to (0.02, 0.08), CENTERED exactly on the eval draw (0.05 +- 0.03),
        # so eval now sees a typical training sample instead of an edge case,
        # while still keeping real height variation for the "position" DR
        # cluster to train against. Side effect (welcome): this also lowers
        # the D25 target_z_capped rate (was 15.7% at the old 0.21 ceiling,
        # against the _TARGET_WORLD_Z_CAP=0.345 reach limit) since the new max
        # (0.08) sits far below the cap regardless of table height.
        # WARNING: this changes the task for EVERY ladder rung, not just the
        # top one -- returns are NOT comparable to W&B data trained under
        # either the original (0.18, 0.21) or the D24 (0.05, 0.21) band.
        target_z_jitter=(0.02, 0.08),
        # Per-episode gripper start position. True (legacy) = sample uniform
        # [0, 0.025] m (anywhere open<->closed) at reset. False = always start
        # FULLY OPEN (0.0) -- for a deterministic DR-ladder L0 baseline.
        finger_random_init=True,
        # --- Lift-target sampling scheme ---------------------------------------
        # "box"        = legacy axis-aligned sampling around the box anchor
        #                (self._init_obj_pos + the fixed X/Y/Z jitter bands +
        #                box_y_center_offset/target_y_center_offset). Unchanged.
        # "base_polar" = sample the target XY in an ANNULUS around the ROBOT BASE
        #                (radius target_r_min‥target_r_max), at an azimuth offset
        #                target_azim_min‥target_azim_max to either side of the
        #                box's bearing from the base. This supersedes the
        #                box_y_center_offset/target_y_center_offset transport
        #                hack (those still work under "box"); the Z band is
        #                unchanged (fixed 0.18‥0.21 above the box anchor).
        target_mode="base_polar",
        # Horizontal radius from the ROBOT BASE for base_polar targets (m).
        target_r_min=0.25,
        # Keep the far target reachable WHILE GRASPING a box. At 0.48 the worst
        # target was 0.52-0.54 m from base (at/over the ~0.54 m reach BEFORE the
        # grasp consumes any), so far targets were unreachable and the box
        # plateaued. 0.42 restored ~6-7 cm of headroom -- but the 95 mm TABLE
        # then raised the box anchor (and with it every target) by that much:
        #   target_z = 0.115 (box on the table) + U(0.18, 0.21) +- the table's
        #              height deviation (0.02) = 0.275..0.345
        #   r=0.42 -> sqrt(0.42^2 + 0.345^2) ~= 0.54 m  (no headroom left)
        #   r=0.35 -> sqrt(0.35^2 + 0.345^2) ~= 0.49 m  (~5 cm, as intended)
        # Hence 0.42 -> 0.35. Still a real carry (chord up to ~0.35 m at the
        # 60 deg azimuth cap); drop target_z_jitter before r if success stalls.
        target_r_max=0.35,
        # Azimuth offset band (rad) from the box's bearing (as seen from the
        # base) for base_polar targets: 30°‥60° to either side, so the target
        # stays in front of the robot (a 90°‥180° band threw it across the base,
        # out of reach within the episode).
        target_azim_min=0.5236,  # pi/6 (30 deg)
        target_azim_max=1.0472,  # pi/3 (60 deg)
        # Box-center distance (m) to the lift target counted as success (3
        # consecutive steps). Tight 3 mm — the box must end up inside the target.
        success_tol=0.005,
        # "Off the resting height" margin (m) a grasped box must clear to set the
        # sticky "lifted" latch that unlocks box_target (anti-push lever).
        lift_eps=0.03,
        # hold_target (reward_config.scales, above) params. hold_radius: box-
        # center distance (m) to the target counted as "in hold" -- inside the
        # 4cm-radius visual mocap_target sphere (xmls/..._ur3.xml), which is
        # decorative for physics but is what "at the target" looks like in the
        # rollout videos, unlike success_tol's much tighter 5mm point gate.
        # hold_tau: dwell-time constant in STEPS for the tanh ramp
        # (tanh(counter/tau) ~=0.76 @10 steps, ~=0.96 @20 steps).
        hold_radius=0.03,
        hold_tau=10.0,
        # Minimum grasp-frame alignment required for the `grasped` sticky latch
        # to set -- blocks the "grab while misaligned" shortcut that let a lucky
        # 2-pad contact on a rotated/tilted cube unlock the whole sticky
        # lift/box_target/hold chain regardless of alignment (DIAGHOLD300 W&B
        # finding). Soft: gripper_align (reward_config.scales, above) still pays
        # a continuous gradient below this bar, so there is no dead zone -- only
        # the LATCH is hard-gated.
        #
        # This latch is the single highest-leverage gate in the reward: it
        # unlocks lift(5) + box_target(20) + hold_target(6) = 31 raw/step for
        # the REST of the episode.
        #
        # UNITS DEPEND ON align_mode -- never carry a value across modes:
        #   axis_free : 0.30 == sqrt(0.30)=0.548 per axis -> a_jaw=a_app=0.774
        #               -> 39.3 deg of misalignment allowed on EACH axis. Far
        #               too loose: measured D23 grasp-stage sim->real retention
        #               was 0.06-0.21, i.e. the grasp is the stage that does not
        #               transfer, and this is why. Sim contact tolerates a 39 deg
        #               grab; the Hand-E does not.
        #   axis_aware: the same 39 deg pose now scores 0.091, because the two
        #               physical preference factors multiply in. Measured
        #               calibration ladder (jaw misaligned in-plane by the same
        #               angle as the approach, nominal 3x3x4 cm box):
        #                 10 deg -> 0.730   15 deg -> 0.582
        #                 20 deg -> 0.443   25 deg -> 0.321
        #                 30 deg -> 0.220   39 deg -> 0.091
        #               0.45 ~= 20 deg on both axes WITH a good jaw span.
        #               0.35 ~= 24 deg (the safer first try).
        #               For reference, the two grasps the axis_free score cannot
        #               tell apart: a top-down grasp on a 3 cm face scores 1.000
        #               under BOTH modes, while a side grasp spanning the 4 cm
        #               axis scores 1.000 under axis_free and 0.086 under
        #               axis_aware. Reaching upward scores 1.000 vs 0.150.
        #
        # Every sweep line MUST set align_mode and grasp_align_thresh EXPLICITLY
        # and TOGETHER, so the pair lands in env_overrides -> metadata.json.
        # Splitting them across a default and an override silently reinterprets
        # the number. Watch eval/episode_grasp_gate_blocked: if it is large
        # while eval/episode_grasped stays ~0, the bar is too high and the
        # lift/box_target/hold chain will never bootstrap -- lower it.
        grasp_align_thresh=0.3,
        # ------------------------------------------------------------------
        # Grasp-frame alignment mode (STATIC; resolved at trace time in __init__)
        # ------------------------------------------------------------------
        # "axis_free"  = legacy: jaw_score * app_score with max|axis . box_axis|
        #                over ALL THREE box axes. Symmetry-invariant, and that
        #                is exactly the defect. The box is a 3x3x4 cm PRISM
        #                (geom size 0.015 0.015 0.020), not a cube, and the jaw
        #                opening is 49.9 mm. So the legacy score rates a jaw
        #                spanning the 4 cm axis (9.9 mm total clearance)
        #                IDENTICALLY to one spanning a 3 cm axis (19.9 mm), and
        #                rates a horizontal approach identically to top-down.
        #                The in-code claim that max|.| buys invariance to the
        #                "cube's 24-fold octahedral symmetry" does not hold: a
        #                3x3x4 prism has only D4h (16-element) symmetry -- the z
        #                axis is NOT interchangeable with x/y.
        # "axis_aware" = the legacy face score TIMES two physically-grounded
        #                preference factors: (a) jaw span measured in real
        #                finger clearance, (b) top-down approach. Both are
        #                floored by align_pref_floor so there is never a dead
        #                zone, and both switch OFF once the box is lifted so
        #                they shape approach/grasp only and never fight
        #                transport. DR-SAFE by construction: it consumes the
        #                REALIZED per-episode half-extents (info["box_half"]),
        #                so a cube_size draw anywhere in 2x2x3 .. 4x4x4 cm is
        #                handled without a hardcoded "local z is the long axis"
        #                assumption -- which is WRONG for ~25% of draws
        #                (P(xy_half > z_half) = 0.25 under
        #                U(0.010,0.020) x U(0.015,0.020)).
        align_mode="axis_free",
        # Floor for each axis-aware preference factor: pref -> f + (1-f)*pref,
        # so the worst grasp still scores f (not 0) and gripper_align keeps a
        # gradient everywhere. Zero here would recreate exactly the
        # chicken-and-egg dead zone the old hard 30 deg cone had (see the
        # _COS_BOUND note in _get_reward): the policy would have to luck into a
        # good pose before the reward ever turned on.
        align_pref_floor=0.15,
        # When True (default, legacy behavior): grasped/lifted are STICKY
        # (jp.maximum) -- once true for an episode, stay true even if the box
        # is dropped, so grasp/lift/box_target/hold_target keep paying after a
        # drop (masks drops in metrics: e.g. lifted=106/200 steps could mean
        # "held the whole time" or "touched height once, dropped, sat on the
        # floor for the rest"). When False, grasp/lift/box_target/hold_target
        # instead gate on LIVE contact/height each step, so a drop stops all
        # downstream reward until the box is actually re-grasped. `reached`
        # stays sticky either way (legit "found the box once" signal, not a
        # holding signal). Default True to keep the main sweep unconfounded;
        # flip per-run to test re-grasp pressure. CAUTION if enabling: watch
        # for stage-farming, the reward-hacking pattern from d24be6b/8ccfc67.
        sticky_latches=True,
        # ------------------------------------------------------------------
        # Physics domain randomization (per-EPISODE, sampled fresh in reset()).
        # ------------------------------------------------------------------
        # Robustness DR: perturb the sim physics so the policy generalizes to
        # the sim-to-real gap. Sampled once per reset(), stashed in info, and
        # applied to a per-episode model via tree_replace at the top of step()
        # (the policy is NOT told the sampled values -- no obs change). This is
        # the per-episode, in-env path -- NOT the brax per-env randomization_fn
        # wrapper (do not enable both). Every axis is INDEPENDENTLY gated for
        # one-at-a-time ablation (like sticky_latches / target_mode), and all
        # default OFF/identity so training is byte-for-byte unchanged when the
        # master `enable` is False (the enable flags are STATIC, so the off path
        # is a trace-time branch that skips the extra rng split entirely).
        # Sweep-overridable via dotted keys, e.g. "domain_rand.cube_mass.enable".
        domain_rand=config_dict.create(
            # Master switch. False -> reset()/step() take the identity path.
            enable=False,
            # Cube mass: x nominal body_mass[box] (and body_inertia, kept
            # consistent). Box is ~0.036 kg (auto from volume); [0.7,1.3]x.
            cube_mass=config_dict.create(enable=False, min=0.7, max=1.3),
            # Cube sliding friction: x nominal geom_friction[box, 0]. [0.5,1.5]x.
            cube_friction=config_dict.create(enable=False, min=0.5, max=1.5),
            # D19 (2026-07-29): cube size DECOUPLED from a single scale factor
            # into two INDEPENDENT absolute half-extent draws (metres):
            #   cube_size_xy ~ U(0.010, 0.020) -> 2 cm .. 4 cm box WIDTH
            #   cube_size_z  ~ U(0.015, 0.020) -> 3 cm .. 4 cm box HEIGHT
            # i.e. the box spans 2x2x3 cm to 4x4x4 cm. cube_size_xy is hard-
            # clamped in _randomize_physics to _dr_max_box_half_xy (graspability;
            # the fingers open ~5 cm) -- that clamp was raised 0.018 -> 0.020 so
            # the 4 cm draw is admitted instead of silently clipped. The sampled
            # Z half-extent also RE-SEATS the box on the table in reset(), so a
            # short cube starts lower and a tall one higher, both flush.
            # Mass/friction DR above are UNCHANGED by this split (they still
            # scale the nominal, they do not track the drawn size).
            cube_size_xy=config_dict.create(enable=False, min=0.010, max=0.020),
            cube_size_z=config_dict.create(enable=False, min=0.015, max=0.020),
            # Gravity: opt.gravity magnitude += U(-g_delta, g_delta) m/s^2, plus a
            # small random directional tilt (rad) off vertical. tilt=0 -> pure
            # magnitude noise. Nominal g = 9.81.
            gravity=config_dict.create(enable=False, g_delta=0.5, tilt=0.0),
            # Arm "stiffness": x arm position-actuator kp (actuator_gainprm[:6,0],
            # with biasprm[:6,1]=-kp mirrored). CAVEAT: raw arm jnt_stiffness is 0
            # in the model (compliance lives in the servos), so this scales the
            # effective servo stiffness the arm feels, not a passive joint spring.
            arm_stiffness=config_dict.create(enable=False, min=0.75, max=1.25),
            # Arm "damping": x arm position-actuator kv (actuator_biasprm[:6,2]).
            # Same caveat -- raw arm dof_damping is 0; this scales the servo's
            # velocity gain (the effective joint damping).
            arm_damping=config_dict.create(enable=False, min=0.75, max=1.25),
            # ---- deferred DeXtreme-style axes (scaffolded, OFF, code stubbed) ----
            # Contact restitution / softness: solref/solimp on box + finger geoms.
            restitution=config_dict.create(enable=False),
            # Arm joint-range jitter: jnt_range on the 6 arm joints.
            joint_limits=config_dict.create(enable=False),
            # Per-STEP random perturbation force (xfrc_applied) on the box body,
            # applied at the box's center of mass -- see step(). Magnitude is
            # UNCALIBRATED (no measured real perturbation to center on; see
            # "Plan - Sim-to-Real Gap Protocol" C4/blind-randomization note).
            #
            # D20 (2026-07-29) RESPEC: was 0.5 N RESAMPLED EVERY STEP -- ~1.4x
            # the box's own weight (~0.036 kg * 9.81 ~= 0.35 N) as continuous
            # white noise at 50 Hz, diagnosed as the likely cause of the old
            # L4/L5 training collapse (D14). Now 0.15 N in BURSTS: each step has
            # probability `force_prob` of TRIGGERING a burst if none is active;
            # once triggered the SAME force vector is HELD for `burst_steps`
            # steps, then drops back to zero. Re-triggering mid-burst is a no-op
            # (hold the vector, don't restart the clock). State machine lives in
            # `info` (jit/vmap-safe), mirroring the action_delay ring buffer.
            # Expected duty cycle: a burst starts on ~2% of idle steps and lasts
            # 10 steps (0.2 s at ctrl_dt=0.02).
            cube_force=config_dict.create(
                enable=False, force_mag=0.15, force_prob=0.02, burst_steps=10
            ),
            # D20 (2026-07-29) NEW axis: arm joint-torque perturbation. Same
            # burst trigger/hold schedule as cube_force above (independent
            # counter + held vector), applied as an additive qfrc_applied nudge
            # on the 6 arm DOFs -- see step(). "Very moderate jitter on all
            # joints" (Matthias): +-0.3 Nm/joint. Same "env" DR cluster as
            # cube_force in the ladder (see batch_runs/sweeps/gen_dr_ladder.py).
            joint_torque=config_dict.create(
                enable=False, torque_mag=0.3, force_prob=0.02, burst_steps=10
            ),
            # Action latency: delay the commanded action via an info ring
            # buffer. max_delay_steps=5 at ctrl_dt=0.02 => up to 100 ms of
            # latency, sampled ONCE per episode (held constant within it) --
            # see reset()/step(). UNCALIBRATED (no measured real action
            # latency to center on); purely exploratory, per the same blind-
            # randomization note as cube_force above.
            action_delay=config_dict.create(enable=False, max_delay_steps=5),
            # Observation noise: per-step JITTER (resampled every step) + a
            # per-episode BIAS (sampled once at reset, held constant) on every
            # *measured* obs quantity -- arm q, finger, box_pos, box
            # orientation. Modeled as two components because the real error
            # is NOT zero-mean: Nokov mocap jitter is sub-mm and negligible,
            # but the mocap-centroid / base-frame CALIBRATION offset is
            # systematic and constant within a run (see [[Base Frame
            # Calibration]], the vault's "Robustness to Initial Position"
            # note). The bias is the component that matters -- precedent:
            # Peng et al. ICRA 2018 (arXiv:1710.06537), DR robust to
            # calibration error. target_pos gets NO noise: it is COMMANDED
            # (chosen by us), not measured, so noising it would model
            # nothing. Injected at the SOURCE in _get_obs (never on the
            # finished obs vector -- the 6 alignment projections are derived
            # from box orientation + FK, so independent noise there would
            # make the obs physically self-contradictory). The REWARD always
            # uses the TRUE (unnoised) state -- see _get_reward, which reads
            # `data`/`info` directly and never consumes the obs vector.
            #
            # box_pos_bias has a HARD GEOMETRIC CEILING: the Hand-E jaw opens
            # ~5 cm, the box cross-section is 3 cm -> ~1 cm total clearance,
            # ~5 mm per side (see UR3Pick Environment's grasp-alignment
            # notes). Past that the policy cannot localize the box to within
            # grasping tolerance and has NO tactile channel to recover -- the
            # task becomes unlearnable, not merely harder. 0.005 (5 mm) is
            # that ceiling; do not raise it without re-deriving the margin.
            obs_noise=config_dict.create(
                enable=False,
                q_jitter=0.001,      # rad/step; arm encoders are excellent,
                q_bias=0.002,        # rad/episode; deliberately small -- not
                                     # a real error source, covers FK/URDF
                                     # mismatch only.
                finger_jitter=0.0005,  # m/step; readback is percent-
                finger_bias=0.001,     # m/episode; quantized (0.25 mm/count)
                                       # + <=10 Hz staleness carry-forward.
                box_pos_jitter=0.002,  # m/step; deliberately conservative,
                box_pos_bias=0.005,    # m/episode -- THE CRITICAL ONE, see
                                       # the geometric-ceiling note above.
                box_quat_jitter=0.0175,  # rad/step (~1 deg); marker-rig
                box_quat_bias=0.0524,    # rad/episode (~3 deg) -- mocap
                                         # measures the RIG, not the box.
            ),
        ),
    )


class UR3Pick(ur3_base.UR3Base):
    """Lift a box to a target point with the UR3 + Hand-E."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "my_ur3"
            / "xmls"
            / "mjx_single_cube_position_ur3.xml"
        )

        super().__init__(xml_path, config, config_overrides)

        init_keyframe = getattr(self._config, "init_keyframe", "low_home")
        self._post_init(obj_name="box", keyframe=init_keyframe)

        # Decoupled action scaling. The arm actuators use config.action_scale
        # (the swept value); the gripper (the LAST actuator) uses its own
        # gripper_action_scale so it stays slow/open regardless of arm speed.
        # This overrides the scalar self._action_scale set in ur3_base.__init__.
        # action_size == nu == 7 (6 arm + 1 gripper).
        n_act = int(self._mjx_model.nu)
        self._nu = n_act  # action/ctrl dim (6 arm + 1 gripper); used for last_action init
        arm_scale = float(self._config.action_scale)
        grip_scale = float(
            getattr(self._config, "gripper_action_scale", arm_scale)
        )
        action_scale_vec = [arm_scale] * n_act
        action_scale_vec[-1] = grip_scale  # gripper actuator is last
        self._action_scale = jp.asarray(action_scale_vec, dtype=float)

        # Floor-collision sensors (Hand-E fingers + hand capsule vs floor AND vs
        # the table). Both surfaces feed the SAME no_floor_collision penalty: with
        # the table at 95 mm the gripper never reaches the floor plane any more, so
        # the floor-only set would be a dead term and nothing would discourage
        # driving the hand into the table top.
        self._floor_hand_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_pad_floor_found",
                "right_finger_pad_floor_found",
                "hand_capsule_floor_found",
                "left_finger_pad_lifter_found",
                "right_finger_pad_lifter_found",
                "hand_capsule_lifter_found",
            ]
        ]

        # Finger-pad <-> box contact sensors (grasp detection); same data="found"
        # read pattern as the floor sensors above.
        self._finger_box_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_box_contact",
                "right_finger_box_contact",
            ]
        ]

        # Margin (m) a grasped box must clear above its per-episode resting height
        # to count as "lifted" (anti-push latch; rest height stored in reset()).
        self._lift_eps = float(self._config.lift_eps)

        # Init-pose library. Loaded once (stdlib+numpy I/O) and stored as a jnp
        # constant so the jitted reset() can index it; "none" keeps the legacy
        # keyframe start. Python-level branch -> resolved at trace time.
        level = getattr(self._config, "init_start_random", "none")
        if level != "none":
            self._init_pose_lib = jp.asarray(load_init_poses(level, "train"))
            self._n_init_poses = int(self._init_pose_lib.shape[0])
        else:
            self._init_pose_lib = None
            self._n_init_poses = 0

        # The table ("lifter") plate. Resolved here (not in the shared
        # ur3_base._post_init) because the picknplace sibling loads a scene
        # without a lifter body. ALWAYS ON: the table is a physical fixture of the
        # lab cell, not an optional riser, so there is no config gate any more
        # (lifter_height_abs_min == lifter_height_abs_max == lifter_height_nom
        # now means "level table at the nominal height").
        # The flag is kept so the reset()/reset_to_state() branches keep their
        # shape and a scene without the body stays one edit away.
        self._lifter_enabled = True
        self._lifter_mocap = self._mj_model.body("lifter").mocapid
        # D24 (2026-07-29): the table's FIXED world XY, read from the scene XML.
        # This is the pose reset() pins the mocap plate at every episode. It used
        # to be pinned at self._init_obj_pos[:2] (the nominal BOX xy) instead --
        # correct only back when the "lifter" was a small movable riser that had
        # to sit under the cube, and only invisible because the box anchor and
        # the plate happened to share x=0.40. The lifter is now a 0.6 x 1.0 m
        # fixed table AND the box centre moved to x=0.32, so pinning it to the
        # box anchor would drag the whole table 8 cm toward the base every reset
        # and silently contradict both the measured table extent and the
        # _post_init geometry guards below. The table does not move; only its
        # height and tilt are randomized.
        self._lifter_xy = jp.asarray(
            self._mj_model.body("lifter").pos[:2], dtype=float
        )
        # Nominal table TOP surface (m) and the matching mocap-body z (the body
        # origin sits half a plate-thickness below its top face).
        self._lifter_top_nom = float(self._config.lifter_height_nom)
        self._lifter_z_nom = self._lifter_top_nom - _LIFTER_HALF_THICKNESS
        # The XML parks the body at that same nominal, and evaluation/gap_target.py
        # READS the XML value to raise the deployment lift target. If the two ever
        # disagree, the real and sim targets silently differ by that gap -- fail
        # loud instead (same rule as the deploy action scales, see CLAUDE.md).
        # Tolerance is 1 um, not 0: _init_obj_pos below is a float32 jp.array
        # (ur3_base.py), so a keyframe that is EXACTLY right still round-trips
        # with ~2e-9 m of error -- a 1e-9 tolerance rejected the correct scene.
        # 1 um is far below anything physically meaningful here and still
        # catches every real desync (the failure this guards is millimetres).
        _xml_lifter_z = float(self._mj_model.body("lifter").pos[2])
        if abs(_xml_lifter_z - self._lifter_z_nom) > _ANCHOR_TOL:
            raise ValueError(
                f"lifter nominal height mismatch: scene XML body 'lifter' is "
                f"parked at z={_xml_lifter_z:.6f} but lifter_height_nom="
                f"{self._lifter_top_nom:.6f} implies z={self._lifter_z_nom:.6f} "
                f"(top surface minus the {_LIFTER_HALF_THICKNESS} m half-"
                f"thickness). evaluation/gap_target.py reads the XML value, so "
                f"this would desync the real-robot lift target from training."
            )
        # ...and the box anchor (task_home keyframe box Z, == _init_obj_pos[2])
        # must be the cube RESTING ON that nominal table. The anchor carries the
        # table height, so reset() adds only the per-episode DEVIATION from the
        # nominal to the lift target; if the anchor drifted off the table the
        # target would be raised by the wrong amount (or the cube would spawn
        # inside/above the table on any non-reset() path). Fail loud.
        _anchor_want = self._lifter_top_nom + _BOX_HALF_EXTENT
        if abs(float(self._init_obj_pos[2]) - _anchor_want) > _ANCHOR_TOL:
            raise ValueError(
                f"box anchor mismatch: keyframe '{self._config.init_keyframe}' "
                f"puts the box at z={float(self._init_obj_pos[2]):.6f}, but the "
                f"cube resting on the nominal {self._lifter_top_nom:.4f} m table "
                f"is z={_anchor_want:.6f} (table top + the {_BOX_HALF_EXTENT} m "
                f"box half-height). Update the scene XML keyframes."
            )

        # Robot base world XY, SOURCED FROM THE MODEL (not assumed to be the
        # world origin). The UR3 "base" body is a direct child of worldbody, so
        # its body.pos IS its world position at qpos0. Used by the "base_polar"
        # target_mode to sample lift targets in an annulus around the base; stays
        # correct if the base is ever repositioned in the scene XML.
        self._robot_base_xy = jp.asarray(
            self._mj_model.body("base").pos[:2], dtype=float
        )

        # D18/D24 reach-envelope check. CONSTRUCTION time only -- it reads the
        # STATIC config values, never traced arrays, so it never enters the jit
        # graph. LOUD WARNING, not a hard error: we cannot verify the true reach
        # envelope locally (no robot, no MJX/IK here), so the risk is made
        # visible up front instead of being discovered after a wasted HPC run.
        # reset()'s hard clip to _TARGET_WORLD_Z_CAP is the actual safeguard;
        # this only tells you whether the CONFIG could ever ask for more.
        # Do not silently ignore this if it fires.
        if self._config.target_mode == "base_polar" and self._lifter_enabled:
            worst_case_z = min(
                float(self._config.lifter_height_abs_max)
                + float(self._config.target_z_jitter[1])
                + _BOX_HALF_EXTENT,
                _TARGET_WORLD_Z_CAP,
            )
            worst_case_reach = (
                float(self._config.target_r_max) ** 2 + worst_case_z**2
            ) ** 0.5
            # 5 mm slack: the BAKED default is marginally (1.4 mm) over by
            # construction -- _TARGET_WORLD_Z_CAP is 0.345, i.e. 2 mm above the
            # 0.34293 that sqrt(0.49^2 - 0.35^2) allows, because 0.345 is also
            # the previously validated max target z (see the constant's comment).
            # Warning on that every construction would be pure noise; anything
            # genuinely over-reaching clears 5 mm easily.
            if worst_case_reach > _UR3_WORKING_RADIUS_M + 5e-3:
                warnings.warn(
                    "D18/D24 REACH RISK: worst-case lift target (target_r_max="
                    f"{float(self._config.target_r_max):.3f} m, world-z="
                    f"{worst_case_z:.3f} m, already clipped at "
                    f"_TARGET_WORLD_Z_CAP={_TARGET_WORLD_Z_CAP:.3f}) implies an "
                    f"approx. 3D reach of {worst_case_reach:.3f} m, exceeding the "
                    f"~{_UR3_WORKING_RADIUS_M:.2f} m UR3e WORKING radius "
                    f"(datasheet reach {_UR3_APPROX_MAX_REACH_M:.2f} m with "
                    "nothing held). Lower target_r_max or _TARGET_WORLD_Z_CAP "
                    "before training on this config -- see D18/D24 in the vault "
                    "Plan - Sim-to-Real Gap Protocol.",
                    RuntimeWarning,
                )

        # ------------------------------------------------------------------
        # Physics domain-randomization setup (see default_config().domain_rand).
        # ------------------------------------------------------------------
        # The master switch is STATIC (read once here), so reset()/step() branch
        # on it at TRACE time and the OFF path is byte-for-byte identical to the
        # pre-DR env (no extra rng split, model untouched). Per-axis enable flags
        # (also static) are read from self._config.domain_rand.* in the sampler.
        self._dr_enable = bool(self._config.domain_rand.enable)
        # Nominal physics captured ONCE so each axis scales from the baseline and
        # never compounds across episodes. Sourced from the numpy mj_model.
        self._dr_nom_friction = jp.asarray(
            self._mj_model.geom_friction[self._obj_geom], dtype=float
        )
        self._dr_nom_size = jp.asarray(
            self._mj_model.geom_size[self._obj_geom], dtype=float
        )
        self._dr_nom_mass = float(self._mj_model.body_mass[self._obj_body])
        self._dr_nom_inertia = jp.asarray(
            self._mj_model.body_inertia[self._obj_body], dtype=float
        )
        self._dr_nom_gravity = jp.asarray(
            self._mj_model.opt.gravity, dtype=float
        )
        # Arm position-actuator gains (first 6 actuators; the gripper is #7 and is
        # left untouched). kp = actuator_gainprm[:, 0]; the affine velocity gain
        # kv = actuator_biasprm[:, 2] (biasprm[:, 1] = -kp mirrors the setpoint).
        self._dr_nom_kp = jp.asarray(
            self._mj_model.actuator_gainprm[:6, 0], dtype=float
        )
        self._dr_nom_kv = jp.asarray(
            self._mj_model.actuator_biasprm[:6, 2], dtype=float
        )
        # Box Z half-extent (rest offset), == _BOX_HALF_EXTENT. Identity value for
        # cube_half_z so the size-off path re-seats the box unchanged.
        self._dr_nom_half_z = _BOX_HALF_EXTENT
        # Hard graspability clamp on the box WIDTH half-extent (xy) regardless of
        # the configured cube_size_xy range: the fingers open ~5 cm, so keep the
        # width half-extent <= 0.020 (4.0 cm box) -- ~5 mm clearance per side.
        # D19 (2026-07-29): raised 0.018 -> 0.020. The 0.018 belonged to the old
        # +-10% scalar scheme and would have silently clipped the top end of the
        # new cube_size_xy range (max 0.020). Height (z) stays unclamped.
        self._dr_max_box_half_xy = 0.020
        # Identity DR factors for the OFF path (same keys as _sample_physics_dr).
        self._dr_identity = {
            "mass_scale": jp.array(1.0, dtype=float),
            "fric_scale": jp.array(1.0, dtype=float),
            # D19: independent xy/z half-extent draws (absolute metres), not one
            # scale factor. Identity == the nominal box half-extents.
            "half_xy": jp.array(_BOX_HALF_EXTENT_XY, dtype=float),
            "grav": self._dr_nom_gravity,
            "kp_scale": jp.array(1.0, dtype=float),
            "kv_scale": jp.array(1.0, dtype=float),
            "cube_half_z": jp.array(self._dr_nom_half_z, dtype=float),
        }

        # ------------------------------------------------------------------
        # Grasp-frame alignment mode (see default_config().align_mode).
        # ------------------------------------------------------------------
        # STATIC, like sticky_latches / target_mode / the DR enables: the branch
        # in _get_reward is resolved at TRACE time, so the axis_free path is
        # byte-for-byte the pre-change env. Validate loudly here rather than
        # silently falling through to legacy on a typo -- a misspelled
        # align_mode in a sweep line would otherwise train the wrong reward and
        # only show up as an unexplained flat result 12 h later.
        _align_mode = str(self._config.align_mode)
        if _align_mode not in ("axis_free", "axis_aware"):
            raise ValueError(
                f"align_mode must be 'axis_free' or 'axis_aware', got "
                f"{_align_mode!r}. See ur3_pick.default_config().align_mode; "
                f"note grasp_align_thresh's UNITS depend on this value, so the "
                f"two must always be set together in a sweep line."
            )
        self._align_axis_aware = _align_mode == "axis_aware"
        # Pad-to-pad jaw opening (m) at full extension, PARSED from the finger
        # collision geoms so it can never drift from the simulated gripper the
        # way a hardcoded constant would. Same derivation as
        # evaluation/report/build_report.py's _gripper_geometry: the inner face
        # of each pad is its centre offset +- its half-extent along the travel
        # axis, and the finger slide joints are at range-min (fully open) in the
        # model's rest pose that geom_pos describes.
        #   (0.0305 - 0.0056) - (-0.0305 + 0.0055) = 0.0499 m
        _lf_g = self._mj_model.geom("left_finger_collision")
        _rf_g = self._mj_model.geom("right_finger_collision")
        self._jaw_opening = float(
            (_rf_g.pos[0] - _rf_g.size[0]) - (_lf_g.pos[0] + _lf_g.size[0])
        )
        if not (0.02 < self._jaw_opening < 0.10):
            raise ValueError(
                f"parsed jaw opening {self._jaw_opening:.4f} m is implausible "
                f"for the Hand-E (expected ~0.0499). The finger pad geoms in "
                f"universal_robots_ur3e/ur3e_position.xml changed shape or "
                f"axis; fix the parse above before training."
            )

    def _sample_physics_dr(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Sample per-episode physics-randomization factors.

        Each axis is gated on its STATIC `domain_rand.<axis>.enable` flag, so a
        disabled axis stays at its identity value AND consumes no randomness
        (the Python `if` is resolved at trace time). Returns the same keys as
        `self._dr_identity`; applied to the model in `_randomize_physics`.
        """
        dr = self._config.domain_rand
        out = dict(self._dr_identity)  # identity; overwrite only enabled axes

        if dr.cube_mass.enable:
            rng, k = jax.random.split(rng)
            out["mass_scale"] = jax.random.uniform(
                k, (), minval=float(dr.cube_mass.min), maxval=float(dr.cube_mass.max)
            )
        if dr.cube_friction.enable:
            rng, k = jax.random.split(rng)
            out["fric_scale"] = jax.random.uniform(
                k,
                (),
                minval=float(dr.cube_friction.min),
                maxval=float(dr.cube_friction.max),
            )
        # D19 (2026-07-29): cube size DECOUPLED into two independent ABSOLUTE
        # half-extent draws (metres) instead of one scalar multiplier -- xy
        # (width) and z (height) are sampled, clamped and applied separately.
        # Each has its own static gate, so enabling only one costs only one
        # rng split and leaves the other at its nominal identity value.
        if dr.cube_size_xy.enable:
            rng, k = jax.random.split(rng)
            out["half_xy"] = jax.random.uniform(
                k,
                (),
                minval=float(dr.cube_size_xy.min),
                maxval=float(dr.cube_size_xy.max),
            )
        if dr.cube_size_z.enable:
            rng, k = jax.random.split(rng)
            # Z half-extent is ALSO used by reset() to re-seat the box on the
            # table, so a drawn cube always starts flush, never intersecting.
            out["cube_half_z"] = jax.random.uniform(
                k,
                (),
                minval=float(dr.cube_size_z.min),
                maxval=float(dr.cube_size_z.max),
            )
        if dr.gravity.enable:
            rng, k1, k2 = jax.random.split(rng, 3)
            gmag = jp.linalg.norm(self._dr_nom_gravity)
            gdir = self._dr_nom_gravity / gmag
            dmag = jax.random.uniform(
                k1, (), minval=-float(dr.gravity.g_delta), maxval=float(dr.gravity.g_delta)
            )
            # Small random directional tilt: perturb the unit down-vector and
            # renormalize (tilt=0 -> pure magnitude noise along nominal g).
            tilt = float(dr.gravity.tilt)
            off = jax.random.uniform(k2, (3,), minval=-tilt, maxval=tilt)
            new_dir = gdir + off
            new_dir = new_dir / jp.linalg.norm(new_dir)
            out["grav"] = new_dir * (gmag + dmag)
        if dr.arm_stiffness.enable:
            rng, k = jax.random.split(rng)
            out["kp_scale"] = jax.random.uniform(
                k,
                (),
                minval=float(dr.arm_stiffness.min),
                maxval=float(dr.arm_stiffness.max),
            )
        if dr.arm_damping.enable:
            rng, k = jax.random.split(rng)
            out["kv_scale"] = jax.random.uniform(
                k, (), minval=float(dr.arm_damping.min), maxval=float(dr.arm_damping.max)
            )
        return out

    def _randomize_physics(
        self, model: mjx.Model, info: Dict[str, Any]
    ) -> mjx.Model:
        """Return a per-episode model with the DR factors (stashed in `info` by
        reset()) applied via tree_replace. Only ENABLED axes touch the model;
        each is gated on its static flag, so this is composable and isolatable
        for one-axis-at-a-time ablation. Called from step() only when the master
        switch is on -- see the `self._dr_enable` branch there.
        """
        dr = self._config.domain_rand
        updates = {}

        if dr.cube_mass.enable:
            # Scale mass AND inertia by the same factor (keeps them consistent).
            updates["body_mass"] = model.body_mass.at[self._obj_body].set(
                self._dr_nom_mass * info["dr_mass_scale"]
            )
            updates["body_inertia"] = model.body_inertia.at[self._obj_body].set(
                self._dr_nom_inertia * info["dr_mass_scale"]
            )
        if dr.cube_friction.enable:
            # Sliding (tangential dim 0) only; torsional/rolling left nominal.
            new_fric = self._dr_nom_friction.at[0].set(
                self._dr_nom_friction[0] * info["dr_fric_scale"]
            )
            updates["geom_friction"] = model.geom_friction.at[self._obj_geom].set(
                new_fric
            )
        if dr.cube_size_xy.enable or dr.cube_size_z.enable:
            # D19: independent xy/z half-extents (ABSOLUTE metres, not a scale of
            # the nominal). xy (width) hard-clamped graspable; z (height)
            # unclamped. Both info keys are always present -- identity (nominal)
            # when their own sub-axis is off, see self._dr_identity -- so
            # enabling one axis alone leaves the other at the nominal size.
            half_xy = jp.minimum(info["dr_half_xy"], self._dr_max_box_half_xy)
            new_size = jp.array(
                [half_xy, half_xy, info["dr_cube_half_z"]], dtype=float
            )
            updates["geom_size"] = model.geom_size.at[self._obj_geom].set(new_size)
        if dr.gravity.enable:
            updates["opt.gravity"] = info["dr_grav"]  # dotted-path tree_replace
        if dr.arm_stiffness.enable or dr.arm_damping.enable:
            # kp_scale / kv_scale are identity (1.0) when their sub-axis is off.
            new_kp = self._dr_nom_kp * info["dr_kp_scale"]
            gainprm = model.actuator_gainprm.at[:6, 0].set(new_kp)
            biasprm = model.actuator_biasprm
            biasprm = biasprm.at[:6, 1].set(-new_kp)  # mirror setpoint: bias1 = -kp
            biasprm = biasprm.at[:6, 2].set(self._dr_nom_kv * info["dr_kv_scale"])
            updates["actuator_gainprm"] = gainprm
            updates["actuator_biasprm"] = biasprm

        # TODO(dr): deferred DeXtreme-style axes, wired to their (off) flags:
        #   - dr.restitution: model.tree_replace geom_solref/solimp on box+fingers.
        #   - dr.joint_limits: model.jnt_range jitter on the 6 arm joints.
        # (cube_force + action_delay are per-STEP -> handled in step(), not here.)

        if not updates:
            return model
        return model.tree_replace(updates)

    def reset(self, rng: jax.Array) -> State:
        (
            rng,
            rng_box,
            rng_quat,
            rng_target,
            rng_robot,
            rng_gripper,
            rng_pose,
            rng_lift,
            rng_tilt,
        ) = jax.random.split(rng, 9)

        # Physics domain randomization: sample the per-episode factors up-front
        # so the cube half-extent is available for box placement below. STATIC
        # master gate -> when off, there is NO extra split (the rng carry is
        # untouched) and `dr` is the identity dict, so the whole reset stays
        # byte-for-byte unchanged. When on, the DR draw consumes only the carry,
        # leaving every task sub-key (box/target/pose/...) bit-identical.
        if self._dr_enable:
            rng, rng_dr = jax.random.split(rng)
            dr = self._sample_physics_dr(rng_dr)
        else:
            dr = self._dr_identity
        # Box Z half-extent for resting the cube (== _BOX_HALF_EXTENT unless the
        # cube_size_z axis is on).
        box_half_z = dr["cube_half_z"]

        # Action-delay DR: sample the per-episode delay length ONCE (held
        # constant for the whole episode -- the real deploy loop's HTTP/RTDE
        # timing does not change tick-to-tick in a way we'd model as
        # per-step). Same STATIC-gate pattern as the physics DR block above:
        # off -> no split, no info keys (see step() for the ring-buffer read).
        ad = self._config.domain_rand.action_delay
        if ad.enable:
            rng, rng_delay = jax.random.split(rng)
            action_delay_steps = jax.random.randint(
                rng_delay, (), 0, int(ad.max_delay_steps) + 1
            )
        # Observation-noise DR: sample the per-episode BIAS terms once here
        # (held constant for the episode); per-step JITTER is instead derived
        # deterministically inside _get_obs via jax.random.fold_in(seed,
        # info["step"]), so it needs no further rng threading through step().
        # Same STATIC-gate pattern: off -> no split, no info keys.
        on = self._config.domain_rand.obs_noise
        if on.enable:
            rng, rng_obs_seed, rng_obs_bias = jax.random.split(rng, 3)
            k_q, k_f, k_p, k_axis = jax.random.split(rng_obs_bias, 4)
            obs_bias_q = jax.random.uniform(
                k_q, (6,), minval=-float(on.q_bias), maxval=float(on.q_bias)
            )
            obs_bias_finger = jax.random.uniform(
                k_f, (), minval=-float(on.finger_bias), maxval=float(on.finger_bias)
            )
            obs_bias_box_pos = jax.random.uniform(
                k_p, (3,),
                minval=-float(on.box_pos_bias), maxval=float(on.box_pos_bias),
            )
            # Small-rotation bias vector for the box axes (see _skew); each
            # component independently ~ uniform(-box_quat_bias, +box_quat_bias)
            # rather than a unit-axis + angle draw -- simpler, and at these
            # small magnitudes (~3 deg) the difference is negligible.
            obs_bias_quat_vec = jax.random.uniform(
                k_axis, (3,),
                minval=-float(on.box_quat_bias), maxval=float(on.box_quat_bias),
            )

        # initialize box XY — wide jitter about the XML keyframe box XY
        # (nominal x=0.32 since D24). Y-center shiftable via box_y_center_offset
        # (0.0 = legacy, centered on the XML keyframe) so a sweep can push the
        # spawn to one side of the target.
        box_xy_jitter = jp.asarray(self._config.box_xy_jitter, dtype=float)
        box_xy = (
            jax.random.uniform(
                rng_box,
                (2,),
                minval=-box_xy_jitter,
                maxval=box_xy_jitter,
            )
            + self._init_obj_pos[:2]  # Box XY from XML keyframe
            + jp.array([0.0, float(self._config.box_y_center_offset)])
        )

        # D24 (2026-07-29) RADIAL CLIP -- project the rectangular draw above onto
        # the arm's REACHABLE ANNULUS around the base, in the base's own XY frame:
        #   r > _UR3_WORKING_RADIUS_M (0.49) -> scale xy by 0.49 / r
        #   r < _BOX_XY_R_MIN         (0.18) -> scale xy by 0.18 / r
        # Rationale: box_xy_jitter was widened to (0.17, 0.24) to cover more of
        # the real 0.6 x 1.0 m table, but the RECTANGLE's far corners reach
        # r = sqrt(0.49^2 + 0.24^2) = 0.546 m, ~11% past the UR3e's ~0.49 m
        # working radius (that radius is derived in _UR3_WORKING_RADIUS_M's
        # comment from this file's own target_r_max arithmetic, sqrt(0.35^2 +
        # 0.345^2) ~= 0.49), while its near edge (0.15, 0) sits inside the base
        # column. Clipping instead of shrinking keeps strictly MORE reachable
        # coverage than the old (0.15, 0.20) rectangle while producing ZERO
        # unreachable spawns.
        # Accepted side effect: draws that would have landed outside the annulus
        # pile up ON its boundary, so the marginal density is not uniform there.
        # That is deliberate -- boundary poses are the hard ones and are worth
        # over-sampling, and no draw is ever discarded (which would need a
        # rejection loop, not JAX-friendly).
        # Angle is preserved exactly (pure radial scale). Measured from the ROBOT
        # BASE, not the world origin, so it stays correct if the base moves.
        box_rel = box_xy - self._robot_base_xy
        box_r = jp.linalg.norm(box_rel)
        # jp.where on a traced scalar (no Python branching); the 1e-9 floor keeps
        # the division finite at the probability-zero exactly-at-base draw.
        box_r_safe = jp.maximum(box_r, 1e-9)
        box_r_clipped = jp.clip(
            box_r_safe, _BOX_XY_R_MIN, _UR3_WORKING_RADIUS_M
        )
        box_xy = self._robot_base_xy + box_rel * (box_r_clipped / box_r_safe)

        # Box yaw about world Z (±box_z_rot_range). The cube is symmetric so this
        # is the dominant spawn DOF; the slight plate tilt below adds the
        # out-of-plane component that makes the full 2-of-3-axis grasp alignment
        # (not just yaw) matter.
        #
        # WHY THE FULL 2*pi RANGE IS REQUIRED -- SETTLED, DO NOT NARROW IT.
        # It is tempting to argue that yaw only needs a pi/2 (or pi/4) range
        # because the cube has 90 deg rotational symmetry and the alignment
        # REWARD is built from max|axis . box_axis| -- absolute value and max,
        # hence invariant to that symmetry and to axis sign flips. That
        # argument is about the reward and does NOT carry over to the
        # OBSERVATION, which is what the network actually consumes.
        #
        # _get_obs feeds the policy `jaw_proj = jaw_axis @ box_axes` and
        # `app_proj = app_axis @ box_axes`: RAW SIGNED dot products onto all
        # three box axes, with no absolute value and no max. A 180 deg yaw
        # negates b0 and b1, so [a, b, c] becomes [-a, -b, c]; a 90 deg yaw
        # permutes and negates them. The MLP has no notion of the cube's
        # symmetry and sees each of those as a distinct input.
        #
        # Narrowing the range therefore DOES shrink the effective training
        # distribution of the observation. Observed empirically: with a
        # narrow yaw range the policy failed outright once the physical cube
        # was rotated 180 deg -- textbook out-of-distribution input. The real
        # mocap streams yaw over the full 2*pi, so training must cover it.
        # See the D26 entry in the vault's "Plan - Sim-to-Real Gap Protocol".
        theta = jax.random.uniform(
            rng_quat,
            (),
            minval=-self._config.box_z_rot_range,
            maxval=self._config.box_z_rot_range,
        )
        q_yaw = jp.array(
            [jp.cos(theta / 2.0), 0.0, 0.0, jp.sin(theta / 2.0)], dtype=float
        )

        # lifter height + tilt + box resting pose. When enabled, sample a
        # per-episode plate height AND a slight roll/pitch tilt, then rest the box
        # FLUSH on the tilted plate top (bottom face parallel to the plate). The
        # box therefore starts at a variable height and slightly tilted, so the
        # policy must reorient the gripper (not just yaw) to grasp it. Disabled ->
        # legacy flat on-floor spawn (yaw only).
        if self._lifter_enabled:
            # D18/D24 (2026-07-29): sample the table TOP SURFACE as an ABSOLUTE
            # height, U(lifter_height_abs_min, lifter_height_abs_max) -- NOT the
            # old symmetric band about lifter_height_nom. Default U(0.0, 0.220)
            # spans bare floor to a 220 mm table. lifter_height_nom stays what it
            # always was: the NOMINAL the box anchor and the XML plate z encode
            # (asserted in __init__), and the reference the target shift below is
            # measured against -- it is no longer the centre of the draw.
            # Converted to the mocap body z so every downstream expression keeps
            # its meaning, and clamped so a low draw can't push the plate bottom
            # below the floor plane z=0 (the collision masks can't separate them).
            lifter_top = jax.random.uniform(
                rng_lift,
                (),
                minval=float(self._config.lifter_height_abs_min),
                maxval=float(self._config.lifter_height_abs_max),
            )
            lifter_h = jp.maximum(
                lifter_top - _LIFTER_HALF_THICKNESS, _LIFTER_HEIGHT_MIN
            )
            # DEVIATION of the realized top surface from the nominal. The box
            # anchor (_init_obj_pos[2]) already sits on the NOMINAL table, so this
            # -- not the absolute height -- is what the lift target is shifted by.
            # Recomputed from lifter_h so the clamp above is accounted for.
            lifter_dz = (lifter_h + _LIFTER_HALF_THICKNESS) - self._lifter_top_nom
            tilt = jax.random.uniform(
                rng_tilt,
                (2,),
                minval=-float(self._config.lifter_tilt_max),
                maxval=float(self._config.lifter_tilt_max),
            )
            roll, pitch = tilt[0], tilt[1]
            q_roll = jp.array([jp.cos(roll / 2.0), jp.sin(roll / 2.0), 0.0, 0.0])
            q_pitch = jp.array([jp.cos(pitch / 2.0), 0.0, jp.sin(pitch / 2.0), 0.0])
            q_tilt = _quat_mul(q_pitch, q_roll)
            # Plate top-surface normal = 3rd column of R(q_tilt).
            tw, tx, ty, tz = q_tilt[0], q_tilt[1], q_tilt[2], q_tilt[3]
            n = jp.array(
                [
                    2 * (tx * tz + tw * ty),
                    2 * (ty * tz - tw * tx),
                    1 - 2 * (tx * tx + ty * ty),
                ]
            )
            # The plate is pinned at its FIXED XML XY (self._lifter_xy, below) and
            # tilts about that origin; its top plane is raised by the half-
            # thickness. Solve the plane height under the (jittered) box XY, then
            # lift the box center a half-extent along the plate normal so the cube
            # sits flush. D24: the lever arm is now |box_xy - TABLE xy|, not
            # |box_xy - box anchor| -- the tilt pivot is the table, which no
            # longer sits at the box anchor (table x=0.40, box anchor x=0.32).
            p_top_z = lifter_h + _LIFTER_HALF_THICKNESS
            z_plane = p_top_z - (
                n[0] * (box_xy[0] - self._lifter_xy[0])
                + n[1] * (box_xy[1] - self._lifter_xy[1])
            ) / n[2]
            box_z = z_plane + box_half_z * n[2]
        else:
            lifter_h = jp.array(0.0, dtype=float)
            lifter_dz = jp.array(0.0, dtype=float)
            q_tilt = jp.array([1.0, 0.0, 0.0, 0.0])
            box_z = box_half_z  # rest at half-height (0.02 unless cube_size_z DR)

        box_pos = jp.array([box_xy[0], box_xy[1], box_z])
        # Box orientation: plate tilt composed with the yaw spin about the plate
        # normal, so the cube rests flush on the (possibly tilted) plate.
        box_quat = _quat_mul(q_tilt, q_yaw)
        lifter_quat = q_tilt

        # initialize target position — a lift point in the air above the box.
        # X biased forward (+0.01‥+0.06) so the lift pulls the box toward the
        # robot's reachable front; Y pulled in (±0.03) so the lift stays near
        # the robot's sagittal plane. Z band RAISED to 0.18‥0.21 m (was
        # 0.12‥0.15) for a clearly higher lift, on request. WARNING: with the
        # 95 mm TABLE the anchor is 0.115, so this band lands at world-z
        # 0.295‥0.325 — target_r_max was cut 0.42 -> 0.35 to keep the worst 3D
        # target at ~0.48 m, inside the UR3's ~0.54 m reach. Watch success/
        # box_target_dist; pull the Z MAX back toward 0.18 if success collapses.
        # Shifted by the table's per-episode DEVIATION from its nominal height
        # (not the absolute height — the anchor already carries the nominal), so
        # the lift target tracks the box up and down with the table.
        # Y-center shiftable via target_y_center_offset (0.0 = legacy) — set
        # to the OPPOSITE sign of box_y_center_offset to force real lateral
        # transport distance between spawn and target (see default_config).
        # CAUTION: this stacks with the existing near-reach-limit Z band
        # (see warning above) — push gradually and watch success/
        # box_target_dist for collapse from unreachable targets.
        # target_mode is a STATIC config value, so this Python if/else is
        # resolved at trace time (jit/vmap-safe).
        tz_jitter = jp.asarray(self._config.target_z_jitter, dtype=float)
        if self._config.target_mode == "box":
            target_pos = (
                jax.random.uniform(
                    rng_target,
                    (3,),
                    minval=jp.array([0.02, -0.03, tz_jitter[0]]),
                    maxval=jp.array([0.06, 0.03, tz_jitter[1]]),
                )
                + self._init_obj_pos
                + jp.array([0.0, float(self._config.target_y_center_offset), 0.0])
            )
            # Shift by the table's DEVIATION from nominal, not its absolute
            # height: _init_obj_pos[2] (the anchor added just above) already
            # places the cube on the nominal table.
            target_pos = target_pos.at[2].add(lifter_dz)
        else:  # "base_polar"
            # Base-centered polar sampling: put the lift target in an annulus
            # around the ROBOT BASE (radius r), at an azimuth phi that is offset
            # by dphi (90°‥180°) to a random side of the box's bearing from the
            # base. Because the box XY is already jittered above, the target
            # tracks where the box actually spawned instead of a fixed anchor —
            # this replaces the box_y_center_offset/target_y_center_offset
            # transport hack. Every draw gets its own split key (jit/vmap-safe).
            rng_r, rng_azim, rng_side, rng_tz = jax.random.split(rng_target, 4)
            base_xy = self._robot_base_xy
            box_bearing = jp.arctan2(
                box_xy[1] - base_xy[1], box_xy[0] - base_xy[0]
            )
            dphi = jax.random.uniform(
                rng_azim,
                (),
                minval=float(self._config.target_azim_min),
                maxval=float(self._config.target_azim_max),
            )
            side = jp.sign(
                jax.random.uniform(rng_side, (), minval=-1.0, maxval=1.0)
            )
            phi = box_bearing + side * dphi
            r = jax.random.uniform(
                rng_r,
                (),
                minval=float(self._config.target_r_min),
                maxval=float(self._config.target_r_max),
            )
            target_xy = base_xy + r * jp.array([jp.cos(phi), jp.sin(phi)])
            # HEIGHT NOTE: Z is kept identical to the legacy "box" path — the
            # fixed 0.18‥0.21 band above the keyframe box height (which now sits
            # on the nominal table), shifted by the table's per-episode deviation
            # from that nominal. Only the XY changes here. For a true reach SPHERE,
            # replace this fixed Z-band with an elevation angle (target_z from r
            # and an elevation draw) at this spot.
            target_z = (
                jax.random.uniform(
                    rng_tz, (), minval=tz_jitter[0], maxval=tz_jitter[1]
                )
                + self._init_obj_pos[2]
            )
            target_pos = jp.array([target_xy[0], target_xy[1], target_z])
            # Shift by the table's DEVIATION from nominal, not its absolute
            # height: _init_obj_pos[2] (the anchor added just above) already
            # places the cube on the nominal table.
            target_pos = target_pos.at[2].add(lifter_dz)

        # D18/D24 reach safeguard: hard-clip the ABSOLUTE world-Z of the sampled
        # target so no episode -- regardless of how the (now absolute, up to
        # 0.220 m) table draw and the target_z draw combine -- can command a
        # target above _TARGET_WORLD_Z_CAP (0.345 m). Applied to BOTH
        # target_mode branches uniformly. Worst case without it:
        #   target_z = draw(0.21) + anchor(0.115) + lifter_dz(0.220 - 0.095)
        #            = 0.45 m, which at target_r_max=0.35 demands
        #   sqrt(0.35^2 + 0.45^2) ~= 0.57 m, well past the ~0.49 m working radius.
        # Physically sensible: you cannot lift as far above an already-high table.
        # The realized cap RATE is logged as the `target_z_capped` metric below --
        # watch it, a high rate means the config is asking for the impossible.
        target_z_capped = (target_pos[2] > _TARGET_WORLD_Z_CAP).astype(jp.float32)
        target_pos = target_pos.at[2].set(
            jp.minimum(target_pos[2], _TARGET_WORLD_Z_CAP)
        )

        # -----------------------------
        # Randomize robot joint positions
        # -----------------------------
        # Base arm pose: a random hand-collected library pose when a difficulty
        # level is set, else the literal keyframe start. init_qpos_noise then
        # jitters the 6 arm joints uniform(-v, +v); set the noise to 0 in a sweep
        # to use the library/keyframe arm pose verbatim. The gripper is handled
        # separately below (randomized full open<->closed, ignoring init_qpos_noise).
        if self._init_pose_lib is not None:
            pose = self._init_pose_lib[
                jax.random.randint(rng_pose, (), 0, self._n_init_poses)
            ]
            base_arm_qpos = pose[:6]
        else:
            base_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])

        noise_amp = jp.asarray(self._config.init_qpos_noise, dtype=float)
        arm_amp = noise_amp[: len(self._robot_arm_qposadr)]  # 6 arm joints

        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),  # 6 arm joints
            minval=-arm_amp,
            maxval=arm_amp,
        )
        noisy_arm_qpos = base_arm_qpos + robot_qpos_noise
        # Clip to the arm actuator ctrlrange so large init noise (e.g. a
        # full-travel wrist) on a base pose near a limit can't write an
        # out-of-range value into qpos OR the position-actuator ctrl below
        # (which would clamp and jerk). Same 6-arm indexing used for init_ctrl.
        n_arm = len(self._robot_arm_qposadr)
        noisy_arm_qpos = jp.clip(
            noisy_arm_qpos, self._lowers[:n_arm], self._uppers[:n_arm]
        )

        # Gripper: the two fingers are coupled into one combined opening, so
        # randomize it as a single value uniformly across the full physical
        # per-finger range [0, 0.025] m (0 = open, 0.025 = closed) and apply it to
        # both fingers. This starts the policy anywhere between fully open and
        # fully closed, independent of the base pose and init_qpos_noise.
        if self._config.finger_random_init:
            finger_sample = jax.random.uniform(
                rng_gripper, (), minval=0.0, maxval=0.025
            )
        else:
            # Deterministic DR-ladder baseline: always start fully OPEN.
            # rng_gripper is still split above (fixed 9-way split, jit/vmap-
            # safe) but simply goes unused here.
            finger_sample = jp.array(0.0, dtype=float)
        noisy_finger_qpos = jp.full((2,), finger_sample)

        # -----------------------------
        # Build initial qpos with randomized arm joints
        # -----------------------------
        init_q = jp.array(self._init_q)
        init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(box_quat)
        init_q = init_q.at[self._robot_arm_qposadr].set(noisy_arm_qpos)
        init_q = init_q.at[self._robot_qposadr[-2:]].set(noisy_finger_qpos)

        # -----------------------------
        # Make ctrl consistent with init_q (avoids residual torques at reset)
        # -----------------------------
        init_ctrl = jp.array(self._init_ctrl)
        init_ctrl = init_ctrl.at[: len(self._robot_arm_qposadr)].set(noisy_arm_qpos)
        # Tendon actuator commands the symmetric finger position (coef 0.5 each)
        init_ctrl = init_ctrl.at[-1].set(noisy_finger_qpos.sum() * 0.5)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # set target mocap (lift-goal marker); pin the lifter plate at its FIXED
        # XML XY (the real table's measured position) so only its height/tilt
        # vary per episode.
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )
        if self._lifter_enabled:
            # D24 (2026-07-29): XY pinned to the TABLE's own XML pos
            # (self._lifter_xy = 0.40, 0), NOT to the box anchor. It used to be
            # self._init_obj_pos[:2], a leftover from when the lifter was a small
            # movable riser that had to be placed under the cube -- harmless only
            # while the two happened to coincide at x=0.40. The lifter is now a
            # fixed 0.6 x 1.0 m table AND the box anchor moved to x=0.32, so the
            # old pin would shove the whole table 8 cm toward the base every
            # reset, contradicting both the measured extent (x in [0.10, 0.70])
            # and gap_target.py's table-anchor read. The table does not move.
            lifter_pos = jp.array(
                [self._lifter_xy[0], self._lifter_xy[1], lifter_h]
            )
            data = data.replace(
                mocap_pos=data.mocap_pos.at[self._lifter_mocap, :].set(lifter_pos),
                mocap_quat=data.mocap_quat.at[self._lifter_mocap, :].set(lifter_quat),
            )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "success": jp.array(0.0, dtype=float),
            "box_target_dist": jp.array(0.0, dtype=float),
            # Non-summed diagnostics (see the "at_terminal" gating in step()):
            # box_target_dist above is a per-step value that brax's EvalWrapper
            # SUMS over the whole episode when logging eval/episode_*, so it
            # reads as a summed proxy dominated by the (long, far) approach
            # phase -- NOT the real final gap. These two are gated to be
            # nonzero only on the episode's one terminal step, so their W&B
            # sum equals that single value and the cross-env mean is the true
            # mean, in meters.
            "box_target_dist_final": jp.array(0.0, dtype=float),
            "box_target_dist_min": jp.array(0.0, dtype=float),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
            "align_jaw": jp.array(0.0, dtype=float),
            "align_app": jp.array(0.0, dtype=float),
            # D18/D24 reach-safeguard diagnostic (see _TARGET_WORLD_Z_CAP):
            # terminal-gated capping RATE, mirroring box_target_dist_final's
            # pattern so the W&B episode-sum equals the single 0/1 per episode.
            # Always 0 out of reset_to_state (its target_pos is caller-supplied
            # and bypasses reset()'s sampling and cap entirely).
            "target_z_capped": jp.array(0.0, dtype=float),
            # 2026-08 approach/grasp/speed diagnostics -- see the module-level
            # _STEP_METRIC_KEYS / _TERMINAL_METRIC_KEYS comment for how to read
            # each one. Seeded identically in reset() and reset_to_state().
            **{k: jp.array(0.0, dtype=float)
               for k in _STEP_METRIC_KEYS + _TERMINAL_METRIC_KEYS},
            **{k: jp.array(0.0, dtype=float)
               for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "step": jp.array(0, dtype=jp.int32),
            "success_counter": jp.array(0, dtype=jp.int32),
            # Sticky "this episode has succeeded at least once" latch. Needed
            # because success no longer ends the episode (see done= in step):
            # the raw `success` flag is true on EVERY step the box stays inside
            # success_tol, so summing it would measure DWELL, not success rate.
            # The latch is emitted once, on the terminal step, so the summed
            # eval/episode_success keeps its old 0/1-per-episode meaning and
            # stays comparable with the pre-2026-07-26 ladders.
            "success_ever": jp.array(0.0, dtype=float),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
            # Per-episode box resting height (top of the lifter plate, or the
            # keyframe floor Z). The "lifted" latch measures lift against this.
            "box_rest_z": box_z,
            # D18/D24: whether THIS episode's raw target-Z draw was clipped by
            # _TARGET_WORLD_Z_CAP (1.0) or not (0.0). See the cap block above.
            "target_z_capped": target_z_capped,
            # Previous action, for the action_rate smoothness penalty. Zeroed
            # at reset (no prior action yet); also appended to the obs (below)
            # so the action_rate-dependent reward stays a function of the
            # observed state (Markov).
            "last_action": jp.zeros(self._nu, dtype=float),
            # Running minimum box_target_dist seen this episode (closest
            # approach), updated in _get_reward. Large init so the very first
            # step's real distance always wins the jp.minimum.
            "dist_min": jp.array(1e3, dtype=float),
            # Consecutive steps the (lifted) box has stayed within hold_radius
            # of the target; drives the hold_target dwell-time bonus. Reset to
            # 0 the instant the box leaves the radius or drops. NOT part of the
            # observation (matches the existing success_counter precedent) --
            # if hold_target proves hard to learn, adding it to obs is the
            # fallback, not done preemptively.
            "hold_counter": jp.array(0, dtype=jp.int32),
            # REALIZED box half-extents (m) along the box's own local axes.
            # ALWAYS present -- deliberately NOT gated on self._dr_enable,
            # because _get_reward's axis-aware alignment reads it every step and
            # must not branch on the DR switch. `dr` is self._dr_identity when
            # DR is off, so this is the nominal (0.015, 0.015, 0.020) there. The
            # jp.minimum mirrors _randomize_physics's _dr_max_box_half_xy
            # graspability clamp, so this is the half-extent the PHYSICS
            # actually got, not the pre-clamp draw.
            "box_half": jp.stack(
                [
                    jp.minimum(dr["half_xy"], self._dr_max_box_half_xy),
                    jp.minimum(dr["half_xy"], self._dr_max_box_half_xy),
                    dr["cube_half_z"],
                ]
            ),
            # This episode's ACTUAL arm start pose, for robot_target_qpos.
            # That term used to measure against self._init_q (the task_home
            # KEYFRAME), but episodes start either from a pose drawn out of
            # init_poses/train/mid.json (init_start_random="mid", the default)
            # or from the keyframe plus init_qpos_noise whose wrist_3 amplitude
            # is 6.28 rad -- so it was rewarding the policy for driving toward a
            # pose the episode never started at, and with the wrist term
            # dominating the norm it was mostly tanh-saturated anyway.
            # evaluation/ur3_reward_replay.py:201 already does it this way on
            # the real side (self.init_arm_q = arm_q on the first tick), so this
            # also closes a silent sim/replay parity gap.
            "init_arm_q": noisy_arm_qpos,
            # Time-to-stage counters (steps), latched on the rising edge of each
            # stage in _get_reward. Initialized to episode_length so "never
            # happened" reads as a right-censored value rather than 0 -- a 0
            # would average in as "reached instantly", inverting the metric.
            "t_reach": jp.array(int(self._config.episode_length), dtype=jp.int32),
            "t_grasp": jp.array(int(self._config.episode_length), dtype=jp.int32),
            "t_lift": jp.array(int(self._config.episode_length), dtype=jp.int32),
            # Alignment and jaw span sampled at the rising edge of `grasped` --
            # the direct readout of whether grasp_align_thresh did what it was
            # raised to do. Expect jaw_span_at_grasp ~= 0.030 (the 3 cm face),
            # not 0.040 (the 4 cm long axis).
            "align_at_grasp": jp.array(0.0, dtype=float),
            "jaw_span_at_grasp": jp.array(0.0, dtype=float),
            # Running minimum gripper-to-box distance (closest approach). The
            # existing "dist_min" tracks BOX-to-TARGET; there was no approach
            # -phase distance diagnostic at all, which is why the approach was
            # effectively unobservable in W&B.
            "grip_dist_min": jp.array(1e3, dtype=float),
            # Previous-step TCP position, for the tcp_speed diagnostic. Zeroed
            # here on purpose: mjx_env.make_data does NOT run forward kinematics,
            # so data.site_xpos is all-zero at this point and seeding from it
            # would be meaningless. _get_reward guards the first step explicitly.
            "prev_tcp": jp.zeros(3, dtype=float),
        }

        # Stash the per-episode DR factors in info (read by _randomize_physics in
        # step()) and seed their W&B metric keys. STATIC gate -> these keys exist
        # only when DR is on, keeping the State pytree (and the off path) clean.
        if self._dr_enable:
            info.update(
                {
                    "dr_mass_scale": dr["mass_scale"],
                    "dr_fric_scale": dr["fric_scale"],
                    # D19: independent xy/z half-extents replace the old scalar.
                    "dr_half_xy": dr["half_xy"],
                    "dr_cube_half_z": dr["cube_half_z"],
                    "dr_grav": dr["grav"],
                    "dr_kp_scale": dr["kp_scale"],
                    "dr_kv_scale": dr["kv_scale"],
                }
            )
            metrics.update(
                {f"dr/{k}": jp.array(0.0, dtype=float) for k in _DR_METRIC_KEYS}
            )

        # Action-delay ring buffer + the sampled per-episode delay length (see
        # the sampling above and step()'s exec_action read). STATIC gate ->
        # keys exist only when this axis is enabled.
        if ad.enable:
            info.update(
                {
                    "action_buffer": jp.zeros(
                        (int(ad.max_delay_steps) + 1, self._nu), dtype=float
                    ),
                    "dr_action_delay_steps": action_delay_steps,
                }
            )
            metrics.update({"dr/action_delay_steps": jp.array(0.0, dtype=float)})

        # D20 (2026-07-29): burst-perturbation state for cube_force /
        # joint_torque -- a per-episode counter + the currently-held vector,
        # mirroring the action_delay ring buffer's "STATIC gate -> keys exist
        # only when the axis is enabled" pattern. counter=0 means "no burst
        # active"; the trigger/hold logic itself lives in step(). This is the
        # natural start state, not a special eval bypass: an episode simply
        # begins with no burst in flight.
        cf_init = self._config.domain_rand.cube_force
        if cf_init.enable:
            info.update(
                {
                    "cf_burst_counter": jp.array(0, dtype=jp.int32),
                    "cf_burst_force": jp.zeros(3, dtype=float),
                }
            )
        jt_init = self._config.domain_rand.joint_torque
        if jt_init.enable:
            info.update(
                {
                    "jt_burst_counter": jp.array(0, dtype=jp.int32),
                    "jt_burst_torque": jp.zeros(6, dtype=float),
                }
            )

        # Obs-noise per-episode bias terms + the fold_in seed for the per-step
        # jitter (see _get_obs). STATIC gate -> keys exist only when enabled.
        if on.enable:
            info.update(
                {
                    "obs_noise_seed": rng_obs_seed,
                    "obs_bias_q": obs_bias_q,
                    "obs_bias_finger": obs_bias_finger,
                    "obs_bias_box_pos": obs_bias_box_pos,
                    "obs_bias_quat_vec": obs_bias_quat_vec,
                }
            )

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def reset_to_state(
        self,
        rng: jax.Array,
        arm_qpos: jax.Array,
        finger: jax.Array,
        box_pos: jax.Array,
        box_quat: jax.Array,
        target_pos: jax.Array,
        cube_half_extents: Optional[jax.Array] = None,
        lifter_top_height: Optional[jax.Array] = None,
        lifter_tilt_rp: Optional[jax.Array] = None,
    ) -> State:
        """EVAL-ONLY reset to an EXACT given state -- bypasses reset()'s random
        sampling entirely. Used by evaluation/run_gap_protocol_sim.py (the
        gap-protocol sim mirror, Commit 7) to reset to a real robot's measured
        initial state (arm pose + box pose, from measured_init.json) instead of
        a randomly sampled episode, so a sim rollout and its matched real run
        start from the identical state (D1/D3: "sim mirrors the measurement").

        DO NOT call this from the training loop. `reset()` above is the ONLY
        reset path used by training and remains byte-for-byte unchanged by this
        method's existence -- this is an additive method, nothing above it was
        touched.

        Args:
          rng: PRNGKey. Threaded into `info["rng"]` (consumed by step()'s
            per-step cube_force draw, if that axis is enabled -- see the
            "per-step DR" note below) and into obs-noise jitter's fold_in seed
            (if that axis is enabled). Pass the SAME key for two calls that
            should reproduce the identical rollout (this is the determinism
            check evaluation/run_gap_protocol_sim.py runs).
          arm_qpos: (6,) rad, the 6 arm joint angles (measured or commanded).
          finger: scalar m in [0, 0.025], applied to BOTH fingers symmetrically
            (matches the tendon actuator's coef 0.5/0.5, same as reset()).
          box_pos: (3,) world m, the box freejoint position.
          box_quat: (4,) MuJoCo (w, x, y, z), the box freejoint orientation.
            Normalized internally (defends against a slightly-off-unit logged
            quaternion, same tolerance-guard as SimFK.geom in
            evaluation/ur3_reward_replay.py).
          target_pos: (3,) world m, the lift-target mocap position. Callers
            MUST obtain this from evaluation/gap_target.target_for_episode()
            (never re-derived here) so a real run and this sim mirror can never
            disagree about which target they were scored against (D2).
          cube_half_extents: OPTIONAL (3,) world m, box geom half-extents
            (x, y, z). D17 cube-size probe -- ADDITIVE, defaults to None,
            which is the pre-D17 behaviour: the box geom stays at its nominal
            XML size (or whatever `_randomize_physics` would apply, which is
            identity here since physics DR is bypassed above -- so `None`
            reproduces this method's exact pre-D17 output byte-for-byte).
            When given, it is applied UNCLAMPED and bypasses BOTH (a) this
            method's own physics-DR-identity path (which never touched
            geom_size to begin with) and (b) step()'s `_randomize_physics`
            `_dr_max_box_half_xy = 0.020` graspability clamp (D19 raised it
            from 0.018) -- that clamp
            exists to keep *training* domain-randomization draws graspable;
            the 4cm eval cube is a deliberate one-off OOD override applied
            only at eval time by evaluation/run_gap_protocol_sim.py, not a DR
            draw, and must not be silently clamped back down. See
            `_DR17_EVAL_CUBE_HALF_EXTENTS_KEY` in step() for how the override
            is threaded through -- it lives in `info`, so it is per-episode
            (this call), not a mutation of `self._mjx_model` (which would
            leak across envs/episodes under vmap).
          lifter_top_height: OPTIONAL scalar world m, the ABSOLUTE table-top
            height to pin the "lifter" mocap body at (same quantity as
            reset()'s `lifter_top` draw -- NOT a deviation from nominal).
            D23/D25 gap-protocol-sim addition (2026-07-29) -- ADDITIVE,
            defaults to None, which is the pre-existing behaviour: the table
            pinned LEVEL at `lifter_height_nom` (see the "NOT bypassed"
            section below, unchanged when this is omitted). Needed to mirror
            D23 Block D ("+125mm lifter block, top ~150mm") in sim -- without
            it, `reset_to_state` puts the measured box at its real height
            while the table geometry underneath stays flat-at-95mm, an
            inconsistent scene (box floating above / clipped through the
            plate). Clamped the same way reset()'s own draw is
            (`_LIFTER_HEIGHT_MIN` floor) -- a call passing an implausible
            value gets the same floor training would have applied, not a
            silent negative-height plate.
          lifter_tilt_rp: OPTIONAL (2,) [roll, pitch] radians, the table tilt
            about its own fixed XY (same convention as reset()'s `tilt`
            draw). D23/D25 addition, ADDITIVE, defaults to None (level, the
            pre-existing behaviour). Needed to mirror D23 Block C ("~5 deg
            wedge"). Unlike `lifter_top_height`, NOT clamped to
            `lifter_tilt_max` -- that config value bounds a *training* draw,
            and an eval-time measured wedge angle is a fact about the real
            board, not a distribution to enforce. Composed into the lifter
            mocap quat with the identical roll-then-pitch quaternion multiply
            reset() uses (`_quat_mul(q_pitch, q_roll)`), so a value equal to
            a training draw reproduces the identical `lifter_quat` bit for
            bit.
          NOTE: passing one of `lifter_top_height`/`lifter_tilt_rp` without
          the other is allowed -- each defaults independently (height stays
          nominal if only tilt is given, and vice versa). Both require
          `self._lifter_enabled` (always True in this env, see __init__);
          passing either while the lifter is disabled raises rather than
          silently doing nothing.

        What is and is NOT bypassed
        ----------------------------------------------------------------------
        Bypassed entirely (forced to IDENTITY, regardless of what this env's
        `self._config.domain_rand.*` says -- i.e. regardless of the DR-ladder
        config the loaded policy trained under): the per-episode PHYSICS DR
        draw (cube mass/friction/size, gravity, arm stiffness/damping -- see
        `self._dr_identity`), the per-episode ACTION-DELAY length (forced to
        0 -- no delay), and the per-episode OBS-NOISE BIAS terms (forced to
        the zero vector). None of reset()'s random sampling (box XY/yaw,
        lifter height/tilt, target draw, arm-pose-library index, init_qpos_noise,
        finger_random_init) runs at all -- every one of those quantities is an
        ARGUMENT here instead, sourced from the measured init.

        NOT bypassed (a documented, deliberate choice, not an oversight):
          - The per-STEP cube_force / joint_torque BURST disturbances (D20,
            domain_rand.cube_force / domain_rand.joint_torque), if this
            policy's config has them enabled, still fire every step() call
            during the rollout -- they are not "reset" quantities to bypass;
            they are part of the deployed policy's OWN dynamics model and
            forcing them off would evaluate a policy that never trained under
            them. Their burst state (info["cf_burst_*"] / info["jt_burst_*"])
            starts at "no burst active" below -- not a special bypass, just
            the natural start state, identical to reset()'s.
          - The per-STEP obs-noise JITTER (domain_rand.obs_noise, the
            q_jitter/finger_jitter/box_pos_jitter/box_quat_jitter amplitudes),
            if enabled, still applies every step via `_get_obs`'s
            `jax.random.fold_in(seed, info["step"])` -- it is a fixed, non-zero,
            config-baked amplitude unrelated to reset()'s random sampling, and
            (like cube_force) is part of what the policy trained under. Only
            the per-episode BIAS component is zeroed above. Because the jitter
            is a pure function of `info["obs_noise_seed"]` + the step counter
            (never resampled), the rollout is still fully deterministic given
            the same `rng` argument -- this is what run_gap_protocol_sim.py's
            determinism check verifies.
          - The lifter mocap body: it models the REAL lab table (0.6 x 1.0 m,
            top surface 95 mm above the base origin), so BY DEFAULT (both
            `lifter_top_height` and `lifter_tilt_rp` omitted) it is pinned at
            the nominal `lifter_height_nom`, level, at its FIXED XML XY --
            the per-episode height/tilt DRAW is bypassed like every other
            reset() sampling, but the table itself is not, because it
            physically exists. (Before 2026-07-28 it was pinned at z=0 under
            a "no lifter on the real robot" convention; see
            evaluation/gap_target.py, updated to match. D24, 2026-07-29: the
            XY pin moved from the box anchor to the table's own XML pos --
            the two no longer coincide.) D23/D25 (2026-07-29): when the
            caller passes `lifter_top_height` and/or `lifter_tilt_rp` (to
            mirror a real D23 Block C/D run, where the table itself was
            physically reconfigured, not just the box's position on it),
            THIS bypass is what gets overridden -- the table is pinned at
            the GIVEN height/tilt instead of nominal/level. Its XY stays
            fixed at `self._lifter_xy` either way (the table does not move
            laterally in any D23 condition).
        """
        arm_qpos = jp.asarray(arm_qpos, dtype=float)
        box_pos = jp.asarray(box_pos, dtype=float)
        box_quat = jp.asarray(box_quat, dtype=float)
        box_quat = box_quat / jp.linalg.norm(box_quat)
        target_pos = jp.asarray(target_pos, dtype=float)
        finger_qpos = jp.full((2,), jp.asarray(finger, dtype=float))

        init_q = jp.array(self._init_q)
        init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(
            box_quat
        )
        init_q = init_q.at[self._robot_arm_qposadr].set(arm_qpos)
        init_q = init_q.at[self._robot_qposadr[-2:]].set(finger_qpos)

        # ctrl consistent with init_q, same rationale as reset() ("avoids
        # residual torques at reset").
        init_ctrl = jp.array(self._init_ctrl)
        init_ctrl = init_ctrl.at[: len(self._robot_arm_qposadr)].set(arm_qpos)
        init_ctrl = init_ctrl.at[-1].set(finger_qpos.sum() * 0.5)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )
        if self._lifter_enabled:
            # The table is real -- by default pin it LEVEL at the nominal
            # height and at its own FIXED XML XY (the per-episode height/tilt
            # DRAW is what gets bypassed here, not the table). D24: XY was
            # self._init_obj_pos[:2], the box anchor, which no longer
            # coincides with the table.
            # D23/D25 (2026-07-29): lifter_top_height / lifter_tilt_rp let a
            # caller override the height and/or tilt to match a REAL D23
            # Block C/D run, where the table itself was physically
            # reconfigured -- not a "bypass" of a random draw (there is none
            # here to bypass), but reproducing a fact about the run being
            # mirrored. XY is never overridden -- the table does not move
            # laterally in any D23 condition. See the docstring's "NOT
            # bypassed" section (last bullet) and the Args entries above.
            if lifter_top_height is None:
                lifter_z = self._lifter_z_nom
            else:
                lifter_z = jp.maximum(
                    jp.asarray(lifter_top_height, dtype=float)
                    - _LIFTER_HALF_THICKNESS,
                    _LIFTER_HEIGHT_MIN,
                )
            lifter_pos = jp.array(
                [self._lifter_xy[0], self._lifter_xy[1], lifter_z]
            )
            if lifter_tilt_rp is None:
                lifter_quat = jp.array([1.0, 0.0, 0.0, 0.0])
            else:
                tilt_rp = jp.asarray(lifter_tilt_rp, dtype=float)
                roll, pitch = tilt_rp[0], tilt_rp[1]
                q_roll = jp.array(
                    [jp.cos(roll / 2.0), jp.sin(roll / 2.0), 0.0, 0.0]
                )
                q_pitch = jp.array(
                    [jp.cos(pitch / 2.0), 0.0, jp.sin(pitch / 2.0), 0.0]
                )
                lifter_quat = _quat_mul(q_pitch, q_roll)
            data = data.replace(
                mocap_pos=data.mocap_pos.at[self._lifter_mocap, :].set(lifter_pos),
                mocap_quat=data.mocap_quat.at[self._lifter_mocap, :].set(
                    lifter_quat
                ),
            )
        elif lifter_top_height is not None or lifter_tilt_rp is not None:
            raise ValueError(
                "reset_to_state: lifter_top_height/lifter_tilt_rp given but "
                "self._lifter_enabled is False -- there is no lifter mocap "
                "body to set. (This env always enables the lifter as of "
                "D24; seeing this means an env config this eval-only method "
                "was not written against.)"
            )

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "success": jp.array(0.0, dtype=float),
            "box_target_dist": jp.array(0.0, dtype=float),
            "box_target_dist_final": jp.array(0.0, dtype=float),
            "box_target_dist_min": jp.array(0.0, dtype=float),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
            "align_jaw": jp.array(0.0, dtype=float),
            "align_app": jp.array(0.0, dtype=float),
            # D18/D24 reach-safeguard diagnostic (see _TARGET_WORLD_Z_CAP):
            # terminal-gated capping RATE, mirroring box_target_dist_final's
            # pattern so the W&B episode-sum equals the single 0/1 per episode.
            # Always 0 out of reset_to_state (its target_pos is caller-supplied
            # and bypasses reset()'s sampling and cap entirely).
            "target_z_capped": jp.array(0.0, dtype=float),
            # 2026-08 approach/grasp/speed diagnostics -- see the module-level
            # _STEP_METRIC_KEYS / _TERMINAL_METRIC_KEYS comment for how to read
            # each one. Seeded identically in reset() and reset_to_state().
            **{k: jp.array(0.0, dtype=float)
               for k in _STEP_METRIC_KEYS + _TERMINAL_METRIC_KEYS},
            **{k: jp.array(0.0, dtype=float)
               for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "step": jp.array(0, dtype=jp.int32),
            "success_counter": jp.array(0, dtype=jp.int32),
            # Sticky "this episode has succeeded at least once" latch. Needed
            # because success no longer ends the episode (see done= in step):
            # the raw `success` flag is true on EVERY step the box stays inside
            # success_tol, so summing it would measure DWELL, not success rate.
            # The latch is emitted once, on the terminal step, so the summed
            # eval/episode_success keeps its old 0/1-per-episode meaning and
            # stays comparable with the pre-2026-07-26 ladders.
            "success_ever": jp.array(0.0, dtype=float),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
            "box_rest_z": box_pos[2],
            # D18/D24: never capped here -- target_pos is caller-supplied and
            # bypasses reset()'s sampling and _TARGET_WORLD_Z_CAP entirely (same
            # "bypassed vs not bypassed" convention as the rest of this method).
            "target_z_capped": jp.array(0.0, dtype=float),
            "last_action": jp.zeros(self._nu, dtype=float),
            "dist_min": jp.array(1e3, dtype=float),
            "hold_counter": jp.array(0, dtype=jp.int32),
            # Nominal box half-extents. reset_to_state bypasses the per-episode
            # physics DR draw (see the docstring above), so the identity values
            # are correct here -- EXCEPT when the D17 cube-size probe supplies
            # cube_half_extents, which overrides this a few lines below.
            "box_half": jp.stack(
                [
                    self._dr_identity["half_xy"],
                    self._dr_identity["half_xy"],
                    self._dr_identity["cube_half_z"],
                ]
            ),
            # Caller-supplied start pose (see reset()'s "init_arm_q" comment).
            "init_arm_q": arm_qpos,
            "t_reach": jp.array(int(self._config.episode_length), dtype=jp.int32),
            "t_grasp": jp.array(int(self._config.episode_length), dtype=jp.int32),
            "t_lift": jp.array(int(self._config.episode_length), dtype=jp.int32),
            "align_at_grasp": jp.array(0.0, dtype=float),
            "jaw_span_at_grasp": jp.array(0.0, dtype=float),
            "grip_dist_min": jp.array(1e3, dtype=float),
            "prev_tcp": jp.zeros(3, dtype=float),
        }

        if self._dr_enable:
            info.update(
                {
                    "dr_mass_scale": self._dr_identity["mass_scale"],
                    "dr_fric_scale": self._dr_identity["fric_scale"],
                    # D19: independent xy/z half-extents replace the old scalar.
                    "dr_half_xy": self._dr_identity["half_xy"],
                    "dr_cube_half_z": self._dr_identity["cube_half_z"],
                    "dr_grav": self._dr_identity["grav"],
                    "dr_kp_scale": self._dr_identity["kp_scale"],
                    "dr_kv_scale": self._dr_identity["kv_scale"],
                }
            )
            metrics.update(
                {f"dr/{k}": jp.array(0.0, dtype=float) for k in _DR_METRIC_KEYS}
            )

        ad = self._config.domain_rand.action_delay
        if ad.enable:
            info.update(
                {
                    "action_buffer": jp.zeros(
                        (int(ad.max_delay_steps) + 1, self._nu), dtype=float
                    ),
                    # Identity: no delay (eval-only bypass -- see docstring).
                    "dr_action_delay_steps": jp.array(0, dtype=jp.int32),
                }
            )
            metrics.update({"dr/action_delay_steps": jp.array(0.0, dtype=float)})

        # D20 (2026-07-29): burst-perturbation state for cube_force /
        # joint_torque -- a per-episode counter + the currently-held vector,
        # mirroring the action_delay ring buffer's "STATIC gate -> keys exist
        # only when the axis is enabled" pattern. counter=0 means "no burst
        # active"; the trigger/hold logic itself lives in step(). This is the
        # natural start state, not a special eval bypass: an episode simply
        # begins with no burst in flight.
        cf_init = self._config.domain_rand.cube_force
        if cf_init.enable:
            info.update(
                {
                    "cf_burst_counter": jp.array(0, dtype=jp.int32),
                    "cf_burst_force": jp.zeros(3, dtype=float),
                }
            )
        jt_init = self._config.domain_rand.joint_torque
        if jt_init.enable:
            info.update(
                {
                    "jt_burst_counter": jp.array(0, dtype=jp.int32),
                    "jt_burst_torque": jp.zeros(6, dtype=float),
                }
            )

        on = self._config.domain_rand.obs_noise
        if on.enable:
            info.update(
                {
                    "obs_noise_seed": rng,
                    # Identity: zero BIAS (eval-only bypass); per-step JITTER
                    # still fires in _get_obs -- see docstring.
                    "obs_bias_q": jp.zeros(6, dtype=float),
                    "obs_bias_finger": jp.array(0.0, dtype=float),
                    "obs_bias_box_pos": jp.zeros(3, dtype=float),
                    "obs_bias_quat_vec": jp.zeros(3, dtype=float),
                }
            )

        # D17: OPTIONAL eval-only box-geometry override (cube-size probe).
        # Stashed in `info` (per-episode, not a self._mjx_model mutation) so
        # step() can apply it every substep -- see the docstring above and
        # the matching block in step(). `cube_half_extents is None` (the
        # default, and every non-D17 caller) means this key is simply absent
        # from `info`, which keeps this reset_to_state's pytree structure --
        # and therefore step()'s behaviour -- byte-for-byte identical to
        # before this parameter existed.
        if cube_half_extents is not None:
            info[_DR17_EVAL_CUBE_HALF_EXTENTS_KEY] = jp.asarray(
                cube_half_extents, dtype=float
            )
            # The caller's value MUST win over the nominal seeded above: step()
            # writes cube_half_extents straight into model.geom_size UNCLAMPED
            # (see the matching block there), so leaving box_half at nominal
            # would score the 4 cm E1/E2 probe against 3 cm geometry and make
            # its axis-aware jaw preference silently wrong.
            info["box_half"] = jp.asarray(cube_half_extents, dtype=float)

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)

        # Action-delay DR: execute a buffered PAST action instead of the
        # fresh one (the delay length was sampled ONCE per episode in
        # reset(), held in info["dr_action_delay_steps"]). Push the
        # just-computed RAW action into the ring buffer (index 0 = most
        # recent, pushed every step) and read back the delayed entry for the
        # CTRL computation only -- info["last_action"] (the obs Markov
        # feature + the action_rate reward, both below) stays on the RAW
        # action, matching real deployment: the policy is never told its own
        # command was delayed, only the arm's physical execution lags. STATIC
        # gate -> the off path is byte-for-byte the old
        # `delta = action * self._action_scale`.
        ad = self._config.domain_rand.action_delay
        if ad.enable:
            buf = jp.concatenate(
                [action[None, :], info["action_buffer"][:-1]], axis=0
            )
            info["action_buffer"] = buf
            exec_action = buf[info["dr_action_delay_steps"]]
        else:
            exec_action = action

        delta = exec_action * self._action_scale
        ctrl = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)

        # Per-episode physics DR: rebuild the model from the factors sampled in
        # reset() (stashed in info). STATIC gate -> when off this is exactly
        # `self._mjx_model` and the physics step is unchanged. Under vmap the
        # per-env info scalars make the model per-env automatically.
        model = (
            self._randomize_physics(self._mjx_model, info)
            if self._dr_enable
            else self._mjx_model
        )

        # D17 cube-size probe: OPTIONAL eval-only override, applied AFTER the
        # normal physics-DR path above and UNCLAMPED -- deliberately bypasses
        # _randomize_physics's `_dr_max_box_half_xy` graspability clamp (that
        # clamp guards *training* DR draws; the 4cm eval cube is a one-off OOD
        # override, not a DR draw -- see reset_to_state()'s docstring). Gated
        # on the key's presence in `info`, which is only ever set by
        # reset_to_state(cube_half_extents=...) -- absent for every training
        # rollout and every eval call that doesn't pass the override, so this
        # branch is a true no-op (not even a tree_replace) in those cases.
        if _DR17_EVAL_CUBE_HALF_EXTENTS_KEY in info:
            model = model.tree_replace({
                "geom_size": model.geom_size.at[self._obj_geom].set(
                    info[_DR17_EVAL_CUBE_HALF_EXTENTS_KEY]
                ),
            })

        # Per-step environment DR: a random 3D force on the box (xfrc_applied at
        # its centre of mass), drawn from info["rng"]. STATIC gate -> the off
        # path never touches state.data.xfrc_applied or info["rng"] at all.
        #
        # D20 (2026-07-29) RESPEC: this used to resample a fresh force EVERY step
        # (0.5 N of continuous 50 Hz white noise) -- diagnosed as the likely cause
        # of the L4/L5 training collapse (D14). It is now a BURST: each step has
        # probability `force_prob` of TRIGGERING a new burst if none is active;
        # once triggered the SAME force vector is HELD for `burst_steps` steps,
        # then drops back to zero. Re-triggering while a burst is already active
        # is a NO-OP (hold the vector, do not restart the clock). All state lives
        # in `info` (not a Python closure), so it is jit/vmap-safe -- same pattern
        # as the action_delay ring buffer above.
        data_in = state.data
        cf = self._config.domain_rand.cube_force
        if cf.enable:
            rng_trig, rng_force, rng_next = jax.random.split(info["rng"], 3)
            info["rng"] = rng_next
            active = info["cf_burst_counter"] > 0
            trigger = (
                jax.random.uniform(rng_trig, ()) < float(cf.force_prob)
            ) & jp.logical_not(active)
            new_force = jax.random.uniform(
                rng_force, (3,),
                minval=-float(cf.force_mag), maxval=float(cf.force_mag),
            )
            force = jp.where(trigger, new_force, info["cf_burst_force"])
            counter = jp.where(
                trigger,
                jp.array(int(cf.burst_steps), dtype=jp.int32),
                jp.where(
                    active,
                    info["cf_burst_counter"] - 1,
                    jp.array(0, dtype=jp.int32),
                ),
            )
            info["cf_burst_force"] = force
            info["cf_burst_counter"] = counter
            applied_force = jp.where(counter > 0, force, jp.zeros(3, dtype=float))
            data_in = data_in.replace(
                xfrc_applied=data_in.xfrc_applied.at[self._obj_body, :3].set(
                    applied_force
                )
            )

        # D20 (2026-07-29) NEW axis: arm joint-torque perturbation. Same burst
        # trigger/hold state machine as cube_force above (its OWN independent
        # counter + held vector), applied as an additive qfrc_applied nudge on
        # the 6 arm DOFs. Indexed via self._robot_arm_dofadr (built from
        # mj_model.jnt_dofadr in ur3_base._post_init -- the EXACT dof address,
        # never assumed equal to the qpos address, which it is not once the box
        # freejoint is in the model). STATIC gate -> the off path never touches
        # qfrc_applied.
        jt = self._config.domain_rand.joint_torque
        if jt.enable:
            rng_trig2, rng_torque, rng_next2 = jax.random.split(info["rng"], 3)
            info["rng"] = rng_next2
            active_jt = info["jt_burst_counter"] > 0
            trigger_jt = (
                jax.random.uniform(rng_trig2, ()) < float(jt.force_prob)
            ) & jp.logical_not(active_jt)
            new_torque = jax.random.uniform(
                rng_torque, (6,),
                minval=-float(jt.torque_mag), maxval=float(jt.torque_mag),
            )
            torque = jp.where(trigger_jt, new_torque, info["jt_burst_torque"])
            counter_jt = jp.where(
                trigger_jt,
                jp.array(int(jt.burst_steps), dtype=jp.int32),
                jp.where(
                    active_jt,
                    info["jt_burst_counter"] - 1,
                    jp.array(0, dtype=jp.int32),
                ),
            )
            info["jt_burst_torque"] = torque
            info["jt_burst_counter"] = counter_jt
            applied_torque = jp.where(
                counter_jt > 0, torque, jp.zeros(6, dtype=float)
            )
            data_in = data_in.replace(
                qfrc_applied=data_in.qfrc_applied.at[
                    jp.asarray(self._robot_arm_dofadr)
                ].set(applied_torque)
            )

        data = mjx_env.step(model, data_in, ctrl, self.n_substeps)

        info["step"] = info["step"] + 1

        # action_rate reward reads info["last_action"] (the PREVIOUS action,
        # still un-overwritten at this point) inside _get_reward below; only
        # after the reward is computed do we overwrite it with the current
        # action for the next step.
        raw_rewards, raw_signals = self._get_reward(data, info, action)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }

        # Success is resolved BEFORE the reward sum so the terminal
        # success_bonus can be folded into it. This block used to sit AFTER the
        # sum; moving it up is a pure reordering -- the success logic itself is
        # unchanged, and nothing between here and the old location read `reward`.
        box_pos = data.xpos[self._obj_body]
        box_target_dist = jp.linalg.norm(info["target_pos"] - box_pos)

        success_now = box_target_dist < self._config.success_tol
        info["success_counter"] = jp.where(
            success_now,
            info["success_counter"] + 1,
            jp.array(0, dtype=jp.int32),
        )
        success = info["success_counter"] >= 3

        # Rising-edge success bonus. EDGE-TRIGGERED, not level-triggered: since
        # success no longer terminates the episode, `success` stays true for
        # every step the box remains inside success_tol, so a level-triggered
        # bonus would pay out on hundreds of steps. This fires exactly once per
        # episode regardless of whether success terminates, so the term stays
        # safe under either done= policy. Default scale is 0.0 (not needed once
        # the episode always runs to the horizon) -- see the scales entry.
        newly_succeeded = success & jp.logical_not(info["success_ever"] > 0.5)
        info["success_ever"] = jp.maximum(
            info["success_ever"], success.astype(float)
        )
        raw_rewards["success_bonus"] = newly_succeeded.astype(float)
        rewards["success_bonus"] = (
            raw_rewards["success_bonus"]
            * self._config.reward_config.scales["success_bonus"]
        )

        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
        info["last_action"] = action

        tcp_pos = data.site_xpos[self._gripper_site]
        out_of_bounds = (
            jp.any(jp.abs(tcp_pos[:2]) > 0.6) | (tcp_pos[2] < 0.0)
        )
        invalid_state = (
            jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )
        # NOTE (2026-07-26): `success` deliberately NOT in done. Episodes now
        # always run the full episode_length, matching the real-robot protocol
        # (D16: "Success is NOT a termination criterion"). Previously success
        # ended the episode, which forfeited every remaining step of
        # box_target(20)+hold_target(6)+lift(5)+grasp(3) ~= 31 raw/step. That
        # made succeeding strictly worse than parking in the 5mm..30mm ring, and
        # the 2026-07-24 ladder learned exactly that: eval/episode_success
        # peaked 0.67-0.69 at 60-70% of training then collapsed below 0.03 while
        # episode reward climbed to its maximum. With no termination there is
        # nothing to forfeit, and box_target is strictly maximised at d=0
        # (inside success_tol), so the shaping now points at success instead of
        # away from it -- no compensating bonus required.
        #
        # out_of_bounds / invalid_state stay: the real protocol also aborts on
        # cube-out-of-bounds and hardware faults, so keeping them is parity, not
        # an exception to it.
        done = (out_of_bounds | invalid_state).astype(jp.float32)

        # Non-summed box_target_dist_final/_min diagnostics (see the metrics
        # dict comment in reset()). brax's EvalWrapper.step SUMS every key in
        # State.metrics over the episode as `episode_metrics[k] += metrics[k]`
        # (brax/envs/wrappers/training.py EpisodeWrapper, then EvalWrapper
        # mirrors it), so a plain per-step value reads as a summed proxy
        # dominated by the (long) approach phase, not the real final gap --
        # this was silently misleading in the FIXVERIFY read. Fix: emit these
        # ONLY on the episode's single terminal step (zero elsewhere), so the
        # wrapper's sum equals that one value and the cross-env mean W&B logs
        # is the true mean, in meters. `done` above already covers
        # success|out_of_bounds|invalid_state; `at_horizon` covers the
        # EpisodeWrapper truncation the base env can't otherwise see (relies
        # on EpisodeWrapper setting exactly one done per episode, which it
        # does: `done = steps >= episode_length` OR the env's own done).
        at_horizon = info["step"] >= jp.array(
            int(self._config.episode_length), dtype=info["step"].dtype
        )
        is_terminal = (done > 0.5) | at_horizon
        box_target_dist_final = jp.where(is_terminal, box_target_dist, 0.0)
        box_target_dist_min = jp.where(is_terminal, info["dist_min"], 0.0)

        metrics = state.metrics
        metrics.update(
            **raw_rewards,
            out_of_bounds=out_of_bounds.astype(jp.float32),
            # Emitted ONLY on the terminal step, from the sticky success_ever
            # latch. brax sums metrics over the episode; now that success no
            # longer terminates, the raw per-step flag stays true for every step
            # the box sits inside success_tol, so summing it would report DWELL
            # STEPS, not a success rate -- and would silently jump from ~0.03 to
            # ~100+, breaking comparability with every earlier ladder. This form
            # keeps the old 0/1-per-episode meaning exactly.
            success=jp.where(
                is_terminal, info["success_ever"].astype(jp.float32), 0.0
            ),
            box_target_dist=box_target_dist,
            box_target_dist_final=box_target_dist_final,
            box_target_dist_min=box_target_dist_min,
            reached_box=info["reached_box"],
            grasped=info["grasped"],
            lifted=info["lifted"],
            align_jaw=raw_signals["a_jaw"],
            align_app=raw_signals["a_app"],
            # --- 2026-08 approach/grasp/speed diagnostics ---
            # PER-STEP (brax sums; divide the logged eval/episode_* by
            # episode_length to read a mean). Note align_jaw/align_app above are
            # the PRE-cone raw cosines, so `alignment` here is not reconstructible
            # from them -- it is the post-cone, post-preference value that
            # actually gates the `grasped` latch, which is why it is logged
            # separately.
            grip_box_dist=raw_signals["grip_box_dist"],
            finger_touch_dist=raw_signals["finger_touch_dist"],
            alignment=raw_signals["alignment"],
            jaw_span=raw_signals["span_jaw"],
            tcp_speed=raw_signals["tcp_speed"],
            grasp_gate_blocked=raw_signals["grasp_gate_blocked"],
            # TERMINAL-GATED, same pattern as box_target_dist_* above. The three
            # t_* counters are initialized to episode_length in reset, so a stage
            # that never fired reads as a right-censored episode_length rather
            # than a 0 that would average in as "happened instantly".
            grip_box_dist_min=jp.where(is_terminal, info["grip_dist_min"], 0.0),
            align_at_grasp=jp.where(is_terminal, info["align_at_grasp"], 0.0),
            jaw_span_at_grasp=jp.where(
                is_terminal, info["jaw_span_at_grasp"], 0.0
            ),
            t_reach=jp.where(is_terminal, info["t_reach"].astype(jp.float32), 0.0),
            t_grasp=jp.where(is_terminal, info["t_grasp"].astype(jp.float32), 0.0),
            t_lift=jp.where(is_terminal, info["t_lift"].astype(jp.float32), 0.0),
            # D18/D24 reach-safeguard diagnostic, terminal-gated exactly like
            # box_target_dist_* so the W&B episode-sum is the single per-episode
            # 0/1 and the cross-env mean is the true capping RATE.
            target_z_capped=jp.where(is_terminal, info["target_z_capped"], 0.0),
        )
        # Realized per-episode DR factors, terminal-gated like box_target_dist_*
        # so brax's EvalWrapper episode-SUM equals the single per-episode value
        # (=> the cross-env W&B mean is the true realized mean of each factor).
        if self._dr_enable:
            metrics.update(
                {
                    "dr/cube_mass": jp.where(is_terminal, info["dr_mass_scale"], 0.0),
                    "dr/cube_friction": jp.where(
                        is_terminal, info["dr_fric_scale"], 0.0
                    ),
                    # D19: realized ABSOLUTE half-extents (metres), not a scale.
                    "dr/cube_size_xy": jp.where(
                        is_terminal, info["dr_half_xy"], 0.0
                    ),
                    "dr/cube_size_z": jp.where(
                        is_terminal, info["dr_cube_half_z"], 0.0
                    ),
                    "dr/gravity_z": jp.where(is_terminal, info["dr_grav"][2], 0.0),
                    "dr/arm_stiffness": jp.where(
                        is_terminal, info["dr_kp_scale"], 0.0
                    ),
                    "dr/arm_damping": jp.where(is_terminal, info["dr_kv_scale"], 0.0),
                }
            )
        # action_delay is sampled/gated independently of self._dr_enable (the
        # physics-DR master switch), so it gets its own terminal-gated entry.
        if ad.enable:
            metrics.update(
                {
                    "dr/action_delay_steps": jp.where(
                        is_terminal,
                        info["dr_action_delay_steps"].astype(jp.float32),
                        0.0,
                    ),
                }
            )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """13D base obs + 6D gripper<->box orientation feats + 7D last_action (26D total).

        Appends where the two grasp-relevant gripper axes point IN THE BOX FRAME:
        the jaw axis (finger-separation) and the approach axis (palm->fingertips),
        each projected onto the three box axes. This is the state the policy needs
        to align its frame with the (rotated + tilted) cube, and it is
        reproducible on the real robot: jaw/approach axes come from arm FK, the box
        axes from the mocap-streamed box quaternion. DEPLOYED: `build_obs_from_
        feedback` in robots/UR3e/ur3_realrobot_dependencies.py appends the same 6
        numbers via its own FK pass (`compute_obs_geometry`, reading the tcp/
        finger-touch sites + box xmat of the same scene XML), gated on the
        loaded policy's obs_dim so 13D (legacy) policies are unaffected. Verified
        bit-parity against evaluation/ur3_reward_replay.SimFK.

        Also appends the previous action (7D, zeros on the reset step). This is
        REQUIRED for the action_rate penalty in _get_reward (below) to keep the
        task Markov: the reward now depends on (s, a, a_prev), so a_prev must be
        observable. DEPLOYED: `build_obs_from_feedback` also appends the
        real-robot last_action (raw, pre action_scale) in its 26D path, tracked
        per-tick in `run_policy_loop`.

        Observation noise (domain_rand.obs_noise, STATIC gate): when enabled,
        the arm q / finger / box_pos / box-orientation channels are corrupted
        with a per-episode BIAS + per-step JITTER (sampled in reset() /
        derived here via fold_in) BEFORE this method's formula runs, so the
        whole obs is assembled from mutually consistent noisy quantities --
        never patched onto the finished vector afterward. TCP-side geometry
        (gripper/finger-touch sites -> jaw_axis/app_axis) stays TRUE: it
        mirrors the real arm's own encoder-driven FK, which is accurate.
        Only the channels that are genuinely *measured* on the real robot
        (arm q readback, finger readback, mocap box pose) get noise. The
        REWARD (_get_reward, below) is never affected: it reads `data`/`info`
        directly and never consumes this obs vector.
        """
        l_pos = data.site_xpos[self._left_finger_touch]
        r_pos = data.site_xpos[self._right_finger_touch]
        g_pos = data.site_xpos[self._gripper_site]
        jaw_axis = r_pos - l_pos
        jaw_axis = jaw_axis / (jp.linalg.norm(jaw_axis) + 1e-6)
        app_axis = 0.5 * (l_pos + r_pos) - g_pos
        app_axis = app_axis / (jp.linalg.norm(app_axis) + 1e-6)
        box_axes_true = data.xmat[self._obj_body].reshape(3, 3)  # columns = box axes

        on = self._config.domain_rand.obs_noise
        if on.enable:
            # Per-step jitter is derived deterministically from the
            # per-episode seed + the current step count -- no extra rng
            # threading through step() needed (mirrors how info["step"] is
            # already a stable per-step counter).
            step_key = jax.random.fold_in(info["obs_noise_seed"], info["step"])
            k_q, k_f, k_p, k_r = jax.random.split(step_key, 4)

            q_noisy = (
                data.qpos[self._robot_arm_qposadr]
                + info["obs_bias_q"]
                + jax.random.uniform(
                    k_q, (6,),
                    minval=-float(on.q_jitter), maxval=float(on.q_jitter),
                )
            )
            finger_noisy = (
                data.qpos[self._robot_qposadr[-2:]].sum()
                + info["obs_bias_finger"]
                + jax.random.uniform(
                    k_f, (),
                    minval=-float(on.finger_jitter), maxval=float(on.finger_jitter),
                )
            )
            box_pos_noisy = (
                data.xpos[self._obj_body]
                + info["obs_bias_box_pos"]
                + jax.random.uniform(
                    k_p, (3,),
                    minval=-float(on.box_pos_jitter),
                    maxval=float(on.box_pos_jitter),
                )
            )
            quat_delta = info["obs_bias_quat_vec"] + jax.random.uniform(
                k_r, (3,),
                minval=-float(on.box_quat_jitter),
                maxval=float(on.box_quat_jitter),
            )
            # First-order small-angle rotation perturbation (see _skew); not
            # re-orthonormalized -- only dot products with jaw/app axes below
            # ever consume this matrix, so exact SO(3) membership is not
            # required, and the residual error at these small angles
            # (~1-3 deg) is negligible.
            box_axes = box_axes_true + _skew(quat_delta) @ box_axes_true

            tcp_pos = g_pos  # TRUE: arm FK/encoders, not a mocap channel
            base_obs = jp.concatenate(
                [
                    q_noisy,
                    finger_noisy.reshape(1),
                    box_pos_noisy - tcp_pos,
                    # target_pos is COMMANDED (chosen by us), not measured --
                    # noising it would model nothing, so it stays exact.
                    info["target_pos"] - box_pos_noisy,
                ]
            )
        else:
            base_obs = super()._get_obs(data, info)
            box_axes = box_axes_true

        jaw_proj = jaw_axis @ box_axes  # [jaw·b0, jaw·b1, jaw·b2]
        app_proj = app_axis @ box_axes  # [app·b0, app·b1, app·b2]
        last_action = info["last_action"]
        obs = jp.concatenate([base_obs, jaw_proj, app_proj, last_action])

        # addvelocity: appended LAST so the first 26D of the obs are IDENTICAL
        # (same values, same order) whether or not this is enabled -- a
        # deployed 26D policy's obs-builder can never accidentally pick up
        # trailing velocity dims, and a 33D policy's obs-builder just appends
        # 7 more numbers after everything else. TRUE state always (no noise
        # wired to these channels yet, see default_config's obs_include_velocity
        # docstring) -- exact in sim (data.qvel), matched on real by RTDE
        # getActualQd + a finger finite-difference (see
        # ur3_realrobot_dependencies.py's build_obs_from_feedback).
        if self._config.obs_include_velocity:
            arm_qvel = data.qvel[self._robot_arm_dofadr]        # 6
            finger_qvel = data.qvel[self._robot_finger_dofadr].sum().reshape(1)  # 1
            obs = jp.concatenate([obs, arm_qvel, finger_qvel])

        return obs

    def _get_reward(
        self, data: mjx.Data, info: Dict[str, Any], action: jax.Array
    ) -> Dict[str, Any]:
        # ==============================
        # Postition world-frame
        # ==============================
        # Endposition of the mocap target - JAX array (3,) float64
        target_pos = info[
            "target_pos"
        ]
        # Current position of the box - JAX array (3,) float64
        box_pos = data.xpos[
            self._obj_body
        ]
        # World-frame Cartesian position of the TCP site - JAX array (3,) float64
        gripper_pos = data.site_xpos[
            self._gripper_site
        ]
        # World-frame Cartesian position of the left_finger_touch site - JAX array (3,) float64
        left_finger_touch_pos = data.site_xpos[
            self._left_finger_touch
        ]
        # World-frame Cartesian position of the right_finger_touch site - JAX array (3,) float64
        right_finger_touch_pos = data.site_xpos[
            self._right_finger_touch
        ]

        # ==============================
        # Distance calcluation
        # ==============================
        # Euclidean distance between box and target - scalar JAX float64
        box_target_dist = jp.linalg.norm(
            target_pos - box_pos
        )
        # Closest approach seen this episode (for the non-summed
        # box_target_dist_min diagnostic emitted in step()). Includes the
        # current step before any terminal emit.
        info["dist_min"] = jp.minimum(info["dist_min"], box_target_dist)
        # Euclidean distance between gripper and box - scalar JAX float64
        gripper_box_dist = jp.linalg.norm(
            box_pos - gripper_pos
        )
        # Euclidean distance between the two finger touch sites - scalar JAX float64
        finger_touch_dist = jp.linalg.norm(
            right_finger_touch_pos - left_finger_touch_pos
        )

        # ==============================
        # --- Grasp-frame alignment (2 of 3 axes) ---
        # ==============================
        # MOVED UP (was "Stage 1c", computed after the sticky latches below) so
        # `alignment` is available to GATE the `grasped` latch itself, not just
        # to shape a reward term after the fact -- DIAGHOLD300 (W&B) showed the
        # latch unlocking at 1.5cm+2-pad contact with no alignment requirement,
        # so a misaligned grab still earned the full sticky lift/box_target/hold
        # chain. A parallel-jaw gripper can close on a cube only when its frame
        # lines up with the cube's frame: the jaw axis (finger separation) and
        # the approach axis (palm -> fingertips) must EACH line up with a box
        # face-normal. If those two align, the third gripper axis is forced by
        # orthonormality — so "2 of 3 axes aligned" == fully aligned. Both axes
        # are built from world-frame site positions (no frame/column
        # assumptions); max over the 3 box axes + abs makes the score invariant
        # to the cube's 24-fold (octahedral) symmetry, so every equivalent grasp
        # scores the same.
        #
        # 2026-08: that symmetry argument is FALSE for this geometry, which is
        # why config.align_mode="axis_aware" adds a second tier below. The box
        # is a 3x3x4 cm PRISM, so it has D4h (16-element) symmetry, not
        # octahedral -- its local z is not interchangeable with x/y. Under the
        # 3-axis max, a jaw spanning the 4 cm axis (9.9 mm of total finger
        # clearance against the 49.9 mm opening) scores exactly as well as one
        # spanning a 3 cm axis (19.9 mm), and a horizontal approach scores
        # exactly as well as top-down.
        #
        # Sticky "lifted" as of the PREVIOUS step, read here before the latch
        # block below updates it. Used to switch the axis-aware preferences off
        # once the box is up: they exist to shape the approach and the grasp,
        # and leaving them on would make gripper_align (scale 5.0, and ~fully
        # paid at gripper_box_dist ~= 0 while holding) fight the transport stage
        # every time the wrist has to tilt to reach a high or lateral target.
        was_lifted = info["lifted"]
        jaw_axis = right_finger_touch_pos - left_finger_touch_pos
        jaw_axis = jaw_axis / (jp.linalg.norm(jaw_axis) + 1e-6)
        app_axis = (
            0.5 * (left_finger_touch_pos + right_finger_touch_pos) - gripper_pos
        )
        app_axis = app_axis / (jp.linalg.norm(app_axis) + 1e-6)
        box_axes = data.xmat[self._obj_body].reshape(3, 3)  # columns = box axes
        a_jaw = jp.max(jp.abs(jaw_axis @ box_axes))
        a_app = jp.max(jp.abs(app_axis @ box_axes))
        # Soft alignment cone (60 deg, cos 60 = 0.5), linear ramp to 1 at perfect
        # alignment. WIDENED from the old hard 30-deg cone (0.866): with a 3-axis
        # max, |axis . nearest box axis| is >= 1/sqrt(3) ~ 0.577, so a 0.5 bound
        # keeps the score STRICTLY POSITIVE everywhere -> there is always a
        # gradient pulling the hand toward alignment, no dead zone. The old 0.866
        # cone gave zero reward AND zero gradient until BOTH axes were already
        # inside 30 deg at once (product of two clipped ramps), a chicken-and-egg
        # trap: the policy had to luck into near-perfect alignment before the
        # reward ever turned on. Still a PRODUCT so both axes must improve ("2 of
        # 3 axes aligned"), still proximity-gated below so it can't be farmed.
        jaw_score = jp.clip((a_jaw - _COS_BOUND) / (1.0 - _COS_BOUND), 0.0, 1.0)
        app_score = jp.clip((a_app - _COS_BOUND) / (1.0 - _COS_BOUND), 0.0, 1.0)
        align_face = jaw_score * app_score

        # Box support width along the jaw axis. For a box this is EXACTLY
        # 2 * sum_i h_i * |u . b_i| (the support function of a rectangular
        # cuboid), i.e. how far apart the fingers must be to clear it on this
        # heading. Continuous in orientation -- no argmax, so no cliff and no
        # tie-breaking on a symmetric cube -- and it already folds "is the jaw
        # face-aligned" into "how wide must the jaw open", because any diagonal
        # heading presents a wider span than the narrowest face.
        span_jaw = 2.0 * (info["box_half"] @ jp.abs(jaw_axis @ box_axes))

        if self._align_axis_aware:
            # (a) JAW SPAN preference, in real finger-clearance units.
            #     self._jaw_opening is parsed from the pad geoms (0.0499 m) in
            #     __init__ so it cannot drift from the model. On the nominal
            #     3x3x4 prism:
            #       jaw || a 3 cm axis -> span 0.030 -> clearance 0.0199 -> 1.00
            #       jaw || the 4 cm axis -> span 0.040 -> clearance 0.0099 -> 0.50
            #     i.e. exactly the 19.9 vs 9.9 mm mechanical ratio -- DERIVED
            #     from the gripper, not tuned. Expressed as a ratio against the
            #     best achievable span for THIS episode's box, so it means the
            #     same thing across the whole 2x2x3 .. 4x4x4 cm cube_size DR
            #     range instead of silently rescaling with the draw.
            clear_now = self._jaw_opening - span_jaw
            clear_best = self._jaw_opening - 2.0 * jp.min(info["box_half"])
            jaw_pref = jp.clip(clear_now / (clear_best + 1e-9), 0.0, 1.0)
            # (b) TOP-DOWN approach preference.
            #     SIGN IS LOAD-BEARING AND VERIFIED: the `tcp` site sits 6 mm
            #     PAST the finger pads (mount-local z 0.145 vs 0.139), so
            #     app_axis = fingertip_midpoint - tcp points BACK toward the
            #     palm -- the opposite of what _get_obs's docstring calls
            #     "palm -> fingertips". A top-down grasp therefore has
            #     app_axis[2] ~ +1 (measured +0.807 at the task_home keyframe);
            #     reaching UPWARD gives -1. Do NOT negate this term. The
            #     axis_free scores above are unaffected either way because they
            #     take jp.abs.
            #     WORLD frame on purpose, not the box frame: the box always
            #     rests upright (spawned with a world-Z yaw, flush on a plate
            #     tilted <= ~4 deg), so "along the box's long axis" and
            #     "downward" coincide at nominal -- but they DISAGREE for the
            #     ~25% of cube_size draws where the box is oblate
            #     (xy_half > z_half), and there only the world rule is right:
            #     a box-frame rule would ask for a vertical jaw axis, i.e. one
            #     finger underneath the table.
            app_down = jp.clip(
                (app_axis[2] - _COS_BOUND) / (1.0 - _COS_BOUND), 0.0, 1.0
            )
            # Floor both preferences so neither can zero the product: the whole
            # point of the 0.5 cone above is that there is never a dead zone,
            # and an unfloored preference would reintroduce one (a horizontal
            # approach has app_down == 0 exactly).
            f = float(self._config.align_pref_floor)
            pref = (f + (1.0 - f) * jaw_pref) * (f + (1.0 - f) * app_down)
            alignment = align_face * jp.where(was_lifted > 0.5, 1.0, pref)
        else:
            jaw_pref = jp.array(1.0, dtype=float)
            app_down = jp.array(1.0, dtype=float)
            alignment = align_face
        # Weighted by approach proximity so it shapes the final approach and
        # can't be farmed in free space. Gate WIDENED tanh(5d)->tanh(3d) so this
        # pays earlier in the approach, not only the final ~cm (DIAGHOLD300:
        # gripper_align was ~11% of return and effectively approach-only).
        gripper_align_Reward = alignment * (1 - jp.tanh(3 * gripper_box_dist))

        # ==============================
        # --- Sticky stage latches (monotone via jp.maximum) ---
        # ==============================
        # reached_box: gripper has been within 1.5 cm of the box at some point.
        # Loosened from 1 cm: on a rotated/tilted cube the open fingers straddle
        # the corners, so the TCP can't get to the exact center without perfect
        # alignment; the tight 1 cm gate then silently locks the whole grasp ->
        # lift -> transport chain behind alignment. 1.5 cm lets the pick chain
        # unlock a bit before perfect alignment (the `grasped` latch below still
        # needs real finger-pad contact, so this can't be farmed).
        reach_now = gripper_box_dist < 0.015
        info["t_reach"] = jp.where(
            reach_now & (info["reached_box"] < 0.5), info["step"], info["t_reach"]
        )
        info["reached_box"] = jp.maximum(
            info["reached_box"], reach_now.astype(float)
        )
        # Closest approach seen this episode (approach-phase counterpart of
        # info["dist_min"], which tracks BOX-to-TARGET).
        info["grip_dist_min"] = jp.minimum(
            info["grip_dist_min"], gripper_box_dist
        )
        # grasped: at the box AND both finger pads in contact with it.
        finger_box_contact = [
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._finger_box_found_sensor
        ]
        both_pads_touch = sum(finger_box_contact) >= 2
        # GATED on alignment (DIAGHOLD300 fix): without this, a lucky 2-pad
        # contact on a rotated/tilted cube -- with the gripper frame nowhere
        # near the box's -- still set the sticky `grasped` latch and unlocked
        # the whole downstream lift/box_target/hold chain. Now the box must
        # actually be grasped ALONG A FACE (grasp_align_thresh, soft: see
        # default_config) before the latch can set. `gripper_align_Reward`
        # above still pays a continuous gradient below this bar, so there is
        # no dead zone -- only the sticky latch is hard-gated.
        grasp_now = (
            (info["reached_box"] > 0.5)
            & both_pads_touch
            & (alignment > self._config.grasp_align_thresh)
        )
        # Rising edge, computed BEFORE the jp.maximum below overwrites the latch.
        # align_at_grasp / jaw_span_at_grasp are the direct readout of whether
        # raising grasp_align_thresh actually changed WHICH grasps latch, rather
        # than just how often: expect jaw_span_at_grasp ~= 0.030 (a 3 cm face),
        # not 0.040 (the 4 cm long axis).
        newly_grasped = grasp_now & (info["grasped"] < 0.5)
        info["align_at_grasp"] = jp.where(
            newly_grasped, alignment, info["align_at_grasp"]
        )
        info["jaw_span_at_grasp"] = jp.where(
            newly_grasped, span_jaw, info["jaw_span_at_grasp"]
        )
        info["t_grasp"] = jp.where(newly_grasped, info["step"], info["t_grasp"])
        # TRIPWIRE for a too-tight grasp_align_thresh: the hand IS at the box
        # with both pads in contact, and only the alignment gate is holding the
        # latch shut. If eval/episode_grasp_gate_blocked is large WHILE
        # eval/episode_grasped stays ~0, the bar is too high and the whole
        # lift/box_target/hold chain will never bootstrap -- lower the threshold
        # rather than waiting out a flat 12 h run.
        grasp_gate_blocked = (
            (info["reached_box"] > 0.5) & both_pads_touch & jp.logical_not(grasp_now)
        ).astype(float)
        info["grasped"] = jp.maximum(info["grasped"], grasp_now.astype(float))
        # lifted: a grasped box has cleared its per-episode resting height by
        # lift_eps. Anti-push latch — box_target only pays once this is set.
        box_off_rest = box_pos[2] > (info["box_rest_z"] + self._lift_eps)
        lift_now = box_off_rest & (info["grasped"] > 0.5)
        info["t_lift"] = jp.where(
            lift_now & (info["lifted"] < 0.5), info["step"], info["t_lift"]
        )
        info["lifted"] = jp.maximum(info["lifted"], lift_now.astype(float))

        # LIVE (non-sticky) counterparts: grasped_live/lifted_live reflect
        # CURRENT contact/height only, unlike info["grasped"]/info["lifted"]
        # above which latch true for the rest of the episode once achieved
        # once (jp.maximum). Used below when sticky_latches=False so a dropped
        # box stops earning grasp/lift/box_target/hold_target until it is
        # actually re-grasped -- with sticky_latches=True (default) these are
        # computed but unused, so behavior is unchanged from before this
        # option existed.
        grasped_live = grasp_now.astype(float)
        lifted_live = (box_off_rest & (grasped_live > 0.5)).astype(float)
        _use_sticky = bool(self._config.sticky_latches)  # Python bool at trace time
        grasped_eff = info["grasped"] if _use_sticky else grasped_live
        lifted_eff = info["lifted"] if _use_sticky else lifted_live

        # ==============================
        # --- Reward terms (staged by the latches above) ---
        # ==============================
        # `reached` stays sticky unconditionally (legit "found the box once"
        # signal, not a holding signal -- see sticky_latches docstring).
        reached = info["reached_box"]
        grasped = grasped_eff
        lifted = lifted_eff

        # Gripper open/closed in [0, 1] from the finger-tip separation.
        finger_open = jp.tanh(finger_touch_dist / 0.05)
        finger_closed = 1 - finger_open

        # Stage 1 — approach (always on): gripper moves onto the box. Two-scale
        # (coarse tanh*5 for a long-range pull + fine tanh*30 for a steep
        # near-goal gradient) so the policy is rewarded for closing the LAST few
        # mm, not just getting within ~1 cm. The old single tanh*5 was nearly flat
        # inside 1 cm (reward ~0.95 at 10 mm), so the TCP settled off-center and a
        # 4 cm cube (5 mm finger clearance per side) got one-fingered.
        #
        # 2026-07-29: promoted to a THREE-scale cascade, mirroring box_target's
        # own fix below exactly (coarse tanh*1.5 pulls from long range, mid
        # tanh*5 covers the old formula's working band, fine tanh*30 drives the
        # last cm). The two-scale version above (tanh*5 + tanh*30) is the SAME
        # shape that box_target used to have and was already diagnosed as going
        # flat past ~0.4 m (W&B Spheretarget_mid_30M) -- at 0.5 m,
        # 1-tanh(5*0.5) ~= 0.013, essentially zero gradient. box_target got the
        # coarse-scale fix; gripper_box did not, and the same symptom (slow
        # approach at long range) showed up here too. /3.0 keeps max=1 at d=0,
        # so gripper_box=4.0's scale is unchanged.
        gripper_box_Reward = (
            (1 - jp.tanh(1.5 * gripper_box_dist))
            + (1 - jp.tanh(5.0 * gripper_box_dist))
            + (1 - jp.tanh(30.0 * gripper_box_dist))
        ) / 3.0

        # Stage 2 — grasp: close the fingers on the box once reached. No
        # near-target fade (unlike picknplace) — this task holds the box AT the
        # air target rather than releasing it, so the grasp must stay rewarded.
        # Shaped by `alignment` (continuous, in addition to the hard gate on the
        # `grasped` latch above) so closing the fingers pays more when the
        # gripper frame actually lines up with the box -- not just when reached.
        grasp_Reward = finger_closed * reached * alignment

        # Stage 1b — approach OPEN: reward open fingers until the box is reached,
        # so the arm arrives ready to grasp instead of bumping the box with a
        # closed hand. Gated by (1 - reached) so it switches off exactly when
        # `grasp` switches on. One-sided V1 form (scale 1.0). NOTE: do NOT boost
        # this above grasp(3.0) — the two-sided @6.0 v2 variant reward-hacked by
        # hovering open just outside the 1 cm reach latch (W&B: reached_box → ~0,
        # lifted = 0), because not-reaching kept the fat open bonus alive.
        approach_open_Reward = finger_open * (1 - reached)

        # Stage 3 — lift: raise the box off its resting height (saturates ~12 cm).
        lift_height = jp.clip(box_pos[2] - info["box_rest_z"], 0.0, 0.12)
        lift_Reward = jp.tanh(lift_height / 0.06) * reached

        # Stage 4 — transport: box to the target. THREE-scale cascade so the
        # pull never vanishes across the full base_polar carry (~0.5 m -> 5 mm):
        # coarse tanh*1.5 pulls from ~0.5 m, mid tanh*6 covers 0.05-0.3 m, fine
        # tanh*40 drives the last cm to the 5 mm success gate. The old two-scale
        # (tanh*5 + tanh*30) went flat past ~0.4 m, so far targets sat in a
        # gradient dead-zone and the box plateaued ~19 cm short (W&B
        # Spheretarget_mid_30M). /3 keeps max=1 at d=0 so the box_target=20.0
        # scale is unchanged. GATED by "lifted" so a floor-pushed box earns 0.
        box_target_Reward = (
            (1 - jp.tanh(1.5 * box_target_dist))
            + (1 - jp.tanh(6.0 * box_target_dist))
            + (1 - jp.tanh(40.0 * box_target_dist))
        ) / 3.0 * lifted

        # Stage 5 — hold: pay for KEEPING a lifted box inside hold_radius of
        # the target (settle-and-stay), not just tapping the point once.
        # in_hold requires the CURRENT lifted state (grasped_eff/lifted_eff,
        # respects sticky_latches) so a drop -- even under sticky_latches
        # where box_target keeps paying via the sticky "lifted" -- still zeros
        # this term and its dwell counter immediately, giving a drop penalty
        # regardless of the sticky_latches setting. Dwell counter ramps a
        # tanh bonus so held-longer > tapped-once; resets to 0 the instant the
        # box leaves the radius.
        in_hold = (lifted > 0.5) & (box_target_dist < self._config.hold_radius)
        info["hold_counter"] = jp.where(
            in_hold, info["hold_counter"] + 1, jp.array(0, dtype=jp.int32)
        )
        hold_target_Reward = in_hold.astype(float) * jp.tanh(
            info["hold_counter"] / self._config.hold_tau
        )

        # Penalty for deviating too far from the initial arm configuration.
        # Gated by (1 - lifted): this term existed to keep the arm from
        # wandering during approach/grasp, but once the box is lifted, the
        # transport stage (box_target, above) NEEDS the arm to move away from
        # its init pose to reach the (raised) target -- an un-gated version
        # actively fights that motion. box_target_dist floored around 30mm /
        # success ~3% on the 20260709 speedtest runs partly because of this.
        #
        # 2026-08 REFERENCE FIX: this measured against self._init_q -- the
        # task_home KEYFRAME -- but episodes do not start there. They start from
        # a pose drawn out of init_poses/train/mid.json (init_start_random="mid",
        # the default) or from the keyframe plus init_qpos_noise, whose wrist_3
        # amplitude is a full 6.28 rad. So the term was paying the policy to
        # drive toward a pose the episode never started at, and with the wrist
        # contribution dominating the norm it sat tanh-saturated (~0) for most
        # of the approach anyway. info["init_arm_q"] is this episode's ACTUAL
        # start pose; evaluation/ur3_reward_replay.py:201 already computed it
        # that way on the real side, so this also closes a sim/replay parity gap
        # that was silently biasing the regularizer stage of every retention
        # number.
        #
        # NOTE this is a BONUS despite the name: the scale is POSITIVE and the
        # raw signal is maximal at zero deviation, so it pays the arm to stay
        # put. Combined with action_rate it is what made long-range motion net
        # negative (measured: -0.163/tick sustained vs a +0.023/tick pull from
        # gripper_box at d=0.40 m). Hence the default scale drop to 0.05.
        robot_target_qpos_penalty = (
            1
            - jp.tanh(
                jp.linalg.norm(
                    data.qpos[self._robot_arm_qposadr] - info["init_arm_q"]
                )
            )
        ) * (1 - lifted)

        # TCP speed diagnostic (m/s). Not a reward term -- purely so the
        # approach phase is observable in W&B, which it previously was not:
        # nothing about speed, time-to-stage, or gripper-box distance was ever
        # logged. First step is guarded because info["prev_tcp"] is seeded to
        # zeros (mjx_env.make_data does not run forward kinematics, so there is
        # no meaningful TCP position at reset time to seed from).
        #
        # The guard is `<= 1`, NOT `< 1`: step() does info["step"] += 1 BEFORE
        # calling _get_reward, so the very first reward evaluation of an episode
        # already sees step == 1. With `< 1` the guard never fired and step 1
        # reported |gripper_pos - 0| / dt, i.e. ~21 m/s instead of 0.
        # (The t_reach/t_grasp/t_lift counters share this 1-based convention:
        # a stage that fires on the first step records 1, and "never" stays at
        # the episode_length seeded in reset.)
        tcp_speed = jp.where(
            info["step"] <= 1,
            0.0,
            jp.linalg.norm(gripper_pos - info["prev_tcp"]) / self.dt,
        )
        info["prev_tcp"] = gripper_pos

        # Action-rate penalty: discourage large deltas between consecutive
        # actions (DeXtreme-style smoothness term). Raw value is a squared L2
        # norm over the full 7D action (arm + gripper); the scale (negative)
        # turns it into a penalty. Targets the visible shaking in rollout
        # videos that knocks the box loose -- action_scale sets the jitter's
        # AMPLITUDE, but nothing previously penalized the jitter itself.
        last_action = info["last_action"]
        action_rate_Reward = jp.sum(jp.square(action - last_action))

        # Floor collision via touch sensors (same as Panda).
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._floor_hand_found_sensor
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision_Reward = (1 - floor_collision).astype(float)

        rewards = {
            "gripper_box": gripper_box_Reward,
            "approach_open": approach_open_Reward,
            "grasp": grasp_Reward,
            "lift": lift_Reward,
            "box_target": box_target_Reward,
            "hold_target": hold_target_Reward,
            "gripper_align": gripper_align_Reward,
            "no_floor_collision": no_floor_collision_Reward,
            "robot_target_qpos": robot_target_qpos_penalty,
            "action_rate": action_rate_Reward,
        }
        # ==============================
        # Raw signals dict (for debug)
        # ==============================
        raw = {
            # positions
            "target_pos": target_pos,
            "box_pos": box_pos,
            "gripper_pos": gripper_pos,
            "left_finger_touch_pos": left_finger_touch_pos,
            "right_finger_touch_pos": right_finger_touch_pos,

            # errors / distances
            "box_target_dist": box_target_dist,
            "grip_box_dist": gripper_box_dist,
            "finger_touch_dist": finger_touch_dist,

            # grasp-frame alignment (2-of-3-axis face score, then the
            # axis_aware preference factors; both are 1.0 under axis_free so
            # the keys exist unconditionally and the pytree stays stable)
            "a_jaw": a_jaw,
            "a_app": a_app,
            "align_face": align_face,
            "jaw_pref": jaw_pref,
            "app_down": app_down,
            "span_jaw": span_jaw,
            "alignment": alignment,

            # speed / gate diagnostics
            "tcp_speed": tcp_speed,
            "grasp_gate_blocked": grasp_gate_blocked,

            # events
            "reached_box": info["reached_box"],
            "grasped": info["grasped"],
            "lifted": info["lifted"],
            "number_floor_collision": floor_collision,
        }

        return rewards, raw
