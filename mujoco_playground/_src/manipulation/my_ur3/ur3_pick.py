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
Optionally the box spawns on a variable-height "lifter" plate (lifter_height_max)
so the policy learns to grasp boxes at different heights, and the arm start pose
can be drawn from a library of hand-collected real-robot poses (init_start_random).
Mirrors the commented-out reward scaffolding of ur10pick.py.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.my_ur3 import ur3_base
from mujoco_playground._src.manipulation.my_ur3.init_poses import load_init_poses
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

# Lifter (box-riser) geometry. A thin static plate placed under the box at reset;
# its top surface sets the box's starting height so the policy learns to grasp
# boxes at different heights. Min height keeps the plate bottom above the floor
# plane (z=0) so it never overlaps the floor (masks alone can't separate them).
_LIFTER_HALF_THICKNESS = 0.0025  # 5 mm plate -> half-extent
_LIFTER_HEIGHT_MIN = 0.003
_BOX_HALF_EXTENT = 0.02  # half the box HEIGHT (3x3x4 cm box, 4 cm tall) -> rest offset

# Per-episode domain-randomization factors logged to W&B (terminal-gated, see
# step()). Realized scale per axis + the gravity z-component.
_DR_METRIC_KEYS = (
    "cube_mass",
    "cube_friction",
    "cube_size",
    "gravity_z",
    "arm_stiffness",
    "arm_damping",
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
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to initial pose. Gated by (1-lifted) in
                # _get_reward so it only regularizes the pre-lift approach —
                # once the box is lifted this stopped fighting the transport
                # stage, which needs the arm to move AWAY from its init pose
                # to reach the (raised) target. Was un-gated and always-on;
                # box_target_dist floored at ~30mm / success ~3% on the
                # 20260709 speedtest runs, partly because this term penalized
                # exactly the motion transport requires.
                robot_target_qpos=0.3,
                # Penalize large action deltas between consecutive steps
                # (DeXtreme-style smoothness term) to stop the visible
                # shaking/jitter in rollout videos that knocks the box loose.
                # Raw signal is a squared L2 norm (unbounded, unlike the other
                # tanh-saturated terms here) so keep this scale small relative
                # to the others; NEGATIVE scale turns the raw magnitude into a
                # penalty (same sign convention `franka_emika_panda_robotiq`
                # and `leap_hand` use for their action_rate terms).
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
            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="task_home",
        # Per-joint per-direction amplitude (rad) for reset randomization: 6 arm
        # + 1 finger. Applied symmetrically as uniform(-v, +v) on top of the init
        # keyframe. Default 0.05 reproduces the legacy uniform(-0.05, 0.05) arm noise.
        init_qpos_noise=(0.05, 0.05, 0.05, 0.05, 0.05, 6.28319, 0.0),
        # Arm/finger start-pose source. "none" = literal keyframe start (then +
        # init_qpos_noise jitter); "light"/"mid"/"hard" = randomly pick one
        # hand-collected pose from init_poses/train/<level>.json each reset.
        init_start_random="mid",
        # Per-episode box-riser height (m). 0.0 disables the lifter (box on the
        # floor, legacy). >0 => h ~ uniform(_LIFTER_HEIGHT_MIN, lifter_height_max)
        # and the box spawns resting on the plate; the lift target is raised by h.
        lifter_height_max=0.02,
        # Per-episode SLIGHT plate tilt (rad). roll AND pitch are sampled
        # INDEPENDENTLY, each ~ uniform(-t, +t) about world X and world Y, then
        # composed — so both axes tilt at once and the worst-case surface-normal
        # tilt is ~sqrt(2)*t when both hit their extremes. The box rests FLUSH on
        # the tilted plate, so it starts both at a variable height and slightly
        # tilted. This is what makes the out-of-plane (approach-axis) component of
        # the 2-of-3-axis grasp alignment matter. 0.0 => flat plate (legacy).
        # Only active with the lifter enabled.
        # Reasonable values (per axis; remember the ~1.41x combined worst case):
        #   0.00  => flat (baked default here)
        #   0.05  => mild,     ~2.9 deg  (~4.0 deg combined)  -- gentle start
        #   0.08  => moderate, ~4.6 deg  (~6.5 deg combined)  -- prior default
        #   0.10  => strong,   ~5.7 deg  (~8.1 deg combined)  -- stickyoff "hard"
        # Keep < ~0.12 rad (~6.9 deg / ~9.7 deg combined) so the cube can't
        # slide/tip off the plate before the grasp.
        lifter_tilt_max=0.08,  # flat plate (baked from Spheretarget_mid_30M)
        # Box spawn yaw about world Z (rad); yaw ~ uniform(-r, +r). pi/4 covers
        # all yaw thanks to the cube's 4-fold symmetry, so the policy must learn
        # to match the jaw axis to a face rather than getting a free alignment.
        box_z_rot_range=6.28319,  # 2*pi (full-rotation yaw coverage)
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
        box_xy_jitter=(0.15, 0.2),
        # Lift-target Z-band jitter (m): target_z ~ uniform(*target_z_jitter) +
        # _init_obj_pos[2] (+ lifter_h). Was a hardcoded [0.18, 0.21] literal,
        # shared verbatim by BOTH target_mode paths ("box" and "base_polar");
        # exposed here for the same reason as box_xy_jitter. A zero-width
        # tuple (e.g. (0.195, 0.195)) makes the goal height deterministic.
        target_z_jitter=(0.18, 0.21),
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
        #                unchanged (fixed 0.18‥0.21 + lifter_h).
        target_mode="base_polar",
        # Horizontal radius from the ROBOT BASE for base_polar targets (m).
        target_r_min=0.25,
        # Keep the far target reachable WHILE GRASPING a box. At 0.48 the worst
        # target was 0.52-0.54 m from base (at/over the ~0.54 m reach BEFORE the
        # grasp consumes any), so far targets were unreachable and the box
        # plateaued. 0.42 -> worst ~0.47-0.49 m, ~6-7 cm headroom for the grasp;
        # drop to 0.40 if success stays 0. Still a real carry (chord up to
        # ~0.40 m at the 60 deg azimuth cap).
        target_r_max=0.42,
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
        # Minimum grasp-frame alignment (jaw_score * app_score, each in [0,1])
        # required for the `grasped` sticky latch to set -- blocks the "grab
        # while misaligned" shortcut that let a lucky 2-pad contact on a
        # rotated/tilted cube unlock the whole sticky lift/box_target/hold
        # chain regardless of alignment (DIAGHOLD300 W&B finding). Soft:
        # gripper_align (reward_config.scales, above) still pays a continuous
        # gradient below this bar, so there is no dead zone -- only the LATCH
        # is hard-gated. 0.3 ~= both jaw and approach axes within ~40 deg of
        # a box face-normal. Lower toward 0.2 if `grasped` fails to unlock at
        # all early in training.
        grasp_align_thresh=0.3,
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
            # Cube size: x nominal geom_size[box] half-extents. Hard-clamped to a
            # graspable width (fingers open ~5 cm; 3 cm cube -> ~1 cm/side) so DR
            # can never produce an ungraspable box. [0.9,1.1]x. NOTE: the sampled
            # Z half-extent also re-seats the box on the (lifter) plate in reset().
            cube_size=config_dict.create(enable=False, min=0.9, max=1.1),
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
            # Per-STEP random perturbation force (xfrc_applied) on the box body.
            # A fresh 3D force ~ uniform(-force_mag, force_mag) N per axis is
            # drawn EVERY step (not held per-episode) from info["rng"] and
            # applied at the box's center of mass -- see step(). Magnitude is
            # UNCALIBRATED (no measured real perturbation to center on; see
            # "Plan - Sim-to-Real Gap Protocol" C4/blind-randomization note) --
            # 0.5 N is ~1.4x the box's own weight (~0.036 kg * 9.81 ~= 0.35 N),
            # a deliberately noticeable nudge, purely exploratory.
            cube_force=config_dict.create(enable=False, force_mag=0.5),
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

        # Floor-collision sensors (Hand-E fingers + hand capsule vs floor).
        self._floor_hand_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_pad_floor_found",
                "right_finger_pad_floor_found",
                "hand_capsule_floor_found",
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

        # Variable-height lifter plate. Resolved here (not in the shared
        # ur3_base._post_init) because the picknplace sibling loads a scene
        # without a lifter body. Disabled (parked at its XML default pose) when
        # lifter_height_max <= 0.
        self._lifter_enabled = float(self._config.lifter_height_max) > 0.0
        self._lifter_mocap = self._mj_model.body("lifter").mocapid

        # Robot base world XY, SOURCED FROM THE MODEL (not assumed to be the
        # world origin). The UR3 "base" body is a direct child of worldbody, so
        # its body.pos IS its world position at qpos0. Used by the "base_polar"
        # target_mode to sample lift targets in an annulus around the base; stays
        # correct if the base is ever repositioned in the scene XML.
        self._robot_base_xy = jp.asarray(
            self._mj_model.body("base").pos[:2], dtype=float
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
        # the configured cube_size range: fingers open ~5 cm, so keep the width
        # half-extent <= 0.018 (3.6 cm). Height (z) is unclamped.
        self._dr_max_box_half_xy = 0.018
        # Identity DR factors for the OFF path (same keys as _sample_physics_dr).
        self._dr_identity = {
            "mass_scale": jp.array(1.0, dtype=float),
            "fric_scale": jp.array(1.0, dtype=float),
            "size_scale": jp.array(1.0, dtype=float),
            "grav": self._dr_nom_gravity,
            "kp_scale": jp.array(1.0, dtype=float),
            "kv_scale": jp.array(1.0, dtype=float),
            "cube_half_z": jp.array(self._dr_nom_half_z, dtype=float),
        }

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
        if dr.cube_size.enable:
            rng, k = jax.random.split(rng)
            s = jax.random.uniform(
                k, (), minval=float(dr.cube_size.min), maxval=float(dr.cube_size.max)
            )
            out["size_scale"] = s
            # Z half-extent used by reset() to re-seat the box on the plate/floor.
            out["cube_half_z"] = self._dr_nom_half_z * s
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
        if dr.cube_size.enable:
            # Uniform half-extent scale; width (xy) hard-clamped graspable.
            new_size = self._dr_nom_size * info["dr_size_scale"]
            new_size = new_size.at[:2].set(
                jp.minimum(new_size[:2], self._dr_max_box_half_xy)
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
        # cube_size axis is on).
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

        # initialize box XY — wide jitter, X-heavy, so the box spawns anywhere
        # in X∈[0.30,0.60] (all graspable; 0.60 was the old nominal). Y-center
        # shiftable via box_y_center_offset (0.0 = legacy, centered on the XML
        # keyframe) so a sweep can push the spawn to one side of the target.
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

        # Box yaw about world Z (±box_z_rot_range). The cube is symmetric so this
        # is the dominant spawn DOF; the slight plate tilt below adds the
        # out-of-plane component that makes the full 2-of-3-axis grasp alignment
        # (not just yaw) matter.
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
            lifter_h = jax.random.uniform(
                rng_lift,
                (),
                minval=_LIFTER_HEIGHT_MIN,
                maxval=float(self._config.lifter_height_max),
            )
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
            # The plate is pinned at the nominal box XY (below); its top plane is
            # raised by the half-thickness. Solve the plane height under the
            # (jittered) box XY, then lift the box center a half-extent along the
            # plate normal so the cube sits flush.
            p_top_z = lifter_h + _LIFTER_HALF_THICKNESS
            z_plane = p_top_z - (
                n[0] * (box_xy[0] - self._init_obj_pos[0])
                + n[1] * (box_xy[1] - self._init_obj_pos[1])
            ) / n[2]
            box_z = z_plane + box_half_z * n[2]
        else:
            lifter_h = jp.array(0.0, dtype=float)
            q_tilt = jp.array([1.0, 0.0, 0.0, 0.0])
            box_z = box_half_z  # rest at half-height (== 0.02 unless cube_size DR)

        box_pos = jp.array([box_xy[0], box_xy[1], box_z])
        # Box orientation: plate tilt composed with the yaw spin about the plate
        # normal, so the cube rests flush on the (possibly tilted) plate.
        box_quat = _quat_mul(q_tilt, q_yaw)
        lifter_quat = q_tilt

        # initialize target position — a lift point in the air above the box.
        # X biased forward (+0.01‥+0.06) so the lift pulls the box toward the
        # robot's reachable front; Y pulled in (±0.03) so the lift stays near
        # the robot's sagittal plane. Z band RAISED to 0.18‥0.21 m (was
        # 0.12‥0.15) for a clearly higher lift, on request. WARNING: the far
        # corner (x≈0.51, world-z≈0.23‥0.25 with the lifter) now sits AT or
        # slightly OVER the UR3's ~0.54 m 3D reach limit, so the hardest targets
        # may be physically unreachable — watch success/box_target_dist; pull the
        # MAX back toward 0.18 if success collapses. Raised further by the lifter
        # height so the lift target stays above the (possibly raised) box.
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
            target_pos = target_pos.at[2].add(lifter_h)
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
            # fixed 0.18‥0.21 band above the keyframe box height, raised by the
            # lifter height. Only the XY changes here. For a true reach SPHERE,
            # replace this fixed Z-band with an elevation angle (target_z from r
            # and an elevation draw) at this spot.
            target_z = (
                jax.random.uniform(
                    rng_tz, (), minval=tz_jitter[0], maxval=tz_jitter[1]
                )
                + self._init_obj_pos[2]
            )
            target_pos = jp.array([target_xy[0], target_xy[1], target_z])
            target_pos = target_pos.at[2].add(lifter_h)

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

        # set target mocap (lift-goal marker); pin the lifter plate at the
        # nominal box XY when enabled so the 40 cm plate covers the jittered
        # box yet keeps its near edge 10 cm off the base (else parked at XML).
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )
        if self._lifter_enabled:
            # XY pinned to the nominal box pos (not the jittered box_xy) so the
            # plate never reaches the base; only the height varies per episode.
            lifter_pos = jp.array(
                [self._init_obj_pos[0], self._init_obj_pos[1], lifter_h]
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
            **{k: jp.array(0.0, dtype=float)
               for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "step": jp.array(0, dtype=jp.int32),
            "success_counter": jp.array(0, dtype=jp.int32),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
            # Per-episode box resting height (top of the lifter plate, or the
            # keyframe floor Z). The "lifted" latch measures lift against this.
            "box_rest_z": box_z,
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
        }

        # Stash the per-episode DR factors in info (read by _randomize_physics in
        # step()) and seed their W&B metric keys. STATIC gate -> these keys exist
        # only when DR is on, keeping the State pytree (and the off path) clean.
        if self._dr_enable:
            info.update(
                {
                    "dr_mass_scale": dr["mass_scale"],
                    "dr_fric_scale": dr["fric_scale"],
                    "dr_size_scale": dr["size_scale"],
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

        # Per-step environment DR: a fresh random 3D force on the box, drawn
        # from info["rng"] and re-split EVERY step (unlike action_delay, this
        # is NOT held per-episode -- it models a continuous small
        # "environment disturbance", e.g. a bumped table, not a one-off
        # event). STATIC gate -> the off path never touches
        # state.data.xfrc_applied or info["rng"] at all.
        cf = self._config.domain_rand.cube_force
        if cf.enable:
            rng_force, rng_next = jax.random.split(info["rng"])
            info["rng"] = rng_next
            force = jax.random.uniform(
                rng_force, (3,),
                minval=-float(cf.force_mag), maxval=float(cf.force_mag),
            )
            data_in = state.data.replace(
                xfrc_applied=state.data.xfrc_applied.at[self._obj_body, :3].set(
                    force
                )
            )
        else:
            data_in = state.data

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
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
        info["last_action"] = action

        box_pos = data.xpos[self._obj_body]
        box_target_dist = jp.linalg.norm(info["target_pos"] - box_pos)

        success_now = box_target_dist < self._config.success_tol
        info["success_counter"] = jp.where(
            success_now,
            info["success_counter"] + 1,
            jp.array(0, dtype=jp.int32),
        )
        success = info["success_counter"] >= 3

        tcp_pos = data.site_xpos[self._gripper_site]
        out_of_bounds = (
            jp.any(jp.abs(tcp_pos[:2]) > 0.6) | (tcp_pos[2] < 0.0)
        )
        invalid_state = (
            jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )
        done = (success | out_of_bounds | invalid_state).astype(jp.float32)

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
            success=success.astype(jp.float32),
            box_target_dist=box_target_dist,
            box_target_dist_final=box_target_dist_final,
            box_target_dist_min=box_target_dist_min,
            reached_box=info["reached_box"],
            grasped=info["grasped"],
            lifted=info["lifted"],
            align_jaw=raw_signals["a_jaw"],
            align_app=raw_signals["a_app"],
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
                    "dr/cube_size": jp.where(is_terminal, info["dr_size_scale"], 0.0),
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
        return jp.concatenate([base_obs, jaw_proj, app_proj, last_action])

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
        _cos_bound = 0.5
        jaw_score = jp.clip((a_jaw - _cos_bound) / (1.0 - _cos_bound), 0.0, 1.0)
        app_score = jp.clip((a_app - _cos_bound) / (1.0 - _cos_bound), 0.0, 1.0)
        alignment = jaw_score * app_score
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
        info["reached_box"] = jp.maximum(
            info["reached_box"],
            (gripper_box_dist < 0.015).astype(float),
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
        info["grasped"] = jp.maximum(info["grasped"], grasp_now.astype(float))
        # lifted: a grasped box has cleared its per-episode resting height by
        # lift_eps. Anti-push latch — box_target only pays once this is set.
        box_off_rest = box_pos[2] > (info["box_rest_z"] + self._lift_eps)
        lift_now = box_off_rest & (info["grasped"] > 0.5)
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
        # 4 cm cube (5 mm finger clearance per side) got one-fingered. Mirrors the
        # box_target shaping.
        gripper_box_Reward = (
            0.5 * (1 - jp.tanh(5 * gripper_box_dist))
            + 0.5 * (1 - jp.tanh(30 * gripper_box_dist))
        )

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
        robot_target_qpos_penalty = (
            1
            - jp.tanh(
                jp.linalg.norm(
                    data.qpos[self._robot_arm_qposadr]
                    - self._init_q[self._robot_arm_qposadr]
                )
            )
        ) * (1 - lifted)

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

            # grasp-frame alignment (2-of-3-axis)
            "a_jaw": a_jaw,
            "a_app": a_app,
            "alignment": alignment,

            # events
            "reached_box": info["reached_box"],
            "grasped": info["grasped"],
            "lifted": info["lifted"],
            "number_floor_collision": floor_collision,
        }

        return rewards, raw
