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
        episode_length=250,
        action_repeat=1,
        # Arm per-step ctrl delta = action * action_scale (swept per run). The
        # gripper is DECOUPLED via gripper_action_scale below so it can be kept
        # slow (stays open on approach) while the arm runs faster.
        action_scale=0.025,
        # Separate per-step scale for the gripper actuator (last ctrl dim). Small
        # + fixed so the hand can't snap shut in one step — full open->close
        # travel is 0.025, which needs >=2.5 steps at 0.01. This is what keeps
        # the hand open on approach independent of the swept arm action_scale.
        gripper_action_scale=0.01,
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
                box_target=8.0,
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
                action_rate=-0.01,
                # Sustained-proximity bonus: pay for KEEPING a lifted box inside
                # the target sphere, ramping with dwell time so the policy
                # settles and holds instead of tapping the point and drifting
                # off. Resets to 0 the instant the box leaves the radius (a drop
                # kills it), so it doubles as a drop penalty. Gated by `lifted`
                # and tanh-capped so it cannot out-pay box_target(8.0)/grasp(3.0)
                # -- the reward-hacking guard: it must never be farmable without
                # an actual lifted box held at the goal. Added because
                # box_target only rewards distance-to-point, nothing rewarded
                # HOLDING it there (FIXVERIFY: box visibly enters the 4cm target
                # sphere but drifts/drops with no recovery pressure).
                hold_target=3.0,
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
        # Per-episode SLIGHT plate tilt (rad). roll,pitch ~ uniform(-t, +t) about
        # world X and Y; the box rests FLUSH on the tilted plate, so it starts
        # both at a variable height and slightly tilted. This is what makes the
        # out-of-plane (approach-axis) component of the 2-of-3-axis grasp
        # alignment matter. 0.0 => flat plate (legacy). Only active with the
        # lifter enabled. Keep small (< ~0.12 rad) so the cube can't slide/tip.
        lifter_tilt_max=0.0,  # flat plate (baked from Spheretarget_mid_30M)
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
        # Keep <= UR3 ~0.54 m reach so base_polar targets stay reachable.
        target_r_max=0.48,
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

        # initialize box XY — wide jitter, X-heavy, so the box spawns anywhere
        # in X∈[0.30,0.60] (all graspable; 0.60 was the old nominal). Y-center
        # shiftable via box_y_center_offset (0.0 = legacy, centered on the XML
        # keyframe) so a sweep can push the spawn to one side of the target.
        box_xy = (
            jax.random.uniform(
                rng_box,
                (2,),
                minval=jp.array([-0.15, -0.2]),
                maxval=jp.array([0.15, 0.2]),
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
            box_z = z_plane + _BOX_HALF_EXTENT * n[2]
        else:
            lifter_h = jp.array(0.0, dtype=float)
            q_tilt = jp.array([1.0, 0.0, 0.0, 0.0])
            box_z = self._init_obj_pos[2]  # legacy on-floor (0.02)

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
        if self._config.target_mode == "box":
            target_pos = (
                jax.random.uniform(
                    rng_target,
                    (3,),
                    minval=jp.array([0.02, -0.03, 0.18]),
                    maxval=jp.array([0.06, 0.03, 0.21]),
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
                jax.random.uniform(rng_tz, (), minval=0.18, maxval=0.21)
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
        finger_sample = jax.random.uniform(
            rng_gripper, (), minval=0.0, maxval=0.025
        )
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

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info = dict(state.info)
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
        """
        base_obs = super()._get_obs(data, info)
        l_pos = data.site_xpos[self._left_finger_touch]
        r_pos = data.site_xpos[self._right_finger_touch]
        g_pos = data.site_xpos[self._gripper_site]
        jaw_axis = r_pos - l_pos
        jaw_axis = jaw_axis / (jp.linalg.norm(jaw_axis) + 1e-6)
        app_axis = 0.5 * (l_pos + r_pos) - g_pos
        app_axis = app_axis / (jp.linalg.norm(app_axis) + 1e-6)
        box_axes = data.xmat[self._obj_body].reshape(3, 3)  # columns = box axes
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

        # Stage 4 — transport: box to the target. Two-scale (coarse tanh*5 keeps a
        # long-range pull, fine tanh*30 adds a steep near-goal gradient so the box
        # is pulled the last ~10 mm) and GATED by "lifted" so a box pushed along
        # the floor earns nothing.
        box_target_Reward = (
            0.5 * (1 - jp.tanh(5 * box_target_dist))
            + 0.5 * (1 - jp.tanh(30 * box_target_dist))
        ) * lifted

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
