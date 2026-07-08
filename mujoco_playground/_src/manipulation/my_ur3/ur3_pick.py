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
        episode_length=200,
        action_repeat=1,
        # Arm per-step ctrl delta = action * action_scale (swept per run). The
        # gripper is DECOUPLED via gripper_action_scale below so it can be kept
        # slow (stays open on approach) while the arm runs faster.
        action_scale=0.01,
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
                # actually close on the (rotated + tilted) cube. Kept modest and
                # proximity-gated so it can't out-pay grasp/lift or be farmed in
                # free space (the reward-hacking lesson from earlier open bonuses).
                gripper_align=2.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to initial pose.
                robot_target_qpos=0.3,
            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="task_home",
        # Per-joint per-direction amplitude (rad) for reset randomization: 6 arm
        # + 1 finger. Applied symmetrically as uniform(-v, +v) on top of the init
        # keyframe. Default 0.05 reproduces the legacy uniform(-0.05, 0.05) arm noise.
        init_qpos_noise=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01),
        # Arm/finger start-pose source. "none" = literal keyframe start (then +
        # init_qpos_noise jitter); "light"/"mid"/"hard" = randomly pick one
        # hand-collected pose from init_poses/train/<level>.json each reset.
        init_start_random="none",
        # Per-episode box-riser height (m). 0.0 disables the lifter (box on the
        # floor, legacy). >0 => h ~ uniform(_LIFTER_HEIGHT_MIN, lifter_height_max)
        # and the box spawns resting on the plate; the lift target is raised by h.
        lifter_height_max=0.03,
        # Per-episode SLIGHT plate tilt (rad). roll,pitch ~ uniform(-t, +t) about
        # world X and Y; the box rests FLUSH on the tilted plate, so it starts
        # both at a variable height and slightly tilted. This is what makes the
        # out-of-plane (approach-axis) component of the 2-of-3-axis grasp
        # alignment matter. 0.0 => flat plate (legacy). Only active with the
        # lifter enabled. Keep small (< ~0.12 rad) so the cube can't slide/tip.
        lifter_tilt_max=0.08,  # ~4.6 deg
        # Box spawn yaw about world Z (rad); yaw ~ uniform(-r, +r). pi/4 covers
        # all yaw thanks to the cube's 4-fold symmetry, so the policy must learn
        # to match the jaw axis to a face rather than getting a free alignment.
        box_z_rot_range=0.7853981633974483,  # pi/4
        # Box-center distance (m) to the lift target counted as success (3
        # consecutive steps). Tight 3 mm — the box must end up inside the target.
        success_tol=0.003,
        # "Off the resting height" margin (m) a grasped box must clear to set the
        # sticky "lifted" latch that unlocks box_target (anti-push lever).
        lift_eps=0.03,
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
        # in X∈[0.30,0.60] (all graspable; 0.60 was the old nominal).
        box_xy = (
            jax.random.uniform(
                rng_box,
                (2,),
                minval=jp.array([-0.15, -0.2]),
                maxval=jp.array([0.15, 0.2]),
            )
            + self._init_obj_pos[:2]  # Box XY from XML keyframe
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
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([0.02, -0.03, 0.18]),
                maxval=jp.array([0.06, 0.03, 0.21]),
            )
            + self._init_obj_pos
        )
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

        raw_rewards, raw_signals = self._get_reward(data, info)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

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

        metrics = state.metrics
        metrics.update(
            **raw_rewards,
            out_of_bounds=out_of_bounds.astype(jp.float32),
            success=success.astype(jp.float32),
            box_target_dist=box_target_dist,
            reached_box=info["reached_box"],
            grasped=info["grasped"],
            lifted=info["lifted"],
            align_jaw=raw_signals["a_jaw"],
            align_app=raw_signals["a_app"],
        )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """13D base obs + 6D gripper<->box orientation features (19D total).

        Appends where the two grasp-relevant gripper axes point IN THE BOX FRAME:
        the jaw axis (finger-separation) and the approach axis (palm->fingertips),
        each projected onto the three box axes. This is the state the policy needs
        to align its frame with the (rotated + tilted) cube, and it is
        reproducible on the real robot: jaw/approach axes come from arm FK, the box
        axes from the mocap-streamed box quaternion. NOTE: when deploying a policy
        trained with this obs, `build_obs_from_feedback` on the real robot must
        append the same 6 numbers (re-enable the mocap orientation it currently
        drops), or sim and real observations will not match.
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
        return jp.concatenate([base_obs, jaw_proj, app_proj])

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
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
        # Euclidean distance between gripper and box - scalar JAX float64
        gripper_box_dist = jp.linalg.norm(
            box_pos - gripper_pos
        )
        # Euclidean distance between the two finger touch sites - scalar JAX float64
        finger_touch_dist = jp.linalg.norm(
            right_finger_touch_pos - left_finger_touch_pos
        )

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
        grasp_now = (info["reached_box"] > 0.5) & both_pads_touch
        info["grasped"] = jp.maximum(info["grasped"], grasp_now.astype(float))
        # lifted: a grasped box has cleared its per-episode resting height by
        # lift_eps. Anti-push latch — box_target only pays once this is set.
        box_off_rest = box_pos[2] > (info["box_rest_z"] + self._lift_eps)
        lift_now = box_off_rest & (info["grasped"] > 0.5)
        info["lifted"] = jp.maximum(info["lifted"], lift_now.astype(float))

        # ==============================
        # --- Reward terms (staged by the latches above) ---
        # ==============================
        reached = info["reached_box"]
        grasped = info["grasped"]
        lifted = info["lifted"]

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
        grasp_Reward = finger_closed * reached

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

        # Stage 1c — grasp-frame alignment (2 of 3 axes). A parallel-jaw gripper
        # can close on a cube only when its frame lines up with the cube's frame:
        # the jaw axis (finger separation) and the approach axis (palm ->
        # fingertips) must EACH line up with a box face-normal. If those two
        # align, the third gripper axis is forced by orthonormality — so "2 of 3
        # axes aligned" == fully aligned. Both axes are built from world-frame
        # site positions (no frame/column assumptions); max over the 3 box axes +
        # abs makes the score invariant to the cube's 24-fold (octahedral)
        # symmetry, so every equivalent grasp scores the same.
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
        # Weighted by approach proximity so it shapes the final approach and can't
        # be farmed in free space (kept below grasp=3.0: reward-hacking lesson).
        gripper_align_Reward = alignment * (1 - jp.tanh(5 * gripper_box_dist))

        # Penalty for deviating too far from the initial arm configuration.
        robot_target_qpos_penalty = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

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
            "gripper_align": gripper_align_Reward,
            "no_floor_collision": no_floor_collision_Reward,
            "robot_target_qpos": robot_target_qpos_penalty,
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
