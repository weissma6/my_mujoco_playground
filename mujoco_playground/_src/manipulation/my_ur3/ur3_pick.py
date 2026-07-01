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
The 4x4x4 cm box spawns with a random Z-axis yaw (range set by box_z_rot_range)
and a rotation-error reward encourages bringing it back to the canonical
(identity) orientation while lifting. Optionally the box spawns on a
variable-height "lifter" plate (lifter_height_max) so the policy learns to grasp
boxes at different heights, and the arm start pose can be drawn from a library of
hand-collected real-robot poses (init_start_random). Mirrors the commented-out
reward scaffolding of ur10pick.py.
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
_LIFTER_HALF_THICKNESS = 0.0005  # 1 mm plate -> half-extent
_LIFTER_HEIGHT_MIN = 0.003
_BOX_HALF_EXTENT = 0.02  # 4 cm cube -> half-extent


def default_config() -> config_dict.ConfigDict:
    """Default config for the UR3 pick task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                ## Staged reward scaling factors (sequenced by sticky latches).
                # Gripper (TCP) approaches the box (always on).
                gripper_box=4.0,
                # Close the fingers on the box once the gripper has reached it.
                grasp=3.0,
                # Raise the box off its resting height — anti-push lever that
                # gates box_target behind a real lift.
                lift=5.0,
                # Box goes to the mocap target (lift point in the air); gated by
                # the sticky "lifted" latch so a sliding box earns nothing.
                box_target=8.0,
                # Box orientation aligns with the canonical (identity) target —
                # gated by sticky reached_box (only active once at the box).
                box_orient=2.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to initial pose.
                robot_target_qpos=0.3,
            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="low_home",
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
        lifter_height_max=0.0,
        # Box spawn yaw about world Z (rad); yaw ~ uniform(-r, +r). 0.0 =
        # axis-aligned. Replaces the old ±45° Y-axis tilt.
        box_z_rot_range=0.0,
        # Box-center distance (m) to the lift target counted as success (3
        # consecutive steps). Tight 5 mm — the box must end up inside the target.
        success_tol=0.005,
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

        # Canonical (identity) box orientation target, pre-flattened to the first
        # two rows of the rotation matrix (matches the obs/reward layout).
        self._target_xmat_flat = jp.eye(3).ravel()[:6]

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
        ) = jax.random.split(rng, 8)

        # initialize box XY — wide jitter, X-heavy, so the box spawns anywhere
        # in X∈[0.30,0.60] (all graspable; 0.60 was the old nominal).
        box_xy = (
            jax.random.uniform(
                rng_box,
                (2,),
                minval=jp.array([-0.15, -0.10]),
                maxval=jp.array([0.15, 0.10]),
            )
            + self._init_obj_pos[:2]  # Box XY from XML keyframe
        )

        # lifter height + box resting Z. When enabled, sample a per-episode plate
        # height and rest the box on the plate top; else keep the legacy on-floor
        # Z from the keyframe.
        if self._lifter_enabled:
            lifter_h = jax.random.uniform(
                rng_lift,
                (),
                minval=_LIFTER_HEIGHT_MIN,
                maxval=float(self._config.lifter_height_max),
            )
            box_z = lifter_h + _LIFTER_HALF_THICKNESS + _BOX_HALF_EXTENT
        else:
            lifter_h = jp.array(0.0, dtype=float)
            box_z = self._init_obj_pos[2]  # legacy on-floor (0.02)
        box_pos = jp.array([box_xy[0], box_xy[1], box_z])

        # initialize box orientation — random yaw about world Z, ±box_z_rot_range.
        # (A symmetric cube just topples under an X/Y tilt, so only yaw is physical.)
        theta = jax.random.uniform(
            rng_quat,
            (),
            minval=-self._config.box_z_rot_range,
            maxval=self._config.box_z_rot_range,
        )
        box_quat = jp.array(
            [jp.cos(theta / 2.0), 0.0, 0.0, jp.sin(theta / 2.0)], dtype=float
        )

        # initialize target position — a lift point in the air above the box.
        # Envelope kept tight so every target stays within reach (worst-case 3D
        # reach ≈0.54 m); required for the 5 mm success criterion to be feasible.
        # X biased forward (+0.01‥+0.06) so the lift pulls the box toward the
        # robot's reachable front; Y pulled in (±0.03) so the lift stays near
        # the robot's sagittal plane. Z band tightened UP to 0.12‥0.15 m (was
        # 0.08‥0.15) so every target is a clearly-visible lift, not hovering at
        # the box start. The MAX stays 0.15: the far corner (x=0.51,z=0.17) is
        # already at UR3's ~0.54 m reach limit, so the ceiling is physics-bound —
        # only the floor could be raised. Raised further by the lifter height so
        # the lift target stays above the (possibly raised) box.
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([0.01, -0.03, 0.12]),
                maxval=jp.array([0.06, 0.03, 0.15]),
            )
            + self._init_obj_pos
        )
        target_pos = target_pos.at[2].add(lifter_h)

        # -----------------------------
        # Randomize robot joint positions
        # -----------------------------
        # Base arm/finger pose: a random hand-collected library pose when a
        # difficulty level is set, else the literal keyframe start. init_qpos_noise
        # then jitters it uniform(-v, +v) per direction (6 arm + 1 finger); set the
        # noise to 0 in a sweep to use the library/keyframe pose verbatim.
        if self._init_pose_lib is not None:
            pose = self._init_pose_lib[
                jax.random.randint(rng_pose, (), 0, self._n_init_poses)
            ]
            base_arm_qpos = pose[:6]
            base_finger_qpos = jp.array([pose[6], pose[6]])  # one finger -> both
        else:
            base_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
            base_finger_qpos = jp.array(self._init_q[self._robot_qposadr[-2:]])

        noise_amp = jp.asarray(self._config.init_qpos_noise, dtype=float)
        arm_amp = noise_amp[: len(self._robot_arm_qposadr)]  # 6 arm joints
        finger_amp = noise_amp[-1]

        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),  # 6 arm joints
            minval=-arm_amp,
            maxval=arm_amp,
        )
        noisy_arm_qpos = base_arm_qpos + robot_qpos_noise

        # Gripper noise — symmetric, clipped to the physical finger range [0, 0.025]
        gripper_noise = jax.random.uniform(
            rng_gripper,
            (2,),  # left and right finger
            minval=-finger_amp,
            maxval=finger_amp,
        )
        noisy_finger_qpos = jp.clip(base_finger_qpos + gripper_noise, 0.0, 0.025)

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

        # set target mocap position (lift goal marker); place the lifter plate
        # under the box when enabled (else it stays parked at its XML default).
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )
        if self._lifter_enabled:
            lifter_pos = jp.array([box_xy[0], box_xy[1], lifter_h])
            data = data.replace(
                mocap_pos=data.mocap_pos.at[self._lifter_mocap, :].set(lifter_pos),
            )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "success": jp.array(0.0, dtype=float),
            "box_target_dist": jp.array(0.0, dtype=float),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
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
        )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

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
        # --- Rotation related computations ---
        # ==============================
        # First two rows of the box world-frame rotation matrix - JAX array (6,) float64
        box_xmat_flat = data.xmat[self._obj_body].ravel()[:6]
        # Rotation error between box and the canonical (identity) target - scalar float64
        rot_err = jp.linalg.norm(self._target_xmat_flat - box_xmat_flat)

        # ==============================
        # --- Sticky stage latches (monotone via jp.maximum) ---
        # ==============================
        # reached_box: gripper has been within 2 cm of the box at some point.
        info["reached_box"] = jp.maximum(
            info["reached_box"],
            (gripper_box_dist < 0.02).astype(float),
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

        # Stage 1 — approach (always on): gripper moves onto the box.
        gripper_box_Reward = 1 - jp.tanh(5 * gripper_box_dist)

        # Stage 2 — grasp: close the fingers on the box once reached. No
        # near-target fade (unlike picknplace) — this task holds the box AT the
        # air target rather than releasing it, so the grasp must stay rewarded.
        grasp_Reward = finger_closed * reached

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

        # Orientation reward — only active once the gripper has reached the box
        # (avoids rewarding orientation noise during the free-floating approach).
        box_orient_Reward = (1 - jp.tanh(2 * rot_err)) * reached

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
            "grasp": grasp_Reward,
            "lift": lift_Reward,
            "box_target": box_target_Reward,
            "box_orient": box_orient_Reward,
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
            "rot_err": rot_err,
            "grip_box_dist": gripper_box_dist,
            "finger_touch_dist": finger_touch_dist,

            # events
            "reached_box": info["reached_box"],
            "grasped": info["grasped"],
            "lifted": info["lifted"],
            "number_floor_collision": floor_collision,
        }

        return rewards, raw
