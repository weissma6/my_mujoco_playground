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
"""UR3 pick-and-place: 6-DOF arm + Hand-E gripper, carry a box to a drop zone.

The 4x4x4 cm box spawns on the +Y side (~20 cm from the workspace center) with a
random Y-axis tilt. The 5x5x5 cm mocap target is a drop zone on the -Y side at
the same radius, at the box resting height so a released box settles inside it.
The reward is staged by sticky latches (reached -> grasped -> lifted ->
delivered): approach, grasp, lift off the floor, carry to the target (gated by a
real lift so pushing earns nothing), seat the box fully inside the target box
(all 8 corners), open the fingers, and retract the gripper clear of the box.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.my_ur3 import ur3_base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member


def default_config() -> config_dict.ConfigDict:
    """Default config for the UR3 pick task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=250,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                ## Staged pick-and-place reward (sequenced by sticky latches).
                # Stage 1 — approach: TCP gets to the box (always on).
                gripper_box=2.0,
                # Stage 2 — grasp: close the fingers on the box (fades near the
                # target so release can take over; no cliff at any latch).
                grasp=3.0,
                # Stage 3 — lift: raise the box off the floor. Anti-push lever —
                # box_target is gated behind a real lift.
                lift=5.0,
                # Stage 4 — transport: box approaches the drop target, GATED by
                # the sticky "lifted" latch (a sliding box earns nothing here).
                box_target=8.0,
                # Stage 5a — placement: 4 cm box fully inside the 5 cm target box
                # (all 8 corners), gated by the sticky "grasped" latch.
                box_inside=6.0,
                # Stage 5b — release: open the fingers near the target (only if
                # the box was actually carried there, i.e. lifted).
                release=3.0,
                # Stage 5c — retract: move the gripper up/away after delivery so
                # the box ends in the zone without the gripper.
                retract=4.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to initial pose.
                robot_target_qpos=0.3,
            )
        ),
        # Curriculum / task knobs (tune locally, ramp difficulty across runs).
        curriculum=config_dict.create(
            tilt_deg=15.0,  # spawn box ±tilt about Y; start small, ramp to 45
            lift_eps=0.03,  # "off the floor" margin (m) for the lifted latch
            spawn_xy_jitter=0.05,  # shared pick/drop XY radius ("same radius")
            success_tol=1e-3,  # containment_max threshold for success
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="low_home",
    )


class UR3PicknPlace(ur3_base.UR3Base):
    """Pick a box and place it in a drop zone with the UR3 + Hand-E."""

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
            / "mjx_single_cube_position_ur3_picknplace.xml"
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

        # Box resting height (z of the box center at the keyframe) and the margin
        # a grasped box must clear to count as "lifted" (anti-push latch).
        self._box_rest_z = float(self._init_obj_pos[2])
        self._lift_eps = float(self._config.curriculum.lift_eps)

        # Canonical (identity) box orientation target, pre-flattened to the first
        # two rows of the rotation matrix (kept only as a logged diagnostic).
        self._target_xmat_flat = jp.eye(3).ravel()[:6]

        # Box (4 cm) and target (5 cm) half-extents — single source of truth is
        # the XML; used by the containment reward (box must sit inside target).
        self._box_half = jp.array(self._mj_model.geom("box").size, dtype=float)
        self._target_half = jp.array(
            self._mj_model.geom("mocap_target_geom").size, dtype=float
        )
        # 8 corner sign combinations of an axis-aligned box, shape (8, 3).
        self._box_corner_signs = jp.array(
            [
                [sx, sy, sz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=float,
        )

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_quat, rng_target, rng_robot, rng_gripper = (
            jax.random.split(rng, 6)
        )

        # initialize box position — spawn ~20 cm to the +Y side of the workspace
        # center, tight jitter (UR3 reach ~0.5 m). The -Y drop target is kept at
        # the same radius (symmetric in Y).
        xy_jitter = self._config.curriculum.spawn_xy_jitter
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-xy_jitter, -xy_jitter, 0.0]),
                maxval=jp.array([xy_jitter, xy_jitter, 0.0]),
            )
            + self._init_obj_pos  # Box position from XML keyframe
            + jp.array([0.0, 0.20, 0.0])  # shift to +Y20cm spawn side
        )

        # initialize box orientation — random rotation around world Y axis. Tilt
        # magnitude is a curriculum knob (start small, ramp toward ±45°).
        tilt = jp.deg2rad(self._config.curriculum.tilt_deg)
        theta = jax.random.uniform(rng_quat, (), minval=-tilt, maxval=tilt)
        box_quat = jp.array(
            [jp.cos(theta / 2.0), 0.0, jp.sin(theta / 2.0), 0.0], dtype=float
        )

        # initialize target position — drop zone ~20 cm to the -Y side, at the
        # box resting height so a released box settles *inside* the zone (the
        # gripper need not hold it there). Same radius as the +Y spawn.
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-xy_jitter, -xy_jitter, 0.0]),
                maxval=jp.array([xy_jitter, xy_jitter, 0.0]),
            )
            + self._init_obj_pos
            + jp.array([0.0, -0.20, 0.0])  # -Y20cm side, z = box rest (0.02)
        )

        # -----------------------------
        # Randomize robot joint positions (arm only, not gripper)
        # -----------------------------
        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),  # 6 arm joints
            minval=-0.05,  # ~3 degrees in radians
            maxval=0.05,
        )
        init_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
        noisy_arm_qpos = init_arm_qpos + robot_qpos_noise

        # Gripper noise (small range since finger range is 0-0.025)
        gripper_noise = jax.random.uniform(
            rng_gripper,
            (2,),  # left and right finger
            minval=0.0,
            maxval=0.01,
        )
        init_finger_qpos = jp.array(self._init_q[self._robot_qposadr[-2:]])
        noisy_finger_qpos = init_finger_qpos + gripper_noise

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

        # set target mocap position (lift goal marker)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "success": jp.array(0.0, dtype=float),
            "box_target_dist": jp.array(0.0, dtype=float),
            "reached_box": jp.array(0.0, dtype=float),
            "grasped": jp.array(0.0, dtype=float),
            "lifted": jp.array(0.0, dtype=float),
            "delivered": jp.array(0.0, dtype=float),
            "box_height": jp.array(0.0, dtype=float),
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
            "delivered": jp.array(0.0, dtype=float),
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

        # Success: the whole 4 cm box sits inside the 5 cm target box
        # (worst corner within success_tol of the bounds).
        success_now = (
            raw_signals["containment_max"] <= self._config.curriculum.success_tol
        )
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
            delivered=info["delivered"],
            box_height=raw_signals["box_height"],
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
        # Rotation error vs the canonical (identity) target - scalar float64 (diagnostic only)
        rot_err = jp.linalg.norm(self._target_xmat_flat - box_xmat_flat)

        # ==============================
        # --- Containment: 4 cm box fully inside the 5 cm target box ---
        # ==============================
        # World-frame rotation matrix of the box - JAX array (3,3) float64
        box_mat = data.xmat[self._obj_body].reshape(3, 3)
        # 8 box corners in world frame: box_pos + R_box @ (signs * box_half) - (8,3)
        box_corners = box_pos + (self._box_corner_signs * self._box_half) @ box_mat.T
        # Per-corner, per-axis overshoot beyond the axis-aligned target box - (8,3) m
        corner_overshoot = jp.maximum(
            jp.abs(box_corners - target_pos) - self._target_half, 0.0
        )
        # Total overshoot (m); zero iff every corner is inside the target box.
        containment_violation = jp.sum(corner_overshoot)
        # Worst single-corner overshoot (m) - used for the success criterion.
        containment_max = jp.max(corner_overshoot)

        # ==============================
        # --- Sticky stage latches (monotone via jp.maximum) ---
        # ==============================
        # reached_box: gripper has been within 2 cm of the box at some point.
        info["reached_box"] = jp.maximum(
            info["reached_box"],
            (gripper_box_dist < 0.02).astype(float),  # Panda threshold was 0.012
        )
        # grasped: at the box AND both finger pads in contact with it.
        finger_box_contact = [
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._finger_box_found_sensor
        ]
        both_pads_touch = sum(finger_box_contact) >= 2
        grasp_now = (info["reached_box"] > 0.5) & both_pads_touch
        info["grasped"] = jp.maximum(info["grasped"], grasp_now.astype(float))
        # lifted: a grasped box has cleared the floor by lift_eps. The anti-push
        # latch — box_target only pays once this is set.
        box_off_floor = box_pos[2] > (self._box_rest_z + self._lift_eps)
        lift_now = box_off_floor & (info["grasped"] > 0.5)
        info["lifted"] = jp.maximum(info["lifted"], lift_now.astype(float))
        # delivered: a carried (lifted) box is essentially inside the target box.
        deliver_now = (containment_violation < 5e-3) & (info["lifted"] > 0.5)
        info["delivered"] = jp.maximum(
            info["delivered"], deliver_now.astype(float)
        )

        # ==============================
        # --- Reward terms (staged by the latches above) ---
        # ==============================
        reached = info["reached_box"]
        grasped = info["grasped"]
        lifted = info["lifted"]
        delivered = info["delivered"]

        # Proximity to the drop target; gripper open/closed in [0, 1].
        near_target = 1 - jp.tanh(10.0 * box_target_dist)
        finger_open = jp.tanh(finger_touch_dist / 0.05)
        finger_closed = 1 - finger_open

        # Stage 1 — approach (always on).
        gripper_box_Reward = 1 - jp.tanh(5.0 * gripper_box_dist)

        # Stage 2 — grasp: close the fingers on the box. Active at the box and
        # faded near the target so it does not fight the release. Not gated by a
        # latch => no reward cliff when a stage latches.
        grasp_Reward = finger_closed * reached * (1 - near_target)

        # Stage 3 — lift: raise the box off the floor (saturates ~12 cm). Faded
        # near the target so lowering the box to place it is not penalized.
        lift_height = jp.clip(box_pos[2] - self._box_rest_z, 0.0, 0.12)
        lift_Reward = jp.tanh(lift_height / 0.06) * reached * (1 - near_target)

        # Stage 4 — transport: box approaches the target. GATED by "lifted" so a
        # box pushed along the floor earns nothing here.
        box_target_Reward = (1 - jp.tanh(5.0 * box_target_dist)) * lifted

        # Stage 5a — placement: 4 cm box fully inside the 5 cm target box, gated
        # by "grasped" (only meaningful after a real grasp, never for pushing).
        box_inside_Reward = (
            1 - jp.tanh(15.0 * containment_violation)
        ) * grasped

        # Stage 5b — release: open the fingers near the target, only if the box
        # was actually carried there (lifted).
        release_Reward = finger_open * near_target * lifted

        # Stage 5c — retract: once delivered, move the TCP up and away from the
        # box so it ends in the zone without the gripper.
        tcp_above_box = jp.clip(gripper_pos[2] - box_pos[2], 0.0, 0.10)
        tcp_box_xy = jp.linalg.norm(gripper_pos[:2] - box_pos[:2])
        retract_Reward = (
            0.5 * jp.tanh(tcp_above_box / 0.05)
            + 0.5 * jp.tanh(tcp_box_xy / 0.05)
        ) * delivered

        # Arm stays close to its initial configuration.
        robot_target_qpos_penalty = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Floor collision via touch sensors (same pattern as Panda).
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
            "box_inside": box_inside_Reward,
            "release": release_Reward,
            "retract": retract_Reward,
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
            "containment_violation": containment_violation,
            "containment_max": containment_max,
            "grip_box_dist": gripper_box_dist,
            "finger_touch_dist": finger_touch_dist,

            # events
            "reached_box": info["reached_box"],
            "grasped": info["grasped"],
            "lifted": info["lifted"],
            "delivered": info["delivered"],
            "box_height": box_pos[2],
            "number_floor_collision": floor_collision,
        }

        return rewards, raw
