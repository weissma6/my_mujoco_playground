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
The 4x4x4 cm box spawns with a random Y-axis tilt and a rotation-error reward
encourages bringing it back to the canonical (identity) orientation while lifting.
Mirrors the commented-out reward scaffolding of ur10pick.py.
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
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                ## Reward scaling factors
                # Box goes to the mocap target (lift point in the air).
                box_target=8.0,
                # Gripper (TCP) approaches the box.
                gripper_box=4.0,
                # Box orientation aligns with the canonical (identity) target —
                # gated by sticky reached_box (only active once at the box).
                box_orient=2.0,
                # Encourage finger opening relative to box distance (ur10pick formula).
                finger_touch=1.0,
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
        # Box-center distance (m) to the lift target counted as success (3
        # consecutive steps). Tight 5 mm — the box must end up inside the target.
        success_tol=0.005,
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

        # Canonical (identity) box orientation target, pre-flattened to the first
        # two rows of the rotation matrix (matches the obs/reward layout).
        self._target_xmat_flat = jp.eye(3).ravel()[:6]

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_quat, rng_target, rng_robot, rng_gripper = (
            jax.random.split(rng, 6)
        )

        # initialize box position — wide jitter, X-heavy, so the box spawns
        # anywhere in X∈[0.30,0.60] (all graspable; 0.60 was the old nominal).
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.15, -0.10, 0.0]),
                maxval=jp.array([0.15, 0.10, 0.0]),
            )
            + self._init_obj_pos  # Box position from XML keyframe
        )

        # initialize box orientation — random rotation around world Y axis, ±45°.
        theta = jax.random.uniform(
            rng_quat, (), minval=-jp.pi / 4.0, maxval=jp.pi / 4.0
        )
        box_quat = jp.array(
            [jp.cos(theta / 2.0), 0.0, jp.sin(theta / 2.0), 0.0], dtype=float
        )

        # initialize target position — a lift point in the air above the box.
        # Envelope kept tight so every target stays within reach (worst-case 3D
        # reach ≈0.54 m); required for the 5 mm success criterion to be feasible.
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.06, -0.06, 0.08]),
                maxval=jp.array([0.06, 0.06, 0.15]),
            )
            + self._init_obj_pos
        )

        # -----------------------------
        # Randomize robot joint positions (per-joint amplitude from config)
        # -----------------------------
        # Per-direction amplitude (rad): 6 arm joints + 1 finger. Applied as
        # uniform(-v, +v) on top of the init keyframe.
        noise_amp = jp.asarray(self._config.init_qpos_noise, dtype=float)
        arm_amp = noise_amp[: len(self._robot_arm_qposadr)]  # 6 arm joints
        finger_amp = noise_amp[-1]

        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),  # 6 arm joints
            minval=-arm_amp,
            maxval=arm_amp,
        )
        init_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
        noisy_arm_qpos = init_arm_qpos + robot_qpos_noise

        # Gripper noise — symmetric, clipped to the physical finger range [0, 0.025]
        gripper_noise = jax.random.uniform(
            rng_gripper,
            (2,),  # left and right finger
            minval=-finger_amp,
            maxval=finger_amp,
        )
        init_finger_qpos = jp.array(self._init_q[self._robot_qposadr[-2:]])
        noisy_finger_qpos = jp.clip(init_finger_qpos + gripper_noise, 0.0, 0.025)

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
            **{k: jp.array(0.0, dtype=float)
               for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "step": jp.array(0, dtype=jp.int32),
            "success_counter": jp.array(0, dtype=jp.int32),
            "reached_box": jp.array(0.0, dtype=float),
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
        # --- Reward terms ---
        # ==============================
        # Reward for box being at target - scalar JAX float64.
        # Two-scale: a coarse term (tanh*5) keeps a long-range lift pull, a fine
        # term (tanh*30) adds a steep gradient near the goal so the policy is
        # actually pulled the last ~10 mm (plain tanh*5 is flat near zero, which
        # is why the box plateaued ~13 mm short).
        box_target_Reward = 0.5 * (1 - jp.tanh(5 * box_target_dist)) + 0.5 * (
            1 - jp.tanh(30 * box_target_dist)
        )
        # Reward for gripper being at box - scalar JAX float64
        gripper_box_Reward = 1 - jp.tanh(
            5 * gripper_box_dist
        )
        # Penalty for deviating too far from the initial arm configuration - scalar JAX float64
        robot_target_qpos_penalty = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )
        # reward for finger distans large, when distance to box large
        finger_touch_Reward = jp.tanh(
            finger_touch_dist / (gripper_box_dist + 1e-6)
        )

        # Floor collision via touch sensors (same as Panda) - scalar JAX float64
        # List of booleans indicating if each sensor detects contact with the floor
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._floor_hand_found_sensor
        ]
        # Boolean indicating if any sensor detects contact with the floor
        floor_collision = (
            sum(hand_floor_collision) > 0
        )
        # Reward for no floor collision - scalar JAX float64
        no_floor_collision_Reward = (1 - floor_collision).astype(
            float
        )
        # ==============================
        # --- Same "reached_box" gate as Panda, but based on fingertip midpoint ---
        # Binary indicator if the gripper has reached the box - scalar JAX float64
        info["reached_box"] = 1.0 * jp.maximum(
            info["reached_box"],
            (gripper_box_dist < 0.01).astype(float),  # Panda threshold was 0.012
        )

        # Orientation reward — only active once the gripper has reached the box
        # (avoids rewarding orientation noise during the free-floating approach).
        box_orient_Reward = (1 - jp.tanh(2 * rot_err)) * info["reached_box"]

        rewards = {
            "box_target": box_target_Reward,
            "gripper_box": gripper_box_Reward,
            "box_orient": box_orient_Reward,
            "finger_touch": finger_touch_Reward,
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
            "number_floor_collision": floor_collision,
        }

        return rewards, raw
