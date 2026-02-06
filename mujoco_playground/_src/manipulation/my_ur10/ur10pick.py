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
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union
from jax import debug
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.my_ur10 import ur10_base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for bring_to_target tasks."""
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                ## Reward scaling factors
                # Gripper goes to the box.
                gripper_box=4.0,
                # Box goes to the target mocap.
                box_target=8.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to target pose.
                robot_target_qpos=0.3,

            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="low_home",
    )
    return config


class UR10PickCube(ur10_base.UR10Base):
    """Bring a box to a target."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        sample_orientation: bool = False,
    ):

        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "my_ur10"
            / "xmls"
            / "mjx_single_multi_shape_position.xml"
        )

        super().__init__(
            xml_path,
            config,
            config_overrides,
        )

        init_keyframe = getattr(self._config, "init_keyframe", "low_home")

        # Multi-shape support — USE np.array NOT jp.array
        self._obj_bodies = jp.array([
            self._mj_model.body("obj_box").id,
            self._mj_model.body("obj_sphere").id,
            self._mj_model.body("obj_cylinder").id,
        ])
        self._obj_qposadrs = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.body(n).jntadr[0]]
            for n in ["obj_box", "obj_sphere", "obj_cylinder"]
        ])

        self._post_init(obj_name="obj_box", keyframe=init_keyframe)
        self._sample_orientation = sample_orientation
        self._gripper_site = self._mj_model.site("tcp").id
        self._left_finger_touch = (self._mj_model.site("left_finger_touch_site").id,)
        self._right_finger_touch = (self._mj_model.site("right_finger_touch_site").id,)

        self._floor_hand_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_pad_floor_found",
                "right_finger_pad_floor_found",
                "hand_capsule_floor_found",
            ]
        ]
        print("Available sensors:", flush=True)
        for i in range(self._mj_model.nsensor):
            print(f"  - {self._mj_model.sensor(i).name}", flush=True)


    def reset(self, rng: jax.Array) -> State:
        rng, rng_obj, rng_target, rng_robot, rng_gripper, rng_shape = jax.random.split(rng, 6)

        # 1. Randomly choose active shape: 0=box, 1=sphere, 2=capsule
        active_shape = jax.random.randint(rng_shape, (), 0, 3)
        active_body_id = self._obj_bodies[active_shape]

        # 2. Randomize object spawn position
        obj_pos = (
            jax.random.uniform(
                rng_obj,
                (3,),
                minval=jp.array([-0.2, -0.3, 0.0]),
                maxval=jp.array([0.2, 0.3, 0.0]),
            )
            + self._init_obj_pos
        )

        # 3. Initialize target position
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, -0.3, 0.3]),
                maxval=jp.array([0.2, 0.3, 0.5]),
            )
            + self._init_obj_pos
        )

        # 4. Randomize robot joint positions (arm only)
        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),
            minval=-0.1,
            maxval=0.1,
        )
        init_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
        noisy_arm_qpos = init_arm_qpos + robot_qpos_noise

        # Gripper noise
        gripper_noise = jax.random.uniform(
            rng_gripper,
            (2,),
            minval=0.0,
            maxval=0.01,
        )
        init_finger_qpos = jp.array(self._init_q[self._robot_qposadr[-2:]])
        noisy_finger_qpos = init_finger_qpos + gripper_noise

        # Optional orientation sampling
        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self._sample_orientation:
            rng, rng_axis, rng_theta = jax.random.split(rng, 3)
            perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
            perturb_axis = perturb_axis / math.norm(perturb_axis)
            perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
            target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

        # 5. Build initial qpos
        init_q = jp.array(self._init_q)

        # Set noisy arm + gripper joint positions
        init_q = init_q.at[self._robot_arm_qposadr].set(noisy_arm_qpos)
        init_q = init_q.at[self._robot_qposadr[-2:]].set(noisy_finger_qpos)

        # Place each shape: active one at obj_pos, others underground
        underground = jp.array([0.0, 0.0, -10.0])
        for i in range(3):
            adr = int(self._obj_qposadrs[i])
            pos_i = jp.where(i == active_shape, obj_pos, underground)
            init_q = init_q.at[adr : adr + 3].set(pos_i)

        # DEBUG: verify all 3 shape positions in qpos
        debug.print(
            "active={a} | box_pos={b} | sphere_pos={s} | cyl_pos={c}",
            a=active_shape,
            b=init_q[int(self._obj_qposadrs[0]):int(self._obj_qposadrs[0])+3],
            s=init_q[int(self._obj_qposadrs[1]):int(self._obj_qposadrs[1])+3],
            c=init_q[int(self._obj_qposadrs[2]):int(self._obj_qposadrs[2])+3],
        )

        # 6. Make ctrl consistent with init_q
        init_ctrl = jp.array(self._init_ctrl)
        init_ctrl = init_ctrl.at[:len(self._robot_arm_qposadr)].set(noisy_arm_qpos)
        init_ctrl = init_ctrl.at[-1].set(noisy_finger_qpos.sum() * 0.5)

        # 7. Create data
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Set target mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        # 8. Initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_body": 0.0,
            "active_shape": active_shape,
            "active_body_id": active_body_id,
            "step": jp.array(0, dtype=jp.int32),
            "debug_every": jp.array(10, dtype=jp.int32),
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
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        # Use active body for out-of-bounds check
        body_id = info["active_body_id"]
        body_pos = data.xpos[body_id]
        out_of_bounds = jp.any(jp.abs(body_pos) > 1.2) | (body_pos[2] < 0.0)
        done = (out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(jp.float32)

        metrics = state.metrics
        metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(jp.float32))

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        # ==============================
        # Position world-frame
        # ==============================
        target_pos = info["target_pos"]
        body_id = info["active_body_id"]
        body_pos = data.xpos[body_id]
        gripper_pos = data.site_xpos[self._gripper_site]
        left_finger_touch_pos = data.site_xpos[self._left_finger_touch]
        right_finger_touch_pos = data.site_xpos[self._right_finger_touch]

        # ==============================
        # Distance calculation
        # ==============================
        body_target_dist = jp.linalg.norm(target_pos - body_pos)
        gripper_body_dist = jp.linalg.norm(body_pos - gripper_pos)
        finger_touch_dist = jp.linalg.norm(right_finger_touch_pos - left_finger_touch_pos)

        # ==============================
        # Rotation related computations
        # ==============================
        box_mat = data.xmat[body_id]
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

        # ==============================
        # Reward terms
        # ==============================
        body_target_Reward = 1 - jp.tanh(5 * (0.9 * body_target_dist + 0.1 * rot_err))
        gripper_body_Reward = 1 - jp.tanh(5 * gripper_body_dist)
        robot_target_qpos_penalty = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Floor collision via touch sensors
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._floor_hand_found_sensor
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision_Reward = (1 - floor_collision).astype(float)

        # Reached body gate
        info["reached_body"] = 1.0 * jp.maximum(
            info["reached_body"],
            (gripper_body_dist < 0.02),
        )

        # NOTE: keys must match self._config.reward_config.scales
        rewards = {
            "gripper_box": gripper_body_Reward,
            "box_target": body_target_Reward * info["reached_body"],
            "no_floor_collision": no_floor_collision_Reward,
            "robot_target_qpos": robot_target_qpos_penalty,
        }

        raw = {
            "target_pos": target_pos,
            "body_pos": body_pos,
            "gripper_pos": gripper_pos,
            "left_finger_touch_pos": left_finger_touch_pos,
            "right_finger_touch_pos": right_finger_touch_pos,
            "body_target_dist": body_target_dist,
            "rot_err": rot_err,
            "gripper_body_dist": gripper_body_dist,
            "finger_touch_dist": finger_touch_dist,
            "reached_body": info["reached_body"],
            "number_floor_collision": floor_collision,
        }

        return rewards, raw

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_mat = data.site_xmat[self._gripper_site].ravel()
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])

        # Use active body instead of fixed self._obj_body
        body_id = info["active_body_id"]

        obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
                gripper_pos,
                gripper_mat[3:],
                data.xmat[body_id].ravel()[3:],
                data.xpos[body_id] - data.site_xpos[self._gripper_site],
                info["target_pos"] - data.xpos[body_id],
                target_mat.ravel()[:6] - data.xmat[body_id].ravel()[:6],
                data.ctrl
                - data.qpos[self._robot_qposadr[:-1]],
            ]
        )

        return obs



class UR10PickCubeOrientation(UR10PickCube):
    """Bring a box to a target and orientation."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides, sample_orientation=True)
