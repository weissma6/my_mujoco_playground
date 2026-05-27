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
"""UR3 pick task: 6-DOF arm + Hand-E gripper, bring a box to a drop target.

Observation (21D):
  [q(6), qd(6), tcp_pos(3), box_pos(3), drop_target(3)]
Action (7D):
  6 arm joint deltas + 1 Hand-E tendon delta.
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
                # Bring the box to the drop target.
                box_target=8.0,
                # Bring the gripper (TCP) to the box.
                reach_box=4.0,
                # Do not collide the gripper/fingers with the floor.
                no_floor_collision=0.25,
                # Stay close to the initial arm pose.
                robot_target_qpos=0.3,
            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
        init_keyframe="low_home",
    )


class UR3Pick(ur3_base.UR3Base):
    """Bring a box to a drop target with the UR3 + Hand-E.

    Observation (21D):
      [q(6), qd(6), tcp_pos(3), box_pos(3), drop_target(3)]
    """

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

        self._gripper_site = self._mj_model.site("tcp").id

        # Floor-collision sensors (Hand-E fingers + hand capsule vs floor).
        self._floor_hand_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_pad_floor_found",
                "right_finger_pad_floor_found",
                "hand_capsule_floor_found",
            ]
        ]

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target, rng_robot, rng_gripper = jax.random.split(rng, 5)

        # Box position: jitter around the keyframe box pose (on the table).
        # UR3 reach is ~0.5 m, so the jitter is tighter than the UR10 task.
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.08, -0.08, 0.0]),
                maxval=jp.array([0.08, 0.08, 0.0]),
            )
            + self._init_obj_pos
        )

        # Drop target: a reachable workspace volume for the UR3.
        drop_target = jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([0.25, -0.15, 0.10]),
            maxval=jp.array([0.40, 0.15, 0.30]),
        )

        # Arm joint noise (±0.05 rad around keyframe).
        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),
            minval=-0.05,
            maxval=0.05,
        )
        init_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
        noisy_arm_qpos = init_arm_qpos + robot_qpos_noise

        # Gripper finger noise (small; finger range is 0-0.025).
        gripper_noise = jax.random.uniform(
            rng_gripper,
            (2,),
            minval=0.0,
            maxval=0.01,
        )
        init_finger_qpos = jp.array(self._init_q[self._robot_qposadr[-2:]])
        noisy_finger_qpos = init_finger_qpos + gripper_noise

        # Build qpos: noisy arm + noisy fingers + box pose.
        init_q = jp.array(self._init_q)
        init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        init_q = init_q.at[self._robot_arm_qposadr].set(noisy_arm_qpos)
        init_q = init_q.at[self._robot_qposadr[-2:]].set(noisy_finger_qpos)

        # ctrl consistent with qpos to avoid residual torques at reset.
        init_ctrl = jp.array(self._init_ctrl)
        init_ctrl = init_ctrl.at[:6].set(noisy_arm_qpos)
        # Tendon actuator commands the symmetric finger position (coef 0.5 each).
        init_ctrl = init_ctrl.at[6].set(noisy_finger_qpos.sum() * 0.5)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Place the mocap target marker at the drop target.
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(drop_target),
        )

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "success": jp.array(0.0, dtype=float),
            "box_target_dist": jp.array(0.0, dtype=float),
            **{k: jp.array(0.0, dtype=float)
               for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "drop_target": drop_target,
            "step": jp.array(0, dtype=jp.int32),
            "success_counter": jp.array(0, dtype=jp.int32),
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

        raw_rewards = self._get_reward(data, info)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[self._obj_body]
        box_target_dist = jp.linalg.norm(info["drop_target"] - box_pos)

        success_now = box_target_dist < 0.03
        info["success_counter"] = jp.where(
            success_now,
            info["success_counter"] + 1,
            jp.array(0, dtype=jp.int32),
        )
        success = info["success_counter"] >= 3

        tcp_pos = data.site_xpos[self._gripper_site]
        out_of_bounds = (
            jp.any(jp.abs(tcp_pos[:2]) > 0.8) | (tcp_pos[2] < 0.0)
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
        )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        tcp_pos = data.site_xpos[self._gripper_site]
        box_pos = data.xpos[self._obj_body]
        drop_target = info["drop_target"]

        tcp_box_dist = jp.linalg.norm(box_pos - tcp_pos)
        box_target_dist = jp.linalg.norm(drop_target - box_pos)

        reach_box = 1 - jp.tanh(5.0 * tcp_box_dist)
        box_target = 1 - jp.tanh(5.0 * box_target_dist)

        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Floor collision via contact sensors.
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._floor_hand_found_sensor
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = (1 - floor_collision).astype(float)

        return {
            "box_target": box_target,
            "reach_box": reach_box,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Returns 21D obs: [q(6), qd(6), tcp_pos(3), box_pos(3), drop_target(3)]."""
        q = data.qpos[self._robot_arm_qposadr]        # (6,)
        qd = data.qvel[self._robot_arm_qveladr]        # (6,)
        tcp_pos = data.site_xpos[self._gripper_site]   # (3,)
        box_pos = data.xpos[self._obj_body]            # (3,)
        drop_target = info["drop_target"]              # (3,)

        return jp.concatenate([q, qd, tcp_pos, box_pos, drop_target])  # (21,)
