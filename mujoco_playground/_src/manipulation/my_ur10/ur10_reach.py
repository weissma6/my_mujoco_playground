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
"""UR10 simplified reach task: 6-DOF arm only, TCP -> mocap_target.

Observation (18D) matches RDTE receive package output:
  [q(6), qd(6), tcp_pos(3), target_pos(3)]
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.my_ur10 import ur10_base
from mujoco_playground._src.mjx_env import State


def default_config() -> config_dict.ConfigDict:
    """Default config for UR10 simple reach task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach_target=8.0,
            )
        ),
        impl="jax",
        nconmax=12 * 8192,
        njmax=44,
        init_keyframe="low_home",
    )


class UR10SimpleReach(ur10_base.UR10Base):
    """6-DOF arm-only reach task. No gripper, no box. 18D observation.

    Observation matches RDTE receive package:
      [q(6), qd(6), tcp_pos(3), target_pos(3)]
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "my_ur10"
            / "xmls"
            / "mjx_reach.xml"
        )

        super().__init__(xml_path, config, config_overrides)

        init_keyframe = getattr(self._config, "init_keyframe", "low_home")

        # obj_name=None: no box body; finger_joints=[]: arm-only model
        self._post_init(obj_name=None, keyframe=init_keyframe, finger_joints=[])

        self._gripper_site = self._mj_model.site("tcp").id

    def reset(self, rng: jax.Array) -> State:
        rng, rng_target, rng_robot = jax.random.split(rng, 3)

        # Randomize target position in a reachable workspace volume
        # UR10e reach: x=[0.3, 0.8], y=[-0.4, 0.4], z=[0.2, 0.8]
        target_pos = jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([0.3, -0.4, 0.2]),
            maxval=jp.array([0.8,  0.4, 0.8]),
        )

        # Randomize arm joint positions (small noise around keyframe)
        robot_qpos_noise = jax.random.uniform(
            rng_robot,
            (len(self._robot_arm_qposadr),),
            minval=-0.05,
            maxval=0.05,
        )
        init_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
        noisy_arm_qpos = init_arm_qpos + robot_qpos_noise

        # Build qpos (6D only for this model)
        init_q = jp.array(self._init_q)
        init_q = init_q.at[self._robot_arm_qposadr].set(noisy_arm_qpos)

        # ctrl consistent with qpos to avoid residual torques at reset
        init_ctrl = jp.array(self._init_ctrl)
        init_ctrl = init_ctrl.at[:6].set(noisy_arm_qpos)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Place mocap target
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )

        metrics = {
            "success": jp.array(0.0, dtype=float),
            "tcp_target_dist": jp.array(0.0, dtype=float),
            **{k: jp.array(0.0, dtype=float)
               for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "step": jp.array(0, dtype=jp.int32),
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

        tcp_pos = data.site_xpos[self._gripper_site]
        tcp_target_dist = jp.linalg.norm(info["target_pos"] - tcp_pos)

        success = tcp_target_dist < 0.0005  # 0.5mm threshold for successs

        out_of_bounds = (
            jp.any(jp.abs(tcp_pos[:2]) > 1.2) | (tcp_pos[2] < 0.0)
        )
        invalid_state = (
            jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )
        done = (success | out_of_bounds | invalid_state).astype(jp.float32)

        metrics = state.metrics
        metrics.update(
            **raw_rewards,
            success=success.astype(jp.float32),
            tcp_target_dist=tcp_target_dist,
        )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        tcp_pos = data.site_xpos[self._gripper_site]
        target_pos = info["target_pos"]
        tcp_target_dist = jp.linalg.norm(target_pos - tcp_pos)

        reach_reward = 1 - jp.tanh(5.0 * tcp_target_dist)

        return {"reach_target": reach_reward}

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Returns 18D observation: [q(6), qd(6), tcp_pos(3), target_pos(3)].

        Matches RDTE receive package layout:
          q        — getActualQ()       [rad]
          qd       — getActualQd()      [rad/s]
          tcp_pos  — getActualTCPPose() [:3]  [m]
          target_pos — provided externally (goal)
        """
        q = data.qpos[self._robot_arm_qposadr]        # (6,)
        qd = data.qvel[self._robot_arm_qveladr]        # (6,)
        tcp_pos = data.site_xpos[self._gripper_site]   # (3,)
        target_pos = info["target_pos"]                # (3,)

        return jp.concatenate([q, qd, tcp_pos, target_pos])  # (18,)
