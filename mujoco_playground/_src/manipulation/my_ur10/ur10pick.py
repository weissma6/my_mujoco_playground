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

        # ------------------------------------------------------------------------------------

        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "my_ur10"
            / "xmls"
            / "mjx_single_cube_position.xml"
        )
        # print("XML PATH:", xml_path, flush=True)


        # ------------------------------------------------------------------------------------

        super().__init__(
            xml_path,
            config,
            config_overrides,
        )
        self._post_init(obj_name="box", keyframe="low_home")
        self._sample_orientation = sample_orientation
        self._gripper_site = (self._mj_model.site("tcp").id,)

        # --- No finger sensors on UR10 ---
        self._floor_hand_found_sensor = []

        # Contact sensor IDs (UR10 + Hand-E).
        self._floor_hand_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_touch",
                "right_finger_touch",
                "gripper_floor_contact",
            ]
        ]

        # print("Model has keyframe support:", hasattr(self._mj_model, "keyframe"), flush=True)


    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # initialize box position
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.15, -0.15, 0.0]),
                maxval=jp.array([0.15, 0.15, 0.0]),
            )
            + self._init_obj_pos
        )

        # initialize target position
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.4]),
                maxval=jp.array([0.2, 0.2, 0.6]),
            )
            + self._init_obj_pos
        )

        # Optional orientation sampling
        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self._sample_orientation:
            rng, rng_axis, rng_theta = jax.random.split(rng, 3)
            perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
            perturb_axis = perturb_axis / math.norm(perturb_axis)
            perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
            target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

        # -----------------------------
        # Build initial qpos - qpos says: “robot is at home pose”
        # -----------------------------
        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(box_pos)
        )

        # Use home pose for all arm joints at reset
        init_q = init_q.at[self._robot_qposadr].set(self._init_q[self._robot_qposadr])

        # -----------------------------
        # IMPORTANT FIX: make ctrl consistent with init_q - ctrl says: “robot should be somewhere else”
        # -----------------------------
        init_ctrl = jp.array(self._init_ctrl)

        # Arm joints: ctrl = joint positions
        init_ctrl = init_ctrl.at[:6].set(init_q[self._robot_arm_qposadr])

        # Gripper: open at reset
        init_ctrl = init_ctrl.at[6].set(self._uppers[6])

        # -----------------------------
        # Create data with CONSISTENT qpos / ctrl
        # -----------------------------
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,  # ✅ FIXED
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )



        # set target mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": 0.0,
            "step": jp.array(0, dtype=jp.int32),        # step counter for debug printing
            "debug_every": jp.array(10, dtype=jp.int32) # print every N steps (set N here)
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        state = State(data, obs, reward, done, metrics, info)


        # -- print debug block --
        # --- robot joint positions (arm + gripper) ---
        robot_qpos = data.qpos[self._robot_qposadr]   # shape (7 or 8)

        # --- box position from qpos (freejoint translation) ---
        box_qpos = data.qpos[self._obj_qposadr : self._obj_qposadr + 3]

        # --- controls ---
        robot_ctrl = data.ctrl[: robot_qpos.shape[0]]

        # --- consistency error ---
        ctrl_qpos_err = jp.linalg.norm(
            data.ctrl[:6] - data.qpos[self._robot_arm_qposadr]
        )

        # debug.print(
        #     "[RESET] robot_qpos={rq} | box_qpos={bq} | ctrl={c} | ctrl-qpos-err={e}",
        #     rq=robot_qpos,
        #     bq=box_qpos,
        #     c=robot_ctrl,
        #     e=ctrl_qpos_err,
        # )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info = dict(state.info)
        info["step"] = info["step"] + 1

        raw_rewards, raw_signals = self._get_reward(data, info)  # <-- pass info
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)
        done = (out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(jp.float32)

        metrics = state.metrics
        metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(jp.float32))

        dummy = jp.array(0, dtype=jp.int32)
        def _do_print(_):
            jax.debug.print(
                "[EP END] t={t} pos_err={pe:.4f} rot_err={re:.4f} dist={d:.4f} "
                "reached={rb} floor={fc} oob={oob} "
                "gb={gb:.3f} bt={bt:.3f} nf={nf:.3f} ah={ah:.3f} TOTAL={tot<[EP END] t={t} ... TOTAL={tot:.3f}",
                t=info["step"],
                pe=raw_signals["pos_err"],
                re=raw_signals["rot_err"],
                d=raw_signals["grip_box_dist"],
                rb=raw_signals["reached_box"],
                fc=raw_signals["floor_collision"],
                oob=out_of_bounds.astype(jp.int32),
                gb=rewards["gripper_box"],
                bt=rewards["box_target"],
                nf=rewards["no_floor_collision"],
                ah=rewards["robot_target_qpos"],
                tot=reward,
            )
            return dummy

        _ = jax.lax.cond(done > 0.0, _do_print, lambda _: dummy, operand=dummy)

        obs = self._get_obs(data, info)  # <-- use info
        return State(data, obs, reward, done, metrics, info)  # <-- return info

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        # --- target + box pose errors ---
        target_pos = info[
            "target_pos"
        ]  # Endposition of the mocap target - JAX array (3,) float64
        box_pos = data.xpos[
            self._obj_body
        ]  # Current position of the box - JAX array (3,) float64
        gripper_pos = data.site_xpos[
            self._gripper_site
        ]  # World-frame Cartesian position of the TCP site - JAX array (3,) float64
        pos_err = jp.linalg.norm(
            target_pos - box_pos
        )  # Euclidean distance between box and target - scalar JAX float64

        # --- Rotation related computations ---
        box_mat = data.xmat[
            self._obj_body
        ]  # World-frame rotation matrix of the box - JAX array (3,3) float64
        target_mat = math.quat_to_mat(
            data.mocap_quat[self._mocap_target]
        )  # World-frame rotation matrix of the mocap target - JAX array (3,3) float64
        rot_err = jp.linalg.norm(
            target_mat.ravel()[:6] - box_mat.ravel()[:6]
        )  # Rotation error between box and target - scalar JAX float64

        # --- Reward terms identical to Panda, only gripper_pos definition changed --
        box_target = 1 - jp.tanh(
            5 * (0.9 * pos_err + 0.1 * rot_err)
        )  # Reward for box being at target - scalar JAX float64
        gripper_box = 1 - jp.tanh(
            5 * jp.linalg.norm(box_pos - gripper_pos)
        )  # Reward for gripper being at box - scalar JAX float64
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )  # Penalty for deviating too far from the initial arm configuration - scalar JAX float64

        # --- Floor collision via touch sensors (same as Panda) ---
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._floor_hand_found_sensor  # List of booleans indicating if each sensor detects contact with the floor
        ]
        floor_collision = (
            sum(hand_floor_collision) > 0
        )  # Boolean indicating if any sensor detects contact with the floor
        no_floor_collision = (1 - floor_collision).astype(
            float
        )  # Reward for no floor collision - scalar JAX float64

        # --- Same "reached_box" gate as Panda, but based on fingertip midpoint ---
        info["reached_box"] = 1.0 * jp.maximum(
            info["reached_box"],
            (
                jp.linalg.norm(box_pos - gripper_pos) < 0.012
            ),  # Panda threshold was 0.012
        )  # Binary indicator if the gripper has reached the box - scalar JAX float64

        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target * info["reached_box"],
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        # ==============================
        # Raw signals dict (for debug)
        # ==============================
        raw = {
            # positions
            "target_pos": target_pos,
            "box_pos": box_pos,
            "gripper_pos": gripper_pos,

            # errors / distances
            "pos_err": pos_err,
            "rot_err": rot_err,
            "grip_box_dist": gripper_box,

            # events
            "reached_box": info["reached_box"],
            "floor_collision": floor_collision,
        }

        return rewards, raw

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_mat = data.site_xmat[self._gripper_site].ravel()
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        # ------------------------------------------------------------------------------------
        # Test for joint observation shapes
        # print("Action Space Size:", self.action_size)
        # print("ctrl:", data.ctrl.shape)
        # print("robot_qpos subset:", data.qpos[self._robot_qposadr[:-1]].shape)
        # ------------------------------------------------------------------------------------
        obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
                gripper_pos,
                gripper_mat[3:],
                data.xmat[self._obj_body].ravel()[3:],
                data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
                info["target_pos"] - data.xpos[self._obj_body],
                target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
                data.ctrl
                - data.qpos[
                    self._robot_qposadr[:-1]
                ],  # the is one freejoint for the box to move that has no actuator
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
