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
        sim_dt=0.002,
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

        # Pull keyframe from config (supports external override)
        init_keyframe = getattr(self._config, "init_keyframe", "low_home")


        self._post_init(obj_name="box", keyframe=init_keyframe)
        self._sample_orientation = sample_orientation
        self._gripper_site = self._mj_model.site("tcp").id
        self._left_finger_touch = (self._mj_model.site("left_finger_touch_site").id,)
        self._right_finger_touch = (self._mj_model.site("right_finger_touch_site").id,)

        # --- No finger sensors on UR10 ---
        self._floor_hand_found_sensor = []

        # Contact sensor IDs (UR10 + Hand-E).
        self._floor_hand_found_sensor = [
            self._mj_model.sensor(name).id
            for name in [
                "left_finger_pad_floor_found",   # ✓ Matches your XML
                "right_finger_pad_floor_found",  # ✓ Matches your XML
                "hand_capsule_floor_found",      # ✓ Matches your XML
            ]
        ]
        print("Available sensors:", flush=True)
        for i in range(self._mj_model.nsensor):
            print(f"  - {self._mj_model.sensor(i).name}", flush=True)


    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target, rng_robot, rng_gripper = jax.random.split(rng, 5)

        # initialize box position
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.0]),
                maxval=jp.array([0.2, 0.2, 0.0]),
            )
            + self._init_obj_pos # Box position from XML Keyframe
        )

        # initialize target position
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.2]),
                maxval=jp.array([0.2, 0.2, 0.4]),
            )
            + self._init_obj_pos # Box position from XML Keyframe
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
        # Get initial arm qpos and add noise
        init_arm_qpos = jp.array(self._init_q[self._robot_arm_qposadr])
        noisy_arm_qpos = init_arm_qpos + robot_qpos_noise

        # Gripper noise (small range since gripper range is 0-0.025)
        gripper_noise = jax.random.uniform(
            rng_gripper,
            (2,),  # left and right finger
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

        # -----------------------------
        # Build initial qpos with randomized arm joints
        # -----------------------------
        init_q = jp.array(self._init_q)
        
        # Set box position
        init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        
        # Set noisy arm joint positions
        init_q = init_q.at[self._robot_arm_qposadr].set(noisy_arm_qpos)
        init_q = init_q.at[self._robot_qposadr[-2:]].set(noisy_finger_qpos)

        # -----------------------------
        # IMPORTANT FIX: make ctrl consistent with init_q - ctrl says: “robot should be somewhere else”
        # -----------------------------
        init_ctrl = jp.array(self._init_ctrl)
        # Update arm control to match noisy arm positions
        init_ctrl = init_ctrl.at[:len(self._robot_arm_qposadr)].set(noisy_arm_qpos)
        # Update gripper control (last actuator)
        init_ctrl = init_ctrl.at[-1].set(noisy_finger_qpos.sum() * 0.5)  # tendon actuator controls sum

        # -----------------------------
        # Create data with CONSISTENT qpos / ctrl
        # -----------------------------
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=init_ctrl,  
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
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4) # Reward clipping to avoid NaNs

        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.2) | (box_pos[2] < 0.0)
        done = (out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(jp.float32)

        metrics = state.metrics
        metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(jp.float32))

        # # env index should be stored in info at reset, e.g. info["env_id"]
        # eid = state.info.get("env_id")  # default to -1 if not found    
        # dummy = jp.array(0, dtype=jp.int32)
        # def _do_print(_):
        #     jax.debug.print(
        #         "[EP END] env={eid} t={t}\n"
        #         "POS tp={tp} bp={bp} gp={gp}\n"
        #         "DIST btd={btd:.4f} re={re:.4f} gbd={gbd:.4f}\n"
        #         "EVT rb={rb} fc={fc} oob={oob}\n"
        #         "REWARDS gb={gb:.3f} bt={bt:.3f} nf={nf:.3f} mp={mp:.3f} TOT={tot:.3f}\n\n",
        #         eid=eid,
        #         t=info["step"],
        #         tp=raw_signals["target_pos"],
        #         bp=raw_signals["box_pos"],
        #         gp=raw_signals["gripper_pos"],
        #         btd=raw_signals["box_target_dist"],
        #         re=raw_signals["rot_err"],
        #         gbd=raw_signals["grip_box_dist"],
        #         rb=raw_signals["reached_box"],
        #         fc=raw_signals["number_floor_collision"],
        #         oob=out_of_bounds.astype(jp.int32),
        #         gb=rewards["gripper_box"],
        #         bt=rewards["box_target"],
        #         nf=rewards["no_floor_collision"],
        #         mp=rewards["robot_target_qpos"],
        #         tot=reward,
        #     )

        #     return dummy

        # print_this = (done > 0.0) & (eid == 0) # only print for env 0 at episode end
        # _ = jax.lax.cond(print_this, _do_print, lambda _: dummy, operand=dummy)

        obs = self._get_obs(data, info)  # <-- use info
        return State(data, obs, reward, done, metrics, info)  # <-- return info

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
        # World-frame rotation matrix of the box - JAX array (3,3) float64
        box_mat = data.xmat[
            self._obj_body
        ]
        # World-frame rotation matrix of the mocap target - JAX array (3,3) float64
        target_mat = math.quat_to_mat(
            data.mocap_quat[self._mocap_target]
        )
        # Rotation error between box and target - scalar JAX float64
        rot_err = jp.linalg.norm(
            target_mat.ravel()[:6] - box_mat.ravel()[:6]
        )  

        # ==============================
        # --- Reward terms ---
        # ==============================
        # Reward for box being at target - scalar JAX float64
        box_target_Reward = 1 - jp.tanh(
            5 * (0.9 * box_target_dist + 0.1 * rot_err)
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
        # rewartd for finger distans large, when distance to boy large
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
            (gripper_box_dist < 0.02),  # Panda threshold was 0.012
        )  

        rewards = {
            "gripper_box": gripper_box_Reward,
            "box_target": box_target_Reward * info["reached_box"],
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
