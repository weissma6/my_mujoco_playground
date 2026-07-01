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
"""UR3 base class (UR3e + Robotiq Hand-E)."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_FINGER_JOINTS = [
    "hande_left_finger_joint",
    "hande_right_finger_joint",
]


def get_assets() -> Dict[str, bytes]:
    """Load XML and mesh assets for UR3e + Hand-E."""
    assets = {}
    path = mjx_env.ROOT_PATH / "manipulation" / "my_ur3" / "xmls"
    mjx_env.update_assets(assets, path, "*.xml")

    path = mjx_env.ROOT_PATH / "manipulation" / "my_ur3" / "universal_robots_ur3e"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        impl="jax",
        nconmax=12 * 8192,
        njmax=44,
    )


class UR3Base(mjx_env.MjxEnv):
    """Base environment for the UR3e + Hand-E (mirrors UR10Base)."""

    def __init__(
        self,
        xml_path: epath.Path,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):

        super().__init__(config, config_overrides)

        self._xml_path = xml_path.as_posix()
        xml = xml_path.read_text()
        self._model_assets = get_assets()

        # Build MuJoCo model
        mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
        mj_model.opt.timestep = self.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
        self._action_scale = config.action_scale

    # Post-init setup (keyframes, IDs, and body references)
    def _post_init(self, obj_name: str = None, keyframe: str = "low_home",
                   finger_joints=None):
        # finger_joints defaults to _FINGER_JOINTS for the pick task
        if finger_joints is None:
            finger_joints = _FINGER_JOINTS
        all_joints = _ARM_JOINTS + finger_joints
        # ------------------------------------------------------------------------------------
        self._robot_arm_qposadr = np.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
                for j in _ARM_JOINTS
            ]
        )
        self._robot_qposadr = np.array(
            [self._mj_model.jnt_qposadr[self._mj_model.joint(j).id] for j in all_joints]
        )
        # ------------------------------------------------------------------------------------
        # The end-effector site (TCP)
        geom_names = [self._mj_model.geom(i).name for i in range(self._mj_model.ngeom)]

        self._gripper_site = self._mj_model.site("tcp").id
        self._left_finger_geom = (
            self._mj_model.geom("left_finger_collision").id
            if "left_finger_collision" in geom_names
            else None
        )
        self._right_finger_geom = (
            self._mj_model.geom("right_finger_collision").id
            if "right_finger_collision" in geom_names
            else None
        )
        self._hand_geom = (
            self._mj_model.geom("hand_base").id if "hand_base" in geom_names else None
        )

        # Finger touch sites on the inner faces of the Hand-E fingers
        # (mirrors ur10pick; used by the UR3PicknPlace / UR3Pick rewards for
        # finger_touch_dist).
        self._left_finger_touch = self._mj_model.site("left_finger_touch_site").id
        self._right_finger_touch = self._mj_model.site("right_finger_touch_site").id

        # Environment object references (like cube) — optional for reach-only tasks
        if obj_name is not None:
            self._obj_body = self._mj_model.body(obj_name).id
            self._obj_qposadr = self._mj_model.jnt_qposadr[
                self._mj_model.body(obj_name).jntadr[0]
            ]
        else:
            self._obj_body = None
            self._obj_qposadr = None

        # Mocap and floor
        self._mocap_target = self._mj_model.body("mocap_target").mocapid
        self._floor_geom = self._mj_model.geom("floor").id

        # Keyframes (initial positions) - with error handling
        try:
            keyframe_id = self._mj_model.key(keyframe).id
            self._init_q = self._mj_model.keyframe(keyframe).qpos
            self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
        except:
            print(f"✗ Keyframe '{keyframe}' not found!")
            print(f"Available keyframes:")
            for i in range(self._mj_model.nkey):
                print(f"  - {self._mj_model.key(i).name}")
            raise ValueError(f"Keyframe '{keyframe}' does not exist in the model")

        # Extract initial object position (None for reach-only tasks)
        if self._obj_qposadr is not None:
            self._init_obj_pos = jp.array(
                self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
                dtype=jp.float32,
            )
        else:
            self._init_obj_pos = None

        # Control limits
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Returns 20D obs: [q(8), (box-tcp)(3), (target-box)(3), box_xmat[:6](6)].

        Canonical observation contract shared by UR3Pick (simple pick) and
        UR3PicknPlace (pick-and-place) — the two tasks differ only in
        reward/termination, never in observation. Hoisted here so it can never
        drift between the subclasses. Relies on handles set in _post_init
        (_robot_qposadr, _obj_body, _gripper_site) and on
        info["target_pos"] (written by each env's reset).
        """
        obs = jp.concatenate(
            [
                data.qpos[self._robot_qposadr],                                # 8 (6 arm + 2 finger)
                data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],  # 3 box - tcp
                info["target_pos"] - data.xpos[self._obj_body],                # 3 target - box
                data.xmat[self._obj_body].ravel()[:6],                         # 6 box orientation (rows 0-1)
            ]
        )
        return obs

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu  # 7 (6 arm + 1 gripper actuator)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
