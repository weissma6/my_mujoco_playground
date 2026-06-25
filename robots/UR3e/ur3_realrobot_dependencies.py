"""
RTDE dependencies for the UR3 + Hand-E pick task.

Mirrors robots/URSim/URSim_RTDE_dependencies.py (the UR10 reach backbone)
but adapted for the UR3PicknPlace policy:

- 26D observation  [q(8), qd(6), (box-tcp)(3), (target-box)(3), box_xmat[:6](6)]
                   q = 6 arm joints + 2 finger positions; box_xmat[:6] = first
                   two rows of the box rotation matrix (from the mocap quaternion)
- 7D action        6 arm joint deltas + 1 Hand-E gripper delta
- box pose comes from a Nokov rigid body (motion_capture.mymocap.mocap_dependencies):
  xyz via get_rigid_body_xyz(), orientation via get_rigid_body_quat()
- the lift target is supplied by the caller (hardcoded in the pick-loop script)
- the ARM and the GRIPPER connect on two separate, independent channels:
    * connect_arm()     — RTDE receive + the External Control URCap (PolyScope X,
                          port ur_cap_port, default 50002). This is the ONLY arm
                          control path; it BLOCKS until the External Control
                          program is PLAYING on the pendant.
    * connect_gripper() — the Hand-E via robots/hande/HandEGripper (Robotiq URCapX
                          XML-RPC, http://<host>:49999/, slaveId 9). servoJ takes
                          only the 6 arm joints, so the gripper rides this channel.
  All gripper logic lives in HandEGripper; the gripper methods here just delegate.

The UR10 reach files are left untouched; this is a sibling, not a replacement.
"""

from typing import Dict, List, Optional, Tuple
import os
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd
import rtde_control
import rtde_receive

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground import registry


class UR3RealRobotPick:
    """RTDE control loop for the UR3 pick task (6-DOF arm + Hand-E, 26D obs)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port_rtde: int = 30004,
        use_ext_urcap: bool = True,
        ur_cap_port: int = 50002,
    ):
        self.host = host
        self.port_rtde = port_rtde
        # ARM channel. The arm ALWAYS connects through the External Control URCap:
        # connect_arm() attaches to the pendant's External Control program (set to
        # this PC's IP + ur_cap_port) and blocks until it is PLAYING. The old
        # script-upload path (PolyScope 5 / URSim) has been removed; use_ext_urcap
        # is kept only so existing call sites still construct — it no longer
        # switches the path.
        self._ur_cap_port = int(ur_cap_port)
        self._receiver: Optional[rtde_receive.RTDEReceiveInterface] = None
        self._control: Optional[rtde_control.RTDEControlInterface] = None
        self._policy_fn = None
        self._ctrl_lowers: Optional[np.ndarray] = None
        self._ctrl_uppers: Optional[np.ndarray] = None
        # Internal gripper command tracker (the obs does not include gripper
        # state, so we integrate the policy's gripper action ourselves).
        self._gripper_ctrl: float = 0.0
        self._gripper_lo: float = 0.0
        self._gripper_hi: float = 0.05
        # GRIPPER channel. The real Hand-E is driven by robots/hande/HandEGripper
        # (Robotiq URCapX XML-RPC, HTTP :49999, slaveId 9, percent units). ALL
        # gripper logic (incl. the verified sim<->percent mapping) lives there;
        # self._gripper holds that instance once connect_gripper() is called.
        self._gripper = None  # HandEGripper once connect_gripper() is called
        self._gripper_xmlrpc_port: int = 49999
        self._gripper_slave_id: int = 9
        self._gripper_speed_pct: int = 100
        self._gripper_force_pct: int = 50
        # Base-frame calibration (mocap world -> robot base). Auto-loaded from
        # calibration/base_frame_calibration.json if present; applied to every
        # mocap reading so the box pose lands in the policy frame (see
        # mocap_pos_to_base / run_policy_loop). None => mocap used raw (identity).
        self._cal_p0 = None     # base origin in world
        self._cal_R0T = None    # world->base rotation (R0.T)
        self._cal_q_w2s = None  # world->sim-base quaternion (incl. the X/Y negation)
        self.load_base_calibration()

    # =========================================================================
    # A — RTDE Connection & Feedback
    # =========================================================================

    def connect_arm(self) -> rtde_receive.RTDEReceiveInterface:
        """Connect the ARM channel: RTDE receive + the External Control URCap.

        The arm has exactly one control path: ur_rtde does NOT upload its own
        script (PolyScope X would not run it); it attaches to the pendant's
        External Control program (configured to this PC's IP + ur_cap_port,
        default 50002). This BLOCKS until that program is PLAYING with the robot
        in Remote Control — pressing Play on the pendant is the robot-side "go".
        Independent of the gripper; see connect_gripper().
        """
        if self._receiver is None:
            self._receiver = rtde_receive.RTDEReceiveInterface(self.host)
        if self._control is None:
            flags = rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP
            self._control = rtde_control.RTDEControlInterface(
                self.host, -1.0, flags, self._ur_cap_port
            )
        return self._receiver

    def connect(self) -> rtde_receive.RTDEReceiveInterface:
        """Back-compat alias for connect_arm() (used by receive_feedback,
        send_movej, and the pick-loop / calibration scripts)."""
        return self.connect_arm()

    def disconnect(self):
        # Gripper channel: drop the HandEGripper handle (XML-RPC holds no
        # persistent socket); leaves the gripper activated on the controller.
        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                pass
            self._gripper = None
        if self._control is not None:
            try:
                self._control.servoStop()
                self._control.stopScript()
            except Exception:
                pass
            self._control.disconnect()
            self._control = None
        if self._receiver is not None:
            self._receiver.disconnect()
            self._receiver = None

    def is_connected(self) -> bool:
        if self._receiver is None or self._control is None:
            return False
        return self._receiver.isConnected() and self._control.isConnected()

    def receive_feedback(self) -> Dict[str, List[float]]:
        """Returns dict with joint state, TCP, currents, control output, force."""
        r = self.connect()
        q = r.getActualQ()
        qd = r.getActualQd()
        tcp_pose = r.getActualTCPPose()
        # X/Y negated to match the MuJoCo base-frame orientation (same
        # convention as the UR10 reach loop).
        tcp_pose[0] = -tcp_pose[0]
        tcp_pose[1] = -tcp_pose[1]
        current = r.getActualCurrent()
        ctrl_output = r.getJointControlOutput()
        tcp_force = r.getActualTCPForce()
        return {
            "q": list(q),
            "qd": list(qd),
            "tcp_xyz": list(tcp_pose[:3]),
            "tcp_pose": list(tcp_pose),
            "current": list(current),
            "ctrl_output": list(ctrl_output),
            "tcp_force": list(tcp_force),
        }

    def print_feedback(self, digits: int = 4):
        fb = self.receive_feedback()
        print("q      =", [round(v, digits) for v in fb["q"]])
        print("qd     =", [round(v, digits) for v in fb["qd"]])
        print("tcp_xyz=", [round(v, digits) for v in fb["tcp_xyz"]])

    # =========================================================================
    # B — Motion Commands via RTDEControlInterface
    # =========================================================================

    def send_movej(self, q, a=0.4, v=0.4, asynchronous=True, textmsg=None):
        """moveJ via rtde_control.

        asynchronous=True returns immediately (for sweeps that record while the
        joint travels); asynchronous=False blocks until the controller reports the
        move converged (repositioning — what the old move_to_start poll loop did).
        NOTE: moveJ arg order is (q, speed, acceleration) — v BEFORE a. Don't flip.
        """
        self.connect()
        if textmsg is not None:
            print(f"[movej] {textmsg}")
        self._control.moveJ(list(map(float, q)), float(v), float(a), bool(asynchronous))

    def send_servoj(self, q, a=1.4, v=1.05, t=0.04, lookahead_time=0.1, gain=300,
                    textmsg=None):
        """Non-blocking servoJ via rtde_control (6 arm joints only).

        NOTE: rtde_control.servoJ argument order is (q, speed, acceleration,
        time, lookahead_time, gain) — speed (v) BEFORE acceleration (a),
        opposite to URScript's servoj(q, a, v, ...). Don't flip these.
        """
        if textmsg is not None:
            print(f"[servoj] {textmsg}")
        self._control.servoJ(
            list(map(float, q)),
            float(v),
            float(a),
            float(t),
            float(lookahead_time),
            float(gain),
        )

    def connect_gripper(self, slave_id: int = None, speed: int = None,
                        force: int = None, reset: bool = False):
        """Connect + activate the real Hand-E on its own (gripper) channel.

        Thin wrapper: builds a robots/hande/HandEGripper (Robotiq URCapX XML-RPC,
        http://<host>:49999/, slaveId 9, percent units) and connects it. ALL
        gripper logic — the verified sim<->percent mapping, open/close, readback —
        lives in HandEGripper; this object only holds the instance so the pick
        loop can drive arm + gripper through one handle. Independent of the arm:
        call it before, after, or without connect_arm(). Errors propagate so a
        missing URCapX server fails loudly.

        (PolyScope 5 / CB-series instead use the 63352 socket; see
        robots/random_sample_code/robotiq_gripper.py, not used here.)
        """
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from robots.hande.HandE_dependency import HandEGripper

        if slave_id is not None:
            self._gripper_slave_id = int(slave_id)
        if speed is not None:
            self._gripper_speed_pct = int(speed)
        if force is not None:
            self._gripper_force_pct = int(force)
        g = HandEGripper(
            self.host,
            port=self._gripper_xmlrpc_port,
            slave_id=self._gripper_slave_id,
            speed_pct=self._gripper_speed_pct,
            force_pct=self._gripper_force_pct,
        )
        g.connect(reset=reset)
        self._gripper = g
        return g

    def _require_gripper(self):
        if self._gripper is None:
            raise RuntimeError(
                "Gripper not connected. Call connect_gripper() before "
                "send_gripper()/open_gripper()/close_gripper()/read_gripper_state()."
            )
        return self._gripper

    def send_gripper(self, norm_cmd: float):
        """Send a normalized [0,1] gripper command to the real Hand-E.

        norm 0.0 = OPEN, 1.0 = CLOSED — matches run_policy_loop's gripper_norm and
        the sim per-finger range [0, 0.025] m (norm == finger_meters / 0.025).
        Delegates to HandEGripper.command (which owns the verified sim->percent
        mapping); norm maps to sim meters as norm * 0.025. servoJ controls only
        the 6 arm joints, so the gripper rides this separate XML-RPC channel —
        rate-limit to <=10 Hz in tight loops (see run_policy_loop).

        Requires connect_gripper() first (raises otherwise).
        """
        g = self._require_gripper()
        norm = float(np.clip(norm_cmd, 0.0, 1.0))
        g.command(norm * 0.025)

    def open_gripper(self):
        """Fully open the Hand-E (sim 0). Direction-independent URCapX call."""
        self._require_gripper().open_gripper()

    def close_gripper(self):
        """Fully close the Hand-E (sim 0.025). Direction-independent URCapX call."""
        self._require_gripper().close_gripper()

    def read_gripper_state(self) -> Dict[str, float]:
        """Read back the real Hand-E state (no motion command sent).

        Delegates to HandEGripper.read_state(), returning:
          pos_pct     getCurrentPosition (0-100)
          obj_flag    getObjectDetectionFlag (0 none, 1 on-open, 2 on-close)
          grasped     obj_flag in (1, 2)
          sim_finger  per-finger meters [0, 0.025] (0 = open, 0.025 = closed),
                      usable as real gripper feedback for the 26D obs
          fault / activated / connected
        (open_frac, 1 = open ... 0 = closed, is added for the bring-up notebook.)
        """
        state = self._require_gripper().read_state()
        state["open_frac"] = 1.0 - state["sim_finger"] / 0.025
        return state

    # =========================================================================
    # C — Movement Helpers
    # =========================================================================

    def joint_error_norm(self, q_target) -> float:
        fb = self.receive_feedback()
        q = np.asarray(fb["q"], dtype=float)
        return float(np.linalg.norm(np.asarray(q_target, dtype=float) - q))

    def reached_joint_target(self, q_target, tol=0.05) -> bool:
        return self.joint_error_norm(q_target) < tol

    # =========================================================================
    # D — Policy Loading
    # =========================================================================

    @staticmethod
    def download_policy_from_wandb(
        run_id: str,
        out_dir: str,
        entity: str = "weissma6-zhaw-school-of-engineering",
        project: str = "UR3_pick_ppo",
        force: bool = False,
    ) -> str:
        """Download a trained PPO policy from W&B into a load_policy_fn-ready dir.

        Thin wrapper kept for backward compatibility (notebooks call it). The
        single implementation lives in the dependency-light
        evaluation/downloaded_policies/policy_downloader module.
        """
        downloader_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../evaluation/downloaded_policies",
        )
        if downloader_dir not in sys.path:
            sys.path.insert(0, downloader_dir)
        from policy_downloader import download_policy

        return download_policy(
            run_id, out_dir, entity=entity, project=project, force=force
        )

    def load_policy_fn(self, policy_path: str, deterministic: bool = True):
        """Load a Brax PPO policy from saved files (metadata.json + params.msgpack).

        Everything (env_name, obs_dim=26, action_dim=7, network architecture)
        is read from the saved artifacts. Same pattern as the UR10 reach loop.
        """
        import json as _json
        from flax import serialization

        meta_file = os.path.join(policy_path, "metadata.json")
        with open(meta_file) as f:
            meta = _json.load(f)

        nf_kwargs = {
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in meta["network_factory"].items()
        }

        ppo_net = ppo_networks.make_ppo_networks(
            observation_size=meta["obs_dim"],
            action_size=meta["action_dim"],
            preprocess_observations_fn=running_statistics.normalize,
            **nf_kwargs,
        )

        rng_init = jax.random.PRNGKey(0)
        template = {
            "0": running_statistics.init_state(
                jax.ShapeDtypeStruct((meta["obs_dim"],), jnp.float32)
            ),
            "1": ppo_net.policy_network.init(rng_init),
            "2": ppo_net.value_network.init(rng_init),
        }

        with open(os.path.join(policy_path, "params.msgpack"), "rb") as f:
            loaded_params = serialization.from_bytes(template, f.read())

        _raw_policy = ppo_networks.make_inference_fn(ppo_net)(
            (loaded_params["0"], loaded_params["1"]), deterministic=deterministic
        )

        @jax.jit
        def policy_fn(obs):
            action, _ = _raw_policy(obs, jax.random.PRNGKey(0))
            return action

        self._policy_fn = policy_fn
        self._meta = meta
        self.policy_path = policy_path

        # Actuator ctrl limits (arm + gripper) from the env model.
        env = registry.load(meta["env_name"])
        lowers, uppers = env.mj_model.actuator_ctrlrange.T
        self._ctrl_lowers = np.array(lowers, dtype=np.float32)
        self._ctrl_uppers = np.array(uppers, dtype=np.float32)
        # Gripper actuator is the last one (tendon "split", ctrlrange 0..0.05).
        self._gripper_lo = float(self._ctrl_lowers[-1])
        self._gripper_hi = float(self._ctrl_uppers[-1])

        # JIT warmup.
        t0 = time.perf_counter()
        dummy = np.zeros((1, meta["obs_dim"]), dtype=np.float32)
        _ = policy_fn(dummy)
        _.block_until_ready()
        print(f"Loaded policy: obs={meta['obs_dim']}, act={meta['action_dim']} "
              f"(JIT warmup: {time.perf_counter() - t0:.2f}s)")

    # =========================================================================
    # E — MuJoCo FK model (optional: computes tcp_pos from joint angles)
    # =========================================================================

    def init_fk_model(self, xml_path: str):
        """Load a MuJoCo model for forward kinematics of the tcp site."""
        self._fk_model = mujoco.MjModel.from_xml_path(xml_path)
        self._fk_data = mujoco.MjData(self._fk_model)
        self._fk_tcp_id = mujoco.mj_name2id(
            self._fk_model, mujoco.mjtObj.mjOBJ_SITE, "tcp"
        )
        print(f"FK model loaded: tcp site id={self._fk_tcp_id}")

    def compute_tcp_pos(self, q: np.ndarray) -> np.ndarray:
        """Compute MuJoCo tcp site position from the 6 arm joint angles via FK."""
        self._fk_data.qpos[:6] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self._fk_model, self._fk_data)
        return self._fk_data.site_xpos[self._fk_tcp_id].copy()

    # =========================================================================
    # F — Observation Building (26D)
    # =========================================================================

    @staticmethod
    def quat_to_xmat_flat(quat: Optional[np.ndarray]) -> np.ndarray:
        """Convert a (w, x, y, z) quaternion to the first two rows of its 3x3
        rotation matrix, flattened to (6,). Returns identity rows if quat is None.
        """
        if quat is None:
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        w, x, y, z = (float(v) for v in quat)
        n = (w * w + x * x + y * y + z * z) ** 0.5
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        w, x, y, z = w / n, x / n, y / n, z / n
        mat = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return mat.ravel()[:6]

    def build_obs_from_feedback(
        self,
        fb: dict,
        box_pos: np.ndarray,
        target_pos: np.ndarray,
        tcp_pos: Optional[np.ndarray] = None,
        box_quat: Optional[np.ndarray] = None,
        dtype=np.float32,
    ) -> np.ndarray:
        """Build 26D obs to match the canonical UR3Base._get_obs (shared by
        UR3PicknPlace, the deployment target, and UR3Pick):
        [q(8), qd(6), (box-tcp)(3), (target-box)(3), box_xmat[:6](6)].

        q(8) = 6 arm joints (RTDE) + 2 finger positions (from the internal gripper
        tracker; the real robot has no finger encoder). qd(6) = arm velocities.
        tcp_pos defaults to RTDE getActualTCPPose()[:3] (X/Y already negated in
        receive_feedback); pass an FK-computed tcp_pos to override. box_xmat[:6]
        is derived from the mocap quaternion (identity rows when box_quat is None).
        Returns shape (1, 26) for batched policy input.
        """
        arm_q = np.array(fb["q"], dtype=dtype)
        finger_q = np.full(2, float(self._gripper_ctrl), dtype=dtype)
        q = np.concatenate([arm_q, finger_q])                       # 8
        qd = np.array(fb["qd"], dtype=dtype)                        # 6 (arm)
        tcp = (np.array(fb["tcp_xyz"], dtype=dtype) if tcp_pos is None
               else np.array(tcp_pos, dtype=dtype))
        box = np.array(box_pos, dtype=dtype)
        tgt = np.array(target_pos, dtype=dtype)
        box_to_tcp = box - tcp                                      # 3
        target_to_box = tgt - box                                  # 3
        box_xmat_flat = self.quat_to_xmat_flat(box_quat).astype(dtype)  # 6
        obs = np.concatenate([q, qd, box_to_tcp, target_to_box, box_xmat_flat])
        return obs[None, :]  # (1, 26)

    # =========================================================================
    # F2 — Mocap base-frame calibration (mocap world -> policy/base frame)
    # =========================================================================

    # Raw UR base frame -> MuJoCo/sim base frame: the same X/Y negation
    # receive_feedback() applies to the TCP (the sim base is 180 deg about Z from
    # the real UR base). The calibration maps mocap -> RAW base (its probes use the
    # raw TCP pose), so this negation is needed for the box to share the TCP frame.
    _XY_NEG = np.array([-1.0, -1.0, 1.0])

    def load_base_calibration(self, json_path: Optional[str] = None) -> bool:
        """Load base_frame_calibration.json (mocap world -> robot base).

        Default path: calibration/base_frame_calibration.json next to this file.
        Sets the transform applied to every mocap reading. If the file is absent,
        the mocap is used raw (identity) and a note is printed. Returns True if a
        calibration was loaded.
        """
        import json as _json

        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "calibration", "base_frame_calibration.json",
            )
        if not os.path.exists(json_path):
            print(f"[cal] no base-frame calibration at {json_path}; mocap used raw.")
            self._cal_p0 = self._cal_R0T = self._cal_q_w2s = None
            return False
        with open(json_path) as f:
            cal = _json.load(f)
        self._cal_p0 = np.array(cal["translation_m"], dtype=float)
        self._cal_R0T = np.array(cal["rotation_matrix"], dtype=float).T  # world->base
        # world->sim quaternion = q_D (x) conj(q_R0): conj(q_R0) is world->raw-base
        # rotation, q_D = 180 deg about Z folds in the X/Y negation so orientation
        # lands in the same (sim) frame as the negated TCP.
        qw, qx, qy, qz = (float(v) for v in cal["rotation_quat_wxyz"])
        q_r0_conj = np.array([qw, -qx, -qy, -qz])
        q_d = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about Z
        self._cal_q_w2s = self._quat_mult(q_d, q_r0_conj)
        print(f"[cal] loaded base-frame calibration from {json_path}")
        return True

    @staticmethod
    def _quat_mult(a, b) -> np.ndarray:
        """Hamilton product of two (w, x, y, z) quaternions."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dtype=float)

    def mocap_pos_to_base(self, p_world):
        """Map a mocap-world position into the policy/base frame: apply the
        calibration (world->raw base) then the X/Y negation (raw->sim base, to
        match the TCP). Identity if no calibration is loaded or p_world is None.
        """
        if self._cal_R0T is None or p_world is None:
            return p_world
        p_raw = self._cal_R0T @ (np.asarray(p_world, dtype=float) - self._cal_p0)
        return p_raw * self._XY_NEG

    def mocap_quat_to_base(self, quat_world):
        """Map a mocap-world (w, x, y, z) quaternion into the policy/base frame.
        Identity if no calibration is loaded or quat_world is None.
        """
        if self._cal_q_w2s is None or quat_world is None:
            return quat_world
        return self._quat_mult(self._cal_q_w2s, np.asarray(quat_world, dtype=float))

    # =========================================================================
    # G — Control Update
    # =========================================================================

    def policy_step_ctrl_update(
        self,
        q: np.ndarray,
        action: np.ndarray,
        action_scale: float = 0.04,
        alpha: float = 1.0,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, float]:
        """Compute arm servoj target and gripper command from a 7D action.

        Returns (arm_ctrl(6,), gripper_norm) where gripper_norm is in [0,1].
        The gripper command is integrated internally (obs has no gripper state).
        """
        action = np.asarray(action, dtype=dtype).reshape(-1)
        q = np.asarray(q, dtype=dtype)

        # --- Arm: delta on measured joint positions ---
        arm_ctrl = q + float(action_scale) * action[:6]
        if self._ctrl_lowers is not None and self._ctrl_uppers is not None:
            arm_ctrl = np.clip(arm_ctrl, self._ctrl_lowers[:6], self._ctrl_uppers[:6])
        if alpha < 1.0:
            arm_ctrl = alpha * arm_ctrl + (1.0 - alpha) * q

        # --- Gripper: integrate the 7th action, clip to actuator range ---
        self._gripper_ctrl = float(
            np.clip(
                self._gripper_ctrl + float(action_scale) * float(action[6]),
                self._gripper_lo,
                self._gripper_hi,
            )
        )
        span = max(self._gripper_hi - self._gripper_lo, 1e-9)
        gripper_norm = (self._gripper_ctrl - self._gripper_lo) / span

        return arm_ctrl, gripper_norm

    # =========================================================================
    # H — Control Loop
    # =========================================================================

    def run_policy_loop(
        self,
        drop_target: np.ndarray,
        mocap_reader,
        control_hz: float = 50.0,
        timeout_s: float = 15.0,
        action_scale: float = 0.04,
        lookahead_time: float = 0.1,
        gain: int = 300,
        servoj_a: float = 1.4,
        servoj_v: float = 1.05,
        alpha: float = 1.0,
        gripper_fn=None,
        use_fk_tcp: bool = False,
        reach_tol: float = None,
        dwell_time_s: float = 0.0,
        dtype=np.float32,
        debug_print: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """Run the UR3 pick policy loop.

        Args:
          drop_target: (3,) world-frame drop location (meters).
          mocap_reader: NokovRigidBodyReader; get_rigid_body_xyz() -> box pos.
          gripper_fn: callable(norm_cmd in [0,1]); defaults to self.send_gripper.
          use_fk_tcp: if True, tcp_pos is computed via FK from joint angles
                      (requires init_fk_model); else RTDE TCP pose is used.
        Returns (per-step DataFrame, summary stats dict).
        """
        if self._policy_fn is None:
            raise RuntimeError("Policy not loaded. Call load_policy_fn() first.")
        if gripper_fn is None:
            gripper_fn = self.send_gripper

        dt = 1.0 / float(control_hz)
        drop_target = np.asarray(drop_target, dtype=np.float32).reshape(3)
        # Seed the internal gripper tracker from the keyframe ctrl if available.
        self._gripper_ctrl = float(getattr(self, "_gripper_ctrl", 0.0))

        log = []
        step_count = 0
        overrun_count = 0
        in_tol_since = None
        last_box_pos = None
        last_box_quat = None
        # The URCapX XML-RPC server must not be commanded above ~10 Hz, so the
        # gripper is sent at most every gripper_min_dt seconds even though the
        # arm loop runs at control_hz (e.g. 50 Hz).
        gripper_min_dt = 0.1
        last_gripper_t = -1.0
        last_gripper_norm = None

        start_time = time.perf_counter()
        next_tick = start_time

        try:
            while True:
                loop_start = time.perf_counter()

                # 1. RTDE receive
                t0_obs = time.perf_counter()
                fb = self.receive_feedback()
                t1_obs = time.perf_counter()

                q = np.asarray(fb["q"], dtype=float)
                qd = np.asarray(fb["qd"], dtype=float)
                tcp_xyz = np.asarray(fb["tcp_xyz"], dtype=float)
                if use_fk_tcp:
                    tcp_xyz = self.compute_tcp_pos(q)

                # 2. Box pose from mocap, mapped into the policy/base frame via the
                # base-frame calibration (identity if none loaded). Fall back to the
                # last good value.
                box_pos = mocap_reader.get_rigid_body_xyz()
                if box_pos is not None:
                    box_pos = self.mocap_pos_to_base(box_pos)   # mocap world -> base
                if box_pos is None:
                    box_pos = (last_box_pos if last_box_pos is not None
                               else drop_target.copy())
                box_pos = np.asarray(box_pos, dtype=float)
                last_box_pos = box_pos.copy()

                # Box orientation (quaternion w,x,y,z), also mapped to the base
                # frame. Identity fallback if unavailable.
                box_quat = mocap_reader.get_rigid_body_quat()
                if box_quat is None:
                    box_quat = last_box_quat
                else:
                    box_quat = self.mocap_quat_to_base(np.asarray(box_quat, dtype=float))
                    last_box_quat = box_quat.copy()

                box_target_dist = float(np.linalg.norm(drop_target - box_pos))

                # 3. Build observation (26D)
                obs_batch = self.build_obs_from_feedback(
                    fb, box_pos, drop_target,
                    tcp_pos=tcp_xyz if use_fk_tcp else None,
                    box_quat=box_quat, dtype=dtype,
                )

                # 4. Policy inference
                t0_policy = time.perf_counter()
                action_raw = self._policy_fn(obs_batch)
                action = np.asarray(action_raw, dtype=dtype).reshape(-1)
                t1_policy = time.perf_counter()

                # 5. Control update (arm servoj + gripper cmd)
                arm_ctrl, gripper_norm = self.policy_step_ctrl_update(
                    q, action, action_scale, alpha=alpha, dtype=dtype,
                )

                # 6. Send arm servoj + gripper
                t0_send = time.perf_counter()
                self.send_servoj(
                    arm_ctrl.tolist(),
                    a=servoj_a, v=servoj_v, t=5 * dt,
                    lookahead_time=lookahead_time, gain=gain,
                )
                # Gripper on a separate (XML-RPC) channel, rate-limited to
                # <=10 Hz and only re-sent when the command changes meaningfully.
                if (loop_start - last_gripper_t >= gripper_min_dt and
                        (last_gripper_norm is None
                         or abs(gripper_norm - last_gripper_norm) > 1e-3)):
                    try:
                        gripper_fn(gripper_norm)
                        last_gripper_t = loop_start
                        last_gripper_norm = gripper_norm
                    except Exception as e:  # noqa: BLE001
                        if step_count == 0 and debug_print:
                            print(f"\n[warn] gripper command failed: {e}; "
                                  "arm runs, gripper ignored.")
                t1_send = time.perf_counter()

                # 7. Timing control
                next_tick += dt
                now = time.perf_counter()
                sleep_time = next_tick - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    overrun_count += 1
                    next_tick = now

                loop_end = time.perf_counter()
                loop_dt_true = loop_end - loop_start
                loop_hz_true = 1.0 / loop_dt_true if loop_dt_true > 1e-12 else np.nan
                elapsed = loop_end - start_time

                # 8. Log
                row = {
                    "step": step_count,
                    "time": elapsed,
                    **{f"q{i}": q[i] for i in range(6)},
                    **{f"qd{i}": qd[i] for i in range(6)},
                    **{f"ctrl{i}": arm_ctrl[i] for i in range(6)},
                    **{f"action{i}": action[i] for i in range(7)},
                    "gripper_norm": gripper_norm,
                    "gripper_ctrl": self._gripper_ctrl,
                    "tcp_x": tcp_xyz[0], "tcp_y": tcp_xyz[1], "tcp_z": tcp_xyz[2],
                    "box_x": box_pos[0], "box_y": box_pos[1], "box_z": box_pos[2],
                    "target_x": drop_target[0], "target_y": drop_target[1],
                    "target_z": drop_target[2],
                    "box_to_target_dist": box_target_dist,
                    "obs_time_s": t1_obs - t0_obs,
                    "policy_time_s": t1_policy - t0_policy,
                    "send_time_s": t1_send - t0_send,
                    "loop_dt_true_s": loop_dt_true,
                    "loop_hz_true": loop_hz_true,
                    "overrun_count": overrun_count,
                }
                log.append(row)

                if debug_print:
                    print(
                        f"\r[{step_count:4d}] box->tgt={box_target_dist:.4f}m "
                        f"| grip={gripper_norm:.2f} | hz={loop_hz_true:.1f} "
                        f"| box={np.round(box_pos, 3)}",
                        end="", flush=True,
                    )

                if reach_tol is not None and box_target_dist < reach_tol:
                    if in_tol_since is None:
                        in_tol_since = time.perf_counter()
                    if (time.perf_counter() - in_tol_since) >= dwell_time_s:
                        if debug_print:
                            print(f"\nTarget reached at step {step_count}!")
                        break
                else:
                    in_tol_since = None

                if elapsed >= timeout_s:
                    break

                step_count += 1
        finally:
            if self._control is not None:
                try:
                    self._control.servoStop()
                except Exception:
                    pass

        total_wall = time.perf_counter() - start_time
        df = pd.DataFrame(log)
        stats = self.summarize_policy_loop(df, control_hz, timeout_s, total_wall)
        if debug_print:
            print("\nPolicy loop finished.")
        return df, stats

    def summarize_policy_loop(self, df, requested_control_hz, timeout_s,
                              total_wall_time_s) -> dict:
        if len(df) == 0:
            return {"num_samples": 0}
        n = len(df) - 1
        true_hz = n / total_wall_time_s if total_wall_time_s > 0 else None
        return {
            "requested_control_hz": round(float(requested_control_hz), 1),
            "timeout_s": float(timeout_s),
            "num_samples": int(len(df)),
            "total_time_s": float(df["time"].iloc[-1]),
            "total_wall_time_s": float(total_wall_time_s),
            "true_inferred_frequency_hz": float(true_hz) if true_hz else None,
            "mean_loop_hz_true": float(df["loop_hz_true"].mean()),
            "mean_policy_time_s": float(df["policy_time_s"].mean()),
            "max_policy_time_s": float(df["policy_time_s"].max()),
            "start_box_to_target_dist": float(df["box_to_target_dist"].iloc[0]),
            "final_box_to_target_dist": float(df["box_to_target_dist"].iloc[-1]),
            "min_box_to_target_dist": float(df["box_to_target_dist"].min()),
            "net_improvement": float(
                df["box_to_target_dist"].iloc[0] - df["box_to_target_dist"].iloc[-1]
            ),
            "num_overruns": int(df["overrun_count"].iloc[-1]),
            "overrun_ratio": float(df["overrun_count"].iloc[-1]) / max(len(df), 1),
        }

    def print_stats(self, stats: dict, keys: Optional[List[str]] = None):
        items = stats.items() if keys is None else [
            (k, stats.get(k, "<missing>")) for k in keys]
        for k, v in items:
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    # =========================================================================
    # I — MuJoCo Rendering
    # =========================================================================

    def mujoco_init_model(self, xml_path: str, height: int = 480, width: int = 640,
                          cam_lookat=(0.3, 0.0, 0.3), cam_distance: float = 1.2,
                          cam_azimuth: float = 130, cam_elevation: float = -20) -> dict:
        """Initialize MuJoCo model, data, renderer, and camera for replay."""
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = list(cam_lookat)
        cam.distance = float(cam_distance)
        cam.azimuth = float(cam_azimuth)
        cam.elevation = float(cam_elevation)

        # qpos addresses for the box freejoint (for replay placement).
        box_jntadr = model.body("box").jntadr[0]
        box_qposadr = int(model.jnt_qposadr[box_jntadr])
        return {"model": model, "data": data, "renderer": renderer, "cam": cam,
                "box_qposadr": box_qposadr}

    def mujoco_sync_and_render(self, mj: dict, q, gripper_ctrl=None, box_pos=None,
                               drop_target=None) -> np.ndarray:
        """Sync robot+box+target state into MuJoCo, run mj_forward, render."""
        data = mj["data"]
        data.qpos[:6] = np.asarray(q, dtype=float)
        if gripper_ctrl is not None:
            # Both fingers ~ gripper_ctrl (tendon coupling; small range).
            data.qpos[6] = float(gripper_ctrl)
            data.qpos[7] = float(gripper_ctrl)
        if box_pos is not None:
            adr = mj["box_qposadr"]
            data.qpos[adr:adr + 3] = np.asarray(box_pos, dtype=float)
        if drop_target is not None and data.mocap_pos.shape[0] > 0:
            data.mocap_pos[0, :] = np.asarray(drop_target, dtype=float)

        mujoco.mj_forward(mj["model"], data)
        mj["renderer"].update_scene(data, camera=mj["cam"])
        return mj["renderer"].render().copy()

    def render_video_from_log(self, mj: dict, df: pd.DataFrame,
                              video_fps: float = 30.0) -> Tuple[list, float]:
        """Render frames from logged DataFrame at the desired video fps."""
        q_cols = [f"q{i}" for i in range(6)]
        has_box = all(c in df.columns for c in ["box_x", "box_y", "box_z"])
        has_tgt = all(c in df.columns for c in ["target_x", "target_y", "target_z"])
        has_grip = "gripper_ctrl" in df.columns

        if "loop_hz_true" in df.columns:
            logged_hz = float(df["loop_hz_true"].mean())
        else:
            dts = df["time"].diff().dropna()
            logged_hz = 1.0 / dts.mean() if len(dts) > 0 else video_fps

        stride = max(1, int(round(logged_hz / video_fps)))
        actual_video_fps = logged_hz / stride

        frames = []
        for i in range(0, len(df), stride):
            row = df.iloc[i]
            q = row[q_cols].to_numpy(dtype=float)
            grip = float(row["gripper_ctrl"]) if has_grip else None
            box = (row[["box_x", "box_y", "box_z"]].to_numpy(dtype=float)
                   if has_box else None)
            tgt = (row[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
                   if has_tgt else None)
            frames.append(self.mujoco_sync_and_render(mj, q, grip, box, tgt))

        print(f"  Rendered {len(frames)} frames from {len(df)} logged steps")
        return frames, actual_video_fps

    def save_video(self, frames: list, out_path: str = "ur3_pick_replay.mp4",
                   fps: float = 30.0) -> str:
        writer = imageio.get_writer(out_path, fps=float(fps))
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        print(f"[save_video] Saved {len(frames)} frames to {out_path}")
        return out_path

    # =========================================================================
    # J — Diagnostics: Plots & Metadata
    # =========================================================================

    JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

    def save_plots(self, df: pd.DataFrame, out_path: str) -> str:
        """Save a stacked diagnostic plot (PNG)."""
        import matplotlib.pyplot as plt

        joint_names = self.JOINT_NAMES
        t = df["time"]
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

        ax = axes[0]
        ax.plot(t, df["box_to_target_dist"] * 100)
        ax.set_ylabel("distance (cm)")
        ax.set_title("Box to drop-target distance")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for i, name in enumerate(joint_names):
            ax.plot(t, df[f"q{i}"], label=name)
        ax.set_ylabel("position (rad)")
        ax.set_title("Joint positions")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        for i, name in enumerate(joint_names):
            ax.plot(t, df[f"qd{i}"], label=name)
        ax.set_ylabel("velocity (rad/s)")
        ax.set_title("Joint velocities")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.plot(t, df["gripper_norm"])
        ax.set_ylabel("gripper [0,1]")
        ax.set_title("Gripper command (0=open, 1=closed)")
        ax.grid(True, alpha=0.3)

        ax = axes[4]
        ax.plot(t, df["loop_hz_true"], alpha=0.7)
        ax.set_ylabel("Hz")
        ax.set_title("Loop frequency")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[save_plots] Saved to {out_path}")
        return out_path

    @staticmethod
    def save_run_metadata(out_path: str, **kwargs) -> str:
        import json as _json

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        data = {k: _convert(v) for k, v in kwargs.items()}
        with open(out_path, "w") as f:
            _json.dump(data, f, indent=2, default=str)
        print(f"[save_run_metadata] Saved to {out_path}")
        return out_path
