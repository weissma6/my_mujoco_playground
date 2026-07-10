"""
RTDE dependencies for the UR3 + Hand-E pick task.

Mirrors robots/URSim/URSim_RTDE_dependencies.py (the UR10 reach backbone)
but adapted for the UR3PicknPlace policy:

- 13D observation  [arm_q(6), gripper(1), (box-tcp)(3), (target-box)(3)]
                   arm_q = 6 arm joints; gripper = combined finger opening in
                   [0, 0.05] (no orientation is exposed to the policy)
- 7D action        6 arm joint deltas + 1 Hand-E gripper delta
- box pose comes from a Nokov rigid body (motion_capture.mymocap.mocap_dependencies):
  xyz via get_rigid_body_xyz() (orientation is no longer needed by the policy)
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


class MocapRigidBodyLost(RuntimeError):
    """Raised when the tracked mocap rigid body stops updating mid-loop.

    The VRPN reader keeps returning the LAST pose after a body leaves the
    cameras' view (it never reverts to None), so loss is detected by staleness:
    last_update_time stops advancing. run_policy_loop catches this, stops the
    arm, and returns the partial log so diagnostics are still saved.
    """


class RobotStoppedExternally(RuntimeError):
    """Raised when the robot itself signals it stopped: a protective/
    emergency stop, or the External Control program is no longer PLAYING
    (e.g. the operator pressed Stop on the PolyScope X pendant). Detected via
    UR3RealRobotPick.is_stopped_externally(). run_policy_loop catches this
    the same way as MocapRigidBodyLost: halt the arm, keep the partial log,
    tag stopped_reason="external_stop".
    """


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
        # _gripper_ctrl lives in the tendon-actuator ctrl space [_gripper_lo,
        # _gripper_hi] = [0, 0.05] (matches the sim actuator ctrlrange and the
        # sim integrator in ur3_pick.step). The PHYSICAL per-finger position is
        # in [0, _finger_hi] = [0, 0.025] (the finger joint range), and the
        # training obs uses the finger JOINT position there — so obs/command are
        # projected from ctrl-space into finger-space 1:1 and clipped at
        # _finger_hi (the tendon coef 0.5 + 1:1 coupling makes ctrl == per-finger
        # position in the unsaturated regime; ctrl in (0.025, 0.05] just over-
        # drives against the joint limit, exactly as in sim).
        self._gripper_ctrl: float = 0.0
        self._gripper_lo: float = 0.0
        self._gripper_hi: float = 0.05
        self._finger_hi: float = 0.025
        # Deployment-only estimate of the physical per-finger position [0, 0.025],
        # produced by the finger-plant low-pass in policy_step_ctrl_update (sim has
        # a real PD finger plant; deployment re-creates its lag so the policy sees
        # what it trained on). Feeds BOTH the command and the obs. Starts open.
        self._finger_pos_est: float = 0.0
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

    def is_stopped_externally(self) -> bool:
        """Best-effort check for a protective/emergency stop or the External
        Control program no longer PLAYING. Used by run_policy_loop to break
        out with stopped_reason="external_stop" instead of running blind
        against a halted robot. Returns False (never blocks the loop) if the
        receiver is unavailable or the check itself raises -- this is a
        best-effort safety signal, not the primary control path.
        """
        if self._receiver is None:
            return False
        try:
            if self._receiver.isProtectiveStopped():
                return True
        except Exception:
            pass
        try:
            if not self._receiver.isConnected():
                return True
        except Exception:
            pass
        return False

    def receive_feedback(self) -> Dict[str, List[float]]:
        """Returns dict with joint state, TCP, currents, control output, force."""
        r = self.connect()
        q = r.getActualQ()
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
            "tcp_xyz": list(tcp_pose[:3]),
            "tcp_pose": list(tcp_pose),
            "current": list(current),
            "ctrl_output": list(ctrl_output),
            "tcp_force": list(tcp_force),
        }

    def print_feedback(self, digits: int = 4):
        fb = self.receive_feedback()
        print("q      =", [round(v, digits) for v in fb["q"]])
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
        # Drives build_obs_from_feedback's 13D/26D branch -- read from the
        # saved metadata so both policy families in POLICY_REGISTRY produce
        # the obs shape they actually trained on, automatically.
        self._obs_dim = int(meta["obs_dim"])

        # Actuator ctrl limits (arm + gripper) from the env model.
        env = registry.load(meta["env_name"])
        lowers, uppers = env.mj_model.actuator_ctrlrange.T
        self._ctrl_lowers = np.array(lowers, dtype=np.float32)
        self._ctrl_uppers = np.array(uppers, dtype=np.float32)
        # Gripper actuator is the last one (tendon "split", ctrlrange 0..0.05).
        # This is the INTEGRATOR range, matching the sim ctrl clip.
        self._gripper_lo = float(self._ctrl_lowers[-1])
        self._gripper_hi = float(self._ctrl_uppers[-1])
        # Physical per-finger close (= the finger joint range max, 0.025). The
        # obs/command are clipped here, because the training obs uses the finger
        # JOINT position [0, 0.025], not the tendon ctrl [0, 0.05].
        self._finger_hi = float(
            env.mj_model.jnt_range[
                env.mj_model.joint("hande_left_finger_joint").id
            ][1]
        )

        # Training-time gripper action scale (the sim decouples arm vs gripper
        # scaling: ur3_pick default_config.gripper_action_scale). It is NOT saved
        # in metadata.json, so it is read from the env default config here as the
        # source of truth the pick loop should match. NaN if the env predates the
        # decoupling. The pick loop passes its own gripper_action_scale into
        # run_policy_loop; compare the two if the gripper behaves oddly.
        self._train_gripper_action_scale = float(
            getattr(getattr(env, "_config", None), "gripper_action_scale",
                    float("nan"))
        )
        print(f"Env gripper_action_scale (training source of truth): "
              f"{self._train_gripper_action_scale}")

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

    # Joint names for the FK model's grasp-geometry qpos addresses (mirrors
    # mujoco_playground._src.manipulation.my_ur3.ur3_base._ARM_JOINTS /
    # _FINGER_JOINTS -- kept as a literal here so this module has no MJX/
    # mujoco_playground import dependency for FK).
    _FK_ARM_JOINTS = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    _FK_FINGER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]

    def init_fk_model(self, xml_path: str):
        """Load a MuJoCo model for forward kinematics: the tcp site (used by
        compute_tcp_pos, unchanged) plus the grasp-frame geometry (finger
        touch sites + box freejoint) needed by the 26D obs's orientation
        features (see compute_obs_geometry / build_obs_from_feedback).
        """
        self._fk_model = mujoco.MjModel.from_xml_path(xml_path)
        self._fk_data = mujoco.MjData(self._fk_model)
        self._fk_tcp_id = mujoco.mj_name2id(
            self._fk_model, mujoco.mjtObj.mjOBJ_SITE, "tcp"
        )
        self._fk_left_touch = self._fk_model.site("left_finger_touch_site").id
        self._fk_right_touch = self._fk_model.site("right_finger_touch_site").id
        self._fk_box_body = self._fk_model.body("box").id
        self._fk_arm_qadr = np.array(
            [self._fk_model.jnt_qposadr[self._fk_model.joint(j).id]
             for j in self._FK_ARM_JOINTS]
        )
        self._fk_finger_qadr = np.array(
            [self._fk_model.jnt_qposadr[self._fk_model.joint(j).id]
             for j in self._FK_FINGER_JOINTS]
        )
        _box_jntadr = self._fk_model.body("box").jntadr[0]
        self._fk_box_qadr = int(self._fk_model.jnt_qposadr[_box_jntadr])
        print(f"FK model loaded: tcp site id={self._fk_tcp_id}, "
              f"left/right touch sites={self._fk_left_touch}/"
              f"{self._fk_right_touch}, box body={self._fk_box_body}")

    def compute_tcp_pos(self, q: np.ndarray) -> np.ndarray:
        """Compute MuJoCo tcp site position from the 6 arm joint angles via FK."""
        self._fk_data.qpos[:6] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self._fk_model, self._fk_data)
        return self._fk_data.site_xpos[self._fk_tcp_id].copy()

    def compute_obs_geometry(self, q, finger_m: float, box_pos, box_quat) -> dict:
        """One mj_forward on the full FK scene (arm + fingers + box freejoint)
        that returns exactly the geometry the 26D obs's orientation features
        need: the tcp site position, the jaw/approach unit axes (built from
        the SAME finger-touch sites sim uses), and the box's world-frame
        axes (from its quaternion, via MuJoCo's own xmat -- never a hand-
        rolled quat->matrix, so this is guaranteed to match sim's
        data.xmat[box] convention bit-for-bit). Mirrors
        evaluation/ur3_reward_replay.SimFK.geom() exactly (kept as a
        separate copy here so this deploy-time module has no dependency on
        the evaluation/ package).

        Args:
          q: (6,) arm joint angles, radians.
          finger_m: scalar meters in [0, 0.025] (both fingers, symmetric --
            matches the sim tendon actuator's coef 0.5/0.5).
          box_pos: (3,) base-frame meters.
          box_quat: (4,) MuJoCo convention (w, x, y, z), base-frame.
        """
        d = self._fk_data
        d.qpos[self._fk_arm_qadr] = np.asarray(q, dtype=float)
        d.qpos[self._fk_finger_qadr] = float(finger_m)
        d.qpos[self._fk_box_qadr:self._fk_box_qadr + 3] = np.asarray(
            box_pos, dtype=float
        )
        bq = np.asarray(box_quat, dtype=float)
        n = np.linalg.norm(bq)
        d.qpos[self._fk_box_qadr + 3:self._fk_box_qadr + 7] = (
            bq / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
        )
        d.qvel[:] = 0.0
        mujoco.mj_forward(self._fk_model, d)

        tcp = d.site_xpos[self._fk_tcp_id].copy()
        lft = d.site_xpos[self._fk_left_touch].copy()
        rgt = d.site_xpos[self._fk_right_touch].copy()
        jaw_axis = rgt - lft
        jaw_axis = jaw_axis / (np.linalg.norm(jaw_axis) + 1e-6)
        app_axis = 0.5 * (lft + rgt) - tcp
        app_axis = app_axis / (np.linalg.norm(app_axis) + 1e-6)
        box_axes = d.xmat[self._fk_box_body].reshape(3, 3).copy()
        return {"tcp": tcp, "jaw_axis": jaw_axis, "app_axis": app_axis,
                "box_axes": box_axes}

    # =========================================================================
    # F — Observation Building (13D / 26D)
    # =========================================================================

    def build_obs_from_feedback(
        self,
        fb: dict,
        box_pos: np.ndarray,
        target_pos: np.ndarray,
        tcp_pos: Optional[np.ndarray] = None,
        box_quat: Optional[np.ndarray] = None,
        last_action: Optional[np.ndarray] = None,
        obs_dim: Optional[int] = None,
        dtype=np.float32,
    ) -> np.ndarray:
        """Build the obs to match the canonical UR3Base/UR3Pick _get_obs,
        branching on `obs_dim` (13 for the legacy orientation-free policies,
        26 for the grasp-frame-aligned policies e.g. AGATE250) so BOTH policy
        families in POLICY_REGISTRY deploy correctly through one method.

        13D (unchanged from before this branch was added):
            [arm_q(6), gripper(1), (box-tcp)(3), (target-box)(3)]

        26D (= 13D + 6D grasp-frame orientation features + 7D last_action,
        matching ur3_pick.py's UR3Pick._get_obs EXACTLY):
            [ ...13D above..., jaw_proj(3), app_proj(3), last_action(7) ]
        where jaw_proj/app_proj are the gripper's jaw axis (finger
        separation) and approach axis (palm->fingertips), each projected
        onto the box's three world-frame axes -- computed via ONE mj_forward
        on the FK scene (compute_obs_geometry, above), which reads the exact
        tcp/finger-touch sites + box xmat sim uses (no hand-rolled
        quat->matrix -- see that method's docstring). In the 26D path the
        base 13D's box-tcp term ALSO uses this FK tcp (not tcp_pos/RTDE), so
        both the position and orientation terms share one consistent TCP,
        exactly like sim's single `_gripper_site`.

        arm_q = 6 arm joints (RTDE). gripper = combined finger opening in
        [0, 0.05] m: sim feeds the SUM of the two finger joint positions, so the
        single-finger plant estimate is doubled here. Uses the finger-PLANT
        estimate (policy_step_ctrl_update's low-pass of the tendon integrator),
        NOT the raw ctrl: sim observes the lagged physical finger, so feeding the
        raw command would be out-of-distribution and, after
        running_statistics.normalize, drive the policy off the rails (and the
        zero-lag feedback was what made the deploy gripper oscillate).
        tcp_pos defaults to RTDE getActualTCPPose()[:3] (X/Y already negated in
        receive_feedback); pass an FK-computed tcp_pos to override -- ignored
        in the 26D path (uses compute_obs_geometry's FK tcp instead, see above).
        box_quat: (4,) MuJoCo convention (w, x, y, z), base-frame -- REQUIRED
        for a correct 26D obs (defaults to identity if None, which silently
        assumes an unrotated box; only safe as a degraded fallback).
        last_action: previous RAW 7D policy action (pre action_scale), zeros
        on the first step -- matches sim's info["last_action"]. Only used in
        the 26D path.
        obs_dim: 13 or 26; defaults to self._obs_dim (set by load_policy_fn
        from the loaded policy's metadata.json) if not given explicitly.
        Returns shape (1, obs_dim) for batched policy input.
        """
        obs_dim = int(obs_dim if obs_dim is not None
                      else getattr(self, "_obs_dim", 13))

        arm_q = np.array(fb["q"], dtype=dtype)                      # 6
        gripper = np.array(
            [2.0 * float(np.clip(self._finger_pos_est, 0.0, self._finger_hi))],
            dtype=dtype,
        )                                                           # 1 (0-0.05)
        box = np.array(box_pos, dtype=dtype)
        tgt = np.array(target_pos, dtype=dtype)

        if obs_dim == 13:
            tcp = (np.array(fb["tcp_xyz"], dtype=dtype) if tcp_pos is None
                   else np.array(tcp_pos, dtype=dtype))
            box_to_tcp = box - tcp                                  # 3
            target_to_box = tgt - box                              # 3
            obs = np.concatenate([arm_q, gripper, box_to_tcp, target_to_box])
            return obs[None, :]  # (1, 13)

        if obs_dim == 26:
            if getattr(self, "_fk_model", None) is None:
                raise RuntimeError(
                    "26D obs requires init_fk_model() to have been called "
                    "(the grasp-frame orientation features need the FK "
                    "scene's finger-touch sites + box body)."
                )
            bq = (np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                  if box_quat is None else np.asarray(box_quat, dtype=float))
            finger_m = float(np.clip(self._finger_pos_est, 0.0, self._finger_hi))
            geo = self.compute_obs_geometry(arm_q, finger_m, box_pos, bq)
            tcp = np.array(geo["tcp"], dtype=dtype)
            box_to_tcp = box - tcp                                  # 3
            target_to_box = tgt - box                              # 3
            jaw_proj = (geo["jaw_axis"] @ geo["box_axes"]).astype(dtype)  # 3
            app_proj = (geo["app_axis"] @ geo["box_axes"]).astype(dtype)  # 3
            la = (np.zeros(7, dtype=dtype) if last_action is None
                  else np.asarray(last_action, dtype=dtype).reshape(7))   # 7
            obs = np.concatenate([
                arm_q, gripper, box_to_tcp, target_to_box,
                jaw_proj, app_proj, la,
            ])
            return obs[None, :]  # (1, 26)

        raise ValueError(f"build_obs_from_feedback: unsupported obs_dim={obs_dim} "
                          "(expected 13 or 26)")

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
        gripper_action_scale: Optional[float] = None,
        alpha: float = 1.0,
        dt: float = 0.02,
        gripper_tau: float = 0.0,
        gripper_max_rate: float = float("inf"),
        dtype=np.float32,
    ) -> Tuple[np.ndarray, float, dict]:
        """Compute arm servoj target and gripper command from a 7D action.

        Returns (arm_ctrl(6,), gripper_norm, diag) where gripper_norm is in [0,1]
        and diag carries the pre-smoothing intermediates for the diagnostic plots
        (arm_ctrl_pre_alpha, gripper_ctrl_raw, finger_pos_est). The gripper command
        is integrated internally (the obs has no real gripper feedback); the
        finger-plant low-pass below re-creates the sim finger dynamics the policy
        was trained against.

        action_scale scales the 6 ARM deltas. gripper_action_scale scales the
        single GRIPPER delta and MUST match the env's gripper_action_scale — the
        sim decouples the two (ur3_pick: self._action_scale = [arm]*6 + [grip],
        delta = action * self._action_scale). The gripper scale is kept small
        (0.01 vs arm 0.025) so the hand can't snap shut in one step; applying the
        arm scale here would integrate the finger ~2.5x too fast and drive it
        out-of-distribution. None => fall back to action_scale (old coupled
        behavior, for policies trained before the decoupling).

        gripper_tau / gripper_max_rate default to the degenerate (no-smoothing)
        values, so this reduces EXACTLY to the prior behavior when they are unset.
        """
        grip_scale = float(
            action_scale if gripper_action_scale is None else gripper_action_scale
        )
        action = np.asarray(action, dtype=dtype).reshape(-1)
        q = np.asarray(q, dtype=dtype)

        # --- Arm: delta on measured joint positions ---
        arm_ctrl = q + float(action_scale) * action[:6]
        if self._ctrl_lowers is not None and self._ctrl_uppers is not None:
            arm_ctrl = np.clip(arm_ctrl, self._ctrl_lowers[:6], self._ctrl_uppers[:6])
        arm_ctrl_pre_alpha = arm_ctrl.copy()
        if alpha < 1.0:
            arm_ctrl = alpha * arm_ctrl + (1.0 - alpha) * q

        # --- Gripper ---
        # 1. Raw integrator in tendon-ctrl space [0, 0.05], faithful to the sim
        # integrator (ur3_pick.step: ctrl += action_scale*action, clipped to the
        # actuator ctrlrange). Uses grip_scale (the DECOUPLED gripper scale), not
        # the arm action_scale — the sim's action-scale vector applies the small
        # gripper_action_scale to this last actuator dim.
        self._gripper_ctrl = float(
            np.clip(
                self._gripper_ctrl + grip_scale * float(action[6]),
                self._gripper_lo,
                self._gripper_hi,
            )
        )
        # 2. Finger-plant low-pass (ROOT CAUSE of the deploy oscillation): in sim
        # the policy observes data.qpos[finger] — the PHYSICAL finger, which lags
        # the command (a PD plant, kp=400/kv=1) and stalls on the box. Deployment
        # has no finger plant, so re-create it as a first-order lag toward the
        # projected finger target. tau=0 => beta=1 => instant (the old behavior).
        # This single estimate feeds BOTH the command (below) and the next obs
        # (build_obs_from_feedback), mirroring sim's ctrl -> plant -> qpos -> obs.
        target = float(np.clip(self._gripper_ctrl, 0.0, self._finger_hi))
        beta = float(dt) / (float(gripper_tau) + float(dt))
        est = (1.0 - beta) * self._finger_pos_est + beta * target
        # 3. Brute-force slew clip (SAFETY NET): cap the per-step finger travel
        # (gripper_max_rate in m/s; inf => disabled).
        dmax = float(gripper_max_rate) * float(dt)
        est = float(
            np.clip(est, self._finger_pos_est - dmax, self._finger_pos_est + dmax)
        )
        self._finger_pos_est = float(np.clip(est, 0.0, self._finger_hi))
        # 4. Command + obs both read this plant estimate. gripper_norm ==
        # finger_meters / 0.025, so gripper_fn(norm) = command(norm*0.025) hands the
        # real finger target (meters) to HandEGripper.
        gripper_norm = self._finger_pos_est / max(self._finger_hi, 1e-9)

        diag = {
            "arm_ctrl_pre_alpha": arm_ctrl_pre_alpha,
            "gripper_ctrl_raw": self._gripper_ctrl,
            "finger_pos_est": self._finger_pos_est,
        }
        return arm_ctrl, gripper_norm, diag

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
        gripper_action_scale: Optional[float] = None,
        lookahead_time: float = 0.1,
        gain: int = 300,
        servoj_a: float = 1.4,
        servoj_v: float = 1.05,
        alpha: float = 1.0,
        gripper_fn=None,
        gripper_state_fn=None,
        gripper_tau: float = 0.0,
        gripper_max_rate: float = float("inf"),
        use_fk_tcp: bool = False,
        reach_tol: float = None,
        dwell_time_s: float = 0.0,
        mocap_stale_s: float = None,
        dtype=np.float32,
        debug_print: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """Run the UR3 pick policy loop.

        Args:
          drop_target: (3,) world-frame drop location (meters).
          mocap_reader: VRPN/Nokov rigid-body reader; get_rigid_body_xyz() -> box.
          gripper_action_scale: per-step scale for the gripper delta; MUST match
                      the env's gripper_action_scale (sim decouples arm vs gripper
                      scaling). None => coupled to action_scale (pre-decoupling
                      policies). See policy_step_ctrl_update.
          gripper_fn: callable(norm_cmd in [0,1]); defaults to self.send_gripper.
          gripper_state_fn: optional callable() -> real gripper readback; either a
                      read_state() dict (sim_finger metres [0,0.025] + pos_pct raw
                      native percent [0,100]) or, for back-compat, a plain float
                      sim_finger. Polled at the same <=10 Hz cadence as the send
                      (diagnostic only — NOT fed into the obs), logged as
                      gripper_fb_pos (metres) + gripper_fb_pct (percent). None /
                      dry-run => both logged as NaN.
          gripper_tau: finger-plant low-pass time constant (s) for the gripper
                      command/obs; 0 => no smoothing (the old behavior). See
                      policy_step_ctrl_update.
          gripper_max_rate: brute-force slew cap (m/s) on the per-step finger
                      command; inf => disabled.
          use_fk_tcp: if True, tcp_pos is computed via FK from joint angles
                      (requires init_fk_model); else RTDE TCP pose is used.
          mocap_stale_s: if set, the loop STOPS as soon as the tracked rigid body
                      has not updated for this many seconds (it left the cameras'
                      view). The reader's pose goes stale, not None, so this
                      staleness window is the only reliable loss signal — needs a
                      reader exposing `last_update_time` (VRPNRigidBodyReader does)
                      subscribed to ONLY the target body. None disables the check.
        Returns (per-step DataFrame, summary stats dict). On mocap loss the loop
        breaks early; stats["stopped_reason"] == "mocap_lost".
        """
        if self._policy_fn is None:
            raise RuntimeError("Policy not loaded. Call load_policy_fn() first.")
        if gripper_fn is None:
            gripper_fn = self.send_gripper

        dt = 1.0 / float(control_hz)
        drop_target = np.asarray(drop_target, dtype=np.float32).reshape(3)
        # Seed the internal gripper tracker from the keyframe ctrl if available, and
        # start the finger-plant estimate open (the pickloop opens the gripper at
        # the start of the run).
        self._gripper_ctrl = float(getattr(self, "_gripper_ctrl", 0.0))
        self._finger_pos_est = 0.0

        log = []
        step_count = 0
        overrun_count = 0
        in_tol_since = None
        last_box_pos = None
        last_box_quat = None  # (w, x, y, z); carried forward like last_box_pos
        # The URCapX XML-RPC server must not be commanded above ~10 Hz, so the
        # gripper is sent at most every gripper_min_dt seconds even though the
        # arm loop runs at control_hz (e.g. 50 Hz).
        gripper_min_dt = 0.1
        last_gripper_t = -1.0
        last_gripper_norm = None
        last_gripper_fb = np.nan  # Hand-E finger readback [0,0.025]; NaN until polled
        last_gripper_fb_pct = float("nan")  # raw native percent [0,100]; NaN until polled
        # Real grasp proxy (Robotiq object-detection flag), carried forward
        # between the <=10 Hz gripper polls like the readback above. Used by
        # evaluation/ur3_reward_replay.py to gate the grasp/lift/box_target/
        # hold_target reward terms the same way sim's finger-pad contact
        # sensors do (coarser: polled at <=10 Hz, not every control tick).
        last_grasped = False
        last_obj_flag = 0
        # Previous RAW policy action (pre action_scale), fed into the 26D
        # obs's last_action slot (matches sim's info["last_action"] -- see
        # build_obs_from_feedback). Ignored by the 13D path. Zeros on the
        # first step, matching the env's reset().
        last_action = np.zeros(7, dtype=dtype)

        stopped_reason = "completed"

        start_time = time.perf_counter()
        next_tick = start_time

        try:
            while True:
                loop_start = time.perf_counter()

                # 0. Robot-side stop check (protective/emergency stop, or
                # External Control no longer PLAYING e.g. operator pressed
                # Stop on the pendant). Best-effort -- see
                # is_stopped_externally(); never raises on its own.
                if self.is_stopped_externally():
                    raise RobotStoppedExternally(
                        f"robot reported a protective/emergency stop or lost "
                        f"External Control at step {step_count}."
                    )

                # 1. RTDE receive
                t0_recv = time.perf_counter()
                fb = self.receive_feedback()
                t1_recv = time.perf_counter()

                q = np.asarray(fb["q"], dtype=float)
                tcp_xyz = np.asarray(fb["tcp_xyz"], dtype=float)
                # Joint currents (≈ torque proxy), UR control output, and the
                # 6-axis TCP wrench — fetched by receive_feedback but otherwise
                # dropped; logged for the diagnostics (currently CSV-only).
                current = np.asarray(fb["current"], dtype=float)
                ctrl_output = np.asarray(fb["ctrl_output"], dtype=float)
                tcp_force = np.asarray(fb["tcp_force"], dtype=float)
                if use_fk_tcp:
                    tcp_xyz = self.compute_tcp_pos(q)

                # 2. Box pose from mocap, mapped into the policy/base frame via the
                # base-frame calibration (identity if none loaded). The reader keeps
                # returning the LAST pose after the body leaves view, so loss is
                # detected by staleness (last_update_time), not a None return.
                t0_mocap = time.perf_counter()
                if mocap_stale_s is not None:
                    last_upd = getattr(mocap_reader, "last_update_time", None)
                    age = None if last_upd is None else time.time() - last_upd
                    if age is None or age > mocap_stale_s:
                        raise MocapRigidBodyLost(
                            f"rigid body '{getattr(mocap_reader, 'rigid_body_name', '?')}' "
                            f"not updated for "
                            f"{'never' if age is None else f'{age * 1000:.0f} ms'} "
                            f"(> {mocap_stale_s * 1000:.0f} ms) at step {step_count} — "
                            "it left the cameras' view."
                        )
                box_pos = mocap_reader.get_rigid_body_xyz()
                if box_pos is not None:
                    box_pos = self.mocap_pos_to_base(box_pos)   # mocap world -> base
                if box_pos is None:
                    box_pos = (last_box_pos if last_box_pos is not None
                               else drop_target.copy())
                box_pos = np.asarray(box_pos, dtype=float)
                last_box_pos = box_pos.copy()

                # Box orientation (w, x, y, z), same mocap->base mapping as the
                # position above. Not used by the 13D policy obs (orientation-
                # free by design), but needed to reconstruct the training
                # reward's gripper_align term post-hoc (see
                # evaluation/ur3_reward_replay.py) -- carried forward like
                # box_pos on a missed mocap frame.
                box_quat = mocap_reader.get_rigid_body_quat()
                if box_quat is not None:
                    box_quat = self.mocap_quat_to_base(box_quat)
                if box_quat is None:
                    box_quat = (last_box_quat if last_box_quat is not None
                                else np.array([1.0, 0.0, 0.0, 0.0]))
                box_quat = np.asarray(box_quat, dtype=float)
                last_box_quat = box_quat.copy()
                t1_mocap = time.perf_counter()

                box_target_dist = float(np.linalg.norm(drop_target - box_pos))

                # 3. Build observation (13D or 26D, per the loaded policy's
                # obs_dim -- see build_obs_from_feedback). box_quat/last_action
                # are only consumed by the 26D path.
                t0_obs = time.perf_counter()
                obs_batch = self.build_obs_from_feedback(
                    fb, box_pos, drop_target,
                    tcp_pos=tcp_xyz if use_fk_tcp else None,
                    box_quat=box_quat,
                    last_action=last_action,
                    dtype=dtype,
                )
                t1_obs = time.perf_counter()

                # 4. Policy inference
                t0_policy = time.perf_counter()
                action_raw = self._policy_fn(obs_batch)
                action = np.asarray(action_raw, dtype=dtype).reshape(-1)
                last_action = action.copy()  # for the NEXT step's 26D obs
                t1_policy = time.perf_counter()

                # 5. Control update (arm servoj + gripper cmd)
                t0_ctrl = time.perf_counter()
                arm_ctrl, gripper_norm, diag = self.policy_step_ctrl_update(
                    q, action, action_scale,
                    gripper_action_scale=gripper_action_scale,
                    alpha=alpha, dt=dt,
                    gripper_tau=gripper_tau, gripper_max_rate=gripper_max_rate,
                    dtype=dtype,
                )
                t1_ctrl = time.perf_counter()

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
                        # Read the real Hand-E finger position back at the same
                        # <=10 Hz cadence (diagnostic only — NOT fed into the obs).
                        # Carried forward between polls; NaN in a dry-run.
                        if gripper_state_fn is not None:
                            fb = gripper_state_fn()
                            if isinstance(fb, dict):
                                # Full read_state() dict: sim_finger metres +
                                # raw native percent.
                                last_gripper_fb = float(
                                    fb.get("sim_finger", np.nan))
                                last_gripper_fb_pct = float(
                                    fb.get("pos_pct", np.nan))
                                # Robotiq object-detection flag/bool -- real
                                # grasp proxy for evaluation/
                                # ur3_reward_replay.py (sim uses finger-pad
                                # contact sensors; this is the closest real
                                # equivalent, polled at the same <=10 Hz
                                # cadence as the rest of this block).
                                last_grasped = bool(fb.get("grasped", False))
                                last_obj_flag = int(fb.get("obj_flag", 0))
                            else:
                                # Back-compat: a plain float is the sim_finger
                                # value; percent/grasp are unavailable.
                                last_gripper_fb = float(fb)
                                last_gripper_fb_pct = float("nan")
                    except Exception as e:  # noqa: BLE001
                        if step_count == 0 and debug_print:
                            print(f"\n[warn] gripper command failed: {e}; "
                                  "arm runs, gripper ignored.")
                t1_send = time.perf_counter()

                # 7. Timing control. compute_end marks the end of all real work
                # (receive..send); whatever is left until the next tick is sleep.
                compute_end = time.perf_counter()
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

                # 8. Log. Per-phase timings (seconds) let a post-run bar graph
                # break each loop tick into receive / mocap / obs / inference /
                # ctrl / send / sleep — see save_timing_breakdown().
                recv_t = t1_recv - t0_recv
                mocap_t = t1_mocap - t0_mocap
                obs_t = t1_obs - t0_obs
                policy_t = t1_policy - t0_policy
                ctrl_t = t1_ctrl - t0_ctrl
                send_t = t1_send - t0_send
                compute_t = compute_end - loop_start
                sleep_t = max(loop_end - compute_end, 0.0)
                row = {
                    "step": step_count,
                    "time": elapsed,
                    **{f"q{i}": q[i] for i in range(6)},
                    **{f"current{i}": current[i] for i in range(6)},
                    **{f"ctrl_output{i}": ctrl_output[i] for i in range(6)},
                    **{f"tcp_force{i}": tcp_force[i] for i in range(6)},
                    **{f"ctrl_pre_alpha{i}": diag["arm_ctrl_pre_alpha"][i]
                       for i in range(6)},
                    **{f"ctrl{i}": arm_ctrl[i] for i in range(6)},
                    **{f"obs_q{i}": float(obs_batch[0, i]) for i in range(6)},
                    **{f"action{i}": action[i] for i in range(7)},
                    "gripper_norm": gripper_norm,
                    "gripper_ctrl": self._gripper_ctrl,
                    "finger_pos_est": diag["finger_pos_est"],
                    "gripper_obs": float(obs_batch[0, 6]),
                    "gripper_fb_pos": last_gripper_fb,
                    "gripper_fb_pct": last_gripper_fb_pct,
                    "grasped": last_grasped,
                    "obj_flag": last_obj_flag,
                    "tcp_x": tcp_xyz[0], "tcp_y": tcp_xyz[1], "tcp_z": tcp_xyz[2],
                    "box_x": box_pos[0], "box_y": box_pos[1], "box_z": box_pos[2],
                    "box_qw": box_quat[0], "box_qx": box_quat[1],
                    "box_qy": box_quat[2], "box_qz": box_quat[3],
                    "target_x": drop_target[0], "target_y": drop_target[1],
                    "target_z": drop_target[2],
                    "box_to_target_dist": box_target_dist,
                    "recv_time_s": recv_t,
                    "mocap_time_s": mocap_t,
                    "obs_time_s": obs_t,
                    "policy_time_s": policy_t,
                    "ctrl_time_s": ctrl_t,
                    "send_time_s": send_t,
                    "compute_time_s": compute_t,
                    "sleep_time_s": sleep_t,
                    "loop_dt_true_s": loop_dt_true,
                    "loop_hz_true": loop_hz_true,
                    "overrun_count": overrun_count,
                }
                log.append(row)

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
                    stopped_reason = "timeout"
                    break

                step_count += 1
        except MocapRigidBodyLost as e:
            # Tracked rigid body lost: stop immediately and report loudly. The
            # arm is halted in `finally`; partial data is still returned/saved.
            stopped_reason = "mocap_lost"
            print("\n" + "!" * 70)
            print("MOCAP RIGID BODY LOST — STOPPING PICK LOOP")
            print(f"  {type(e).__name__}: {e}")
            print("!" * 70)
        except RobotStoppedExternally as e:
            # Protective/emergency stop, or External Control no longer
            # PLAYING (e.g. operator pressed Stop on the pendant). Arm is
            # halted in `finally`; partial data is still returned/saved.
            stopped_reason = "external_stop"
            print("\n" + "!" * 70)
            print("ROBOT STOPPED EXTERNALLY — STOPPING PICK LOOP")
            print(f"  {type(e).__name__}: {e}")
            print("!" * 70)
        except Exception as e:  # noqa: BLE001
            # Any other RTDE/robot-side exception (connection drop, servoj
            # rejection, etc.) -- stop the loop rather than crash the script,
            # keep the partial log, and tag it distinctly from a clean
            # completion/timeout/mocap-loss/external-stop.
            stopped_reason = "robot_error"
            print("\n" + "!" * 70)
            print("ROBOT/RTDE ERROR — STOPPING PICK LOOP")
            print(f"  {type(e).__name__}: {e}")
            print("!" * 70)
        finally:
            if self._control is not None:
                try:
                    self._control.servoStop()
                except Exception:
                    pass

        total_wall = time.perf_counter() - start_time
        df = pd.DataFrame(log)
        stats = self.summarize_policy_loop(df, control_hz, timeout_s, total_wall)
        stats["stopped_reason"] = stopped_reason
        if debug_print:
            print(f"\nPolicy loop finished (reason: {stopped_reason}).")
        return df, stats

    def summarize_policy_loop(self, df, requested_control_hz, timeout_s,
                              total_wall_time_s) -> dict:
        if len(df) == 0:
            return {"num_samples": 0}
        n = len(df) - 1
        true_hz = n / total_wall_time_s if total_wall_time_s > 0 else None
        phase_means = {
            f"mean_{c}": float(df[c].mean())
            for c, _ in self.TIMING_PHASES if c in df.columns
        }
        return {
            **phase_means,
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
                               drop_target=None, box_quat=None) -> np.ndarray:
        """Sync robot+box+target state into MuJoCo, run mj_forward, render.

        box_quat (w, x, y, z, base frame) sets the box freejoint ORIENTATION.
        Without it the rendered cube stays axis-aligned even when the real
        cube is rotated (the mocap streams the cube's orientation, mapped to
        the base frame and logged as box_qw/qx/qy/qz) -- so the replay looked
        "straight" while the physical 3x3x4 cm cube was yawed. Left None ->
        identity (old behavior, for logs without the quaternion columns).
        """
        data = mj["data"]
        data.qpos[:6] = np.asarray(q, dtype=float)
        if gripper_ctrl is not None:
            # Both fingers ~ gripper_ctrl (tendon coupling; small range).
            data.qpos[6] = float(gripper_ctrl)
            data.qpos[7] = float(gripper_ctrl)
        if box_pos is not None:
            adr = mj["box_qposadr"]
            data.qpos[adr:adr + 3] = np.asarray(box_pos, dtype=float)
            # Box orientation: default to identity so a missing quaternion
            # reproduces the old axis-aligned render exactly.
            q_box = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if box_quat is not None:
                q_box = np.asarray(box_quat, dtype=float)
                nrm = np.linalg.norm(q_box)
                q_box = q_box / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
            data.qpos[adr + 3:adr + 7] = q_box
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
        has_quat = all(c in df.columns
                       for c in ["box_qw", "box_qx", "box_qy", "box_qz"])
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
            box_q = (row[["box_qw", "box_qx", "box_qy", "box_qz"]].to_numpy(dtype=float)
                     if has_quat else None)
            tgt = (row[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
                   if has_tgt else None)
            frames.append(self.mujoco_sync_and_render(mj, q, grip, box, tgt,
                                                       box_quat=box_q))

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

    def save_robot_plots(self, df: pd.DataFrame, out_path: str) -> str:
        """Save the ROBOT (arm) diagnostic plot (PNG): the per-signal control
        pipeline top-to-bottom — measured state, the state sent to the policy, the
        raw action, then the command after each processing stage.

        Panels: box→target distance (context), joint positions (RTDE receive),
        joint velocities (RTDE receive), joint positions in the obs sent to the
        policy, policy action before scaling (arm), arm ctrl before alpha
        smoothing, actual command sent to servoj. Panels whose columns are absent
        (an older log) are skipped. House style mirrors the UR10 sibling.
        """
        import matplotlib.pyplot as plt

        joint_names = self.JOINT_NAMES
        t = df["time"]

        def _has(prefix):
            return all(f"{prefix}{i}" in df.columns for i in range(6))

        def _per_joint(ax, prefix):
            for i, name in enumerate(joint_names):
                ax.plot(t, df[f"{prefix}{i}"], label=name)
            ax.legend(loc="upper right", ncol=3, fontsize=8)

        # (key, kind, ylabel, title). "box" is a single-line context strip.
        specs = []
        if "box_to_target_dist" in df.columns:
            specs.append(("box", "scalar", "distance (cm)",
                          "Box→target and gripper→box distance"))
        if _has("q"):
            specs.append(("q", "joint", "position (rad)",
                          "Joint positions (RTDE receive)"))
        if _has("obs_q"):
            specs.append(("obs_q", "joint", "position (rad)",
                          "Joint positions in state sent to policy"))
        if _has("action"):
            specs.append(("action", "joint", "action [-1,1]",
                          "Policy action before scaling (arm)"))
        if _has("ctrl_pre_alpha"):
            specs.append(("ctrl_pre_alpha", "joint", "ctrl (rad)",
                          "Arm ctrl before alpha smoothing"))
        if _has("ctrl"):
            specs.append(("ctrl", "joint", "ctrl (rad)",
                          "Actual command sent to servoj"))

        n = len(specs)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (key, kind, ylabel, title) in zip(axes, specs):
            if kind == "scalar":
                ax.plot(t, df["box_to_target_dist"] * 100, label="box → target")
                tcp_box_cols = ("tcp_x", "tcp_y", "tcp_z",
                                "box_x", "box_y", "box_z")
                if all(c in df.columns for c in tcp_box_cols):
                    finger_box = np.sqrt(
                        (df["box_x"] - df["tcp_x"]) ** 2
                        + (df["box_y"] - df["tcp_y"]) ** 2
                        + (df["box_z"] - df["tcp_z"]) ** 2
                    )
                    ax.plot(t, finger_box * 100, label="gripper → box (tcp)")
                    ax.legend(loc="upper right", fontsize=8)
            else:
                _per_joint(ax, key)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[save_robot_plots] Saved to {out_path}")
        return out_path

    def save_gripper_plots(self, df: pd.DataFrame, out_path: str) -> str:
        """Save the GRIPPER diagnostic plot (PNG): the finger control pipeline,
        mirroring save_robot_plots for the single gripper DOF — so the oscillation
        is visible per signal.

        Panels: measured finger position (Hand-E readback) vs the plant estimate,
        finger velocity (numeric d/dt), finger position in the obs sent to the
        policy, policy gripper action before scaling, raw integrator [0,0.05] vs
        the plant estimate (the low-pass + slew that tames the command), and the
        gripper command actually sent. Panels are skipped when columns are absent;
        the readback line is empty in a dry-run (gripper_fb_pos all NaN).
        """
        import matplotlib.pyplot as plt

        t = df["time"]
        t_np = t.to_numpy()

        def has(c):
            return c in df.columns

        def _deriv(col):
            return np.gradient(df[col].to_numpy(dtype=float), t_np)

        kinds = []
        if has("finger_pos_est") or has("gripper_fb_pos"):
            kinds += ["pos", "vel"]
        if has("gripper_obs"):
            kinds.append("obs")
        if has("action6"):
            kinds.append("action")
        if has("gripper_ctrl"):
            kinds.append("ctrl")
        if has("gripper_norm"):
            kinds.append("cmd")

        n = len(kinds)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, kind in zip(axes, kinds):
            if kind == "pos":
                if has("finger_pos_est"):
                    ax.plot(t, df["finger_pos_est"] * 1000, label="plant estimate")
                if has("gripper_fb_pos"):
                    ax.plot(t, df["gripper_fb_pos"] * 1000, label="Hand-E readback",
                            color="k", ls="--", lw=1.2)
                ax.set_ylabel("finger (mm)")
                ax.set_title("Finger position: plant estimate vs Hand-E readback "
                             "(0 open, 25 closed)")
                ax.legend(loc="upper right", fontsize=8)
            elif kind == "vel":
                if has("finger_pos_est"):
                    ax.plot(t, _deriv("finger_pos_est") * 1000, label="plant estimate")
                if has("gripper_fb_pos"):
                    ax.plot(t, _deriv("gripper_fb_pos") * 1000, color="k", ls="--",
                            lw=1.0, alpha=0.7, label="Hand-E readback")
                ax.set_ylabel("finger vel (mm/s)")
                ax.set_title("Finger velocity (numeric d/dt)")
                ax.legend(loc="upper right", fontsize=8)
            elif kind == "obs":
                ax.plot(t, df["gripper_obs"] * 1000)
                ax.set_ylabel("finger (mm)")
                ax.set_title("Finger position in state sent to policy")
            elif kind == "action":
                ax.plot(t, df["action6"])
                ax.set_ylabel("action [-1,1]")
                ax.set_title("Policy gripper action before scaling (action6)")
            elif kind == "ctrl":
                ax.plot(t, df["gripper_ctrl"] * 1000, label="raw integrator [0,50]")
                if has("finger_pos_est"):
                    ax.plot(t, df["finger_pos_est"] * 1000,
                            label="plant estimate (low-pass + slew)")
                ax.set_ylabel("finger (mm)")
                ax.set_title("Gripper ctrl before vs after smoothing")
                ax.legend(loc="upper right", fontsize=8)
            elif kind == "cmd":
                ax.plot(t, df["gripper_norm"], drawstyle="steps-post")
                ax.set_ylabel("gripper [0,1]")
                ax.set_title("Gripper command actually sent (0 open, 1 closed)")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[save_gripper_plots] Saved to {out_path}")
        return out_path

    # Per-phase timing columns logged each loop tick, in execution order, with
    # the bar labels used by save_timing_breakdown().
    TIMING_PHASES = [
        ("recv_time_s", "RTDE receive"),
        ("mocap_time_s", "mocap read"),
        ("obs_time_s", "build obs"),
        ("policy_time_s", "inference"),
        ("ctrl_time_s", "ctrl update"),
        ("send_time_s", "servoj+gripper"),
        ("sleep_time_s", "sleep (idle)"),
    ]

    def save_timing_breakdown(self, df: pd.DataFrame, out_path: str) -> str:
        """Save a per-loop-step timing bar graph (PNG).

        Two panels:
          * top — mean COMPUTE time per phase (ms) across the run, with std-dev
            error bars: where each 1/control_hz tick's real work goes. The
            "sleep (idle)" phase is excluded — it is the pacing wait that fills
            the tick out to the control period, not work (still kept in the CSV
            and in summarize_policy_loop's mean_sleep_time_s).
          * bottom — a stacked bar PER STEP (the compute phases only) so spikes
            and overruns are visible against the control-period budget line.
        """
        import matplotlib.pyplot as plt

        phases = [
            (c, lbl)
            for c, lbl in self.TIMING_PHASES
            if c in df.columns and c != "sleep_time_s"
        ]
        cols = [c for c, _ in phases]
        labels = [lbl for _, lbl in phases]
        means_ms = [df[c].mean() * 1e3 for c in cols]
        stds_ms = [df[c].std() * 1e3 for c in cols]

        fig, axes = plt.subplots(2, 1, figsize=(14, 11))

        # Panel 1: mean per-phase time (ms).
        ax = axes[0]
        bars = ax.bar(labels, means_ms, yerr=stds_ms, capsize=4,
                      color="tab:blue", alpha=0.8)
        for rect, m in zip(bars, means_ms):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("mean time per step (ms)")
        ax.set_title("Loop compute timing — mean ± std per control step "
                     "(sleep/idle excluded)")
        ax.grid(True, axis="y", alpha=0.3)

        # Panel 2: stacked compute phases per step (exclude sleep), plus the
        # control-period budget line.
        ax = axes[1]
        compute_cols = [c for c in cols if c != "sleep_time_s"]
        compute_lbls = [lbl for c, lbl in phases if c != "sleep_time_s"]
        steps = df["step"].to_numpy()
        bottom = np.zeros(len(df))
        for c, lbl in zip(compute_cols, compute_lbls):
            vals = df[c].to_numpy() * 1e3
            ax.bar(steps, vals, bottom=bottom, label=lbl, width=1.0)
            bottom += vals
        if "loop_dt_true_s" in df.columns:
            mean_period_ms = df["loop_dt_true_s"].mean() * 1e3
            ax.axhline(mean_period_ms, color="k", ls="--", lw=1,
                       label=f"mean loop period ({mean_period_ms:.1f} ms)")
        ax.set_xlabel("step")
        ax.set_ylabel("compute time (ms)")
        ax.set_title("Per-step compute breakdown (stacked, sleep excluded)")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[save_timing_breakdown] Saved to {out_path}")
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
