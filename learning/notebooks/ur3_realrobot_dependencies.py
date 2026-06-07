"""
RTDE dependencies for the UR3 + Hand-E pick task.

Mirrors learning/notebooks/URSim_RTDE_dependencies.py (the UR10 reach backbone)
but adapted for the UR3Pick policy:

- 26D observation  [q(8), qd(6), (box-tcp)(3), (target-box)(3), box_xmat[:6](6)]
                   q = 6 arm joints + 2 finger positions; box_xmat[:6] = first
                   two rows of the box rotation matrix (from the mocap quaternion)
- 7D action        6 arm joint deltas + 1 Hand-E gripper delta
- box pose comes from a Nokov rigid body (motion_capture.mocap_dependencies):
  xyz via get_rigid_body_xyz(), orientation via get_rigid_body_quat()
- the lift target is supplied by the caller (hardcoded in the pick-loop script)
- the gripper command is sent on a SEPARATE channel from servoj (servoJ only
  accepts the 6 arm joints); see send_gripper(), which is stubbed until the
  lab Hand-E wiring (Robotiq URCap register vs. tool I/O) is confirmed.

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
    ):
        self.host = host
        self.port_rtde = port_rtde
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

    # =========================================================================
    # A — RTDE Connection & Feedback
    # =========================================================================

    def connect(self) -> rtde_receive.RTDEReceiveInterface:
        if self._receiver is None:
            self._receiver = rtde_receive.RTDEReceiveInterface(self.host)
        if self._control is None:
            self._control = rtde_control.RTDEControlInterface(self.host)
        return self._receiver

    def disconnect(self):
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

    def send_movej(self, q, a=0.4, v=0.4, textmsg=None):
        """Asynchronous moveJ via rtde_control."""
        self.connect()
        if textmsg is not None:
            print(f"[movej] {textmsg}")
        self._control.moveJ(list(map(float, q)), float(v), float(a), True)

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

    def send_gripper(self, norm_cmd: float):
        """Send a normalized [0,1] gripper command to the real Hand-E.

        0.0 = fully closed, 1.0 = fully open (matches the sim tendon ctrl
        mapped from [0, 0.05]).

        STUB: servoJ controls only the 6 arm joints, so the Hand-E command
        must go on a separate channel. On the lab UR3 this is typically the
        Robotiq URCap reading an RTDE register, or tool digital/analog I/O.
        Wire one of these up before running on hardware, e.g.:

            # via RTDE IO register (Robotiq URCap listening on it):
            # self._io.setInputIntRegister(18, int(norm_cmd * 255))

        On URSim there is no gripper, so the pick-loop script passes a no-op
        for `gripper_fn` during dry runs.
        """
        raise NotImplementedError(
            "send_gripper() is unwired. Provide a gripper_fn to run_policy_loop "
            "(or override this method) matching the lab Hand-E setup."
        )

    # =========================================================================
    # C — Movement Helpers
    # =========================================================================

    def joint_error_norm(self, q_target) -> float:
        fb = self.receive_feedback()
        q = np.asarray(fb["q"], dtype=float)
        return float(np.linalg.norm(np.asarray(q_target, dtype=float) - q))

    def reached_joint_target(self, q_target, tol=0.05) -> bool:
        return self.joint_error_norm(q_target) < tol

    def move_to_start(self, q_start, a=0.5, v=0.1, timeout_s=20.0, tol=0.01,
                      check_hz=5.0, debug_print=True):
        """Send movej to q_start and poll RTDE until converged."""
        dt = 1.0 / float(check_hz)
        q_start = np.asarray(q_start, dtype=float)

        if debug_print:
            print(f"Moving to start pose with movej (a={a}, v={v})...")

        self.send_movej(q_start.tolist(), a=a, v=v, textmsg="go_start")
        t0 = time.perf_counter()

        while True:
            q_now = np.array(self.receive_feedback()["q"], dtype=float)
            err = np.linalg.norm(q_now - q_start)
            if debug_print:
                print(f"\r  err={err:.4f}  q={np.round(q_now, 3)}", end="", flush=True)
            if err < tol:
                break
            if (time.perf_counter() - t0) > timeout_s:
                print("\n  Pre-position timeout!")
                break
            time.sleep(dt)

        if debug_print:
            print(f"\n  Reached start in {time.perf_counter() - t0:.2f}s")

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

        Fetches the run's `model` artifact (params.msgpack) and writes a
        metadata.json alongside it (env_name + network_factory from run.config,
        obs_dim/action_dim resolved from the registry so they track the env).
        Skips the download if out_dir already has both files unless force=True.
        Returns out_dir.
        """
        import json as _json
        import shutil

        params_out = os.path.join(out_dir, "params.msgpack")
        meta_out = os.path.join(out_dir, "metadata.json")
        if not force and os.path.exists(params_out) and os.path.exists(meta_out):
            print(f"[wandb] policy already present at {out_dir} (skipping download)")
            return out_dir

        import wandb

        os.makedirs(out_dir, exist_ok=True)
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        policy_art = next(
            (a for a in run.logged_artifacts() if a.type == "model"), None
        )
        if policy_art is None:
            raise ValueError(f"No 'model' artifact found for run {run_id}")
        art_dir = policy_art.download(root=os.path.join(out_dir, "_artifact"))
        shutil.copyfile(os.path.join(art_dir, "params.msgpack"), params_out)

        env_name = run.config.get("env_name", "UR3Pick")
        env = registry.load(env_name)
        nf = run.config.get("network_factory", {}) or {}
        metadata = {
            "env_name": env_name,
            "obs_dim": int(env.observation_size),
            "action_dim": int(env.action_size),
            "network_factory": nf,
            "wandb_run_id": run_id,
            "wandb_entity": entity,
            "wandb_project": project,
            "action_scale": run.config.get("action_scale", 0.04),
            "ctrl_dt": run.config.get("ctrl_dt", 0.02),
            "episode_length": run.config.get("episode_length"),
        }
        with open(meta_out, "w") as f:
            _json.dump(metadata, f, indent=2)
        print(f"[wandb] downloaded policy -> {out_dir} "
              f"(obs={metadata['obs_dim']}, act={metadata['action_dim']})")
        return out_dir

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
        """Build 26D obs to match UR3Pick._get_obs:
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

                # 2. Box pose from mocap (fall back to last good value)
                box_pos = mocap_reader.get_rigid_body_xyz()
                if box_pos is None:
                    box_pos = (last_box_pos if last_box_pos is not None
                               else drop_target.copy())
                box_pos = np.asarray(box_pos, dtype=float)
                last_box_pos = box_pos.copy()

                # Box orientation (quaternion w,x,y,z) — internal obs detail; the
                # caller only places the box. Identity fallback if unavailable.
                box_quat = mocap_reader.get_rigid_body_quat()
                if box_quat is None:
                    box_quat = last_box_quat
                else:
                    box_quat = np.asarray(box_quat, dtype=float)
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
                try:
                    gripper_fn(gripper_norm)
                except NotImplementedError:
                    if step_count == 0 and debug_print:
                        print("\n[warn] gripper unwired (send_gripper stub); "
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
        ax.set_title("Gripper command (0=closed, 1=open)")
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
