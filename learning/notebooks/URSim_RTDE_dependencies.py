"""
URSim RTDE dependencies for UR10 Simple Reach task.

Simplified from the VT2 URSimRTDEControlFeedback class:
- 18D observation (q, qd, tcp_pos, target_pos) instead of 57D
- 6D action (arm only) instead of 7D (no gripper)
- No rotation matrices, no object tracking
"""

from typing import Dict, List, Optional, Tuple
import os
import pickle
import socket
import time

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd
import rtde_receive
from IPython.display import clear_output

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params


class URSimRTDESimpleReach:
    """RTDE control loop for the UR10 simple reach task (6-DOF arm, 18D obs)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port_rtde: int = 30004,
        port_urscript: int = 30002,
        port_dashboard: int = 29999,
    ):
        self.host = host
        self.port_rtde = port_rtde
        self.port_urscript = port_urscript
        self.port_dashboard = port_dashboard
        self._receiver: Optional[rtde_receive.RTDEReceiveInterface] = None
        self._policy_fn = None
        self._ctrl_state: Optional[np.ndarray] = None
        self._ctrl_lowers: Optional[np.ndarray] = None
        self._ctrl_uppers: Optional[np.ndarray] = None

    # =========================================================================
    # A — RTDE Connection & Feedback
    # =========================================================================

    def connect(self) -> rtde_receive.RTDEReceiveInterface:
        if self._receiver is None:
            self._receiver = rtde_receive.RTDEReceiveInterface(self.host)
        return self._receiver

    def disconnect(self):
        if self._receiver is not None:
            self._receiver.disconnect()
            self._receiver = None

    def is_connected(self) -> bool:
        if self._receiver is None:
            return False
        return self._receiver.isConnected()

    def receive_feedback(self) -> Dict[str, List[float]]:
        """Returns dict: q(6), qd(6), tcp_xyz(3), tcp_pose(6)."""
        r = self.connect()
        q = r.getActualQ()
        qd = r.getActualQd()
        tcp_pose = r.getActualTCPPose()
        tcp_pose[0] = -tcp_pose[0]
        tcp_pose[1] = -tcp_pose[1]
        return {
            "q": list(q),
            "qd": list(qd),
            "tcp_xyz": list(tcp_pose[:3]),
            "tcp_pose": list(tcp_pose),
        }

    def print_feedback(self, digits: int = 4):
        fb = self.receive_feedback()
        print("q      =", [round(v, digits) for v in fb["q"]])
        print("qd     =", [round(v, digits) for v in fb["qd"]])
        print("tcp_xyz=", [round(v, digits) for v in fb["tcp_xyz"]])

    # =========================================================================
    # B — Socket Commands (URScript on port 30002, Dashboard on port 29999)
    # =========================================================================

    def send_urscript(self, script: str, timeout: float = 2.0):
        with socket.create_connection((self.host, self.port_urscript), timeout=timeout) as s:
            s.sendall(script.encode("utf-8"))

    def dashboard_send(self, cmd: str, timeout: float = 2.0) -> str:
        with socket.create_connection((self.host, self.port_dashboard), timeout=timeout) as s:
            banner = s.recv(4096).decode("utf-8", errors="ignore")
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            time.sleep(0.05)
            resp = s.recv(4096).decode("utf-8", errors="ignore")
        return (banner + resp).strip()

    def urscript_program(self, lines, name="py_prog") -> str:
        body = "\n  ".join(lines)
        return f"def {name}():\n  {body}\nend\n{name}()\n"

    def urscript_movej(self, q, a=0.3, v=0.3) -> str:
        return f"movej({list(map(float, q))}, a={a}, v={v})"

    def urscript_textmsg(self, msg: str) -> str:
        safe = msg.replace('"', "'")
        return f'textmsg("{safe}")'

    def send_movej(self, q, a=0.4, v=0.4, textmsg=None):
        lines = []
        if textmsg is not None:
            lines.append(self.urscript_textmsg(textmsg))
        lines.append(self.urscript_movej(q, a=a, v=v))
        prog = self.urscript_program(lines, name="movej_cmd")
        self.send_urscript(prog)

    def urscript_servoj(self, q, t=0.04, lookahead_time=0.1, gain=300) -> str:
        return (
            f"servoj({list(map(float, q))}, "
            f"t={float(t)}, "
            f"lookahead_time={float(lookahead_time)}, "
            f"gain={int(gain)})"
        )

    def send_servoj(self, q, t=0.04, lookahead_time=0.1, gain=300, textmsg=None):
        lines = []
        if textmsg is not None:
            lines.append(self.urscript_textmsg(textmsg))
        lines.append(self.urscript_servoj(q, t=t, lookahead_time=lookahead_time, gain=gain))
        prog = self.urscript_program(lines, name="servoj_cmd")
        self.send_urscript(prog)

    # =========================================================================
    # C — Movement Helpers
    # =========================================================================

    def joint_error_norm(self, q_target) -> float:
        fb = self.receive_feedback()
        q = np.asarray(fb["q"], dtype=float)
        return float(np.linalg.norm(np.asarray(q_target, dtype=float) - q))

    def reached_joint_target(self, q_target, tol=0.05) -> bool:
        return self.joint_error_norm(q_target) < tol

    def move_to_start(
        self,
        q_start,
        a=2.5,
        v=2.0,
        timeout_s=20.0,
        tol=0.01,
        check_hz=5.0,
        debug_print=True,
    ):
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

    def load_policy_fn(
        self,
        policy_path: str = "../../evaluation/downloaded_policies/simple_reach_policy",
        deterministic: bool = True,
    ):
        """Load a Brax PPO policy from saved files (metadata.json + params.msgpack).

        Everything is read from the saved artifacts — env_name, obs_dim, action_dim,
        network architecture. Same pattern as UR10_SimpleReach.ipynb chunk 9.
        """
        import json as _json
        from flax import serialization

        # ---- Load metadata (env_name, dims, architecture) ----
        meta_file = os.path.join(policy_path, "metadata.json")
        with open(meta_file) as f:
            meta = _json.load(f)

        nf_kwargs = {
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in meta["network_factory"].items()
        }

        # ---- Build PPO network from metadata ----
        ppo_net = ppo_networks.make_ppo_networks(
            observation_size=meta["obs_dim"],
            action_size=meta["action_dim"],
            preprocess_observations_fn=running_statistics.normalize,
            **nf_kwargs,
        )

        # ---- Build template and deserialize params.msgpack ----
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

        # ---- Build JIT-compiled inference function ----
        _raw_policy = ppo_networks.make_inference_fn(ppo_net)(
            (loaded_params["0"], loaded_params["1"]), deterministic=deterministic
        )

        @jax.jit
        def policy_fn(obs):
            action, _ = _raw_policy(obs, jax.random.PRNGKey(0))
            return action

        # ---- Store on self ----
        self._policy_fn = policy_fn
        self._meta = meta
        self.policy_path = policy_path

        # Extract actuator ctrl limits (needs env model)
        env = registry.load(meta["env_name"])
        lowers, uppers = env.mj_model.actuator_ctrlrange.T
        self._ctrl_lowers = np.array(lowers, dtype=np.float32)
        self._ctrl_uppers = np.array(uppers, dtype=np.float32)

        print(f"Loaded policy: obs={meta['obs_dim']}, act={meta['action_dim']}")

    # =========================================================================
    # E — MuJoCo FK model (computes tcp_pos from joint angles)
    # =========================================================================

    def init_fk_model(self, xml_path: str):
        """Load a MuJoCo model for forward kinematics.

        RTDE's getActualTCPPose() returns the UR controller's TCP which has a
        different offset than the MuJoCo 'tcp' site the policy was trained with
        (12 cm difference). Using MuJoCo FK from joint angles gives exactly
        the tcp_pos the policy expects.
        """
        self._fk_model = mujoco.MjModel.from_xml_path(xml_path)
        self._fk_data = mujoco.MjData(self._fk_model)
        self._fk_tcp_id = mujoco.mj_name2id(
            self._fk_model, mujoco.mjtObj.mjOBJ_SITE, "tcp"
        )
        print(f"FK model loaded: tcp site id={self._fk_tcp_id}")

    def compute_tcp_pos(self, q: np.ndarray) -> np.ndarray:
        """Compute MuJoCo tcp site position from joint angles via FK."""
        self._fk_data.qpos[:6] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self._fk_model, self._fk_data)
        return self._fk_data.site_xpos[self._fk_tcp_id].copy()

    # =========================================================================
    # F — Observation Building (18D)
    # =========================================================================

    def build_obs_from_feedback(
        self,
        fb: dict,
        target_pos: np.ndarray,
        dtype=np.float32,
    ) -> np.ndarray:
        """Build 18D observation: [q(6), qd(6), tcp_pos(3), target_pos(3)].

        tcp_pos from RTDE getActualTCPPose()[:3]. This matches training because
        the MuJoCo TCP site is now at the tool flange (pos="0 0 0"), same as
        the UR default TCP.
        Returns shape (1, 18) for batched policy input.
        """
        q = np.array(fb["q"], dtype=dtype)
        qd = np.array(fb["qd"], dtype=dtype)
        tcp = np.array(fb["tcp_xyz"], dtype=dtype)
        tgt = np.array(target_pos, dtype=dtype)
        obs = np.concatenate([q, qd, tcp, tgt])
        return obs[None, :]  # (1, 18)

    # =========================================================================
    # G — Control Update
    # =========================================================================

    def policy_step_ctrl_update(
        self,
        action: np.ndarray,
        action_scale: float = 0.04,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Delta-style control update:
        ctrl_next = clip(ctrl_prev + action_scale * action, lowers, uppers).
        Returns ctrl_next (6,) — the joint command to send via servoj.
        Upper and Lower Limits are the actuator ctrlrange
        """
        if self._ctrl_state is None:
            raise RuntimeError("ctrl_state not initialized (call run_policy_loop first)")

        action = np.asarray(action, dtype=dtype).reshape(-1)
        if action.shape[0] != 6:
            raise ValueError(f"action must have length 6, got {action.shape[0]}")

        ctrl_prev = self._ctrl_state.copy()
        ctrl_next = ctrl_prev + float(action_scale) * action

        if self._ctrl_lowers is not None and self._ctrl_uppers is not None:
            ctrl_next = np.clip(ctrl_next, self._ctrl_lowers, self._ctrl_uppers)

        self._ctrl_state = ctrl_next.astype(dtype)
        return self._ctrl_state.copy()

    # =========================================================================
    # H — Control Loop
    # =========================================================================

    def run_policy_loop(
        self,
        target_pos: np.ndarray,
        control_hz: float = 50.0,
        timeout_s: float = 10.0,
        action_scale: float = 0.04,
        lookahead_time: float = 0.1,
        gain: int = 300,
        dtype=np.float32,
        debug_print: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """Run the policy control loop.

        Returns (DataFrame with per-step logs, summary stats dict).
        """
        if self._policy_fn is None:
            raise RuntimeError("Policy not loaded. Call load_policy_fn() first.")

        dt = 1.0 / float(control_hz)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)

        # Initialize ctrl_state from current robot joint positions
        fb0 = self.receive_feedback()
        self._ctrl_state = np.array(fb0["q"], dtype=dtype)

        log = []
        step_count = 0
        overrun_count = 0
        prev_tcp_xyz = None
        prev_tcp_to_target_dist = None

        start_time = time.perf_counter()
        next_tick = start_time

        while True:
            loop_start = time.perf_counter()

            # 1. RTDE receive
            t0_obs = time.perf_counter()
            fb = self.receive_feedback()
            t1_obs = time.perf_counter()

            q = np.asarray(fb["q"], dtype=float)
            qd = np.asarray(fb["qd"], dtype=float)
            tcp_xyz = np.asarray(fb["tcp_xyz"], dtype=float)

            # TCP motion tracking (using RTDE TCP directly)
            if prev_tcp_xyz is None:
                tcp_delta = np.zeros(3, dtype=float)
                tcp_dist_loop = 0.0
            else:
                tcp_delta = tcp_xyz - prev_tcp_xyz
                tcp_dist_loop = float(np.linalg.norm(tcp_delta))
            prev_tcp_xyz = tcp_xyz.copy()

            # TCP to target distance
            tcp_to_target_dist = float(np.linalg.norm(target_pos - tcp_xyz))
            if prev_tcp_to_target_dist is None:
                tcp_to_target_improvement = 0.0
            else:
                tcp_to_target_improvement = prev_tcp_to_target_dist - tcp_to_target_dist
            prev_tcp_to_target_dist = tcp_to_target_dist

            # 2. Build observation
            obs_batch = self.build_obs_from_feedback(fb, target_pos, dtype)

            # 3. Policy inference
            t0_policy = time.perf_counter()
            action_raw = self._policy_fn(obs_batch)
            action = np.asarray(action_raw, dtype=dtype).reshape(-1)
            t1_policy = time.perf_counter()

            # 4. Control update (with clipping)
            ctrl_prev = self._ctrl_state.copy() if self._ctrl_state is not None else np.zeros(6)
            ctrl_next = self.policy_step_ctrl_update(action, action_scale, dtype)

            # 5. Send servoj
            t0_send = time.perf_counter()
            self.send_servoj(
                ctrl_next.tolist(),
                t=dt,
                lookahead_time=lookahead_time,
                gain=gain,
            )
            t1_send = time.perf_counter()

            # 6. Timing control
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

            # 7. Log
            qd_norm = float(np.linalg.norm(qd))
            cmd_err_norm = float(np.linalg.norm(ctrl_next - q))
            ctrl_delta_norm = float(np.linalg.norm(ctrl_next - ctrl_prev))

            row = {
                "step": step_count,
                "time": elapsed,
                **{f"q{i}": q[i] for i in range(6)},
                **{f"qd{i}": qd[i] for i in range(6)},
                **{f"ctrl{i}": ctrl_next[i] for i in range(6)},
                **{f"action{i}": action[i] for i in range(6)},
                "tcp_x": tcp_xyz[0], "tcp_y": tcp_xyz[1], "tcp_z": tcp_xyz[2],
                "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
                "tcp_to_target_dist": tcp_to_target_dist,
                "tcp_to_target_improvement": tcp_to_target_improvement,
                "tcp_dx": tcp_delta[0], "tcp_dy": tcp_delta[1], "tcp_dz": tcp_delta[2],
                "tcp_dist_loop_m": tcp_dist_loop,
                "tcp_speed_mps": tcp_dist_loop / max(loop_dt_true, 1e-9),
                "qd_norm": qd_norm,
                "cmd_err_norm": cmd_err_norm,
                "ctrl_delta_norm": ctrl_delta_norm,
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
                    f"\r[{step_count:4d}] tcp→tgt={tcp_to_target_dist:.4f}m "
                    f"| hz={loop_hz_true:.1f} "
                    f"| tcp={np.round(tcp_xyz, 3)}",
                    end="", flush=True,
                )

            if elapsed >= timeout_s:
                break

            step_count += 1

        total_wall = time.perf_counter() - start_time
        df = pd.DataFrame(log)

        stats = self.summarize_policy_loop(df, control_hz, timeout_s, total_wall)

        if debug_print:
            print("\nPolicy loop finished.")

        return df, stats

    def summarize_policy_loop(
        self,
        df: pd.DataFrame,
        requested_control_hz: float,
        timeout_s: float,
        total_wall_time_s: float,
    ) -> dict:
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
            "mean_loop_dt_true_s": float(df["loop_dt_true_s"].mean()),
            "std_loop_dt_true_s": float(df["loop_dt_true_s"].std()),
            "mean_loop_hz_true": float(df["loop_hz_true"].mean()),
            "mean_obs_time_s": float(df["obs_time_s"].mean()),
            "mean_policy_time_s": float(df["policy_time_s"].mean()),
            "mean_send_time_s": float(df["send_time_s"].mean()),
            "max_obs_time_s": float(df["obs_time_s"].max()),
            "max_policy_time_s": float(df["policy_time_s"].max()),
            "max_send_time_s": float(df["send_time_s"].max()),
            "start_tcp_to_target_dist": float(df["tcp_to_target_dist"].iloc[0]),
            "final_tcp_to_target_dist": float(df["tcp_to_target_dist"].iloc[-1]),
            "min_tcp_to_target_dist": float(df["tcp_to_target_dist"].min()),
            "net_improvement": float(
                df["tcp_to_target_dist"].iloc[0] - df["tcp_to_target_dist"].iloc[-1]
            ),
            "mean_tcp_speed_mps": float(df["tcp_speed_mps"].mean()),
            "max_tcp_speed_mps": float(df["tcp_speed_mps"].max()),
            "mean_qd_norm": float(df["qd_norm"].mean()),
            "max_qd_norm": float(df["qd_norm"].max()),
            "mean_cmd_err_norm": float(df["cmd_err_norm"].mean()),
            "max_cmd_err_norm": float(df["cmd_err_norm"].max()),
            "num_overruns": int(df["overrun_count"].iloc[-1]),
            "overrun_ratio": float(df["overrun_count"].iloc[-1]) / max(len(df), 1),
        }

    def print_stats(self, stats: dict, keys: Optional[List[str]] = None):
        """Pretty-print stats dict (all keys or selected subset)."""
        items = stats.items() if keys is None else [(k, stats.get(k, "<missing>")) for k in keys]
        for k, v in items:
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    # =========================================================================
    # I — MuJoCo Rendering
    # =========================================================================

    def mujoco_init_model(
        self,
        xml_path: str,
        height: int = 480,
        width: int = 640,
        cam_lookat=(0.4, 0.0, 0.4),
        cam_distance: float = 1.8,
        cam_azimuth: float = 130,
        cam_elevation: float = -20,
    ) -> dict:
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

        return {
            "model": model,
            "data": data,
            "renderer": renderer,
            "cam": cam,
        }

    def mujoco_sync_and_render(
        self,
        mj: dict,
        q: np.ndarray,
        qd: Optional[np.ndarray] = None,
        target_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sync robot state into MuJoCo, run mj_forward, render a frame."""
        data = mj["data"]
        data.qpos[:6] = np.asarray(q, dtype=float)
        if qd is not None:
            data.qvel[:6] = np.asarray(qd, dtype=float)
        if target_pos is not None and data.mocap_pos.shape[0] > 0:
            data.mocap_pos[0, :] = np.asarray(target_pos, dtype=float)

        mujoco.mj_forward(mj["model"], data)
        mj["renderer"].update_scene(data, camera=mj["cam"])
        return mj["renderer"].render().copy()

    def render_video_from_log(
        self,
        mj: dict,
        df: pd.DataFrame,
        video_fps: float = 30.0,
        q_cols: Optional[List[str]] = None,
        qd_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
    ) -> list:
        """Render frames from logged DataFrame at the desired video fps.

        Uses stride-based decimation so the video plays at approximately real-time.
        Returns a list of numpy frames for use with mediapy.show_video().
        """
        if q_cols is None:
            q_cols = [f"q{i}" for i in range(6)]
        if qd_cols is None:
            qd_cols = [f"qd{i}" for i in range(6)]
        if target_cols is None:
            target_cols = ["target_x", "target_y", "target_z"]

        has_qd = all(c in df.columns for c in qd_cols)
        has_target = all(c in df.columns for c in target_cols)

        # Compute stride from logged Hz
        if "loop_hz_true" in df.columns:
            logged_hz = float(df["loop_hz_true"].mean())
        else:
            dts = df["time"].diff().dropna()
            logged_hz = 1.0 / dts.mean() if len(dts) > 0 else video_fps

        stride = max(1, int(round(logged_hz / video_fps)))
        actual_video_fps = logged_hz / stride

        print(f"[render_video_from_log] logged_hz={logged_hz:.1f}, stride={stride}, "
              f"effective_video_fps={actual_video_fps:.1f}")

        frames = []
        for i in range(0, len(df), stride):
            row = df.iloc[i]
            q = row[q_cols].to_numpy(dtype=float)
            qd = row[qd_cols].to_numpy(dtype=float) if has_qd else None
            target = row[target_cols].to_numpy(dtype=float) if has_target else None
            frame = self.mujoco_sync_and_render(mj, q, qd, target)
            frames.append(frame)

        print(f"  Rendered {len(frames)} frames from {len(df)} logged steps")
        return frames, actual_video_fps

    def save_video(
        self,
        frames: list,
        out_path: str = "simple_reach_replay.mp4",
        fps: float = 30.0,
    ) -> str:
        """Write frames list to MP4 file."""
        writer = imageio.get_writer(out_path, fps=float(fps))
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        print(f"[save_video] Saved {len(frames)} frames to {out_path}")
        return out_path
