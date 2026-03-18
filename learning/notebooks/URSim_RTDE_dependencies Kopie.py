'''
Create functions an classes for the URSim RTDE control feedback demo.
'''

from typing import Dict, List, Optional
import time
import mujoco
import mujoco.viewer
import socket
import numpy as np
import pandas as pd
import rtde_receive
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imageio.v2 as imageio
import os




class URSimRTDEControlFeedback:
    def __init__(self, host="127.0.0.1", port_rtde=30004, port_urscript=30002, port_dashboard=29999):
        self.host = host
        self.port_rtde = port_rtde
        self.port_urscript = port_urscript
        self.port_dashboard = port_dashboard
        self._receiver: Optional[rtde_receive.RTDEReceiveInterface] = None

    # ==========================================
    # Connection / feedback
    # ==========================================

    def connect(self):
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
        """
        Returns:
            {
                "q": [q1..q6],                # joint positions [rad]
                "qd": [qd1..qd6],            # joint velocities [rad/s]
                "tau": [tau1..tau6],         # estimated joint torques [Nm]
                "tcp_xyz": [x, y, z],        # TCP position in base/world frame [m]
                "tcp_pose": [x,y,z,rx,ry,rz] # full TCP pose
                "tcp_speed": [vx, vy, vz, vrx, vry, vrz] # TCP speed in base/world frame [m/s, rad/s]
                "tcp_speed_3d": tcp_speed_3d     # TCP speed in 3D space (magnitude of linear velocity) [m/s]
            }
        """
        r = self.connect()

        q = r.getActualQ()
        qd = r.getActualQd()
        cur = r.getActualCurrent()
        tcp_pose = r.getActualTCPPose()
        tcp_xyz = tcp_pose[:3]
        tcp_speed = r.getActualTCPSpeed()
        vx, vy, vz = tcp_speed[:3]
        tcp_speed_3d = (vx**2 + vy**2 + vz**2) ** 0.5

        return {
            "q": list(q),
            "qd": list(qd),
            "cur": list(cur),
            "tcp_xyz": list(tcp_xyz),
            "tcp_pose": list(tcp_pose),
            "tcp_speed": list(tcp_speed),
            "tcp_speed_3d": list([tcp_speed_3d])
        }

    def print_feedback(self, digits: int = 4):
        fb = self.receive_feedback()
        print("q      =", [round(v, digits) for v in fb["q"]])
        print("qd     =", [round(v, digits) for v in fb["qd"]])
        print("cur    =", [round(v, digits) for v in fb["cur"]])
        print("tcp_xyz=", [round(v, digits) for v in fb["tcp_xyz"]])
        print("tcp_speed_3d=", [round(v, digits) for v in fb["tcp_speed_3d"]])
    
    # ==========================================
    # Mujoco Render Functions
    # ==========================================
    def render_q_in_mujoco(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer,
        joint_qpos_indices: Optional[List[int]] = None,
        frequency_hz: float = 50.0,
    ):
        """
        Read RTDE joint positions and render them in an existing MuJoCo viewer at fixed frequency.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            viewer: launched mujoco viewer handle
            joint_qpos_indices: indices in data.qpos where the 6 UR joints should be written.
                               If None, uses the first 6 qpos entries.
            frequency_hz: render/update frequency
        """
        if joint_qpos_indices is None:
            joint_qpos_indices = list(range(6))

        if len(joint_qpos_indices) != 6:
            raise ValueError("joint_qpos_indices must contain exactly 6 indices for UR joints.")

        dt = 1.0 / frequency_hz
        r = self.connect()

        next_t = time.perf_counter()

        while viewer.is_running():
            q = r.getActualQ()

            for i, q_idx in enumerate(joint_qpos_indices):
                data.qpos[q_idx] = q[i]

            # optional: also write velocities if your qvel mapping is known
            # qd = r.getActualQd()
            # for i, qv_idx in enumerate(joint_qvel_indices):
            #     data.qvel[qv_idx] = qd[i]

            mujoco.mj_forward(model, data)
            viewer.sync()

            next_t += dt
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def launch_and_render_q_in_mujoco(
        self,
        xml_path: str,
        joint_qpos_indices: Optional[List[int]] = None,
        frequency_hz: float = 50.0,
    ):
        """
        Convenience function:
        - loads model from xml_path
        - creates data
        - launches passive viewer
        - streams RTDE q into MuJoCo viewer at fixed frequency
        """
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            self.render_q_in_mujoco(
                model=model,
                data=data,
                viewer=viewer,
                joint_qpos_indices=joint_qpos_indices,
                frequency_hz=frequency_hz,
            )
    # ==========================================
    # Control helper functions
    # ==========================================
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
        return f"""def {name}():
  {body}
end
{name}()
"""

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

    def joint_error_norm(self, q_target):
        fb = self.receive_feedback()
        q = np.asarray(fb["q"], dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        return np.linalg.norm(q_target - q)

    def reached_joint_target(self, q_target, tol=0.05):
        return self.joint_error_norm(q_target) < tol

    # ==========================================
    # Control ServoJ helper functions
    # ==========================================
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


    # ==========================================
    # Helper function for clean movements in the notebooktest without a policy
    # (e.g. for testing the dashboard connection or the servoj stream)
    # ==========================================

    def step_toward_joint_target(self, q_target, max_step=0.03):
        q_now = np.asarray(self.receive_feedback()["q"], dtype=float)
        q_target = np.asarray(q_target, dtype=float)

        delta = q_target - q_now
        dist = np.linalg.norm(delta)

        if dist < 1e-12:
            return q_target.tolist(), 0.0

        if dist > max_step:
            q_cmd = q_now + (delta / dist) * max_step
        else:
            q_cmd = q_target.copy()

        return q_cmd.tolist(), float(dist)

    # ==========================================
    # MuJoCo helper functions
    # ==========================================
    def mujoco_init_model(
        self,
        xml_path,
        height=480,
        width=640,
        cam_lookat=(0.0, 0.0, 0.35),
        cam_distance=2.0,
        cam_azimuth=130,
        cam_elevation=-20,
        joint_qpos_indices=None,
        joint_qvel_indices=None,
    ):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = list(cam_lookat)
        cam.distance = float(cam_distance)
        cam.azimuth = float(cam_azimuth)
        cam.elevation = float(cam_elevation)

        if joint_qpos_indices is None:
            joint_qpos_indices = [0, 1, 2, 3, 4, 5]
        if joint_qvel_indices is None:
            joint_qvel_indices = [0, 1, 2, 3, 4, 5]

        return {
            "model": model,
            "data": data,
            "renderer": renderer,
            "cam": cam,
            "joint_qpos_indices": list(joint_qpos_indices),
            "joint_qvel_indices": list(joint_qvel_indices),
        }

    def mujoco_sync_from_robot(self, mj, q, qd=None):
        data = mj["data"]

        for i, idx in enumerate(mj["joint_qpos_indices"]):
            data.qpos[idx] = float(q[i])

        if qd is not None:
            for i, idx in enumerate(mj["joint_qvel_indices"]):
                data.qvel[idx] = float(qd[i])

        mujoco.mj_forward(mj["model"], data)

    def mujoco_render(self, mj, title="MuJoCo view", figsize=(8, 6)):
        mj["renderer"].update_scene(mj["data"], camera=mj["cam"])
        img = mj["renderer"].render()

        clear_output(wait=True)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()

    def mujoco_render_video_by_decimation(
        self,
        mj,
        df,
        out_path="mujoco_robot_replay.mp4",
        logged_hz=None,
        video_fps=50.0,
        q_cols=None,
        qd_cols=None,
        verbose=True,
    ):
        if q_cols is None:
            q_cols = [f"q{i}" for i in range(6)]
        if qd_cols is None:
            qd_cols = [f"qd{i}" for i in range(6)]

        if logged_hz is None:
            if "loop_hz_true" not in df.columns:
                raise ValueError("logged_hz not provided and df has no 'loop_hz_true' column.")
            logged_hz = float(df["loop_hz_true"].mean())

        step = max(1, int(round(logged_hz / video_fps)))

        writer = imageio.get_writer(out_path, fps=float(video_fps))
        try:
            for i in range(0, len(df), step):
                q = df.iloc[i][q_cols].to_numpy(dtype=float)
                has_qd = all(c in df.columns for c in qd_cols)
                qd = df.iloc[i][qd_cols].to_numpy(dtype=float) if has_qd else None

                self.mujoco_sync_from_robot(mj, q, qd)
                mj["renderer"].update_scene(mj["data"], camera=mj["cam"])
                img = mj["renderer"].render()
                writer.append_data(img)
        finally:
            writer.close()

        return out_path

    # ====================================================================================
    # Loop functions
    # ====================================================================================
    def run_timed_control_loop_no_render(
        self,
        q_start,
        q_goal,
        policy_fn,
        control_hz=200.0,
        tol=0.02,
        timeout_s=20.0,
        lookahead_time=0.1,
        gain=300,
        pre_movej_a=2.5,
        pre_movej_v=2.0,
        pre_timeout_s=15.0,
        start_reach_tol=0.05,
        textmsg_start="go_start_fast",
        send_t=None,
    ):
        """
        Generic timed experiment loop without MuJoCo rendering.

        policy_fn signature:
            q_cmd, policy_info = policy_fn(q, qd, q_goal, dt)

        where:
            q        : current joint positions, np.ndarray shape (6,)
            qd       : current joint velocities, np.ndarray shape (6,)
            q_goal   : target joint positions, np.ndarray shape (6,)
            dt       : requested loop period

            q_cmd    : commanded joint target, list or np.ndarray shape (6,)
            policy_info : optional dict, can contain extra logged fields, e.g.
                          {"dist": float_value, ...}
        """

        dt = 1.0 / float(control_hz)
        if send_t is None:
            send_t = dt

        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        # ------------------------------------------
        # Fast pre-positioning to start pose
        # ------------------------------------------
        self.send_movej(q_start.tolist(), a=pre_movej_a, v=pre_movej_v, textmsg=textmsg_start)

        pre_start_t = time.perf_counter()
        while True:
            q_now = np.array(self.receive_feedback()["q"], dtype=float)
            err = np.linalg.norm(q_now - np.array(q_start, dtype=float))
            if err < start_reach_tol:
                break
            if time.perf_counter() - pre_start_t > pre_timeout_s:
                print("\nPre-position timeout")
                break
            time.sleep(dt)

        # print("\nReached start pose. Beginning timed loop.")

        # ------------------------------------------
        # Timed loop
        # ------------------------------------------
        log = []
        prev_tcp_xyz = None
        start_time = time.perf_counter()
        step_count = 0
        reached_goal = False
        timed_out = False

        loop_wall_t0 = time.perf_counter()
        while True:
            loop_start = time.perf_counter()

            # ---- RTDE receive timing
            t0_recv = time.perf_counter()
            fb = self.receive_feedback()
            t1_recv = time.perf_counter()

            q = np.asarray(fb["q"], dtype=float)
            qd = np.asarray(fb["qd"], dtype=float)
            tcp_xyz = np.asarray(fb["tcp_xyz"], dtype=float)

            if prev_tcp_xyz is None:
                tcp_delta = np.zeros(3, dtype=float)
                tcp_dist_loop = 0.0
            else:
                tcp_delta = tcp_xyz - prev_tcp_xyz
                tcp_dist_loop = float(np.linalg.norm(tcp_delta))
            prev_tcp_xyz = tcp_xyz.copy()

            # ---- policy timing
            t0_policy = time.perf_counter()
            q_cmd, policy_info = policy_fn(q, qd, q_goal, dt)
            t1_policy = time.perf_counter()

            q_cmd = np.asarray(q_cmd, dtype=float)

            if policy_info is None:
                policy_info = {}

            dist = float(policy_info.get("dist", np.linalg.norm(q_goal - q)))

            # ---- send timing
            t0_send = time.perf_counter()
            self.send_servoj(
                q_cmd.tolist(),
                t=send_t,
                lookahead_time=lookahead_time,
                gain=gain,
            )
            t1_send = time.perf_counter()

            # ---- logging preparation
            elapsed = time.perf_counter() - start_time

            row = {
                "step": step_count,
                "time": elapsed,
                "dist": dist,

                "recv_time_s": t1_recv - t0_recv,
                "policy_time_s": t1_policy - t0_policy,
                "send_time_s": t1_send - t0_send,

                "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3], "q4": q[4], "q5": q[5],
                "qd0": qd[0], "qd1": qd[1], "qd2": qd[2], "qd3": qd[3], "qd4": qd[4], "qd5": qd[5],
                "cmd0": q_cmd[0], "cmd1": q_cmd[1], "cmd2": q_cmd[2], "cmd3": q_cmd[3], "cmd4": q_cmd[4], "cmd5": q_cmd[5],

                "tcp_x": tcp_xyz[0],
                "tcp_y": tcp_xyz[1],
                "tcp_z": tcp_xyz[2],

                "tcp_dx": tcp_delta[0],
                "tcp_dy": tcp_delta[1],
                "tcp_dz": tcp_delta[2],
                "tcp_dist_loop_m": tcp_dist_loop,
            }

            for k, v in policy_info.items():
                if k not in row:
                    row[k] = v

            # ---- stop conditions
            if dist < tol:
                reached_goal = True
            elif elapsed > timeout_s:
                timed_out = True

            # ---- enforce requested period
            elapsed_loop_work = time.perf_counter() - loop_start
            sleep_time = dt - elapsed_loop_work
            if sleep_time > 0:
                time.sleep(sleep_time)

            # ---- true full loop timing AFTER sleep
            loop_end = time.perf_counter()
            loop_dt_true = loop_end - loop_start
            loop_hz_true = 1.0 / loop_dt_true if loop_dt_true > 1e-12 else np.nan
            elapsed = loop_end - start_time

            # ---- stop conditions AFTER full loop
            if dist < tol:
                reached_goal = True
            elif elapsed >= timeout_s:
                timed_out = True

            # ---- log timing
            t0_log = time.perf_counter()
            row["time"] = elapsed
            row["loop_dt_true_s"] = loop_dt_true
            row["loop_hz_true"] = loop_hz_true
            row["tcp_speed_mps"] = tcp_dist_loop / max(loop_dt_true, 1e-9)
            t1_log = time.perf_counter()
            row["log_time_s"] = t1_log - t0_log

            log.append(row)

            if reached_goal or timed_out:
                break

            step_count += 1

        loop_wall_t1 = time.perf_counter()
        total_loop_wall_time_s = loop_wall_t1 - loop_wall_t0
        df = pd.DataFrame(log)

        if len(df) > 0:
            df["tcp_speed_mps"] = df["tcp_dist_loop_m"] / df["loop_dt_true_s"].clip(lower=1e-9)
            df["qd_norm"] = np.sqrt(
                df["qd0"]**2 + df["qd1"]**2 + df["qd2"]**2 +
                df["qd3"]**2 + df["qd4"]**2 + df["qd5"]**2
            )
            df["cmd_err_norm"] = np.sqrt(
                (df["cmd0"] - df["q0"])**2 + (df["cmd1"] - df["q1"])**2 + (df["cmd2"] - df["q2"])**2 +
                (df["cmd3"] - df["q3"])**2 + (df["cmd4"] - df["q4"])**2 + (df["cmd5"] - df["q5"])**2
            )

        stats = self.summarize_timed_loop(
            df,
            control_hz,
            reached_goal,
            timed_out,
            timeout_s,
            total_loop_wall_time_s=total_loop_wall_time_s,
        )
        
        return df, stats

    def summarize_timed_loop(
        self,
        df,
        requested_control_hz,
        reached_goal,
        timed_out,
        timeout_s,
        total_loop_wall_time_s=None,
    ):
        if len(df) == 0:
            return {
                "requested_control_hz": round(requested_control_hz, 1),
                "reached_goal": reached_goal,
                "timed_out": timed_out,
                "timeout_s": float(timeout_s),
                "num_samples": 0,
                "total_loop_wall_time_s": total_loop_wall_time_s,
                "true_inferred_frequency_hz": None,
            }

        num_samples = int(len(df))

        if total_loop_wall_time_s is None or total_loop_wall_time_s <= 0:
            true_inferred_frequency_hz = None
        else:
            true_inferred_frequency_hz = num_samples / float(total_loop_wall_time_s)

        return {
            "requested_control_hz": round(float(requested_control_hz),1),
            "reached_goal": bool(reached_goal),
            "timed_out": bool(timed_out),
            "timeout_s": float(timeout_s),
            
            "num_samples": num_samples,
            "total_time_s": float(df["time"].iloc[-1]),
            "total_loop_wall_time_s": float(total_loop_wall_time_s),
            "true_inferred_frequency_hz": (
                float(true_inferred_frequency_hz)
                if true_inferred_frequency_hz is not None else None
            ),


            "mean_loop_dt_true_s": float(df["loop_dt_true_s"].mean()),
            "std_loop_dt_true_s": float(df["loop_dt_true_s"].std()),
            "mean_loop_hz_true": float(df["loop_hz_true"].mean()),

            "mean_recv_time_s": float(df["recv_time_s"].mean()),
            "mean_policy_time_s": float(df["policy_time_s"].mean()),
            "mean_send_time_s": float(df["send_time_s"].mean()),
            "mean_log_time_s": float(df["log_time_s"].mean()),

            "max_recv_time_s": float(df["recv_time_s"].max()),
            "max_policy_time_s": float(df["policy_time_s"].max()),
            "max_send_time_s": float(df["send_time_s"].max()),
            "max_log_time_s": float(df["log_time_s"].max()),

            "start_dist": float(df["dist"].iloc[0]),
            "final_dist": float(df["dist"].iloc[-1]),

            "mean_tcp_speed_mps": float(df["tcp_speed_mps"].mean()),
            "median_tcp_speed_mps": float(df["tcp_speed_mps"].median()),
            "max_tcp_speed_mps": float(df["tcp_speed_mps"].max()),

            "mean_qd_norm": float(df["qd_norm"].mean()),
            "max_qd_norm": float(df["qd_norm"].max()),
            "mean_cmd_err_norm": float(df["cmd_err_norm"].mean()),
            "max_cmd_err_norm": float(df["cmd_err_norm"].max()),
            "---" : "-------------------------------"
        }

    # ==========================================
    # Wrapper using step_toward_joint_target
    # ==========================================
    def run_step_toward_joint_target_experiment(
        self,
        q_start,
        q_goal,
        control_hz=20.0,
        tol=0.02,
        timeout_s=20.0,
        max_speed=2.0,   # rad/s
        lookahead_time=0.1,
        gain=300,
        pre_movej_a=2.5,
        pre_movej_v=2.0,
        pre_timeout_s=15.0,
        start_reach_tol=0.05,
        send_t=None,
    ):
        dt = 1.0 / float(control_hz)
        max_step = max_speed * dt

        def _policy_fn(q, qd, q_goal_np, dt_inner):
            q_cmd, dist = self.step_toward_joint_target(q_goal_np, max_step=max_step)
            return q_cmd, {"dist": float(dist)}

        return self.run_timed_control_loop_no_render(
            q_start=q_start,
            q_goal=q_goal,
            policy_fn=_policy_fn,
            control_hz=control_hz,
            tol=tol,
            timeout_s=timeout_s,
            lookahead_time=lookahead_time,
            gain=gain,
            pre_movej_a=pre_movej_a,
            pre_movej_v=pre_movej_v,
            pre_timeout_s=pre_timeout_s,
            start_reach_tol=start_reach_tol,
            send_t=send_t,
        )

    def print_timed_loop_stats(self, stats):
        print("Timed loop statistics")
        print("------------------------------")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    def print_stats_keys(self, stats, keys):
        """
        Print only selected keys from the stats dictionary.

        Example:
            self.print_stats_keys(stats, ["mean_loop_hz_true", "mean_tcp_speed_mps"])
        """
        for k in keys:
            if k in stats:
                v = stats[k]
                if isinstance(v, float):
                    print(f"{k}: {v:.6f}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: <missing>")

    def mujoco_render_loop_video_frames(
        self,
        mj,
        df,
        q_cols=None,
        qd_cols=None,
        step_stride=1,
        video_dir="render_video",
        timestamp=None,
        video_fps=50.0,
        save_video=False,
    ):
        """
        Replay logged robot states in MuJoCo and collect rendered frames.

        Notes
        -----
        - This is posthoc replay from the logged dataframe.
        - Actual video saving is intentionally commented out.
        - Prepared output path format:
            render_video/<datetimestamp>.video
        """
        if q_cols is None:
            q_cols = [f"q{i}" for i in range(6)]
        if qd_cols is None:
            qd_cols = [f"qd{i}" for i in range(6)]

        if len(df) == 0:
            raise ValueError("df is empty, nothing to render.")

        missing_q = [c for c in q_cols if c not in df.columns]
        if missing_q:
            raise ValueError(f"Missing q columns in df: {missing_q}")

        has_qd = all(c in df.columns for c in qd_cols)

        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{timestamp}.video")

        frames = []
        frame_times = []

        t0 = df["t_loop_start"].iloc[0]

        for i in range(0, len(df), step_stride):
            row = df.iloc[i]

            q = row[q_cols].to_numpy(dtype=float)
            qd = row[qd_cols].to_numpy(dtype=float) if has_qd else None

            self.mujoco_sync_from_robot(mj, q, qd)
            mj["renderer"].update_scene(mj["data"], camera=mj["cam"])
            img = mj["renderer"].render()

            frames.append(img.copy())
            frame_times.append(row["t_loop_start"] - t0)

        # Intentionally commented out
        # imageio.mimsave(video_path, frames, fps=float(video_fps))

        return frames, video_path


    def mujoco_render_loop_video_frames(
        self,
        mj,
        df,
        q_cols=None,
        qd_cols=None,
        step_stride=1,
        video_dir="render_video",
        timestamp=None,
        video_fps=50.0,
        save_video=False,
    ):
        """
        Replay logged robot states in MuJoCo and collect rendered frames.

        Notes
        -----
        - This is posthoc replay from the logged dataframe.
        - Replay timing is based on df["time"].
        - Frames are resampled to exactly `video_fps`.
        - Actual video saving is intentionally commented out.
        - Prepared output path format:
            render_video/<datetimestamp>.video
        """
        if q_cols is None:
            q_cols = [f"q{i}" for i in range(6)]
        if qd_cols is None:
            qd_cols = [f"qd{i}" for i in range(6)]

        if len(df) == 0:
            raise ValueError("df is empty, nothing to render.")

        if "time" not in df.columns:
            raise ValueError("df must contain a 'time' column for time-based replay.")

        missing_q = [c for c in q_cols if c not in df.columns]
        if missing_q:
            raise ValueError(f"Missing q columns in df: {missing_q}")

        has_qd = all(c in df.columns for c in qd_cols)

        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{timestamp}.video")

        raw_frames = []
        raw_times = []

        t0 = float(df["time"].iloc[0])

        for i in range(0, len(df), step_stride):
            row = df.iloc[i]

            q = row[q_cols].to_numpy(dtype=float)
            qd = row[qd_cols].to_numpy(dtype=float) if has_qd else None

            self.mujoco_sync_from_robot(mj, q, qd)
            mj["renderer"].update_scene(mj["data"], camera=mj["cam"])
            img = mj["renderer"].render()

            raw_frames.append(img.copy())
            raw_times.append(float(row["time"]) - t0)

        if len(raw_frames) == 0:
            raise ValueError("No frames were generated.")

        if len(raw_frames) == 1:
            frames = [raw_frames[0]]
            return frames, video_path

        target_dt = 1.0 / float(video_fps)
        t_max = raw_times[-1]

        frames = []
        t_target = 0.0
        idx = 0

        while t_target <= t_max:
            while idx < len(raw_times) - 1 and raw_times[idx] < t_target:
                idx += 1
            frames.append(raw_frames[idx])
            t_target += target_dt

        # Intentionally commented out
        # imageio.mimsave(video_path, frames, fps=float(video_fps))

        return frames, video_path


    def run_timed_control_loop_then_render(
        self,
        mj,
        q_start,
        q_goal,
        policy_fn,
        control_hz=200.0,
        tol=0.02,
        timeout_s=20.0,
        lookahead_time=0.1,
        gain=300,
        pre_movej_a=2.5,
        pre_movej_v=2.0,
        pre_timeout_s=15.0,
        start_reach_tol=0.05,
        textmsg_start="go_start_fast",
        send_t=None,
        render_video_fps=50.0,
        render_step_stride=1,
        render_q_cols=None,
        render_qd_cols=None,
        video_dir="render_video",
        timestamp=None,
        save_video=False,
    ):
        """
        Run the live control loop without rendering, then replay the logged states in MuJoCo.

        Returns
        -------
        df : pandas.DataFrame
            Logged live-loop data.
        stats : dict
            Summary statistics from the live loop.
        frames : list[np.ndarray]
            Rendered replay frames, resampled in time to `render_video_fps`.
        video_path : str
            Prepared output path. Saving is intentionally commented out.
        """
        df, stats = self.run_timed_control_loop_no_render(
            q_start=q_start,
            q_goal=q_goal,
            policy_fn=policy_fn,
            control_hz=control_hz,
            tol=tol,
            timeout_s=timeout_s,
            lookahead_time=lookahead_time,
            gain=gain,
            pre_movej_a=pre_movej_a,
            pre_movej_v=pre_movej_v,
            pre_timeout_s=pre_timeout_s,
            start_reach_tol=start_reach_tol,
            textmsg_start=textmsg_start,
            send_t=send_t,
        )

        frames, video_path = self.mujoco_render_loop_video_frames(
            mj=mj,
            df=df,
            q_cols=render_q_cols,
            qd_cols=render_qd_cols,
            step_stride=render_step_stride,
            video_dir=video_dir,
            timestamp=timestamp,
            video_fps=render_video_fps,
            save_video=save_video,
        )

        return df, stats, frames, video_path