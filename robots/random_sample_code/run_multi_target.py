"""
Multi-target reach script for the UR10e with a trained PPO policy.

Moves the robot through a list of TCP target coordinates using the
policy control loop at 50 Hz. Once the TCP is within 5 mm of a target
the script advances to the next one. At the end a MuJoCo replay video
is saved.

Usage:
    python run_multi_target.py
"""

import os
import platform
import sys
import time

# ── Backend setup ──────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import numpy as np
import pandas as pd

# URSimRTDESimpleReach lives in robots/URSim/ (this script is in robots/random_sample_code/).
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "URSim"),
)
from URSim_RTDE_dependencies import URSimRTDESimpleReach

# ===========================================================================
# CONFIGURATION — edit these before running
# ===========================================================================

ROBOT_IP = "192.168.1.2"

# TCP target positions in world frame (meters).
# Training workspace: x=[0.3, 0.8], y=[-0.4, 0.4], z=[0.2, 0.8]
TARGETS = [
    [0.5,  0.0, 0.5],
    [0.6,  0.2, 0.4],
    [0.4, -0.2, 0.6],
    [0.5,  0.0, 0.5],
]

# Start joint pose — "low_home" keyframe from mjx_reach.xml
Q_START = [0, -1.7, 2.25, -2.15, -1.5, -1.5]

# Convergence tolerance (meters)
REACH_TOL = 0.005  # 5 mm

# Control loop settings (must match the 50 Hz policy training)
CONTROL_HZ = 50.0
ACTION_SCALE = 0.04
TIMEOUT_PER_TARGET_S = 10.0

# servoj parameters
LOOKAHEAD_TIME = 0.1
GAIN = 300

# Paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "../../mujoco_playground/_src/manipulation/my_ur10/xmls/mjx_reach.xml",
)
POLICY_PATH = os.path.join(
    SCRIPT_DIR,
    "../../evaluation/downloaded_policies/simple_reach_policy_50hz",
)

# Video output
VIDEO_OUT = os.path.join(SCRIPT_DIR, "multi_target_replay.mp4")
VIDEO_FPS = 30.0

# ===========================================================================
# MAIN
# ===========================================================================


def main():
    # ── Connect ────────────────────────────────────────────────────────────
    print(f"Connecting to robot at {ROBOT_IP} ...")
    robot = URSimRTDESimpleReach(host=ROBOT_IP)
    robot.connect()
    robot.print_feedback()

    # ── Load policy ────────────────────────────────────────────────────────
    print(f"\nLoading policy from {POLICY_PATH}")
    robot.load_policy_fn(policy_path=POLICY_PATH, deterministic=True)

    # ── JIT warmup ─────────────────────────────────────────────────────────
    dummy = np.zeros((1, 18), dtype=np.float32)
    t0 = time.perf_counter()
    _ = robot._policy_fn(dummy)
    _.block_until_ready()
    print(f"JIT warmup: {time.perf_counter() - t0:.2f}s")

    # ── Move to start ─────────────────────────────────────────────────────
    print(f"\nMoving to start pose {Q_START}")
    robot.move_to_start(Q_START, a=1.5, v=1.0, timeout_s=15.0, tol=0.01)

    # ── Loop through targets ──────────────────────────────────────────────
    all_dfs = []
    all_stats = []

    for idx, target in enumerate(TARGETS):
        target = np.array(target, dtype=np.float32)
        print(f"\n{'='*60}")
        print(f"  Target {idx + 1}/{len(TARGETS)}: {target.tolist()}")
        print(f"{'='*60}")

        df, stats = run_until_reached(
            robot,
            target_pos=target,
            reach_tol=REACH_TOL,
            control_hz=CONTROL_HZ,
            timeout_s=TIMEOUT_PER_TARGET_S,
            action_scale=ACTION_SCALE,
            lookahead_time=LOOKAHEAD_TIME,
            gain=GAIN,
        )

        all_dfs.append(df)
        all_stats.append(stats)

        final_dist = stats["final_tcp_to_target_dist"]
        reached = final_dist < REACH_TOL
        print(f"\n  Final distance: {final_dist * 1000:.1f} mm — {'REACHED' if reached else 'TIMEOUT'}")

        # Return to start before next target
        if idx < len(TARGETS) - 1:
            print("  Returning to start pose...")
            robot.move_to_start(Q_START, a=2.0, v=1.5, timeout_s=10.0, tol=0.01)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for idx, (target, stats) in enumerate(zip(TARGETS, all_stats)):
        d = stats["final_tcp_to_target_dist"]
        ok = "OK" if d < REACH_TOL else "MISS"
        print(f"  Target {idx + 1} {target} — {d * 1000:.1f} mm [{ok}]")

    # ── Combine logs and render video ─────────────────────────────────────
    df_all = combine_logs(all_dfs)
    print(f"\nTotal logged steps: {len(df_all)}")

    print(f"\nRendering MuJoCo replay video...")
    mj = robot.mujoco_init_model(
        xml_path=MODEL_PATH,
        height=480,
        width=640,
        cam_lookat=(0.4, 0.0, 0.4),
        cam_distance=1.8,
        cam_azimuth=130,
        cam_elevation=-20,
    )
    frames, actual_fps = robot.render_video_from_log(mj, df_all, video_fps=VIDEO_FPS)
    robot.save_video(frames, out_path=VIDEO_OUT, fps=actual_fps)
    print(f"Video saved to {VIDEO_OUT}")

    # ── Disconnect ────────────────────────────────────────────────────────
    robot.disconnect()
    print("\nDone.")


def run_until_reached(
    robot: URSimRTDESimpleReach,
    target_pos: np.ndarray,
    reach_tol: float,
    control_hz: float,
    timeout_s: float,
    action_scale: float,
    lookahead_time: float,
    gain: int,
    dtype=np.float32,
) -> tuple:
    """Run the policy loop, but stop early when TCP is within reach_tol of target."""

    if robot._policy_fn is None:
        raise RuntimeError("Policy not loaded.")

    dt = 1.0 / float(control_hz)
    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)

    # Initialize ctrl_state from current joint positions
    fb0 = robot.receive_feedback()
    robot._ctrl_state = np.array(fb0["q"], dtype=dtype)

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
        fb = robot.receive_feedback()
        t1_obs = time.perf_counter()

        q = np.asarray(fb["q"], dtype=float)
        qd = np.asarray(fb["qd"], dtype=float)
        tcp_xyz = np.asarray(fb["tcp_xyz"], dtype=float)

        # TCP motion tracking
        if prev_tcp_xyz is None:
            tcp_delta = np.zeros(3)
            tcp_dist_loop = 0.0
        else:
            tcp_delta = tcp_xyz - prev_tcp_xyz
            tcp_dist_loop = float(np.linalg.norm(tcp_delta))
        prev_tcp_xyz = tcp_xyz.copy()

        # Distance to target
        tcp_to_target_dist = float(np.linalg.norm(target_pos - tcp_xyz))
        if prev_tcp_to_target_dist is None:
            tcp_to_target_improvement = 0.0
        else:
            tcp_to_target_improvement = prev_tcp_to_target_dist - tcp_to_target_dist
        prev_tcp_to_target_dist = tcp_to_target_dist

        # 2. Build observation
        obs_batch = robot.build_obs_from_feedback(fb, target_pos, dtype)

        # 3. Policy inference
        t0_policy = time.perf_counter()
        action_raw = robot._policy_fn(obs_batch)
        action = np.asarray(action_raw, dtype=dtype).reshape(-1)
        t1_policy = time.perf_counter()

        # 4. Control update
        ctrl_prev = robot._ctrl_state.copy()
        ctrl_next = robot.policy_step_ctrl_update(action, action_scale, dtype)

        # 5. Send servoj
        t0_send = time.perf_counter()
        robot.send_servoj(
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
        loop_hz_true = 1.0 / loop_dt_true if loop_dt_true > 1e-12 else float("nan")
        elapsed = loop_end - start_time

        # 7. Log
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
            "qd_norm": float(np.linalg.norm(qd)),
            "cmd_err_norm": float(np.linalg.norm(ctrl_next - q)),
            "ctrl_delta_norm": ctrl_delta_norm,
            "obs_time_s": t1_obs - t0_obs,
            "policy_time_s": t1_policy - t0_policy,
            "send_time_s": t1_send - t0_send,
            "loop_dt_true_s": loop_dt_true,
            "loop_hz_true": loop_hz_true,
            "overrun_count": overrun_count,
        }
        log.append(row)

        print(
            f"\r  [{step_count:4d}] tcp->tgt={tcp_to_target_dist * 1000:.1f}mm "
            f"| hz={loop_hz_true:.1f} "
            f"| tcp={np.round(tcp_xyz, 3)}",
            end="", flush=True,
        )

        # Early stop if target reached
        if tcp_to_target_dist < reach_tol:
            print(f"\n  Target reached at step {step_count}!")
            break

        if elapsed >= timeout_s:
            print(f"\n  Timeout after {elapsed:.1f}s")
            break

        step_count += 1

    total_wall = time.perf_counter() - start_time
    df = pd.DataFrame(log)
    stats = robot.summarize_policy_loop(df, control_hz, timeout_s, total_wall)
    return df, stats


def combine_logs(dfs: list) -> pd.DataFrame:
    """Concatenate per-target DataFrames with continuous time."""
    combined = []
    time_offset = 0.0
    for df in dfs:
        if len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy["time"] = df_copy["time"] + time_offset
        combined.append(df_copy)
        time_offset = df_copy["time"].iloc[-1] + 0.02  # small gap between targets
    if not combined:
        return pd.DataFrame()
    result = pd.concat(combined, ignore_index=True)
    result["step"] = range(len(result))
    return result


if __name__ == "__main__":
    main()
