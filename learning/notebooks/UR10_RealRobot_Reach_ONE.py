"""
Single-target reach from the current robot pose.

Position the robot by hand, then run this script. The policy will
drive the TCP toward TARGET_POS at 50 Hz using servoj — no movej
pre-positioning. A MuJoCo replay video is saved at the end.

Usage:
    python UR10_RealRobot_Reach_ONE.py
"""

import os
import platform
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pandas as pd

from URSim_RTDE_dependencies import URSimRTDESimpleReach

# ===========================================================================
# CONFIGURATION
# ===========================================================================

ROBOT_IP = "192.168.1.2"

# Target: high up, centred — safe direction for any starting pose
# Training workspace: x=[0.3, 0.8], y=[-0.4, 0.4], z=[0.2, 0.8]
TARGET_POS = [0.7, 0.2, 0.7]

# Start pose — "low_home" keyframe from mjx_reach.xml
Q_START = [0, -1.7, 2.25, -2.15, -1.5, -1.5]

# Convergence
REACH_TOL = 0.01       # 1cm
DWELL_TIME_S = 3     # Must stay within REACH_TOL for this many seconds before declaring convergence
TIMEOUT_S = 20.0

# Control
CONTROL_HZ = 200.0        # Loop frequency [Hz]. Must match policy training (50 Hz -> ctrl_dt=0.02)
ACTION_SCALE = 0.04       # Joint delta per step [rad]. Training default=0.04. Lower=slower/safer, higher=faster/shakier
LOOKAHEAD_TIME = 0.02      # servoj smoothing horizon [s]. Range 0.03-0.2. Higher=smoother but laggier
GAIN = 200               # servoj tracking stiffness. Range 100-2000. Lower=softer/smoother, higher=stiffer/jerkier
SERVOJ_A = 0.5          # Max joint acceleration [rad/s^2] enforced by UR controller. UR10e max ~2.5. Conservative=0.5-1.0, Med=1.4, Fast=2.5
SERVOJ_V = 2         # Max joint velocity [rad/s] enforced by UR controller. UR10e max ~2.1. Conservative=0.5, Med=1.05, Fast=2.1
ALPHA = 1             # Blend factor: 1.0=full policy, <1.0=blend with measured q (smoother but slower)

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
FOLDER_OUT = os.path.join(SCRIPT_DIR, "results")
VIDEO_OUT = os.path.join(FOLDER_OUT, "reach_one_replay.mp4")
PLOTS_OUT = os.path.join(FOLDER_OUT, "reach_one_plots.png")
META_OUT = os.path.join(FOLDER_OUT, "reach_one_meta.json")
VIDEO_FPS = 50.0

# ===========================================================================
# MAIN
# ===========================================================================

os.makedirs(FOLDER_OUT, exist_ok=True)

# ── Connect ────────────────────────────────────────────────────────────
print(f"Connecting to {ROBOT_IP} ...")
robot = URSimRTDESimpleReach(host=ROBOT_IP)
robot.connect()
print("Current pose:")
robot.print_feedback()

# ── Move to start ─────────────────────────────────────────────────────
print(f"\nMoving to start pose {Q_START}")
robot.move_to_start(Q_START, a=1.5, v=0.8, timeout_s=15.0, tol=0.01)

# ── Load policy ────────────────────────────────────────────────────────
robot.load_policy_fn(policy_path=POLICY_PATH, deterministic=True)

target = np.array(TARGET_POS, dtype=np.float32)
fb = robot.receive_feedback()
tcp_now = np.array(fb["tcp_xyz"])
dist = np.linalg.norm(target - tcp_now)
print(f"\nTarget : {target.tolist()}")
print(f"TCP now: {np.round(tcp_now, 4).tolist()}")
print(f"Distance: {dist * 1000:.1f} mm")
print(f"\nStarting policy loop at {CONTROL_HZ} Hz ...")

# ── Run policy loop ───────────────────────────────────────────────────
df, stats = robot.run_policy_loop(
    target_pos=target,
    control_hz=CONTROL_HZ,
    timeout_s=TIMEOUT_S,
    action_scale=ACTION_SCALE,
    lookahead_time=LOOKAHEAD_TIME,
    gain=GAIN,
    servoj_a=SERVOJ_A,
    servoj_v=SERVOJ_V,
    alpha=ALPHA,
    reach_tol=REACH_TOL,
    dwell_time_s=DWELL_TIME_S,
)

# ── Results ───────────────────────────────────────────────────────────
final_dist = stats["final_tcp_to_target_dist"]
print(f"\nFinal distance: {final_dist * 1000:.1f} mm — "
        f"{'REACHED' if final_dist < REACH_TOL else 'NOT REACHED'}")
print(f"Steps: {len(df)}, wall time: {stats['total_wall_time_s']:.2f}s")
robot.print_stats(stats, keys=[
    "true_inferred_frequency_hz",
    "mean_loop_hz_true",
    "start_tcp_to_target_dist",
    "final_tcp_to_target_dist",
    "min_tcp_to_target_dist",
    "net_improvement",
    "mean_tcp_speed_mps",
    "num_overruns",
])

# ── Render video ──────────────────────────────────────────────────────
print(f"\nRendering MuJoCo replay ...")
mj = robot.mujoco_init_model(
    xml_path=MODEL_PATH,
    height=480, width=640,
    cam_lookat=(0.4, 0.0, 0.4),
    cam_distance=1.8, cam_azimuth=130, cam_elevation=-20,
)
frames, actual_fps = robot.render_video_from_log(mj, df, video_fps=VIDEO_FPS)
robot.save_video(frames, out_path=VIDEO_OUT, fps=actual_fps)
print(f"Video: {VIDEO_OUT}")

# ── Save plots ───────────────────────────────────────────────────────
robot.save_plots(df, out_path=PLOTS_OUT)

# ── Save run metadata ────────────────────────────────────────────────
robot.save_run_metadata(
    META_OUT,
    robot_ip=ROBOT_IP,
    target_pos=TARGET_POS,
    q_start=Q_START,
    reach_tol=REACH_TOL,
    timeout_s=TIMEOUT_S,
    control_hz=CONTROL_HZ,
    action_scale=ACTION_SCALE,
    lookahead_time=LOOKAHEAD_TIME,
    gain=GAIN,
    servoj_a=SERVOJ_A,
    servoj_v=SERVOJ_V,
    alpha=ALPHA,
    policy_path=POLICY_PATH,
    model_path=MODEL_PATH,
    stats=stats,
)

# ── Disconnect ────────────────────────────────────────────────────────
robot.disconnect()
print("Done.")

