"""
UR3 + Hand-E pick loop with a mocap-derived box position.

Mirror of UR10_RealRobot_Reach_ONE.py, adapted for the UR3PicknPlace policy:
  - 26D obs [q(8), qd(6), (box-tcp)(3), (target-box)(3), box_xmat[:6](6)]
  - 7D action (6 arm + 1 gripper)
  - the box pose (xyz + orientation) is streamed live from a Nokov rigid body;
    you only place the box, the loop reads its pose every tick
  - the lift target is hardcoded below

Position the robot, place the tracked rigid body in the cameras' view, then run.
A MuJoCo replay video + diagnostic plots + run metadata are written to
real_robot_results/ (git-tracked).

Usage:
    python ur3_realrobot_pickloop.py
For a URSim dry-run set ROBOT_IP=127.0.0.1 and ENABLE_GRIPPER=False.
"""

import os
import platform
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

from ur3_realrobot_dependencies import UR3RealRobotPick

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../evaluation/downloaded_policies"))
from motion_capture.mymocap.vrpn_dependencies import VRPNRigidBodyReader
from robots.hande.HandE_dependency import HandEGripper
from policy_downloader import default_policy_dir, download_policy

# ===========================================================================
# CONFIGURATION
# ===========================================================================

ROBOT_IP = "192.168.1.4"          # real UR3e (PolyScope X); URSim = "127.0.0.1"

# Arm control path. On PolyScope X the headless script ur_rtde normally uploads
# does NOT run; set USE_EXT_URCAP=True to drive the arm via the External Control
# URCapX (port UR_CAP_PORT). robot.connect() then BLOCKS until an External Control
# program (Host IP = this PC, port = UR_CAP_PORT) is PLAYING on the pendant with the
# robot in Remote Control — i.e. pressing Play on the pendant is the robot-side
# "go" gate. Leave False for the URSim dry-run (script-upload path, no gate).
USE_EXT_URCAP = True
UR_CAP_PORT = 50002

# Drop target (where to bring the box). UR3-reachable workspace.
DROP_TARGET = [0.3, -0.20, 0.30]

# Start pose — UR3 "task_home" keyframe arm angles from mjx_single_cube_position_ur3.xml
# (gripper is opened separately below, since servoJ/moveJ only covers the 6 arm joints)
Q_START = [0, -2.0, 1.6, -1.6, -1.5, 0]

# Motion capture
MOCAP_SERVER_IP = "10.1.1.198"
MOCAP_RIGID_BODY_NAME = "CubeInCube2"  # streamed VRPN tracker name
MOCAP_RIGID_BODY_ID = None        # kept for run-metadata only (VRPN reads by name)
# Rigid-body loss guard: if the tracked body has not updated for this many
# seconds (it left the cameras' view), STOP the loop and report. The mocap
# streams ~60 Hz, so 0.25 s ~= 15 dropped frames — long enough to ride out a
# single missed frame, short enough to halt fast. The loop subscribes to ONLY
# this body so its last-update time is not kept fresh by other trackers.
MOCAP_STALE_S = 0.25

# Gripper: True drives the real Hand-E via the HandEGripper wrapper (Robotiq URCapX
# XML-RPC, PolyScope X); False = no-op (URSim dry-run / arm-only test).
ENABLE_GRIPPER = True
GRIPPER_PORT = 49999              # Robotiq URCapX XML-RPC server (PolyScope X)
GRIPPER_SLAVE_ID = 9             # this Hand-E = slaveId 9 ("Gripper ID 1")
GRIPPER_SPEED_PCT = 30           # low = gentle physical gripper motion (see below)
GRIPPER_FORCE_PCT = 50

# Convergence
REACH_TOL = 0.03                  # 3 cm (box-to-target)
DWELL_TIME_S = 2.0
TIMEOUT_S = 10.0

# Control (must match training: 50 Hz -> ctrl_dt=0.02, action_scale=0.04)
CONTROL_HZ = 50.0
ACTION_SCALE = 0.04              # MUST match training (UR3Pick default 0.04)
LOOKAHEAD_TIME = 0.1              # servoj smoothing [0.03, 0.2]
GAIN = 300                        # servoj stiffness [100, 2000]
SERVOJ_A = 0.2                    # max joint accel [rad/s^2]
SERVOJ_V = 0.2                    # max joint vel  [rad/s]
ALPHA = 0.5                      # 1.0 = send the policy's full action (no blend; matches training)
USE_FK_TCP = True                 # compute tcp_pos via MuJoCo FK (matches sim site)
# Gripper smoothing disabled: the raw integrator target goes straight to the
# hardware/obs. Physical smoothing now comes from the low GRIPPER_SPEED_PCT above,
# not a software low-pass (which double-lagged the real Hand-E's own dynamics).
GRIPPER_TAU = 0.0                # s; 0 = no finger-plant low-pass
GRIPPER_MAX_RATE = float("inf")  # m/s; inf = no slew cap

# Paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This policy trains on UR3Pick (env_name in metadata.json), so render/FK use the
# UR3Pick scene — NOT the pick-and-place XML (different box size + target).
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "../../mujoco_playground/_src/manipulation/my_ur3/xmls/"
    "mjx_single_cube_position_ur3.xml",
)
# Policy registry: friendly name -> W&B run id. Add a trained run here, then
# point POLICY_NAME at it. Values are bare run ids; a policy trained to a
# different W&B project (e.g. pick-and-place -> "UR3_PicknDrop") also needs
# WANDB_PROJECT below adjusted. The selected policy is loaded from
# evaluation/downloaded_policies/{run_id}/ if present, else downloaded from W&B.
POLICY_REGISTRY = {
    "old_version_pick": "Pick_12M_env1024_20260625_193041_6311",
    "base90_lr4e-10": "base90_j10_fin25_20M_lr4e-4_20260626_101918_6311",
    "reasonable starting positions": "Reso_Pos_lr6e-4_20260626_110022_7585",
    "pick_12M_rand_base85_fin25": "Pick_12M_rand_base85_fin25_20260626_094554_6311",
    "pick_un20_env2048":          "Pick_un20_env2048_20260624_160023_6311",
    "pick_dr_medium":             "ur3pick_DR_MFR_medium_20260603_092056_5305",
}
POLICY_NAME = "reasonable starting positions"  # <- select which policy to run

WANDB_ENTITY = "weissma6-zhaw-school-of-engineering"
WANDB_PROJECT = "UR3_pick_ppo"
WANDB_RUN_ID = POLICY_REGISTRY[POLICY_NAME]
# Local cache is run-id keyed: evaluation/downloaded_policies/{run_id}/
POLICY_PATH = default_policy_dir(WANDB_RUN_ID)
# real_robot_results/ (NOT "results/", which is git-ignored) so the run outputs
# are tracked and uploaded.
FOLDER_OUT = os.path.join(SCRIPT_DIR, "real_robot_results")
VIDEO_OUT = os.path.join(FOLDER_OUT, "ur3_pick_replay.mp4")
CSV_OUT = os.path.join(FOLDER_OUT, "ur3_pick_states.csv")
ROBOT_PLOTS_OUT = os.path.join(FOLDER_OUT, "ur3_pick_robot_plots.png")
GRIPPER_PLOTS_OUT = os.path.join(FOLDER_OUT, "ur3_pick_gripper_plots.png")
TIMING_OUT = os.path.join(FOLDER_OUT, "ur3_pick_timing.png")
META_OUT = os.path.join(FOLDER_OUT, "ur3_pick_meta.json")
VIDEO_FPS = 50.0

# ===========================================================================
# MAIN
# ===========================================================================

os.makedirs(FOLDER_OUT, exist_ok=True)

# ── Connect robot ──────────────────────────────────────────────────────
print(f"Connecting to robot {ROBOT_IP} ...")
robot = UR3RealRobotPick(
    host=ROBOT_IP, use_ext_urcap=USE_EXT_URCAP, ur_cap_port=UR_CAP_PORT,
)
if USE_EXT_URCAP:
    print(f"  Waiting for the go from the robot: press Play on the pendant "
          f"(External Control → Host IP = this PC, port {UR_CAP_PORT}, "
          f"Remote Control). connect() blocks until it is PLAYING ...")
robot.connect()
if not robot.is_connected():
    raise RuntimeError(
        "RTDE failed to connect. Check robot IP and that PolyScope is in "
        "Remote Control mode."
    )
robot.print_feedback()

# ── Connect mocap (VRPN) ───────────────────────────────────────────────
# Subscribe to ONLY the target body (names=[...]) so its last-update time is the
# loss signal — with the default "stream all", another tracker would keep the
# timestamp fresh and mask the box leaving view.
print(f"\nConnecting to mocap {MOCAP_SERVER_IP} (rigid body {MOCAP_RIGID_BODY_NAME}) ...")
mocap = VRPNRigidBodyReader(
    MOCAP_SERVER_IP,
    rigid_body_name=MOCAP_RIGID_BODY_NAME,
    names=[MOCAP_RIGID_BODY_NAME],
)
if not mocap.start(timeout=5.0):
    raise RuntimeError(
        "Mocap failed to connect. Check the server IP, that 'SDK Enabled' is "
        "on, and that the rigid body is defined and visible to the cameras."
    )
if not mocap.wait_for_data(timeout=5.0):
    raise RuntimeError(
        f"Mocap subscribed but rigid body '{MOCAP_RIGID_BODY_NAME}' never "
        "reported. Check the name (case-sensitive) and that it is visible to "
        "the cameras."
    )
box0 = mocap.get_rigid_body_xyz()
print(f"Initial box (mocap): {None if box0 is None else np.round(box0, 4).tolist()}")

# ── Gripper wiring ─────────────────────────────────────────────────────
# run_policy_loop passes gripper_norm in [0,1] (0=open, 1=closed); the policy's
# tendon-ctrl span [0,0.05] maps to per-finger [0,0.025] m, so norm*0.025 is the
# sim finger value the wrapper expects. ENABLE_GRIPPER=False keeps the dry-run no-op.
gripper = None
if ENABLE_GRIPPER:
    gripper = HandEGripper(
        ROBOT_IP, port=GRIPPER_PORT, slave_id=GRIPPER_SLAVE_ID,
        speed_pct=GRIPPER_SPEED_PCT, force_pct=GRIPPER_FORCE_PCT,
    )
    gripper.connect()
    gripper.open_gripper()  # start the task with fingers open
    gripper_fn = lambda norm: gripper.command(norm * 0.025)  # noqa: E731
    # Read the real finger position back (diagnostic only — see run_policy_loop);
    # polled at the same <=10 Hz cadence as the command, logged as gripper_fb_pos
    # (sim_finger metres) + gripper_fb_pct (raw native percent). Return the full
    # state dict so no readback field (pos_pct, sim_finger, ...) is lost.
    gripper_state_fn = lambda: gripper.read_state()  # noqa: E731
else:
    gripper_fn = lambda norm: None  # noqa: E731
    gripper_state_fn = None

target = np.array(DROP_TARGET, dtype=np.float32)
print(f"\nDrop target : {target.tolist()}")
print(f"Starting pick loop at {CONTROL_HZ} Hz ...")

# ── Move to start ──────────────────────────────────────────────────────
print(f"\nMoving to start pose {Q_START}")
robot.send_movej(Q_START, a=1.0, v=0.5, asynchronous=False)  # blocking moveJ

# ── Download policy from W&B if missing (cache-aware, run-id keyed) ─────
download_policy(
    WANDB_RUN_ID, out_dir=POLICY_PATH,
    entity=WANDB_ENTITY, project=WANDB_PROJECT,
)

# ── Load policy + FK model ─────────────────────────────────────────────
robot.load_policy_fn(policy_path=POLICY_PATH, deterministic=True)
if USE_FK_TCP:
    robot.init_fk_model(MODEL_PATH)



# ── Run policy loop ────────────────────────────────────────────────────
df, stats = robot.run_policy_loop(
    drop_target=target,
    mocap_reader=mocap,
    control_hz=CONTROL_HZ,
    timeout_s=TIMEOUT_S,
    action_scale=ACTION_SCALE,
    lookahead_time=LOOKAHEAD_TIME,
    gain=GAIN,
    servoj_a=SERVOJ_A,
    servoj_v=SERVOJ_V,
    alpha=ALPHA,
    gripper_fn=gripper_fn,
    gripper_state_fn=gripper_state_fn,
    gripper_tau=GRIPPER_TAU,
    gripper_max_rate=GRIPPER_MAX_RATE,
    use_fk_tcp=USE_FK_TCP,
    reach_tol=REACH_TOL,
    dwell_time_s=DWELL_TIME_S,
    mocap_stale_s=MOCAP_STALE_S,
)

# ── Results ────────────────────────────────────────────────────────────
stop_reason = stats.get("stopped_reason", "completed")
final_dist = stats.get("final_box_to_target_dist", float("nan"))
print(f"\nStopped: {stop_reason}")
print(f"Final box->target: {final_dist * 1000:.1f} mm — "
      f"{'REACHED' if final_dist < REACH_TOL else 'NOT REACHED'}")
print(f"Steps: {len(df)}, wall time: {stats.get('total_wall_time_s', 0.0):.2f}s")

# ── Success + release ──────────────────────────────────────────────────
# Only treat as success if the loop actually converged — a mocap-loss stop with
# the box happening to be near target is NOT a real reach.
reached = stop_reason != "mocap_lost" and final_dist < REACH_TOL
if reached:
    print("\nSUCCESS")
    time.sleep(3.0)
    if gripper is not None:
        gripper.open_gripper()   # direction-independent URCapX open
    else:
        gripper_fn(0.0)          # dry-run no-op / norm=0 = open
    print("Gripper opened.")
    print(f"Returning to task_home {Q_START}")
    robot.send_movej(Q_START, a=1.0, v=0.5, asynchronous=False)  # blocking moveJ

robot.print_stats(stats, keys=[
    "stopped_reason",
    "true_inferred_frequency_hz",
    "mean_loop_hz_true",
    "mean_recv_time_s",
    "mean_mocap_time_s",
    "mean_policy_time_s",
    "mean_send_time_s",
    "start_box_to_target_dist",
    "final_box_to_target_dist",
    "min_box_to_target_dist",
    "net_improvement",
    "num_overruns",
])

if len(df) == 0:
    # Loop stopped before logging any step (e.g. mocap lost immediately).
    print("\nNo steps logged — skipping video/plots. Check mocap + robot.")
    mocap.stop()
    if gripper is not None:
        gripper.disconnect()
    robot.disconnect()
    print("Done.")
    sys.exit(0)

# ── Render video ───────────────────────────────────────────────────────
print("\nRendering MuJoCo replay ...")
mj = robot.mujoco_init_model(
    xml_path=MODEL_PATH, height=480, width=640,
    cam_lookat=(0.3, 0.0, 0.3), cam_distance=1.2,
    cam_azimuth=130, cam_elevation=-20,
)
frames, actual_fps = robot.render_video_from_log(mj, df, video_fps=VIDEO_FPS)
robot.save_video(frames, out_path=VIDEO_OUT, fps=actual_fps)
print(f"Video: {VIDEO_OUT}")

# ── Save full per-step state + plots + metadata ────────────────────────
df.to_csv(CSV_OUT, index=False)
print(f"States: {CSV_OUT}")
# Two split diagnostic figures: the arm control pipeline and the gripper control
# pipeline (measured -> obs -> action -> ctrl stages -> command), so the gripper
# oscillation is visible per signal.
robot.save_robot_plots(df, out_path=ROBOT_PLOTS_OUT)
print(f"Robot plots: {ROBOT_PLOTS_OUT}")
robot.save_gripper_plots(df, out_path=GRIPPER_PLOTS_OUT)
print(f"Gripper plots: {GRIPPER_PLOTS_OUT}")
# Per-loop-step timing bar graph (receive / mocap / obs / inference / ctrl /
# send) — where each control tick's compute time goes (the pacing sleep is
# excluded from the bar chart; still in the CSV).
robot.save_timing_breakdown(df, out_path=TIMING_OUT)
print(f"Timing: {TIMING_OUT}")
robot.save_run_metadata(
    META_OUT,
    robot_ip=ROBOT_IP,
    drop_target=DROP_TARGET,
    q_start=Q_START,
    mocap_server_ip=MOCAP_SERVER_IP,
    mocap_rigid_body_name=MOCAP_RIGID_BODY_NAME,
    mocap_rigid_body_id=MOCAP_RIGID_BODY_ID,
    mocap_stale_s=MOCAP_STALE_S,
    enable_gripper=ENABLE_GRIPPER,
    policy_name=POLICY_NAME,
    reach_tol=REACH_TOL,
    timeout_s=TIMEOUT_S,
    control_hz=CONTROL_HZ,
    action_scale=ACTION_SCALE,
    lookahead_time=LOOKAHEAD_TIME,
    gain=GAIN,
    servoj_a=SERVOJ_A,
    servoj_v=SERVOJ_V,
    alpha=ALPHA,
    gripper_tau=GRIPPER_TAU,
    gripper_max_rate=GRIPPER_MAX_RATE,
    use_fk_tcp=USE_FK_TCP,
    policy_path=POLICY_PATH,
    model_path=MODEL_PATH,
    stats=stats,
)

# ── Disconnect ─────────────────────────────────────────────────────────
mocap.stop()
if gripper is not None:
    gripper.disconnect()
robot.disconnect()
print("Done.")
