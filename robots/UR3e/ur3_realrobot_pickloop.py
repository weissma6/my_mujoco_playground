"""
UR3 + Hand-E pick loop with a mocap-derived box position.

Mirror of UR10_RealRobot_Reach_ONE.py, adapted for the UR3PicknPlace policy:
  - 13D obs [arm_q(6), gripper(1), (box-tcp)(3), (target-box)(3)]
  - 7D action (6 arm + 1 gripper)
  - the box xyz is streamed live from a Nokov rigid body (orientation no longer
    used); you only place the box, the loop reads its position every tick
  - the lift target is hardcoded below

Position the robot, place the tracked rigid body in the cameras' view, then run.
A MuJoCo replay video + diagnostic plots + run metadata are written to
real_robot_results/ (git-tracked).

Usage:
    python ur3_realrobot_pickloop.py
For a URSim dry-run set ROBOT_IP=127.0.0.1 and ENABLE_GRIPPER=False.
"""

import json
import os
import platform
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

from ur3_realrobot_dependencies import (
    UR3RealRobotPick,
    print_action_scale_banner,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../evaluation/downloaded_policies"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../evaluation"))
from motion_capture.mymocap.vrpn_dependencies import VRPNRigidBodyReader
from robots.hande.HandE_dependency import HandEGripper
from policy_downloader import default_policy_dir, download_policy
from ur3_reward_replay import replay_dataframe, save_reward_terms_plot

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

# Lift target — the SAME fixed air-point UR3Pick trains on, NOT a place location.
# Training anchors the target to the fixed keyframe box pos self._init_obj_pos
# (task_home box = (0.4, 0, 0.115)) plus uniform([0.02,-0.03,0.18],[0.06,0.03,0.21]);
# the midpoint => (0.44, 0.0, 0.310) in the base/sim frame. It is a STATIC point
# (does not track the live box) and the policy lifts the box straight up into it.
# The anchor's Z is 0.115 (not 0.02) since the "lifter" body became the real lab
# TABLE: the cube rests on the table's 95 mm top surface + its own 20 mm half-
# height. Training adds only the table's per-episode deviation from that nominal
# on top, which is 0 for the real (fixed) table — so the old "no lifter on the
# real robot, so no lifter-height Z add" note is obsolete; the height is now in
# the anchor. See evaluation/gap_target.py.
# WARNING — reach: 0.44 m out at z=0.310 is ~0.538 m from the base, i.e. AT the
# UR3e's usable limit before the grasp consumes any of it. If the arm stalls
# short of the target, pull the X in (~0.38) rather than lowering Z.
# The old far-sideways "place" target [0.3,-0.20,0.30] was out-of-distribution
# and made the policy release mid-grasp (see diagnosis) — drop/place is disabled
# below; lift-and-hold.
DROP_TARGET = [0.44, 0.0, 0.310]

# Diagnostic offset (m) added to the mocap-derived box Z every tick, BEFORE
# it feeds the obs/gripper-align reward-replay/render -- does NOT move the
# physical box. Use this to nudge the policy's belief of the box height up
# (e.g. +0.015 = 15 mm) to visually check whether it would still
# attempt/complete a grasp at that offset, without touching the setup.
# 0.0 = disabled (raw mocap height, matches training exactly).
BOX_Z_OFFSET = 0

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
GRIPPER_SPEED_PCT = 60           # 30-60 = gentle finger motion; the value the first success ran
GRIPPER_FORCE_PCT = 80           # 50-80; a Robotiq move ends on the force limit, so keep force high
                                 # so the fingers clamp the cube instead of stalling short of it

# Convergence
REACH_TOL = 0.02                  # 2 cm (box-to-target)
DWELL_TIME_S = 2.0
TIMEOUT_S = 15.0

# Control (must match training: 50 Hz -> ctrl_dt=0.02).
CONTROL_HZ = 50.0
# ARM per-step delta scale. MUST match the run's trained action_scale. Read it
# from the run's sweep line, or — if the sweep does not override it, as with this
# policy — the ur3_pick default_config at the run's git_commit.
# For DR_cube_mass_light (1636c989) = 0.015.
# For L2_pos_cube (DR-ladder, 2026-07-22): metadata.json says "action_scale":
# 0.04, confirmed correct (this is the value the DR-ladder policies were
# ACTUALLY trained with, per run_gap_protocol.py's 2026-07-22 fix -- the env
# default was later lowered to 0.015, which made every ladder policy on the
# real robot ~2.7x too slow to reach the cube until this was caught).
# RealDR_vel_as04_gas0{1,2}_s0 (2026-07-27 velocity re-train) train at 0.04.
#
# BOTH scales are now RESOLVED FROM THE CHECKPOINT (see below) -- leave these
# as None for any policy whose sweep line sets them explicitly. They are only
# a manual fallback for older policies whose metadata.json carries no
# env_overrides entry (the 26D pre-2026-07-26 ones).
#
# Why: these used to be two hand-edited constants that had to be kept in sync
# with POLICY_NAME by comment alone. On 2026-07-27 the gas01 policy (trained
# gripper_action_scale=0.01) was rolled out twice at 0.02 -- 2x the trained
# finger rate, which ur3_realrobot_dependencies.compute_ctrl documents as
# driving the finger out-of-distribution. Two of three rollouts were wasted.
# print_action_scale_banner() existed to catch exactly this but was never
# actually called from anywhere, so nothing was printed and nothing failed.
ACTION_SCALE = None              # None => take from metadata env_overrides
# GRIPPER per-step delta scale — DECOUPLED from the arm in the new setup. MUST
# match the env's gripper_action_scale (ur3_pick default_config at the run's
# commit). L2_pos_cube (and every DR-ladder config) trains with 0.02 -- printed
# as "Env gripper_action_scale (training source of truth): 0.02" by
# run_gap_protocol.py for this exact policy.
GRIPPER_ACTION_SCALE = None      # None => take from metadata env_overrides
LOOKAHEAD_TIME = 0.1              # servoj smoothing [0.03, 0.2]
GAIN = 300                        # servoj stiffness [100, 2000]
SERVOJ_A = 0.3                    # max joint accel [rad/s^2]
SERVOJ_V = 1.0                    # max joint vel  [rad/s]
ALPHA = 1                      # 1.0 = send the policy's full action (no blend; matches training)
USE_FK_TCP = True                 # compute tcp_pos via MuJoCo FK (matches sim site)
# Finger-plant low-pass. The first hardware success (9a6e399) ran TAU=0.1 s and
# it was part of the winning config, so it is restored here (it was later set to
# 0.0 justified by a "low GRIPPER_SPEED_PCT", but the speed was actually 80, not
# low). This re-creates the sim finger lag the policy trained against.
GRIPPER_TAU = 0.1               # s; the value the first success ran (0 = no low-pass)
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
    "DR_cube_mass_light_1636c989": "DR_cube_mass_light_20260715_133502_2201",
    "Align_logic_easy_052d79fb30eeff144a0e3b1e3ef8cf495b070f9b": "Align_logic_easy_20260710_132446_2201",
    "Align_logic_mid_052d79fb30eeff144a0e3b1e3ef8cf495b070f9b": "Align_logic_mid_20260710_132449_2201",
    "more_rot_weight_e919791fe2e332b8fe2a1ad345b2528570730727": "DIAGHOLD300_posemid_hard_offON_AS0.025_LR6e-4_2048env_25M_s1_20260710_084027_2201",
    "AGATE250_posemid_hard_offOFF_4035f12ccc5925b7fac8af367a8096deff107707": "AGATE250_posemid_hard_offOFF_AS0.025_LR6e-4_2048env_25M_s1_20260710_102830_2201",
    "EP200_AS0.025_78fd8855a6eeb72d8ef1a0395cd6e4860db1439d": "EP200_AS0.025_d0.99_20260703_132735_2201",
    "Simplified_9cd87e0bef819dcca2d86ea25451eaa98bf78eb5": "Simplified_lightRandom_lr1e-3_20260701_173930_3867",
    "NoVelocity_mid_ea1ffd26f79c25db5c62af8e68022f6677b5aff6": "cur_mid_base_20260701_164529_3867",
    "Nolocity_ea1ffd26f79c25db5c62af8e68022f6677b5aff6": "cur_light_lift8rot_20260701_163113_3867",
    "base90_lr4e-10": "base90_j10_fin25_20M_lr4e-4_20260626_101918_6311",
    "reasonable starting positions": "Reso_Pos_lr6e-4_20260626_110022_7585",
    "pick_12M_rand_base85_fin25": "Pick_12M_rand_base85_fin25_20260626_094554_6311",
    "L2_pos_cube": "L2_pos_cube_s2_20260717_131601_926",
    "L1_pos": "L1_pos_s0_20260717_121957_6311",
    # --- 2026-07-27 velocity re-train of the deployed real-robot DR config ---
    "RealDR_vel_as04_gas02_s0": "RealDR_vel_as04_gas02_s0_20260727_161152_6311",  # gripper_action_scale=0.02
    "RealDR_vel_as04_gas01_s0": "RealDR_vel_as04_gas01_s0_20260727_161344_6311",  # gripper_action_scale=0.01
    "Grasp_light_as03_vel":          "Grasp_light_as03_vel_dr_14M_s1_20260727_201658_2201",
    "Grasp_mid_as03_vel":             "Grasp_mid_as03_vel_dr_14M_s1_20260727_201502_2201",
}
# Velocity re-train OOD check: point at gas02 or gas01. Action scales resolve
# automatically from each checkpoint's metadata, so switching policy is a
# ONE-LINE change here. Paste each run's W&B id above after training. Switch
# back to "L1_pos" etc. for the older 26D policies (those have no
# env_overrides, so they still need the manual constants set).
POLICY_NAME =  "Grasp_mid_as03_vel"  # pick policy to run

WANDB_ENTITY = "weissma6-zhaw-school-of-engineering"
WANDB_PROJECT = "UR3_pick_ppo"
WANDB_RUN_ID = POLICY_REGISTRY[POLICY_NAME]
# Local cache is run-id keyed: evaluation/downloaded_policies/{run_id}/
POLICY_PATH = default_policy_dir(WANDB_RUN_ID)
# real_robot_results/{POLICY_NAME}/{run_stamp}/ (NOT "results/", which is
# git-ignored) so the run outputs are tracked and uploaded, and so repeated
# runs of the SAME policy accumulate as separate folders instead of
# overwriting each other -- needed to collect e.g. 10 real rollouts per
# policy for evaluation/compare_reward_train_sim_real.py. run_stamp is a
# wall-clock timestamp taken once at script start (time.strftime, not
# reused across runs).
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
FOLDER_OUT = os.path.join(SCRIPT_DIR, "real_robot_results", POLICY_NAME, RUN_STAMP)
VIDEO_OUT = os.path.join(FOLDER_OUT, "ur3_pick_replay.mp4")
CSV_OUT = os.path.join(FOLDER_OUT, "ur3_pick_states.csv")
ROBOT_PLOTS_OUT = os.path.join(FOLDER_OUT, "ur3_pick_robot_plots.png")
GRIPPER_PLOTS_OUT = os.path.join(FOLDER_OUT, "ur3_pick_gripper_plots.png")
TIMING_OUT = os.path.join(FOLDER_OUT, "ur3_pick_timing.png")
META_OUT = os.path.join(FOLDER_OUT, "ur3_pick_meta.json")
REWARD_CSV_OUT = os.path.join(FOLDER_OUT, "ur3_pick_reward.csv")
REWARD_PLOT_OUT = os.path.join(FOLDER_OUT, "ur3_pick_reward.png")
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
# print(f"\nMoving to start pose {Q_START}")
# robot.send_movej(Q_START, a=1.0, v=0.5, asynchronous=False)  # blocking moveJ

# ── Download policy from W&B if missing (cache-aware, run-id keyed) ─────
download_policy(
    WANDB_RUN_ID, out_dir=POLICY_PATH,
    entity=WANDB_ENTITY, project=WANDB_PROJECT,
)

# ── Resolve action scales FROM THE CHECKPOINT ──────────────────────────
# Runs after download_policy (metadata.json is guaranteed on disk) and before
# any policy-driven motion. A value in metadata's env_overrides means the sweep
# line set it explicitly, so it is the training source of truth; a bare
# top-level metadata value is NOT trusted (policy_downloader writes a hardcoded
# action_scale=0.04 when the run's config carried none).
def _resolve_scale(name, manual, overrides):
    trained = overrides.get(name)
    if trained is None:
        if manual is None:
            raise SystemExit(
                f"FATAL: {name} is not in metadata env_overrides for "
                f"{os.path.basename(POLICY_PATH)} and no manual value is set.\n"
                f"       Look up the env default at the run's git_commit and set "
                f"{name.upper()} explicitly."
            )
        print(f"  !! {name}: not in env_overrides -- using MANUAL {manual} "
              f"(UNVERIFIED against training)")
        return float(manual)
    if manual is not None and abs(float(manual) - float(trained)) > 1e-9:
        raise SystemExit(
            f"FATAL: {name} mismatch -- manual {manual} vs trained {trained} "
            f"({float(manual)/float(trained):.2f}x).\n"
            f"       This is the 2026-07-27 gas01/gas02 bug. Set "
            f"{name.upper()} = None to use the trained value."
        )
    return float(trained)


with open(os.path.join(POLICY_PATH, "metadata.json"), "r", encoding="utf-8") as _f:
    _env_overrides = (json.load(_f).get("env_overrides") or {})
ACTION_SCALE = _resolve_scale("action_scale", ACTION_SCALE, _env_overrides)
GRIPPER_ACTION_SCALE = _resolve_scale(
    "gripper_action_scale", GRIPPER_ACTION_SCALE, _env_overrides)

# Report-only banner (now actually wired in -- it was dead code before).
print_action_scale_banner(
    policy_path=POLICY_PATH,
    rollout_action_scale=ACTION_SCALE,
    rollout_gripper_action_scale=GRIPPER_ACTION_SCALE,
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
    gripper_action_scale=GRIPPER_ACTION_SCALE,
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
    box_z_offset=BOX_Z_OFFSET,
)

# ── Results ────────────────────────────────────────────────────────────
stop_reason = stats.get("stopped_reason", "completed")
final_dist = stats.get("final_box_to_target_dist", float("nan"))
print(f"\nStopped: {stop_reason}")
print(f"Final box->target: {final_dist * 1000:.1f} mm — "
      f"{'REACHED' if final_dist < REACH_TOL else 'NOT REACHED'}")
print(f"Steps: {len(df)}, wall time: {stats.get('total_wall_time_s', 0.0):.2f}s")

# ── Success (lift-and-hold; drop/return DISABLED for now) ──────────────
# Only treat as success if the loop actually converged — a mocap-loss stop with
# the box happening to be near target is NOT a real reach. The drop (open gripper)
# and return-to-home are intentionally removed for now: the task is just "lift the
# box to the training target and hold it there", so keep the fingers closed and
# leave the arm where it converged.
reached = stop_reason != "mocap_lost" and final_dist < REACH_TOL
if reached:
    print("\nSUCCESS — box lifted to the training target; holding (no drop).")
    time.sleep(3.0)

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
    box_z_offset=BOX_Z_OFFSET,
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
    gripper_action_scale=GRIPPER_ACTION_SCALE,
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

# ── Reconstruct + plot the same reward terms as training (Task 1) ──────
# Replays the logged geometry (arm q, finger position, mocap box pose +
# orientation, action) through ur3_reward_replay's RewardReplayer -- a
# numpy port of the training env's _get_reward, keyed to the SAME
# default_config().reward_config.scales -- so this run's reward is directly
# comparable to eval/episode_<term> in W&B. One multi-line chart: every
# scaled reward term as its own line + a bold total, over the run's steps.
try:
    reward_df = replay_dataframe(df, xml_path=MODEL_PATH, target=DROP_TARGET)
    reward_df.to_csv(REWARD_CSV_OUT, index=False)
    print(f"Reward CSV: {REWARD_CSV_OUT}")
    save_reward_terms_plot(
        reward_df, REWARD_PLOT_OUT,
        title=f"{POLICY_NAME} — real run reward ({stop_reason})",
    )
    print(f"Reward plot: {REWARD_PLOT_OUT}")
except Exception as e:  # noqa: BLE001
    # Reward reconstruction is a diagnostic add-on, not required for the
    # pick loop itself -- never let it fail the run's other outputs (video/
    # CSV/plots/metadata above are already saved by this point).
    print(f"\n[warn] reward reconstruction failed: {e}; "
          "video/CSV/plots/metadata above are unaffected.")

# ── Disconnect ─────────────────────────────────────────────────────────
mocap.stop()
if gripper is not None:
    gripper.disconnect()
robot.disconnect()
print("Done.")
