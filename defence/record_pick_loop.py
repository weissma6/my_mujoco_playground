"""Pick the cube, drop it in the taped square, pause, repeat — for 60 seconds.

Runs on the LINUX lab machine (UR3e + Robotiq Hand-E + Nokov/VRPN mocap).
The continuous-demo sibling of `record_real_rollout.py`: same policy, same
50 Hz loop, same D22 drop — but in a session loop instead of one episode.

One cycle is:

    run_policy_loop (400 steps, 8.0 s)   the policy picks and carries
    open gripper + DROP_SETTLE_S         the cube falls into the square
    REPOSITION_S                         <-- YOU re-place the cube here
    re-seed the gripper + SETTLE_S       see "THE RE-SEED" below

~12 s per cycle, so SESSION_SECONDS = 60 gives about 5 picks.

NO HOMING MOVE between cycles: each cycle starts from wherever the drop left
the arm. That is deliberate (it buys ~3-4 s per cycle) but it does mean the
arm starts cycle 2+ in a pose the policy never saw at reset during training —
`arm_q` is 6 of its 33 obs dims. If cycle 1 picks cleanly and later cycles
wander, that is the cause: set HOME_BETWEEN = True.

THE RE-SEED — the one thing that makes this loop different from running
`record_real_rollout.py` five times. `run_policy_loop` seeds its gripper
integrator from the `task_home` keyframe (per-finger 0.0125 m), but the D22
drop leaves the physical fingers wide open at 0.0. Handing the next cycle a
12.5 mm disagreement between the estimate and the hardware re-creates exactly
the defect fixed in b12f2bc, where the policy flew on a stale gripper channel
and never closed. `_prepare_next_cycle` commands the fingers back to
START_FINGER and waits SETTLE_S so the seed is true again. Do not remove it.

THIN DRIVER, same posture as record_real_rollout.py and run_gap_protocol.py:
the control loop is NOT duplicated here. `UR3RealRobotPick.run_policy_loop`
still owns RTDE receive/servoJ/the Hand-E worker thread/logging, and the force
cutoff, scale resolution, timing assertion and CSV/manifest writers are
imported from `record_real_rollout` rather than re-implemented.

ATTENDED BY DESIGN — and this is the deliberate divergence from
`record_real_rollout.py`, which states it is unattended. There is still no
`input()` (nothing to approve mid-session), but REPOSITION_S exists precisely
so a human can reach into the workspace between cycles. The arm is stopped
during that window: `run_policy_loop`'s `finally` calls `servoStop()` on every
exit path. Keep a hand on the E-stop anyway.

Artifacts, one session directory holding replayable per-cycle runs:

    defence/runs/<UTC>_loop_<policy>/session.json      config + per-cycle rows
    defence/runs/<UTC>_loop_<policy>/cycle_01/manifest.json
    defence/runs/<UTC>_loop_<policy>/cycle_01/real_states.csv

Each cycle_NN/ is a complete "defence/1" run, so the sim mirror works on any
one of them unchanged:

    python defence/render_sim_rollout.py --run-dir defence/runs/<session>/cycle_03

Usage (from this folder — ../.. paths depend on it, per CLAUDE.md). Every knob
is a constant in the block below; there are no command-line arguments:

    python record_pick_loop.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "robots", "UR3e"),
    os.path.join(REPO_ROOT, "evaluation"),
    os.path.join(REPO_ROOT, "evaluation", "downloaded_policies"),
    _THIS_DIR,
):
  if _p not in sys.path:
    sys.path.insert(0, _p)

from ur3_realrobot_dependencies import UR3RealRobotPick  # noqa: E402
from motion_capture.mymocap.vrpn_dependencies import (  # noqa: E402
    VRPNRigidBodyReader,
)
from robots.hande.HandE_dependency import HandEGripper  # noqa: E402
from evaluation.gap_metrics import place_error  # noqa: E402 -- D22 drop metric
from policy_downloader import (  # noqa: E402
    default_policy_dir,
    policy_exists_locally,
)

# Imported, never edited (repo rule): the gap protocol owns these helpers and
# the model path, and reusing them is what keeps this run comparable with the
# submitted results.
import run_gap_protocol as gap  # noqa: E402

# Everything below is imported, not re-implemented -- see the THIN DRIVER note
# in the module docstring. record_real_rollout is import-safe (its main() is
# behind an `if __name__ == "__main__"` guard and it has no module-level side
# effects).
from record_real_rollout import (  # noqa: E402
    GRIPPER_CONVENTION,
    SCHEMA_VERSION,
    _assert_trained_timing,
    _build_states_csv,
    _default_run_name,
    _ForceGuardedUR3,
    _resolve_scales,
    _task_home_qpos,
)

# ###########################################################################
# ###                                                                     ###
# ###   >>>>>>   ALL RUN PARAMETERS -- EDIT HERE   <<<<<<                 ###
# ###                                                                     ###
# ###########################################################################

# --- Policy ----------------------------------------------------------------
CHECKPOINT        = "Snappy2_as04_ar70_g01_s1_20260817_202921_2201"  # policy to deploy
ACTION_SCALE      = None      # arm delta scale; None => from checkpoint metadata
GRIPPER_SCALE     = None      # finger delta scale; None => from checkpoint metadata
EPISODE_LENGTH    = 400       # hard horizon in steps; must equal trained value
CONTROL_HZ        = 50.0      # control rate, Hz; must equal trained 1/ctrl_dt

# --- Session ---------------------------------------------------------------
SESSION_SECONDS   = 60.0      # total wall-clock budget; a started cycle always finishes
REPOSITION_S      = 2.0       # pause after each drop to re-place the cube by hand
DROP_SETTLE_S     = 1.5       # wait after gripper-open so the cube lands before scoring
SETTLE_S          = 1.0       # wait after commanding the start finger opening
DROP_TARGET       = (0.212, 0.212, 0.165)  # taped square centre, BASE frame
HOME_BETWEEN      = False     # True => moveJ back to task_home between cycles

# --- Control law -----------------------------------------------------------
ALPHA             = 1.0       # servoJ command blend; 1.0 = no smoothing
CONTROL_LAW       = "rebase"  # "rebase" (legacy) | "integrate" (sim parity)
LOOKAHEAD_TIME    = 0.1       # servoJ smoothing, s
GAIN              = 300       # servoJ stiffness
SERVOJ_A          = 0.3       # servoJ max joint accel, rad/s^2
SERVOJ_V          = 1.0       # servoJ max joint vel, rad/s
MOVEJ_A           = 0.4       # moveJ accel for the optional homing move, rad/s^2
MOVEJ_V           = 0.4       # moveJ speed for the optional homing move, rad/s
GRIPPER_TAU       = 0.1       # finger-plant low-pass time constant, s
INIT_KEYFRAME     = "task_home"  # scene keyframe the gripper integrator seeds from
START_FINGER      = 0.0125    # per-finger start opening, m; MUST match INIT_KEYFRAME
BOX_Z_OFFSET      = 0.0       # metres added to the mocap box Z before it feeds the obs

# --- Safety ----------------------------------------------------------------
FORCE_LIMIT_N     = 30.0      # hard cutoff on ||F - F_bias||, N
FORCE_CONSECUTIVE = 2         # consecutive over-limit ticks required to trip
FORCE_WARMUP      = 5         # ticks ignored while the wrench estimate settles
FORCE_BIAS_SAMPLES = 25       # standstill samples averaged into the force bias
MOCAP_STALE_S     = 0.25      # halt the cycle if the cube goes unseen this long, s
MAX_CONSEC_FAILS  = 0         # 0 = never abort; N = abort after N failures in a row
SETUP_WAIT_S      = 6.0       # countdown before cycle 1: place the cube, stand clear

# --- Hardware --------------------------------------------------------------
ROBOT_IP          = "192.168.1.4"   # UR3e (PolyScope X)
MOCAP_SERVER_IP   = "10.1.1.198"    # Nokov/VRPN server
MOCAP_BODY        = "CubeInCube2"   # streamed VRPN rigid-body name
GRIPPER_PORT      = 49999           # Robotiq URCapX XML-RPC port
GRIPPER_SLAVE_ID  = 9               # this Hand-E = slaveId 9
GRIPPER_SPEED_PCT = 60              # 30-60 = gentle finger motion
GRIPPER_FORCE_PCT = 80              # high, so fingers clamp instead of stalling short
ENABLE_GRIPPER    = True            # False = dry run, gripper channel is a no-op

# NOT deploy parameters, deliberately absent: discount, learning rate, entropy
# cost and the rest of the PPO hyperparameters shape TRAINING only. They reach
# no code path at inference and are not even recorded in metadata.json, so
# listing them here would imply a knob that does nothing.

# ###########################################################################
# ###   END OF THE EDIT-HERE BLOCK                                        ###
# ###########################################################################

LOOP_SCHEMA_VERSION = "defence/loop/1"


# ===========================================================================
# Session setup.
# ===========================================================================

def _git_branch():
  """Best-effort current branch. Never raises — same convention as
  gap.git_sha(), which has no branch equivalent: a logging helper must not be
  able to crash a real-robot run."""
  import subprocess  # local: keeps module import cheap
  try:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT,
        stderr=subprocess.DEVNULL).decode().strip()
  except Exception:  # noqa: BLE001
    return "unknown"


def _countdown(seconds, banner):
  """Timed pause with a banner — no input(), same as record_real_rollout."""
  if seconds <= 0:
    return
  print("\n" + "!" * 70)
  print(banner)
  print("!" * 70)
  for remaining in range(int(seconds), 0, -1):
    print(f"  {remaining} ...", flush=True)
    time.sleep(1.0)


def setup_once():
  """Connect everything and resolve the policy. Returns a context dict.

  Everything here happens exactly once per session. In particular
  `init_fk_model` MUST run before the first `run_policy_loop`: the gripper
  seed reads the keyframe off that model and raises without it, which would
  fail every cycle rather than just the first.
  """
  if not policy_exists_locally(CHECKPOINT):
    print(f"[checkpoint] {CHECKPOINT} not cached -- fetching from W&B ...")
    UR3RealRobotPick.download_policy_from_wandb(
        CHECKPOINT, out_dir=default_policy_dir(CHECKPOINT),
        entity=gap.WANDB_ENTITY, project=gap.WANDB_PROJECT,
    )
  policy_path = default_policy_dir(CHECKPOINT)
  meta_path = os.path.join(policy_path, "metadata.json")
  params_path = os.path.join(policy_path, "params.msgpack")
  for p in (meta_path, params_path):
    if not os.path.exists(p):
      raise SystemExit(f"checkpoint incomplete: {p} missing")
  with open(meta_path, encoding="utf-8") as f:
    ckpt_meta = json.load(f)

  ctrl_dt = 1.0 / float(CONTROL_HZ)
  action_scale, gripper_action_scale, scale_source = _resolve_scales(
      CHECKPOINT, policy_path, ACTION_SCALE)
  if GRIPPER_SCALE is not None:
    gripper_action_scale = float(GRIPPER_SCALE)
    scale_source += " (gripper overridden by GRIPPER_SCALE)"
  _assert_trained_timing(ckpt_meta, CHECKPOINT, ctrl_dt, EPISODE_LENGTH)
  if int(ckpt_meta["action_dim"]) != 7:
    raise SystemExit(
        f"{CHECKPOINT} has action_dim {ckpt_meta['action_dim']}, expected 7.")

  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  session_name = f"{stamp}_loop_{_default_run_name(CHECKPOINT)}"
  out_dir = os.path.join(_THIS_DIR, "runs", session_name)
  os.makedirs(out_dir, exist_ok=True)

  print("=" * 70)
  print(f"checkpoint  : {CHECKPOINT}")
  print(f"obs/act     : {ckpt_meta['obs_dim']} / {ckpt_meta['action_dim']}")
  print(f"scales      : arm {action_scale}  gripper {gripper_action_scale}")
  print(f"              ({scale_source})")
  print(f"session     : {SESSION_SECONDS:.0f} s budget, "
        f"{EPISODE_LENGTH} steps @ {CONTROL_HZ:.0f} Hz per cycle")
  print(f"home between: {HOME_BETWEEN}")
  print(f"out dir     : {out_dir}")
  print(f"force cutoff: {FORCE_LIMIT_N:.1f} N")
  print("=" * 70)

  robot = _ForceGuardedUR3(ROBOT_IP)
  robot.connect()
  if not robot.is_connected():
    raise SystemExit("robot did not connect.")

  mocap = VRPNRigidBodyReader(
      MOCAP_SERVER_IP, rigid_body_name=MOCAP_BODY, names=[MOCAP_BODY])
  mocap.start(timeout=5.0)
  mocap.wait_for_data(timeout=5.0)

  if ENABLE_GRIPPER:
    gripper = HandEGripper(
        ROBOT_IP, port=GRIPPER_PORT, slave_id=GRIPPER_SLAVE_ID,
        speed_pct=GRIPPER_SPEED_PCT, force_pct=GRIPPER_FORCE_PCT,
    )
    gripper.connect()
    gripper.open_gripper()
    gripper_fn = lambda norm: gripper.command(norm * 0.025)  # noqa: E731
    gripper_state_fn = lambda: gripper.read_state()  # noqa: E731
  else:
    gripper = None
    gripper_fn = lambda norm: None  # noqa: E731
    gripper_state_fn = None
    print("  [dry run] gripper channel is a no-op.")

  robot.load_policy_fn(policy_path=policy_path, deterministic=True)
  robot.init_fk_model(gap.MODEL_PATH)

  force_bias, bias_spread = robot.measure_force_bias(
      n_samples=FORCE_BIAS_SAMPLES)
  if bias_spread > FORCE_LIMIT_N * 0.5:
    raise SystemExit(
        f"force bias is unstable (spread {bias_spread:.1f} N) -- the arm was "
        "not at standstill. Re-run without touching it.")
  print(f"[force] bias {np.round(force_bias, 2).tolist()} N "
        f"(spread {bias_spread:.2f} N)")

  return {
      "robot": robot, "mocap": mocap, "gripper": gripper,
      "gripper_fn": gripper_fn, "gripper_state_fn": gripper_state_fn,
      "ckpt_meta": ckpt_meta, "params_path": params_path,
      "ctrl_dt": ctrl_dt, "action_scale": action_scale,
      "gripper_action_scale": gripper_action_scale,
      "scale_source": scale_source, "force_bias": force_bias,
      "checkpoint_sha256": gap.sha256_file(params_path),
      "out_dir": out_dir, "session_name": session_name,
  }


# ===========================================================================
# One cycle.
# ===========================================================================

def run_cycle(ctx, cycle_idx, t_session):
  """One pick -> carry -> drop -> score. Returns the session.json row."""
  robot = ctx["robot"]
  mocap = ctx["mocap"]
  gripper = ctx["gripper"]
  cycle_dir = os.path.join(ctx["out_dir"], f"cycle_{cycle_idx:02d}")
  os.makedirs(cycle_dir, exist_ok=True)
  t_start_rel = time.perf_counter() - t_session

  # Disarm for the pre-flight reads. _fg_armed is set True by arm_force_guard
  # and never cleared (it only goes False in __init__), so from cycle 2 on the
  # guard would still be live here -- and an armed receive_feedback() stamps
  # t0 and logs a wrench row. Disarming makes every cycle behave exactly like
  # cycle 1: t0 comes from run_policy_loop's first tick, not from this read.
  robot._fg_armed = False

  # Pre-flight reads, BEFORE the guard is re-armed -- same order as
  # record_real_rollout, where receive_feedback() returns early while disarmed.
  fb = robot.receive_feedback()
  measured_arm_qpos = [float(v) for v in np.asarray(fb["q"], dtype=float)]
  box_xyz, box_quat = mocap.get_rigid_body_pose()
  if box_xyz is None:
    raise RuntimeError("mocap has no data for the tracked cube")
  box_pos_base = robot.mocap_pos_to_base(box_xyz)
  box_quat_base = robot.mocap_quat_to_base(box_quat)
  target_pos = np.asarray(DROP_TARGET, dtype=np.float32)

  print(f"\n{'=' * 70}\ncycle {cycle_idx}  (t+{t_start_rel:.1f} s)  "
        f"cube at {np.round(box_pos_base, 3).tolist()}\n{'=' * 70}")

  # Re-arming per cycle also clears _fg_tripped / _fg_trip_step / _fg_peak_n
  # and wrench_log. Without it cycle 2's completeness check still sees cycle
  # 1's trip, and its CSV inherits cycle 1's wrench rows.
  robot.arm_force_guard(
      limit_n=FORCE_LIMIT_N, bias=ctx["force_bias"],
      consecutive=FORCE_CONSECUTIVE, warmup=FORCE_WARMUP,
  )

  df, stats = robot.run_policy_loop(
      drop_target=target_pos,
      mocap_reader=mocap,
      control_hz=CONTROL_HZ,
      timeout_s=EPISODE_LENGTH / CONTROL_HZ * 1.12 + 0.5,
      action_scale=ctx["action_scale"],
      gripper_action_scale=ctx["gripper_action_scale"],
      lookahead_time=LOOKAHEAD_TIME,
      gain=GAIN,
      servoj_a=SERVOJ_A,
      servoj_v=SERVOJ_V,
      alpha=ALPHA,
      gripper_fn=ctx["gripper_fn"],
      gripper_state_fn=ctx["gripper_state_fn"],
      gripper_tau=GRIPPER_TAU,
      gripper_max_rate=float("inf"),
      control_law=CONTROL_LAW,
      use_fk_tcp=True,
      reach_tol=None,      # never terminate early on reach
      dwell_time_s=0.0,
      max_steps=int(EPISODE_LENGTH),
      init_keyframe=INIT_KEYFRAME,
      mocap_stale_s=MOCAP_STALE_S,
      box_z_offset=BOX_Z_OFFSET,
      debug_print=False,
  )
  stopped_reason = stats.get("stopped_reason", "unknown")
  if robot._fg_tripped:
    stopped_reason = "force_limit"

  # D22 THE DROP -- open the gripper and let the cube fall the last ~50 mm
  # into the taped square. Zero scripted arm motion; the policy carried it.
  # Only scored on a cycle that ran its full horizon: an aborted one may not
  # have the cube at all, so "where did it land" would be fabricated.
  complete = (not robot._fg_tripped) and len(df) >= EPISODE_LENGTH
  landed_box_xy = place_error_m = place_in_square = None
  drop_attempted = False
  if complete and gripper is not None:
    drop_attempted = True
    print("  D22: opening gripper -- dropping into the taped square ...")
    gripper.open_gripper()
    time.sleep(DROP_SETTLE_S)
    landed_xyz, _ = mocap.get_rigid_body_pose()
    if landed_xyz is not None:
      landed_base = robot.mocap_pos_to_base(landed_xyz)
      landed_box_xy = (float(landed_base[0]), float(landed_base[1]))
      place_error_m, place_in_square = place_error(landed_box_xy)
      print(f"  D22 place_error: {place_error_m * 1000:.1f} mm "
            f"(in_square={place_in_square})")
  else:
    print(f"  D22: no drop -- stop_reason={stopped_reason!r}")

  out_df = _build_states_csv(
      df, robot.wrench_log, os.path.join(cycle_dir, "real_states.csv"))
  n_steps = int(len(out_df))
  grasped_steps = (int(out_df["grasped"].astype(str).eq("True").sum())
                   if "grasped" in out_df.columns else None)

  ckpt_meta = ctx["ckpt_meta"]
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "run_name": f"{ctx['session_name']}/cycle_{cycle_idx:02d}",
      "t0_unix": float(robot.t0_unix),
      "t0_iso": datetime.fromtimestamp(
          robot.t0_unix, timezone.utc).isoformat(timespec="milliseconds"),
      "t0_monotonic": float(robot.t0_monotonic),
      "init": {
          "arm_qpos": measured_arm_qpos,
          # No moveJ under HOME_BETWEEN=False, so the commanded pose IS the
          # measured one: the cycle starts from wherever the drop left the arm.
          "commanded_arm_qpos": measured_arm_qpos,
          "finger": float(START_FINGER),
          "box_pos": [float(v) for v in box_pos_base],
          "box_quat_wxyz": [float(v) for v in box_quat_base],
          "target_pos": [float(v) for v in target_pos],
          "cube_half_extents": None,
          "lifter_top_height": None,
          "lifter_tilt_rp": None,
          "settle_s": float(SETTLE_S),
      },
      "pose_ref": {
          "source": "task_home" if (HOME_BETWEEN or cycle_idx == 1)
                    else "previous_cycle_drop_pose",
          "arm_pose_source": "task_home" if (HOME_BETWEEN or cycle_idx == 1)
                             else "previous_cycle_drop_pose",
          "protocol_id": None, "poses_sha256": None,
          "episode_id": None, "level": None, "split": None,
      },
      "policy": {
          "checkpoint_id": CHECKPOINT,
          "checkpoint_sha256": ctx["checkpoint_sha256"],
          "obs_dim": int(ckpt_meta["obs_dim"]),
          "action_dim": int(ckpt_meta["action_dim"]),
          "policy_hidden_layer_sizes": list(
              ckpt_meta["network_factory"]["policy_hidden_layer_sizes"]),
      },
      "control": {
          "ctrl_dt": ctx["ctrl_dt"],
          "action_scale": float(ctx["action_scale"]),
          "gripper_action_scale": float(ctx["gripper_action_scale"]),
          "scale_source": ctx["scale_source"],
          "gripper_convention": GRIPPER_CONVENTION,
          "episode_length": n_steps,
          "nominal_horizon": int(EPISODE_LENGTH),
          "max_steps": int(EPISODE_LENGTH),
          "horizon_enforced": True,
          "control_hz_requested": float(CONTROL_HZ),
          "achieved_hz": float(
              stats.get("true_inferred_frequency_hz") or float("nan")),
          "control_law": CONTROL_LAW,
          "box_z_offset": float(BOX_Z_OFFSET),
      },
      "result": {
          "stop_reason": stopped_reason,
          "stop_step": int(robot._fg_trip_step)
                       if robot._fg_trip_step is not None else n_steps - 1,
          "n_steps": n_steps,
          "force_limit_n": float(FORCE_LIMIT_N),
          "force_bias_n": [float(v) for v in ctx["force_bias"]],
          "peak_force_n": float(robot._fg_peak_n),
          "overrun_count": int(stats.get("num_overruns") or 0),
          "drop_attempted": bool(drop_attempted),
          "landed_box_xy": landed_box_xy,
          "place_error_m": place_error_m,
          "place_in_square": place_in_square,
          "drop_settle_s": float(DROP_SETTLE_S),
      },
      "env": {
          "env_name": ckpt_meta.get("env_name", "UR3Pick"),
          "xml_path": os.path.relpath(gap.MODEL_PATH, REPO_ROOT),
          "git_sha": gap.git_sha(),
          "branch": _git_branch(),
      },
      # All-null on purpose: filming is external. render_sim_rollout's
      # validate_manifest requires these keys PRESENT, not non-null.
      "video": {
          "real_mp4": None, "webcam_fps_nominal": None,
          "n_webcam_frames": 0, "frame_index_csv": None,
          "source": "external",
      },
  }
  with open(os.path.join(cycle_dir, "manifest.json"), "w",
            encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

  print(f"  cycle {cycle_idx}: {stopped_reason}  {n_steps} steps  "
        f"peak {robot._fg_peak_n:.1f} N  grasped {grasped_steps} steps")

  return {
      "cycle": cycle_idx,
      "t_start_rel": round(t_start_rel, 2),
      "stop_reason": stopped_reason,
      "n_steps": n_steps,
      "peak_force_n": round(float(robot._fg_peak_n), 2),
      "grasped_steps": grasped_steps,
      "landed_box_xy": landed_box_xy,
      "place_error_m": place_error_m,
      "place_in_square": place_in_square,
  }


def prepare_next_cycle(ctx):
  """Between-cycle reset. Runs after a FAILED cycle too, so it must not assume
  the previous one got anywhere.

  The gripper re-command is the load-bearing line -- see THE RE-SEED in the
  module docstring. What is deliberately NOT here: _gripper_ctrl, _arm_ctrl,
  _finger_pos_est and _prev_finger_pos_est, all of which run_policy_loop
  resets itself as of b12f2bc.
  """
  robot = ctx["robot"]
  gripper = ctx["gripper"]

  if HOME_BETWEEN:
    home_q = _task_home_qpos(gap.MODEL_PATH)[:6]
    robot.send_movej(list(home_q), a=MOVEJ_A, v=MOVEJ_V, asynchronous=False,
                     textmsg="pick-loop home")

  print(f"\n  >>> REPOSITION THE CUBE -- {REPOSITION_S:.0f} s (arm stopped) <<<")
  time.sleep(REPOSITION_S)

  # THE RE-SEED. The drop left the fingers wide open (0.0); run_policy_loop
  # will seed its integrator from the keyframe (START_FINGER). Command the
  # hardware back to that opening and let it get there, or the next cycle
  # starts with the estimate and the physical gripper 12.5 mm apart.
  if gripper is not None:
    gripper.command(START_FINGER)
  time.sleep(SETTLE_S)

  # t0 is set on the FIRST receive_feedback() after arming and never again, so
  # without this every later cycle would stamp its manifest with cycle 1's t0.
  robot.t0_unix = None
  robot.t0_perf = None
  robot.t0_monotonic = None


# ===========================================================================
# Main.
# ===========================================================================

def main():
  ctx = setup_once()
  robot, mocap, gripper = ctx["robot"], ctx["mocap"], ctx["gripper"]
  results = []
  consec_fail = 0
  aborted = None

  try:
    if gripper is not None:
      gripper.command(START_FINGER)
    _countdown(SETUP_WAIT_S,
               "PLACE THE CUBE on the marker, then stand clear. "
               f"The session runs {SESSION_SECONDS:.0f} s.")
    time.sleep(SETTLE_S)

    t_session = time.perf_counter()
    cycle_idx = 0
    while time.perf_counter() - t_session < SESSION_SECONDS:
      cycle_idx += 1
      try:
        results.append(run_cycle(ctx, cycle_idx, t_session))
        consec_fail = 0
      except KeyboardInterrupt:
        raise
      except Exception as e:  # noqa: BLE001
        # One bad cycle never kills the session -- the gap protocol's rule.
        print(f"  cycle {cycle_idx} FAILED: {type(e).__name__}: {e}")
        results.append({
            "cycle": cycle_idx,
            "t_start_rel": round(time.perf_counter() - t_session, 2),
            "stop_reason": f"error:{type(e).__name__}",
            "n_steps": 0, "peak_force_n": None, "grasped_steps": None,
            "landed_box_xy": None, "place_error_m": None,
            "place_in_square": None,
        })
        consec_fail += 1
        if MAX_CONSEC_FAILS and consec_fail >= MAX_CONSEC_FAILS:
          aborted = f"{consec_fail} consecutive failures"
          print(f"  ABORTING: {aborted}")
          break

      if time.perf_counter() - t_session >= SESSION_SECONDS:
        break
      prepare_next_cycle(ctx)

  except KeyboardInterrupt:
    aborted = "KeyboardInterrupt"
    print("\n  interrupted -- writing session.json for the cycles so far.")
  finally:
    for obj, name in ((gripper, "gripper"), (robot, "robot"), (mocap, "mocap")):
      if obj is None:
        continue
      try:
        obj.stop() if name == "mocap" else obj.disconnect()
      except Exception:  # noqa: BLE001
        pass

  session = {
      "schema_version": LOOP_SCHEMA_VERSION,
      "session_name": ctx["session_name"],
      "aborted": aborted,
      "config": {
          "checkpoint": CHECKPOINT,
          "action_scale": float(ctx["action_scale"]),
          "gripper_action_scale": float(ctx["gripper_action_scale"]),
          "scale_source": ctx["scale_source"],
          "episode_length": int(EPISODE_LENGTH),
          "control_hz": float(CONTROL_HZ),
          "session_seconds": float(SESSION_SECONDS),
          "reposition_s": float(REPOSITION_S),
          "drop_settle_s": float(DROP_SETTLE_S),
          "settle_s": float(SETTLE_S),
          "drop_target": [float(v) for v in DROP_TARGET],
          "home_between": bool(HOME_BETWEEN),
          "alpha": float(ALPHA),
          "control_law": CONTROL_LAW,
          "init_keyframe": INIT_KEYFRAME,
          "start_finger": float(START_FINGER),
          "force_limit_n": float(FORCE_LIMIT_N),
      },
      "cycles": results,
  }
  with open(os.path.join(ctx["out_dir"], "session.json"), "w",
            encoding="utf-8") as f:
    json.dump(session, f, indent=2)

  n_ok = sum(1 for r in results if r["stop_reason"] == "horizon")
  n_placed = sum(1 for r in results if r.get("place_in_square"))
  print("\n" + "=" * 70)
  print(f"session   : {len(results)} cycles, {n_ok} full-horizon, "
        f"{n_placed} landed in the square")
  print(f"artifacts : {ctx['out_dir']}")
  print(f"replay    : python defence/render_sim_rollout.py --run-dir "
        f"{os.path.relpath(ctx['out_dir'], REPO_ROOT)}/cycle_01")
  print("=" * 70)


if __name__ == "__main__":
  main()
