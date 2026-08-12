"""Record ONE real UR3e pick episode for the VT2 defence side-by-side video.

Runs on the LINUX lab machine (UR3e + Robotiq Hand-E + Nokov/VRPN mocap +
Microsoft LifeCam). Drives one episode from a named frozen start pose to the
matched drop target while a webcam records throughout, then writes the three
artifacts `render_sim_rollout.py` needs on the MacBook:

    defence/runs/<UTC>_<name>/manifest.json      full sim-rollout determinant
    defence/runs/<UTC>_<name>/real_states.csv    per-control-step log
    defence/runs/<UTC>_<name>/real.mp4           H.264/yuv420p webcam video
    defence/runs/<UTC>_<name>/frame_index.csv    per-frame timestamps vs t0

THIN DRIVER, same posture as run_gap_protocol.py: the 50 Hz control loop is
NOT duplicated here. `UR3RealRobotPick.run_policy_loop` still owns RTDE
receive/servoJ/the Hand-E worker thread/logging. This script adds only what
that loop has no reason to know about: the webcam, the force cutoff, and the
manifest.

Two things are grafted on without touching any existing file:

1. WEBCAM (`_WebcamWorker`) — mirrors `_GripperWorker`'s single-slot mailbox
   pattern. The arm loop never touches the camera: capture, encode and
   timestamping all live on the worker thread, and the only arm-side
   interaction is a non-blocking `mark_step()` (one lock-guarded int store)
   that stamps each frame with the control step in flight. Zero added latency
   to the 50 Hz tick by construction.

2. FORCE CUTOFF (`_ForceGuardedUR3`) — `run_policy_loop` has no force limit
   and `getActualTCPForce()` was read but never used or logged. Rather than
   edit that loop, this subclass overrides `receive_feedback()` (called
   exactly once per tick, as stage 1, INSIDE the loop's try block) and raises
   `RobotStoppedExternally` the moment the bias-corrected TCP force exceeds
   the limit. That is the loop's existing emergency path: it is caught, the
   arm is halted by `servoStop()` in `finally`, the gripper worker is torn
   down, and the partial log is still returned. Detection and stop happen in
   the SAME tick — no extra latency, no polling thread.

   The limit is applied to the bias-corrected linear force ||F - F_bias||,
   where F_bias is measured at standstill just before the run. Raw
   `getActualTCPForce()` includes the payload/gravity offset, so an absolute
   threshold on it would either trip instantly or never; the bias correction
   is what makes a conservative number meaningful.

Usage (from this folder — ../.. paths depend on it, per CLAUDE.md):

    python record_real_rollout.py --name defence_take1 \
        --checkpoint L0_none_vel_s1_20260729_104930_2201 \
        --episode-id 0 --force-limit-n 30
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "robots", "UR3e"),
    os.path.join(REPO_ROOT, "evaluation"),
    os.path.join(REPO_ROOT, "evaluation", "downloaded_policies"),
):
  if _p not in sys.path:
    sys.path.insert(0, _p)

from ur3_realrobot_dependencies import (  # noqa: E402
    RobotStoppedExternally,
    UR3RealRobotPick,
)
from motion_capture.mymocap.vrpn_dependencies import (  # noqa: E402
    VRPNRigidBodyReader,
)
from robots.hande.HandE_dependency import HandEGripper  # noqa: E402
from evaluation.protocols import load_protocol  # noqa: E402
from evaluation import gap_target  # noqa: E402

# Imported, never edited (repo rule): the gap protocol owns these helpers and
# the model path, and reusing them is what keeps this run comparable with the
# submitted results.
import run_gap_protocol as gap  # noqa: E402

SCHEMA_VERSION = "defence/1"
POLICY_ROOT = os.path.join(REPO_ROOT, "evaluation", "downloaded_policies")
GRIPPER_CONVENTION = "0=open,0.025=closed"

# Leading columns of real_states.csv. render_sim_rollout.py writes the same
# names, in the same order, as the head of sim_states.csv — that shared prefix
# is the contract that lets the two runs be diffed column-wise.
CORE_COLUMNS = (
    ["step", "t_rel"]
    + [f"q{i}" for i in range(6)]
    + ["finger", "tcp_x", "tcp_y", "tcp_z"]
    + ["box_x", "box_y", "box_z"]
    + ["box_qw", "box_qx", "box_qy", "box_qz"]
    + ["target_x", "target_y", "target_z", "box_to_target_dist"]
    + [f"action{i}" for i in range(7)]
)


# ===========================================================================
# Webcam worker — mirrors _GripperWorker's mailbox pattern.
# ===========================================================================


class _WebcamWorker:
  """Background thread owning the LifeCam: capture, encode, timestamp.

  The arm loop's ONLY interaction is `mark_step()`, a non-blocking store into
  a single-slot mailbox (exactly like `_GripperWorker.command()`). Frame
  grabbing (~33 ms blocking read) and H.264 encoding both happen here, off
  the 50 Hz path. Encoding is a pipe write to an ffmpeg subprocess, so it
  releases the GIL rather than stalling the control thread.

  Every frame is stamped with absolute wall-clock `time.time()`. t_rel is
  computed later, once the control loop's t0 is known — the camera starts
  BEFORE the loop, so early frames legitimately carry negative t_rel and the
  compositor relies on that pre-roll to align the two clips.
  """

  def __init__(self, device, width, height, fps, out_path,
               debug_print=True):
    self._device = device
    self._width = int(width)
    self._height = int(height)
    self._fps = float(fps)
    self._out_path = out_path
    self._debug_print = debug_print

    # Single-slot mailbox for the control step in flight. Its own lock, held
    # only for one int store/load, so the arm loop can never queue behind the
    # frame-index append below.
    self._step_lock = threading.Lock()
    self._cur_step = -1

    # list.append is atomic under the GIL, so the index needs no lock.
    self._index = []
    self._n_frames = 0
    self._error = None

    self._ready = threading.Event()
    self._stop_event = threading.Event()
    self._thread = threading.Thread(
        target=self._run, daemon=True, name="LifeCam-capture")

  def start(self, timeout=10.0):
    """Start capturing and block until the first frame lands (or fail)."""
    self._thread.start()
    if not self._ready.wait(timeout=timeout):
      self.stop()
      raise RuntimeError(
          f"webcam produced no frame within {timeout:.0f}s "
          f"(device={self._device!r}). Check the LifeCam is plugged in and "
          f"not held by another process."
      )
    if self._error is not None:
      raise RuntimeError(f"webcam failed to start: {self._error}")

  def stop(self, timeout=10.0):
    self._stop_event.set()
    if self._thread.is_alive():
      self._thread.join(timeout=timeout)

  def mark_step(self, step):
    """Non-blocking: publish the control step currently executing."""
    with self._step_lock:
      self._cur_step = int(step)

  def latest_state(self):
    """Non-blocking: frame count and last stamped step."""
    with self._step_lock:
      cur = self._cur_step
    return {"n_frames": self._n_frames, "cur_step": cur}

  @property
  def index(self):
    """[(frame_idx, t_unix, ctrl_step)] — safe to read after stop()."""
    return list(self._index)

  def _run(self):
    try:
      import cv2
    except ImportError as e:
      self._error = (
          f"{e}. opencv-python is not installed in this venv — "
          f"`uv pip install opencv-python`."
      )
      self._ready.set()
      return

    import imageio.v2 as imageio

    cap = None
    writer = None
    try:
      cap = cv2.VideoCapture(self._device)
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
      cap.set(cv2.CAP_PROP_FPS, self._fps)
      # Shallow buffer: we want the freshest frame, not a backlog, because a
      # queued frame would carry a timestamp later than the scene it shows.
      try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
      except Exception:
        pass
      if not cap.isOpened():
        self._error = f"cv2.VideoCapture({self._device!r}) did not open."
        self._ready.set()
        return

      # yuv420p requires even dimensions; take what the driver actually gave
      # us rather than what we asked for, then force even.
      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._width
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._height
      w -= w % 2
      h -= h % 2
      if self._debug_print:
        print(f"  [webcam] capturing {w}x{h} -> {self._out_path}")

      writer = imageio.get_writer(
          self._out_path,
          fps=self._fps,
          codec="libx264",
          pixelformat="yuv420p",
          macro_block_size=1,
          ffmpeg_params=[
              "-profile:v", "high", "-level", "4.0",
              "-crf", "18", "-preset", "veryfast",
          ],
      )

      while not self._stop_event.is_set():
        ok, frame = cap.read()
        t_unix = time.time()
        if not ok or frame is None:
          continue
        if frame.shape[1] != w or frame.shape[0] != h:
          frame = frame[:h, :w]
        with self._step_lock:
          step = self._cur_step
        writer.append_data(frame[:, :, ::-1])  # BGR -> RGB
        self._index.append((self._n_frames, t_unix, step))
        self._n_frames += 1
        if not self._ready.is_set():
          self._ready.set()
    except Exception as e:  # noqa: BLE001
      self._error = f"{type(e).__name__}: {e}"
      if self._debug_print:
        print(f"\n[warn] webcam thread died: {self._error}")
      self._ready.set()
    finally:
      if writer is not None:
        try:
          writer.close()
        except Exception:
          pass
      if cap is not None:
        try:
          cap.release()
        except Exception:
          pass


# ===========================================================================
# Force-guarded robot.
# ===========================================================================


class _ForceGuardedUR3(UR3RealRobotPick):
  """UR3RealRobotPick + a hard TCP-force cutoff and a webcam step tap.

  `receive_feedback()` is stage 1 of every `run_policy_loop` tick and runs
  inside that loop's try block, which makes it the correct and only place to
  intervene without editing the loop. Overriding it gives us, per tick and in
  order: the wall-clock t0 anchor, the webcam step stamp, the wrench log, and
  the cutoff itself.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._fg_armed = False
    self._fg_limit_n = float("inf")
    self._fg_bias = np.zeros(3, dtype=float)
    self._fg_consecutive = 2
    self._fg_warmup = 5
    self._fg_streak = 0
    self._fg_calls = 0
    self._fg_tripped = False
    self._fg_trip_step = None
    self._fg_peak_n = 0.0
    self._webcam = None
    self.wrench_log = []
    self.t0_unix = None
    self.t0_perf = None
    self.t0_monotonic = None

  def measure_force_bias(self, n_samples=25, dt=0.02):
    """Average the standstill wrench — the payload/gravity offset."""
    samples = []
    for _ in range(int(n_samples)):
      fb = super().receive_feedback()
      samples.append(np.asarray(fb["tcp_force"], dtype=float)[:3])
      time.sleep(dt)
    bias = np.mean(np.asarray(samples), axis=0)
    spread = float(np.max(np.linalg.norm(
        np.asarray(samples) - bias, axis=1)))
    return bias, spread

  def arm_force_guard(self, limit_n, bias, consecutive, warmup, webcam):
    self._fg_limit_n = float(limit_n)
    self._fg_bias = np.asarray(bias, dtype=float).reshape(3)
    self._fg_consecutive = int(consecutive)
    self._fg_warmup = int(warmup)
    self._fg_streak = 0
    self._fg_calls = 0
    self._fg_tripped = False
    self._fg_trip_step = None
    self._fg_peak_n = 0.0
    self._webcam = webcam
    self.wrench_log = []
    self._fg_armed = True

  def receive_feedback(self):
    fb = super().receive_feedback()
    if not self._fg_armed:
      return fb

    # First tick defines t0 for the whole run: the CSV, the manifest and
    # every webcam frame are expressed against this instant.
    if self.t0_unix is None:
      self.t0_unix = time.time()
      self.t0_perf = time.perf_counter()
      self.t0_monotonic = time.monotonic()

    step = self._fg_calls
    self._fg_calls += 1
    if self._webcam is not None:
      self._webcam.mark_step(step)

    wrench = np.asarray(fb["tcp_force"], dtype=float)
    f_ext = wrench[:3] - self._fg_bias
    mag = float(np.linalg.norm(f_ext))
    self.wrench_log.append({
        "step": step,
        **{f"tcp_force{i}": float(wrench[i]) for i in range(6)},
        "force_ext_x": float(f_ext[0]),
        "force_ext_y": float(f_ext[1]),
        "force_ext_z": float(f_ext[2]),
        "force_ext_mag": mag,
    })

    # Warmup: the estimator settles over the first few ticks as the arm
    # breaks standstill; tripping there would abort every run.
    if step < self._fg_warmup:
      return fb

    self._fg_peak_n = max(self._fg_peak_n, mag)
    if mag > self._fg_limit_n:
      self._fg_streak += 1
    else:
      self._fg_streak = 0

    if self._fg_streak >= self._fg_consecutive:
      self._fg_tripped = True
      self._fg_trip_step = step
      raise RobotStoppedExternally(
          f"FORCE LIMIT: |F - F_bias| = {mag:.1f} N > {self._fg_limit_n:.1f} N "
          f"for {self._fg_streak} consecutive ticks at step {step}. "
          f"Stopping the arm."
      )
    return fb


# ===========================================================================
# Helpers.
# ===========================================================================


def _resolve_scales(checkpoint_id, policy_path, override):
  """action_scale / gripper_action_scale for this checkpoint.

  CLAUDE.md line 18: these are resolved from the checkpoint's own
  metadata.json, and **deploy scale must equal trained scale**.

  The authoritative record of what a run actually trained with is
  `metadata.json["env_overrides"]`. This deliberately does NOT go through
  `run_gap_protocol.resolve_loop_kwargs()`: that helper reads the gripper
  scale from `gen_dr_ladder._CONFIGS`, whose override dict is EMPTY for every
  velocity-ladder config, so it silently falls back to the env
  `default_config()` value. The ladder checkpoints trained at
  gripper_action_scale=0.01 while the current env default is 0.02 — taking
  the fallback would drive the real Hand-E at twice the trained scale. The
  arm scale is unaffected (env_overrides and the top-level key agree at
  0.04), but it is resolved the same way here for consistency.

  render_sim_rollout.py re-derives both numbers from the same env_overrides
  and hard-fails on any mismatch with the manifest.
  """
  meta_path = os.path.join(policy_path, "metadata.json")
  with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
  overrides = meta.get("env_overrides") or {}

  action_scale = overrides.get("action_scale", meta.get("action_scale"))
  source = "checkpoint metadata.json env_overrides"
  if override is not None:
    action_scale = float(override)
    source = "--action-scale CLI override"
  if action_scale is None:
    raise SystemExit(
        f"{checkpoint_id}: metadata.json records no action_scale (neither "
        f"env_overrides nor top-level), so the trained value cannot be "
        f"recovered. Refusing to guess — a wrong scale silently changes the "
        f"policy's behaviour on hardware."
    )

  gripper_action_scale = overrides.get(
      "gripper_action_scale", meta.get("gripper_action_scale"))
  if gripper_action_scale is None:
    raise SystemExit(
        f"{checkpoint_id}: metadata.json records no gripper_action_scale. "
        f"Refusing to fall back to the env default — the env default and "
        f"the trained value have diverged before (0.02 vs 0.01), and using "
        f"the wrong one drives the real Hand-E at the wrong rate. Re-export "
        f"the checkpoint metadata or pick a checkpoint that records it."
    )

  return float(action_scale), float(gripper_action_scale), source


def _press_enter(prompt, auto_yes):
  if auto_yes:
    print(f"{prompt} [auto-yes]")
    return
  input(f"{prompt} ")


def _confirm(prompt, auto_yes):
  if auto_yes:
    print(f"{prompt} [auto-yes]")
    return True
  return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _build_states_csv(df, wrench_log, out_path):
  """real_states.csv: the shared core columns first, then everything else."""
  if len(df) == 0:
    raise SystemExit("no control steps were logged — nothing to write.")

  out = pd.DataFrame()
  out["step"] = df["step"]
  out["t_rel"] = df["time"]
  for i in range(6):
    out[f"q{i}"] = df[f"q{i}"]
  # The physical per-finger position estimate (metres, 0=open) — the same
  # signal that feeds the obs, and the direct analogue of the sim finger
  # joint that render_sim_rollout.py writes.
  out["finger"] = df["finger_pos_est"]
  for c in ("tcp_x", "tcp_y", "tcp_z", "box_x", "box_y", "box_z",
            "box_qw", "box_qx", "box_qy", "box_qz",
            "target_x", "target_y", "target_z", "box_to_target_dist"):
    out[c] = df[c]
  for i in range(7):
    out[f"action{i}"] = df[f"action{i}"]

  assert list(out.columns) == CORE_COLUMNS, (
      f"core column drift: {list(out.columns)} != {CORE_COLUMNS}")

  # Wrench columns, aligned by step (receive_feedback runs exactly once per
  # logged tick, so the indices correspond 1:1).
  if wrench_log:
    wdf = pd.DataFrame(wrench_log).set_index("step")
    out = out.join(wdf, on="step")

  extras = [c for c in df.columns if c not in ("step", "time")
            and c not in out.columns]
  for c in extras:
    out[c] = df[c]

  out.to_csv(out_path, index=False)
  return out


def _write_frame_index(webcam, t0_unix, out_path):
  rows = webcam.index
  with open(out_path, "w", encoding="utf-8") as f:
    f.write("frame_idx,t_unix,t_rel,ctrl_step\n")
    for idx, t_unix, step in rows:
      f.write(f"{idx},{t_unix:.6f},{t_unix - t0_unix:.6f},{step}\n")
  return len(rows)


# ===========================================================================
# Main.
# ===========================================================================


def main():
  ap = argparse.ArgumentParser(
      description="Record one real UR3e pick episode + webcam for the "
                  "defence side-by-side video.")
  ap.add_argument("--name", required=True,
                  help="short run name; output goes to "
                       "defence/runs/<UTC>_<name>/")
  ap.add_argument("--checkpoint", required=True,
                  help="dir name under evaluation/downloaded_policies/")
  ap.add_argument("--config-id", default="L0_none",
                  help="DR-ladder config, used ONLY to resolve action scales "
                       "the same way the gap protocol does.")
  ap.add_argument("--action-scale", type=float, default=None,
                  help="override the trained action_scale (do not use "
                       "unless you know the training value).")

  # Start pose.
  ap.add_argument("--protocol", default=gap.DEFAULT_PROTOCOL,
                  help="frozen protocol json holding the named start poses.")
  ap.add_argument("--episode-id", type=int, default=None,
                  help="episode id in --protocol; its arm_qpos/finger are "
                       "the named start pose.")
  ap.add_argument("--arm-qpos", type=float, nargs=6, default=None,
                  help="explicit 6-joint start pose (rad), instead of "
                       "--episode-id.")
  ap.add_argument("--finger", type=float, default=0.0,
                  help="commanded per-finger position (m), 0=open. Used with "
                       "--arm-qpos.")
  ap.add_argument("--drop-target", type=float, nargs=3, default=None,
                  help="explicit drop target (m); default is the protocol's "
                       "matched target for this episode.")

  # Hardware.
  # Defaults mirror run_gap_protocol.py's, so a defence take lands on the
  # same hardware configuration as the submitted campaign.
  ap.add_argument("--robot-ip", default="192.168.1.4")
  ap.add_argument("--mocap-server-ip", default="10.1.1.198")
  ap.add_argument("--mocap-rigid-body-name", default="CubeInCube2")
  ap.add_argument("--mocap-stale-s", type=float, default=0.25)
  ap.add_argument("--gripper-port", type=int, default=49999)
  ap.add_argument("--gripper-slave-id", type=int, default=9)
  ap.add_argument("--gripper-speed-pct", type=int, default=100)
  ap.add_argument("--gripper-force-pct", type=int, default=50)
  ap.add_argument("--no-gripper", action="store_true",
                  help="dry run: no-op gripper channel.")

  # Loop.
  ap.add_argument("--control-hz", type=float, default=50.0)
  ap.add_argument("--episode-length", type=int, default=400,
                  help="nominal horizon; the hang watchdog is derived from "
                       "it unless --hang-watchdog-s is given.")
  ap.add_argument("--hang-watchdog-s", type=float, default=None)
  ap.add_argument("--settle-s", type=float, default=1.0)
  ap.add_argument("--lookahead-time", type=float, default=0.1)
  ap.add_argument("--gain", type=int, default=300)
  ap.add_argument("--servoj-a", type=float, default=1.4)
  ap.add_argument("--servoj-v", type=float, default=1.05)
  ap.add_argument("--movej-a", type=float, default=0.4)
  ap.add_argument("--movej-v", type=float, default=0.4)
  ap.add_argument("--gripper-tau", type=float, default=0.0)
  ap.add_argument("--control-law", default="rebase",
                  choices=("rebase", "integrate"))

  # SAFETY — the top open item. See _ForceGuardedUR3.
  ap.add_argument("--force-limit-n", type=float, default=30.0,
                  help="HARD CUTOFF on bias-corrected TCP force ||F-F_bias|| "
                       "(N). Tripping stops the arm and sets "
                       "stop_reason=force_limit. Conservative default 30 N; "
                       "a 3 cm cube pick should never approach it.")
  ap.add_argument("--force-limit-consecutive", type=int, default=2,
                  help="consecutive over-limit ticks required to trip "
                       "(rejects single-sample sensor spikes).")
  ap.add_argument("--force-warmup-steps", type=int, default=5,
                  help="ticks ignored at the start while the wrench "
                       "estimate settles.")
  ap.add_argument("--force-bias-samples", type=int, default=25)

  # Webcam.
  ap.add_argument("--camera-device", default="0",
                  help="cv2 device index or /dev/videoN path.")
  ap.add_argument("--camera-width", type=int, default=1920)
  ap.add_argument("--camera-height", type=int, default=1080)
  ap.add_argument("--camera-fps", type=float, default=30.0)
  ap.add_argument("--preroll-s", type=float, default=1.5,
                  help="seconds of webcam recorded before the loop starts; "
                       "these frames carry negative t_rel and give the "
                       "compositor its alignment handle.")

  ap.add_argument("--out-root", default=os.path.join(_THIS_DIR, "runs"))
  ap.add_argument("--yes", action="store_true",
                  help="skip the interactive confirmations.")
  args = ap.parse_args()

  if args.episode_id is None and args.arm_qpos is None:
    raise SystemExit("give either --episode-id or --arm-qpos.")

  device = args.camera_device
  if device.isdigit():
    device = int(device)

  # ---- checkpoint -------------------------------------------------------
  policy_path = os.path.join(POLICY_ROOT, args.checkpoint)
  meta_path = os.path.join(policy_path, "metadata.json")
  params_path = os.path.join(policy_path, "params.msgpack")
  for p in (meta_path, params_path):
    if not os.path.exists(p):
      raise SystemExit(f"checkpoint incomplete: {p} not found.")
  with open(meta_path, "r", encoding="utf-8") as f:
    ckpt_meta = json.load(f)
  checkpoint_sha256 = gap.sha256_file(params_path)

  action_scale, gripper_action_scale, scale_source = _resolve_scales(
      args.checkpoint, policy_path, args.action_scale)

  ctrl_dt = 1.0 / float(args.control_hz)
  hang_watchdog_s = args.hang_watchdog_s
  if hang_watchdog_s is None:
    hang_watchdog_s = args.episode_length * ctrl_dt * 1.12 + 0.5

  # ---- start pose -------------------------------------------------------
  protocol = None
  episode = None
  if args.arm_qpos is not None:
    commanded_arm_qpos = [float(v) for v in args.arm_qpos]
    commanded_finger = float(args.finger)
    pose_ref = {"source": "adhoc", "protocol_id": None, "poses_sha256": None,
                "episode_id": None, "level": None, "split": None}
  else:
    protocol = load_protocol(args.protocol)
    by_id = {e.episode_id: e for e in protocol.episodes}
    if args.episode_id not in by_id:
      raise SystemExit(
          f"--episode-id {args.episode_id} not in {args.protocol} "
          f"(have {sorted(by_id)}).")
    episode = by_id[args.episode_id]
    commanded_arm_qpos = [float(v) for v in np.asarray(episode.arm_qpos)]
    commanded_finger = float(episode.finger)
    pose_ref = {
        "source": "protocol",
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "episode_id": int(episode.episode_id),
        "level": getattr(episode, "level", None),
        "split": getattr(episode, "split", None),
    }

  # ---- output dir -------------------------------------------------------
  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  run_name = f"{stamp}_{args.name}"
  out_dir = os.path.join(args.out_root, run_name)
  os.makedirs(out_dir, exist_ok=True)
  real_mp4 = os.path.join(out_dir, "real.mp4")

  print("=" * 70)
  print(f"Defence rollout {run_name}")
  print(f"  checkpoint  : {args.checkpoint}")
  print(f"  obs/act     : {ckpt_meta['obs_dim']}D / {ckpt_meta['action_dim']}D")
  print(f"  scales      : action={action_scale} gripper={gripper_action_scale}")
  print(f"                (source: {scale_source})")
  print(f"  start pose  : {np.round(commanded_arm_qpos, 3).tolist()}")
  print(f"  out         : {out_dir}")
  print("!" * 70)
  print(f"  FORCE CUTOFF: {args.force_limit_n:.1f} N on |F - F_bias|, "
        f"{args.force_limit_consecutive} consecutive ticks")
  print("!" * 70)
  if not _confirm("Proceed with these settings?", args.yes):
    raise SystemExit("aborted before touching hardware.")

  robot = _ForceGuardedUR3(args.robot_ip)
  mocap = None
  gripper = None
  webcam = None
  df = pd.DataFrame()
  stats = {}
  stopped_reason = "not_started"

  try:
    # ---- connect --------------------------------------------------------
    print("Connecting arm (needs Remote Control + External Control PLAYING)…")
    robot.connect()
    if not robot.is_connected():
      raise RuntimeError("RTDE failed to connect.")

    print(f"Connecting mocap {args.mocap_server_ip} "
          f"({args.mocap_rigid_body_name})…")
    mocap = VRPNRigidBodyReader(
        args.mocap_server_ip,
        rigid_body_name=args.mocap_rigid_body_name,
        names=[args.mocap_rigid_body_name],
    )
    if not mocap.start(timeout=5.0) or not mocap.wait_for_data(timeout=5.0):
      raise RuntimeError("Mocap failed to connect or never reported data.")

    if args.no_gripper:
      gripper_fn = lambda norm: None  # noqa: E731
      gripper_state_fn = None
      gripper_command = lambda m: None  # noqa: E731
      print("  [dry run] gripper channel is a no-op.")
    else:
      gripper = HandEGripper(
          args.robot_ip, port=args.gripper_port,
          slave_id=args.gripper_slave_id,
          speed_pct=args.gripper_speed_pct,
          force_pct=args.gripper_force_pct,
      )
      gripper.connect()
      gripper.open_gripper()
      gripper_fn = lambda norm: gripper.command(norm * 0.025)  # noqa: E731
      gripper_state_fn = lambda: gripper.read_state()  # noqa: E731
      gripper_command = gripper.command

    robot.load_policy_fn(policy_path=policy_path, deterministic=True)
    robot.init_fk_model(gap.MODEL_PATH)

    if int(ckpt_meta["action_dim"]) != 7:
      raise SystemExit(
          f"action_dim {ckpt_meta['action_dim']} != 7 — this driver assumes "
          f"6 arm joints + 1 Hand-E tendon.")

    # ---- move to the named start pose -----------------------------------
    if not _confirm(
        f"Move arm to start pose "
        f"{np.round(commanded_arm_qpos, 3).tolist()}?", args.yes):
      raise SystemExit("aborted at the move-to-start prompt.")
    robot.send_movej(
        list(commanded_arm_qpos), a=args.movej_a, v=args.movej_v,
        asynchronous=False, textmsg=f"defence {run_name}")

    _press_enter("Place the cube, then press Enter to settle and measure…")

    gripper_command(commanded_finger)
    time.sleep(args.settle_s)

    # ---- measured init --------------------------------------------------
    fb = robot.receive_feedback()
    measured_arm_qpos = [float(v) for v in np.asarray(fb["q"], dtype=float)]

    box_xyz, box_quat = mocap.get_rigid_body_pose()
    if box_xyz is None:
      raise SystemExit(
          "mocap has no pose for the tracked body — check visibility.")
    box_pos_base = robot.mocap_pos_to_base(np.asarray(box_xyz, dtype=float))
    box_quat_base = robot.mocap_quat_to_base(
        np.asarray(box_quat, dtype=float))

    if args.drop_target is not None:
      target_pos = np.asarray(args.drop_target, dtype=float)
    elif protocol is not None:
      target_pos, _ = gap_target.target_for_episode(
          protocol.protocol_id, episode.episode_id, args.config_id,
          box_xy=np.asarray(box_xyz, dtype=float)[:2],
          xml_path=gap.MODEL_PATH,
      )
      target_pos = np.asarray(target_pos, dtype=float)
    else:
      raise SystemExit(
          "--drop-target is required when using --arm-qpos (no protocol to "
          "derive the matched target from).")

    print(f"  measured box : {np.round(box_pos_base, 4).tolist()}")
    print(f"  drop target  : {np.round(target_pos, 4).tolist()}")

    # ---- force bias at standstill ---------------------------------------
    print(f"  measuring force bias ({args.force_bias_samples} samples, "
          f"arm must be still)…")
    force_bias, bias_spread = robot.measure_force_bias(
        n_samples=args.force_bias_samples)
    print(f"  F_bias = {np.round(force_bias, 2).tolist()} N "
          f"(standstill spread {bias_spread:.2f} N)")
    if bias_spread > args.force_limit_n * 0.5:
      raise SystemExit(
          f"standstill wrench noise ({bias_spread:.1f} N) is more than half "
          f"the {args.force_limit_n:.1f} N limit — the cutoff would be "
          f"unreliable. Check the arm is truly still and the payload is "
          f"configured, or raise --force-limit-n deliberately.")

    # ---- webcam pre-roll -------------------------------------------------
    webcam = _WebcamWorker(
        device, args.camera_width, args.camera_height, args.camera_fps,
        real_mp4)
    webcam.start()
    print(f"  webcam rolling; {args.preroll_s:.1f}s pre-roll…")
    time.sleep(args.preroll_s)

    robot.arm_force_guard(
        limit_n=args.force_limit_n,
        bias=force_bias,
        consecutive=args.force_limit_consecutive,
        warmup=args.force_warmup_steps,
        webcam=webcam,
    )

    # ---- the episode ----------------------------------------------------
    print(f"  running episode (watchdog {hang_watchdog_s:.1f}s)…")
    df, stats = robot.run_policy_loop(
        drop_target=np.asarray(target_pos, dtype=np.float32),
        mocap_reader=mocap,
        control_hz=args.control_hz,
        timeout_s=hang_watchdog_s,
        action_scale=action_scale,
        gripper_action_scale=gripper_action_scale,
        lookahead_time=args.lookahead_time,
        gain=args.gain,
        servoj_a=args.servoj_a,
        servoj_v=args.servoj_v,
        alpha=1.0,
        gripper_fn=gripper_fn,
        gripper_state_fn=gripper_state_fn,
        gripper_tau=args.gripper_tau,
        gripper_max_rate=float("inf"),
        control_law=args.control_law,
        use_fk_tcp=True,
        reach_tol=None,  # never terminate early on reach
        mocap_stale_s=args.mocap_stale_s,
        debug_print=True,
    )
    stopped_reason = stats.get("stopped_reason", "unknown")

  finally:
    if webcam is not None:
      webcam.stop()
    if robot is not None:
      try:
        robot.disconnect()
      except Exception:
        pass
    if mocap is not None:
      try:
        mocap.stop()
      except Exception:
        pass

  # ---- the force trip outranks the generic external_stop label ----------
  if robot._fg_tripped:
    stopped_reason = "force_limit"
    print("!" * 70)
    print(f"FORCE LIMIT TRIPPED at step {robot._fg_trip_step} — arm stopped.")
    print("!" * 70)

  if robot.t0_unix is None:
    raise SystemExit(
        "the control loop never completed a tick — no t0, nothing to sync. "
        f"(stop reason: {stopped_reason})")

  # ---- artifacts --------------------------------------------------------
  states_csv = os.path.join(out_dir, "real_states.csv")
  out_df = _build_states_csv(df, robot.wrench_log, states_csv)
  frame_csv = os.path.join(out_dir, "frame_index.csv")
  n_frames = _write_frame_index(webcam, robot.t0_unix, frame_csv)
  n_steps = int(len(out_df))

  manifest = {
      "schema_version": SCHEMA_VERSION,
      "run_name": run_name,
      "t0_unix": float(robot.t0_unix),
      "t0_iso": datetime.fromtimestamp(
          robot.t0_unix, timezone.utc).isoformat(timespec="milliseconds"),
      "t0_monotonic": float(robot.t0_monotonic),
      "init": {
          "arm_qpos": measured_arm_qpos,
          "commanded_arm_qpos": commanded_arm_qpos,
          "finger": commanded_finger,
          "box_pos": [float(v) for v in box_pos_base],
          "box_quat_wxyz": [float(v) for v in box_quat_base],
          "target_pos": [float(v) for v in target_pos],
          "cube_half_extents": None,
          "lifter_top_height": None,
          "lifter_tilt_rp": None,
          "settle_s": float(args.settle_s),
      },
      "pose_ref": pose_ref,
      "policy": {
          "checkpoint_id": args.checkpoint,
          "checkpoint_sha256": checkpoint_sha256,
          "obs_dim": int(ckpt_meta["obs_dim"]),
          "action_dim": int(ckpt_meta["action_dim"]),
          "policy_hidden_layer_sizes": list(
              ckpt_meta["network_factory"]["policy_hidden_layer_sizes"]),
      },
      "control": {
          "ctrl_dt": ctrl_dt,
          "action_scale": float(action_scale),
          "gripper_action_scale": float(gripper_action_scale),
          "scale_source": scale_source,
          "gripper_convention": GRIPPER_CONVENTION,
          # The number of steps the sim must reproduce for a frame-matched
          # side-by-side: the actual length of THIS episode, not the nominal
          # horizon (which is kept below for provenance).
          "episode_length": n_steps,
          "nominal_horizon": int(args.episode_length),
          "control_hz_requested": float(args.control_hz),
          "achieved_hz": float(
              stats.get("true_inferred_frequency_hz") or float("nan")),
          "control_law": args.control_law,
      },
      "result": {
          "stop_reason": stopped_reason,
          "stop_step": int(robot._fg_trip_step)
                       if robot._fg_trip_step is not None else n_steps - 1,
          "n_steps": n_steps,
          "force_limit_n": float(args.force_limit_n),
          "force_bias_n": [float(v) for v in force_bias],
          "peak_force_n": float(robot._fg_peak_n),
          "overrun_count": int(stats.get("num_overruns") or 0),
      },
      "env": {
          "env_name": ckpt_meta.get("env_name", "UR3Pick"),
          "xml_path": os.path.relpath(gap.MODEL_PATH, REPO_ROOT),
          "git_sha": gap.git_sha(),
          "branch": "addvelocity",
      },
      "video": {
          "real_mp4": os.path.relpath(real_mp4, out_dir),
          "webcam_fps_nominal": float(args.camera_fps),
          "n_webcam_frames": int(n_frames),
          "frame_index_csv": os.path.relpath(frame_csv, out_dir),
      },
  }
  with open(os.path.join(out_dir, "manifest.json"), "w",
            encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

  print("\n" + "=" * 70)
  print(f"stop_reason : {stopped_reason}")
  print(f"steps       : {n_steps}   peak |F-F_bias|: "
        f"{robot._fg_peak_n:.1f} N / {args.force_limit_n:.1f} N limit")
  print(f"webcam      : {n_frames} frames -> {real_mp4}")
  print(f"artifacts   : {out_dir}")
  print("Next: git-sync, then on the MacBook run")
  print(f"  python defence/render_sim_rollout.py --run-dir "
        f"defence/runs/{run_name}")
  print("=" * 70)


if __name__ == "__main__":
  main()
