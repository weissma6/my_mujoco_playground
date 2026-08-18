"""Record ONE real UR3e pick episode for the VT2 defence side-by-side video.

Runs on the LINUX lab machine (UR3e + Robotiq Hand-E + Nokov/VRPN mocap).
Drives one episode from the task_home start pose to the matched drop target,
then writes the two artifacts `render_sim_rollout.py` needs:

    defence/runs/<UTC>_<name>/manifest.json      full sim-rollout determinant
    defence/runs/<UTC>_<name>/real_states.csv    per-control-step log

NO VIDEO IS RECORDED HERE. Filming is external (Sony Alpha, started by hand
during the pre-roll countdown); the manifest's `video` block is emitted
all-null with source="external" purely to satisfy the sim mirror's schema.
The footage is aligned by hand — see the SYNC RULE printed at the end of a run.

UNATTENDED BY DESIGN: there is no `input()` anywhere. The operator's approval
is pressing Play on the pendant, which `connect()` already blocks on; from
there the run proceeds through move -> settle -> episode -> drop on its own.
The only pauses are two fixed countdowns (`--setup-wait-s`, `--preroll-s`).

THIN DRIVER, same posture as run_gap_protocol.py: the 50 Hz control loop is
NOT duplicated here. `UR3RealRobotPick.run_policy_loop` still owns RTDE
receive/servoJ/the Hand-E worker thread/logging. This script adds only what
that loop has no reason to know about: the force cutoff and the manifest.

Like the gap protocol, the episode ENDS WITH THE D22 DROP: once the policy has
carried the cube through the full horizon, the gripper simply opens and the
cube falls into the taped square. No scripted arm motion — the policy did all
the carrying — and `place_error` is scored from a post-drop mocap read, reusing
`evaluation.gap_metrics.place_error`. See the "D22: THE DROP" block in `main`.

Two things are grafted on without touching any existing file:

1. FORCE CUTOFF (`_ForceGuardedUR3`) — `run_policy_loop` has no force limit
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

2. FRAME (see the target block in `main`) — the drop target is computed from
   the BASE-frame box position, not the raw mocap one. This is a deliberate
   divergence from run_gap_protocol.py, which is left as-is so the already-run
   campaign stays reproducible; the reasoning is spelled out at the call site.

Usage (from this folder — ../.. paths depend on it, per CLAUDE.md).
EVERY argument has a default, so this alone is a complete run: it uses
L2_pos_cube (the campaign's best policy — see _BEST_CONFIG_EVIDENCE below),
resolves its checkpoint from the gap protocol's own policy map, and starts the
arm at the scene's task_home keyframe:

    python record_real_rollout.py

    # start from the collected eval pose nearest task_home instead, which
    # keeps full frozen-protocol provenance for the pose AND the target:
    python record_real_rollout.py --start-pose nearest_home
"""

import argparse
import json
import os
import sys
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
from evaluation.gap_metrics import place_error  # noqa: E402 -- D22 drop metric
from policy_downloader import (  # noqa: E402
    default_policy_dir,
    download_policy,
    policy_exists_locally,
)

# Imported, never edited (repo rule): the gap protocol owns these helpers and
# the model path, and reusing them is what keeps this run comparable with the
# submitted results.
import run_gap_protocol as gap  # noqa: E402

SCHEMA_VERSION = "defence/1"
POLICY_ROOT = os.path.join(REPO_ROOT, "evaluation", "downloaded_policies")
GRIPPER_CONVENTION = "0=open,0.025=closed"

# D22 drop square, mirrored from evaluation/gap_metrics.py:963-964 -- the
# 15x15 cm square taped on the table at base-frame (0.212, 0.212). The Z is
# box_z_anchor (0.115, gap_target.scene_constants) + the 0.05 lift the
# L0_deterministic profile draws, i.e. the same 0.165 the L0_none run
# 20260814T114737Z targeted. Pinning it makes the target identical in every
# take instead of redrawn per episode.
#
# Why pin at all: target_for_episode() routes L1-L5 through the "default"
# profile, which for episode 0 draws base-frame (0.1967, -0.1656, 0.1413) --
# 0.374 m from the L0 target and place_error 0.378 m OUTSIDE the taped
# square. That is why 7 of the 8 runs on 2026-08-14 missed the tape. The
# sampler itself is deliberately NOT changed here (see the plan's Out of
# scope); the target is pinned instead.
DROP_SQUARE_TARGET = (0.212, 0.212, 0.165)


# ###########################################################################
# ###                                                                     ###
# ###   >>>>>>   SELECT WHICH POLICY TO RUN -- EDIT HERE   <<<<<<         ###
# ###                                                                     ###
# ###########################################################################
#
# Pick ONE rung. This is the default for --config-id; the CLI still wins if
# you pass --config-id explicitly.
#
#   "L0_none"            -> L0_none_vel_s1_20260729_104930_2201
#   "L1_pos"             -> L1_pos_vel_s1_20260729_112044_2201
#   "L2_pos_cube"        -> L2_pos_cube_vel_s1_20260729_112246_2201   <-- best
#   "L3_pos_cube_robot"  -> L3_pos_cube_robot_vel_s1_20260729_115408_2201
#   "L4_full"            -> L4_full_vel_s1_20260729_122736_2201
#
# All five are addvelocity checkpoints (33-D obs). The run_id is resolved from
# robots/UR3e/gap_protocol_policy_map.json -- see _BEST_CONFIG_EVIDENCE below
# for why L2 is the default (only config that actually lifts the cube: 81 mm
# median rise, 85% completion, 116 mm final distance).
#
# For the defence film, L0/L1 never leave the table -- picking them records
# the arm failing to grasp.

SELECT_CONFIG_ID = "L1_pos"

# Pin an EXACT checkpoint, bypassing the policy map entirely.
#   None = resolve from gap_protocol_policy_map.json via SELECT_CONFIG_ID
#   "L2_pos_cube_vel_s1_20260729_112246_2201" = force this exact one
# Note the target profile still comes from SELECT_CONFIG_ID, so if you pin a
# checkpoint from a different rung, set SELECT_CONFIG_ID to match it.

SELECT_CHECKPOINT = "Snappy2_as04_ar70_g01_s1_20260817_202921_2201"

# NOTE the SELECT_CONFIG_ID above is COSMETIC on this pinned path, and the
# snappy2/L1_pos mismatch is deliberate rather than an oversight:
#   * with SELECT_CHECKPOINT set, `if args.checkpoint is None` below is False,
#     so gap.resolve_policy_run_id() -- the only reader of
#     gap_protocol_policy_map.json -- never runs, and
#   * with an explicit drop target (now the default, DROP_SQUARE_TARGET),
#     gap_target.target_for_episode() never runs either.
# CONFIG_TO_PROFILE is therefore never consulted and cannot reject "snappy2"
# as an unmapped rung. --config-id survives only as a manifest/label field.

# ###########################################################################
# ###   END OF THE EDIT-HERE BLOCK                                        ###
# ###########################################################################

# Why --config-id defaults to L2_pos_cube: it is the best policy of the five in
# the REAL gap-protocol campaign, aggregated over the 144 velocity-ladder runs
# under robots/UR3e/real_robot_results/gap_protocol/ (the older *_s0/_s1/_s2
# 2026-07-17 runs are a different obs size and excluded).
#
#   config              n  compl%  final_d  min_d  improve  lift  grasp_end
#   L0_none            36    75%    358mm   358mm     0mm    0mm      0.00
#   L1_pos             27    63%    248mm   248mm     0mm    0mm      0.24
#   L2_pos_cube        27    85%    116mm   116mm   107mm   81mm      0.62
#   L3_pos_cube_robot  27    48%    194mm   194mm    33mm    0mm      0.30
#   L4_full            27    67%    233mm   229mm    24mm    0mm      0.23
#
# L2 wins on every column: the highest completion rate, less than half the
# final box-to-target distance of any other config, and it is the ONLY config
# that actually lifts the cube on hardware (81 mm median rise, gripper still
# closed at the end of 62% of runs). L0/L1 never leave the table -- zero lift,
# zero net improvement -- so a defence take on them would film the arm failing
# to pick.
#
# These columns are read from each run's ur3_pick_states.csv, NOT from the
# metas' place_error_m: those were scored against a raw-mocap landed position
# (run_gap_protocol.py:704, the frame issue this file fixes for itself), which
# is why every campaign run reports ~600 mm and in_square=0. box_to_target_dist
# and box_z come from run_policy_loop, which does convert to base frame, so
# they are sound -- and every config was scored identically, so the ranking
# holds regardless.
_BEST_CONFIG_EVIDENCE = "campaign aggregate, 2026-08-14; see comment above"

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
# Force-guarded robot.
# ===========================================================================


class _ForceGuardedUR3(UR3RealRobotPick):
  """UR3RealRobotPick + a hard TCP-force cutoff.

  `receive_feedback()` is stage 1 of every `run_policy_loop` tick and runs
  inside that loop's try block, which makes it the correct and only place to
  intervene without editing the loop. Overriding it gives us, per tick and in
  order: the wall-clock t0 anchor, the wrench log, and the cutoff itself.
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

  def arm_force_guard(self, limit_n, bias, consecutive, warmup):
    self._fg_limit_n = float(limit_n)
    self._fg_bias = np.asarray(bias, dtype=float).reshape(3)
    self._fg_consecutive = int(consecutive)
    self._fg_warmup = int(warmup)
    self._fg_streak = 0
    self._fg_calls = 0
    self._fg_tripped = False
    self._fg_trip_step = None
    self._fg_peak_n = 0.0
    self.wrench_log = []
    self._fg_armed = True

  def receive_feedback(self):
    fb = super().receive_feedback()
    if not self._fg_armed:
      return fb

    # First tick defines t0 for the whole run: the CSV and the manifest are
    # expressed against this instant, and it is the moment the external
    # camera's footage has to be cut to (see the SYNC RULE printed at the end).
    if self.t0_unix is None:
      self.t0_unix = time.time()
      self.t0_perf = time.perf_counter()
      self.t0_monotonic = time.monotonic()

    step = self._fg_calls
    self._fg_calls += 1
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


def _task_home_qpos(xml_path):
  """The scene XML's `task_home` keyframe qpos (6 arm joints + 2 fingers).

  Read from the model rather than hardcoded: task_home is the anchor
  ur3_pick.py and gap_target.py both key off (gap_target.scene_constants reads
  its box Z as box_z_anchor), so a scene edit must move this pose too rather
  than silently disagreeing with a copy pasted in here.
  """
  import mujoco  # local: keeps module import cheap for --help
  model = mujoco.MjModel.from_xml_path(xml_path)
  try:
    qpos = np.array(model.key("task_home").qpos, dtype=float)
  except KeyError:
    raise SystemExit(
        f"{xml_path} has no 'task_home' keyframe -- cannot use "
        f"--start-pose task_home. Pass --arm-qpos explicitly instead.")
  if qpos.size < 8:
    raise SystemExit(
        f"task_home qpos has {qpos.size} entries, expected >= 8 "
        f"(6 arm joints + 2 fingers).")
  return qpos


def _countdown(seconds, banner):
  """A timed, NON-BLOCKING-on-input pause with a visible countdown.

  This script has no `input()` anywhere: the operator's approval is pressing
  Play on the pendant (Remote Control + External Control PLAYING), which the
  arm connect already blocks on. Once that is done the run proceeds by itself
  to the end, so it can be started and left alone. These countdowns are the
  only pauses, and they exist to give the operator a fixed, predictable window
  to place the cube / start the external camera / stand clear.
  """
  seconds = float(seconds)
  if seconds <= 0:
    return
  print("!" * 70)
  print(f"  {banner}")
  print("!" * 70)
  whole = int(seconds)
  for s in range(whole, 0, -1):
    print(f"  {s}...", flush=True)
    time.sleep(1.0)
  frac = seconds - whole
  if frac > 0:
    time.sleep(frac)


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


# ===========================================================================
# Main.
# ===========================================================================


def main():
  ap = argparse.ArgumentParser(
      description="Run one real UR3e pick episode (policy rollout + drop) and "
                  "write the manifest the defence sim mirror replays. Video "
                  "is filmed externally; this script records no video.")
  ap.add_argument("--name", default=None,
                  help="short run name; output goes to "
                       "defence/runs/<UTC>_<name>/. Default: --config-id.")
  ap.add_argument("--checkpoint", default=SELECT_CHECKPOINT,
                  help="dir name under evaluation/downloaded_policies/. "
                       "Default: SELECT_CHECKPOINT at the top of this file, or "
                       "if that is None, resolved from --config-id via "
                       "robots/UR3e/gap_protocol_policy_map.json, exactly as "
                       "the gap protocol campaign resolves it (and downloaded "
                       "from W&B if not cached yet).")
  ap.add_argument("--config-id", default=SELECT_CONFIG_ID,
                  help="DR-ladder config. Selects the checkpoint (via the gap "
                       "protocol's policy map) and the target profile. "
                       f"Default {SELECT_CONFIG_ID} (SELECT_CONFIG_ID at the "
                       "top of this file); L2_pos_cube is the best performer "
                       "in the real gap-protocol campaign, see "
                       "_BEST_CONFIG_EVIDENCE.")
  ap.add_argument("--action-scale", type=float, default=None,
                  help="override the trained action_scale (do not use "
                       "unless you know the training value).")

  # Start pose.
  ap.add_argument("--protocol", default=gap.DEFAULT_PROTOCOL,
                  help="frozen protocol json holding the named start poses.")
  ap.add_argument("--episode-id", type=int, default=0,
                  help="episode id in --protocol; its arm_qpos/finger are "
                       "the named start pose. Default 0, so a bare "
                       "invocation runs the protocol's first episode.")
  ap.add_argument("--start-pose",
                  choices=("task_home", "nearest_home", "episode"),
                  default="task_home",
                  help="where the ARM start pose comes from. 'task_home' "
                       "(default): the scene XML's task_home keyframe -- the "
                       "standard home the policy trains from. 'nearest_home': "
                       "the collected eval episode whose frozen pose is "
                       "closest to task_home (keeps full protocol provenance; "
                       "also switches --episode-id to that episode, so the "
                       "target matches the pose). 'episode': the frozen pose "
                       "of --episode-id exactly. With task_home the DROP "
                       "TARGET still comes from --episode-id.")
  ap.add_argument("--arm-qpos", type=float, nargs=6, default=None,
                  help="explicit 6-joint start pose (rad); overrides "
                       "--start-pose. May be combined with --episode-id, in "
                       "which case the episode still supplies the target.")
  ap.add_argument("--finger", type=float, default=None,
                  help="commanded per-finger position (m), 0=open. Default: "
                       "whatever the chosen start pose carries (the episode's "
                       "finger, or task_home's 0.0125).")
  ap.add_argument("--drop-target", type=float, nargs=3,
                  default=DROP_SQUARE_TARGET,
                  help="explicit drop target in BASE (MuJoCo/sim) frame "
                       "metres -- NOT mocap world. It is handed straight to "
                       "run_policy_loop, which compares it against a "
                       "base-converted box position. Default is "
                       f"DROP_SQUARE_TARGET {DROP_SQUARE_TARGET}, the centre "
                       "of the taped square, so every take aims at the same "
                       "physical point.")
  ap.add_argument("--drop-target-from-protocol", action="store_true",
                  help="restore the OLD behaviour: redraw the target per "
                       "episode via gap_target.target_for_episode instead of "
                       "using the pinned DROP_SQUARE_TARGET. Requires "
                       "--episode-id. Mutually exclusive with an explicit "
                       "--drop-target.")

  # Hardware.
  # Defaults mirror run_gap_protocol.py's, so a defence take lands on the
  # same hardware configuration as the submitted campaign.
  ap.add_argument("--robot-ip", default="192.168.1.4")
  ap.add_argument("--mocap-server-ip", default="10.1.1.198")
  ap.add_argument("--mocap-rigid-body-name", default="CubeInCube2")
  ap.add_argument("--mocap-stale-s", type=float, default=0.25)
  ap.add_argument("--gripper-port", type=int, default=49999)
  ap.add_argument("--gripper-slave-id", type=int, default=9)
  # 60/80 mirror run_gap_protocol.py's campaign values.
  ap.add_argument("--gripper-speed-pct", type=int, default=60)
  ap.add_argument("--gripper-force-pct", type=int, default=80)
  ap.add_argument("--no-gripper", action="store_true",
                  help="dry run: no-op gripper channel.")

  # Loop.
  ap.add_argument("--control-hz", type=float, default=50.0)
  ap.add_argument("--episode-length", type=int, default=400,
                  help="HARD horizon: forwarded to run_policy_loop as "
                       "max_steps, so the loop stops after exactly this many "
                       "control steps with stop_reason 'horizon'. Also sizes "
                       "the hang watchdog unless --hang-watchdog-s is given. "
                       "Default 400 = the trained episode_length, which is "
                       "what makes the sim replay frame-matched. Was "
                       "previously None (= the protocol's own horizon), which "
                       "capped nothing and let the loop run to timeout.")
  ap.add_argument("--hang-watchdog-s", type=float, default=None)
  ap.add_argument("--settle-s", type=float, default=1.0)
  ap.add_argument("--lookahead-time", type=float, default=0.1)
  ap.add_argument("--gain", type=int, default=300)
  # servoJ + gripper-filter values mirror run_gap_protocol.py's campaign
  # settings, so a defence take is driven exactly like a scored run.
  ap.add_argument("--servoj-a", type=float, default=0.3)
  ap.add_argument("--servoj-v", type=float, default=1.0)
  ap.add_argument("--movej-a", type=float, default=0.4)
  ap.add_argument("--movej-v", type=float, default=0.4)
  ap.add_argument("--gripper-tau", type=float, default=0.1)
  ap.add_argument("--box-z-offset", type=float, default=0.0,
                  help="metres added to the mocap-derived box Z before it "
                       "feeds the policy obs (deploy calibration nudge; "
                       "recorded in the manifest). 0.0 = off.")
  ap.add_argument("--control-law", default="rebase",
                  choices=("rebase", "integrate"))

  # D22 drop -- mirrors run_gap_protocol.py's post-episode release.
  ap.add_argument("--no-drop", dest="drop", action="store_false",
                  help="skip the D22 drop entirely; the cube stays held at "
                       "the end of the episode.")
  ap.set_defaults(drop=True)
  ap.add_argument("--drop-on-abort", action="store_true",
                  help="drop even when the episode did not run its full "
                       "horizon. Off by default, matching the gap protocol: "
                       "an aborted episode may not have the cube grasped at "
                       "all, so 'where did it land' is meaningless.")
  ap.add_argument("--drop-settle-s", type=float, default=1.5,
                  help="pause after the gripper-open drop, before the "
                       "post-drop mocap read -- lets the fall and any bounce "
                       "settle so place_error is not scored against a cube "
                       "still in mid-air.")

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

  # Timed windows. There are NO input() prompts anywhere in this script --
  # these countdowns are the only pauses. See _countdown.
  ap.add_argument("--setup-wait-s", type=float, default=6.0,
                  help="window after the arm reaches the start pose: place "
                       "the cube and stand clear. The only chance to set the "
                       "scene, since nothing blocks on a keypress -- raise it "
                       "if 6s is tight, the run measures when it expires.")
  ap.add_argument("--preroll-s", type=float, default=4.0,
                  help="window immediately before the episode starts: hit "
                       "record on the external camera.")

  ap.add_argument("--out-root", default=os.path.join(_THIS_DIR, "runs"))
  args = ap.parse_args()

  # --drop-target now DEFAULTS to the pinned DROP_SQUARE_TARGET, so the
  # protocol draw is opt-in. `is DROP_SQUARE_TARGET` is an identity test on
  # the argparse default object: argparse hands back that very tuple when the
  # flag is absent, and a fresh list when it is passed, so this distinguishes
  # "not given" from "given, and happens to equal the default".
  if args.drop_target_from_protocol:
    if args.drop_target is not DROP_SQUARE_TARGET:
      raise SystemExit(
          "--drop-target and --drop-target-from-protocol are mutually "
          "exclusive: the first pins a target, the second redraws one.")
    args.drop_target = None   # fall through to the target_for_episode path

  # --episode-id is what fixes the DROP TARGET; an explicit --drop-target
  # replaces it. The arm pose is chosen separately (--start-pose/--arm-qpos),
  # so giving one does not switch the other off.
  if args.episode_id is None and args.drop_target is None:
    raise SystemExit(
        "give --episode-id (the target is derived from its frozen draws) or "
        "an explicit --drop-target.")

  # ---- checkpoint -------------------------------------------------------
  # No --checkpoint: resolve the SAME way the gap protocol campaign does, so
  # `--config-id L0_none` alone is a complete invocation. resolve_policy_run_id
  # reads robots/UR3e/gap_protocol_policy_map.json and raises a SystemExit
  # naming the file if the config is not mapped.
  if args.checkpoint is None:
    args.checkpoint = gap.resolve_policy_run_id(args.config_id, None)
    print(f"  [checkpoint] {args.config_id} -> {args.checkpoint} "
          f"(gap_protocol_policy_map.json)")
    if not policy_exists_locally(args.checkpoint):
      print(f"  [checkpoint] not cached locally; downloading from W&B ...")
      download_policy(
          args.checkpoint, out_dir=default_policy_dir(args.checkpoint),
          entity=gap.WANDB_ENTITY, project=gap.WANDB_PROJECT,
      )
  if args.name is None:
    args.name = args.config_id

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

  # ---- start pose -------------------------------------------------------
  # The ARM start pose and the DROP TARGET are chosen independently. The
  # target needs a protocol episode (its frozen dphi/side/r draws); the arm
  # pose does not have to come from the same place, and for a defence take it
  # is often nicer to start from the familiar task_home. So: load the protocol
  # whenever an --episode-id is given (that fixes the target), then let
  # --start-pose / --arm-qpos override the arm pose on top of it.
  protocol = None
  episode = None
  if args.episode_id is not None:
    protocol = load_protocol(args.protocol)
    by_id = {e.episode_id: e for e in protocol.episodes}
    if args.episode_id not in by_id:
      raise SystemExit(
          f"--episode-id {args.episode_id} not in {args.protocol} "
          f"(have {sorted(by_id)}).")
    episode = by_id[args.episode_id]

  if args.arm_qpos is None and args.start_pose == "nearest_home":
    # The collected eval pose closest to task_home. Switches the EPISODE too,
    # not just the pose: the target is derived from the episode's frozen
    # draws, so taking a pose from one episode and a target from another would
    # score the run against a target that was never matched to it.
    if protocol is None:
      raise SystemExit(
          "--start-pose nearest_home needs a protocol (drop --drop-target, or "
          "pass an --episode-id) to pick the nearest collected pose from.")
    home_q = _task_home_qpos(gap.MODEL_PATH)[:6]
    ranked = sorted(
        protocol.episodes,
        key=lambda e: float(np.linalg.norm(
            np.asarray(e.arm_qpos, dtype=float) - home_q)))
    episode = ranked[0]
    dist = float(np.linalg.norm(
        np.asarray(episode.arm_qpos, dtype=float) - home_q))
    print(f"  [start-pose] nearest_home -> episode {episode.episode_id} "
          f"(marker {episode.box_marker}), {dist:.3f} rad from task_home")
    args.episode_id = int(episode.episode_id)

  if args.arm_qpos is not None:
    commanded_arm_qpos = [float(v) for v in args.arm_qpos]
    pose_finger = 0.0
    arm_pose_source = "adhoc"
  elif args.start_pose == "task_home":
    # Read from the scene XML's own keyframe -- the single source of truth
    # that ur3_pick.py and gap_target.py also anchor on. Never hardcoded here.
    key_qpos = _task_home_qpos(gap.MODEL_PATH)
    commanded_arm_qpos = [float(v) for v in key_qpos[:6]]
    pose_finger = float(key_qpos[6])  # per-finger, 0=open
    arm_pose_source = "task_home"
  else:
    if episode is None:
      raise SystemExit(
          "--start-pose episode needs an --episode-id (or give --arm-qpos / "
          "--start-pose task_home).")
    commanded_arm_qpos = [float(v) for v in np.asarray(episode.arm_qpos)]
    pose_finger = float(episode.finger)
    arm_pose_source = "protocol"

  # --finger overrides whatever the chosen pose carried.
  commanded_finger = (pose_finger if args.finger is None
                      else float(args.finger))

  pose_ref = {
      # Kept for the manifest contract: where the ARM pose came from.
      "source": arm_pose_source,
      "arm_pose_source": arm_pose_source,
      "protocol_id": protocol.protocol_id if protocol is not None else None,
      "poses_sha256": protocol.poses_sha256 if protocol is not None else None,
      "episode_id": int(episode.episode_id) if episode is not None else None,
      "level": getattr(episode, "level", None) if episode is not None else None,
      "split": getattr(episode, "split", None) if episode is not None else None,
  }
  print(f"  start pose source: {arm_pose_source}"
        + (f" (target still from episode {episode.episode_id})"
           if episode is not None and arm_pose_source != "protocol" else ""))

  # ---- horizon + watchdog ----------------------------------------------
  # Default to the PROTOCOL's own horizon, same as run_gap_protocol.py, so a
  # defence take covers exactly as many steps as a scored campaign run. The
  # --arm-qpos path has no protocol to read, hence the 400 fallback.
  episode_length = args.episode_length
  if episode_length is None:
    episode_length = int(protocol.horizon) if protocol is not None else 400
  hang_watchdog_s = args.hang_watchdog_s
  if hang_watchdog_s is None:
    hang_watchdog_s = episode_length * ctrl_dt * 1.12 + 0.5

  # ---- output dir -------------------------------------------------------
  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  run_name = f"{stamp}_{args.name}"
  out_dir = os.path.join(args.out_root, run_name)
  os.makedirs(out_dir, exist_ok=True)

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
  print("  UNATTENDED: no prompts from here on. Your approval is Play on the "
        "pendant — connect() blocks until then, and the run then goes "
        "through to the drop by itself.")

  robot = _ForceGuardedUR3(args.robot_ip)
  mocap = None
  gripper = None
  df = pd.DataFrame()
  stats = {}
  stopped_reason = "not_started"
  # D22 drop outcome. All stay None unless a drop was actually attempted AND
  # mocap saw where the cube landed -- never fabricated.
  drop_attempted = False
  landed_box_xy = None
  place_error_m = None
  place_in_square = None

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
    # asynchronous=False blocks until the controller reports convergence,
    # which IS the standstill check.
    print(f"  moving to start pose "
          f"{np.round(commanded_arm_qpos, 3).tolist()} ({arm_pose_source})…")
    robot.send_movej(
        list(commanded_arm_qpos), a=args.movej_a, v=args.movej_v,
        asynchronous=False, textmsg=f"defence {run_name}")

    _countdown(args.setup_wait_s,
               "PLACE THE CUBE on the marker, then stand clear.")

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
      # BASE frame, deliberately -- gap_target.target_for_episode documents
      # box_xy as base frame (gap_target.py:332) and builds the target as
      # base_xy + r*[cos φ, sin φ] off arctan2(box_xy - base_xy)
      # (gap_target.py:311-313), where base_xy comes from the scene XML. The
      # live mocap calibration is a ~180 deg Z rotation plus a ~0.37 m offset,
      # so feeding raw mocap XY here does not shift the bearing slightly -- it
      # replaces it: a cube at base (0.355, -0.009) sits at raw mocap
      # (-0.013, -0.122), i.e. a bearing of -1.5 deg read as -96 deg.
      #
      # NOTE run_gap_protocol.py:616 passes the RAW mocap XY here. That is a
      # known issue, left untouched ON PURPOSE so the already-run campaign
      # stays internally consistent and reproducible -- this divergence is
      # deliberate, not an oversight in one of the two files. It also matters
      # more here: render_sim_rollout.py resets the sim box to the base-frame
      # init.box_pos below and the mocap target to init.target_pos, so a
      # mixed-frame manifest can never produce a matching sim mirror.
      target_pos, _ = gap_target.target_for_episode(
          protocol.protocol_id, episode.episode_id, args.config_id,
          box_xy=np.asarray(box_pos_base, dtype=float)[:2],
          xml_path=gap.MODEL_PATH,
      )
      target_pos = np.asarray(target_pos, dtype=float)
    else:
      raise SystemExit(
          "no --episode-id, so there are no frozen draws to derive the "
          "matched target from -- pass an explicit --drop-target (base "
          "frame).")

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

    # ---- external camera pre-roll ---------------------------------------
    _countdown(args.preroll_s,
               "START THE EXTERNAL CAMERA NOW — the episode begins when this "
               "countdown ends.")

    robot.arm_force_guard(
        limit_n=args.force_limit_n,
        bias=force_bias,
        consecutive=args.force_limit_consecutive,
        warmup=args.force_warmup_steps,
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
        dwell_time_s=0.0,
        # The horizon is now ENFORCED, not just watchdog-sized: without this
        # the loop ran to timeout_s and logged ~473 steps against a policy
        # trained at 400, so the sim replay could never be frame-matched.
        max_steps=int(episode_length),
        mocap_stale_s=args.mocap_stale_s,
        box_z_offset=args.box_z_offset,
        debug_print=True,
    )
    stopped_reason = stats.get("stopped_reason", "unknown")

    # ---- D22: THE DROP ---------------------------------------------------
    # Mirrors run_gap_protocol.py's post-episode release. The policy has just
    # carried the cube through the full horizon to the eval target, which
    # sits inside the trained distribution; opening the gripper lets the cube
    # fall the remaining few centimetres into the taped square. ZERO scripted
    # arm motion -- the policy did all the carrying.
    #
    # This runs INSIDE the try, before the teardown below: run_policy_loop's
    # own finally has already called servoStop(), so the arm is halted but
    # still connected, and mocap/gripper are both still live.
    #
    # Gated on completion exactly as the gap protocol gates it: an aborted
    # episode may not have the cube grasped at all, so "where did it land" is
    # meaningless and the three fields stay None rather than being fabricated.
    complete = (not robot._fg_tripped) and len(df) >= episode_length
    if not args.drop:
      print("  D22: drop disabled (--no-drop).")
    elif args.no_gripper:
      print("  D22: no drop -- gripper channel is a no-op (--no-gripper).")
    elif not complete and not args.drop_on_abort:
      print(f"  D22: no drop attempted -- {len(df)}/{episode_length} steps, "
            f"force_tripped={robot._fg_tripped} (pass --drop-on-abort to "
            f"drop anyway).")
    else:
      print("  D22: opening gripper — dropping into the taped square …")
      drop_attempted = True
      gripper.open_gripper()
      time.sleep(args.drop_settle_s)  # let the fall + bounce settle
      landed_xyz, _landed_quat = mocap.get_rigid_body_pose()
      if landed_xyz is None:
        print("  D22: mocap lost the cube after the drop — place_error not "
              "scored (the cube may have left the capture volume).")
      else:
        # place_error documents landed_xy as BASE frame
        # (evaluation/gap_metrics.py:991), so convert before scoring.
        landed_base = robot.mocap_pos_to_base(
            np.asarray(landed_xyz, dtype=float))
        landed_box_xy = (float(landed_base[0]), float(landed_base[1]))
        place_error_m, place_in_square = place_error(landed_box_xy)
        print(f"  D22 place_error: {place_error_m * 1000:.1f} mm "
              f"(in_square={place_in_square}), landed_xy={landed_box_xy}")

  finally:
    if gripper is not None:
      try:
        gripper.disconnect()
      except Exception:
        pass
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
          "nominal_horizon": int(episode_length),
          # The cap actually handed to run_policy_loop. With horizon_enforced
          # True, a healthy take has n_steps == max_steps and
          # result.stop_reason == "horizon"; anything else means the loop
          # exited on force/mocap/timeout instead.
          "max_steps": int(episode_length),
          "horizon_enforced": True,
          "control_hz_requested": float(args.control_hz),
          "achieved_hz": float(
              stats.get("true_inferred_frequency_hz") or float("nan")),
          "control_law": args.control_law,
          "box_z_offset": float(args.box_z_offset),
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
          # D22 scripted-drop outcome. Additive to the "defence/1" schema:
          # render_sim_rollout.validate_manifest only checks that its own
          # fixed key list is PRESENT and ignores extras, so no version bump
          # is needed and older manifests keep loading.
          "drop_attempted": bool(drop_attempted),
          "landed_box_xy": landed_box_xy,
          "place_error_m": place_error_m,
          "place_in_square": place_in_square,
          "drop_settle_s": float(args.drop_settle_s),
      },
      "env": {
          "env_name": ckpt_meta.get("env_name", "UR3Pick"),
          "xml_path": os.path.relpath(gap.MODEL_PATH, REPO_ROOT),
          "git_sha": gap.git_sha(),
          "branch": "addvelocity",
      },
      # This script records NO video -- filming is external. The block is
      # still emitted, all-null, because render_sim_rollout.validate_manifest
      # requires these four keys to be PRESENT (it does not require them to be
      # non-null), so dropping the block would break the sim mirror. Any
      # consumer that needs real frames (composite.py) reads source ==
      # "external" and asks for the footage to be cut by hand -- see the SYNC
      # RULE printed at the end of this run.
      "video": {
          "real_mp4": None,
          "webcam_fps_nominal": None,
          "n_webcam_frames": 0,
          "frame_index_csv": None,
          "source": "external",
      },
  }
  with open(os.path.join(out_dir, "manifest.json"), "w",
            encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

  print("\n" + "=" * 70)
  print(f"stop_reason : {stopped_reason}")
  print(f"steps       : {n_steps}/{episode_length}   peak |F-F_bias|: "
        f"{robot._fg_peak_n:.1f} N / {args.force_limit_n:.1f} N limit")
  if place_error_m is None:
    why = ("attempted, but mocap lost the cube" if drop_attempted
           else "not attempted")
    print(f"drop        : {why}")
  else:
    print(f"drop        : place_error {place_error_m * 1000:.1f} mm, "
          f"in_square={place_in_square}")
  print("video       : none recorded (filmed externally)")
  print("SYNC RULE: no automatic alignment was recorded. Between the "
        "send_movej to the start pose and the first policy tick the arm is "
        "completely still (setup countdown -> gripper command -> --settle-s "
        "-> force-bias sampling -> pre-roll countdown). The arm's FIRST "
        "MOVEMENT in the footage is therefore t0, to within one control "
        "tick — the same instant as sim frame 0. Cut the footage there by "
        "hand.")
  print(f"artifacts   : {out_dir}")
  print("Next: git-sync, then on the MacBook run")
  print(f"  python defence/render_sim_rollout.py --run-dir "
        f"defence/runs/{run_name}")
  print("=" * 70)


if __name__ == "__main__":
  main()
