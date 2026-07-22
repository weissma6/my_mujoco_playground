"""Real-robot gap-protocol runner (Commit 6 of "Plan - Sim-to-Real Gap Protocol").

A THIN DRIVER over UR3RealRobotPick.run_policy_loop -- the control loop itself
(RTDE receive/servoJ/gripper-thread/logging, all 77 CSV columns) is NOT
duplicated here, exactly as the plan specifies. This script only adds the
protocol-specific bookkeeping run_policy_loop has no reason to know about:
iterating (config, episode, repeat, cube_size) tuples, moving to each frozen
pose, prompting for cube placement, measuring the box, computing the matched
target, and writing the two small artifacts the loop doesn't produce on its
own (measured_init.json, ur3_pick_meta.json).

Two invocation modes
--------------------------------------------------------------------------
1. Single-config (original mode, still supported -- useful for a pilot or for
   filling in one config's remaining runs): `--config_id L1_pos`. Runs that
   one config over the requested episodes x repeats x cube_sizes.
2. Full campaign (D16/D17, new in this commit): `--campaign`. Loops over ALL
   four real-deployed DR-ladder configs (D14: L0_none, L1_pos, L2_pos_cube,
   L3_pos_cube_robot -- L4_full/L5_full_obs are cut, they never trained) x 10
   episodes x 3 repeats x {3cm, 4cm} cubes = 240 runs, in the ordering D16/D17
   specify (default: episode-major, cube-grouped within each marker -- see
   `build_campaign_schedule`). All four checkpoints are downloaded and loaded
   ONCE up front and hot-swapped between runs rather than reconnecting per
   config (D16's ordering tip: "costs only loading 4 small checkpoints").

Both modes share the same per-run worker (`run_episode_repeat`) and the same
between-run reset sequence (`between_run_reset`) via `_execute_schedule`, so
the D16 human-in-the-loop mechanics are identical regardless of which mode
launched them.

Resumable: a run folder that already exists is skipped, so an interrupted
240-run session picks up where it left off on the next invocation. Pass
--force to redo specific folders (delete them first, or the resume check
would just skip). The output folder key includes `cube_size`
(`ep{ID}_rep{K}_{cube_size}/`) -- WITHOUT this, a 3cm and a 4cm run of the
same (config, episode, repeat) would land in the same folder and resume would
silently skip one of them (D17).

Fixed horizon (D6/D16) -- how it is actually enforced
--------------------------------------------------------------------------
run_policy_loop has no "run exactly N steps" mode -- it stops on reach_tol+
dwell convergence, `timeout_s`, mocap loss, an external stop, or a robot
error (see its source; this script does not touch that logic). D6/D16 want
every episode to run exactly H = the protocol's horizon, with early
termination on reach disabled, and success is NOT a termination criterion.
This driver gets there with TWO choices:

  1. `reach_tol=None` is passed explicitly -- run_policy_loop's convergence
     branch is `if reach_tol is not None and ...`, so None disables it
     unconditionally. No early stop on reaching the target, ever -- a
     successful grasp keeps running so `hold_target` accrues (D16).
  2. `timeout_s` is now the D16 **hang watchdog** (`--hang_watchdog_s`,
     default 9.0s) -- set just above the nominal real-time duration of H
     steps at the loop's observed ~50.4 Hz (~7.94s), not a generous multiple
     of it. With `reach_tol=None`, "timeout" is the ONLY normal exit path
     run_policy_loop has (besides an exception) -- so this watchdog fires on
     EVERY run, healthy or hung. What distinguishes the two is how many rows
     got logged before it fired:
       - `len(df) >= horizon` -> the healthy case: the loop reached (or
         slightly overshot, given the ~0.8-1% fast bias) H steps before the
         9s cap -- `stop_reason = "complete"`, `stop_step = horizon`,
         truncate to exactly H rows.
       - `len(df) < horizon` -> genuinely incomplete, whatever the cause
         (the watchdog caught a hang, or mocap_lost / external_stop /
         robot_error fired inside run_policy_loop and returned early) --
         `stop_reason = "abort"`, `stop_step = len(df)`. The RAW row count
         is kept (not truncated -- there is nothing valid to truncate to),
         for diagnosis.
     See `_finalize_run`. This REPLACES the prior generous-margin-plus-
     truncation design (TIMEOUT_MARGIN_FACTOR=1.4/+2s) -- that scheme was
     tuned to make "timeout" a designed always-succeeds exit; D16 tightens it
     into an actual hang watchdog, at the cost of relying on the same
     len(df) >= horizon check to tell the two cases apart (which the old
     design already did for its own "anomalous early timeout" case, so nyher
     detection logic here is new, only the threshold changed).
  Limitation, flagged rather than silently assumed: "cube ejected out of
  bounds" (one of D16's named abort triggers) has no dedicated detector in
  this pipeline -- run_policy_loop's stopped_reason vocabulary is
  {completed, timeout, mocap_lost, external_stop, robot_error}. In practice
  a cube ejected far enough to matter is expected to leave the mocap
  capture volume and surface as `mocap_lost` (-> len(df) < horizon -> abort
  here); there is no independent "cube left the table" sensor. Documented as
  a known gap, not implemented as new hardware-detection logic (out of this
  commit's file scope -- would touch ur3_realrobot_dependencies.py).

Between-run reset sequence (D16) -- see `between_run_reset` + the per-run
preamble in `run_episode_repeat`
--------------------------------------------------------------------------
  1. catch all errors from the just-finished episode (`_execute_schedule`'s
     try/except around each schedule entry) -- one bad episode never kills
     the campaign.
  2. wait `--brake_wait_s` (default 10s) -- the window to clear a protective
     stop / release the arm brakes.
  3. open the gripper.
  (1-3 are `between_run_reset`, called after every schedule entry.)
  4. moveJ to the NEXT run's init pose (confirm-before-move, conservative
     speed/accel, standstill check -- send_movej(asynchronous=False) blocks
     until the controller reports convergence, which IS the standstill
     check; same pattern this script already used pre-D16).
  5. Enter-gated cube-placement prompt naming the cube for the run about to
     happen and reminding to hide the other one (D17) -- blocks on
     `input()`, replacing the old timed/y-N wait.
  6. settle -> read mocap box pos+quat = measured init -> run the episode.
  (4-6 are the top of the NEXT call to `run_episode_repeat`.)

Cube-size factor (D17)
--------------------------------------------------------------------------
Every (config, episode, repeat) combination is run with BOTH physical cubes
-- full cross, 240 runs (4 configs x 10 episodes x 3 repeats x 2 cubes).
`cube_size` is recorded in both measured_init.json and ur3_pick_meta.json;
evaluation/run_gap_protocol_sim.py (Commit 7) reads it from there to set the
matching sim box geometry. The campaign schedule groups runs by cube_size
within each marker to minimize physical cube swaps (D16's ordering tip,
D17's "keep D16's episode-major interleave and group by cube within it") --
see `build_campaign_schedule`.

Target computation -- see evaluation/gap_target.py
--------------------------------------------------------------------------
D2 as originally written ("target = deterministic function of measured box")
is only true for target_mode="box"; every DR-ladder config trains with the
baked default "base_polar", where the target ALSO depends on 3 more
per-episode random draws (dphi/side/r) plus target_z jitter -- not derivable
from the box position alone. gap_target.target_for_episode() freezes those
draws deterministically per (protocol_id, episode_id, config_id) and computes
the same base_polar formula ur3_pick.py's reset() uses. This script and
evaluation/run_gap_protocol_sim.py call the SAME function, so a real run and
its sim counterpart can never disagree about which target they were scored
against.

Loop-mechanics knobs (action_scale, gripper_action_scale, servoJ gains, ...)
--------------------------------------------------------------------------
Read from ur3_pick.default_config(), overridden per the DR-ladder sweep's
OWN overrides dict if that config happens to set them -- never hardcoded
here as a magic-number literal.

What is intentionally NOT done here
--------------------------------------------------------------------------
No video rendering, no reward-replay plot, no robot/gripper/timing plots per
run -- see ur3_realrobot_pickloop.py for those; running them per protocol run
would be far too slow across ~240 runs. The raw CSV (77 columns, unchanged)
plus measured_init.json plus ur3_pick_meta.json is everything Commit 7/8/9
need; plots can always be regenerated later from the CSV via
evaluation/ur3_reward_replay.replay_dataframe, same as any other run.

Usage
--------------------------------------------------------------------------
    # Single config (pilot / backfill):
    python robots/UR3e/run_gap_protocol.py \\
        --config_id L1_pos --policy_run_id <wandb_run_id> \\
        --cube_sizes 3cm            # pilot: 1 config x 10 episodes x 3 repeats

    # Full campaign (needs gap_protocol_policy_map.json filled for all 4
    # configs -- D10):
    python robots/UR3e/run_gap_protocol.py --campaign
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (
    REPO_ROOT,
    _THIS_DIR,
    os.path.join(REPO_ROOT, "evaluation"),
    os.path.join(REPO_ROOT, "evaluation", "downloaded_policies"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ur3_realrobot_dependencies import UR3RealRobotPick  # noqa: E402
from motion_capture.mymocap.vrpn_dependencies import VRPNRigidBodyReader  # noqa: E402
from robots.hande.HandE_dependency import HandEGripper  # noqa: E402
from policy_downloader import default_policy_dir, download_policy  # noqa: E402

from evaluation.protocols import load_protocol  # noqa: E402
from evaluation import gap_target  # noqa: E402
from mujoco_playground._src.manipulation.my_ur3 import ur3_pick  # noqa: E402
from batch_runs.sweeps.gen_dr_ladder import _CONFIGS as _LADDER_CONFIGS  # noqa: E402

_LADDER_OVERRIDES = {cid: overrides for cid, overrides, _tags in _LADDER_CONFIGS}

MODEL_PATH = os.path.join(
    REPO_ROOT,
    "mujoco_playground",
    "_src",
    "manipulation",
    "my_ur3",
    "xmls",
    "mjx_single_cube_position_ur3.xml",
)
DEFAULT_PROTOCOL = os.path.join(
    REPO_ROOT, "evaluation", "protocols", "gap_protocol_v1.json"
)
DEFAULT_OUT_ROOT = os.path.join(_THIS_DIR, "real_robot_results", "gap_protocol")
POLICY_MAP_PATH = os.path.join(_THIS_DIR, "gap_protocol_policy_map.json")

WANDB_ENTITY = "weissma6-zhaw-school-of-engineering"
WANDB_PROJECT = "UR3_pick_ppo"

# D14: only these 4 configs ever reach the real robot -- L4_full/L5_full_obs
# collapsed in training (0% success, all seeds) and are cut everywhere
# downstream, including the campaign driver's default config set.
LADDER_REAL_CONFIGS = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot"]

# D17: full cross, both physical cubes get every (config, episode, repeat).
CUBE_SIZES = ("3cm", "4cm")

# D16: hang watchdog, replaces the old generous-timeout-plus-truncation
# design -- see the module docstring's "Fixed horizon" section.
DEFAULT_HANG_WATCHDOG_S = 9.0
# D16: window to clear a protective stop / release the arm brakes between runs.
DEFAULT_BRAKE_WAIT_S = 10.0


def git_sha() -> str:
    """Best-effort HEAD SHA; never raises (mirrors is_stopped_externally's
    'never raises on its own' convention -- a logging helper must not be able
    to crash a real-robot run)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_loop_kwargs(config_id: str) -> dict:
    """action_scale/gripper_action_scale for this config: sweep override if
    present, else the baked default -- see the module docstring."""
    if config_id not in _LADDER_OVERRIDES:
        raise ValueError(
            f"unknown config_id {config_id!r}; not one of "
            f"{sorted(_LADDER_OVERRIDES)} (from gen_dr_ladder.py's _CONFIGS)."
        )
    overrides = _LADDER_OVERRIDES[config_id]
    cfg = ur3_pick.default_config()
    action_scale = float(overrides.get("action_scale", cfg.action_scale))
    gripper_action_scale = float(
        overrides.get("gripper_action_scale", cfg.gripper_action_scale)
    )
    return {"action_scale": action_scale, "gripper_action_scale": gripper_action_scale}


def resolve_policy_run_id(config_id: str, explicit: str) -> str:
    if explicit:
        return explicit
    if os.path.exists(POLICY_MAP_PATH):
        with open(POLICY_MAP_PATH, "r", encoding="utf-8") as f:
            policy_map = json.load(f)
        if config_id in policy_map:
            return policy_map[config_id]
    raise SystemExit(
        f"No policy run id for {config_id!r}. Either pass --policy_run_id "
        f"explicitly, or add {{\"{config_id}\": \"<wandb_run_id>\"}} to "
        f"{POLICY_MAP_PATH} once D10's median-seed selection has picked it "
        f"(median by sim return on the protocol episodes -- Commit 7, not "
        f"training reward)."
    )


def confirm(prompt: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"{prompt} [auto-yes]")
        return True
    resp = input(f"{prompt} [y/N] ").strip().lower()
    return resp in ("y", "yes")


def press_enter(prompt: str, auto_yes: bool) -> None:
    """D17's Enter-gated cube-placement prompt -- BLOCKS on input(), not a
    timed wait, and not a y/N confirm (there is nothing to decline; the
    operator either places the cube and presses Enter, or the campaign
    just waits). --yes (unattended use only) skips the block.
    """
    if auto_yes:
        print(f"{prompt}\n  [auto-yes: not actually blocking]")
        return
    input(f"{prompt}\n  Press Enter when ready...")


def cube_prompt_message(cube_size: str, other_cube_size: str, marker: str,
                         cube_changed: bool) -> str:
    """D17: name the cube for the upcoming run, remind to hide the other one
    (shared mocap rigid body -- only one cube may be in the capture volume at
    a time). `cube_changed` banners the swap boundary loudly (D16/D17's
    "group by cube within a marker, prompt once at the swap") -- the prompt
    itself still fires every run (a pick moves the cube; D9 mechanics), only
    the banner is swap-specific.
    """
    banner = ""
    if cube_changed:
        banner = (
            f"\n{'=' * 70}\n"
            f"=== CUBE SWAP: switch to the {cube_size} cube now ===\n"
            f"{'=' * 70}\n"
        )
    return (
        f"{banner}Place the {cube_size} cube on marker {marker!r}. "
        f"Remove/hide the {other_cube_size} cube from the mocap capture volume."
    )


def run_dir(out_root, config_id, protocol_id, episode_id, repeat, cube_size):
    """D17: cube_size is part of the folder key. Without it a 3cm and a 4cm
    run of the same (config, episode, repeat) collide and resume silently
    skips one of them.
    """
    return os.path.join(
        out_root, config_id, protocol_id, f"ep{episode_id}_rep{repeat}_{cube_size}"
    )


def _finalize_run(df: pd.DataFrame, horizon: int, loop_stopped_reason: str):
    """D16 fixed-horizon scoring: time (H reached) or abort, nothing else.

    With reach_tol=None and timeout_s == the D16 hang watchdog (see the
    module docstring's "Fixed horizon" section), the watchdog is the ONLY
    normal exit path -- it fires on every run, healthy or hung. What tells
    the two apart is the row count:
      - n >= horizon -> stop_reason="complete", stop_step=horizon, truncate.
      - n <  horizon -> stop_reason="abort", stop_step=n, NOT truncated (kept
        for diagnosis) -- covers a genuine hang, mocap_lost, external_stop,
        and robot_error alike (see the module docstring's watchdog note).

    Returns (df_final, stop_step, stop_reason, note).
    """
    n = len(df)
    if n >= horizon:
        return df.iloc[:horizon].reset_index(drop=True), horizon, "complete", ""
    note = (
        f"ABORT: only {n}/{horizon} steps logged before the "
        f"{DEFAULT_HANG_WATCHDOG_S}s-class hang watchdog or an error stopped "
        f"the loop (underlying run_policy_loop stopped_reason="
        f"{loop_stopped_reason!r}). Excluded from the paired reward gap "
        f"(D16); kept on disk for diagnosis. The per-(config, cube_size) "
        f"abort RATE is itself a reported result (D16/D17) -- do not delete "
        f"this folder."
    )
    print(f"  {note}")
    return df, n, "abort", note


def between_run_reset(robot, gripper, brake_wait_s: float) -> None:
    """D16 steps 1-3: (error already caught by the caller) wait for the brake
    window, then open the gripper. Steps 4-6 (moveJ, cube prompt, settle +
    mocap read) are the top of the NEXT `run_episode_repeat` call.
    """
    print(
        f"\n--- Reset window: waiting {brake_wait_s:.0f}s -- clear any "
        f"protective stop / release the arm brakes now if needed ---"
    )
    time.sleep(brake_wait_s)
    gripper.open_gripper()
    print("--- Gripper opened. Moving to the next run. ---")


def run_episode_repeat(
    robot,
    mocap,
    gripper,
    gripper_fn,
    gripper_state_fn,
    episode,
    repeat,
    cube_size,
    other_cube_size,
    cube_changed,
    hang_watchdog_s,
    args,
    protocol,
    loop_kwargs,
    policy_run_id,
    checkpoint_hash,
    config_id,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(
        f"Config {config_id}  Episode {episode.episode_id} "
        f"(marker {episode.box_marker}), repeat {repeat}/{protocol.real_repeats}, "
        f"cube {cube_size}  ->  {out_dir}"
    )
    print(f"{'=' * 70}")

    # D16 step 4: moveJ to this run's frozen start pose. confirm-before-move +
    # asynchronous=False blocking convergence IS the standstill check.
    if not confirm(
        f"  Move arm to episode {episode.episode_id}'s frozen start pose "
        f"{np.round(episode.arm_qpos, 3).tolist()}?",
        args.yes,
    ):
        print("  Skipped (move declined).")
        return False
    robot.send_movej(
        list(episode.arm_qpos), a=0.3, v=0.4, asynchronous=False,
        textmsg=f"gap-protocol ep{episode.episode_id} rep{repeat} {cube_size}",
    )

    # D16 step 5 / D17: Enter-gated cube-placement prompt, blocks on input().
    press_enter(
        cube_prompt_message(cube_size, other_cube_size, episode.box_marker,
                             cube_changed),
        args.yes,
    )

    # D16 step 6: settle -> command finger -> read mocap = measured init.
    gripper.command(episode.finger)
    time.sleep(args.settle_s)

    fb = robot.receive_feedback()
    measured_arm_qpos = list(np.asarray(fb["q"], dtype=float))

    box_xyz, box_quat = mocap.get_rigid_body_pose()
    if box_xyz is None:
        print("  ABORTED: mocap has no data for the tracked body. Check "
              "visibility and try again (this run's folder is left empty; "
              "resume will re-attempt it).")
        return False

    target_pos, components = gap_target.target_for_episode(
        protocol.protocol_id, episode.episode_id, config_id,
        box_xy=box_xyz[:2], xml_path=MODEL_PATH,
    )

    measured_init = {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "episode_id": episode.episode_id,
        "repeat": repeat,
        "config_id": config_id,
        "cube_size": cube_size,  # D17
        "commanded_arm_qpos": list(episode.arm_qpos),
        "measured_arm_qpos": measured_arm_qpos,
        "commanded_finger": episode.finger,
        "box_marker": episode.box_marker,
        "box_pos": list(np.asarray(box_xyz, dtype=float)),
        "box_quat_wxyz": list(np.asarray(box_quat, dtype=float)),
        "target_pos": list(target_pos),
        "target_components": {
            "dphi": components.dphi, "side": components.side, "r": components.r,
            "target_z_draw": components.target_z_draw,
            "profile_name": components.profile_name, "seed": components.seed,
        },
        "settle_s": args.settle_s,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "measured_init.json"), "w", encoding="utf-8") as f:
        json.dump(measured_init, f, indent=2)

    horizon = protocol.horizon
    # D16: timeout_s IS the hang watchdog now (not a generous multiple) --
    # see the module docstring's "Fixed horizon" section for why this alone
    # is sufficient to recover stop_reason/stop_step post-hoc.
    timeout_s = hang_watchdog_s

    df, stats = robot.run_policy_loop(
        drop_target=np.asarray(target_pos, dtype=np.float32),
        mocap_reader=mocap,
        control_hz=args.control_hz,
        timeout_s=timeout_s,
        action_scale=loop_kwargs["action_scale"],
        gripper_action_scale=loop_kwargs["gripper_action_scale"],
        lookahead_time=args.lookahead_time,
        gain=args.gain,
        servoj_a=args.servoj_a,
        servoj_v=args.servoj_v,
        alpha=1.0,
        gripper_fn=gripper_fn,
        gripper_state_fn=gripper_state_fn,
        gripper_tau=args.gripper_tau,
        gripper_max_rate=float("inf"),
        use_fk_tcp=True,
        reach_tol=None,  # D6/D16: no early termination on reach, ever
        dwell_time_s=0.0,
        mocap_stale_s=args.mocap_stale_s,
        box_z_offset=0.0,
    )

    loop_stopped_reason = stats.get("stopped_reason", "completed")
    df_final, stop_step, stop_reason, note = _finalize_run(
        df, horizon, loop_stopped_reason
    )
    df_final.to_csv(os.path.join(out_dir, "ur3_pick_states.csv"), index=False)

    meta = {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "config_id": config_id,
        "episode_id": episode.episode_id,
        "repeat": repeat,
        "cube_size": cube_size,  # D17 -- NEW
        "policy_run_id": policy_run_id,
        "checkpoint_hash_sha256": checkpoint_hash,
        "git_sha": git_sha(),
        # Existing column, UNCHANGED semantics: the raw run_policy_loop exit
        # reason (completed/timeout/mocap_lost/external_stop/robot_error).
        "stopped_reason": loop_stopped_reason,
        # D16 -- NEW: the canonical {complete, abort} classification derived
        # from stop_step vs horizon (see _finalize_run). This is what
        # downstream (gap_metrics.py, run_gap_protocol_sim.py) reads.
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        "expected_steps": horizon,
        "actual_steps_raw": len(df),
        "actual_steps_used": len(df_final),
        "protocol_complete": stop_reason == "complete",
        "note": note,
        "achieved_hz": stats.get("mean_loop_hz_true", float("nan")),
        "overrun_count": stats.get("num_overruns", None),
        "control_hz_requested": args.control_hz,
        "hang_watchdog_s": hang_watchdog_s,
        "action_scale": loop_kwargs["action_scale"],
        "gripper_action_scale": loop_kwargs["gripper_action_scale"],
        "servoj_a": args.servoj_a,
        "servoj_v": args.servoj_v,
        "lookahead_time": args.lookahead_time,
        "gain": args.gain,
        "gripper_tau": args.gripper_tau,
        "robot_ip": args.robot_ip,
        "mocap_server_ip": args.mocap_server_ip,
        "mocap_rigid_body_name": args.mocap_rigid_body_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "ur3_pick_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(
        f"  Done: {len(df_final)}/{horizon} steps, stop_reason={stop_reason} "
        f"(loop stopped_reason={loop_stopped_reason})"
    )
    return True


# ===========================================================================
# Campaign scheduling (D16/D17).
# ===========================================================================


def build_campaign_schedule(episode_ids, configs, repeats, cube_sizes, order):
    """Returns a list of dicts {episode_id, config_id, repeat, cube_size} in
    the requested order.

    episode_major (default, D16/D17): outer = episode/marker, then cube_size
    (grouped -- do all of a marker's 3cm runs across every policy x repeat,
    THEN swap once to 4cm), then config, then repeat. Minimizes physical cube
    swaps (240 runs -> ~20 swaps: one pair per marker) while keeping policy
    interleaved against session time within each cube block, per D16's
    ordering tip.

    policy_major (override): outer = config, then episode, then cube_size
    (grouped), then repeat -- fewer policy (re)loads, but confounds policy
    with session time (D16 explicitly recommends against this as the
    default).
    """
    if order not in ("episode_major", "policy_major"):
        raise ValueError(f"unknown order {order!r}")
    entries = []
    if order == "episode_major":
        for eid in episode_ids:
            for cube in cube_sizes:
                for cfg in configs:
                    for rep in range(1, repeats + 1):
                        entries.append(dict(
                            episode_id=eid, config_id=cfg, repeat=rep, cube_size=cube,
                        ))
    else:  # policy_major
        for cfg in configs:
            for eid in episode_ids:
                for cube in cube_sizes:
                    for rep in range(1, repeats + 1):
                        entries.append(dict(
                            episode_id=eid, config_id=cfg, repeat=rep, cube_size=cube,
                        ))
    return entries


def _resolve_config_ctx(config_ids, explicit_policy_run_id=None):
    """{config_id: {loop_kwargs, policy_run_id, policy_path, checkpoint_hash}}
    for every config in `config_ids` -- resolved and downloaded up front so
    the campaign can hot-swap policies between runs without re-downloading.
    `explicit_policy_run_id` only applies when there is exactly ONE config
    (single-config mode's --policy_run_id override); campaign mode always
    resolves from gap_protocol_policy_map.json.
    """
    ctx = {}
    for cid in config_ids:
        loop_kwargs = resolve_loop_kwargs(cid)
        policy_run_id = resolve_policy_run_id(
            cid, explicit_policy_run_id if len(config_ids) == 1 else None
        )
        policy_path = default_policy_dir(policy_run_id)
        print(f"Config {cid}: policy {policy_run_id}")
        print(f"  action_scale={loop_kwargs['action_scale']}, "
              f"gripper_action_scale={loop_kwargs['gripper_action_scale']}")
        download_policy(
            policy_run_id, out_dir=policy_path, entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
        )
        checkpoint_hash = sha256_file(os.path.join(policy_path, "params.msgpack"))
        ctx[cid] = {
            "loop_kwargs": loop_kwargs,
            "policy_run_id": policy_run_id,
            "policy_path": policy_path,
            "checkpoint_hash": checkpoint_hash,
        }
    return ctx


def _execute_schedule(schedule, args, protocol, episodes_by_id, config_ctx):
    """Shared executor for both single-config and campaign mode.

    `schedule`: list of dicts {episode_id, config_id, repeat, cube_size}
    (already filtered to pending -- resume-skipped entries are not passed
    in). `config_ctx`: from `_resolve_config_ctx` -- may hold 1 (single-
    config) or up to 4 (campaign) entries; the policy is hot-swapped via
    robot.load_policy_fn only when config_id changes between consecutive
    entries, so a single-config run never reloads and a campaign only
    reloads at policy boundaries.
    """
    print(f"\nConnecting to robot {args.robot_ip} ...")
    robot = UR3RealRobotPick(
        host=args.robot_ip, use_ext_urcap=args.use_ext_urcap, ur_cap_port=args.ur_cap_port,
    )
    robot.connect()
    if not robot.is_connected():
        raise RuntimeError("RTDE failed to connect.")
    robot.print_feedback()

    print(f"Connecting to mocap {args.mocap_server_ip} "
          f"(rigid body {args.mocap_rigid_body_name}) ...")
    mocap = VRPNRigidBodyReader(
        args.mocap_server_ip, rigid_body_name=args.mocap_rigid_body_name,
        names=[args.mocap_rigid_body_name],
    )
    if not mocap.start(timeout=5.0) or not mocap.wait_for_data(timeout=5.0):
        raise RuntimeError("Mocap failed to connect or never reported data.")

    gripper = HandEGripper(
        args.robot_ip, port=args.gripper_port, slave_id=args.gripper_slave_id,
        speed_pct=args.gripper_speed_pct, force_pct=args.gripper_force_pct,
    )
    gripper.connect()
    gripper.open_gripper()
    gripper_fn = lambda norm: gripper.command(norm * 0.025)  # noqa: E731
    gripper_state_fn = lambda: gripper.read_state()  # noqa: E731

    robot.init_fk_model(MODEL_PATH)

    loaded_config = None
    last_cube_size = None
    completed = 0
    try:
        for entry in schedule:
            cid, eid, rep, cube = (
                entry["config_id"], entry["episode_id"], entry["repeat"],
                entry["cube_size"],
            )
            ctx = config_ctx[cid]
            if cid != loaded_config:
                print(f"\n[policy swap] loading {cid} ({ctx['policy_run_id']}) ...")
                robot.load_policy_fn(policy_path=ctx["policy_path"], deterministic=True)
                loaded_config = cid
            cube_changed = last_cube_size is not None and cube != last_cube_size
            other_cube = next((c for c in CUBE_SIZES if c != cube), cube)
            out_dir = run_dir(args.out_root, cid, protocol.protocol_id, eid, rep, cube)

            try:
                ok = run_episode_repeat(
                    robot, mocap, gripper, gripper_fn, gripper_state_fn,
                    episodes_by_id[eid], rep, cube, other_cube, cube_changed,
                    args.hang_watchdog_s, args, protocol, ctx["loop_kwargs"],
                    ctx["policy_run_id"], ctx["checkpoint_hash"], cid, out_dir,
                )
                completed += int(ok)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                # D16 step 1: catch everything -- one bad episode never kills
                # the campaign. No folder is guaranteed written here (unlike
                # the watchdog/abort path inside run_episode_repeat, which
                # DOES write a folder with stop_reason="abort"); resume will
                # retry this exact entry next session.
                print(f"  ERROR on {cid} ep{eid} rep{rep} {cube}: "
                      f"{type(e).__name__}: {e}")
                print("  Continuing with the next run (resume will not skip "
                      "this one -- its folder was not fully written).")

            last_cube_size = cube
            # D16 steps 1-3 (error already caught above): brake window, then
            # open the gripper. Steps 4-6 happen at the top of the next loop
            # iteration's run_episode_repeat call.
            between_run_reset(robot, gripper, args.brake_wait_s)
    except KeyboardInterrupt:
        print(f"\nInterrupted. {completed}/{len(schedule)} pending runs "
              f"completed this session. Re-run the same command to resume.")
    finally:
        mocap.stop()
        gripper.disconnect()
        robot.disconnect()

    print(f"\nDone: {completed}/{len(schedule)} pending runs completed.")
    return completed


# ===========================================================================
# Single-config mode (original CLI shape, extended with cube_size).
# ===========================================================================


def run_single_config(args):
    protocol = load_protocol(args.protocol)  # raises if validated_on_robot=false
    print(
        f"Protocol {protocol.protocol_id}: {len(protocol)} episodes, "
        f"horizon={protocol.horizon}, real_repeats={protocol.real_repeats}, "
        f"hash={protocol.poses_sha256[:16]}..."
    )

    episode_ids = args.episodes or [e.episode_id for e in protocol.episodes]
    episodes_by_id = {e.episode_id: e for e in protocol.episodes}
    unknown = set(episode_ids) - set(episodes_by_id)
    if unknown:
        raise SystemExit(f"--episodes contains unknown episode_id(s): {sorted(unknown)}")
    repeats = args.repeats or protocol.real_repeats
    cube_sizes = args.cube_sizes or list(CUBE_SIZES)

    config_ctx = _resolve_config_ctx([args.config_id], args.policy_run_id)

    full_schedule = build_campaign_schedule(
        sorted(episode_ids), [args.config_id], repeats, cube_sizes, args.order
    )
    pending = []
    for entry in full_schedule:
        out_dir = run_dir(args.out_root, entry["config_id"], protocol.protocol_id,
                          entry["episode_id"], entry["repeat"], entry["cube_size"])
        if os.path.exists(out_dir) and not args.force:
            continue
        pending.append(entry)
    print(f"{len(pending)}/{len(full_schedule)} runs pending "
          f"({len(full_schedule) - len(pending)} already done, resume-skipped).")
    if not pending:
        print("Nothing to do.")
        return

    _execute_schedule(pending, args, protocol, episodes_by_id, config_ctx)


# ===========================================================================
# Campaign mode (D16/D17 -- new in this commit).
# ===========================================================================


def run_campaign(args):
    protocol = load_protocol(args.protocol)
    print(
        f"Protocol {protocol.protocol_id}: {len(protocol)} episodes, "
        f"horizon={protocol.horizon}, real_repeats={protocol.real_repeats}, "
        f"hash={protocol.poses_sha256[:16]}..."
    )

    episode_ids = args.episodes or [e.episode_id for e in protocol.episodes]
    episodes_by_id = {e.episode_id: e for e in protocol.episodes}
    unknown = set(episode_ids) - set(episodes_by_id)
    if unknown:
        raise SystemExit(f"--episodes contains unknown episode_id(s): {sorted(unknown)}")
    configs = args.configs or LADDER_REAL_CONFIGS
    unknown_cfg = set(configs) - set(_LADDER_OVERRIDES)
    if unknown_cfg:
        raise SystemExit(f"--configs contains unknown config_id(s): {sorted(unknown_cfg)}")
    repeats = args.repeats or protocol.real_repeats
    cube_sizes = args.cube_sizes or list(CUBE_SIZES)

    print(f"Campaign: {len(configs)} configs x {len(episode_ids)} episodes x "
          f"{repeats} repeats x {len(cube_sizes)} cube size(s) = "
          f"{len(configs) * len(episode_ids) * repeats * len(cube_sizes)} runs "
          f"(order={args.order}).")

    config_ctx = _resolve_config_ctx(configs)

    full_schedule = build_campaign_schedule(
        sorted(episode_ids), configs, repeats, cube_sizes, args.order
    )
    pending = []
    for entry in full_schedule:
        out_dir = run_dir(args.out_root, entry["config_id"], protocol.protocol_id,
                          entry["episode_id"], entry["repeat"], entry["cube_size"])
        if os.path.exists(out_dir) and not args.force:
            continue
        pending.append(entry)
    print(f"{len(pending)}/{len(full_schedule)} runs pending "
          f"({len(full_schedule) - len(pending)} already done, resume-skipped).")
    if not pending:
        print("Nothing to do.")
        return

    _execute_schedule(pending, args, protocol, episodes_by_id, config_ctx)


def _build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--campaign", action="store_true",
                    help="run the full D16/D17 campaign over LADDER_REAL_CONFIGS "
                         "(or --configs) x episodes x repeats x cube_sizes, "
                         "instead of a single --config_id")
    ap.add_argument("--config_id", default=None,
                    help="DR-ladder config id, e.g. L1_pos (see gen_dr_ladder.py). "
                         "Required unless --campaign is given.")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="--campaign only: override the config set (default "
                         "LADDER_REAL_CONFIGS, the 4 D14-surviving configs)")
    ap.add_argument("--policy_run_id", default=None,
                    help="W&B run id of this config's checkpoint (single-config "
                         "mode only); falls back to gap_protocol_policy_map.json "
                         "if omitted. Campaign mode always resolves per-config "
                         "from the policy map.")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    help="subset of episode_ids to run; default = all in the protocol")
    ap.add_argument("--repeats", type=int, default=None,
                    help="override protocol.real_repeats")
    ap.add_argument("--cube_sizes", nargs="+", default=None, choices=list(CUBE_SIZES),
                    help="D17: which physical cube size(s) to run; default = "
                         "both (full cross). Pass a single size (e.g. '3cm') "
                         "for the Commit-6 pilot / a 3cm-only backfill.")
    ap.add_argument("--order", choices=["episode_major", "policy_major"],
                    default="episode_major",
                    help="D16/D17 campaign ordering; default episode_major "
                         "(cube-grouped within each marker, minimizes swaps)")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="re-run folders that already exist (resume is the default)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the placement/move/cube-Enter confirm prompts "
                         "(unattended use only -- there is no safety gate on arm "
                         "motion without it)")
    ap.add_argument("--robot_ip", default="192.168.1.4")
    ap.add_argument("--use_ext_urcap", action="store_true", default=True)
    ap.add_argument("--ur_cap_port", type=int, default=50002)
    ap.add_argument("--mocap_server_ip", default="10.1.1.198")
    ap.add_argument("--mocap_rigid_body_name", default="CubeInCube2")
    ap.add_argument("--mocap_stale_s", type=float, default=0.25)
    ap.add_argument("--gripper_port", type=int, default=49999)
    ap.add_argument("--gripper_slave_id", type=int, default=9)
    ap.add_argument("--gripper_speed_pct", type=int, default=60)
    ap.add_argument("--gripper_force_pct", type=int, default=80)
    ap.add_argument("--gripper_tau", type=float, default=0.1)
    ap.add_argument("--control_hz", type=float, default=50.0,
                    help="MUST match training ctrl_dt=0.02 for the horizon to mean "
                         "what the protocol says it means; override only if you "
                         "know why")
    ap.add_argument("--lookahead_time", type=float, default=0.1)
    ap.add_argument("--gain", type=int, default=300)
    ap.add_argument("--servoj_a", type=float, default=0.3)
    ap.add_argument("--servoj_v", type=float, default=1.0)
    ap.add_argument("--settle_s", type=float, default=1.0,
                    help="pause after placement/gripper command before reading "
                         "the mocap 'measured init'")
    ap.add_argument("--hang_watchdog_s", type=float, default=DEFAULT_HANG_WATCHDOG_S,
                    help="D16: wall-clock cap per episode, just above the nominal "
                         "H-step duration at ~50.4Hz (~7.94s) -- catches a hung "
                         "loop within ~1s. Also the ONLY normal exit path with "
                         "reach_tol=None; see the module docstring.")
    ap.add_argument("--brake_wait_s", type=float, default=DEFAULT_BRAKE_WAIT_S,
                    help="D16: window between episodes to clear a protective "
                         "stop / release the arm brakes.")
    return ap


def main():
    ap = _build_argparser()
    args = ap.parse_args()

    if args.control_hz != 50.0:
        print(
            f"WARNING: --control_hz {args.control_hz} != 50.0. The protocol's "
            f"horizon (H steps) was fixed at training's ctrl_dt=0.02s "
            f"(50 Hz); running at a different rate changes how much wall-clock "
            f"time H steps covers and is very likely not what you want.",
            file=sys.stderr,
        )

    if args.campaign:
        if args.config_id:
            raise SystemExit("--config_id and --campaign are mutually exclusive "
                              "(--campaign always uses --configs / "
                              "LADDER_REAL_CONFIGS).")
        run_campaign(args)
    else:
        if not args.config_id:
            raise SystemExit("--config_id is required unless --campaign is given.")
        run_single_config(args)


if __name__ == "__main__":
    main()
