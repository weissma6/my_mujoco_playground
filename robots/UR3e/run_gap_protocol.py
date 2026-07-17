"""Real-robot gap-protocol runner (Commit 6 of "Plan - Sim-to-Real Gap Protocol").

A THIN DRIVER over UR3RealRobotPick.run_policy_loop -- the control loop itself
(RTDE receive/servoJ/gripper-thread/logging, all 77 CSV columns) is NOT
duplicated here, exactly as the plan specifies. This script only adds the
protocol-specific bookkeeping run_policy_loop has no reason to know about:
iterating (episode, repeat) pairs, moving to each frozen pose, measuring the
box, computing the matched target, and writing the two small artifacts the
loop doesn't produce on its own (measured_init.json, meta.json).

Usage (one DR-ladder config per invocation -- run it once per config you want
real data for):

    python robots/UR3e/run_gap_protocol.py \
        --config_id L1_pos --policy_run_id <wandb_run_id>

Resumable: a run folder that already exists is skipped, so an interrupted
150-run session (D9: 6 configs x 10 episodes x 3 repeats = 180, or per-config
30) picks up where it left off on the next invocation. Pass --force to redo
specific folders (delete them first, or the resume check would just skip).

Fixed horizon (D6) -- how it is actually enforced
--------------------------------------------------------------------------
run_policy_loop has no "run exactly N steps" mode -- it stops on reach_tol+
dwell convergence, `timeout_s`, mocap loss, an external stop, or a robot
error (see its source; this script does not touch that logic). D6 wants
every episode to run exactly H = the protocol's horizon, with early
termination on reach disabled. This driver gets there with TWO choices, not
one:

  1. `reach_tol=None` is passed explicitly -- run_policy_loop's convergence
     branch is `if reach_tol is not None and ...`, so None disables it
     unconditionally. No early stop on reaching the target.
  2. `timeout_s` is set GENEROUSLY above the nominal H*dt, not tuned to hit H
     exactly. On 2026-07-17 the same control loop measured loop_hz_true ~=
     50.4-50.5 Hz against a 50.0 Hz nominal -- a systematic ~0.8-1% fast bias
     that would drift a tightly-tuned timeout off H by several steps over an
     8s episode. Tuning the timeout precisely is the wrong tool for that.

  D6 itself already anticipates the fix: "Sim truncates to H." So real does
  too -- request a timeout no real run should ever hit (TIMEOUT_MARGIN below,
  ~40% headroom + 2s), let the loop run until that generous timeout fires,
  and TRUNCATE the logged dataframe to exactly the first H rows afterward. A
  run stopped by `reach_tol`/`dwell` is impossible (never enabled). With that
  branch off, EVERY successful run's `stopped_reason` is legitimately
  "timeout" -- that is the designed, expected exit path, not an anomaly, and
  `_finalize_run` does not flag it. The one genuinely anomalous case is the
  opposite: the generous timeout firing BEFORE H rows are logged
  (`stopped_reason == "timeout"` but `n < H`), which the margin is supposed
  to make essentially impossible -- `_finalize_run` gives that combination
  its own louder note, since it means `--control_hz` is likely wrong or the
  control loop is running far slower than every run measured this session. A
  run stopped by mocap_lost/external_stop/robot_error before H rows are
  logged is a genuine abort: it is NOT truncated (there is nothing valid to
  truncate to), it is written with `protocol_complete: false`, and the
  partial data is kept for diagnosis but must be excluded from any headline
  metric.

Target computation -- see evaluation/gap_target.py
--------------------------------------------------------------------------
D2 as originally written ("target = deterministic function of measured box")
is only true for target_mode="box"; every DR-ladder config trains with the
baked default "base_polar", where the target ALSO depends on 3 more
per-episode random draws (dphi/side/r) plus target_z jitter -- not derivable
from the box position alone. gap_target.target_for_episode() freezes those
draws deterministically per (protocol_id, episode_id, config_id) and computes
the same base_polar formula ur3_pick.py's reset() uses. This script and the
future Commit 7 sim mirror call the SAME function, so a real run and its sim
counterpart can never disagree about which target they were scored against.

Loop-mechanics knobs (action_scale, gripper_action_scale, servoJ gains, ...)
--------------------------------------------------------------------------
Read from ur3_pick.default_config(), overridden per the DR-ladder sweep's
OWN overrides dict if that config happens to set them (none of the 10
current configs do -- action_scale/gripper_action_scale are untouched by
every DR-ladder line, see gen_dr_ladder.py's _CONFIGS) -- never hardcoded
here as a magic-number literal the way the older ur3_realrobot_pickloop.py's
ACTION_SCALE has to be (that script predates the DR ladder and manually
tracks "this policy = 0.05" per policy in a comment; this script derives it
so it cannot go stale the same way).

What is intentionally NOT done here
--------------------------------------------------------------------------
No video rendering, no reward-replay plot, no robot/gripper/timing plots per
run -- see ur3_realrobot_pickloop.py for those; running them per protocol run
would be far too slow across ~30-180 runs. The raw CSV (77 columns, unchanged)
plus measured_init.json plus meta.json is everything Commit 7/8/9 need; plots
can always be regenerated later from the CSV via
evaluation/ur3_reward_replay.replay_dataframe, same as any other run.
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

# Generous timeout headroom -- see the module docstring's "Fixed horizon"
# section. NEVER meant to bind; the horizon is enforced by post-hoc
# truncation, not by tuning this to land on H.
TIMEOUT_MARGIN_FACTOR = 1.4
TIMEOUT_MARGIN_SLACK_S = 2.0


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
        f"training reward). This map is intentionally NOT pre-filled: the "
        f"DR-ladder sweep this needs was still training as of this commit."
    )


def confirm(prompt: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"{prompt} [auto-yes]")
        return True
    resp = input(f"{prompt} [y/N] ").strip().lower()
    return resp in ("y", "yes")


def run_dir(out_root, config_id, protocol_id, episode_id, repeat):
    return os.path.join(out_root, config_id, protocol_id, f"ep{episode_id}_rep{repeat}")


def _finalize_run(df: pd.DataFrame, horizon: int, stopped_reason: str):
    """Enforce D6's fixed horizon by truncation, per the module docstring.

    With reach_tol=None, EVERY successful run stops via stopped_reason ==
    "timeout" at the generous margin -- that is the designed, expected exit
    path, not an anomaly, so n >= horizon is never flagged here regardless
    of stopped_reason. The only genuinely anomalous case is the timeout
    firing BELOW horizon (n < horizon with stopped_reason == "timeout"): the
    margin is supposed to never bind, so that combination means the loop ran
    far slower than every run measured this session and is worth a distinct,
    louder note than a plain abort (mocap_lost / external_stop / robot_error).

    Returns (df_final, protocol_complete: bool, note: str).
    """
    n = len(df)
    if n >= horizon:
        return df.iloc[:horizon].reset_index(drop=True), True, ""
    # n < horizon: genuine incompleteness -- either a real abort, or (should
    # not happen) the generous timeout firing early.
    if stopped_reason == "timeout":
        note = (
            f"INCOMPLETE + ANOMALOUS: only {n}/{horizon} steps logged before "
            f"the GENEROUS timeout ({TIMEOUT_MARGIN_FACTOR}x + "
            f"{TIMEOUT_MARGIN_SLACK_S}s margin) fired. This margin is designed "
            f"to never bind -- check --control_hz and this run's achieved Hz; "
            f"the control loop may be running far slower than every run "
            f"measured this session (~50.4-50.5 Hz)."
        )
    else:
        note = (
            f"INCOMPLETE: only {n}/{horizon} steps logged (stopped_reason="
            f"{stopped_reason!r}). Not usable as protocol data; kept for "
            f"diagnosis only."
        )
    print(f"  {note}")
    return df, False, note


def run_episode_repeat(
    robot,
    mocap,
    gripper,
    gripper_fn,
    gripper_state_fn,
    episode,
    repeat,
    args,
    protocol,
    loop_kwargs,
    policy_run_id,
    checkpoint_hash,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(
        f"Episode {episode.episode_id} (marker {episode.box_marker}), "
        f"repeat {repeat}/{protocol.real_repeats}  ->  {out_dir}"
    )
    print(f"{'=' * 70}")
    print(f"  Place the box roughly on marker {episode.box_marker!r}.")
    if not confirm("  Box placed and clear of the arm's path?", args.yes):
        print("  Skipped (not placed).")
        return False

    if not confirm(
        f"  Move arm to episode {episode.episode_id}'s frozen start pose "
        f"{np.round(episode.arm_qpos, 3).tolist()}?",
        args.yes,
    ):
        print("  Skipped (move declined).")
        return False

    # send_movej(asynchronous=False) blocks until the controller reports
    # convergence -- this IS the standstill check the plan asks for; no
    # separate poll is needed on top of it.
    robot.send_movej(
        list(episode.arm_qpos), a=0.3, v=0.4, asynchronous=False,
        textmsg=f"gap-protocol ep{episode.episode_id} rep{repeat}",
    )
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
        protocol.protocol_id, episode.episode_id, args.config_id,
        box_xy=box_xyz[:2], xml_path=MODEL_PATH,
    )

    measured_init = {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "episode_id": episode.episode_id,
        "repeat": repeat,
        "config_id": args.config_id,
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
    dt = 1.0 / args.control_hz
    timeout_s = horizon * dt * TIMEOUT_MARGIN_FACTOR + TIMEOUT_MARGIN_SLACK_S

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
        reach_tol=None,  # D6: no early termination on reach -- see docstring
        dwell_time_s=0.0,
        mocap_stale_s=args.mocap_stale_s,
        box_z_offset=0.0,
    )

    stopped_reason = stats.get("stopped_reason", "completed")
    df_final, protocol_complete, note = _finalize_run(df, horizon, stopped_reason)
    df_final.to_csv(os.path.join(out_dir, "ur3_pick_states.csv"), index=False)

    meta = {
        "protocol_id": protocol.protocol_id,
        "poses_sha256": protocol.poses_sha256,
        "config_id": args.config_id,
        "episode_id": episode.episode_id,
        "repeat": repeat,
        "policy_run_id": policy_run_id,
        "checkpoint_hash_sha256": checkpoint_hash,
        "git_sha": git_sha(),
        "stopped_reason": stopped_reason,
        "expected_steps": horizon,
        "actual_steps_raw": len(df),
        "actual_steps_used": len(df_final),
        "protocol_complete": protocol_complete,
        "note": note,
        "achieved_hz": stats.get("mean_loop_hz_true", float("nan")),
        "overrun_count": stats.get("num_overruns", None),
        "control_hz_requested": args.control_hz,
        "timeout_s_requested": timeout_s,
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

    gripper.open_gripper()  # safe handoff before the next episode's placement
    print(
        f"  Done: {len(df_final)}/{horizon} steps, stopped_reason={stopped_reason}, "
        f"protocol_complete={protocol_complete}"
    )
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--config_id", required=True,
                    help="DR-ladder config id, e.g. L1_pos (see gen_dr_ladder.py)")
    ap.add_argument("--policy_run_id", default=None,
                    help="W&B run id of this config's checkpoint; falls back to "
                         "gap_protocol_policy_map.json if omitted")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    help="subset of episode_ids to run; default = all in the protocol")
    ap.add_argument("--repeats", type=int, default=None,
                    help="override protocol.real_repeats")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="re-run folders that already exist (resume is the default)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the placement/move confirm prompts (unattended use "
                         "only -- there is no safety gate on arm motion without it)")
    ap.add_argument("--robot_ip", default="192.168.1.4")
    ap.add_argument("--use_ext_urcap", action="store_true", default=True)
    ap.add_argument("--ur_cap_port", type=int, default=50002)
    ap.add_argument("--mocap_server_ip", default="10.1.1.198")
    ap.add_argument("--mocap_rigid_body_name", default="CubeInCube2")
    ap.add_argument("--mocap_stale_s", type=float, default=0.25)
    # No --enable_gripper toggle: unlike ur3_realrobot_pickloop.py's demo
    # script, the gripper is not optional here. episode.finger is part of
    # D2's matched-init spec (arm_q + finger + box_pos + box_quat + target),
    # so every protocol run must command it -- a "no gripper" mode would
    # silently produce episodes that don't match their own frozen spec.
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
    args = ap.parse_args()

    if args.control_hz != 50.0:
        print(
            f"WARNING: --control_hz {args.control_hz} != 50.0. The protocol's "
            f"horizon ({{H}} steps) was fixed at training's ctrl_dt=0.02s "
            f"(50 Hz); running at a different rate changes how much wall-clock "
            f"time H steps covers and is very likely not what you want.",
            file=sys.stderr,
        )

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

    loop_kwargs = resolve_loop_kwargs(args.config_id)
    policy_run_id = resolve_policy_run_id(args.config_id, args.policy_run_id)
    policy_path = default_policy_dir(policy_run_id)
    print(f"Config {args.config_id}: policy {policy_run_id}")
    print(f"  action_scale={loop_kwargs['action_scale']}, "
          f"gripper_action_scale={loop_kwargs['gripper_action_scale']}")

    download_policy(
        policy_run_id, out_dir=policy_path, entity=WANDB_ENTITY, project=WANDB_PROJECT,
    )
    checkpoint_hash = sha256_file(os.path.join(policy_path, "params.msgpack"))

    # Skip-if-exists resume pass BEFORE connecting to anything -- lets the
    # operator see what's left without touching the robot.
    pending = []
    for eid in sorted(episode_ids):
        for rep in range(1, repeats + 1):
            out_dir = run_dir(args.out_root, args.config_id, protocol.protocol_id, eid, rep)
            if os.path.exists(out_dir) and not args.force:
                continue
            pending.append((eid, rep, out_dir))
    total = len(episode_ids) * repeats
    print(f"{len(pending)}/{total} runs pending "
          f"({total - len(pending)} already done, resume-skipped).")
    if not pending:
        print("Nothing to do.")
        return

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

    robot.load_policy_fn(policy_path=policy_path, deterministic=True)
    robot.init_fk_model(MODEL_PATH)

    completed = 0
    try:
        for eid, rep, out_dir in pending:
            try:
                ok = run_episode_repeat(
                    robot, mocap, gripper, gripper_fn, gripper_state_fn,
                    episodes_by_id[eid], rep, args, protocol, loop_kwargs,
                    policy_run_id, checkpoint_hash, out_dir,
                )
                completed += int(ok)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR on ep{eid} rep{rep}: {type(e).__name__}: {e}")
                print("  Continuing with the next run (resume will not skip this "
                      "one -- its folder was not fully written).")
    except KeyboardInterrupt:
        print(f"\nInterrupted. {completed}/{len(pending)} pending runs completed "
              f"this session. Re-run the same command to resume.")
    finally:
        mocap.stop()
        gripper.disconnect()
        robot.disconnect()

    print(f"\nDone: {completed}/{len(pending)} pending runs completed.")


if __name__ == "__main__":
    main()
