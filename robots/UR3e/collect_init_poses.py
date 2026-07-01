"""Hand-guided UR3e init-pose collector (RTDE freedrive capture).

Record real UR3e joint configurations while you hand-guide the arm in PolyScope
freedrive, to build the init-position libraries that feed BOTH training (the env
reset's "library" start-pose condition) and evaluation (sim + real eval harnesses).
The canonical loader lives in the env package
(``mujoco_playground/_src/manipulation/my_ur3/init_poses/__init__.py``) so there is a
single source of truth; this script writes the files it reads.

The script NEVER commands motion. You move the arm:
  * default — press the physical Freedrive button on the pendant; the script just
    reads joint angles over RTDE (``rtde_receive.RTDEReceiveInterface``, no pendant
    program / go-gate needed, like UR3_RTDE_Tests.ipynb step 1).
  * ``--teach`` — software freedrive via the External Control URCap: reuses
    ``UR3RealRobotPick.connect_arm()`` (blocks until you press Play on the pendant),
    calls ``RTDEControlInterface.teachMode()`` at start and ``endTeachMode()`` in a
    try/finally so freedrive is always cleanly disabled on exit / Ctrl-C.

Output: one JSON file per level+split at ``<out>/<split>/<level>.json`` (format
documented in the env-package ``init_poses`` module). train and test are collected in
SEPARATE sessions → SEPARATE files; no pose appears in both.

Examples:
    # physical Freedrive, 30 light TRAIN poses (60 s, every 2 s) on URSim dry-run
    python robots/UR3e/collect_init_poses.py --level light --split train --ip 127.0.0.1
    # software freedrive (teachMode) on the real robot, extend an existing file
    python robots/UR3e/collect_init_poses.py --level hard --split test --teach --append
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
# Run from any cwd: the sibling dependency module is imported by name (only on the
# --teach path), so its directory must be importable.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
DEFAULT_OUT = os.path.join(
    REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "init_poses"
)
FINGER_MIN, FINGER_MAX = 0.0, 0.025  # sim Hand-E per-finger meters (0 open, 0.025 closed)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _prompt_choice(name, choices):
    """Interactively pick one of ``choices`` (used when the flag is omitted)."""
    while True:
        print(f"\nSelect {name}:")
        for i, c in enumerate(choices, 1):
            print(f"  {i}) {c}")
        raw = input(f"{name} [1-{len(choices)}, or name]: ").strip().lower()
        if raw in choices:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  '{raw}' is not valid — pick a number or one of {choices}.")


def _canonical_load_init_poses():
    """Import the env-package ``load_init_poses`` by file path.

    Loaded standalone (not ``import mujoco_playground...``) so the collector stays
    hermetic — no jax/mujoco/menagerie pulled in — while still exercising the ONE
    real loader for the end-of-run read-back.
    """
    path = os.path.join(
        REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3",
        "init_poses", "__init__.py",
    )
    spec = importlib.util.spec_from_file_location("ur3_init_poses_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_init_poses


# ---------------------------------------------------------------------------
# Connection (no motion commanded)
# ---------------------------------------------------------------------------

def connect(ip: str, teach: bool):
    """Return ``(receiver, robot, teach_active)``.

    ``robot`` is a UR3RealRobotPick only on the --teach path (else None); ``receiver``
    is an RTDEReceiveInterface either way. ``teach_active`` says whether teachMode was
    enabled (so the caller ends it in a finally).
    """
    if teach:
        # Software freedrive: reuse the verified RTDE + External Control URCap setup.
        from ur3_realrobot_dependencies import UR3RealRobotPick  # heavy import, lazy

        robot = UR3RealRobotPick(host=ip, use_ext_urcap=True)
        print(
            "[teach] connecting via External Control URCap — press PLAY on the pendant "
            "(robot in Remote Control) to continue ...",
            flush=True,
        )
        receiver = robot.connect_arm()  # blocks until the program is PLAYING
        if not hasattr(robot, "_control") or robot._control is None:
            raise RuntimeError("connect_arm() did not create a control interface")
        robot._control.teachMode()
        print("[teach] teachMode ENABLED — guide the arm by hand.", flush=True)
        return receiver, robot, True

    # Physical Freedrive button: pure receive, no control interface, no go-gate.
    import rtde_receive  # lazy so --help works without ur_rtde installed

    print(f"[recv] connecting RTDE receive to {ip} ...", flush=True)
    receiver = rtde_receive.RTDEReceiveInterface(ip)
    print(
        "[recv] connected. Press the FREEDRIVE button on the pendant and guide the "
        "arm by hand.",
        flush=True,
    )
    return receiver, None, False


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_loop(receiver, n_samples, interval, level, finger_rng, store_tcp, poses_out):
    """Append captured poses to ``poses_out`` in place (so a Ctrl-C keeps partials)."""
    whole = int(interval)
    frac = float(interval) - whole
    for i in range(n_samples):
        for r in range(whole, 0, -1):
            print(f"\r  pose {i + 1}/{n_samples}: sampling in {r:>2d}s ... ",
                  end="", flush=True)
            time.sleep(1.0)
        if frac > 0:
            time.sleep(frac)

        q = [float(x) for x in receiver.getActualQ()]
        if len(q) != 6:
            print(f"\r  [warn] getActualQ returned {len(q)} values, expected 6 — skipped.   ")
            continue

        rec = {
            "arm_qpos": q,
            "finger": float(finger_rng.uniform(FINGER_MIN, FINGER_MAX)),
            "level": level,
            "captured_at": _now_iso(),
        }
        if store_tcp:
            try:
                rec["tcp_pose"] = [float(x) for x in receiver.getActualTCPPose()]
            except Exception:
                pass
        poses_out.append(rec)
        joints = ", ".join(f"{x:+.3f}" for x in q)
        print(f"\r  pose {i + 1}/{n_samples} captured: [{joints}]  "
              f"finger={rec['finger']:.4f}     ")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_poses(out_path, new_poses, metadata, append):
    """Write (or --append-extend) the per-level JSON. Returns total pose count."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if append and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        meta = existing.get("metadata", metadata)  # keep the original session's metadata
        all_poses = existing.get("poses", []) + new_poses
    else:
        meta, all_poses = metadata, new_poses

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "poses": all_poses}, f, indent=2)
    return len(all_poses)


def print_summary(poses, level):
    arr = np.array([p["arm_qpos"] for p in poses], dtype=float)  # (N, 6)
    fingers = np.array([p["finger"] for p in poses], dtype=float)
    print(f"\n=== summary: {len(poses)} poses (level={level}) ===")
    for j, name in enumerate(JOINT_ORDER):
        col = arr[:, j]
        print(
            f"  {name:13s}  min {col.min():+.3f}  max {col.max():+.3f}  "
            f"range {col.max() - col.min():.3f}"
        )
    print(
        f"  {'finger':13s}  min {fingers.min():.4f}  max {fingers.max():.4f}  "
        f"range {fingers.max() - fingers.min():.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Record UR3e init poses via RTDE while hand-guiding in freedrive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--level", choices=["light", "mid", "hard"],
                    help="difficulty tier (prompted if omitted)")
    ap.add_argument("--split", choices=["train", "test"],
                    help="train or test set (prompted if omitted)")
    ap.add_argument("--ip", default="192.168.1.4", help="UR3e IP (URSim = 127.0.0.1)")
    ap.add_argument("--duration", type=float, default=60.0, help="capture window (s)")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between samples")
    ap.add_argument("--out", default=DEFAULT_OUT, help="init_poses root dir")
    ap.add_argument("--append", action="store_true", help="extend an existing file")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty file")
    ap.add_argument("--teach", action="store_true",
                    help="software freedrive via teachMode (blocks until pendant Play)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the finger RNG")
    args = ap.parse_args()

    # Zero-friction choices: prompt for level/split when the flag is omitted.
    level = args.level or _prompt_choice("level", ["light", "mid", "hard"])
    split = args.split or _prompt_choice("split", ["train", "test"])

    out_path = os.path.join(args.out, split, f"{level}.json")
    n_samples = max(1, int(round(args.duration / args.interval)))

    # Fail loudly BEFORE asking the user to freedrive for a minute.
    if (os.path.exists(out_path) and os.path.getsize(out_path) > 0
            and not args.append and not args.force):
        raise SystemExit(
            f"{out_path} already exists and is non-empty.\n"
            f"  pass --append to add to it, or --force to overwrite."
        )

    print(f"Target file : {out_path}")
    print(f"Plan        : {n_samples} samples, every {args.interval}s for ~{args.duration}s")
    print(f"Mode        : "
          f"{'teachMode (software freedrive)' if args.teach else 'physical Freedrive button'}")

    finger_rng = np.random.default_rng(args.seed)
    metadata = {
        "robot": "UR3e",
        "gripper": "Robotiq HandE",
        "split": split,
        "joint_order": JOINT_ORDER,
        "finger_units": "sim_m[0,0.025]",
        "capture": {
            "duration_s": args.duration,
            "interval_s": args.interval,
            "source": "RTDE getActualQ",
        },
        "created": _now_iso(),
        "robot_ip": args.ip,
        "seed": args.seed,
    }

    poses = []
    robot = None
    receiver = None
    teach_active = False
    try:
        receiver, robot, teach_active = connect(args.ip, args.teach)
        try:
            capture_loop(receiver, n_samples, args.interval, level,
                         finger_rng, store_tcp=True, poses_out=poses)
        except KeyboardInterrupt:
            print("\n[ctrl-c] stopping capture — saving the poses collected so far ...")
    finally:
        if teach_active and robot is not None:
            try:
                robot._control.endTeachMode()
                print("[teach] teachMode disabled.")
            except Exception as e:
                print(f"[warn] endTeachMode failed: {e}")
        try:
            if robot is not None:
                robot.disconnect()
            elif receiver is not None:
                receiver.disconnect()
        except Exception:
            pass

    if not poses:
        print("No poses collected — nothing written.")
        return

    total = write_poses(out_path, poses, metadata, args.append)
    print(f"\nWrote {len(poses)} new pose(s) → {out_path} ({total} total).")
    print_summary(poses, level)

    # Read back through the canonical loader (single source of truth).
    load_init_poses = _canonical_load_init_poses()
    arr = load_init_poses(level, split, root=args.out)
    print(f"[verify] load_init_poses('{level}', '{split}') -> {arr.shape}")


if __name__ == "__main__":
    main()
