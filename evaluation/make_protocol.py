"""Freeze the sim-to-real gap-protocol episode set (Commit 5 of the vault plan).

Draws N start poses from the HELD-OUT split of the init-pose library, freezes
their VALUES into evaluation/protocols/gap_protocol_v1.json, and hashes the
episodes block. See evaluation/protocols/__init__.py for what a protocol pins
and why the poses come from the library's test split rather than from synthetic
draws around a keyframe (this corrects D4 as originally written -- the env's
reset() does not use a keyframe at all under init_start_random="mid").

Usage:
    .venv/bin/python evaluation/make_protocol.py            # defaults below
    .venv/bin/python evaluation/make_protocol.py --force    # overwrite

Prerequisite -- the TEST split must exist. It is collected on the robot, once:

    python robots/UR3e/collect_init_poses.py --level mid --split test

(60 s of physical Freedrive at 2 s/sample -> 30 poses. Arm powered + RTDE
reachable is enough: the collector never commands motion, and `finger` is drawn
from an RNG rather than read from the gripper, so no Hand-E / URCap / pendant
Play is needed.) The env trains against the "mid" TRAIN split, so the protocol
must use the SAME level's TEST split -- same collection process and same
distribution, separate session, never trained on.

HORIZON -- read this before changing anything
---------------------------------------------
The horizon is NOT a literal here and is deliberately NOT read from
ur3_pick.default_config(). D6 fixes H = the episode_length the policies were
TRAINED at, and on 2026-07-17 exactly that inference cost 30 GPU-runs: the
DR-ladder sweep generator assumed default_config() held the winner values, but
its episode_length=250 is stale (it predates action_scale=0.015), while every
validated run trains at 400 set per-line in the sweep JSONL (see b170af5).

So the horizon is derived from the SWEEP ARTIFACT the policies are actually
trained by: every line is read, the values must agree, and that value becomes
H. If the sweep's horizon changes, the protocol follows it. If the sweep is
internally inconsistent, this errors instead of guessing. Do not "simplify"
this into a constant or a default_config() lookup -- both are how this breaks.

Why the horizon matters for the measurement itself: the ladder's policies do
the lift/transport LATE (gripper_align is approach-only and fixed-length, while
lift/box_target/hold are per-step and gated behind box_off_rest). Truncating
evaluation below the training horizon would score every policy over the
approach phase only -- the phase where sim and real agree MOST -- systematically
understating the gap and flattening every DR config toward each other. That is
a metric that cannot see the effect this study exists to measure.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.protocols import canonical_episode_hash  # noqa: E402
from mujoco_playground._src.manipulation.my_ur3 import ur3_pick  # noqa: E402
from mujoco_playground._src.manipulation.my_ur3.init_poses import (  # noqa: E402
    JOINT_ORDER,
    load_init_poses,
)

DEFAULT_SWEEP = os.path.join(
    REPO_ROOT, "batch_runs", "sweeps", "UR3Pick_dr_ladder.jsonl"
)
DEFAULT_OUT = os.path.join(_THIS_DIR, "protocols", "gap_protocol_v1.json")
DEFAULT_LIB = os.path.join(
    REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "init_poses"
)


def horizon_from_sweep(sweep_path: str) -> int:
    """Read the training horizon off the sweep the policies are trained by.

    Every data line must agree: a sweep that trains different DR levels for
    different episode lengths cannot be evaluated on one shared horizon, and
    silently picking one would compare policies across different tasks.
    """
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f"sweep not found: {sweep_path} -- the protocol horizon is derived "
            f"from the sweep the policies are trained by (see this module's "
            f"docstring). Pass --sweep, or --horizon to override explicitly."
        )
    lengths = {}
    with open(sweep_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = json.loads(line)
            if "episode_length" not in entry:
                raise ValueError(
                    f"sweep line {entry.get('run_id', '?')} in {sweep_path} does "
                    f"not set episode_length. It would inherit "
                    f"ur3_pick.default_config()'s STALE 250 (see b170af5), so the "
                    f"training horizon is not knowable from this artifact. Fix "
                    f"the sweep generator before building a protocol against it."
                )
            lengths.setdefault(int(entry["episode_length"]), []).append(
                entry.get("run_id", "?")
            )
    if not lengths:
        raise ValueError(f"sweep has no data lines: {sweep_path}")
    if len(lengths) > 1:
        detail = "; ".join(
            f"{k}: {len(v)} runs (e.g. {v[0]})" for k, v in sorted(lengths.items())
        )
        raise ValueError(
            f"sweep {sweep_path} trains MIXED episode lengths -- {detail}. A "
            f"single protocol horizon cannot evaluate policies trained on "
            f"different horizons (D6: H = the training episode_length)."
        )
    return next(iter(lengths))


def training_support(level: str, lib_root: str, noise):
    """Per-joint interval the policies were actually trained to start from.

    The env's reset() draws a base pose from the TRAIN split of `level` and adds
    uniform(-init_qpos_noise, +init_qpos_noise) per joint. So the support is the
    train library's per-joint range widened by that noise. Both inputs are read
    from the live env/library -- never hardcoded -- so this cannot drift.

    Returns (lo[6], hi[6], unconstrained[6]) where `unconstrained[j]` marks a
    joint whose noise is >= pi: it is randomized over the full circle during
    training, so EVERY angle is in-distribution and no bound applies. This is
    not a technicality -- init_qpos_noise's wrist_3 term is 2*pi, so a raw
    range comparison flags large wrist_3 "excursions" that are in fact fully
    trained. Filtering on them would discard good poses for no reason.
    """
    tr = load_init_poses(level, "train", root=lib_root)[:, :6]
    noise = np.asarray(noise, dtype=float)[:6]
    unconstrained = noise >= np.pi
    return tr.min(0) - noise, tr.max(0) + noise, unconstrained


def filter_to_support(poses: np.ndarray, lo, hi, unconstrained):
    """Keep only poses lying inside the training support on every bounded joint.

    Rationale: the protocol exists to measure the SIM-TO-REAL gap. A start pose
    the policy never trained from measures out-of-distribution generalization
    instead, and the two are not separable after the fact. Worse, an OOD start
    tends to fail in BOTH domains, driving sim and real return toward the floor
    together -- which makes the D8 retention ratio a 0/0 and quietly destroys
    that episode's contribution to the headline metric.

    The test split is hand-collected in a SEPARATE freedrive session from train,
    and "mid" is a subjective difficulty label rather than an enforced envelope,
    so the two sessions need not cover the same region -- and empirically they
    do not (2026-07-17: 6/30 test poses sit up to +41.7 deg outside the trained
    shoulder_pan range). Filtering keeps the poses real and held out while making
    "in-distribution" true rather than assumed.
    """
    keep = np.ones(poses.shape[0], dtype=bool)
    violations = {}
    for j in range(6):
        if unconstrained[j]:
            continue
        bad = (poses[:, j] < lo[j]) | (poses[:, j] > hi[j])
        if bad.any():
            excursion = float(
                max(np.max(lo[j] - poses[:, j]), np.max(poses[:, j] - hi[j]))
            )
            violations[JOINT_ORDER[j]] = (int(bad.sum()), excursion)
        keep &= ~bad
    return keep, violations


def library_sha256(level: str, split: str, root: str) -> str:
    """Hash the source library file, so a re-collection is detectable."""
    import hashlib

    path = os.path.join(root, split, f"{level}.json")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_episodes(poses: np.ndarray, n: int, seed: int):
    """Draw `n` poses WITHOUT replacement and freeze them into episode dicts.

    Poses are frozen VERBATIM -- no init_qpos_noise is layered on top. Training
    samples library_pose + noise; a protocol episode is the noise=0 case, i.e. a
    point inside that distribution's support, and is exactly reproducible on the
    real arm via a single moveJ. Adding a frozen noise draw would buy nothing
    and only obscure where the pose came from.
    """
    if n > poses.shape[0]:
        raise ValueError(
            f"asked for {n} episodes but the library holds only "
            f"{poses.shape[0]} poses -- collect a longer session "
            f"(--duration) or lower --n"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(poses.shape[0], size=n, replace=False)
    idx = np.sort(idx)  # stable, readable ordering; the draw is still seeded
    episodes = []
    for episode_id, i in enumerate(idx):
        episodes.append(
            {
                "episode_id": episode_id,
                "arm_qpos": [float(x) for x in poses[i, :6]],
                "finger": float(poses[i, 6]),
                "box_marker": f"A{episode_id}",
                "notes": f"place box roughly on marker A{episode_id}",
                "source_pose_index": int(i),
            }
        )
    return episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol_id", default="gap_v1")
    ap.add_argument("--level", choices=["light", "mid", "hard"], default=None,
                    help="pose library tier; default = the env's live "
                         "init_start_random, which is what the policies train on")
    ap.add_argument("--no_support_filter", action="store_true",
                    help="do NOT restrict the draw to the training support "
                         "(measures OOD generalization, not the sim-to-real gap)")
    ap.add_argument("--split", choices=["train", "test"], default="test",
                    help="test = held out from training; see the module docstring")
    ap.add_argument("--n", type=int, default=10, help="episodes to freeze")
    ap.add_argument("--seed", type=int, default=0, help="pose-draw seed")
    ap.add_argument("--real_repeats", type=int, default=3,
                    help="k real runs per episode (D9 noise floor)")
    ap.add_argument("--sweep", default=DEFAULT_SWEEP,
                    help="sweep JSONL the horizon is derived from")
    ap.add_argument("--horizon", type=int, default=None,
                    help="explicit override; normally derived from --sweep")
    ap.add_argument("--lib_root", default=DEFAULT_LIB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists; pass --force to overwrite.\n"
            f"WARNING: overwriting changes poses_sha256, which INVALIDATES every "
            f"result already recorded against the old hash (D5). If runs exist, "
            f"bump --protocol_id and --out instead of overwriting."
        )

    if args.split == "train":
        print(
            "WARNING: --split train draws poses the policies TRAINED on. The "
            "protocol then measures performance on training starts, not held-out "
            "ones. Only do this deliberately, and disclose it in the paper.",
            file=sys.stderr,
        )

    if args.horizon is not None:
        horizon = int(args.horizon)
        print(
            f"WARNING: --horizon {horizon} given explicitly, NOT derived from "
            f"{os.path.relpath(args.sweep, REPO_ROOT)}. D6 requires H == the "
            f"episode_length the policies were trained at; you are asserting "
            f"that yourself.",
            file=sys.stderr,
        )
    else:
        horizon = horizon_from_sweep(args.sweep)

    # The level is NOT assumed: it is read from the live env config, so the
    # protocol can never silently evaluate against a different pose tier than
    # the policies trained on.
    cfg = ur3_pick.default_config()
    train_level = str(cfg.init_start_random)
    level = args.level or train_level
    if level != train_level:
        print(
            f"WARNING: --level {level!r} differs from the env's live "
            f"init_start_random={train_level!r}. The policies train on "
            f"{train_level!r} poses; evaluating on {level!r} measures a "
            f"different task.",
            file=sys.stderr,
        )
    if train_level == "none":
        raise SystemExit(
            "env init_start_random='none': reset() starts from the keyframe, not "
            "the pose library, so a library-drawn protocol would be off-"
            "distribution. Pass --level explicitly if you know what you want."
        )

    try:
        poses = load_init_poses(level, args.split, root=args.lib_root)
    except FileNotFoundError as e:
        raise SystemExit(
            f"{e}\n\n"
            f"The {args.split!r} split of the {args.level!r} library has not been "
            f"collected yet. It needs the robot, once:\n\n"
            f"    python robots/UR3e/collect_init_poses.py "
            f"--level {args.level} --split {args.split}\n\n"
            f"(60 s of physical Freedrive; arm powered + RTDE reachable is "
            f"enough -- no gripper, no URCap, no pendant Play.)"
        )

    pool_size_raw = int(poses.shape[0])
    support = None
    if args.no_support_filter:
        print(
            "WARNING: --no_support_filter -- poses outside the trained start "
            "distribution will be eligible. Any gap measured on them conflates "
            "sim-to-real with OOD generalization.",
            file=sys.stderr,
        )
        violations = {}
    else:
        lo, hi, unconstrained = training_support(level, args.lib_root, cfg.init_qpos_noise)
        keep, violations = filter_to_support(poses, lo, hi, unconstrained)
        # Never truncate silently: say exactly what was dropped and why.
        print(
            f"Support filter ({level}/train range +/- init_qpos_noise): "
            f"{int(keep.sum())}/{pool_size_raw} test poses are in-distribution."
        )
        for jname, (count, exc) in sorted(violations.items()):
            print(
                f"  dropped on {jname:14s}: {count:2d} pose(s), worst excursion "
                f"{exc:+.3f} rad ({np.degrees(exc):+.1f} deg)"
            )
        for j in range(6):
            if unconstrained[j]:
                print(
                    f"  {JOINT_ORDER[j]:14s}: unbounded (init_qpos_noise "
                    f">= pi -> every angle trained; not filtered)"
                )
        if int(keep.sum()) < args.n:
            raise SystemExit(
                f"only {int(keep.sum())} of {pool_size_raw} test poses lie inside "
                f"the training support, but --n {args.n} were requested.\n"
                f"Collect more test poses (--duration) aiming at the SAME arm "
                f"region as the train session, or lower --n. Do NOT pass "
                f"--no_support_filter to make this go away: it would trade a "
                f"loud shortage for a silently OOD protocol."
            )
        poses = poses[keep]
        support = {
            "filtered_to": f"{level}/train range +/- init_qpos_noise",
            "pool_size_raw": pool_size_raw,
            "pool_size_in_support": int(keep.sum()),
            "dropped": {k: {"n": v[0], "worst_excursion_rad": round(v[1], 4)}
                        for k, v in violations.items()},
            "unconstrained_joints": [
                JOINT_ORDER[j] for j in range(6) if unconstrained[j]
            ],
        }

    episodes = build_episodes(poses, args.n, args.seed)
    doc = {
        "protocol_id": args.protocol_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "kind": "init_pose_library_heldout",
            "level": level,
            "split": args.split,
            "library_sha256": library_sha256(level, args.split, args.lib_root),
            "train_library_sha256": library_sha256(level, "train", args.lib_root),
            "pool_size": int(poses.shape[0]),
            "draw_seed": args.seed,
            "support_filter": support,
            "note": (
                "Poses are real UR3e configurations recorded via freedrive "
                "(collect_init_poses.py) and frozen verbatim -- reachable by "
                "construction, in the training distribution's support, never "
                "trained on. NOT synthetic draws around a keyframe (D4 as "
                "originally written misread the env: with "
                "init_start_random='mid' the reset samples this same library, "
                "not a keyframe)."
            ),
        },
        "horizon": horizon,
        "horizon_source": (
            "explicit --horizon override"
            if args.horizon is not None
            else f"derived from {os.path.relpath(args.sweep, REPO_ROOT)} "
                 f"(all lines agree); D6: H == training episode_length"
        ),
        "real_repeats": args.real_repeats,
        # Poses were RECORDED FROM the real arm, so each is reachable and
        # collision-free AT the pose by construction -- there is nothing for a
        # separate moveJ-validation pass to discover (this is why the plan's
        # validate_protocol_poses.py is not needed once poses come from the
        # library). The PATH to a pose from an arbitrary current configuration
        # is a different question, and stays the runner's responsibility:
        # confirm-before-move with conservative speed/accel limits.
        "validated_on_robot": True,
        "episodes": episodes,
    }
    doc["poses_sha256"] = canonical_episode_hash(episodes)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(episodes)} episodes -> {args.out}")
    print(f"  horizon      : {horizon}  ({doc['horizon_source']})")
    print(f"  real_repeats : {args.real_repeats}  -> {len(episodes) * args.real_repeats} real runs per config")
    print(f"  source       : {level}/{args.split}, pool={poses.shape[0]} "
          f"(of {pool_size_raw} collected), draw_seed={args.seed}")
    print(f"  poses_sha256 : {doc['poses_sha256']}")


if __name__ == "__main__":
    main()
