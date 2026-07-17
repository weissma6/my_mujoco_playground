"""Frozen, hashed episode sets for the sim-to-real gap protocol.

See the vault plan "Plan - Sim-to-Real Gap Protocol" (VT2-SimToReal-Robotics),
D2/D4/D5/D6. A *protocol* is the immutable answer to "which episodes does every
policy get evaluated on", so that a real rollout and its matched MuJoCo mirror
are comparable across DR-ladder configs, across seeds, and across weeks.

What a protocol pins
--------------------
  * ``episodes``  -- N frozen arm start poses (6 joint angles, rad) + a finger
    opening (sim m, [0, 0.025]) + a physical box marker label. The arm side is
    PRESCRIBED (sim->real: ``moveJ`` is repeatable). The box side is only
    "place it roughly on this tape mark" -- mocap MEASURES the truth and the
    sim mirrors that measurement (D1/D3). That is why an episode carries a
    marker label, not a box pose.
  * ``horizon``   -- exactly H steps per episode, no early termination (D6).
  * ``real_repeats`` -- k real runs per episode; their within-episode spread is
    the measurement noise floor, and no gap smaller than it is reportable (D9).
  * ``poses_sha256`` -- SHA-256 over the episodes block. Every result records
    it; a mismatch means "these numbers are not comparable", and the loader
    refuses rather than silently comparing across a changed episode set (D5).

Why the poses are hashed rather than indexed
--------------------------------------------
An earlier design stored indices into the init-pose library. Indices break the
moment anyone runs ``collect_init_poses.py --append``: the same index silently
means a different pose, and every historical result is quietly invalidated with
no error. Freezing the pose VALUES and hashing them makes that failure loud.

Provenance of the poses (corrects D4 as originally written)
-----------------------------------------------------------
D4 specified synthetic poses drawn around the ``low_home`` keyframe. That was
based on a misreading of the env and is NOT what this module does:

  * ``ur3_pick.default_config()`` sets ``init_keyframe="task_home"``; the
    string "low_home" survives only as a dead ``getattr`` fallback.
  * More importantly, with the baked default ``init_start_random="mid"`` the
    env's ``reset()`` does not use a keyframe at all -- it samples a base pose
    from ``load_init_poses("mid", "train")`` (real, freedrive-recorded UR3e
    configurations) and jitters it with ``init_qpos_noise``.

So synthetic poses around a keyframe would have started every evaluation
episode OUTSIDE the distribution the policies actually trained on, confounding
the sim-to-real gap with an out-of-distribution generalization gap -- and would
still have needed a robot pass to prove each pose reachable. Instead the
protocol draws from the ``test`` split of the SAME library
(``init_poses/<split>/<level>.json``), which per that package's own docstring is
exactly what the split exists for:

    "The SAME files feed BOTH: training -- the env reset samples a start pose
    ... and evaluation -- the sim and real eval harnesses reset/move to each
    held-out pose. train and test poses are collected in SEPARATE sessions ->
    SEPARATE files; no pose appears in both."

This buys three things at once: the poses are real-robot reachable BY
CONSTRUCTION (they were recorded from the arm, not computed), they are in the
training distribution's support, and they were never trained on.
"""

import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, Tuple

PROTOCOL_DIR = os.path.dirname(os.path.abspath(__file__))


def canonical_episode_hash(episodes) -> str:
    """SHA-256 over the episodes block, canonicalized.

    Canonical form = sorted keys, no whitespace, so the hash is stable against
    JSON formatting churn (an ``indent=`` change must NOT invalidate a
    protocol). Only the ``episodes`` block is hashed -- metadata like
    ``created`` is descriptive, and ``horizon``/``real_repeats`` are checked
    separately by the runners. Anything that changes WHICH physical episode is
    executed lives in this block and is therefore covered.
    """
    blob = json.dumps(
        episodes, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class Episode:
    """One frozen protocol episode. Immutable by construction."""

    episode_id: int
    arm_qpos: Tuple[float, ...]  # 6 joint angles, rad, in init_poses.JOINT_ORDER
    finger: float  # sim Hand-E per-finger m, [0, 0.025]; 0 = open
    box_marker: str  # physical tape label; mocap supplies the truth (D1)
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class Protocol:
    """A loaded, hash-verified protocol. Immutable by construction."""

    protocol_id: str
    created: str
    source: Dict[str, Any]
    horizon: int
    real_repeats: int
    validated_on_robot: bool
    episodes: Tuple[Episode, ...]
    poses_sha256: str
    path: str

    def __len__(self) -> int:
        return len(self.episodes)


def load_protocol(path: str, dry_run: bool = False) -> Protocol:
    """Load + verify a protocol JSON.

    Verifies, in order:
      1. the recorded ``poses_sha256`` matches a recomputed hash of the
         episodes block -- catches hand-edits, partial writes, and any drift
         between the artifact and the results that cite its hash;
      2. ``validated_on_robot`` is true (unless ``dry_run``) -- refuses to
         drive real hardware toward poses nobody has confirmed;
      3. episode shapes (6 arm joints, finger in range, unique ids).

    Args:
      path: path to the protocol JSON.
      dry_run: skip only the ``validated_on_robot`` gate -- for offline/sim
        work that never commands the arm. The hash check is NEVER skipped.

    Raises:
      FileNotFoundError: no such protocol.
      ValueError: hash mismatch, unvalidated protocol, or malformed episodes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"protocol not found: {path} (build it with "
            f"evaluation/make_protocol.py)"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_episodes = data.get("episodes", [])
    if not raw_episodes:
        raise ValueError(f"protocol has no episodes: {path}")

    recorded = data.get("poses_sha256")
    actual = canonical_episode_hash(raw_episodes)
    if recorded != actual:
        raise ValueError(
            f"protocol hash mismatch in {path}:\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            f"The episodes block has changed since this file was generated. "
            f"Results recorded against the OLD hash are not comparable to runs "
            f"on this file -- regenerate with evaluation/make_protocol.py and "
            f"treat it as a NEW protocol_id, or restore the original file."
        )

    if not data.get("validated_on_robot", False) and not dry_run:
        raise ValueError(
            f"protocol {path} has validated_on_robot=false. Refusing to hand it "
            f"to the robot. Pass dry_run=True for offline/sim use only."
        )

    episodes = []
    seen_ids = set()
    for i, ep in enumerate(raw_episodes):
        arm = ep["arm_qpos"]
        if len(arm) != 6:
            raise ValueError(
                f"episode {i} in {path}: arm_qpos has length {len(arm)}, "
                f"expected 6"
            )
        finger = float(ep["finger"])
        if not 0.0 <= finger <= 0.025:
            raise ValueError(
                f"episode {i} in {path}: finger={finger} outside the physical "
                f"Hand-E range [0, 0.025] m"
            )
        eid = int(ep["episode_id"])
        if eid in seen_ids:
            raise ValueError(f"duplicate episode_id {eid} in {path}")
        seen_ids.add(eid)
        episodes.append(
            Episode(
                episode_id=eid,
                arm_qpos=tuple(float(x) for x in arm),
                finger=finger,
                box_marker=str(ep["box_marker"]),
                notes=str(ep.get("notes", "")),
            )
        )

    return Protocol(
        protocol_id=str(data["protocol_id"]),
        created=str(data.get("created", "")),
        source=dict(data.get("source", {})),
        horizon=int(data["horizon"]),
        real_repeats=int(data["real_repeats"]),
        validated_on_robot=bool(data.get("validated_on_robot", False)),
        episodes=tuple(episodes),
        poses_sha256=str(recorded),
        path=os.path.abspath(path),
    )
