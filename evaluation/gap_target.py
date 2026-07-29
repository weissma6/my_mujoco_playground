"""Deterministic lift-target sampling, shared VERBATIM between the real
gap-protocol runner (Commit 6) and its matched sim mirror (Commit 7).

Why this module has to exist -- D2 as written in the vault plan is wrong
--------------------------------------------------------------------------
D2 ("Plan - Sim-to-Real Gap Protocol") says: "target = deterministic function
of measured box (mirror the env rule exactly)." That is true for
target_mode="box" (the legacy path), but FALSE for target_mode="base_polar" --
the BAKED DEFAULT every DR-ladder config actually trains with. Reading
ur3_pick.py's reset() (the "base_polar" branch):

    box_bearing = atan2(box_xy - base_xy)
    dphi   ~ uniform(target_azim_min, target_azim_max)   # PER-EPISODE DRAW
    side   = sign(uniform(-1, 1))                        # PER-EPISODE DRAW
    r      ~ uniform(target_r_min, target_r_max)          # PER-EPISODE DRAW
    phi    = box_bearing + side * dphi
    target_xy = base_xy + r * [cos(phi), sin(phi)]
    target_z  ~ uniform(*target_z_jitter) + box_z_anchor + lifter_dz  # PER-EP.

Three of four target components (dphi, side, r) and half of the fourth
(target_z's jitter term) are INDEPENDENT per-episode random draws, not
functions of the box position. Given only a measured box_xy, "the" target
under base_polar is not one point -- it is a whole distribution, and `side`
alone flips it between two mirror-image regions. A real run and its sim
mirror computing "the" target independently, even from the identical
measured box_xy, would with ~50% probability disagree about which SIDE of
the box the lift point is on.

So D1/D2/D3's pairing guarantee (real measures the box, sim mirrors that
exact measurement) is necessary but not sufficient for the target. The
missing piece: dphi/side/r/target_z must ALSO be frozen per protocol episode,
exactly like the arm pose in Commit 5, and reused deterministically by BOTH
domains. That is what this module does. It is a plan-level finding, not a
bug in ur3_pick.py -- the env's randomness is correct and intentional; it is
the protocol's job to freeze a specific draw from it, the same way Commit 5
freezes one draw from the arm-pose distribution.

`side`'s independent randomness is real, not an oversight to "fix" here
--------------------------------------------------------------------------
Note that even the DR-ladder's "fully deterministic" L0_none/LOO_no_pos
config (box_xy_jitter=0, target_azim_min==target_azim_max) still draws `side`
independently -- so L0's target position is not actually a single point, it
is one of two mirror-image points, chosen randomly per training episode. This
module reproduces that faithfully (freezing one of the two per protocol
episode) rather than "fixing" it, because Commit 6/7's job is to replicate
the trained distribution, not correct it.

Why only TWO target profiles cover every real-robot-bound config
--------------------------------------------------------------------------
target_r_min/max, target_azim_min/max, target_z_jitter are untouched by every
DR-ladder line except L0_none and LOO_no_pos (see gen_dr_ladder.py's
_DETERMINISTIC_POSITION) -- L1_pos..L5_full_obs and LOO_no_cube/robot/env all
inherit the SAME baked ur3_pick.default_config() ranges. The plan's
"Real-robot budget" only sends L0_none..L5_full_obs to the robot (LOO stays
sim-only) -- so exactly two profiles are ever needed on real hardware:
"L0_deterministic" (L0_none) and "default" (L1..L5). Both ranges are IMPORTED
from their single source of truth below, never re-typed as literals -- typing
them again here is exactly the mistake that cost 30 GPU-runs today (b170af5).

Why components are frozen ONCE per (protocol, episode, profile), not redrawn
per repeat
--------------------------------------------------------------------------
D9: real repeats exist to measure the MEASUREMENT noise floor -- "no gap
smaller than it is reportable." If the target flipped side/radius/azimuth on
every repeat, the k=3 spread would mix in real TASK variance (a materially
different lift point) with measurement variance, inflating the noise floor
and making it meaningless. Freezing per (protocol_id, episode_id, profile)
means every repeat of the same episode gets the identical target; only the
box's OWN measured pose (which genuinely does vary slightly between
"place it roughly on the marker" repeats) contributes real noise, which is
the intended behaviour.

The lifter term is 0, but the ANCHOR moved (changed 2026-07-28)
--------------------------------------------------------------------------
ur3_pick.py's reset() shifts target_z by the table's per-episode height
DEVIATION from its nominal (and tilts the box). Until 2026-07-28 the "lifter"
was a small 3-20 mm riser with no physical counterpart, the box anchor was the
cube on the FLOOR (0.02), and this module forced the lifter term to 0 ("no
lifter on the real robot"), accepting the resulting one-axis real<->sim gap.

The lifter body now models the actual lab TABLE -- 1.2 x 0.8 m, top surface
95 mm above the UR3e base origin -- which does exist on the real setup. The
table's nominal height is carried by the ANCHOR instead of a separate term:
the task_home keyframe box Z is now 0.115, the cube resting ON the nominal
table. So the formula here is unchanged in shape,

    target_z = target_z_draw + box_z_anchor      # anchor 0.115, was 0.02

and the lifter term stays 0 -- not because there is no table, but because the
only thing left in it is the per-episode DEVIATION, which is exactly the kind
of randomness this module's whole job is to freeze at 0. UR3Pick's
reset_to_state() pins the table level at the same nominal on the sim side, so
both domains agree by construction. scene_constants() asserts the anchor
really is table-top + box-half-height, so a scene edit cannot desync it.

Consequence: protocol runs recorded BEFORE this change store targets 95 mm
lower than target_for_episode now returns, and run_gap_protocol_sim.py's
stored-vs-recomputed check will warn on them. That warning is correct.

Verified (see this module's __main__ smoke test): compute_target_pos's numpy
port of the base_polar formula reproduced against a jax.numpy port of the
SAME formula (independent re-derivation from ur3_pick.py's reset() lines,
not copy-pasted) agrees to float32 precision across 200 random draws.
base_xy=(0,0) and the task_home box anchor (0.4, 0, 0.115) are READ from the
scene XML via plain `mujoco` (mirroring evaluation/ur3_reward_replay.SimFK's
convention), never hardcoded, so a scene change cannot silently desync this
module from the env.
"""

import dataclasses
import hashlib
import os
import sys
from typing import Optional, Tuple

import mujoco
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mujoco_playground._src.manipulation.my_ur3 import ur3_pick  # noqa: E402

# _DETERMINISTIC_POSITION is imported, not re-typed, from its single source of
# truth. gen_dr_ladder.py flags at its definition that this import exists --
# keep that name/shape stable if it ever moves.
from batch_runs.sweeps.gen_dr_ladder import _DETERMINISTIC_POSITION  # noqa: E402

DEFAULT_XML = os.path.join(
    REPO_ROOT,
    "mujoco_playground",
    "_src",
    "manipulation",
    "my_ur3",
    "xmls",
    "mjx_single_cube_position_ur3.xml",
)

# The DR-ladder config -> target profile mapping. Only L0_none/LOO_no_pos use
# the deterministic-position ranges; every other config (incl. every LOO
# config other than no_pos) inherits the SAME baked defaults. See the module
# docstring for why this is exhaustive over every config the plan ever sends
# to the real robot.
# D21 (2026-07-29): the ladder is now 5 configs -- L5_full_obs was absorbed
# into L4_full and the whole LOO block was dropped. See
# batch_runs/sweeps/gen_dr_ladder.py.
CONFIG_TO_PROFILE = {
    "L0_none": "L0_deterministic",
    "L1_pos": "default",
    "L2_pos_cube": "default",
    "L3_pos_cube_robot": "default",
    "L4_full": "default",
}


@dataclasses.dataclass(frozen=True)
class TargetProfile:
    name: str
    target_r_min: float
    target_r_max: float
    target_azim_min: float
    target_azim_max: float
    target_z_jitter: Tuple[float, float]


def _default_profile() -> TargetProfile:
    """"default" profile -- L1_pos..L5_full_obs, LOO_no_{cube,robot,env}.

    Values read from ur3_pick.default_config() at import time -- if that
    default ever changes, this profile changes with it automatically.
    """
    cfg = ur3_pick.default_config()
    return TargetProfile(
        name="default",
        target_r_min=float(cfg.target_r_min),
        target_r_max=float(cfg.target_r_max),
        target_azim_min=float(cfg.target_azim_min),
        target_azim_max=float(cfg.target_azim_max),
        target_z_jitter=tuple(float(x) for x in cfg.target_z_jitter),
    )


def _l0_profile() -> TargetProfile:
    """"L0_deterministic" profile -- L0_none, LOO_no_pos.

    Values read from gen_dr_ladder._DETERMINISTIC_POSITION -- the actual dict
    written into those sweep lines -- not re-derived here.
    """
    d = _DETERMINISTIC_POSITION
    return TargetProfile(
        name="L0_deterministic",
        target_r_min=float(d["target_r_min"]),
        target_r_max=float(d["target_r_max"]),
        target_azim_min=float(d["target_azim_min"]),
        target_azim_max=float(d["target_azim_max"]),
        target_z_jitter=tuple(float(x) for x in d["target_z_jitter"]),
    )


TARGET_PROFILES = {"default": _default_profile(), "L0_deterministic": _l0_profile()}


@dataclasses.dataclass(frozen=True)
class TargetComponents:
    """The 4 per-episode random draws base_polar needs beyond the box pos."""

    dphi: float
    side: float  # +1.0 or -1.0 (np.sign; 0.0 only at probability-zero exactly-0 draw)
    r: float
    target_z_draw: float
    profile_name: str
    seed: int


def sample_target_components(
    protocol_id: str, episode_id: int, config_id: str
) -> TargetComponents:
    """Deterministically draw dphi/side/r/target_z for one protocol episode.

    Seeded from sha256(protocol_id|episode_id|profile_name) so it is stable
    across processes/machines/Python versions (unlike Python's built-in
    hash(), which is randomized per-process for strings) and reproducible by
    Commit 7's sim rollout without sharing any runtime state with the real
    runner -- it only needs (protocol_id, episode_id, config_id), all three
    already logged in every run's meta.json.
    """
    profile_name = CONFIG_TO_PROFILE[config_id]
    profile = TARGET_PROFILES[profile_name]
    seed_str = f"{protocol_id}|{episode_id}|{profile_name}"
    seed = int.from_bytes(hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    dphi = float(rng.uniform(profile.target_azim_min, profile.target_azim_max))
    # D18/D24 (2026-07-29): freeze `side` for the L0_deterministic profile.
    # L0's fixed target must equal the D22 eval drop point EXACTLY
    # (r=0.30, azim=45 deg, world z 0.165). With a randomly drawn side, L0
    # would still flip between two mirror-image targets each episode -- which
    # was tolerable when L0 was merely "deterministic position", but is not
    # once L0 has one specific physical drop square to reproduce. Every other
    # profile ("default") keeps drawing side randomly, unchanged: that
    # randomness is a genuine property of the trained distribution.
    if profile_name == "L0_deterministic":
        side = 1.0
    else:
        side = float(np.sign(rng.uniform(-1.0, 1.0)))
    r = float(rng.uniform(profile.target_r_min, profile.target_r_max))
    target_z_draw = float(
        rng.uniform(profile.target_z_jitter[0], profile.target_z_jitter[1])
    )
    return TargetComponents(
        dphi=dphi, side=side, r=r, target_z_draw=target_z_draw,
        profile_name=profile_name, seed=seed,
    )


_SCENE_CACHE = {}


def scene_constants(xml_path: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """(base_xy, box_z_anchor) read from the scene XML via plain mujoco.

    Mirrors evaluation/ur3_reward_replay.SimFK's model-loading convention
    (mujoco.MjModel.from_xml_path, not mjx). Cached per xml_path so repeated
    calls (once per real episode) don't reload/reparse the XML each time.

    box_z_anchor is the task_home keyframe box Z, which since 2026-07-28 is the
    cube RESTING ON THE NOMINAL TABLE (0.115 = 0.095 table top + 0.02 box half-
    height), not on the floor. That is checked here against the table geometry
    read from the same model -- ur3_pick.py's _post_init asserts the identical
    invariant on the training side, so a scene edit that breaks it fails loud in
    both domains instead of silently shifting the real-robot lift target.
    """
    xml_path = xml_path or DEFAULT_XML
    if xml_path not in _SCENE_CACHE:
        model = mujoco.MjModel.from_xml_path(xml_path)
        base_xy = np.array(model.body("base").pos[:2], dtype=float)
        box_jntadr = model.body("box").jntadr[0]
        box_qposadr = int(model.jnt_qposadr[box_jntadr])
        box_z_anchor = float(model.key("task_home").qpos[box_qposadr + 2])
        table_top = float(
            model.body("lifter").pos[2] + model.geom("lifter").size[2]
        )
        box_half_z = float(model.geom("box").size[2])
        want = table_top + box_half_z
        if abs(box_z_anchor - want) > 1e-9:
            raise ValueError(
                f"{xml_path}: task_home box Z ({box_z_anchor:.6f}) is not the "
                f"cube resting on the nominal table ({want:.6f} = table top "
                f"{table_top:.6f} + box half-height {box_half_z:.6f}). The lift "
                f"target is anchored to this value, so real and sim would "
                f"disagree by the difference."
            )
        _SCENE_CACHE[xml_path] = (base_xy, box_z_anchor)
    return _SCENE_CACHE[xml_path]


def compute_target_pos(
    box_xy: np.ndarray,
    components: TargetComponents,
    base_xy: np.ndarray,
    box_z_anchor: float,
) -> np.ndarray:
    """Port of ur3_pick.py reset()'s base_polar branch (lines ~901-943).

    No separate lifter term: box_z_anchor IS the cube on the nominal table, and
    the per-episode table deviation is frozen to 0 here -- see the module
    docstring ("Lifter component = the real table's nominal height").
    """
    box_xy = np.asarray(box_xy, dtype=float)
    box_bearing = np.arctan2(box_xy[1] - base_xy[1], box_xy[0] - base_xy[0])
    phi = box_bearing + components.side * components.dphi
    target_xy = base_xy + components.r * np.array([np.cos(phi), np.sin(phi)])
    target_z = components.target_z_draw + box_z_anchor
    return np.array([target_xy[0], target_xy[1], target_z], dtype=float)


def target_for_episode(
    protocol_id: str,
    episode_id: int,
    config_id: str,
    box_xy,
    xml_path: Optional[str] = None,
) -> Tuple[np.ndarray, TargetComponents]:
    """The one call site both the real runner and the sim mirror should use.

    Args:
      protocol_id: e.g. "gap_v1" (Protocol.protocol_id).
      episode_id: the protocol episode index.
      config_id: DR-ladder config id (e.g. "L2_pos_cube") -- selects the
        target profile via CONFIG_TO_PROFILE.
      box_xy: (2,) measured (real) or mirrored (sim) box XY, base frame.
      xml_path: override scene XML; default matches the deployed policies.

    Returns:
      (target_pos (3,) float64, the TargetComponents drawn -- log both; the
      components make the draw auditable and are needed nowhere else, but
      cost nothing to keep).
    """
    if config_id not in CONFIG_TO_PROFILE:
        raise ValueError(
            f"unknown config_id {config_id!r}; not in CONFIG_TO_PROFILE. If "
            f"this is a new DR-ladder config, add it there first -- silently "
            f"defaulting to a profile would risk evaluating against the "
            f"wrong target-sampling range."
        )
    components = sample_target_components(protocol_id, episode_id, config_id)
    base_xy, box_z_anchor = scene_constants(xml_path)
    target_pos = compute_target_pos(box_xy, components, base_xy, box_z_anchor)
    return target_pos, components


if __name__ == "__main__":
    # Smoke test: verify compute_target_pos's numpy port against an
    # INDEPENDENTLY re-derived jax.numpy port of the same formula (not
    # copy-pasted), over many random draws, box positions, and both profiles.
    import jax.numpy as jnp

    def _jax_base_polar(box_xy, base_xy, dphi, side, r, target_z_draw, box_z_anchor):
        bearing = jnp.arctan2(box_xy[1] - base_xy[1], box_xy[0] - base_xy[0])
        phi = bearing + side * dphi
        txy = base_xy + r * jnp.array([jnp.cos(phi), jnp.sin(phi)])
        tz = target_z_draw + box_z_anchor
        return jnp.array([txy[0], txy[1], tz])

    base_xy, box_z_anchor = scene_constants()
    print(f"base_xy={base_xy}, box_z_anchor={box_z_anchor}")

    rng = np.random.default_rng(0)
    max_err = 0.0
    n = 0
    for profile_name, profile in TARGET_PROFILES.items():
        for _ in range(200):
            box_xy = rng.uniform([0.25, -0.25], [0.65, 0.25])
            comp = TargetComponents(
                dphi=float(rng.uniform(profile.target_azim_min, profile.target_azim_max)),
                side=float(np.sign(rng.uniform(-1.0, 1.0))),
                r=float(rng.uniform(profile.target_r_min, profile.target_r_max)),
                target_z_draw=float(rng.uniform(*profile.target_z_jitter)),
                profile_name=profile_name,
                seed=0,
            )
            got = compute_target_pos(box_xy, comp, base_xy, box_z_anchor)
            want = np.array(
                _jax_base_polar(
                    jnp.asarray(box_xy), jnp.asarray(base_xy), comp.dphi,
                    comp.side, comp.r, comp.target_z_draw, box_z_anchor,
                )
            )
            err = float(np.max(np.abs(got - want)))
            max_err = max(max_err, err)
            n += 1
    print(f"compute_target_pos vs independent jax re-derivation: "
          f"{n} draws, max abs error {max_err:.2e} "
          f"({'PASS' if max_err < 1e-5 else 'FAIL'})")

    # Determinism + profile sanity.
    t1, c1 = target_for_episode("gap_v1", 0, "L1_pos", box_xy=(0.42, 0.05))
    t2, c2 = target_for_episode("gap_v1", 0, "L1_pos", box_xy=(0.42, 0.05))
    print(f"determinism (same call twice): "
          f"{'PASS' if np.allclose(t1, t2) and c1 == c2 else 'FAIL'}")
    t0, c0 = target_for_episode("gap_v1", 0, "L0_none", box_xy=(0.42, 0.05))
    print(f"L0_none uses L0_deterministic profile: "
          f"{'PASS' if c0.profile_name == 'L0_deterministic' else 'FAIL'}")
    print(f"L1_pos uses default profile: "
          f"{'PASS' if c1.profile_name == 'default' else 'FAIL'}")
