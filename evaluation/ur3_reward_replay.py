"""Reconstruct the UR3Pick per-term training reward from LOGGED geometry.

Single source of truth used by:
  - the real-robot pickloop (robots/UR3e/ur3_realrobot_pickloop.py), which
    replays a logged run through `replay_dataframe` to plot the same reward
    terms the policy was trained on (Task 1);
  - the train-vs-sim-vs-real comparison grid
    (evaluation/compare_reward_train_sim_real.py), which additionally uses
    `sim_rollout_reward` for MuJoCo (MJX) rollouts (Task 2).

Design: `SimFK` drives the UR3Pick XML with PLAIN MuJoCo forward kinematics
(mujoco.MjModel/MjData/mj_forward -- NOT MJX) from full qpos (6 arm joints +
2 finger joints + the box's 7D freejoint pose), so it reads the exact same
`tcp` / `left_finger_touch_site` / `right_finger_touch_site` sites and box
`xmat` the training env's `_get_reward` uses. `RewardReplayer` is a line-for-
line numpy port of `ur3_pick.py`'s `_get_reward`, keyed to
`default_config().reward_config.scales` (and `grasp_align_thresh`,
`hold_radius`, `hold_tau`, `lift_eps`) so it can never silently drift from
the live training weights -- those are imported, never hardcoded here.

PRECISION CAVEAT (measured 2026-08, do not re-litigate): this module's FK is
float64 MuJoCo, while the training env runs MJX in float32 (MJX cannot run
under JAX_ENABLE_X64 -- its integer carry dtypes break jax.lax.scan). Feeding
the same qpos to both gives TCP / finger-site positions that differ by ~0.6 mm,
which propagates to ~3e-2 in `alignment`. That is MJX round-off, NOT a port
error: fed IDENTICAL geometry (the env's own data.site_xpos and data.xmat), the
two implementations agree to 2.4e-6 on alignment and 5e-9 on span_jaw, in both
align_mode branches. So compare formulas on shared geometry, never by handing
this module a qpos read out of an MJX rollout.

On the real robot, contact (`grasp`) and floor-collision are not directly
observable the way sim's touch sensors are; the caller supplies a proxy
(Hand-E object-detection flag for `grasp_contact`, default no floor contact).
All GEOMETRIC terms (gripper_box, box_target, gripper_align, lift,
robot_target_qpos, action_rate) are exact given accurate logged geometry.

`sim_rollout_reward` (MJX -- do NOT run locally; no GPU here and CPU MJX
OOM-kills this machine, see CLAUDE.md) reuses the
evaluation/render_ur3_policy_rollout.py load+rollout pattern and collects
`state.metrics[<term>]` (raw, unscaled, produced every step by the env)
multiplied by SCALES, plus the geometry needed for a replayer-vs-sim parity
check.

Usage (module import):
    from ur3_reward_replay import replay_dataframe, save_reward_terms_plot
    reward_df = replay_dataframe(df, xml_path=MODEL_PATH, target=DROP_TARGET)
    save_reward_terms_plot(reward_df, "run_reward.png", title="...")

Usage (smoke test, pure MuJoCo FK, safe to run locally -- no MJX, no robot):
    python evaluation/ur3_reward_replay.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mujoco_playground._src.manipulation.my_ur3 import ur3_pick  # noqa: E402

# ===========================================================================
# Reward config -- imported from the env, never hardcoded here.
# ===========================================================================

_CFG = ur3_pick.default_config()
SCALES = {k: float(v) for k, v in _CFG.reward_config.scales.items()}
GRASP_ALIGN_THRESH = float(_CFG.grasp_align_thresh)
HOLD_RADIUS = float(_CFG.hold_radius)
HOLD_TAU = float(_CFG.hold_tau)
LIFT_EPS = float(_CFG.lift_eps)
SUCCESS_TOL = float(_CFG.success_tol)

# Box half-extents per protocol cube size, mirroring
# evaluation/run_gap_protocol_sim.CUBE_HALF_EXTENTS. Needed by the axis-aware
# alignment (jaw-span preference is measured against the REALIZED box).
CUBE_HALF_EXTENTS = {
    "3cm": (0.015, 0.015, 0.020),
    "4cm": (0.020, 0.020, 0.020),
}
_POLICY_ROOT = os.path.join(REPO_ROOT, "evaluation", "downloaded_policies")
# Built configs, keyed by (policy_run_id, root). gap_metrics calls
# config_for_policy once per RUN (179 leaf dirs in the D23 campaign) but there
# are only ~5 distinct policies, so without this the disk read and the
# provenance print would repeat ~36x each. Sharing the ConfigDict is safe:
# RewardReplayer.__init__ copies every value it needs out into plain floats.
_CONFIG_CACHE = {}


def config_for_policy(policy_run_id, policy_root=None):
    """Return ur3_pick.default_config() with a TRAINED run's env_overrides applied.

    Why this exists: this module used to score every real-robot run with the
    module-level SCALES / GRASP_ALIGN_THRESH read from default_config() at
    import time. That was fine while every deployed policy shared the defaults,
    but the v3 ladder trains with align_mode="axis_aware", a raised
    grasp_align_thresh, and reduced action_rate / robot_target_qpos. Scoring a
    v3 real run with v2 reward would silently compare two different objectives
    and corrupt every retention number -- exactly the class of bug the stale
    two-scale cascade already caused once.

    Reads evaluation/downloaded_policies/<policy_run_id>/metadata.json, whose
    "env_overrides" is written verbatim from the training run's
    inference_config.json. Only a value that appears there is a real trained
    value (see robots/UR3e/ur3_realrobot_pickloop.py's _resolve_scale, which
    refuses to trust metadata.json's bare top-level action_scale for the same
    reason). Returns the plain default config, with a printed warning, if the
    policy directory or the key is missing -- old campaigns keep working.
    """
    cfg = ur3_pick.default_config()
    if not policy_run_id:
        return cfg
    root = policy_root or _POLICY_ROOT
    ck = (str(policy_run_id), root)
    if ck in _CONFIG_CACHE:
        return _CONFIG_CACHE[ck]
    meta_path = os.path.join(root, str(policy_run_id), "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"[ur3_reward_replay] warning: no metadata.json for policy "
              f"{policy_run_id!r} under {root} -- scoring with DEFAULT reward "
              f"config. If this policy trained with non-default reward scales "
              f"or align_mode, its returns will be wrong.")
        return cfg
    with open(meta_path, "r", encoding="utf-8") as f:
        overrides = (json.load(f) or {}).get("env_overrides") or {}
    applied = {}
    for k, v in overrides.items():
        if k.startswith("reward_config.scales."):
            name = k.split(".")[-1]
            if name in cfg.reward_config.scales:
                cfg.reward_config.scales[name] = float(v)
                applied[k] = float(v)
        elif k in ("grasp_align_thresh", "align_pref_floor", "success_tol",
                   "lift_eps", "hold_radius", "hold_tau"):
            cfg[k] = float(v)
            applied[k] = float(v)
        elif k == "align_mode":
            cfg[k] = str(v)
            applied[k] = str(v)
    if applied:
        print(f"[ur3_reward_replay] {policy_run_id}: reward config from "
              f"metadata.json -> {applied}")
    _CONFIG_CACHE[ck] = cfg
    return cfg
TERMS = list(SCALES.keys())  # gripper_box, approach_open, grasp, lift, box_target,
                              # hold_target, gripper_align, no_floor_collision,
                              # robot_target_qpos, action_rate

_DEFAULT_XML = os.path.join(
    REPO_ROOT,
    "mujoco_playground", "_src", "manipulation", "my_ur3", "xmls",
    "mjx_single_cube_position_ur3.xml",
)

_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
_FINGER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]


# ===========================================================================
# SimFK -- plain MuJoCo forward kinematics (NOT MJX). Cheap, CPU, safe to run
# locally and on the robot PC.
# ===========================================================================


class SimFK:
    """Drives the UR3Pick MuJoCo scene with full qpos and reads back the
    exact sites/axes the training reward uses. No MJX -- `mujoco.MjModel` /
    `MjData` / `mj_forward` only.
    """

    def __init__(self, xml_path: str = None):
        import mujoco  # local import: keep mujoco off the module import path
                        # for callers that only need SCALES/TERMS.

        self._mujoco = mujoco
        xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.tcp_site = self.model.site("tcp").id
        self.left_touch_site = self.model.site("left_finger_touch_site").id
        self.right_touch_site = self.model.site("right_finger_touch_site").id
        self.box_body = self.model.body("box").id

        self.arm_qposadr = np.array(
            [self.model.jnt_qposadr[self.model.joint(j).id] for j in _ARM_JOINTS]
        )
        self.finger_qposadr = np.array(
            [self.model.jnt_qposadr[self.model.joint(j).id] for j in _FINGER_JOINTS]
        )
        box_jntadr = self.model.body("box").jntadr[0]
        self.box_qposadr = int(self.model.jnt_qposadr[box_jntadr])

    def geom(self, arm_q, finger_m, box_pos, box_quat) -> dict:
        """Set qpos (arm + fingers + box freejoint), forward-kinematics, and
        return the sites/axes `_get_reward` needs.

        Args:
          arm_q: (6,) radians.
          finger_m: scalar meters in [0, 0.025] (both fingers, symmetric --
            matches the sim tendon actuator's coef 0.5/0.5).
          box_pos: (3,) world meters.
          box_quat: (4,) MuJoCo convention (w, x, y, z).
        """
        d = self.data
        d.qpos[self.arm_qposadr] = np.asarray(arm_q, dtype=float)
        d.qpos[self.finger_qposadr] = float(finger_m)
        d.qpos[self.box_qposadr : self.box_qposadr + 3] = np.asarray(
            box_pos, dtype=float
        )
        q = np.asarray(box_quat, dtype=float)
        n = np.linalg.norm(q)
        d.qpos[self.box_qposadr + 3 : self.box_qposadr + 7] = (
            q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
        )
        d.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, d)
        return {
            "tcp": d.site_xpos[self.tcp_site].copy(),
            "lft": d.site_xpos[self.left_touch_site].copy(),
            "rgt": d.site_xpos[self.right_touch_site].copy(),
            "box_axes": d.xmat[self.box_body].reshape(3, 3).copy(),
            "box_pos": d.xpos[self.box_body].copy(),
        }


def frame_alignment_np(jaw_axis, app_axis, box_axes, cos_bound=0.5):
    """Numpy port of alignment.frame_alignment (same math, no jax import --
    this module runs inside the real-robot 50 Hz loop).

    jaw_axis, app_axis: (3,) vectors in world frame (renormalised inside).
    box_axes: (3, 3) rotation matrix, columns = box body axes in world frame.
    Returns {"jaw", "app", "third", "face"}: floats in [0, 1].
    """
    jaw_axis = np.asarray(jaw_axis, dtype=float)
    app_axis = np.asarray(app_axis, dtype=float)
    box_axes = np.asarray(box_axes, dtype=float)

    jaw_axis = jaw_axis / np.maximum(np.linalg.norm(jaw_axis), 1e-6)
    app_axis = app_axis / np.maximum(np.linalg.norm(app_axis), 1e-6)
    third_axis = np.cross(jaw_axis, app_axis)
    third_axis = third_axis / np.maximum(np.linalg.norm(third_axis), 1e-6)

    def _score(axis):
        a = float(np.max(np.abs(axis @ box_axes)))
        return float(np.clip((a - cos_bound) / (1.0 - cos_bound), 0.0, 1.0))

    jaw = _score(jaw_axis)
    app = _score(app_axis)
    third = _score(third_axis)
    face = jaw * app * third
    return {"jaw": jaw, "app": app, "third": third, "face": face}


# ===========================================================================
# RewardReplayer -- numpy port of ur3_pick.py's _get_reward. Sticky-latch
# semantics (sticky_latches=True, the default) -- mirrors info["reached"],
# info["grasped"], info["lifted"], info["hold_counter"], info["dist_min"].
# ===========================================================================


class RewardReplayer:
    """Stateful per-episode reward replayer. Call `reset()` once per episode,
    then `step()` once per logged tick, in order.
    """

    def __init__(self, fk: SimFK, cfg=None):
        """cfg: the env config THIS policy trained with (see
        `config_for_policy`). None -> ur3_pick.default_config(), i.e. the
        module-level SCALES / GRASP_ALIGN_THRESH, which is what every caller got
        before per-run configs existed and keeps the v2 campaign reproducible.
        """
        self.fk = fk
        self.cfg = ur3_pick.default_config() if cfg is None else cfg
        self.scales = {k: float(v) for k, v in self.cfg.reward_config.scales.items()}
        self.grasp_align_thresh = float(self.cfg.grasp_align_thresh)
        self.hold_radius = float(self.cfg.hold_radius)
        self.hold_tau = float(self.cfg.hold_tau)
        self.lift_eps = float(self.cfg.lift_eps)
        self.success_tol = float(self.cfg.success_tol)
        # align_mode / align_pref_floor are absent from configs produced before
        # the v3 reward fix, so fall back to the legacy behaviour rather than
        # raising -- this module must keep replaying archived campaigns.
        self.align_axis_aware = (
            str(self.cfg.get("align_mode", "axis_free")) == "axis_aware"
        )
        self.align_pref_floor = float(self.cfg.get("align_pref_floor", 0.15))
        # Pad-to-pad jaw opening, parsed from the same geoms ur3_pick.__init__
        # parses so the two can never disagree.
        _lf = fk.model.geom("left_finger_collision")
        _rf = fk.model.geom("right_finger_collision")
        self.jaw_opening = float(
            (_rf.pos[0] - _rf.size[0]) - (_lf.pos[0] + _lf.size[0])
        )
        self.reset()

    def reset(self, box_rest_z: float = None, init_arm_q=None, box_half=None):
        self.box_rest_z = None if box_rest_z is None else float(box_rest_z)
        self.init_arm_q = (
            None if init_arm_q is None else np.asarray(init_arm_q, dtype=float)
        )
        # Realized box half-extents (m) for this episode -- drives the
        # axis-aware jaw-span preference. Defaults to the nominal 3x3x4 cm
        # prism; the D17 4 cm probe must pass CUBE_HALF_EXTENTS["4cm"] or its
        # jaw preference is computed against the wrong geometry.
        self.box_half = np.array(
            CUBE_HALF_EXTENTS["3cm"] if box_half is None else box_half,
            dtype=float,
        )
        self.reached = 0.0
        self.grasped = 0.0
        self.lifted = 0.0
        self.hold_counter = 0
        self.dist_min = 1e3
        self.prev_action = np.zeros(7, dtype=float)
        # success_bonus state. TERMS is derived from reward_config.scales, which
        # carries success_bonus, so `raw` must carry it too or every consumer
        # that iterates TERMS raises KeyError (the module smoke test did).
        # Mirrors ur3_pick.py:2146-2165: 3 CONSECUTIVE in-tolerance steps arm
        # `success`, and the bonus is EDGE-triggered so it pays at most once.
        self.success_counter = 0
        self.success_ever = 0.0

    def step(
        self,
        arm_q,
        finger_m: float,
        box_pos,
        box_quat,
        target_pos,
        action,
        grasp_contact: bool,
        floor_collision: bool = False,
    ) -> dict:
        arm_q = np.asarray(arm_q, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 7:
            action = np.concatenate([action, np.zeros(7 - action.shape[0])])

        if self.box_rest_z is None:
            self.box_rest_z = float(np.asarray(box_pos, dtype=float)[2])
        if self.init_arm_q is None:
            self.init_arm_q = arm_q.copy()

        geom = self.fk.geom(arm_q, finger_m, box_pos, box_quat)
        tcp, lft, rgt = geom["tcp"], geom["lft"], geom["rgt"]
        box_axes, box_pos_fk = geom["box_axes"], geom["box_pos"]

        # --- Distances ---
        gripper_box_dist = float(np.linalg.norm(box_pos_fk - tcp))
        finger_touch_dist = float(np.linalg.norm(rgt - lft))
        box_target_dist = float(np.linalg.norm(target_pos - box_pos_fk))
        self.dist_min = min(self.dist_min, box_target_dist)

        # --- Grasp-frame alignment (3 axes: jaw, app, third = jaw x app) --
        # mirrors ur3_pick.py's moved-up alignment block. ---
        jaw_axis = rgt - lft
        jaw_axis = jaw_axis / np.maximum(np.linalg.norm(jaw_axis), 1e-6)
        app_axis = 0.5 * (lft + rgt) - tcp
        app_axis = app_axis / np.maximum(np.linalg.norm(app_axis), 1e-6)
        _cos_bound = float(ur3_pick._COS_BOUND)
        _scores = frame_alignment_np(jaw_axis, app_axis, box_axes, _cos_bound)
        a_jaw, a_app, align_face = _scores["jaw"], _scores["app"], _scores["face"]
        # Box support width along the jaw axis (see ur3_pick._get_reward).
        span_jaw = float(2.0 * (self.box_half @ np.abs(jaw_axis @ box_axes)))
        if self.align_axis_aware:
            # Line-for-line mirror of ur3_pick._get_reward's axis_aware branch.
            clear_now = self.jaw_opening - span_jaw
            clear_best = self.jaw_opening - 2.0 * float(np.min(self.box_half))
            jaw_pref = float(np.clip(clear_now / (clear_best + 1e-9), 0.0, 1.0))
            f = self.align_pref_floor
            pref = f + (1.0 - f) * jaw_pref
            # Gated on the PREVIOUS step's sticky lifted latch, matching the env
            # (which reads info["lifted"] before the latch block updates it).
            alignment = align_face * (1.0 if self.lifted > 0.5 else pref)
        else:
            jaw_pref = 1.0
            alignment = align_face
        # Proximity gate WIDENED tanh(5d)->tanh(3d), matching ur3_pick.py.
        gripper_align_Reward = alignment * (1.0 - np.tanh(3.0 * gripper_box_dist))

        # --- Sticky stage latches ---
        reached_now = gripper_box_dist < 0.015
        self.reached = max(self.reached, float(reached_now))
        # GATED on alignment (grasp_align_thresh), matching ur3_pick.py.
        grasp_now = (
            (self.reached > 0.5)
            and bool(grasp_contact)
            and (alignment > self.grasp_align_thresh)
        )
        self.grasped = max(self.grasped, float(grasp_now))
        box_off_rest = box_pos_fk[2] > (self.box_rest_z + self.lift_eps)
        lift_now = box_off_rest and (self.grasped > 0.5)
        self.lifted = max(self.lifted, float(lift_now))

        reached, grasped, lifted = self.reached, self.grasped, self.lifted

        # --- Reward terms (staged by the latches above) ---
        finger_open = float(np.tanh(finger_touch_dist / 0.05))
        finger_closed = 1.0 - finger_open

        # THREE-scale cascade, mirroring ur3_pick.py:2590-2594. This replayer
        # carried the old TWO-scale form (0.5*(1-tanh(5d)) + 0.5*(1-tanh(30d)))
        # long after the env gained the coarse 1.5 scale in commit 7c83352, so
        # every real-robot return it produced was under-counted -- by 0.59
        # raw/step at d=0.40 m and 0.79 at 0.20 m, i.e. worst exactly where the
        # real arm spends most of its time. Because the real arm is ~5x slower
        # than sim (deploy control-law rebase, see robots/UR3e/
        # ur3_realrobot_dependencies.py), it dwells at long range far longer
        # than the sim rollouts it is compared against, so the bias was
        # ASYMMETRIC and inflated every reported sim->real gap.
        gripper_box_Reward = (
            (1.0 - np.tanh(1.5 * gripper_box_dist))
            + (1.0 - np.tanh(5.0 * gripper_box_dist))
            + (1.0 - np.tanh(30.0 * gripper_box_dist))
        ) / 3.0
        # Shaped by alignment (continuous), matching ur3_pick.py.
        grasp_Reward = finger_closed * reached * alignment
        approach_open_Reward = finger_open * (1.0 - reached)

        lift_height = float(np.clip(box_pos_fk[2] - self.box_rest_z, 0.0, 0.12))
        lift_Reward = float(np.tanh(lift_height / 0.06)) * reached

        # THREE-scale cascade, mirroring ur3_pick.py:2625-2629. Note the scales
        # are (1.5, 6, 40) here, NOT gripper_box's (1.5, 5, 30). Same stale-form
        # bug as gripper_box above and far larger in absolute terms because
        # box_target's weight is 20.0: the old two-scale form under-counted by
        # 3.03 raw/step at d=0.40 m and 4.17 at 0.20 m.
        box_target_Reward = (
            (1.0 - np.tanh(1.5 * box_target_dist))
            + (1.0 - np.tanh(6.0 * box_target_dist))
            + (1.0 - np.tanh(40.0 * box_target_dist))
        ) / 3.0 * lifted

        in_hold = (lifted > 0.5) and (box_target_dist < self.hold_radius)
        self.hold_counter = self.hold_counter + 1 if in_hold else 0
        hold_target_Reward = float(in_hold) * float(
            np.tanh(self.hold_counter / self.hold_tau)
        )

        robot_target_qpos_penalty = (
            1.0 - np.tanh(float(np.linalg.norm(arm_q - self.init_arm_q)))
        ) * (1.0 - lifted)

        action_rate_Reward = float(np.sum(np.square(action - self.prev_action)))
        self.prev_action = action.copy()

        # No floor-contact sensor on the real robot -- default no collision
        # (caller may pass floor_collision=True from another signal).
        no_floor_collision_Reward = 1.0 - float(bool(floor_collision))

        # Edge-triggered success bonus, mirroring ur3_pick.py:2146-2165. The
        # DEFAULT scale is 0.0 (deliberately off -- see the hover-farming note
        # in ur3_pick.default_config()), so this contributes nothing today; it
        # is computed faithfully rather than stubbed so the replayer stays
        # correct if a sweep ever turns success_bonus on. Note gap_metrics.py's
        # TERM_ORDER does not include it, so it never reaches the report.
        success_now = box_target_dist < self.success_tol
        self.success_counter = self.success_counter + 1 if success_now else 0
        newly_succeeded = (self.success_counter >= 3) and (self.success_ever < 0.5)
        self.success_ever = max(self.success_ever, float(self.success_counter >= 3))

        raw = {
            "success_bonus": float(newly_succeeded),
            "gripper_box": gripper_box_Reward,
            "approach_open": approach_open_Reward,
            "grasp": grasp_Reward,
            "lift": lift_Reward,
            "box_target": box_target_Reward,
            "hold_target": hold_target_Reward,
            "gripper_align": gripper_align_Reward,
            "no_floor_collision": no_floor_collision_Reward,
            "robot_target_qpos": robot_target_qpos_penalty,
            "action_rate": action_rate_Reward,
        }
        scaled = {k: raw[k] * self.scales[k] for k in raw}
        total = float(np.clip(sum(scaled.values()), -1e4, 1e4))

        out = dict(scaled)
        out.update({f"raw_{k}": v for k, v in raw.items()})
        out.update(
            {
                "reward_total": total,
                "reached": reached,
                "grasped": grasped,
                "lifted": lifted,
                "a_jaw": a_jaw,
                "a_app": a_app,
                # Mirrors ur3_pick._get_reward's raw_signals; align_face /
                # jaw_pref / span_jaw exist in BOTH modes (the preference is
                # 1.0 under axis_free) so the parity test against the env's
                # metrics can compare them unconditionally.
                "align_face": align_face,
                "jaw_pref": jaw_pref,
                "span_jaw": span_jaw,
                "alignment": alignment,
                "gripper_box_dist": gripper_box_dist,
                "box_target_dist": box_target_dist,
                "dist_min": self.dist_min,
            }
        )
        return out


# ===========================================================================
# Real-robot convenience: replay a logged pickloop DataFrame.
# ===========================================================================


def replay_dataframe(
    df: pd.DataFrame,
    xml_path: str = None,
    target=None,
    contact_col: str = "grasped",
    box_rest_z: float = None,
    cfg=None,
    box_half=None,
) -> pd.DataFrame:
    """Replay a logged real-robot run (as produced by
    `UR3RealRobotPick.run_policy_loop`) through `RewardReplayer`.

    cfg: the env config the REPLAYED POLICY trained with -- build it with
      `config_for_policy(meta["policy_run_id"])`. None keeps the module
      defaults, which is correct for every policy trained before the v3 reward
      fix and wrong for every one after it (a v3 policy trained with
      align_mode="axis_aware" and grasp_align_thresh=0.45 scored under v2
      reward would have a completely different `grasped` latch, and therefore
      different grasp/lift/box_target/hold_target returns).
    box_half: (3,) realized box half-extents, from
      CUBE_HALF_EXTENTS[meta["cube_size"]]. Only affects the axis-aware jaw-span
      preference; None -> the nominal 3 cm prism.

    Expects columns q0..q5, finger_pos_est, box_x/y/z, action0..action6, and
    (ideally) box_qw/qx/qy/qz (from the mocap-orientation logging commit) --
    falls back to an identity box quaternion with a printed warning if those
    are absent, so old logs (pre box-quat logging) still plot the geometric
    terms (gripper_align / grasp gating will be approximate without real box
    orientation). `contact_col` (default "grasped") supplies the real grasp
    proxy per step (Hand-E object-detection flag); absent -> always False.
    target: fixed (3,) drop target, or None to read per-row target_x/y/z.
    """
    if len(df) == 0:
        return pd.DataFrame()

    fk = SimFK(xml_path)
    replayer = RewardReplayer(fk, cfg=cfg)

    has_quat = all(c in df.columns for c in ["box_qw", "box_qx", "box_qy", "box_qz"])
    if not has_quat:
        print(
            "[ur3_reward_replay] warning: no box orientation columns "
            "(box_qw/box_qx/box_qy/box_qz) in this log -- using an identity "
            "box quaternion. gripper_align / the alignment-gated grasp latch "
            "will be approximate; all other geometric terms are unaffected."
        )

    has_action = all(f"action{i}" in df.columns for i in range(7))
    has_contact = contact_col in df.columns
    if not has_contact:
        print(
            f"[ur3_reward_replay] warning: no '{contact_col}' column -- "
            "treating grasp_contact as always False (grasp/lift/box_target/"
            "hold_target will stay at 0)."
        )

    if box_rest_z is None:
        box_rest_z = float(df.iloc[0]["box_z"])
    init_arm_q = df.iloc[0][[f"q{i}" for i in range(6)]].to_numpy(dtype=float)
    replayer.reset(box_rest_z=box_rest_z, init_arm_q=init_arm_q,
                   box_half=box_half)

    rows = []
    prev_action = np.zeros(7, dtype=float)
    for _, r in df.iterrows():
        arm_q = r[[f"q{i}" for i in range(6)]].to_numpy(dtype=float)
        finger_m = float(r.get("finger_pos_est", 0.0))
        box_pos = r[["box_x", "box_y", "box_z"]].to_numpy(dtype=float)
        box_quat = (
            r[["box_qw", "box_qx", "box_qy", "box_qz"]].to_numpy(dtype=float)
            if has_quat
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        if target is not None:
            target_pos = np.asarray(target, dtype=float)
        else:
            target_pos = r[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
        action = (
            r[[f"action{i}" for i in range(7)]].to_numpy(dtype=float)
            if has_action
            else prev_action
        )
        grasp_contact = bool(r[contact_col]) if has_contact else False

        out = replayer.step(
            arm_q, finger_m, box_pos, box_quat, target_pos, action,
            grasp_contact=grasp_contact,
        )
        out["step"] = r.get("step", _)
        out["time"] = r.get("time", np.nan)
        rows.append(out)
        prev_action = action

    return pd.DataFrame(rows)


# ===========================================================================
# Task 1: multi-line reward-terms chart.
# ===========================================================================


def save_reward_terms_plot(
    reward_df: pd.DataFrame,
    out_path: str,
    title: str = "Reward terms",
    train_ref: dict = None,
) -> str:
    """One figure: every scaled reward term as its own line + a bold total,
    over the run's steps. `train_ref` (optional) is a {term: episode_sum}
    reference dict (e.g. from a training W&B run) drawn as faint dashed
    horizontal lines so a real/sim run reads directly against training.
    Saves both .png and .pdf next to `out_path` (dpi=300). Returns the png
    path.
    """
    base, _ext = os.path.splitext(out_path)
    png_path, pdf_path = base + ".png", base + ".pdf"

    fig, ax = plt.subplots(figsize=(12, 7))
    x = reward_df["step"] if "step" in reward_df.columns else reward_df.index
    colors = plt.cm.tab10.colors

    for i, term in enumerate(TERMS):
        if term in reward_df.columns:
            ax.plot(
                x, reward_df[term], label=term,
                color=colors[i % len(colors)], linewidth=1.5, alpha=0.85,
            )
    if "reward_total" in reward_df.columns:
        ax.plot(x, reward_df["reward_total"], label="total",
                 color="black", linewidth=2.5)

    if train_ref:
        for term, val in train_ref.items():
            color = "black" if term == "reward_total" else None
            ax.axhline(val, linestyle="--", alpha=0.3, color=color,
                        label=f"train {term}" if term == "reward_total" else None)

    ax.set_xlabel("step")
    ax.set_ylabel("scaled reward")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ===========================================================================
# Task 2 helper (MJX -- do NOT run locally; HPC / robot PC only).
# ===========================================================================


def sim_rollout_reward(policy_dir: str, seed: int, xml_path: str = None) -> pd.DataFrame:
    """Roll out a downloaded policy in the MuJoCo Playground sim env (MJX)
    and collect the per-step SCALED reward terms from `state.metrics`, which
    the env already produces raw/unscaled every step (see ur3_pick.py
    `step()`'s `metrics.update(**raw_rewards, ...)`).

    MJX -- this function builds and steps an MJX env. Per CLAUDE.md, never
    run this locally (no GPU; CPU MJX OOM-kills the machine). Run on the
    ZHAW HPC or the robot PC only.

    Mirrors evaluation/render_ur3_policy_rollout.py's load+rollout pattern.
    Also returns geometry columns (arm qpos, finger qpos, box qpos, action)
    so a replayer-vs-sim parity check can feed the SAME geometry through
    `RewardReplayer` and assert the two agree (contacts taken from sim's own
    `grasped`, so the proxy is exact in that check, unlike the real-robot
    Hand-E approximation).
    """
    import json

    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from flax import serialization

    from mujoco_playground import registry

    with open(os.path.join(policy_dir, "metadata.json")) as f:
        meta = json.load(f)

    nf_kwargs = {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in meta["network_factory"].items()
    }
    ppo_net = ppo_networks.make_ppo_networks(
        observation_size=meta["obs_dim"],
        action_size=meta["action_dim"],
        preprocess_observations_fn=running_statistics.normalize,
        **nf_kwargs,
    )
    template = {
        "0": running_statistics.init_state(
            jax.ShapeDtypeStruct((meta["obs_dim"],), jnp.float32)
        ),
        "1": ppo_net.policy_network.init(jax.random.PRNGKey(0)),
        "2": ppo_net.value_network.init(jax.random.PRNGKey(0)),
    }
    with open(os.path.join(policy_dir, "params.msgpack"), "rb") as f:
        params = serialization.from_bytes(template, f.read())
    raw_policy = ppo_networks.make_inference_fn(ppo_net)(
        (params["0"], params["1"]), deterministic=True
    )
    inference_fn = jax.jit(raw_policy)

    env_overrides = meta.get("env_overrides", {})
    env = registry.load(meta["env_name"], config_overrides=env_overrides)
    episode_length = int(env._config.episode_length)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    rows = []
    for t in range(episode_length):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, action)

        raw = {k: float(state.metrics[k]) for k in TERMS}
        scaled = {k: raw[k] * SCALES[k] for k in TERMS}
        row = dict(scaled)
        row["reward_total"] = float(state.reward)
        row["step"] = t
        row["seed"] = int(seed)
        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# Smoke test -- pure MuJoCo FK, synthetic trajectory. NO MJX, NO robot. Safe
# to run locally: `python evaluation/ur3_reward_replay.py`.
# ===========================================================================


def _smoke_test():
    print(f"SCALES ({len(SCALES)} terms): {SCALES}")
    print(f"grasp_align_thresh={GRASP_ALIGN_THRESH} hold_radius={HOLD_RADIUS} "
          f"hold_tau={HOLD_TAU} lift_eps={LIFT_EPS}")

    fk = SimFK(_DEFAULT_XML)
    replayer = RewardReplayer(fk)

    # Synthetic 3-step trajectory: a box resting at (0.30, 0.0, 0.02), a TCP
    # descending toward it over 3 steps, fingers closing, no rotation (all
    # box axes world-aligned -> alignment should be near 1.0 once the jaw is
    # roughly aligned with a world axis). This is NOT a physically-integrated
    # trajectory -- it only exercises the reward math end-to-end.
    box_pos = np.array([0.30, 0.0, 0.02])
    box_quat = np.array([1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([0.32, 0.0, 0.20])
    # A task_home-ish arm pose (from ur3_pick default init note), then two
    # small nudges "toward" the box -- just needs to be a valid, finite qpos.
    q_home = np.array([0.0, -2.0, 1.6, -1.6, -1.5, 0.0])
    trajectory = [
        (q_home, 0.025, False),  # open, far
        (q_home + 0.01, 0.02, False),  # slightly closer, fingers closing
        (q_home + 0.02, 0.0, True),  # fingers closed, contact reported
    ]
    replayer.reset(box_rest_z=float(box_pos[2]), init_arm_q=q_home)

    header = ["step"] + TERMS + ["reward_total", "reached", "grasped", "lifted",
                                  "alignment", "gripper_box_dist", "box_target_dist"]
    print("\n" + " | ".join(f"{h:>16s}" for h in header))
    for i, (arm_q, finger_m, contact) in enumerate(trajectory):
        action = np.zeros(7)
        out = replayer.step(
            arm_q, finger_m, box_pos, box_quat, target_pos, action,
            grasp_contact=contact,
        )
        row = [i] + [out[t] for t in TERMS] + [
            out["reward_total"], out["reached"], out["grasped"], out["lifted"],
            out["alignment"], out["gripper_box_dist"], out["box_target_dist"],
        ]
        print(" | ".join(f"{v:16.4f}" if isinstance(v, float) else f"{v:16d}"
                          for v in row))
        for k, v in out.items():
            assert np.isfinite(v), f"step {i}: non-finite {k}={v}"
        assert -1e4 <= out["reward_total"] <= 1e4, out["reward_total"]
        assert 0.0 <= out["alignment"] <= 1.0, out["alignment"]
        assert 0.0 <= out["reached"] <= 1.0
        assert 0.0 <= out["grasped"] <= 1.0
        assert 0.0 <= out["lifted"] <= 1.0

    # Sanity: a synthetic plotting smoke test too (pure matplotlib, no MJX).
    reward_df = pd.DataFrame(
        [
            {**{t: 0.0 for t in TERMS}, "reward_total": 0.0, "step": i}
            for i in range(3)
        ]
    )
    out_dir = os.path.join(REPO_ROOT, "evaluation", "__pycache__")
    os.makedirs(out_dir, exist_ok=True)
    png = save_reward_terms_plot(
        reward_df, os.path.join(out_dir, "_smoke_reward_plot.png"),
        title="smoke test",
    )
    assert os.path.exists(png), png
    os.remove(png)
    os.remove(png.replace(".png", ".pdf"))

    print("\nSMOKE TEST OK (pure MuJoCo FK, no MJX, no robot).")


if __name__ == "__main__":
    _smoke_test()
