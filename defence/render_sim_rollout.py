"""Render the SIM half of a defence side-by-side video from a real-robot run.

Companion to `defence/record_real_rollout.py` (never edited by this file --
see CLAUDE.md's "surgical changes" rule). That script drives one real UR3e
pick episode and writes `defence/runs/<run>/manifest.json` +
`real_states.csv` + `real.mp4`. This script reads ONLY the manifest, resets a
plain-MuJoCo (CPU) copy of the same scene to the MEASURED initial state, rolls
the checkpoint's policy closed-loop for `control.episode_length` steps, and
writes `sim.mp4` + `sim_states.csv` into the same run directory so the two
clips/logs can be compared frame-for-frame.

HARD RULES (see the task this file was written against / CLAUDE.md):
  * plain `mujoco` (CPU) ONLY -- `mujoco.mjx` is NEVER *used*. It does land
    in `sys.modules` anyway (brax's transitive import), so presence is not
    the thing being checked. `_assert_no_mjx()` below instead re-verifies
    that `_forbid_mjx_use()`'s raiser patches are still intact (and neuters
    mjx if it only just appeared unpatched), so a call into a live MJX entry
    point cannot pass unnoticed -- CPU MJX OOM-kills this machine.
  * `mujoco_playground._src.manipulation.my_ur3.ur3_pick` is NEVER imported
    (it pulls in mjx at module import time, and its reset_to_state/step/
    _get_obs are jax/mjx-typed -- they cannot be called from a CPU loop).
    Its reset/obs/control-law LOGIC is instead reimplemented here in plain
    numpy against `mujoco.MjModel`/`MjData`, read from the source (with line
    citations in the comments below) but never executed via that module.
  * Never run the sim/training locally beyond what this script itself does
    (one policy-driven CPU episode, which is cheap): this file is verified
    with `python -m py_compile` and static reading only.

Usage:
    python defence/render_sim_rollout.py --run-dir defence/runs/<run>
    python defence/render_sim_rollout.py --run-dir defence/runs/<run> \\
        --camera side_130
"""

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# CLAUDE.md: macOS has no osmesa, so mujoco.Renderer needs MUJOCO_GL=glfw or
# it has no GL backend at all. setdefault so an explicit env var still wins.
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))


import mujoco  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import serialization  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def _forbid_mjx_use():
  """Make MJX *usable* impossible, without banning its mere presence.

  `brax` imports `mujoco.mjx` transitively, so `mujoco.mjx in sys.modules` is
  true in any environment that can load a Brax PPO policy at all — checking
  for presence would make this script unrunnable. What actually OOM-kills a
  MacBook is USING MJX: putting the model on device and stepping it.

  So the entry points that would start a CPU-MJX rollout are replaced with
  raisers. Importing mjx stays fine; calling it dies loudly and immediately.
  This file never calls them — the guard exists to catch a future edit, or a
  transitive call from library code, before it eats all the machine's RAM.
  """
  mjx = sys.modules.get("mujoco.mjx")
  if mjx is None:
    return []

  def _raiser(name):
    def _blocked(*_args, **_kwargs):
      raise SystemExit(
          f"HARD RULE VIOLATION: mujoco.mjx.{name}() was called. This script "
          f"is plain-CPU-MuJoCo only -- CPU MJX OOM-kills this machine. Step "
          f"with mujoco.mj_step(MjModel, MjData) instead."
      )
    return _blocked

  blocked = []
  for name in ("step", "make_data", "put_model", "put_data", "forward"):
    if hasattr(mjx, name):
      setattr(mjx, name, _raiser(name))
      blocked.append(name)
  return blocked


_BLOCKED_MJX = _forbid_mjx_use()


def _assert_no_mjx(when: str) -> None:
  """Re-verify that `_forbid_mjx_use()`'s guard is still intact.

  This does NOT assert that `mujoco.mjx` is absent from `sys.modules` --
  brax's own transitive import puts it there in any environment that can
  load a policy at all (see `_forbid_mjx_use`'s docstring), so a presence
  check would abort every run. It only checks that the raiser patches are
  still in place, covering two cases:
    (i)  `_BLOCKED_MJX` is empty but `mujoco.mjx` is now in `sys.modules`
         -- it arrived (or was reloaded) after `_forbid_mjx_use()` already
         ran with nothing to patch. Neuter it now.
    (ii) `_BLOCKED_MJX` is non-empty but one of those names no longer
         resolves to the `_raiser`-built stub (importlib.reload, a
         monkey-patch, a library re-import) -- hard-fail.
  Even fully intact, this is a best-effort check: only five top-level
  `mujoco.mjx` attributes are patched, so e.g.
  `from mujoco.mjx._src.forward import step` bypasses it entirely.
  """
  global _BLOCKED_MJX
  mjx = sys.modules.get("mujoco.mjx")
  if mjx is None:
    return
  if not _BLOCKED_MJX:
    _BLOCKED_MJX = _forbid_mjx_use()
    print(
        f"[_assert_no_mjx] mujoco.mjx appeared in sys.modules {when} -- "
        f"neutered it now (blocked: {_BLOCKED_MJX})."
    )
    return
  for name in _BLOCKED_MJX:
    fn = getattr(mjx, name, None)
    if getattr(fn, "__name__", None) != "_blocked":
      raise SystemExit(
          f"HARD RULE VIOLATION: mujoco.mjx.{name} no longer resolves to "
          f"the _forbid_mjx_use() raiser stub {when} -- the MJX guard was "
          "removed or replaced (reload/monkeypatch/re-import). Refusing to "
          "continue."
      )


def _assert_cpu_physics(model, data):
  """Physics objects must be the plain CPU types, never MJX device structs."""
  if not isinstance(model, mujoco.MjModel):
    raise SystemExit(
        f"HARD RULE VIOLATION: model is {type(model)!r}, not mujoco.MjModel "
        f"-- this rollout must stay on plain CPU MuJoCo.")
  if not isinstance(data, mujoco.MjData):
    raise SystemExit(
        f"HARD RULE VIOLATION: data is {type(data)!r}, not mujoco.MjData "
        f"-- this rollout must stay on plain CPU MuJoCo.")

# ===========================================================================
# Constants MIRRORED from ur3_pick.py / ur3_base.py -- NOT imported (see the
# module docstring). Each has a citation to where it was read from; if the
# source values ever change, these will silently drift and need re-syncing.
# ===========================================================================

# ur3_base._ARM_JOINTS / _FINGER_JOINTS.
_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_FINGER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]

# ur3_pick.default_config(): action_scale=0.015, gripper_action_scale=0.02,
# obs_include_velocity default (absent -> False, addvelocity STATIC gate).
# Used only as the FALLBACK when a checkpoint's metadata.json env_overrides
# doesn't set these keys -- every checkpoint seen so far sets them
# explicitly, so this is a defensive default, not the expected path.
_DEFAULT_ACTION_SCALE = 0.015
_DEFAULT_GRIPPER_ACTION_SCALE = 0.02
_DEFAULT_OBS_INCLUDE_VELOCITY = False

# ur3_pick.py: _LIFTER_HALF_THICKNESS = 0.0025, _LIFTER_HEIGHT_MIN = 0.003.
# Only used for the OPTIONAL lifter_top_height override path -- the nominal
# (override-absent) lifter pose is read straight off the loaded MjModel's
# own body pos/quat instead of being re-derived from these constants.
_LIFTER_HALF_THICKNESS = 0.0025
_LIFTER_HEIGHT_MIN = 0.003

# evaluation/gap_metrics.py's drop-square target zone + tape-mark poses,
# mirrored (not imported) for the table-marker overlay drawn in
# draw_table_markers(). A pytest asserts the two TAPE_MARK_XY copies (this
# one and gap_metrics.TAPE_MARK_XY) stay equal.
DROP_SQUARE_CENTER = (0.212, 0.212)        # mirror of gap_metrics.py:963
DROP_SQUARE_HALF_WIDTH = 0.075             # mirror of gap_metrics.py:964
TAPE_MARK_XY = {"P1": (0.216, -0.185),
                "P2": (0.355, -0.009),
                "P3": (0.495, 0.188)}      # mirror of gap_metrics.TAPE_MARK_XY
_MARK_RADIUS = 0.015
_TAPE_HALF_WIDTH = 0.010
_OVERLAY_HALF_THICKNESS = 0.0005
_OVERLAY_Z_GAP = 0.0005
_SQUARE_RGB = (1.00, 0.80, 0.10)
_MARK_RGB = {"P1": (0.15, 0.85, 1.00),     # cyan
             "P2": (1.00, 0.35, 0.85),     # magenta
             "P3": (0.35, 1.00, 0.45)}     # green

# record_real_rollout.py's manifest contract (never edited -- read only).
SCHEMA_VERSION = "defence/1"
GRIPPER_CONVENTION = "0=open,0.025=closed"

# record_real_rollout.py's CORE_COLUMNS, duplicated here (not imported --
# that module drives real hardware) so sim_states.csv's leading columns are
# byte-identical in name and order to real_states.csv's.
CORE_COLUMNS = (
    ["step", "t_rel"]
    + [f"q{i}" for i in range(6)]
    + ["finger", "tcp_x", "tcp_y", "tcp_z"]
    + ["box_x", "box_y", "box_z"]
    + ["box_qw", "box_qx", "box_qy", "box_qz"]
    + ["target_x", "target_y", "target_z", "box_to_target_dist"]
    + [f"action{i}" for i in range(7)]
)

DEFAULT_XML_PATH = os.path.join(
    REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "xmls",
    "mjx_single_cube_position_ur3.xml",
)
POLICY_ROOT = os.path.join(REPO_ROOT, "evaluation", "downloaded_policies")
DEFAULT_CAMERA_JSON = os.path.join(_THIS_DIR, "camera.json")


# ===========================================================================
# Manifest loading / validation.
# ===========================================================================


def _check(cond, msg):
  if not cond:
    raise SystemExit(f"manifest validation failed: {msg}")


def _get(d, path):
  node = d
  for key in path.split("."):
    if not isinstance(node, dict) or key not in node:
      raise SystemExit(
          f"manifest validation failed: missing required field {path!r}"
      )
    node = node[key]
  return node


def _check_len(path, value, n):
  _check(
      isinstance(value, (list, tuple)) and len(value) == n,
      f"{path!r} must be a length-{n} array, got {value!r}",
  )


def load_manifest(run_dir):
  manifest_path = os.path.join(run_dir, "manifest.json")
  if not os.path.exists(manifest_path):
    raise SystemExit(f"no manifest.json in {run_dir!r}")
  with open(manifest_path, "r", encoding="utf-8") as f:
    m = json.load(f)
  validate_manifest(m)
  return m


def validate_manifest(m):
  """Fail loudly on any missing/mismatched field (see the task contract).

  This only checks PRESENCE + basic shape for most fields; the four
  explicitly-called-out CONTRACT ASSERTIONS (obs_dim, action_dim==7,
  action_scale/gripper_action_scale re-derivation, gripper_convention) are
  checked later in `main()`, once the checkpoint metadata is available.
  """
  _check(
      _get(m, "schema_version") == SCHEMA_VERSION,
      f"schema_version {m.get('schema_version')!r} != {SCHEMA_VERSION!r} "
      "(this script was written against record_real_rollout.py's current "
      "contract -- a version bump on one side needs a matching update).",
  )
  _get(m, "run_name")
  _get(m, "t0_unix")
  _get(m, "t0_iso")
  _get(m, "t0_monotonic")

  arm_qpos = _get(m, "init.arm_qpos")
  _check_len("init.arm_qpos", arm_qpos, 6)
  _check_len("init.commanded_arm_qpos", _get(m, "init.commanded_arm_qpos"), 6)
  _get(m, "init.finger")
  _check_len("init.box_pos", _get(m, "init.box_pos"), 3)
  _check_len("init.box_quat_wxyz", _get(m, "init.box_quat_wxyz"), 4)
  _check_len("init.target_pos", _get(m, "init.target_pos"), 3)
  cube_half = _get(m, "init.cube_half_extents")
  if cube_half is not None:
    _check_len("init.cube_half_extents", cube_half, 3)
  lifter_h = _get(m, "init.lifter_top_height")
  if lifter_h is not None:
    _check(isinstance(lifter_h, (int, float)), "init.lifter_top_height")
  lifter_tilt = _get(m, "init.lifter_tilt_rp")
  if lifter_tilt is not None:
    _check_len("init.lifter_tilt_rp", lifter_tilt, 2)
  _get(m, "init.settle_s")

  for k in ("source", "protocol_id", "poses_sha256", "episode_id", "level",
            "split"):
    _get(m, f"pose_ref.{k}")

  _get(m, "policy.checkpoint_id")
  _get(m, "policy.checkpoint_sha256")
  _get(m, "policy.obs_dim")
  action_dim = _get(m, "policy.action_dim")
  _check(int(action_dim) == 7, f"policy.action_dim must be 7, got {action_dim}")
  _get(m, "policy.policy_hidden_layer_sizes")

  ctrl_dt = _get(m, "control.ctrl_dt")
  _check(float(ctrl_dt) > 0, "control.ctrl_dt must be > 0")
  _get(m, "control.action_scale")
  _get(m, "control.gripper_action_scale")
  _get(m, "control.scale_source")
  conv = _get(m, "control.gripper_convention")
  _check(
      conv == GRIPPER_CONVENTION,
      f"control.gripper_convention {conv!r} != {GRIPPER_CONVENTION!r}",
  )
  ep_len = _get(m, "control.episode_length")
  _check(int(ep_len) > 0, "control.episode_length must be > 0")
  _get(m, "control.control_hz_requested")
  _get(m, "control.achieved_hz")

  for k in ("stop_reason", "stop_step", "n_steps", "force_limit_n",
            "peak_force_n", "overrun_count"):
    _get(m, f"result.{k}")

  for k in ("env_name", "xml_path", "git_sha", "branch"):
    _get(m, f"env.{k}")

  for k in ("real_mp4", "webcam_fps_nominal", "n_webcam_frames",
            "frame_index_csv"):
    _get(m, f"video.{k}")


# ===========================================================================
# Checkpoint resolution.
# ===========================================================================


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def resolve_checkpoint(policy):
  ckpt_dir = os.path.join(POLICY_ROOT, policy["checkpoint_id"])
  meta_path = os.path.join(ckpt_dir, "metadata.json")
  params_path = os.path.join(ckpt_dir, "params.msgpack")
  for p in (meta_path, params_path):
    if not os.path.exists(p):
      raise SystemExit(
          f"checkpoint incomplete: {p!r} not found. Expected "
          f"evaluation/downloaded_policies/{policy['checkpoint_id']}/ to "
          "hold both metadata.json and params.msgpack."
      )
  actual_sha = sha256_file(params_path)
  want_sha = str(policy["checkpoint_sha256"])
  if actual_sha.lower() != want_sha.lower():
    raise SystemExit(
        f"checkpoint_sha256 mismatch for {policy['checkpoint_id']!r}: "
        f"manifest says {want_sha}, params.msgpack on disk hashes to "
        f"{actual_sha}. Refusing to render against a different checkpoint "
        "than the one the real run used."
    )
  with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
  return ckpt_dir, meta


def load_policy(ckpt_dir, meta):
  """Rebuild the deterministic inference_fn from params.msgpack.

  Mirrors evaluation/render_ur3_policy_rollout.py's network reconstruction
  (params.msgpack, NOT best_params.msgpack -- that best/final distinction
  belongs to evaluation/run_gap_protocol_sim.py's HPC-only path and is not
  part of this manifest's contract).
  """
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
  params_path = os.path.join(ckpt_dir, "params.msgpack")
  with open(params_path, "rb") as f:
    params = serialization.from_bytes(template, f.read())
  raw_policy = ppo_networks.make_inference_fn(ppo_net)(
      (params["0"], params["1"]), deterministic=True
  )
  return jax.jit(raw_policy)


def derive_scales(meta, control):
  """Re-derive action_scale/gripper_action_scale/obs_include_velocity from
  the checkpoint's OWN metadata.json (CLAUDE.md: deploy scale must equal
  trained scale), then hard-fail if they disagree with the manifest.
  """
  overrides = meta.get("env_overrides", {})
  action_scale = float(overrides.get("action_scale", _DEFAULT_ACTION_SCALE))
  gripper_action_scale = float(
      overrides.get("gripper_action_scale", _DEFAULT_GRIPPER_ACTION_SCALE)
  )
  obs_include_velocity = bool(
      overrides.get("obs_include_velocity", _DEFAULT_OBS_INCLUDE_VELOCITY)
  )

  manifest_as = float(control["action_scale"])
  manifest_gas = float(control["gripper_action_scale"])
  if action_scale != manifest_as:
    raise SystemExit(
        f"action_scale mismatch: checkpoint metadata.json (env_overrides) "
        f"says {action_scale}, manifest control.action_scale says "
        f"{manifest_as}. Deploy scale must equal trained scale -- refusing "
        "to render with a scale that doesn't match this checkpoint."
    )
  if gripper_action_scale != manifest_gas:
    raise SystemExit(
        f"gripper_action_scale mismatch: checkpoint metadata.json says "
        f"{gripper_action_scale}, manifest control.gripper_action_scale "
        f"says {manifest_gas}."
    )
  return action_scale, gripper_action_scale, obs_include_velocity


# ===========================================================================
# Plain-MuJoCo scene setup (mirrors ur3_base._post_init's name lookups --
# read from mujoco_playground/_src/manipulation/my_ur3/ur3_base.py, never
# imported).
# ===========================================================================


def load_model(xml_path, width, height):
  model = mujoco.MjModel.from_xml_path(xml_path)
  # The XML's <visual><global offwidth="960" offheight="720"/> is smaller
  # than our default 1920x1080 render request; mujoco.Renderer refuses to
  # exceed these, so bump them first (same pattern as e.g.
  # mujoco_playground/_src/locomotion/g1/base.py's offwidth/offheight bump).
  model.vis.global_.offwidth = max(int(width), model.vis.global_.offwidth)
  model.vis.global_.offheight = max(int(height), model.vis.global_.offheight)
  return model


def resolve_ids(model):
  """Name -> id/address lookups, mirroring ur3_base._post_init exactly."""
  ids = {}
  ids["arm_qposadr"] = np.array(
      [model.jnt_qposadr[model.joint(j).id] for j in _ARM_JOINTS]
  )
  ids["finger_qposadr"] = np.array(
      [model.jnt_qposadr[model.joint(j).id] for j in _FINGER_JOINTS]
  )
  ids["arm_dofadr"] = np.array(
      [model.jnt_dofadr[model.joint(j).id] for j in _ARM_JOINTS]
  )
  ids["finger_dofadr"] = np.array(
      [model.jnt_dofadr[model.joint(j).id] for j in _FINGER_JOINTS]
  )
  ids["tcp_site"] = model.site("tcp").id
  ids["left_touch_site"] = model.site("left_finger_touch_site").id
  ids["right_touch_site"] = model.site("right_finger_touch_site").id
  ids["obj_body"] = model.body("box").id
  ids["obj_geom"] = model.geom("box").id
  box_jntadr = model.body("box").jntadr[0]
  ids["obj_qposadr"] = int(model.jnt_qposadr[box_jntadr])
  ids["mocap_target"] = int(np.asarray(model.body("mocap_target").mocapid)[0])
  ids["lifter_body"] = model.body("lifter").id
  ids["lifter_mocap"] = int(np.asarray(model.body("lifter").mocapid)[0])
  ids["lowers"], ids["uppers"] = model.actuator_ctrlrange.T.copy()
  return ids


def _quat_mul(a, b):
  """Hamilton product of two [w, x, y, z] quaternions (MuJoCo convention).

  Numpy port of ur3_pick.py's `_quat_mul` (jax) -- read, not imported.
  """
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array([
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
  ])


def reset_state(model, ids, init):
  """Reset qpos/ctrl/mocap to the MEASURED state, then mj_forward.

  Mirrors ur3_pick.py's `UR3Pick.reset_to_state` (arm_qpos/finger/box_pos/
  box_quat/target_pos/cube_half_extents/lifter_top_height/lifter_tilt_rp),
  reimplemented in plain numpy against `mujoco.MjData` -- that method is
  jax/mjx-typed and is read here, never called.
  """
  arm_qpos = np.asarray(init["arm_qpos"], dtype=float)
  finger = float(init["finger"])
  box_pos = np.asarray(init["box_pos"], dtype=float)
  box_quat = np.asarray(init["box_quat_wxyz"], dtype=float)
  box_quat = box_quat / np.linalg.norm(box_quat)
  target_pos = np.asarray(init["target_pos"], dtype=float)

  cube_half_extents = init["cube_half_extents"]
  if cube_half_extents is not None:
    # ur3_pick.py's D17 override: applied UNCLAMPED, straight into
    # geom_size, bypassing the training-time graspability clamp.
    model.geom_size[ids["obj_geom"]] = np.asarray(
        cube_half_extents, dtype=float
    )

  qpos = np.zeros(model.nq, dtype=float)
  qpos[ids["arm_qposadr"]] = arm_qpos
  qpos[ids["finger_qposadr"]] = finger
  qpos[ids["obj_qposadr"]:ids["obj_qposadr"] + 3] = box_pos
  qpos[ids["obj_qposadr"] + 3:ids["obj_qposadr"] + 7] = box_quat

  ctrl = np.zeros(model.nu, dtype=float)
  ctrl[:6] = arm_qpos
  # finger_qpos.sum() * 0.5 == finger (both fingers set symmetrically).
  ctrl[6] = finger

  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = 0.0
  data.ctrl[:] = ctrl
  data.mocap_pos[ids["mocap_target"]] = target_pos

  # Lifter (table) mocap pose. Nominal (override absent) is read straight
  # off the loaded model's own static body pos/quat -- exactly the value
  # ur3_pick.py's reset_to_state pins it to when lifter_top_height /
  # lifter_tilt_rp are None, since that nominal is itself sourced from the
  # same XML (see ur3_pick.py's __init__ D24 anchor assertions).
  lifter_top_height = init["lifter_top_height"]
  lifter_tilt_rp = init["lifter_tilt_rp"]
  lifter_xy = model.body_pos[ids["lifter_body"]][:2]
  if lifter_top_height is None:
    lifter_z = model.body_pos[ids["lifter_body"]][2]
  else:
    lifter_z = max(
        float(lifter_top_height) - _LIFTER_HALF_THICKNESS,
        _LIFTER_HEIGHT_MIN,
    )
  if lifter_tilt_rp is None:
    lifter_quat = np.array([1.0, 0.0, 0.0, 0.0])
  else:
    roll, pitch = float(lifter_tilt_rp[0]), float(lifter_tilt_rp[1])
    q_roll = np.array([np.cos(roll / 2.0), np.sin(roll / 2.0), 0.0, 0.0])
    q_pitch = np.array([np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0])
    lifter_quat = _quat_mul(q_pitch, q_roll)
  data.mocap_pos[ids["lifter_mocap"]] = np.array(
      [lifter_xy[0], lifter_xy[1], lifter_z]
  )
  data.mocap_quat[ids["lifter_mocap"]] = lifter_quat

  # Last gate before any stepping: these must be the plain CPU types.
  _assert_cpu_physics(model, data)
  mujoco.mj_forward(model, data)
  return data, target_pos


def build_obs(data, ids, target_pos, last_action, obs_include_velocity):
  """26D/33D obs, mirroring ur3_pick.py's `UR3Pick._get_obs` (the non-noisy
  branch -- domain_rand.obs_noise is NOT reproduced here, see the report
  note) composed on top of ur3_base.py's `UR3Base._get_obs` (the 13D base).
  """
  arm_q = data.qpos[ids["arm_qposadr"]]
  gripper = data.qpos[ids["finger_qposadr"]].sum()
  tcp = data.site_xpos[ids["tcp_site"]]
  box_pos = data.xpos[ids["obj_body"]]
  base13 = np.concatenate([
      arm_q,
      [gripper],
      box_pos - tcp,
      target_pos - box_pos,
  ])

  l_pos = data.site_xpos[ids["left_touch_site"]]
  r_pos = data.site_xpos[ids["right_touch_site"]]
  g_pos = tcp
  jaw_axis = r_pos - l_pos
  jaw_axis = jaw_axis / (np.linalg.norm(jaw_axis) + 1e-6)
  app_axis = 0.5 * (l_pos + r_pos) - g_pos
  app_axis = app_axis / (np.linalg.norm(app_axis) + 1e-6)
  box_axes = data.xmat[ids["obj_body"]].reshape(3, 3)
  jaw_proj = jaw_axis @ box_axes
  app_proj = app_axis @ box_axes

  obs = np.concatenate([base13, jaw_proj, app_proj, last_action])
  if obs_include_velocity:
    arm_qvel = data.qvel[ids["arm_dofadr"]]
    finger_qvel = data.qvel[ids["finger_dofadr"]].sum()
    obs = np.concatenate([obs, arm_qvel, [finger_qvel]])
  return obs.astype(np.float32)


def apply_control(model, data, action, action_scale_vec, n_substeps, ids):
  """ctrl = clip(prev_ctrl + action * action_scale, lowers, uppers), then
  n_substeps physics steps. Mirrors ur3_pick.py's `UR3Pick.step` control law
  exactly (`delta = exec_action * self._action_scale; ctrl = jp.clip(
  state.data.ctrl + delta, self._lowers, self._uppers)`) -- domain-
  randomization action-delay is not reproduced (this manifest's reset always
  bypasses it; see the report note).
  """
  ctrl = np.clip(
      data.ctrl + action * action_scale_vec, ids["lowers"], ids["uppers"]
  )
  data.ctrl[:] = ctrl
  for _ in range(n_substeps):
    mujoco.mj_step(model, data)


# ===========================================================================
# Camera.
# ===========================================================================


def load_or_default_camera(camera_json_path):
  if os.path.exists(camera_json_path):
    with open(camera_json_path, "r", encoding="utf-8") as f:
      cam = json.load(f)
  else:
    # Mirrors the XML "front" camera's pose (mjx_scene_position_ur3.xml) as
    # a reasonable starting webcam-like view; edit camera.json to match the
    # real webcam framing and re-render (cheap: one CPU episode).
    cam = {"pos": [0.85, 0.0, 0.4], "lookat": [0.3, 0.0, 0.1], "fovy": 42.0}
    os.makedirs(os.path.dirname(camera_json_path), exist_ok=True)
    with open(camera_json_path, "w", encoding="utf-8") as f:
      json.dump(cam, f, indent=2)
    print(
        f"[camera] {camera_json_path!r} not found -- wrote a default "
        f"free-camera view (pos={cam['pos']}, lookat={cam['lookat']}, "
        f"fovy={cam['fovy']}). Edit it to match the real webcam framing and "
        "re-render."
    )
  for key, n in (("pos", 3), ("lookat", 3)):
    if not (isinstance(cam.get(key), (list, tuple)) and len(cam[key]) == n):
      raise SystemExit(f"camera.json field {key!r} must be a length-{n} array")
  if "fovy" not in cam:
    raise SystemExit("camera.json missing required field 'fovy'")
  return cam


def build_free_camera(cam_dict, model):
  """pos + lookat + fovy -> a MuJoCo free camera (azimuth/elevation/
  distance/lookat). fovy is a model-level (not per-camera) setting for the
  free camera, so it goes on `model.vis.global_.fovy`.
  """
  pos = np.asarray(cam_dict["pos"], dtype=float)
  lookat = np.asarray(cam_dict["lookat"], dtype=float)
  fovy = float(cam_dict["fovy"])

  fwd = lookat - pos
  distance = float(np.linalg.norm(fwd))
  if distance < 1e-6:
    raise SystemExit("camera.json: 'pos' and 'lookat' must not coincide")
  fwd_unit = fwd / distance
  elevation = float(np.degrees(np.arcsin(np.clip(fwd_unit[2], -1.0, 1.0))))
  azimuth = float(np.degrees(np.arctan2(fwd_unit[1], fwd_unit[0])))

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = lookat
  cam.distance = distance
  cam.azimuth = azimuth
  cam.elevation = elevation
  model.vis.global_.fovy = fovy
  return cam


# ===========================================================================
# Table markers (drop-square outline + P1/P2/P3 tape marks) overlay.
# ===========================================================================


def _append_decor_geom(scene, gtype, size, pos, mat, rgb):
  """Append one non-interactive "decor" geom to `scene`. Returns True on
  success, False (appending nothing) if the scene is already full.

  There is no bounds check in the `ngeom` setter -- writing past capacity
  makes `scene.geoms[i]` raise IndexError instead of failing cleanly, so
  that check is done by hand here.
  """
  if scene.ngeom >= scene.maxgeom:
    return False
  g = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
      g, gtype,
      np.asarray(size, dtype=float),
      np.asarray(pos, dtype=float),
      np.asarray(mat, dtype=float).reshape(9),
      np.array([*rgb, 1.0], dtype=float),
  )
  # Flat tape lying on the table must not cast a shadow onto the very table
  # it lies on.
  g.category = mujoco.mjtCatBit.mjCAT_DECOR
  g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
  g.objid = -1
  # Default segid==0 ALIASES real geom 0 in Renderer's segmentation remap
  # (mujoco/renderer.py:222); -1 opts these overlay geoms out entirely.
  g.segid = -1
  g.emission = 0.35
  # g.label is deliberately left empty: labels DO render, but as a
  # depth-ignoring 2D overlay, so e.g. a "P3" label would float on top of
  # the arm regardless of real occlusion.
  scene.ngeom = scene.ngeom + 1
  return True


def _drop_square_segments(cx, cy, h, w, t):
  """4 box segments -- ((centre_x, centre_y), (half_x, half_y, half_z)) --
  outlining a picture-frame-style square: half-width `h`, centred at
  (cx, cy), drawn from tape strips of half-width `w` and half-thickness `t`.

  Pure function, no MuJoCo objects, so the corner math is unit-testable
  without a live mjvScene.

  The verticals are SHORTENED by `w` (half-length h-w instead of h) so their
  end faces butt exactly against the horizontals' inner faces (vertical
  ends at cy +- (h-w); horizontal inner face at cy +- h -+ w -- the same
  number). Extending both pairs to the full half-length `h` would leave two
  overlapping coplanar boxes at each of the four corners, z-fighting in the
  render.
  """
  return [
      ((cx, cy - h), (h + w, w, t)),
      ((cx, cy + h), (h + w, w, t)),
      ((cx - h, cy), (w, h - w, t)),
      ((cx + h, cy), (w, h - w, t)),
  ]


def draw_table_markers(scene, model, data, ids):
  """Draw the drop-square outline + P1/P2/P3 tape marks as a scene overlay.

  The markers are tape stuck to the REAL table, so they must ride with the
  table body, not be pinned in the base frame: if init.lifter_top_height
  raises the top to 0.150 m (D1) or init.lifter_tilt_rp tilts it (C1), the
  tape moves with it -- pinning at base-frame z=0.095 would bury the tape
  inside a raised slab.

  Uses data.xpos/data.xmat, NOT mocap_pos/mocap_quat: xpos/xmat are already
  populated by the mj_forward call in reset_state() and hand us the body's
  rotation matrix for free.
  """
  bid = ids["lifter_body"]
  offset_x, offset_y = model.body_pos[bid][:2]
  # 0.0025 (lifter geom half-thickness -> its local top face) + a 0.5 mm
  # gap + the overlay slab's own half-thickness, so its BOTTOM face clears
  # the table top.
  local_z = _LIFTER_HALF_THICKNESS + _OVERLAY_Z_GAP + _OVERLAY_HALF_THICKNESS

  R = data.xmat[bid].reshape(3, 3)
  mat = data.xmat[bid]  # already flat-9 float64 C-contiguous

  def _world(local_xy):
    local_pos = np.array([local_xy[0], local_xy[1], local_z])
    return data.xpos[bid] + R @ local_pos

  segments = _drop_square_segments(
      DROP_SQUARE_CENTER[0] - offset_x,
      DROP_SQUARE_CENTER[1] - offset_y,
      DROP_SQUARE_HALF_WIDTH,
      _TAPE_HALF_WIDTH,
      _OVERLAY_HALF_THICKNESS,
  )
  for local_xy, half_size in segments:
    if not _append_decor_geom(
        scene, mujoco.mjtGeom.mjGEOM_BOX, half_size, _world(local_xy), mat,
        _SQUARE_RGB,
    ):
      print("[draw_table_markers] scene is full -- stopped appending decor "
            "geoms.")
      return

  # mjGEOM_LINE's width is specified in PIXELS, not world units, so it would
  # not scale with camera distance/perspective -- cylinders are used
  # instead so the marks read at a consistent physical size.
  for name, (mx, my) in TAPE_MARK_XY.items():
    local_xy = (mx - offset_x, my - offset_y)
    size = (_MARK_RADIUS, _OVERLAY_HALF_THICKNESS, 0.0)
    if not _append_decor_geom(
        scene, mujoco.mjtGeom.mjGEOM_CYLINDER, size, _world(local_xy), mat,
        _MARK_RGB[name],
    ):
      print("[draw_table_markers] scene is full -- stopped appending decor "
            "geoms.")
      return


# ===========================================================================
# Main.
# ===========================================================================


def main():
  ap = argparse.ArgumentParser(
      description="Render the sim half of a defence side-by-side video from "
                  "a real-robot run's manifest.json."
  )
  ap.add_argument("--run-dir", required=True,
                   help="defence/runs/<run>/ dir containing manifest.json")
  ap.add_argument("--camera", default=None,
                   help="name of an existing XML camera (e.g. 'side_130') "
                        "-- overrides --camera-json.")
  ap.add_argument("--camera-json", default=DEFAULT_CAMERA_JSON)
  ap.add_argument("--width", type=int, default=1920)
  ap.add_argument("--height", type=int, default=1080)
  ap.add_argument("--out", default=None,
                   help="default: <run-dir>/sim.mp4")
  ap.add_argument("--no-markers", dest="markers", action="store_false",
                   help="disable the drop-square/tape-mark overlay drawn on "
                        "the table (see draw_table_markers).")
  ap.set_defaults(markers=True)
  args = ap.parse_args()

  run_dir = os.path.abspath(args.run_dir)
  out_path = args.out or os.path.join(run_dir, "sim.mp4")

  m = load_manifest(run_dir)
  init = m["init"]
  policy = m["policy"]
  control = m["control"]
  env = m["env"]

  ckpt_dir, meta = resolve_checkpoint(policy)
  if int(meta.get("obs_dim", -1)) != int(policy["obs_dim"]):
    raise SystemExit(
        f"obs_dim mismatch: checkpoint metadata.json says "
        f"{meta.get('obs_dim')}, manifest policy.obs_dim says "
        f"{policy['obs_dim']}."
    )
  if int(meta.get("action_dim", -1)) != int(policy["action_dim"]):
    raise SystemExit(
        f"action_dim mismatch: checkpoint metadata.json says "
        f"{meta.get('action_dim')}, manifest policy.action_dim says "
        f"{policy['action_dim']}."
    )
  action_scale, gripper_action_scale, obs_include_velocity = derive_scales(
      meta, control
  )
  inference_fn = load_policy(ckpt_dir, meta)
  _assert_no_mjx("after loading the policy")

  xml_rel = env["xml_path"]
  xml_path = (
      xml_rel if os.path.isabs(xml_rel) else os.path.join(REPO_ROOT, xml_rel)
  )
  if not os.path.exists(xml_path):
    raise SystemExit(f"env.xml_path does not exist: {xml_path!r}")

  model = load_model(xml_path, args.width, args.height)
  ids = resolve_ids(model)
  data, target_pos = reset_state(model, ids, init)

  if args.markers:
    # DROP_SQUARE_CENTER is the L0 deterministic fixed target (r=0.30,
    # azim=45deg); for L1-L4 the target is drawn randomly per episode, so
    # the cube may land off the taped square in the video. The tape is a
    # fixed PHYSICAL feature on the real table and is drawn at its own
    # location regardless -- this is just a heads-up, not an error.
    off_dist = float(np.linalg.norm(
        np.asarray(target_pos[:2]) - np.asarray(DROP_SQUARE_CENTER)))
    if off_dist > 0.02:
      print(
          f"[warn] manifest init.target_pos xy = "
          f"({target_pos[0]:.3f}, {target_pos[1]:.3f}) is {off_dist:.3f} m "
          f"from DROP_SQUARE_CENTER = {DROP_SQUARE_CENTER} -- the taped "
          "drop square is a fixed physical feature on the real table and "
          "will be drawn there regardless; the cube may visibly miss it in "
          "this render."
      )

  obs_dim_expected = int(policy["obs_dim"])
  last_action = np.zeros(7, dtype=np.float32)
  obs = build_obs(data, ids, target_pos, last_action, obs_include_velocity)
  if obs.shape[0] != obs_dim_expected:
    raise SystemExit(
        f"obs_dim mismatch: built a {obs.shape[0]}D obs "
        f"(obs_include_velocity={obs_include_velocity}) but "
        f"policy.obs_dim={obs_dim_expected}."
    )

  action_scale_vec = np.array([action_scale] * 6 + [gripper_action_scale])
  ctrl_dt = float(control["ctrl_dt"])
  sim_dt = float(model.opt.timestep)
  n_substeps = int(round(ctrl_dt / sim_dt))
  if abs(n_substeps * sim_dt - ctrl_dt) > 1e-6:
    print(
        f"[warn] ctrl_dt={ctrl_dt} is not an exact multiple of "
        f"model.opt.timestep={sim_dt} (n_substeps={n_substeps})."
    )
  fps = round(1.0 / ctrl_dt)
  episode_length = int(control["episode_length"])

  if args.camera is not None:
    camera_arg = args.camera
  else:
    cam_dict = load_or_default_camera(args.camera_json)
    camera_arg = build_free_camera(cam_dict, model)

  renderer = mujoco.Renderer(model, height=args.height, width=args.width)

  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  writer = imageio.get_writer(
      out_path,
      fps=fps,
      codec="libx264",
      pixelformat="yuv420p",
      macro_block_size=16,
      ffmpeg_params=["-profile:v", "high", "-level", "4.0", "-crf", "18"],
  )

  rng = jax.random.PRNGKey(0)  # deterministic policy: rng is unused.
  rows = []
  print(
      f"Rolling out {episode_length} steps "
      f"(action_scale={action_scale}, gripper_action_scale="
      f"{gripper_action_scale}, obs_include_velocity={obs_include_velocity})"
  )
  try:
    for step in range(episode_length):
      obs = build_obs(
          data, ids, target_pos, last_action, obs_include_velocity
      )
      action_jax, _ = inference_fn(jnp.asarray(obs), rng)
      action = np.asarray(action_jax, dtype=np.float64).reshape(7)

      apply_control(model, data, action, action_scale_vec, n_substeps, ids)

      renderer.update_scene(data, camera=camera_arg)
      # mjv_updateScene REBUILDS scene.ngeom from the model on every call,
      # so the decor overlay must be re-appended every single frame.
      if args.markers:
        draw_table_markers(renderer.scene, model, data, ids)
      writer.append_data(renderer.render())

      arm_q = data.qpos[ids["arm_qposadr"]]
      finger = float(data.qpos[ids["finger_qposadr"]].mean())
      tcp = data.site_xpos[ids["tcp_site"]]
      box_pos = data.xpos[ids["obj_body"]]
      box_quat = data.xquat[ids["obj_body"]]
      dist = float(np.linalg.norm(box_pos - target_pos))

      rows.append(
          [step, step * ctrl_dt]
          + list(arm_q)
          + [finger]
          + list(tcp)
          + list(box_pos)
          + list(box_quat)
          + list(target_pos)
          + [dist]
          + list(action)
      )
      last_action = action.astype(np.float32)
  finally:
    writer.close()

  _assert_no_mjx("after the rollout")

  csv_path = os.path.join(run_dir, "sim_states.csv")
  out_df = pd.DataFrame(rows, columns=CORE_COLUMNS)
  out_df["step"] = out_df["step"].astype(int)
  out_df.to_csv(csv_path, index=False)

  print(f"wrote {len(rows)} frames @ {fps} fps -> {out_path}")
  print(f"wrote {len(rows)} rows -> {csv_path}")


if __name__ == "__main__":
  main()
