"""Tests for defence/render_sim_rollout.py, defence/composite.py, and
(source-level only) defence/record_real_rollout.py.

HARD RULES (see the task contract this suite was written against):
  * NEVER import defence/record_real_rollout.py -- it pulls in a Linux-only
    vrpn.so and cv2, neither present on this machine. Section G below
    inspects it with py_compile.compile(..., doraise=True) and the `ast`
    module only.
  * NEVER render. No mujoco.Renderer, no offscreen GL, no rollout, no
    mj_step -- only mujoco.MjModel.from_xml_path + mujoco.mj_forward, via
    the `model`/`data` fixtures in conftest.py.

Implementations were being written concurrently while this file was
authored, against the frozen contract described in the task. Run with:
    python -m pytest defence/tests -x -q -p no:cacheprovider
(the `-p no:cacheprovider` avoids creating a tracked .pytest_cache/ dir --
.pytest_cache is not in this repo's .gitignore, unlike __pycache__).
"""

import argparse
import ast
import json
import os
import py_compile
import re
import sys
import types

import mujoco
import numpy as np
import pytest

import composite
import gap_metrics
import render_sim_rollout

from conftest import REPO_ROOT, _DEFENCE_DIR  # noqa: E402 -- shared paths

_RECORD_REAL_ROLLOUT_PATH = os.path.join(_DEFENCE_DIR, "record_real_rollout.py")

_MJX_NAMES = ("step", "make_data", "put_model", "put_data", "forward")


# ===========================================================================
# A. _assert_no_mjx
# ===========================================================================


def _blocked(*_args, **_kwargs):
  raise SystemExit("blocked")


def _make_blocked_stub():
  """A stub `mujoco.mjx` module whose 5 tracked names are ALL the same
  function literally named "_blocked" (mirrors _forbid_mjx_use's raiser)."""
  stub = types.ModuleType("mujoco.mjx")
  for n in _MJX_NAMES:
    setattr(stub, n, _blocked)
  return stub


def test_assert_no_mjx_noop_when_fully_blocked(monkeypatch):
  stub = _make_blocked_stub()
  monkeypatch.setitem(sys.modules, "mujoco.mjx", stub)
  monkeypatch.setattr(render_sim_rollout, "_BLOCKED_MJX", list(_MJX_NAMES))
  assert render_sim_rollout._assert_no_mjx("test") is None


def test_assert_no_mjx_raises_when_one_name_unblocked(monkeypatch):
  stub = _make_blocked_stub()

  def _normal_step(*_a, **_k):
    return None

  _normal_step.__name__ = "step"  # deliberately NOT "_blocked"
  setattr(stub, "step", _normal_step)

  monkeypatch.setitem(sys.modules, "mujoco.mjx", stub)
  monkeypatch.setattr(render_sim_rollout, "_BLOCKED_MJX", list(_MJX_NAMES))

  with pytest.raises(SystemExit):
    render_sim_rollout._assert_no_mjx("test")


def test_assert_no_mjx_neuters_stub_when_blocked_list_empty(monkeypatch):
  """`_BLOCKED_MJX == []` but `mujoco.mjx` present (e.g. it arrived after
  `_forbid_mjx_use()` already ran with nothing to patch): must NOT raise,
  and must neuter the stub in place -- afterwards all 5 names resolve to
  functions literally named "_blocked"."""
  stub = types.ModuleType("mujoco.mjx")
  for n in _MJX_NAMES:
    def _fn(*_a, __n=n, **_k):
      return __n

    _fn.__name__ = n  # normal name, NOT "_blocked"
    setattr(stub, n, _fn)

  monkeypatch.setitem(sys.modules, "mujoco.mjx", stub)
  monkeypatch.setattr(render_sim_rollout, "_BLOCKED_MJX", [])

  result = render_sim_rollout._assert_no_mjx("test")
  assert result is None
  for n in _MJX_NAMES:
    fn = getattr(stub, n)
    assert fn.__name__ == "_blocked", (
        f"{n} was not neutered: still named {fn.__name__!r}")


# ===========================================================================
# B. _drop_square_segments
# ===========================================================================

_TOL = 1e-12


def _footprint_contains(seg, px, py, eps=1e-9):
  (cx0, cy0), (sx, sy, _sz) = seg
  return (cx0 - sx - eps) <= px <= (cx0 + sx + eps) and (
      cy0 - sy - eps) <= py <= (cy0 + sy + eps)


@pytest.mark.parametrize("cx,cy", [
    (0.212, 0.212),    # the real drop-square center
    (0.355, -0.009),   # a tape-mark-like center
    (-0.15, -0.25),    # negative-quadrant
])
def test_drop_square_segments_geometry(cx, cy):
  h, w, t = 0.075, 0.010, 0.0005
  segs = render_sim_rollout._drop_square_segments(cx, cy, h, w, t)

  assert len(segs) == 4

  for (_px, _py), (sx, sy, sz) in segs:
    assert sx > 0
    assert sy > 0
    assert sz > 0
    assert sz == t  # z half-size == t, exactly (pure function, no float noise)

  # Classify by position: "horizontal"-type bars sit at (cx, cy +- h),
  # "vertical"-type bars sit at (cx +- h, cy). This is a black-box
  # classification (doesn't assume which pair ends up "long") that still
  # pins down exactly 2 of each and lets us find top/bottom/left/right by
  # name below.
  def _find(pred):
    matches = [s for s in segs if pred(s)]
    assert len(matches) == 1, f"expected exactly 1 match, got {len(matches)}"
    return matches[0]

  top = _find(lambda s: abs(s[0][1] - (cy + h)) < _TOL and abs(s[0][0] - cx) < _TOL)
  bot = _find(lambda s: abs(s[0][1] - (cy - h)) < _TOL and abs(s[0][0] - cx) < _TOL)
  right = _find(lambda s: abs(s[0][0] - (cx + h)) < _TOL and abs(s[0][1] - cy) < _TOL)
  left = _find(lambda s: abs(s[0][0] - (cx - h)) < _TOL and abs(s[0][1] - cy) < _TOL)

  (_, _), (_shx_top, shy_top, _) = top
  (_, _), (_shx_bot, shy_bot, _) = bot
  (_, _), (svx_r, svy_r, _) = right
  (_, _), (svx_l, svy_l, _) = left

  # BUTT JOINT: each vertical's end face (centre_y +- half_size_y) equals
  # the corresponding horizontal's INNER face (centre_y -+ half_size_y),
  # exactly (1e-12) -- zero overlap AND zero gap.
  top_inner_face = (cy + h) - shy_top
  bot_inner_face = (cy - h) + shy_bot
  assert abs(top_inner_face - (cy + svy_r)) < _TOL
  assert abs(top_inner_face - (cy + svy_l)) < _TOL
  assert abs(bot_inner_face - (cy - svy_r)) < _TOL
  assert abs(bot_inner_face - (cy - svy_l)) < _TOL

  # Side lengths: outer == 0.17, inner == 0.13, centreline == 0.15.
  # The horizontal bars' own (long) half-size doubled is the outer side
  # (they are the pair forced to span corner-to-corner -- see the "closed"
  # check below); the verticals' own (long) half-size doubled is the inner
  # side; the centreline is the classification offset `h` itself, doubled.
  assert abs(2 * _shx_top - 0.17) < _TOL
  assert abs(2 * _shx_bot - 0.17) < _TOL
  assert abs(2 * svy_r - 0.13) < _TOL
  assert abs(2 * svy_l - 0.13) < _TOL
  assert abs(2 * h - 0.15) < _TOL

  # CLOSED outline: no gap at any of the 4 corners of the centreline
  # square. (Note: the literal "covered by both a horizontal and a
  # vertical" phrasing in the task is geometrically impossible to satisfy
  # simultaneously with the "Zero overlap" butt-joint requirement above --
  # for a corner to be inside the positive-area footprint of BOTH an
  # (axis-aligned) horizontal and a vertical bar, those two footprints
  # would necessarily overlap by a w*w patch there, contradicting "Zero
  # overlap". What's both checkable and actually meaningful is "no gap":
  # every corner is covered by the UNION of the four segments -- verified
  # here directly from the returned geometry, no implementation internals
  # assumed.)
  for sxg in (1, -1):
    for syg in (1, -1):
      corner = (cx + sxg * h, cy + syg * h)
      covered = any(_footprint_contains(s, *corner) for s in segs)
      assert covered, f"corner {corner} not covered by any segment -- gap"


# ===========================================================================
# C. draw_table_markers
# ===========================================================================


def test_draw_table_markers_appends_exactly_seven(model, data, ids,
                                                    fake_scene_factory):
  scene = fake_scene_factory(64)
  assert scene.ngeom == 0
  render_sim_rollout.draw_table_markers(scene, model, data, ids)
  assert scene.ngeom == 7


def test_draw_table_markers_decor_flags(model, data, ids, fake_scene_factory):
  scene = fake_scene_factory(64)
  render_sim_rollout.draw_table_markers(scene, model, data, ids)
  assert scene.ngeom == 7
  for i in range(scene.ngeom):
    g = scene.geoms[i]
    assert g.category == mujoco.mjtCatBit.mjCAT_DECOR
    assert g.segid == -1
    assert g.objid == -1


def test_draw_table_markers_within_table_extent(model, data, ids,
                                                  fake_scene_factory):
  scene = fake_scene_factory(64)
  render_sim_rollout.draw_table_markers(scene, model, data, ids)
  assert scene.ngeom == 7

  lifter_xpos = data.xpos[ids["lifter_body"]].copy()
  R = data.xmat[ids["lifter_body"]].reshape(3, 3).copy()
  for i in range(scene.ngeom):
    world = np.array(scene.geoms[i].pos, dtype=float)
    local = R.T @ (world - lifter_xpos)
    assert abs(local[0]) <= 0.30 + 1e-6, f"geom {i} local x {local[0]} outside table"
    assert abs(local[1]) <= 0.50 + 1e-6, f"geom {i} local y {local[1]} outside table"


def test_draw_table_markers_shape_counts(model, data, ids, fake_scene_factory):
  scene = fake_scene_factory(64)
  render_sim_rollout.draw_table_markers(scene, model, data, ids)
  assert scene.ngeom == 7
  n_box = sum(1 for i in range(7)
              if scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_BOX)
  n_cyl = sum(1 for i in range(7)
              if scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_CYLINDER)
  assert n_box == 4
  assert n_cyl == 3


# NOTE on tolerances below: draw_table_markers writes marker positions into
# real mujoco.MjvGeom objects, whose `.pos` field is float32 (verified: a
# fresh mujoco.MjvGeom()'s `.pos` array has dtype float32). Round-tripping
# float64 world positions (~0.1-0.4 m magnitude) through that float32 store
# unavoidably loses ~1e-8-1e-7 in the deltas/projections checked below, no
# matter how correct the implementation is. The task's literal "1e-9"
# tolerance is tighter than float32 can represent at these magnitudes, so
# these two checks use 1e-6 instead -- comfortably above the float32 noise
# floor while still many orders of magnitude tighter than any real bug
# (e.g. a marker not moving with the table at all, or the wrong delta sign)
# would produce.
_FLOAT32_TOL = 1e-6


def test_draw_table_markers_rides_with_table_on_raise(model, ids,
                                                        fake_scene_factory):
  # Baseline: fresh MjData, default pose.
  data0 = mujoco.MjData(model)
  mujoco.mj_forward(model, data0)
  scene0 = fake_scene_factory(64)
  render_sim_rollout.draw_table_markers(scene0, model, data0, ids)
  assert scene0.ngeom == 7
  base_z = [float(scene0.geoms[i].pos[2]) for i in range(7)]

  # Raised: a SEPARATE fresh MjData -- never mutates the shared `data`
  # fixture, so nothing here can leak into another test.
  data1 = mujoco.MjData(model)
  mujoco.mj_forward(model, data1)
  lifter_body = ids["lifter_body"]
  lifter_mocap = ids["lifter_mocap"]
  z_before = float(data1.xpos[lifter_body][2])

  data1.mocap_pos[lifter_mocap][2] = 0.1475  # table top -> 0.150 m
  mujoco.mj_forward(model, data1)
  z_after = float(data1.xpos[lifter_body][2])
  table_delta = z_after - z_before
  assert table_delta > 0.05  # sanity: the table actually moved

  scene1 = fake_scene_factory(64)
  render_sim_rollout.draw_table_markers(scene1, model, data1, ids)
  assert scene1.ngeom == 7
  for i in range(7):
    raised_z = float(scene1.geoms[i].pos[2])
    marker_delta = raised_z - base_z[i]
    assert abs(marker_delta - table_delta) < _FLOAT32_TOL, (
        f"marker {i}: rose by {marker_delta}, table rose by {table_delta}")


def test_draw_table_markers_rides_with_table_on_tilt(model, ids,
                                                       fake_scene_factory):
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  lifter_body = ids["lifter_body"]
  lifter_mocap = ids["lifter_mocap"]

  # 0.1 rad roll about x (half-angle 0.05), MuJoCo wxyz convention.
  data.mocap_quat[lifter_mocap] = np.array(
      [np.cos(0.05), np.sin(0.05), 0.0, 0.0])
  mujoco.mj_forward(model, data)

  R_tilted = data.xmat[lifter_body].reshape(3, 3).copy()
  normal = R_tilted[:, 2]
  lifter_xpos = data.xpos[lifter_body].copy()

  scene = fake_scene_factory(64)
  render_sim_rollout.draw_table_markers(scene, model, data, ids)
  assert scene.ngeom == 7

  projections = []
  for i in range(7):
    world = np.array(scene.geoms[i].pos, dtype=float)
    proj = float((world - lifter_xpos) @ normal)
    projections.append(proj)

  for proj in projections:
    assert abs(proj - 0.0035) < _FLOAT32_TOL, projections
  # All 7 equal to each other, not just each individually close to 0.0035.
  assert max(projections) - min(projections) < _FLOAT32_TOL, projections


# ===========================================================================
# D. _append_decor_geom
# ===========================================================================


def _decor_geom_args():
  gtype = mujoco.mjtGeom.mjGEOM_BOX
  size = np.array([0.01, 0.01, 0.001])
  pos = np.array([0.1, 0.1, 0.15])
  mat = np.eye(3).flatten()
  rgb = np.array([1.0, 0.0, 0.0])
  return gtype, size, pos, mat, rgb


def test_append_decor_geom_full_scene_returns_false(fake_scene_factory):
  scene = fake_scene_factory(3)
  scene.ngeom = 3  # simulate an already-full scene
  gtype, size, pos, mat, rgb = _decor_geom_args()
  result = render_sim_rollout._append_decor_geom(scene, gtype, size, pos, mat, rgb)
  assert result is False
  assert scene.ngeom == 3


def test_append_decor_geom_normal_append_increments(fake_scene_factory):
  scene = fake_scene_factory(5)
  gtype, size, pos, mat, rgb = _decor_geom_args()
  result = render_sim_rollout._append_decor_geom(scene, gtype, size, pos, mat, rgb)
  assert result is True
  assert scene.ngeom == 1
  g = scene.geoms[0]
  assert g.type == gtype
  np.testing.assert_allclose(np.array(g.pos), pos, atol=1e-6)


def test_append_decor_geom_mat_3x3_reshapes_cleanly(fake_scene_factory):
  """DOCUMENTS the mjv_initGeom contract: mjv_initGeom itself requires a
  FLAT length-9 mat and raises TypeError on a (3,3) array (verified
  separately against the raw mujoco binding). The landed implementation of
  _append_decor_geom, however, does `np.asarray(mat, dtype=float).reshape(9)`
  before calling mjv_initGeom -- and reshape(3,3 -> 9) is a valid,
  exception-free reshape (9 elements either way). So per this test's own
  contract note ("if the implementation reshapes internally this test
  should be relaxed to assert the reshape works instead"), THIS is the
  relaxed form: a (3,3) mat must NOT raise, and the resulting geom's mat
  must equal the flattened input.
  """
  scene = fake_scene_factory(5)
  gtype, size, pos, _mat, rgb = _decor_geom_args()
  mat_3x3 = np.eye(3)

  result = render_sim_rollout._append_decor_geom(
      scene, gtype, size, pos, mat_3x3, rgb)

  assert result is True
  assert scene.ngeom == 1
  g = scene.geoms[0]
  np.testing.assert_allclose(
      np.array(g.mat).reshape(3, 3), mat_3x3, atol=1e-6)


# ===========================================================================
# E. Mirror-drift guard.
# ===========================================================================


def test_gap_metrics_tape_mark_xy_contract():
  assert set(gap_metrics.TAPE_MARK_XY.keys()) == {"P1", "P2", "P3"}
  assert gap_metrics.TAPE_MARK_XY["P1"] == (0.216, -0.185)
  assert gap_metrics.TAPE_MARK_XY["P2"] == (0.355, -0.009)
  assert gap_metrics.TAPE_MARK_XY["P3"] == (0.495, 0.188)


def test_render_sim_rollout_mirrors_gap_metrics_constants():
  assert render_sim_rollout.TAPE_MARK_XY == gap_metrics.TAPE_MARK_XY
  assert (render_sim_rollout.DROP_SQUARE_CENTER
          == gap_metrics.DROP_SQUARE_CENTER)
  assert (render_sim_rollout.DROP_SQUARE_HALF_WIDTH
          == gap_metrics.DROP_SQUARE_HALF_WIDTH)


# ===========================================================================
# F. composite.build_command
# ===========================================================================


def _composite_args(run_dir):
  return argparse.Namespace(
      run_dir=str(run_dir), sim=None, real=None, out=None, fps=30,
      height=1080, dry_run=True)


def test_build_command_raises_on_external_source_no_frame_csv(tmp_path):
  # This is the EXACT video block record_real_rollout.py --no-video emits:
  # real_mp4 and webcam_fps_nominal are nulled too, not just frame_index_csv.
  # An earlier fixture left real_mp4 as a real path, which hid a TypeError
  # raised while resolving real_mp4 BEFORE the guard could fire.
  manifest = {
      "t0_unix": 1000.0,
      "control": {"ctrl_dt": 0.02, "episode_length": 5},
      "video": {
          "real_mp4": None,
          "webcam_fps_nominal": None,
          "n_webcam_frames": 0,
          "frame_index_csv": None,
          "source": "external",
      },
  }
  (tmp_path / "manifest.json").write_text(json.dumps(manifest))

  args = _composite_args(tmp_path)
  with pytest.raises(SystemExit) as excinfo:
    composite.build_command(args)
  assert "external" in str(excinfo.value)


def test_build_command_normal_webcam_manifest_returns_ffmpeg_argv(tmp_path):
  manifest = {
      "t0_unix": 1000.0,
      "control": {"ctrl_dt": 0.02, "episode_length": 5},
      "video": {
          "real_mp4": "real.mp4",
          "frame_index_csv": "frame_index.csv",
          "source": "webcam",
      },
  }
  (tmp_path / "manifest.json").write_text(json.dumps(manifest))
  (tmp_path / "frame_index.csv").write_text(
      "frame_idx,t_unix,t_rel,ctrl_step\n"
      "0,1000.0,-0.50,-1\n"
      "1,1000.02,-0.48,-1\n"
  )

  args = _composite_args(tmp_path)
  cmd, out_path = composite.build_command(args)
  assert cmd[0] == "ffmpeg"
  assert out_path is not None


# ===========================================================================
# G. record_real_rollout.py -- SOURCE LEVEL ONLY. Never imported.
# ===========================================================================


def test_record_real_rollout_compiles(tmp_path):
  # cfile redirected into tmp_path/pytest's tmp dir so this NEVER writes a
  # __pycache__/*.pyc under the repo (this suite may only write its own two
  # owned files).
  py_compile.compile(
      _RECORD_REAL_ROLLOUT_PATH, cfile=str(tmp_path / "record_real_rollout.pyc"),
      doraise=True)


def _parse_record_real_rollout():
  with open(_RECORD_REAL_ROLLOUT_PATH, "r", encoding="utf-8") as f:
    src = f.read()
  tree = ast.parse(src, filename=_RECORD_REAL_ROLLOUT_PATH)
  return src, tree


def _call_func_name(node):
  if not isinstance(node, ast.Call):
    return None
  return getattr(node.func, "attr", None) or getattr(node.func, "id", None)


def _add_argument_flags(tree):
  """Every flag string passed to an add_argument() call in the file."""
  flags = set()
  for node in ast.walk(tree):
    if _call_func_name(node) != "add_argument" or not node.args:
      continue
    for a in node.args:
      if isinstance(a, ast.Constant) and isinstance(a.value, str):
        flags.add(a.value)
  return flags


def test_record_real_rollout_records_no_video():
  """The webcam is gone: filming is external, this script only drives the arm.

  Guards the whole removal, not just the flag: no capture library, no worker
  class, no camera flags. A future edit that reintroduces any of them has to
  update this test deliberately.
  """
  src, tree = _parse_record_real_rollout()

  imported = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imported.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      imported.add(node.module.split(".")[0])
  for mod in ("cv2", "imageio", "threading"):
    assert mod not in imported, (
        f"{mod!r} is imported again -- the webcam was removed from this "
        "script; video is filmed externally")

  assert "_WebcamWorker" not in src, "the _WebcamWorker class is back"

  flags = _add_argument_flags(tree)
  camera_flags = sorted(f for f in flags
                        if f.startswith("--camera") or f == "--no-video")
  assert not camera_flags, f"camera flags are back: {camera_flags}"

  # The manifest's video block must still be EMITTED (all-null) -- dropping it
  # would break render_sim_rollout.validate_manifest, which requires the keys
  # to be present.
  for key in ("real_mp4", "webcam_fps_nominal", "n_webcam_frames",
              "frame_index_csv"):
    assert f'"{key}"' in src, (
        f"manifest video key {key!r} is missing -- render_sim_rollout's "
        "validate_manifest requires it to be present even when null")


def test_record_real_rollout_manifest_video_block_has_source_key():
  _src, tree = _parse_record_real_rollout()
  found = False
  for node in ast.walk(tree):
    if isinstance(node, ast.Dict):
      for k in node.keys:
        if isinstance(k, ast.Constant) and k.value == "source":
          found = True
          break
    if found:
      break
  assert found, 'no dict literal with key "source" found (manifest video block)'


def _walk_all(nodes):
  for n in nodes:
    yield from ast.walk(n)


def _contains_string_literal(nodes, s):
  return any(isinstance(n, ast.Constant) and n.value == s for n in nodes)


def _contains_call_named(nodes, name):
  return any(_call_func_name(n) == name for n in nodes)


def test_record_real_rollout_always_writes_states_csv():
  """real_states.csv is unconditional -- it is the run's only data artifact.

  (Superseded the old no_video-gating test: with the webcam gone there is no
  video branch left for it to be gated on, but the CSV must still be written
  on every path.)
  """
  _src, tree = _parse_record_real_rollout()
  all_nodes = list(ast.walk(tree))
  assert _contains_string_literal(all_nodes, "real_states.csv")
  assert _contains_call_named(all_nodes, "_build_states_csv")


def test_record_real_rollout_never_blocks_on_input():
  """No hard stops: the run must go through unattended once Play is pressed.

  Any input()/_confirm()/_press_enter() would strand the arm mid-sequence
  waiting on a keypress, which is exactly what was removed.
  """
  _src, tree = _parse_record_real_rollout()
  all_nodes = list(ast.walk(tree))
  for name in ("input", "_confirm", "_press_enter"):
    assert not _contains_call_named(all_nodes, name), (
        f"{name}() is called again -- this script must run unattended; use "
        "_countdown() for a timed window instead")
  assert "--yes" not in _add_argument_flags(tree), (
      "--yes is back, which only makes sense if something blocks on input")


def test_record_real_rollout_starts_from_task_home_by_default():
  """Default start pose is task_home, read from the XML rather than hardcoded."""
  _src, tree = _parse_record_real_rollout()
  for node in ast.walk(tree):
    if _call_func_name(node) != "add_argument" or not node.args:
      continue
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and first.value == "--start-pose"):
      continue
    default = next(
        (kw.value for kw in node.keywords if kw.arg == "default"), None)
    assert isinstance(default, ast.Constant), "--start-pose has no default"
    assert default.value == "task_home", (
        f"--start-pose default is {default.value!r}, expected 'task_home'")
    choices = next(
        (kw.value for kw in node.keywords if kw.arg == "choices"), None)
    got = {e.value for e in choices.elts} if choices is not None else set()
    assert {"task_home", "nearest_home"} <= got, (
        f"--start-pose choices {sorted(got)} must offer task_home and "
        "nearest_home")
    break
  else:
    raise AssertionError("no --start-pose argument found")

  assert _contains_call_named(list(ast.walk(tree)), "_task_home_qpos"), (
      "task_home is not read from the scene XML via _task_home_qpos()")


def _required_add_arguments(tree):
  """{flag: True} for every add_argument(..., required=True) in the file."""
  out = {}
  for node in ast.walk(tree):
    if _call_func_name(node) != "add_argument" or not node.args:
      continue
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
      continue
    for kw in node.keywords:
      if (kw.arg == "required" and isinstance(kw.value, ast.Constant)
          and kw.value.value is True):
        out[first.value] = True
  return out


def test_record_real_rollout_runs_without_required_flags():
  """A bare `--config-id L0_none` must be a complete invocation.

  The original failure this guards: running the script with no arguments
  produced only an argparse usage dump, because --name and --checkpoint were
  both required=True. The checkpoint is now resolved from the gap protocol's
  policy map instead.
  """
  _src, tree = _parse_record_real_rollout()
  required = _required_add_arguments(tree)
  assert "--name" not in required, "--name is required=True again"
  assert "--checkpoint" not in required, "--checkpoint is required=True again"
  assert not required, (
      f"record_real_rollout.py must be runnable with no flags, but these are "
      f"required=True: {sorted(required)}")

  # ...and the fallback that makes --checkpoint optional must be present.
  all_nodes = list(ast.walk(tree))
  assert _contains_call_named(all_nodes, "resolve_policy_run_id"), (
      "no gap.resolve_policy_run_id() call -- --checkpoint is optional but "
      "nothing resolves it from gap_protocol_policy_map.json")


def test_record_real_rollout_drops_the_cube_at_the_end():
  """The D22 drop must exist, and must NOT sit in the teardown.

  A drop inside `finally` would fire after robot/mocap are torn down (and on
  every failure path), so it could neither be scored nor filmed. It has to run
  inside the try, right after run_policy_loop returns.
  """
  _src, tree = _parse_record_real_rollout()
  all_nodes = list(ast.walk(tree))
  assert _contains_call_named(all_nodes, "open_gripper"), (
      "no open_gripper() call -- the D22 drop is missing")
  assert _contains_call_named(all_nodes, "place_error"), (
      "no place_error() call -- the drop is never scored")

  for node in ast.walk(tree):
    if not isinstance(node, ast.Try) or not node.finalbody:
      continue
    final_nodes = list(_walk_all(node.finalbody))
    assert not _contains_call_named(final_nodes, "open_gripper"), (
        "open_gripper() found inside a `finally:` block -- the drop must "
        "happen in the try, before teardown, or it cannot be filmed or scored")
    assert not _contains_call_named(final_nodes, "place_error"), (
        "place_error() found inside a `finally:` block")


def test_record_real_rollout_target_uses_base_frame_box():
  """target_for_episode's box_xy must be the BASE-frame box, not raw mocap.

  gap_target.target_for_episode documents box_xy as base frame and derives the
  target bearing from it. The mocap calibration is a ~180 deg Z rotation plus a
  ~0.37 m offset, so passing the raw mocap XY silently produces a target at an
  unrelated azimuth -- and makes the manifest self-inconsistent, since
  init.box_pos is logged in base frame.
  """
  src, tree = _parse_record_real_rollout()
  found = False
  for node in ast.walk(tree):
    if _call_func_name(node) != "target_for_episode":
      continue
    box_xy = next(
        (kw.value for kw in node.keywords if kw.arg == "box_xy"), None)
    assert box_xy is not None, "target_for_episode called without box_xy="
    seg = ast.get_source_segment(src, box_xy) or ""
    assert "box_pos_base" in seg, (
        f"target_for_episode(box_xy={seg!r}) is not the base-frame box; "
        "pass box_pos_base[:2], not the raw mocap box_xyz")
    found = True
  assert found, "no target_for_episode() call found"


# ===========================================================================
# H. run_policy_loop max_steps horizon cap -- SOURCE LEVEL ONLY.
#    robots/UR3e/ur3_realrobot_dependencies.py is not importable here (RTDE +
#    vrpn), so this section uses ast only, exactly like section G.
# ===========================================================================

_REALROBOT_DEPS_PATH = os.path.join(
    REPO_ROOT, "robots", "UR3e", "ur3_realrobot_dependencies.py")


def _parse_realrobot_deps():
  with open(_REALROBOT_DEPS_PATH, "r", encoding="utf-8") as f:
    src = f.read()
  return src, ast.parse(src, filename=_REALROBOT_DEPS_PATH)


def _run_policy_loop_def(tree):
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "run_policy_loop":
      return node
  raise AssertionError("run_policy_loop not found")


def _parent_map(root):
  """child id -> parent node, over the subtree rooted at `root`."""
  parents = {}
  for parent in ast.walk(root):
    for child in ast.iter_child_nodes(parent):
      parents[id(child)] = parent
  return parents


def _guarding_if(node, parents):
  """The nearest enclosing ast.If of `node`, or None."""
  cur = parents.get(id(node))
  while cur is not None:
    if isinstance(cur, ast.If):
      return cur
    cur = parents.get(id(cur))
  return None


def _stopped_reason_assigns(fn, value):
  """Every `stopped_reason = <value>` Assign inside run_policy_loop."""
  out = []
  for node in ast.walk(fn):
    if not isinstance(node, ast.Assign):
      continue
    if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "stopped_reason"):
      continue
    if isinstance(node.value, ast.Constant) and node.value.value == value:
      out.append(node)
  return out


def test_run_policy_loop_max_steps_defaults_to_none():
  """max_steps must exist AND default to None.

  None is the load-bearing default: run_policy_loop is shared with
  evaluation/run_gap_protocol.py, which must keep running to timeout_s /
  reach_tol exactly as before. A non-None default would silently truncate
  every gap-protocol episode.
  """
  _src, tree = _parse_realrobot_deps()
  fn = _run_policy_loop_def(tree)

  names = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
  assert "max_steps" in names, f"max_steps not in run_policy_loop args: {names}"

  if "max_steps" in [a.arg for a in fn.args.args]:
    idx = [a.arg for a in fn.args.args].index("max_steps")
    # defaults align to the TAIL of args
    offset = len(fn.args.args) - len(fn.args.defaults)
    assert idx >= offset, "max_steps has no default at all"
    default = fn.args.defaults[idx - offset]
  else:
    idx = [a.arg for a in fn.args.kwonlyargs].index("max_steps")
    default = fn.args.kw_defaults[idx]

  assert isinstance(default, ast.Constant) and default.value is None, (
      f"max_steps default is {ast.dump(default)}, must be the None constant")


def test_run_policy_loop_horizon_break_is_guarded():
  """Exactly one `stopped_reason = "horizon"`, and it is unreachable unless
  the caller opted in with max_steps."""
  src, tree = _parse_realrobot_deps()
  fn = _run_policy_loop_def(tree)
  parents = _parent_map(fn)

  assigns = _stopped_reason_assigns(fn, "horizon")
  assert len(assigns) == 1, (
      f"expected exactly 1 stopped_reason = 'horizon' assignment, "
      f"found {len(assigns)}")

  guard = _guarding_if(assigns[0], parents)
  assert guard is not None, "the horizon assignment sits outside any `if`"
  seg = ast.get_source_segment(src, guard.test) or ""
  assert "max_steps is not None" in seg, (
      f"horizon break guarded by {seg!r}, which does not test "
      "`max_steps is not None` -- with max_steps=None the old code path must "
      "be provably unreachable")


def test_run_policy_loop_timeout_break_unchanged():
  """The pre-existing timeout exit is untouched by the max_steps addition."""
  src, tree = _parse_realrobot_deps()
  fn = _run_policy_loop_def(tree)
  parents = _parent_map(fn)

  assigns = _stopped_reason_assigns(fn, "timeout")
  assert len(assigns) == 1, (
      f"expected exactly 1 stopped_reason = 'timeout' assignment, "
      f"found {len(assigns)}")

  guard = _guarding_if(assigns[0], parents)
  assert guard is not None, "the timeout assignment sits outside any `if`"
  seg = (ast.get_source_segment(src, guard.test) or "").strip()
  assert seg == "elapsed >= timeout_s", (
      f"timeout break guard is now {seg!r}, was `elapsed >= timeout_s`")


# ===========================================================================
# I. record_real_rollout.py 400-step contract -- SOURCE LEVEL ONLY.
#    (pinned checkpoint, pinned drop square, enforced horizon)
# ===========================================================================


def _module_constant(tree, name):
  """The literal value of a top-level `NAME = <literal>` assignment."""
  for node in tree.body:
    if not isinstance(node, ast.Assign):
      continue
    if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
      return ast.literal_eval(node.value)
  raise AssertionError(f"module constant {name} not found")


def _add_argument_kwargs(tree, flag):
  """The keyword arguments of the add_argument() call declaring `flag`."""
  for node in ast.walk(tree):
    if _call_func_name(node) != "add_argument":
      continue
    if not any(isinstance(a, ast.Constant) and a.value == flag
               for a in node.args):
      continue
    return {kw.arg: kw.value for kw in node.keywords}
  raise AssertionError(f"no add_argument() for {flag}")


def test_record_real_rollout_drop_square_matches_gap_metrics():
  """DROP_SQUARE_TARGET is the THIRD copy of the drop-square centre.

  gap_metrics.py owns it, render_sim_rollout.py mirrors it (guarded by
  test_render_sim_rollout_mirrors_gap_metrics_constants), and this is the
  third. A drift between any two of them aims the real take and the sim
  replay at different physical points.
  """
  _src, tree = _parse_record_real_rollout()
  target = _module_constant(tree, "DROP_SQUARE_TARGET")

  assert len(target) == 3, f"DROP_SQUARE_TARGET is not a 3-vector: {target}"
  assert tuple(target[:2]) == tuple(gap_metrics.DROP_SQUARE_CENTER), (
      f"DROP_SQUARE_TARGET[:2] = {tuple(target[:2])} but "
      f"gap_metrics.DROP_SQUARE_CENTER = {gap_metrics.DROP_SQUARE_CENTER}")
  # box_z_anchor 0.115 + the 0.05 lift the L0_deterministic profile draws.
  assert target[2] == 0.165, f"DROP_SQUARE_TARGET[2] = {target[2]}, want 0.165"

  # and it really is inside the taped square, not merely equal to a constant
  dist, inside = gap_metrics.place_error(tuple(target[:2]))
  assert inside and dist == 0.0, (dist, inside)


def test_record_real_rollout_forwards_max_steps():
  """The horizon must reach the loop, not just size the watchdog.

  Without max_steps=, --episode-length only fed timeout_s, so the loop ran
  400 * 0.02 * 1.12 + 0.5 = 9.46 s and logged ~473 steps at 50 Hz.
  """
  src, tree = _parse_record_real_rollout()
  calls = [n for n in ast.walk(tree)
           if _call_func_name(n) == "run_policy_loop"]
  assert calls, "no run_policy_loop() call found"
  for call in calls:
    kw = next((k for k in call.keywords if k.arg == "max_steps"), None)
    assert kw is not None, (
        "run_policy_loop() called without max_steps= -- the horizon is not "
        "enforced and the sim replay cannot be frame-matched")
    seg = ast.get_source_segment(src, kw.value) or ""
    assert "episode_length" in seg, (
        f"max_steps={seg!r} is not derived from episode_length")


def test_record_real_rollout_episode_length_defaults_to_400():
  """400 is the trained episode_length; the default must not be None."""
  _src, tree = _parse_record_real_rollout()
  kwargs = _add_argument_kwargs(tree, "--episode-length")
  default = kwargs.get("default")
  assert default is not None, "--episode-length has no default= at all"
  value = ast.literal_eval(default)
  assert value == 400, f"--episode-length default is {value}, want 400"


def test_record_real_rollout_pins_snappy2_checkpoint():
  """SELECT_CHECKPOINT is pinned, so the policy map is never consulted."""
  _src, tree = _parse_record_real_rollout()
  ckpt = _module_constant(tree, "SELECT_CHECKPOINT")
  assert ckpt == "Snappy2_as04_ar70_g01_s1_20260817_202921_2201", (
      f"SELECT_CHECKPOINT is {ckpt!r}")


def test_record_real_rollout_drop_target_flags_are_exclusive():
  """--drop-target and --drop-target-from-protocol must not both apply.

  The pinned target is now the default, so the protocol draw is opt-in.
  Passing both is a contradiction (pin a point / redraw one) and must exit
  rather than silently letting one win.
  """
  src, tree = _parse_record_real_rollout()
  flags = _add_argument_flags(tree)
  assert "--drop-target-from-protocol" in flags, (
      "no --drop-target-from-protocol escape hatch -- pinning the target "
      "removed the per-episode draw with no way back")

  found = False
  for node in ast.walk(tree):
    if not isinstance(node, ast.If):
      continue
    test_seg = ast.get_source_segment(src, node.test) or ""
    if "drop_target_from_protocol" not in test_seg:
      continue
    body_seg = "\n".join(
        ast.get_source_segment(src, b) or "" for b in node.body)
    if "SystemExit" in body_seg and "drop_target" in body_seg:
      found = True
  assert found, (
      "no `if args.drop_target_from_protocol:` branch raising SystemExit on a "
      "conflicting explicit --drop-target")


# ===========================================================================
# J. render_sim_rollout.py -> compare_tcp wiring + the 400-step horizon guard.
# ===========================================================================

_RENDER_SIM_ROLLOUT_PATH = os.path.join(_DEFENCE_DIR, "render_sim_rollout.py")


def _parse_render_sim_rollout():
  with open(_RENDER_SIM_ROLLOUT_PATH, "r", encoding="utf-8") as f:
    src = f.read()
  return src, ast.parse(src, filename=_RENDER_SIM_ROLLOUT_PATH)


def test_render_sim_rollout_has_no_compare_flag():
  """--no-compare must exist: the auto-run has to be switchable off."""
  _src, tree = _parse_render_sim_rollout()
  flags = _add_argument_flags(tree)
  assert "--no-compare" in flags, (
      f"--no-compare not among the declared flags: {sorted(flags)}")


def test_render_sim_rollout_imports_compare_tcp_not_subprocess():
  """compare_tcp runs IN-PROCESS.

  Shelling out would turn compare_tcp's SystemExit on a real/sim length
  mismatch into an exit code nobody reads -- the loudest failure in the
  pipeline would become the quietest.
  """
  src, tree = _parse_render_sim_rollout()

  imported = False
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imported |= any(a.name == "compare_tcp" for a in node.names)
    elif isinstance(node, ast.ImportFrom):
      imported |= (node.module == "compare_tcp")
  assert imported, "render_sim_rollout.py never imports compare_tcp"

  called = any(
      _call_func_name(n) == "compare_run" for n in ast.walk(tree))
  assert called, "compare_tcp.compare_run() is never called"

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      assert not any(a.name == "subprocess" for a in node.names), (
          "render_sim_rollout.py imports subprocess -- compare_tcp must be "
          "called in-process, not shelled out")
    elif isinstance(node, ast.ImportFrom):
      assert node.module != "subprocess", (
          "render_sim_rollout.py imports from subprocess")

  # ...and no other way of shelling out either. Checked as CALLS rather than
  # as a substring search, so the comment in the source explaining why we do
  # not shell out is not itself a violation.
  shell_outs = {"system", "popen", "spawnl", "spawnv", "execv"}
  for node in ast.walk(tree):
    name = _call_func_name(node)
    if name in shell_outs:
      seg = ast.get_source_segment(src, node) or name
      raise AssertionError(
          f"render_sim_rollout.py shells out: {seg!r}. compare_tcp must be "
          "called in-process.")


def test_render_sim_rollout_warns_off_the_trained_horizon():
  """A non-400 episode_length warns; it must NOT raise.

  The eight archived 2026-08-14 runs are 153-473 steps (they exited on
  force_limit or timeout, before max_steps existed) and have to stay
  re-renderable, so this guard is deliberately non-fatal.
  """
  src, tree = _parse_render_sim_rollout()

  assert _module_constant(tree, "TRAINED_EPISODE_LENGTH") == 400

  found = False
  for node in ast.walk(tree):
    if not isinstance(node, ast.If):
      continue
    test_seg = ast.get_source_segment(src, node.test) or ""
    if "episode_length" not in test_seg:
      continue
    if "TRAINED_EPISODE_LENGTH" not in test_seg and "400" not in test_seg:
      continue
    body = list(_walk_all(node.body))
    if not _contains_call_named(body, "print"):
      continue
    assert not any(isinstance(n, ast.Raise) for n in body), (
        "the horizon guard raises; it must only warn, or the archived "
        "2026-08-14 runs stop being re-renderable")
    found = True
  assert found, "no non-fatal episode_length != 400 warning found"


# ===========================================================================
# The run directory must name the policy that actually ran.
#
# `args.name = args.config_id` meant a pinned snappy2 checkpoint still wrote
# into `defence/runs/<UTC>_L1_pos/`, i.e. a directory naming a rung whose
# policy was never loaded. These pin the replacement.
# ===========================================================================


def _exec_default_run_name():
  """`_default_run_name` executed in isolation, without importing the module.

  record_real_rollout.py cannot be imported here (Linux-only vrpn/rtde), but
  this helper is pure: lift its source out of the AST and exec it against a
  namespace holding only `re`. That tests real behaviour rather than shape.
  """
  src, tree = _parse_record_real_rollout()
  fn = next((n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "_default_run_name"),
            None)
  assert fn is not None, "_default_run_name is gone from record_real_rollout.py"
  ns = {"re": re}
  exec(compile(ast.Module(body=[fn], type_ignores=[]), "<lifted>", "exec"), ns)
  return ns["_default_run_name"]


@pytest.mark.parametrize("checkpoint,expected", [
    # The pinned defence policy: W&B stamp stripped, cell identity kept.
    ("Snappy2_as04_ar70_g01_s1_20260817_202921_2201", "Snappy2_as04_ar70_g01_s1"),
    # An L-rung checkpoint, same shape.
    ("L2_pos_cube_vel_s1_20260729_112246_2201", "L2_pos_cube_vel_s1"),
    ("L0_none_vel_s1_20260729_104930_2201", "L0_none_vel_s1"),
    # Not the W&B shape -> returned whole, never guessed at or truncated.
    ("some_hand_made_checkpoint", "some_hand_made_checkpoint"),
    ("trailing_20260817_202921", "trailing_20260817_202921"),
])
def test_default_run_name_strips_only_the_wandb_stamp(checkpoint, expected):
  assert _exec_default_run_name()(checkpoint) == expected


def test_default_run_name_never_returns_a_rung_for_the_pinned_checkpoint():
  """The regression itself: the pinned take must not be named after a rung."""
  _, tree = _parse_record_real_rollout()
  ckpt = _module_constant(tree, "SELECT_CHECKPOINT")
  config_id = _module_constant(tree, "SELECT_CONFIG_ID")
  name = _exec_default_run_name()(ckpt)
  assert name != config_id, (
      f"default run name {name!r} equals SELECT_CONFIG_ID {config_id!r} — the "
      "run directory would name a rung instead of the pinned policy")
  assert name.startswith("Snappy2"), name


def test_run_name_default_comes_from_the_checkpoint_not_the_config_id():
  """`args.name = _default_run_name(args.checkpoint)`, not `args.config_id`."""
  _, tree = _parse_record_real_rollout()
  main = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
  assigns = [n for n in ast.walk(main)
             if isinstance(n, ast.Assign)
             and any(getattr(t, "attr", None) == "name" for t in n.targets)]
  assert assigns, "nothing assigns args.name in main()"
  for a in assigns:
    rendered = ast.dump(a.value)
    assert "config_id" not in rendered, (
        "args.name is still derived from config_id — a pinned checkpoint "
        "would write into a directory named after the wrong policy")
    assert _call_func_name(a.value) == "_default_run_name", ast.dump(a.value)


# ===========================================================================
# The checkpoint must be fetched however it was chosen.
#
# The download used to live INSIDE `if args.checkpoint is None:`, so a pinned
# checkpoint (SELECT_CHECKPOINT / --checkpoint) was never downloaded -- it had
# to already be on disk, and otherwise the run died at the "checkpoint
# incomplete" guard. The pinned path is the defence path, so it must fetch.
# ===========================================================================


def _main_func(tree):
  return next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")


def _enclosing_if_tests(root, target):
  """Source-ish dump of every `if` test that encloses `target` inside `root`."""
  tests = []

  def walk(node, stack):
    if node is target:
      tests.extend(stack)
      return True
    for child in ast.iter_child_nodes(node):
      nxt = stack + [ast.dump(node.test)] if isinstance(node, ast.If) else stack
      if walk(child, nxt if child in getattr(node, "body", []) else stack):
        return True
    return False

  walk(root, [])
  return tests


def test_checkpoint_download_is_not_gated_on_an_unpinned_checkpoint():
  """The fetch must not sit under `if args.checkpoint is None`."""
  _, tree = _parse_record_real_rollout()
  main = _main_func(tree)
  calls = [n for n in ast.walk(main)
           if _call_func_name(n) == "download_policy_from_wandb"]
  assert calls, ("record_real_rollout.py no longer calls "
                 "download_policy_from_wandb anywhere in main()")
  for call in calls:
    for test in _enclosing_if_tests(main, call):
      assert "checkpoint" not in test or "None" not in test, (
          "the W&B download is gated on the checkpoint being unpinned; a "
          f"pinned SELECT_CHECKPOINT would never be fetched. Guard: {test}")


def test_checkpoint_download_goes_through_the_ur3e_dependency():
  """One way to obtain a policy: the UR3e driver's own staticmethod."""
  src, tree = _parse_record_real_rollout()
  main = _main_func(tree)
  names = {_call_func_name(n) for n in ast.walk(main)}
  assert "download_policy_from_wandb" in names
  assert "download_policy" not in names, (
      "main() calls policy_downloader.download_policy directly; route it "
      "through UR3RealRobotPick.download_policy_from_wandb instead")
  # And the direct import is gone, so there is no second route left open.
  assert "    download_policy,\n" not in src


def test_checkpoint_download_is_guarded_by_a_cache_check():
  """Fetch only when missing -- a cached checkpoint must not re-download."""
  _, tree = _parse_record_real_rollout()
  main = _main_func(tree)
  call = next(n for n in ast.walk(main)
              if _call_func_name(n) == "download_policy_from_wandb")
  tests = _enclosing_if_tests(main, call)
  assert any("policy_exists_locally" in t for t in tests), (
      f"download is unconditional -- it would refetch every run. Guards: {tests}")
