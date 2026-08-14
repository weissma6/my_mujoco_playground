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


def test_record_real_rollout_has_no_video_flag():
  _src, tree = _parse_record_real_rollout()
  found = False
  for node in ast.walk(tree):
    if _call_func_name(node) == "add_argument" and node.args:
      first = node.args[0]
      if isinstance(first, ast.Constant) and first.value == "--no-video":
        found = True
        break
  assert found, "no add_argument(\"--no-video\", ...) call found"


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


def test_record_real_rollout_states_csv_never_gated_on_no_video():
  src, tree = _parse_record_real_rollout()

  all_nodes = list(ast.walk(tree))
  assert _contains_string_literal(all_nodes, "real_states.csv")
  assert _contains_call_named(all_nodes, "_build_states_csv")

  # Every `if` whose TEST mentions no_video / args.no_video: walk its body
  # + orelse (NOT its test) and assert neither the "real_states.csv"
  # literal nor a _build_states_csv() call is nested inside.
  checked_any = False
  for node in ast.walk(tree):
    if not isinstance(node, ast.If):
      continue
    test_src = ast.get_source_segment(src, node.test) or ""
    if "no_video" not in test_src:
      continue
    checked_any = True
    sub_nodes = list(_walk_all(node.body + node.orelse))
    assert not _contains_string_literal(sub_nodes, "real_states.csv"), (
        f'"real_states.csv" literal found nested inside `if {test_src}:`')
    assert not _contains_call_named(sub_nodes, "_build_states_csv"), (
        f"_build_states_csv() call found nested inside `if {test_src}:`")

  # At least one such `if` must exist -- otherwise the two asserts above
  # never actually ran, and this test would be silently vacuous.
  assert checked_any, "no `if ... no_video ...:` block found to check"
