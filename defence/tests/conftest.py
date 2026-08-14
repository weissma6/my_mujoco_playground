"""Shared fixtures for defence/tests.

Scope/ownership: this file and test_defence.py are the ONLY files this test
suite owns (see the task contract). Everything else under defence/ and
evaluation/ is read-only from here.

HARD RULES this conftest itself must respect (same as the contract given to
the test-suite author):
  * NEVER import defence/record_real_rollout.py (Linux-only vrpn.so + missing
    cv2 on this machine) -- it is inspected at source level only, from
    test_defence.py, via py_compile/ast.
  * NEVER render: no mujoco.Renderer, no offscreen GL, no rollout, no
    mj_step. Fixtures here build a plain MjModel/MjData via
    mujoco.MjModel.from_xml_path + mujoco.mj_forward and stop there.
"""

import os
import sys

import mujoco
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFENCE_DIR = os.path.dirname(_THIS_DIR)
REPO_ROOT = os.path.dirname(_DEFENCE_DIR)
_EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")

# sys.path wiring: defence/ (for render_sim_rollout.py, composite.py) and
# evaluation/ (for gap_metrics.py) as plain top-level modules -- no
# __init__.py anywhere in either dir, so this is the whole story.
for _p in (_DEFENCE_DIR, _EVALUATION_DIR):
  if _p not in sys.path:
    sys.path.insert(0, _p)

# Importing render_sim_rollout pulls in jax/brax (a few seconds) -- done once
# here at module scope, not per test. composite/gap_metrics are cheap.
import render_sim_rollout  # noqa: E402
import composite  # noqa: E402
import gap_metrics  # noqa: E402

XML_PATH = os.path.join(
    REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "xmls",
    "mjx_single_cube_position_ur3.xml",
)


class FakeScene:
  """Minimal stand-in for a mujoco.MjvScene.

  A plain-python object carrying `ngeom` (int), `maxgeom` (int), and `geoms`
  -- a LIST OF REAL mujoco.MjvGeom() objects, preallocated to `maxgeom`
  entries (mirroring the real scene's fixed-size geoms array), so
  mjv_initGeom and direct field assignment on scene.geoms[i] behave exactly
  as they would against a real MjvScene.
  """

  def __init__(self, maxgeom=64):
    self.maxgeom = int(maxgeom)
    self.ngeom = 0
    self.geoms = [mujoco.MjvGeom() for _ in range(self.maxgeom)]


@pytest.fixture(scope="session")
def model():
  """Session-scoped MjModel -- loading is the expensive part, never mutated."""
  return mujoco.MjModel.from_xml_path(XML_PATH)


@pytest.fixture(scope="session")
def ids(model):
  """Resolved name -> id/address lookups, reusing render_sim_rollout's own
  resolver (not reimplemented here)."""
  return render_sim_rollout.resolve_ids(model)


@pytest.fixture
def data(model):
  """A fresh MjData at the model's default pose, forwarded once.

  Function-scoped on purpose: each test gets its own MjData instance, so a
  test that mutates mocap_pos/mocap_quat (the "rides with the table" checks
  in test C) can never leak state into another test -- there is nothing
  shared to restore.
  """
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  return d


@pytest.fixture
def fake_scene_factory():
  """Factory fixture: fake_scene_factory(maxgeom=64) -> a fresh FakeScene."""

  def _make(maxgeom=64):
    return FakeScene(maxgeom=maxgeom)

  return _make
