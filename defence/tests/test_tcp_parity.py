"""TCP frame parity: the real robot's FK TCP vs the sim's `tcp` site.

The real log's tcp_x/y/z comes from the robot loop's own forward kinematics
(run_policy_loop(use_fk_tcp=True) -> init_fk_model / compute_tcp_pos). The sim
log's comes from data.site_xpos[model.site("tcp").id]. If those are not the
same physical point, every number in a real-vs-sim trajectory comparison
carries a constant unexplained offset -- and it would read as a sim-to-real
gap rather than as the bookkeeping bug it actually is.

Checkable with no robot, against data already on disk.

HARD RULES (same as the rest of this suite, see conftest.py):
  * NEVER render, NEVER mj_step, NEVER MJX. mujoco.mj_forward only, via the
    session-scoped `model` fixture.
  * A missing run directory or CSV is a FAILURE, not a skip. The baseline
    being absent is the finding; skipping would hide it.
"""

import json
import os

import mujoco
import numpy as np
import pandas as pd

from conftest import REPO_ROOT

# The one archived run that carries both a manifest with the measured encoder
# pose and a real_states.csv -- resolved from the repo root, never absolute.
_RUN_DIR = os.path.join(
    REPO_ROOT, "defence", "runs", "20260814T121140Z_L1_pos")
_MANIFEST = os.path.join(_RUN_DIR, "manifest.json")
_REAL_CSV = os.path.join(_RUN_DIR, "real_states.csv")

# Agreement budget. The two TCPs are either the same point or they are not;
# 2 mm is a generous allowance for encoder quantisation and the URDF-vs-XML
# kinematic constants, not a knob to widen if the test goes red.
_TOL_M = 2e-3


def _require(path):
  assert os.path.exists(path), (
      f"{path} is missing. This is a FAILURE, not a skip: the parity claim "
      "has no baseline to stand on, which is itself the finding.")
  return path


def _load_baseline():
  with open(_require(_MANIFEST), "r", encoding="utf-8") as f:
    manifest = json.load(f)
  df = pd.read_csv(_require(_REAL_CSV))
  assert len(df) > 0, f"{_REAL_CSV} has no data rows"
  return manifest, df


def _fk_tcp(model, ids, arm_qpos, finger):
  """The sim `tcp` site position at a given arm/finger configuration."""
  data = mujoco.MjData(model)
  data.qpos[ids["arm_qposadr"]] = np.asarray(arm_qpos, dtype=float)
  data.qpos[ids["finger_qposadr"]] = float(finger)
  mujoco.mj_forward(model, data)
  return data.site_xpos[ids["tcp_site"]].copy()


def test_real_fk_tcp_matches_sim_tcp_site(model, ids, capsys):
  """The measured encoder pose put through the sim must land on the logged TCP.

  Uses init.arm_qpos / init.finger (the pose the manifest records as measured
  at reset) against row 0 of real_states.csv, which is the first logged TCP.
  """
  manifest, df = _load_baseline()
  arm_qpos = np.asarray(manifest["init"]["arm_qpos"], dtype=float)
  finger = float(manifest["init"]["finger"])
  assert arm_qpos.shape == (6,), arm_qpos.shape

  row0 = df.iloc[0]
  real_tcp = np.array([row0["tcp_x"], row0["tcp_y"], row0["tcp_z"]], float)
  sim_tcp = _fk_tcp(model, ids, arm_qpos, finger)
  delta = sim_tcp - real_tcp

  # Printed on pass as well as on failure -- this number is the WP's result.
  with capsys.disabled():
    print(f"\n  real FK TCP (row 0) : {np.round(real_tcp, 6).tolist()}")
    print(f"  sim tcp site        : {np.round(sim_tcp, 6).tolist()}")
    print(f"  offset (mm)         : {np.round(delta * 1000.0, 4).tolist()}")
    print(f"  max |offset|        : {np.abs(delta).max() * 1000.0:.4f} mm")

  assert np.abs(delta).max() < _TOL_M, (
      f"TCP frames disagree by {np.abs(delta).max() * 1000:.3f} mm "
      f"(offset {np.round(delta * 1000, 3).tolist()} mm). Do NOT widen "
      "_TOL_M: either fix the definition at the source or subtract the "
      "offset explicitly and visibly in the comparison plot.")


def test_real_fk_tcp_matches_sim_tcp_site_at_logged_joints(model, ids, capsys):
  """The same check with the pose difference removed.

  init.arm_qpos is sampled at reset, row 0's q0..q5 a control tick later, so
  the test above carries that small encoder drift. Feeding row 0's OWN joint
  angles isolates the frame question: if the two TCP definitions really are
  the same point, this must agree to floating-point noise, not merely to
  2 mm. A regression that shifts the site would fail here long before it
  reached the tolerance above.
  """
  _manifest, df = _load_baseline()
  row0 = df.iloc[0]
  arm_qpos = np.array([row0[f"q{i}"] for i in range(6)], dtype=float)
  real_tcp = np.array([row0["tcp_x"], row0["tcp_y"], row0["tcp_z"]], float)
  sim_tcp = _fk_tcp(model, ids, arm_qpos, float(row0["finger"]))
  delta = sim_tcp - real_tcp

  with capsys.disabled():
    print(f"  at logged joints    : max |offset| "
          f"{np.abs(delta).max() * 1000.0:.6f} mm")

  assert np.abs(delta).max() < 1e-6, (
      f"at row 0's own joint angles the two TCPs differ by "
      f"{np.abs(delta).max() * 1000:.6f} mm -- they are not the same site")


def test_tcp_site_is_independent_of_finger_opening(model, ids):
  """The `tcp` site rides the wrist, not the fingers.

  Load-bearing for the comparison: the real loop's compute_tcp_pos is a
  function of the six arm joints alone, so if the sim site moved with the
  finger opening the two would diverge every time the gripper actuated --
  and the divergence would look like tracking error.
  """
  _manifest, df = _load_baseline()
  row0 = df.iloc[0]
  arm_qpos = np.array([row0[f"q{i}"] for i in range(6)], dtype=float)
  open_tcp = _fk_tcp(model, ids, arm_qpos, 0.0)
  closed_tcp = _fk_tcp(model, ids, arm_qpos, 0.025)
  assert np.abs(closed_tcp - open_tcp).max() < 1e-9, (
      f"the tcp site moves by "
      f"{np.abs(closed_tcp - open_tcp).max() * 1000:.6f} mm between a fully "
      "open and a fully closed gripper -- it is not a pure wrist frame")
