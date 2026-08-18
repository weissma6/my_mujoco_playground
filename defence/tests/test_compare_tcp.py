"""Tests for defence/compare_tcp.py.

compare_tcp.py is pure pandas/matplotlib/imageio -- unlike
record_real_rollout.py it MUST import and run on this machine, so these are
real round-trip tests, not source inspection.

HARD RULES (see conftest.py): no MuJoCo rendering, no mj_step, no MJX.
compare_tcp touches none of those; it only reads CSVs and draws.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import compare_tcp
from render_sim_rollout import CORE_COLUMNS

# A constant analytic offset with an exactly representable norm:
# sqrt(3^2 + 4^2 + 12^2) mm = 13 mm.
_OFFSET = np.array([0.003, -0.004, 0.012])
_OFFSET_NORM = 0.013


def _states_frame(n_rows, tcp):
  """An n_rows x CORE_COLUMNS frame; everything zero but step/t_rel/tcp_*."""
  df = pd.DataFrame(0.0, index=range(n_rows), columns=list(CORE_COLUMNS))
  df["step"] = np.arange(n_rows, dtype=int)
  df["t_rel"] = np.arange(n_rows, dtype=float) * 0.02
  df["tcp_x"], df["tcp_y"], df["tcp_z"] = tcp[:, 0], tcp[:, 1], tcp[:, 2]
  return df


def _make_run(tmp_path, n_real=400, n_sim=None, episode_length=400):
  """A run directory with a minimal manifest and two state CSVs."""
  n_sim = n_real if n_sim is None else n_sim
  run_dir = tmp_path / "run"
  run_dir.mkdir(exist_ok=True)

  t = np.linspace(0.0, 1.0, max(n_real, n_sim))
  base = np.stack([0.25 + 0.10 * t, 0.05 + 0.15 * t, 0.32 - 0.15 * t], axis=1)

  real = base[:n_real]
  sim = base[:n_sim] + _OFFSET

  _states_frame(n_real, real).to_csv(run_dir / "real_states.csv", index=False)
  _states_frame(n_sim, sim).to_csv(run_dir / "sim_states.csv", index=False)

  manifest = {
      "control": {"ctrl_dt": 0.02, "episode_length": episode_length},
      "init": {"target_pos": [0.212, 0.212, 0.165]},
  }
  (run_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")
  return run_dir


def test_compare_run_writes_mp4_and_png_and_reports_exact_offset(tmp_path):
  """Round trip: both artifacts land, and the reported divergence is exact.

  --max-frames caps only the video; the reported numbers must still describe
  the whole 400-step run, so a capped test and a full run agree.
  """
  run_dir = _make_run(tmp_path)
  stats = compare_tcp.compare_run(str(run_dir), max_frames=10, quiet=True)

  mp4 = run_dir / "tcp_compare.mp4"
  png = run_dir / "tcp_compare.png"
  assert mp4.exists() and mp4.stat().st_size > 0, "no/empty tcp_compare.mp4"
  assert png.exists() and png.stat().st_size > 0, "no/empty tcp_compare.png"

  assert stats["n_frames"] == 10, stats["n_frames"]
  assert stats["n_rows"] == 400, stats["n_rows"]
  assert stats["fps"] == 50, stats["fps"]
  assert stats["frame_matched"] is True

  # The offset is constant, so all three statistics are the same number.
  for key in ("mean_dev_m", "max_dev_m", "final_dev_m"):
    assert abs(stats[key] - _OFFSET_NORM) < 1e-9, (key, stats[key])


def test_compare_run_honours_explicit_out_path(tmp_path):
  run_dir = _make_run(tmp_path)
  out = tmp_path / "elsewhere" / "cmp.mp4"
  stats = compare_tcp.compare_run(
      str(run_dir), out_path=str(out), max_frames=3, quiet=True)
  assert out.exists() and out.stat().st_size > 0
  assert (tmp_path / "elsewhere" / "cmp.png").exists()
  assert stats["out_path"] == str(out)


def test_compare_run_rejects_length_mismatch(tmp_path):
  """399 sim rows against 400 real rows must abort, not silently truncate."""
  run_dir = _make_run(tmp_path, n_real=400, n_sim=399)
  with pytest.raises(SystemExit) as exc:
    compare_tcp.compare_run(str(run_dir), max_frames=5, quiet=True)
  msg = str(exc.value)
  assert "400" in msg and "399" in msg, (
      f"the mismatch message must name BOTH counts, got: {msg}")
  assert not (run_dir / "tcp_compare.mp4").exists(), (
      "aborted on mismatch but still wrote a video")


def test_compare_run_warns_but_proceeds_off_400(tmp_path, capsys):
  """A non-400 run still compares; only the parity claim is withdrawn."""
  run_dir = _make_run(tmp_path, n_real=153, episode_length=153)
  stats = compare_tcp.compare_run(str(run_dir), max_frames=4, quiet=True)
  assert stats["frame_matched"] is False
  assert "[warn]" in capsys.readouterr().out
  assert (run_dir / "tcp_compare.mp4").exists()


def test_compare_run_fails_on_missing_sim_states(tmp_path):
  run_dir = _make_run(tmp_path, n_real=10, episode_length=10)
  os.remove(run_dir / "sim_states.csv")
  with pytest.raises(SystemExit) as exc:
    compare_tcp.compare_run(str(run_dir), max_frames=2, quiet=True)
  assert "sim_states.csv" in str(exc.value)


def test_draw_frame_does_not_accumulate_figure_text(tmp_path):
  """The footer is ONE artist, mutated -- not a new fig.text() per frame.

  Regression guard for a real bug: fig.text() appends a fresh Text artist on
  every call, so N frames stacked N overlapping footers. The static words
  stayed legible (identical every frame) but every digit position filled in
  solid black, and the figure grew without bound over a 400-frame run.
  """
  real = np.zeros((5, 3))
  sim = real + _OFFSET
  target = np.array([0.212, 0.212, 0.165])
  lims = compare_tcp._axis_limits(real, sim, target)

  fig = plt.figure(figsize=(4, 3), dpi=50)
  try:
    gs = fig.add_gridspec(3, 2)
    axes3d = fig.add_subplot(gs[:, 0], projection="3d")
    axes1d = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    footer = fig.text(0.5, 0.015, "", ha="center")

    def draw(k):
      compare_tcp._draw_frame(fig, axes3d, axes1d, footer, real, sim, target,
                              k, lims, 0.0, 5, "unit-test")

    # Baseline AFTER frame 0: fig.suptitle() legitimately creates its single
    # Text on first call and reuses it thereafter, so counting from before
    # frame 0 would charge that one-off to the loop. What must be flat is the
    # PER-FRAME growth.
    draw(0)
    before = len(fig.texts)
    for k in range(1, 5):
      draw(k)
    assert len(fig.texts) == before, (
        f"figure text artists grew {before} -> {len(fig.texts)} over 4 "
        "further frames; the footer is being re-created instead of updated")
    assert footer.get_text(), "the footer was never populated"
  finally:
    plt.close(fig)
