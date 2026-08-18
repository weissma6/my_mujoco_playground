"""Animated real-vs-sim TCP trajectory comparison for one defence run.

Companion to `defence/record_real_rollout.py` (the real take) and
`defence/render_sim_rollout.py` (the sim replay). This script reads ONLY the
three files those two leave in a run directory -- `manifest.json`,
`real_states.csv`, `sim_states.csv` -- and writes:

    <run-dir>/tcp_compare.mp4   the animation, one frame per control step
    <run-dir>/tcp_compare.png   the final frame, which is what goes in a paper

Layout, per frame:
  * LEFT   a 3D view. Both TCP paths draw themselves out to the current step
           (real solid, sim dashed) with a marker at each current position.
           Axis limits are computed ONCE from the union of both full
           trajectories, so the view never jumps mid-video.
  * RIGHT  three stacked panels -- tcp_x, tcp_y, tcp_z against step. The full
           curves are drawn from frame 0 with a vertical cursor at the
           current step, so the reader can see where the run is going.
  * FOOT   current step, current per-axis error, and the running max ||d||.

HARD RULES (mirroring render_sim_rollout.py's posture):
  * NEVER import record_real_rollout.py -- it drives real hardware and pulls
    in a Linux-only vrpn.so plus cv2. Nothing here needs it.
  * NEVER touch MuJoCo or MJX. This file renders no scene; it is pure
    pandas/matplotlib/imageio and MUST import cleanly on the Mac.
  * Do NOT set MUJOCO_GL. There is no MuJoCo rendering here, and assigning it
    would only break the *other* scripts' backend selection.
  * A real/sim length mismatch is a HARD FAILURE. Silently truncating to the
    shorter of the two would hide exactly the frame-matching bug this
    comparison exists to detect.

Usage:
    python defence/compare_tcp.py --run-dir defence/runs/<run>
    python defence/compare_tcp.py --run-dir defence/runs/<run> --max-frames 20
"""

import argparse
import json
import os
import sys

import matplotlib
# Agg before pyplot: this runs headless on the robotics PC and in CI, and
# there is no interactive display to attach to anywhere it is used.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402 -- registers '3d'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

# The three TCP columns, in the order CORE_COLUMNS declares them. Shared by
# both CSVs by construction -- record_real_rollout.py asserts its own copy of
# CORE_COLUMNS in _build_states_csv, render_sim_rollout.py writes the tuple
# directly. Deliberately NOT imported from either (the first is a hardware
# module); the pair is guarded by the tests instead.
TCP_COLUMNS = ("tcp_x", "tcp_y", "tcp_z")

# The trained horizon. A run of any other length still compares fine -- the
# geometry is unaffected -- but it is not frame-matched to training, so the
# parity claim does not hold and the video says so.
TRAINED_EPISODE_LENGTH = 400

# The pinned D22 drop square centre, mirrored from
# evaluation/gap_metrics.py:963 (and record_real_rollout.DROP_SQUARE_TARGET /
# render_sim_rollout.DROP_SQUARE_CENTER). Drawn once in the 3D view as the
# point both trajectories are aiming at. Only ever used when the manifest
# does not carry its own init.target_pos.
DROP_SQUARE_TARGET = (0.212, 0.212, 0.165)

_REAL_STYLE = dict(color="#1f77b4", linestyle="-", linewidth=1.8)
_SIM_STYLE = dict(color="#d62728", linestyle="--", linewidth=1.6)

# 1280x720 at dpi 100. Both divisible by 16, so imageio's macro_block_size=16
# never has to silently resize the frames.
_FIG_W_IN, _FIG_H_IN, _DPI = 12.8, 7.2, 100


def _fail(msg):
  raise SystemExit(f"compare_tcp: {msg}")


def _read_run(run_dir):
  """manifest + the two state CSVs, with every absence named explicitly."""
  manifest_path = os.path.join(run_dir, "manifest.json")
  real_path = os.path.join(run_dir, "real_states.csv")
  sim_path = os.path.join(run_dir, "sim_states.csv")

  if not os.path.isdir(run_dir):
    _fail(f"--run-dir {run_dir} is not a directory")
  for path, what in ((manifest_path, "manifest.json"),
                     (real_path, "real_states.csv"),
                     (sim_path, "sim_states.csv")):
    if not os.path.exists(path):
      _fail(f"{what} missing from {run_dir}. "
            + ("Run render_sim_rollout.py first."
               if what == "sim_states.csv" else ""))

  with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
  return manifest, pd.read_csv(real_path), pd.read_csv(sim_path)


def _tcp(df, which):
  missing = [c for c in TCP_COLUMNS if c not in df.columns]
  if missing:
    _fail(f"{which}_states.csv is missing column(s) {missing}")
  return df.loc[:, list(TCP_COLUMNS)].to_numpy(dtype=float)


def _fps_from_manifest(manifest):
  """fps is 1/ctrl_dt from the manifest -- never hardcoded."""
  try:
    ctrl_dt = float(manifest["control"]["ctrl_dt"])
  except (KeyError, TypeError, ValueError):
    _fail("manifest.json has no numeric control.ctrl_dt")
  if not ctrl_dt > 0:
    _fail(f"manifest control.ctrl_dt = {ctrl_dt}, must be positive")
  return int(round(1.0 / ctrl_dt)), ctrl_dt


def _target_from_manifest(manifest):
  target = (manifest.get("init") or {}).get("target_pos")
  if target is None:
    return np.asarray(DROP_SQUARE_TARGET, dtype=float)
  return np.asarray(target, dtype=float).reshape(3)


def _axis_limits(real, sim, target, pad=0.02):
  """One set of limits from the union of BOTH full paths, computed once.

  Equal spans on all three axes so the 3D view is not visually distorted --
  a 5 cm wobble in Z must not look like a 20 cm one because Z happens to
  have the narrowest range.
  """
  pts = np.vstack([real, sim, target.reshape(1, 3)])
  lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
  span = float((hi - lo).max())
  mid = (hi + lo) / 2.0
  return mid - span / 2.0, mid + span / 2.0


def _draw_frame(fig, axes3d, axes1d, footer, real, sim, target, k, lims,
                max_norm_so_far, episode_length, run_name):
  """Redraw every axis for step index `k` (0-based).

  `footer` is a Text artist created ONCE by the caller and mutated here.
  fig.text() would append a brand-new artist on every call, so 400 frames
  would stack 400 overlapping footers -- the static words stay legible but
  every digit position fills in solid black, and the figure grows unboundedly.
  fig.suptitle() is safe by contrast: it replaces the figure's single title.
  """
  lo, hi = lims
  delta = sim[k] - real[k]

  axes3d.clear()
  axes3d.plot(real[:k + 1, 0], real[:k + 1, 1], real[:k + 1, 2],
              label="real (FK TCP)", **_REAL_STYLE)
  axes3d.plot(sim[:k + 1, 0], sim[:k + 1, 1], sim[:k + 1, 2],
              label="sim (tcp site)", **_SIM_STYLE)
  axes3d.scatter(*real[k], s=42, color=_REAL_STYLE["color"], depthshade=False)
  axes3d.scatter(*sim[k], s=42, color=_SIM_STYLE["color"], depthshade=False,
                 marker="^")
  axes3d.scatter(*target, s=70, color="#2ca02c", marker="*",
                 depthshade=False, label="drop target")
  axes3d.set_xlim(lo[0], hi[0])
  axes3d.set_ylim(lo[1], hi[1])
  axes3d.set_zlim(lo[2], hi[2])
  axes3d.set_xlabel("x (m)")
  axes3d.set_ylabel("y (m)")
  axes3d.set_zlabel("z (m)")
  axes3d.set_title("TCP path, base frame")
  axes3d.legend(loc="upper left", fontsize=8)

  steps = np.arange(len(real))
  for ax, col, name in zip(axes1d, range(3), TCP_COLUMNS):
    ax.clear()
    ax.plot(steps, real[:, col], **_REAL_STYLE)
    ax.plot(steps, sim[:, col], **_SIM_STYLE)
    ax.axvline(k, color="0.35", linewidth=1.0)
    ax.set_ylabel(f"{name} (m)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(len(real) - 1, 1))
  axes1d[-1].set_xlabel("control step")

  fig.suptitle(
      f"{run_name}   step {k}/{len(real) - 1}   "
      f"episode_length {episode_length}",
      fontsize=11)
  footer.set_text(
      f"error (sim - real)  dx {delta[0] * 1000:+7.2f} mm   "
      f"dy {delta[1] * 1000:+7.2f} mm   dz {delta[2] * 1000:+7.2f} mm     "
      f"|d| {np.linalg.norm(delta) * 1000:6.2f} mm     "
      f"running max |d| {max_norm_so_far * 1000:6.2f} mm")


def _frame_rgb(fig):
  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba())
  return buf[:, :, :3].copy()


def compare_run(run_dir, out_path=None, max_frames=None, quiet=False):
  """Write tcp_compare.mp4 + .png for `run_dir`; return the divergence stats.

  Returns a dict with mean/max/final ||sim - real|| over the WHOLE trajectory
  (not merely the frames rendered), plus the frame counts. --max-frames caps
  only how much video is written; the reported numbers always describe the
  full run, so a capped test run and a full run report the same divergence.
  """
  manifest, real_df, sim_df = _read_run(run_dir)

  n_real, n_sim = len(real_df), len(sim_df)
  if n_real != n_sim:
    _fail(
        f"length mismatch: real_states.csv has {n_real} rows, "
        f"sim_states.csv has {n_sim}. Refusing to truncate -- a frame-matched "
        "comparison is the entire point, and quietly clipping to "
        f"{min(n_real, n_sim)} rows would hide the bug. Re-render the sim "
        "against this manifest, or re-take the real episode.")
  if n_real == 0:
    _fail("real_states.csv and sim_states.csv are both empty")

  real, sim = _tcp(real_df, "real"), _tcp(sim_df, "sim")
  target = _target_from_manifest(manifest)
  fps, _ctrl_dt = _fps_from_manifest(manifest)
  episode_length = int((manifest.get("control") or {}).get(
      "episode_length", n_real))

  # Warn, do not fail: the geometry is still comparable at any length, but
  # the frame-matched-to-training claim is not.
  if n_real != TRAINED_EPISODE_LENGTH:
    print(f"[warn] this run is {n_real} steps, not the trained "
          f"{TRAINED_EPISODE_LENGTH} — the comparison is still valid but is "
          "NOT frame-matched to training.")

  dev = np.linalg.norm(sim - real, axis=1)
  stats = {
      "n_rows": int(n_real),
      "fps": int(fps),
      "mean_dev_m": float(dev.mean()),
      "max_dev_m": float(dev.max()),
      "final_dev_m": float(dev[-1]),
      "frame_matched": bool(n_real == TRAINED_EPISODE_LENGTH),
  }

  if out_path is None:
    out_path = os.path.join(run_dir, "tcp_compare.mp4")
  png_path = os.path.splitext(out_path)[0] + ".png"

  n_frames = n_real if max_frames is None else min(int(max_frames), n_real)
  if n_frames <= 0:
    _fail(f"--max-frames {max_frames} leaves nothing to render")

  lims = _axis_limits(real, sim, target)
  run_name = os.path.basename(os.path.normpath(run_dir))

  fig = plt.figure(figsize=(_FIG_W_IN, _FIG_H_IN), dpi=_DPI)
  gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0],
                        left=0.05, right=0.97, top=0.90, bottom=0.09,
                        hspace=0.28, wspace=0.22)
  axes3d = fig.add_subplot(gs[:, 0], projection="3d")
  axes1d = [fig.add_subplot(gs[i, 1]) for i in range(3)]
  # Created once, mutated per frame -- see _draw_frame's docstring.
  footer = fig.text(0.5, 0.015, "", ha="center", fontsize=10,
                    family="monospace")

  os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
              exist_ok=True)
  writer = imageio.get_writer(
      out_path,
      fps=fps,
      codec="libx264",
      pixelformat="yuv420p",
      macro_block_size=16,
      ffmpeg_params=["-profile:v", "high", "-level", "4.0", "-crf", "18"],
  )
  try:
    for k in range(n_frames):
      _draw_frame(fig, axes3d, axes1d, footer, real, sim, target, k, lims,
                  float(dev[:k + 1].max()), episode_length, run_name)
      writer.append_data(_frame_rgb(fig))
    fig.savefig(png_path, dpi=_DPI)
  finally:
    writer.close()
    plt.close(fig)

  stats["n_frames"] = int(n_frames)
  stats["out_path"] = out_path
  stats["png_path"] = png_path
  if not quiet:
    print(f"wrote {n_frames} frames @ {fps} fps -> {out_path}")
    print(f"wrote final frame -> {png_path}")
    print(f"TCP divergence  mean {stats['mean_dev_m']:.4f} m  "
          f"max {stats['max_dev_m']:.4f} m  "
          f"final {stats['final_dev_m']:.4f} m")
  return stats


def build_parser():
  ap = argparse.ArgumentParser(
      description="Animated real-vs-sim TCP trajectory comparison.")
  ap.add_argument("--run-dir", required=True,
                  help="run directory holding manifest.json, "
                       "real_states.csv and sim_states.csv")
  ap.add_argument("--out", default=None,
                  help="output mp4. Default <run-dir>/tcp_compare.mp4; the "
                       "still is written alongside it as .png")
  ap.add_argument("--max-frames", type=int, default=None,
                  help="render only the first N frames (tests / quick looks). "
                       "The reported divergence still covers the whole run.")
  ap.add_argument("--quiet", action="store_true")
  return ap


def main(argv=None):
  args = build_parser().parse_args(argv)
  compare_run(args.run_dir, out_path=args.out, max_frames=args.max_frames,
              quiet=args.quiet)
  return 0


if __name__ == "__main__":
  sys.exit(main())
