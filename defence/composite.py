#!/usr/bin/env python3
"""Compose a real-robot video + sim video into a side-by-side clip.

Both clips are resampled onto a shared 30 fps timeline anchored to the
control loop's wall-clock t0 (see <run-dir>/manifest.json), then stacked
side by side for a thesis-defence slide.

Alignment rule (the whole point): align on shared wall-clock t0, never on
frame counts. manifest.json's video.frame_index_csv gives, for every real
webcam frame, t_rel = t_unix - t0_unix (negative for pre-roll frames
recorded before the control loop started). Sim frame k sits at exactly
k * ctrl_dt. So the real clip is shifted (via -ss / -itsoffset) so its
t_rel == 0 frame lines up with sim frame 0.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _resolve(path_str, base):
  """Resolve `path_str` against `base` unless it is already absolute."""
  p = Path(path_str)
  return p if p.is_absolute() else (base / p)


def _find_font():
  for candidate in _FONT_CANDIDATES:
    if Path(candidate).is_file():
      return candidate
  return None


def _read_real_bounds(csv_path):
  """Return (first_t_rel, last_t_rel) from the webcam frame-index CSV."""
  with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise ValueError(f"{csv_path} has no rows")
  return float(rows[0]["t_rel"]), float(rows[-1]["t_rel"])


def _drawtext(label, font):
  return (
      f"drawtext=fontfile='{font}':text='{label}':x=10:y=10:"
      "fontsize=36:fontcolor=white:box=1:boxcolor=black@0.5"
  )


def build_command(args):
  """Resolve inputs/alignment and build the ffmpeg argv list."""
  run_dir = _resolve(args.run_dir, _SCRIPT_DIR)
  manifest = json.loads((run_dir / "manifest.json").read_text())

  t0_unix = manifest["t0_unix"]
  ctrl_dt = manifest["control"]["ctrl_dt"]
  episode_length = manifest["control"]["episode_length"]

  # Must come BEFORE any _resolve of a video.* field: an external-camera run
  # nulls real_mp4 as well as frame_index_csv, so resolving those first would
  # die in pathlib (Path(None) -> TypeError) before ever reaching this guard.
  if manifest["video"].get("frame_index_csv") is None:
    sys.exit(
        f"[composite] run source={manifest['video'].get('source')!r} has no "
        "frame index CSV -- no webcam recorded this run, so automatic t0 "
        "alignment is impossible.\n"
        "[composite] cut the footage BY HAND against sim.mp4 instead. "
        "Rule: sim frame 0 is t0 = the arm's FIRST MOVEMENT (the arm is "
        "completely still before that)."
    )

  sim_path = (
      _resolve(args.sim, _SCRIPT_DIR) if args.sim else run_dir / "sim.mp4"
  )
  real_path = (
      _resolve(args.real, _SCRIPT_DIR)
      if args.real
      else _resolve(manifest["video"]["real_mp4"], run_dir)
  )
  out_path = (
      _resolve(args.out, _SCRIPT_DIR)
      if args.out
      else run_dir / "side_by_side.mp4"
  )

  frame_csv_path = _resolve(manifest["video"]["frame_index_csv"], run_dir)

  first_t_rel, last_t_rel = _read_real_bounds(frame_csv_path)

  # Shift the real clip so its t_rel == 0 frame lines up with sim frame 0.
  real_ss = None
  real_itsoffset = None
  if first_t_rel < 0:
    real_ss = -first_t_rel  # trim the pre-roll recorded before t0.
  elif first_t_rel > 0:
    real_itsoffset = first_t_rel  # webcam started late; delay it instead.

  sim_duration = episode_length * ctrl_dt
  real_duration = last_t_rel - max(first_t_rel, 0.0)
  dropped = abs(sim_duration - real_duration)
  shorter = "sim" if sim_duration < real_duration else "real"
  print(
      f"[composite] t0_unix={t0_unix} ctrl_dt={ctrl_dt} "
      f"episode_length={episode_length}"
  )
  print(
      f"[composite] sim duration={sim_duration:.3f}s, aligned real "
      f"duration={real_duration:.3f}s -> output ends at the shorter "
      f"clip ({shorter}), dropping {dropped:.3f}s from the longer one."
  )

  height = args.height if args.height % 2 == 0 else args.height + 1
  font = _find_font()
  if font:
    real_chain = f"fps={args.fps},scale=-2:{height},{_drawtext('REAL', font)}"
    sim_chain = f"fps={args.fps},scale=-2:{height},{_drawtext('SIM', font)}"
  else:
    print("[composite] no usable font found, skipping REAL/SIM labels")
    real_chain = f"fps={args.fps},scale=-2:{height}"
    sim_chain = f"fps={args.fps},scale=-2:{height}"

  filter_complex = (
      f"[0:v]{real_chain}[realv];"
      f"[1:v]{sim_chain}[simv];"
      "[realv][simv]hstack=inputs=2:shortest=1[stacked];"
      "[stacked]pad=ceil(iw/2)*2:ceil(ih/2)*2[outv]"
  )

  cmd = ["ffmpeg", "-y"]
  if real_ss is not None:
    cmd += ["-ss", f"{real_ss:.6f}"]
  if real_itsoffset is not None:
    cmd += ["-itsoffset", f"{real_itsoffset:.6f}"]
  cmd += ["-i", str(real_path)]
  cmd += ["-i", str(sim_path)]
  cmd += [
      "-filter_complex",
      filter_complex,
      "-map",
      "[outv]",
      "-an",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      str(out_path),
  ]
  return cmd, out_path


def main():
  parser = argparse.ArgumentParser(
      description="Composite a real + sim episode video side by side."
  )
  parser.add_argument("--run-dir", required=True)
  parser.add_argument("--sim", default=None)
  parser.add_argument("--real", default=None)
  parser.add_argument("--out", default=None)
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument("--height", type=int, default=1080)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  cmd, out_path = build_command(args)
  print("[composite] " + " ".join(cmd))

  if args.dry_run:
    return

  if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg binary not found on PATH")

  subprocess.run(cmd, check=True)
  print(f"[composite] wrote {out_path}")


if __name__ == "__main__":
  main()
