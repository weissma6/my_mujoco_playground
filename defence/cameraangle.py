#!/usr/bin/env python3
"""Render ONE frame previewing the camera angle `render_sim_rollout.py` uses.

Framing the defence camera by re-rendering a whole episode is wasteful: the
policy load dominates, and all you wanted to see was where the table, the
taped drop square and the P1/P2/P3 marks land in frame. This script renders a
SINGLE CPU frame instead, so tweaking `camera.json` is a ~2 s loop.

It IMPORTS `render_sim_rollout` and reuses that module's `load_model`,
`resolve_ids`, `reset_state`, `load_or_default_camera`, `build_free_camera`
and `draw_table_markers` verbatim -- none of the camera math or the marker
overlay is re-derived here, so this preview cannot drift from the real render.

Two ways to pick the pose that gets drawn:
  * `--run-dir <dir>`  reset to that run's MEASURED init (manifest.json), i.e.
    exactly what render_sim_rollout.py starts from.
  * no `--run-dir`     reset to an XML keyframe (default `task_home`), so you
    can frame the camera before any run exists.

Two reasons this PNG is not byte-identical to sim.mp4's frame 0, neither of
which affects framing:
  * render_sim_rollout.py applies one control step BEFORE writing its first
    video frame, so frame 0 is one control tick (default 20 ms) later than
    what this draws. Reproducing that exactly would mean loading the policy,
    which is the cost this script exists to avoid.
  * its writer uses macro_block_size=16, and imageio RESIZES (does not pad) up
    to the next multiple of 16 -- so a 900-px-tall request becomes a 912-px
    mp4, and the default 1080 becomes 1088, a silent +0.74% vertical stretch.
    This PNG is written at the exact requested size, so measured against
    sim.mp4 it differs by that scale factor. Verified: the drop square's
    centroid agrees to 0.34 px vertically once the rescale is undone.

Usage:
    python defence/cameraangle.py
    python defence/cameraangle.py --run-dir defence/runs/<run>
    python defence/cameraangle.py --camera-json defence/camera.json \\
        --out /tmp/preview.png
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)

# Import FIRST: render_sim_rollout's module body sets JAX_PLATFORM_NAME and
# MUJOCO_GL via os.environ.setdefault before it imports mujoco, and those only
# take effect if they are set before the GL backend is chosen. Importing it up
# front means this script inherits that setup instead of duplicating it.
import render_sim_rollout as R  # noqa: E402

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def _camera_basis(cam_dict):
  """(pos, forward, right, up) unit basis for a camera.json free camera.

  Mirrors build_free_camera's convention: forward = lookat - pos, world +Z up.
  """
  pos = np.asarray(cam_dict["pos"], dtype=float)
  lookat = np.asarray(cam_dict["lookat"], dtype=float)
  fwd = lookat - pos
  fwd = fwd / np.linalg.norm(fwd)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
  right = right / np.linalg.norm(right)
  up = np.cross(right, fwd)
  return pos, fwd, right, up


def _report_framing(cam_dict, width, height):
  """Print, for each taped feature, whether it falls inside the frame.

  This is the whole point of the preview: the drop square at (0.212, 0.212)
  and P3 at (0.495, 0.188) sit ~0.5 m apart, so a camera framed on the robot
  can easily push one of them off-screen. At the documented default
  (pos [0.85,0,0.4], lookat [0.3,0,0.1], fovy 42) nothing actually clips at
  16:9 -- but the square lands hard right (u ~ +0.26..+0.66) and P3 sits low
  (v ~ -0.55), which is a lopsided, cramped composition rather than a broken
  one. Reported in normalised device coords: |u| <= 1 and |v| <= 1 is
  on-screen, and values near +-1 are near an edge.
  """
  pos, fwd, right, up = _camera_basis(cam_dict)
  fovy = float(cam_dict["fovy"])
  tan_half_v = np.tan(np.radians(fovy / 2.0))
  aspect = float(width) / float(height)

  cx, cy = R.DROP_SQUARE_CENTER
  h = R.DROP_SQUARE_HALF_WIDTH
  z = 0.095 + R._OVERLAY_Z_GAP + R._OVERLAY_HALF_THICKNESS
  features = [
      ("square NE corner", (cx + h, cy + h, z)),
      ("square NW corner", (cx - h, cy + h, z)),
      ("square SE corner", (cx + h, cy - h, z)),
      ("square SW corner", (cx - h, cy - h, z)),
  ]
  for name, (mx, my) in sorted(R.TAPE_MARK_XY.items()):
    features.append((f"mark {name}", (mx, my, z)))

  print(f"[framing] fovy={fovy} aspect={aspect:.3f} ({width}x{height})")
  offscreen = []
  for name, p in features:
    v = np.asarray(p, dtype=float) - pos
    depth = float(v @ fwd)
    if depth <= 1e-9:
      print(f"  {name:18s} BEHIND CAMERA")
      offscreen.append(name)
      continue
    u = float(v @ right) / (depth * tan_half_v * aspect)
    w = float(v @ up) / (depth * tan_half_v)
    inside = abs(u) <= 1.0 and abs(w) <= 1.0
    px = (u + 1.0) * 0.5 * width
    py = (1.0 - w) * 0.5 * height
    flag = "ok " if inside else "OFF"
    print(f"  {name:18s} {flag} ndc=({u:+.2f},{w:+.2f}) px=({px:7.1f},{py:6.1f})")
    if not inside:
      offscreen.append(name)

  if offscreen:
    print(
        f"[framing] {len(offscreen)} feature(s) OUT OF FRAME: "
        f"{', '.join(offscreen)}. Widen 'fovy' or move 'pos'/'lookat' in "
        "camera.json and re-run -- this is a one-frame render, it is cheap."
    )
  else:
    print("[framing] all taped features are inside the frame.")


def main():
  ap = argparse.ArgumentParser(
      description="Render one preview frame of the camera angle "
                  "render_sim_rollout.py will use."
  )
  ap.add_argument("--run-dir", default=None,
                  help="defence/runs/<run>/ with a manifest.json; reset to "
                       "that run's measured init. Omit to use --keyframe.")
  ap.add_argument("--keyframe", default="task_home",
                  help="XML keyframe used when --run-dir is omitted "
                       "(task_home, low_home, tucked).")
  ap.add_argument("--camera", default=None,
                  help="name of an existing XML camera (e.g. 'side_130') -- "
                       "overrides --camera-json.")
  ap.add_argument("--camera-json", default=R.DEFAULT_CAMERA_JSON)
  ap.add_argument("--xml-path", default=R.DEFAULT_XML_PATH,
                  help="only used when --run-dir is omitted; with --run-dir "
                       "the manifest's env.xml_path wins.")
  ap.add_argument("--width", type=int, default=1600)
  ap.add_argument("--height", type=int, default=900)
  ap.add_argument("--no-markers", dest="markers", action="store_false",
                  help="do not draw the taped drop square / P1-P3 marks.")
  ap.set_defaults(markers=True)
  ap.add_argument("--out", default=None,
                  help="default: <run-dir>/camera_preview.png, or "
                       "defence/camera_preview.png without --run-dir.")
  args = ap.parse_args()

  # Resolve the scene + initial state.
  if args.run_dir is not None:
    run_dir = os.path.abspath(args.run_dir)
    m = R.load_manifest(run_dir)
    xml_rel = m["env"]["xml_path"]
    xml_path = (xml_rel if os.path.isabs(xml_rel)
                else os.path.join(R.REPO_ROOT, xml_rel))
    out_path = args.out or os.path.join(run_dir, "camera_preview.png")
    state_desc = f"measured init from {os.path.join(run_dir, 'manifest.json')}"
  else:
    xml_path = args.xml_path
    out_path = args.out or os.path.join(_THIS_DIR, "camera_preview.png")
    state_desc = f"XML keyframe {args.keyframe!r}"

  if not os.path.exists(xml_path):
    raise SystemExit(f"xml_path does not exist: {xml_path!r}")

  model = R.load_model(xml_path, args.width, args.height)
  ids = R.resolve_ids(model)

  if args.run_dir is not None:
    data, _target_pos = R.reset_state(model, ids, m["init"])
  else:
    data = mujoco.MjData(model)
    try:
      key_id = model.keyframe(args.keyframe).id
    except KeyError:
      names = [model.keyframe(i).name for i in range(model.nkey)]
      raise SystemExit(
          f"no keyframe {args.keyframe!r} in {xml_path!r}; available: {names}"
      )
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    R._assert_cpu_physics(model, data)
    mujoco.mj_forward(model, data)

  # Camera -- identical resolution order to render_sim_rollout.main().
  cam_dict = None
  if args.camera is not None:
    camera_arg = args.camera
  else:
    cam_dict = R.load_or_default_camera(args.camera_json)
    camera_arg = R.build_free_camera(cam_dict, model)

  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  renderer.update_scene(data, camera=camera_arg)
  # mjv_updateScene rebuilt scene.ngeom, so the overlay is appended after it
  # and before render() -- the same ordering the rollout loop uses.
  if args.markers:
    R.draw_table_markers(renderer.scene, model, data, ids)
  frame = renderer.render()

  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  imageio.imwrite(out_path, frame)

  print(f"[cameraangle] state   : {state_desc}")
  if cam_dict is not None:
    print(f"[cameraangle] camera  : {args.camera_json} "
          f"pos={cam_dict['pos']} lookat={cam_dict['lookat']} "
          f"fovy={cam_dict['fovy']}")
    _report_framing(cam_dict, args.width, args.height)
  else:
    print(f"[cameraangle] camera  : XML camera {args.camera!r} "
          "(framing report is free-camera only)")
  print(f"[cameraangle] markers : {'on' if args.markers else 'off'}")
  print(f"[cameraangle] wrote {args.width}x{args.height} -> {out_path}")


if __name__ == "__main__":
  main()
