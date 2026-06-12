"""VRPN scene-inventory streamer: discover + stream ALL points over VRPN.

The VRPN analog of the old Nokov-SDK connection test. Instead of the Nokov
SDK, this goes through the project's VRPN pipeline
(`motion_capture/mymocap/vrpn_dependencies.py` -> `VRPNRigidBodyReader`), which
the Nokov (XINGYING) server feeds by broadcasting every rigid body and marker
as a VRPN tracker on port 3883.

It:
  1. discovers every sender (tracker) the server publishes,
  2. subscribes to ALL of them (the reader's `names=None` "stream all" mode),
  3. streams for a few seconds, printing each point that reports per tick.

Run this first to verify the VRPN pipeline end-to-end before wiring it into the
robot loop.

Usage:
  python motion_capture/mymocap/test_mocap_connection.py
  python motion_capture/mymocap/test_mocap_connection.py --server-ip 10.1.1.198
  python motion_capture/mymocap/test_mocap_connection.py --duration 20
  python motion_capture/mymocap/test_mocap_connection.py --list   # names, then exit
"""

import argparse
import sys
import time
from pathlib import Path

# Reuse the project's VRPN pipeline (DRY): same reader the pick loop uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from motion_capture.mymocap.vrpn_dependencies import (  # noqa: E402
    DEFAULT_PORT,
    DEFAULT_SERVER_IP,
    VRPNRigidBodyReader,
    discover_senders,
)


def main():
  ap = argparse.ArgumentParser(
      description="Discover + stream ALL VRPN points from the Nokov server."
  )
  ap.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
  ap.add_argument("--port", type=int, default=DEFAULT_PORT)
  ap.add_argument("--duration", type=float, default=10.0,
                  help="seconds to stream (default 10)")
  ap.add_argument("--list", action="store_true",
                  help="just discover + print sender names, then exit")
  args = ap.parse_args()

  # 1. Discover every tracker the server publishes (rigid bodies AND markers).
  print(f"=== discovering VRPN senders on {args.server_ip}:{args.port} ===")
  senders = discover_senders(args.server_ip, args.port)
  if not senders:
    sys.exit(
        "FAIL: server publishes no VRPN senders. Check that XINGYING has "
        "SDK/VRPN broadcast enabled, a rigid body/markerset is defined and "
        "visible, and this host is on the mocap subnet."
    )
  for nm in sorted(senders):
    print(f"  {nm}")
  print(f"({len(senders)} sender(s))")

  if args.list:
    return

  # 2. Subscribe to ALL discovered senders (names=None => stream all).
  reader = VRPNRigidBodyReader(args.server_ip, names=None, port=args.port)
  if not reader.start(timeout=8.0):
    sys.exit("FAIL: could not subscribe to the VRPN server")

  # 3. Wait for the first report, then stream every reporting point per tick.
  if not reader.wait_for_data(timeout=5.0):
    print("WARN: subscribed but no reports yet — are the bodies visible to "
          "the cameras? Streaming anyway (poses appear as they arrive).")

  print(f"\n=== streaming ALL {len(reader._names)} tracker(s) for "
        f"~{args.duration:.0f} s ===")
  t_end = time.time() + args.duration
  try:
    while time.time() < t_end:
      bodies = reader.get_body()  # {name: {"xyz": np(3), "quat": np(4)}}
      stamp = f"[{time.time() - (t_end - args.duration):5.1f}s]"
      print(f"{stamp} reporting {len(bodies)}/{len(reader._names)}:")
      for nm in sorted(bodies):
        p = bodies[nm]["xyz"]
        q = bodies[nm]["quat"]
        print(f"    {nm:28s} xyz(m) {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f}"
              f"  quat(wxyz) {q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {q[3]:6.3f}")
      time.sleep(0.5)
  except KeyboardInterrupt:
    pass
  finally:
    reader.stop()

  stats = reader.get_stats()
  print(f"\nDone. {stats['num_reporting']}/{stats['num_subscribed']} "
        f"tracker(s) reported, {stats['frame_count']} total updates.")


if __name__ == "__main__":
  main()
