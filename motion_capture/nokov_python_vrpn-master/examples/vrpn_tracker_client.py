"""Minimal raw-`vrpn`-module tracker client for the Nokov VRPN broadcast.

This is the VRPN analog of nokov_python_sdk-master/examples/Nokov_SDK_Client.py:
it uses the hand-coded `vrpn` Python module directly (no project wrappers) to
subscribe to one or more named trackers and print every pose report.

The project's reusable, thread-safe wrapper lives in
  motion_capture/mymocap/vrpn_dependencies.py  (VRPNRigidBodyReader)
— prefer that for the robot loop. This file is the bare-bones reference.

Usage:
  python vrpn_tracker_client.py 10.1.1.198 PUBeam_11 CubeInCube_Marker1
  python vrpn_tracker_client.py 10.1.1.198          # no names -> hint to use --list

Discover the names the server publishes with the helper binary:
  ../bin/discover_senders 10.1.1.198
"""

import sys
import time
from pathlib import Path

# Make the vendored `vrpn` binding importable when run from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dist"))
import vrpn  # noqa: E402


def make_callback(name):
    def callback(userdata, data):
        p = data["position"]            # (x, y, z) meters
        q = data["quaternion"]          # VRPN order (x, y, z, w)
        print(f"{name:24s} pos(m) {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f}"
              f"  quat(xyzw) {q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {q[3]:6.3f}")
    return callback


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: vrpn_tracker_client.py <server-ip> [name ...]")
    host = sys.argv[1]
    names = sys.argv[2:]
    if not names:
        sys.exit("No tracker names given. List them with: "
                 f"../bin/discover_senders {host}")

    trackers = []
    for name in names:
        t = vrpn.receiver.Tracker(f"{name}@{host}")
        t.register_change_handler(name, make_callback(name), "position")
        trackers.append(t)
    print(f"Streaming {len(trackers)} tracker(s) from {host} "
          f"(Ctrl-C to stop) ...")

    try:
        while True:
            for t in trackers:
                t.mainloop()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
