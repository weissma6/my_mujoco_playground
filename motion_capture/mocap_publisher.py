"""NOKOV -> UDP publisher (run on a Linux machine on the lab switch).

Reads marker frames from the NOKOV server via the XING SDK and broadcasts
them as JSON over UDP. The Mac (or any other host on the subnet) runs
mocap_receiver.py to consume the stream.

Usage on the Linux box:
  # 1) put this whole motion_capture/ folder there
  # 2) configure the NIC on the lab subnet:
  #      sudo ip addr add 10.1.1.51/24 dev <iface>
  # 3) verify:
  #      ping 10.1.1.198
  # 4) run:
  #      python motion_capture/mocap_publisher.py
  #
  # Optional flags:
  #   --server 10.1.1.198          NOKOV server IP
  #   --target 10.1.1.255          where to send (subnet broadcast by default)
  #   --port 9870
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from threading import Lock

sdk_path = (
    Path(__file__).parent
    / "XING_Python_SDK-4.1.0.5873"
    / "dist"
    / "nokovpy-3.0.1-py3.9"
)
sys.path.insert(0, str(sdk_path))

from nokov.nokovsdk import PySDKClient  # noqa: E402

INVALID_MARKER_THRESHOLD = 9000000.0

state = {"sock": None, "addr": None, "frames": 0, "t0": None, "drops": 0}
lock = Lock()


def _build_markers(frame):
  markers = []
  for i_set in range(frame.nMarkerSets):
    ms = frame.MocapData[i_set]
    for i_m in range(ms.nMarkers):
      x, y, z = ms.Markers[i_m]
      if x < INVALID_MARKER_THRESHOLD:
        markers.append({
            "id": f"labeled_{i_set}_{i_m}",
            "xyz": [x / 1000.0, y / 1000.0, z / 1000.0],
        })
  for i_o in range(frame.nOtherMarkers):
    x, y, z = frame.OtherMarkers[i_o]
    if x < INVALID_MARKER_THRESHOLD:
      markers.append({
          "id": f"unlabeled_{i_o}",
          "xyz": [x / 1000.0, y / 1000.0, z / 1000.0],
      })
  return markers


def callback(p_frame, _user):
  if p_frame is None:
    return
  frame = p_frame.contents
  markers = _build_markers(frame)
  payload = json.dumps({
      "frame": frame.iFrame,
      "ts": time.time(),
      "markers": markers,
  }).encode()

  try:
    state["sock"].sendto(payload, state["addr"])
  except OSError as e:
    state["drops"] += 1
    if state["drops"] % 60 == 1:
      print(f"sendto failed (#{state['drops']}): {e}", file=sys.stderr)

  with lock:
    state["frames"] += 1
    if state["t0"] is None:
      state["t0"] = time.time()
    if state["frames"] % 60 == 0:
      elapsed = time.time() - state["t0"]
      hz = state["frames"] / elapsed if elapsed > 0 else 0.0
      print(
          f"[{elapsed:6.1f}s] frame {frame.iFrame} | "
          f"{len(markers):3d} markers | {hz:5.1f}Hz | "
          f"-> {state['addr'][0]}:{state['addr'][1]} ({len(payload)}B)"
      )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--server", default="10.1.1.198", help="NOKOV server IP")
  ap.add_argument(
      "--target",
      default="10.1.1.255",
      help="UDP destination (subnet broadcast 10.1.1.255, or a specific IP)",
  )
  ap.add_argument("--port", type=int, default=9870)
  args = ap.parse_args()

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
  state["sock"] = sock
  state["addr"] = (args.target, args.port)

  print(f"connecting to NOKOV at {args.server} ...")
  client = PySDKClient()
  ret = client.Initialize(bytes(args.server, encoding="utf8"))
  if ret != 0:
    print(f"NOKOV Initialize failed (code {ret})", file=sys.stderr)
    sys.exit(1)
  print("connected. registering callback ...")
  client.PySetDataCallback(callback, None)
  print(f"publishing -> udp://{args.target}:{args.port}  (Ctrl-C to stop)")

  try:
    while True:
      time.sleep(1.0)
  except KeyboardInterrupt:
    print("\ninterrupted")
  finally:
    client.Uninitialize()
    sock.close()
    print("done")


if __name__ == "__main__":
  main()
