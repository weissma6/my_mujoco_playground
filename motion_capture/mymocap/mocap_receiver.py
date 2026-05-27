"""UDP receiver for the NOKOV marker stream.

Pairs with motion_capture/mocap_publisher.py running on a Linux box on the
lab switch. No NOKOV SDK needed on this side.

Public API (importable from other scripts):
  start_listener(port=9870)
  get_latest_markers()       -> {marker_id: {"xyz": [x,y,z], "ts": t, "frame": n}}
  get_marker_by_id(mid)      -> {"xyz": [...], "ts": t, "frame": n} or None
  get_marker_centroid(ids)   -> {"xyz": [...], "ts": t} or None if any missing
  get_marker_count()         -> int
  last_frame_age_s()         -> float (time since last received frame)

Run as a script for a 30s connectivity demo:
  python motion_capture/mocap_receiver.py --port 9870
"""

import argparse
import json
import socket
import time
from threading import Lock, Thread

_state = {"frame": None, "ts": None, "markers": {}}
_lock = Lock()
_listener_started = False


def _listener_loop(port):
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(("0.0.0.0", port))
  while True:
    data, _src = sock.recvfrom(65535)
    try:
      msg = json.loads(data.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
      continue
    by_id = {
        m["id"]: {"xyz": m["xyz"], "ts": msg["ts"], "frame": msg["frame"]}
        for m in msg.get("markers", [])
    }
    with _lock:
      _state["frame"] = msg.get("frame")
      _state["ts"] = msg.get("ts")
      _state["markers"] = by_id


def start_listener(port=9870):
  """Start the background UDP listener (idempotent)."""
  global _listener_started
  if _listener_started:
    return
  Thread(target=_listener_loop, args=(port,), daemon=True).start()
  _listener_started = True


def get_latest_markers():
  with _lock:
    return {k: dict(v) for k, v in _state["markers"].items()}


def get_marker_by_id(marker_id):
  with _lock:
    m = _state["markers"].get(marker_id)
    return dict(m) if m else None


def get_marker_centroid(marker_ids):
  with _lock:
    positions = []
    timestamps = []
    for mid in marker_ids:
      m = _state["markers"].get(mid)
      if m is None:
        return None
      positions.append(m["xyz"])
      timestamps.append(m["ts"])
  if not positions:
    return None
  n = len(positions)
  centroid = [sum(p[i] for p in positions) / n for i in range(3)]
  return {"xyz": centroid, "ts": sum(timestamps) / n}


def get_marker_count():
  with _lock:
    return len(_state["markers"])


def last_frame_age_s():
  with _lock:
    ts = _state["ts"]
  return float("inf") if ts is None else max(0.0, time.time() - ts)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", type=int, default=9870)
  ap.add_argument("--seconds", type=float, default=30.0)
  args = ap.parse_args()

  start_listener(args.port)
  print(f"listening udp://0.0.0.0:{args.port} ... waiting for first frame")

  t0 = time.time()
  last_frame = None
  while time.time() - t0 < args.seconds:
    time.sleep(0.5)
    n = get_marker_count()
    with _lock:
      f = _state["frame"]
    if n == 0:
      print(f"  [{time.time() - t0:5.1f}s] no frames yet")
      continue
    age_ms = last_frame_age_s() * 1000.0
    delta = "" if last_frame is None else f" (+{f - last_frame})"
    print(
        f"  [{time.time() - t0:5.1f}s] frame {f}{delta} | "
        f"{n} markers | age {age_ms:5.1f}ms"
    )
    last_frame = f
    markers = get_latest_markers()
    for mid, m in list(markers.items())[:3]:
      x, y, z = m["xyz"]
      print(f"    {mid:20s} -> X:{x:7.3f} Y:{y:7.3f} Z:{z:7.3f} m")
    print()


if __name__ == "__main__":
  main()
