"""
VRPN rigid-body reader for the UR3 pick loop — the VRPN analog of
motion_capture/mymocap/mocap_dependencies.py (which uses the Nokov SDK).

The Nokov (XINGYING) server also broadcasts every rigid body and marker as a
VRPN *tracker* on TCP/UDP port 3883. This module subscribes to one or more
named VRPN trackers and exposes the SAME public API as NokovRigidBodyReader,
so it is a drop-in alternative for the pick loop / any consumer of the reader.

Two streaming modes (see __init__):
    * specific bodies -> pass names=["PUBeam_11", ...]
    * stream all      -> names=None  (every sender the server publishes is
                         auto-discovered and subscribed)

The `vrpn` Python module is the hand-coded VRPN 3.x binding built from source
and vendored at
  motion_capture/nokov_python_vrpn-master/dist/vrpn.so
(see that folder's README.md for build provenance / how to rebuild it).

Usage:
    from motion_capture.mymocap.vrpn_dependencies import VRPNRigidBodyReader

    reader = VRPNRigidBodyReader("10.1.1.198", rigid_body_name="PUBeam_11")
    reader.start()                     # subscribes + spins background thread
    xyz  = reader.get_rigid_body_xyz()  # np.ndarray (3,) meters, or None
    quat = reader.get_rigid_body_quat() # np.ndarray (4,) (w,x,y,z), or None
    reader.stop()

Discover the tracker names the server publishes:
    from motion_capture.mymocap.vrpn_dependencies import discover_senders
    print(discover_senders("10.1.1.198"))

Notes
-----
* VRPN positions are already in METERS (no mm->m conversion, unlike Nokov).
* VRPN quaternions arrive as (x, y, z, w); they are stored here as (w, x, y, z)
  to match NokovRigidBodyReader (and the 26D obs builder in
  ur3_realrobot_dependencies.py).
* A tracker the server *knows* but isn't currently streaming (rigid body not
  visible to the cameras) yields no reports — its pose stays None and VRPN
  prints a "No response from server" warning. That is not an error.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, List, Optional

import numpy as np

# The vendored hand-coded VRPN 3.x Python binding lives next door. Prefer an
# already-importable `vrpn` (e.g. copied into site-packages); else add the
# vendored dist dir to sys.path (mirrors the Nokov SDK fallback pattern).
try:
    import vrpn
except ImportError:
    _VRPN_DIST = (
        Path(__file__).resolve().parents[1]
        / "nokov_python_vrpn-master"
        / "dist"
    )
    sys.path.insert(0, str(_VRPN_DIST))
    try:
        import vrpn
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Could not import the `vrpn` module. Expected the vendored binding "
            f"at {_VRPN_DIST}/vrpn.so — rebuild it per "
            "motion_capture/nokov_python_vrpn-master/README.md."
        ) from e

# Discovery helper binary (built from nokov_python_vrpn-master/src). The
# hand-coded `vrpn` module does not expose connection-level sender enumeration,
# so listing the server's senders shells out to this tool. Prefer one on PATH
# (e.g. copied into the venv's bin/, so the vendored folder isn't needed); else
# fall back to the vendored copy next door.
def _resolve_discover_bin() -> Optional[Path]:
    on_path = shutil.which("discover_senders")
    if on_path:
        return Path(on_path)
    vendored = (
        Path(__file__).resolve().parents[1]
        / "nokov_python_vrpn-master"
        / "bin"
        / "discover_senders"
    )
    return vendored if vendored.exists() else None


_DISCOVER_BIN = _resolve_discover_bin()

# Internal control sender every VRPN server advertises — never a rigid body.
_VRPN_CONTROL = "VRPN Control"

DEFAULT_SERVER_IP = "10.1.1.198"
DEFAULT_PORT = 3883


def _host_arg(server_ip: str, port: int) -> str:
    """VRPN host string: 'ip' for the default port, else 'ip:port'."""
    return server_ip if port == DEFAULT_PORT else f"{server_ip}:{port}"


def _rotate_vec_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` (3,) by unit quaternion `quat_wxyz` (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rot @ vec


def discover_senders(
    server_ip: str = DEFAULT_SERVER_IP,
    port: int = DEFAULT_PORT,
    timeout: float = 8.0,
) -> List[str]:
    """Return the VRPN sender (tracker) names the server publishes.

    Connects, pumps the control channel for a couple of seconds, and lists
    every sender (rigid bodies AND individual markers). The internal
    'VRPN Control' sender is excluded. Returns [] if the helper is missing or
    the server is unreachable. This is the VRPN analog of
    mocap_dependencies.list_rigid_bodies().
    """
    if _DISCOVER_BIN is None:
        print("[VRPN] discovery helper 'discover_senders' not found on PATH "
              "or in nokov_python_vrpn-master/bin")
        return []
    try:
        proc = subprocess.run(
            [str(_DISCOVER_BIN), _host_arg(server_ip, port)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[VRPN] discovery timed out after {timeout}s")
        return []

    names: List[str] = []
    in_senders = False
    for line in proc.stdout.splitlines():
        if line.startswith("Senders:"):
            in_senders = True
            continue
        if line.startswith("Message types:"):
            break
        if in_senders and "]" in line:
            name = line.split("]", 1)[1].strip()
            if name and name != _VRPN_CONTROL:
                names.append(name)
    return names


class VRPNRigidBodyReader:
    """Thread-safe reader exposing pose(s) from named VRPN trackers."""

    def __init__(
        self,
        server_ip: str = DEFAULT_SERVER_IP,
        rigid_body_name: Optional[str] = None,
        names: Optional[List[str]] = None,
        port: int = DEFAULT_PORT,
        offset: Optional[np.ndarray] = None,
    ):
        """Args:
        server_ip: Nokov VRPN server IP (USB-Ethernet subnet, default
            10.1.1.198).
        rigid_body_name: name of the tracker the get_rigid_body_* accessors
            return. If None, the first tracker that reports is used. Always
            included in the subscription even if not in `names`.
        names: explicit list of tracker names to subscribe to ("stream specific
            rigid bodies"). If None, every sender the server publishes is
            auto-discovered and subscribed ("stream all the data"). All
            subscribed bodies are stored; see get_body().
        port: VRPN port (default 3883).
        offset: body-frame offset (3,) added to the centerpoint by the
            get_offset_* accessors, rotated by the body's orientation (so it
            stays fixed relative to the body as it rotates). Default zeros.
            Change live with set_offset().
        """
        self.server_ip = server_ip
        self.port = port
        self.rigid_body_name = rigid_body_name
        self._requested_names = names  # None => discover all at connect
        self._offset = (
            np.zeros(3) if offset is None
            else np.asarray(offset, dtype=np.float64).reshape(3).copy()
        )

        self.thread: Optional[Thread] = None
        self.running = Event()
        self.connected = Event()

        self._lock = Lock()
        # Latest pose per tracker name: {name: {"xyz": np(3), "quat": np(4)}}.
        self._bodies: Dict[str, Dict[str, np.ndarray]] = {}
        self._names: List[str] = []          # names actually subscribed
        self._trackers: list = []            # vrpn.receiver.Tracker objects
        self.frame_count = 0
        self.last_update_time: Optional[float] = None

    # ------------------------------------------------------------------ VRPN
    def _make_handler(self, name: str):
        """Build the per-tracker position callback (closes over `name`)."""

        def handler(userdata, data):  # VRPN calls handler(userdata, data dict)
            pos = data["position"]      # (x, y, z) in meters
            q = data["quaternion"]      # VRPN order (x, y, z, w)
            xyz = np.array(pos, dtype=np.float64)
            quat = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # wxyz
            with self._lock:
                self._bodies[name] = {"xyz": xyz, "quat": quat}
                self.frame_count += 1
                self.last_update_time = time.time()

        return handler

    def _resolve_names(self) -> List[str]:
        """Decide which tracker names to subscribe to."""
        if self._requested_names is None:
            names = discover_senders(self.server_ip, self.port)
            if not names:
                print("[VRPN] discovery returned no senders")
        else:
            names = list(self._requested_names)
        if self.rigid_body_name and self.rigid_body_name not in names:
            names.append(self.rigid_body_name)
        return names

    def _connection_thread(self):
        try:
            print(f"[VRPN] Connecting to {self.server_ip}:{self.port} ...")
            self._names = self._resolve_names()
            if not self._names:
                print("[VRPN] No trackers to subscribe to — aborting.")
                self.connected.clear()
                return

            host = _host_arg(self.server_ip, self.port)
            for nm in self._names:
                tracker = vrpn.receiver.Tracker(f"{nm}@{host}")
                tracker.register_change_handler(nm, self._make_handler(nm),
                                                "position")
                self._trackers.append(tracker)

            print(f"[VRPN] ✓ Subscribed to {len(self._trackers)} tracker(s): "
                  f"{self._names}")
            self.connected.set()

            while self.running.is_set():
                for tracker in self._trackers:
                    tracker.mainloop()
                time.sleep(0.001)
        except Exception as e:  # pragma: no cover - hardware dependent
            print(f"[VRPN] Exception in connection thread: {e}")
            self.connected.clear()

    # --------------------------------------------------------------- public
    def start(self, timeout: float = 8.0) -> bool:
        """Subscribe and spin the background thread.

        Returns True once the subscription pipeline is up (analogous to the
        Nokov reader's connect). Note that `connected` means "trackers
        subscribed", not "data flowing" — poses stay None until the first
        report arrives. Discovery (names=None) can take a few seconds, so the
        default timeout is generous.
        """
        if self.thread and self.thread.is_alive():
            return self.connected.is_set()

        self.running.set()
        self.connected.clear()
        self.thread = Thread(target=self._connection_thread, daemon=True)
        self.thread.start()

        if self.connected.wait(timeout=timeout):
            return True
        print(f"[VRPN] Subscription timeout after {timeout}s")
        self.stop()
        return False

    def stop(self):
        """Stop the background thread."""
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.connected.clear()

    def is_connected(self) -> bool:
        return self.connected.is_set()

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Block until at least one tracker report has arrived (or timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._bodies:
                    return True
            time.sleep(0.02)
        return False

    def _target_body_locked(self) -> Optional[Dict[str, np.ndarray]]:
        """Latest pose of the tracked body. Call while holding _lock."""
        if self.rigid_body_name is not None:
            return self._bodies.get(self.rigid_body_name)
        # No target requested: fall back to the first body that has reported.
        return next(iter(self._bodies.values()), None)

    def get_rigid_body_xyz(self) -> Optional[np.ndarray]:
        """Latest tracked-body center as np.ndarray (3,) in meters, or None."""
        with self._lock:
            body = self._target_body_locked()
            return None if body is None else body["xyz"].copy()

    def get_rigid_body_quat(self) -> Optional[np.ndarray]:
        """Latest tracked-body orientation (w, x, y, z), or None."""
        with self._lock:
            body = self._target_body_locked()
            return None if body is None else body["quat"].copy()

    def get_rigid_body_pose(self):
        """Return (xyz[3] meters, quat[4] as w,x,y,z), or (None, None)."""
        with self._lock:
            body = self._target_body_locked()
            if body is None:
                return None, None
            return body["xyz"].copy(), body["quat"].copy()

    # --------------------------------------------------------------- offset
    def set_offset(self, offset: np.ndarray) -> None:
        """Set the body-frame offset (3,) used by the get_offset_* accessors."""
        off = np.asarray(offset, dtype=np.float64).reshape(3).copy()
        with self._lock:
            self._offset = off

    def get_offset(self) -> np.ndarray:
        """Return the current body-frame offset (3,)."""
        with self._lock:
            return self._offset.copy()

    def _body_by_name_locked(
        self, name: Optional[str]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Latest pose of `name` (or the tracked body if None). Hold _lock."""
        if name is None:
            return self._target_body_locked()
        return self._bodies.get(name)

    def get_offset_point(
        self, name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Centerpoint + body-frame offset, as world-frame np(3) meters, or None.

        The stored offset is rotated by the body's orientation, so it stays
        fixed relative to the body as the body rotates. `name` defaults to the
        tracked body (rigid_body_name, or the first to report).
        """
        with self._lock:
            body = self._body_by_name_locked(name)
            if body is None:
                return None
            return body["xyz"] + _rotate_vec_by_quat(body["quat"], self._offset)

    def get_offset_pose(self, name: Optional[str] = None):
        """Return (offset_point[3] meters, quat[4] as w,x,y,z), or (None, None).

        Same offset rule as get_offset_point; the orientation is the body's
        unmodified quaternion. Use this to stream a single body's origin
        (offset-adjusted) and orientation.
        """
        with self._lock:
            body = self._body_by_name_locked(name)
            if body is None:
                return None, None
            point = body["xyz"] + _rotate_vec_by_quat(body["quat"],
                                                       self._offset)
            return point, body["quat"].copy()

    def get_body(self, name: Optional[str] = None):
        """Filter the subscribed trackers by name (or return them all).

        name None  -> dict of every subscribed body that has reported, keyed by
                      name: {name: {"xyz": np(3), "quat": np(4)}}.
        name given -> {"xyz": np(3), "quat": np(4)} for that body, or None if
                      it is subscribed but hasn't reported yet.

        Raises KeyError (listing the subscribed names) if `name` was never
        subscribed.
        """
        with self._lock:
            if name is None:
                return {
                    nm: {"xyz": p["xyz"].copy(), "quat": p["quat"].copy()}
                    for nm, p in self._bodies.items()
                }
            if name not in self._names:
                available = ", ".join(self._names) or "(none)"
                raise KeyError(
                    f"tracker '{name}' not subscribed. Subscribed: {available}"
                )
            pose = self._bodies.get(name)
            if pose is None:
                return None
            return {"xyz": pose["xyz"].copy(), "quat": pose["quat"].copy()}

    def get_stats(self) -> dict:
        with self._lock:
            body = self._target_body_locked()
            return {
                "connected": self.is_connected(),
                "server_ip": self.server_ip,
                "rigid_body_name": self.rigid_body_name,
                "num_subscribed": len(self._names),
                "num_reporting": len(self._bodies),
                "frame_count": self.frame_count,
                "last_update": self.last_update_time,
                "latest_xyz": None if body is None else body["xyz"].tolist(),
            }


# ----------------------------------------------------------------------- demo
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="VRPN rigid-body reader demo")
    ap.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--name", default=None,
                    help="Tracker NAME to track. If given without --names, "
                         "subscribes to ONLY this body (default: first to "
                         "report among all)")
    ap.add_argument("--names", default=None,
                    help="Comma-separated names to subscribe to "
                         "(default: discover + stream ALL)")
    ap.add_argument("--offset", default=None,
                    help="Body-frame offset 'x,y,z' (m) added to the "
                         "centerpoint, rotated by the body's orientation")
    ap.add_argument("--list", action="store_true",
                    help="Just discover and print sender names, then exit")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    if args.list:
        for nm in discover_senders(args.server_ip, args.port):
            print(nm)
        sys.exit(0)

    # --name without --names => stream ONLY that body.
    if args.names:
        names = args.names.split(",")
    elif args.name:
        names = [args.name]
    else:
        names = None  # discover + stream all
    offset = ([float(v) for v in args.offset.split(",")]
              if args.offset else None)
    reader = VRPNRigidBodyReader(
        args.server_ip,
        rigid_body_name=args.name,
        names=names,
        port=args.port,
        offset=offset,
    )
    if not reader.start(timeout=8.0):
        print("Failed to subscribe to VRPN server")
        sys.exit(1)
    print(f"[VRPN] subscribed trackers: {reader._names}")
    if offset is not None:
        print(f"[VRPN] body-frame offset: {reader.get_offset().tolist()}")

    try:
        n = int(args.duration * 10)
        for i in range(n):
            time.sleep(0.1)
            if i % 10 == 0:
                xyz, quat = reader.get_offset_pose()
                if xyz is not None:
                    print(f"[{i * 0.1:5.1f}s] xyz (m): "
                          f"{xyz[0]:7.3f} {xyz[1]:7.3f} {xyz[2]:7.3f}  | "
                          f"quat (w,x,y,z): "
                          f"{quat[0]:6.3f} {quat[1]:6.3f} "
                          f"{quat[2]:6.3f} {quat[3]:6.3f}  | "
                          f"reporting {len(reader.get_body())}/"
                          f"{len(reader._names)}")
                else:
                    print(f"[{i * 0.1:5.1f}s] no report yet "
                          f"(subscribed {len(reader._names)})")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        print("✓ Reader stopped")
