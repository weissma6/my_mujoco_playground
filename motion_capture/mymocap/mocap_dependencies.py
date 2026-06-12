"""
Nokov (XINGYING) rigid-body reader for the UR3 pick loop.

Unlike motion_capture/mymocap/motion_capture_reader.py (markers-only), this
module reads a *rigid body* from the Nokov stream and exposes only the XYZ of
its center. A background daemon thread keeps the latest pose; the control loop
reads it at its own rate (frequency-independent: mocap ~60 Hz, loop 50 Hz).

Usage:
    from motion_capture.mymocap.mocap_dependencies import NokovRigidBodyReader

    reader = NokovRigidBodyReader("10.1.1.198", rigid_body_id=1)
    reader.start()                     # connects + spins background thread
    xyz = reader.get_rigid_body_xyz()  # np.ndarray (3,) in meters, or None
    reader.stop()

Run `motion_capture/mymocap/test_mocap_connection.py` first to discover the
rigid-body IDs the server publishes (PyGetDataDescriptions dump at startup).

Notes
-----
* The Nokov stream is in millimeters; positions are converted to meters here.
* PySDKClient has NO Uninitialize() method — do not call it on shutdown
  (see CLAUDE.md). The background thread simply exits.
"""

import sys
import time
from ctypes import POINTER
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Optional

import numpy as np

# Prefer the pip-installed wheel (`from nokov.nokovsdk import PySDKClient`).
# Fall back to the vendored SDK dist on sys.path if the wheel isn't installed.
try:
    from nokov.nokovsdk import DataDescriptions, DataDescriptors, PySDKClient
except ImportError:
    _sdk_path = (
        Path(__file__).parent
        / "XING_Python_SDK-4.1.0.5873"
        / "dist"
        / "nokovpy-3.0.1-py3.9"
    )
    sys.path.insert(0, str(_sdk_path))
    try:
        from nokov.nokovsdk import DataDescriptions, DataDescriptors, PySDKClient
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Could not import Nokov SDK. Install the wheel "
            "(uv pip install motion_capture/nokov_python_sdk-master/dist/"
            "nokovpy-3.0.1-py3-none-any.whl) or keep the vendored SDK on disk."
        ) from e


def list_rigid_bodies(client) -> Dict[str, int]:
    """Return {name: id} for every rigid body the server publishes.

    Reads the static scene definition via PyGetDataDescriptions() (NOKOV frame
    data carries only integer ids, never names — the name->id mapping lives in
    the descriptions). Returns an empty dict if the scene has no rigid bodies
    (e.g. 'SDK Enabled' is on but no rigid body is defined).
    """
    pdds = POINTER(DataDescriptions)()
    client.PyGetDataDescriptions(pdds)
    out: Dict[str, int] = {}
    descs = pdds.contents
    for i in range(descs.nDataDescriptions):
        d = descs.arrDataDescriptions[i]
        if d.type == DataDescriptors.Descriptor_RigidBody.value:
            rb = d.Data.RigidBodyDescription.contents
            name = rb.szName.decode("utf-8").rstrip("\x00")
            out[name] = rb.ID
    return out


class NokovRigidBodyReader:
    """Thread-safe reader exposing the XYZ center of one Nokov rigid body."""

    def __init__(
        self,
        server_ip: str = "10.1.1.198",
        rigid_body_id: Optional[int] = None,
        rigid_body_name: Optional[str] = None,
    ):
        """Args:
        server_ip: Nokov server IP (USB-Ethernet subnet, default 10.1.1.198).
        rigid_body_id: ID of the rigid body the back-compat accessors track (as
            dumped by test_mocap_connection.py). Takes precedence over
            rigid_body_name.
        rigid_body_name: Name of the rigid body to track (e.g. "CubeInCube"),
            resolved to an id once at connect via PyGetDataDescriptions. If
            neither id nor name is given, the first rigid body in each frame is
            used. All bodies are always stored; see get_body().
        """
        self.server_ip = server_ip
        self.rigid_body_id = rigid_body_id
        self.rigid_body_name = rigid_body_name

        self.client: Optional[PySDKClient] = None
        self.thread: Optional[Thread] = None
        self.running = Event()
        self.connected = Event()

        self._lock = Lock()
        # Every rigid body in the latest frame: {id: {"xyz": np(3), "quat": np(4)}}.
        self._bodies: Dict[int, Dict[str, np.ndarray]] = {}
        # name<->id maps from the scene descriptions (resolved at connect).
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        # The body the get_rigid_body_* accessors return (id/name resolved).
        self._target_id: Optional[int] = rigid_body_id
        self.frame_count = 0
        self.last_update_time: Optional[float] = None

    # ------------------------------------------------------------------ SDK
    def _mocap_callback(self, pFrameOfMocapData, pUserData):
        """SDK callback (~60 Hz). Stores every rigid body's pose (m)."""
        if pFrameOfMocapData is None:
            return

        frame = pFrameOfMocapData.contents

        bodies: Dict[int, Dict[str, np.ndarray]] = {}
        for i in range(frame.nRigidBodies):
            body = frame.RigidBodies[i]
            bodies[body.ID] = {
                "xyz": np.array(
                    [body.x / 1000.0, body.y / 1000.0, body.z / 1000.0],
                    dtype=np.float64,
                ),
                "quat": np.array(
                    [body.qw, body.qx, body.qy, body.qz],
                    dtype=np.float64,
                ),
            }

        if not bodies:
            return  # no rigid bodies in this frame

        with self._lock:
            self._bodies = bodies
            self.frame_count += 1
            self.last_update_time = time.time()

    def _connection_thread(self):
        try:
            print(f"[Nokov] Connecting to {self.server_ip} ...")
            self.client = PySDKClient()
            ret = self.client.Initialize(bytes(self.server_ip, encoding="utf8"))
            if ret != 0:
                print(f"[Nokov] ❌ Connection failed (error {ret})")
                self.connected.clear()
                return

            print("[Nokov] ✓ Connected")

            # Resolve the scene's rigid-body name<->id maps once (static for a
            # session). If a name was requested, resolve it to the target id
            # now and fail loudly if the server doesn't publish it.
            self._name_to_id = list_rigid_bodies(self.client)
            self._id_to_name = {v: k for k, v in self._name_to_id.items()}
            if self.rigid_body_name is not None and self.rigid_body_id is None:
                if self.rigid_body_name not in self._name_to_id:
                    available = ", ".join(sorted(self._name_to_id)) or "(none)"
                    print(
                        f"[Nokov] ❌ rigid body '{self.rigid_body_name}' not "
                        f"found. Available rigid bodies: {available}"
                    )
                    self.connected.clear()
                    return
                self._target_id = self._name_to_id[self.rigid_body_name]

            self.client.PySetDataCallback(self._mocap_callback, None)
            self.connected.set()

            while self.running.is_set():
                time.sleep(0.1)
        except Exception as e:  # pragma: no cover - hardware dependent
            print(f"[Nokov] Exception in connection thread: {e}")
            self.connected.clear()
        # NOTE: no client.Uninitialize() — that method does not exist.

    # --------------------------------------------------------------- public
    def start(self, timeout: float = 5.0) -> bool:
        """Connect and spin the background thread. Returns True if connected."""
        if self.thread and self.thread.is_alive():
            return self.connected.is_set()

        self.running.set()
        self.connected.clear()
        self.thread = Thread(target=self._connection_thread, daemon=True)
        self.thread.start()

        if self.connected.wait(timeout=timeout):
            return True
        print(f"[Nokov] Connection timeout after {timeout}s")
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

    def _target_body_locked(self) -> Optional[Dict[str, np.ndarray]]:
        """Latest pose of the tracked (target) body. Call while holding _lock."""
        if self._target_id is not None:
            return self._bodies.get(self._target_id)
        # No target requested: fall back to the first body in the frame.
        return next(iter(self._bodies.values()), None)

    def get_rigid_body_xyz(self) -> Optional[np.ndarray]:
        """Return the latest tracked-body center as np.ndarray (3,) in meters.

        Returns None if no frame containing the tracked body has arrived yet.
        """
        with self._lock:
            body = self._target_body_locked()
            return None if body is None else body["xyz"].copy()

    def get_rigid_body_quat(self) -> Optional[np.ndarray]:
        """Return the latest tracked-body orientation as quaternion (w, x, y, z).

        Returns None if no frame containing the tracked body has arrived yet.
        """
        with self._lock:
            body = self._target_body_locked()
            return None if body is None else body["quat"].copy()

    def get_rigid_body_pose(self):
        """Return (xyz[3] in meters, quat[4] as w,x,y,z), or (None, None)."""
        with self._lock:
            body = self._target_body_locked()
            if body is None:
                return None, None
            return body["xyz"].copy(), body["quat"].copy()

    def get_body(self, name: Optional[str] = None):
        """Filter the streamed rigid bodies by name (or return them all).

        name None  -> dict of EVERY body in the latest frame, keyed by name
                      (or by str(id) if the body has no name in the scene
                      descriptions): {name: {"xyz": np(3), "quat": np(4)}}.
        name given -> {"xyz": np(3), "quat": np(4)} for that body, or None if
                      the body is known to the scene but absent from the latest
                      frame.

        Raises KeyError (listing the available names) if `name` is not a rigid
        body the server publishes.
        """
        with self._lock:
            if name is None:
                out = {}
                for bid, pose in self._bodies.items():
                    key = self._id_to_name.get(bid, str(bid))
                    out[key] = {
                        "xyz": pose["xyz"].copy(),
                        "quat": pose["quat"].copy(),
                    }
                return out

            if name not in self._name_to_id:
                available = ", ".join(sorted(self._name_to_id)) or "(none)"
                raise KeyError(
                    f"rigid body '{name}' not published. Available: {available}"
                )
            pose = self._bodies.get(self._name_to_id[name])
            if pose is None:
                return None
            return {"xyz": pose["xyz"].copy(), "quat": pose["quat"].copy()}

    def get_stats(self) -> dict:
        with self._lock:
            body = self._target_body_locked()
            return {
                "connected": self.is_connected(),
                "rigid_body_id": self.rigid_body_id,
                "rigid_body_name": self.rigid_body_name,
                "target_id": self._target_id,
                "num_bodies": len(self._bodies),
                "frame_count": self.frame_count,
                "last_update": self.last_update_time,
                "latest_xyz": None if body is None else body["xyz"].tolist(),
            }


# ----------------------------------------------------------------------- demo
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Nokov rigid-body reader demo")
    ap.add_argument("--server-ip", default="10.1.1.198")
    ap.add_argument("--rigid-body-id", type=int, default=None,
                    help="Rigid body ID to track (default: first body)")
    ap.add_argument("--rigid-body-name", default=None,
                    help="Rigid body NAME to track, e.g. CubeInCube "
                         "(resolved to an id at connect)")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    reader = NokovRigidBodyReader(
        args.server_ip,
        rigid_body_id=args.rigid_body_id,
        rigid_body_name=args.rigid_body_name,
    )
    if not reader.start(timeout=5.0):
        print("Failed to connect to Nokov server")
        sys.exit(1)
    print(f"[Nokov] rigid bodies in scene: {reader._name_to_id}")

    try:
        n = int(args.duration * 10)
        for i in range(n):
            time.sleep(0.1)
            if i % 10 == 0:
                xyz, quat = reader.get_rigid_body_pose()
                if xyz is not None:
                    print(f"[{i * 0.1:5.1f}s] xyz (m): "
                          f"{xyz[0]:7.3f} {xyz[1]:7.3f} {xyz[2]:7.3f}  | "
                          f"quat (w,x,y,z): "
                          f"{quat[0]:6.3f} {quat[1]:6.3f} "
                          f"{quat[2]:6.3f} {quat[3]:6.3f}")
                else:
                    print(f"[{i * 0.1:5.1f}s] no rigid body yet")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        print("✓ Reader stopped")
