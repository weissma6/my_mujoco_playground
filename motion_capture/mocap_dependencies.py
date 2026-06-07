"""
Nokov (XINGYING) rigid-body reader for the UR3 pick loop.

Unlike motion_capture/mymocap/motion_capture_reader.py (markers-only), this
module reads a *rigid body* from the Nokov stream and exposes only the XYZ of
its center. A background daemon thread keeps the latest pose; the control loop
reads it at its own rate (frequency-independent: mocap ~60 Hz, loop 50 Hz).

Usage:
    from motion_capture.mocap_dependencies import NokovRigidBodyReader

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
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np

# Prefer the pip-installed wheel (`from nokov.nokovsdk import PySDKClient`).
# Fall back to the vendored SDK dist on sys.path if the wheel isn't installed.
try:
    from nokov.nokovsdk import PySDKClient
except ImportError:
    _sdk_path = (
        Path(__file__).parent
        / "XING_Python_SDK-4.1.0.5873"
        / "dist"
        / "nokovpy-3.0.1-py3.9"
    )
    sys.path.insert(0, str(_sdk_path))
    try:
        from nokov.nokovsdk import PySDKClient
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Could not import Nokov SDK. Install the wheel "
            "(uv pip install motion_capture/nokov_python_sdk-master/dist/"
            "nokovpy-3.0.1-py3-none-any.whl) or keep the vendored SDK on disk."
        ) from e


class NokovRigidBodyReader:
    """Thread-safe reader exposing the XYZ center of one Nokov rigid body."""

    def __init__(
        self,
        server_ip: str = "10.1.1.198",
        rigid_body_id: Optional[int] = None,
    ):
        """Args:
        server_ip: Nokov server IP (USB-Ethernet subnet, default 10.1.1.198).
        rigid_body_id: ID of the rigid body to track (as dumped by
            test_mocap_connection.py). If None, the first rigid body in each
            frame is used.
        """
        self.server_ip = server_ip
        self.rigid_body_id = rigid_body_id

        self.client: Optional[PySDKClient] = None
        self.thread: Optional[Thread] = None
        self.running = Event()
        self.connected = Event()

        self._lock = Lock()
        self._latest_xyz: Optional[np.ndarray] = None
        self._latest_quat: Optional[np.ndarray] = None
        self.frame_count = 0
        self.last_update_time: Optional[float] = None

    # ------------------------------------------------------------------ SDK
    def _mocap_callback(self, pFrameOfMocapData, pUserData):
        """SDK callback (~60 Hz). Stores the tracked body's XYZ center (m)."""
        if pFrameOfMocapData is None:
            return

        frame = pFrameOfMocapData.contents

        chosen = None
        for i in range(frame.nRigidBodies):
            body = frame.RigidBodies[i]
            if self.rigid_body_id is None or body.ID == self.rigid_body_id:
                chosen = body
                break

        if chosen is None:
            return  # tracked body not present in this frame

        xyz = np.array(
            [chosen.x / 1000.0, chosen.y / 1000.0, chosen.z / 1000.0],
            dtype=np.float64,
        )
        quat = np.array(
            [chosen.qw, chosen.qx, chosen.qy, chosen.qz],
            dtype=np.float64,
        )

        with self._lock:
            self._latest_xyz = xyz
            self._latest_quat = quat
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

    def get_rigid_body_xyz(self) -> Optional[np.ndarray]:
        """Return the latest rigid-body center as np.ndarray (3,) in meters.

        Returns None if no frame containing the tracked body has arrived yet.
        """
        with self._lock:
            return None if self._latest_xyz is None else self._latest_xyz.copy()

    def get_rigid_body_quat(self) -> Optional[np.ndarray]:
        """Return the latest rigid-body orientation as quaternion (w, x, y, z).

        Returns None if no frame containing the tracked body has arrived yet.
        """
        with self._lock:
            return None if self._latest_quat is None else self._latest_quat.copy()

    def get_rigid_body_pose(self):
        """Return (xyz[3] in meters, quat[4] as w,x,y,z), or (None, None)."""
        with self._lock:
            xyz = None if self._latest_xyz is None else self._latest_xyz.copy()
            quat = None if self._latest_quat is None else self._latest_quat.copy()
            return xyz, quat

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "connected": self.is_connected(),
                "rigid_body_id": self.rigid_body_id,
                "frame_count": self.frame_count,
                "last_update": self.last_update_time,
                "latest_xyz": None if self._latest_xyz is None
                else self._latest_xyz.tolist(),
            }


# ----------------------------------------------------------------------- demo
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Nokov rigid-body reader demo")
    ap.add_argument("--server-ip", default="10.1.1.198")
    ap.add_argument("--rigid-body-id", type=int, default=None,
                    help="Rigid body ID to track (default: first body)")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    reader = NokovRigidBodyReader(args.server_ip, rigid_body_id=args.rigid_body_id)
    if not reader.start(timeout=5.0):
        print("Failed to connect to Nokov server")
        sys.exit(1)

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
