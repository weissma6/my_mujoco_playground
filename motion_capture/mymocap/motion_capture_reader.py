"""
Frequency-independent XINYANG Motion Capture Reader.

This module provides a clean interface to query marker positions
regardless of the motion capture frame rate (runs at ~60Hz).
Control loops (50Hz) can query this whenever they need target positions.

Usage:
  from motion_capture_reader import XINYINGReader

  reader = XINYINGReader("10.1.1.198")
  reader.start()  # Runs background thread

  # In your 50Hz control loop:
  target_xyz = reader.get_target_position()  # Latest XYZ in meters

  reader.stop()
"""

import sys
import time
import threading
from threading import Thread, Lock, Event
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add XINYING SDK to path (SDK lives one level up at motion_capture/XING_Python_SDK-...)
sdk_path = Path(__file__).parent.parent / "XING_Python_SDK-4.1.0.5873" / "dist" / "nokovpy-3.0.1-py3.9"
sys.path.insert(0, str(sdk_path))

try:
  from nokov.nokovsdk import PySDKClient
except ImportError as e:
  raise ImportError(f"Could not import XINYING SDK from {sdk_path}: {e}")


class XINYINGReader:
  """
  Thread-safe wrapper for XINYANG motion capture data.

  Features:
  - Background thread continuously updates from mocap system
  - Frequency-independent querying (control loop asks when it wants)
  - Filters invalid markers automatically
  - Provides centroid calculation for rigid bodies
  """

  def __init__(self, server_ip: str = "10.1.1.198"):
    """Initialize XINYANG reader (does not connect yet)."""
    self.server_ip = server_ip
    self.client: Optional[PySDKClient] = None
    self.thread: Optional[Thread] = None
    self.running = Event()
    self.connected = Event()

    # Thread-safe marker storage
    self.lock = Lock()
    self.latest_markers: Dict = {}
    self.frame_count = 0
    self.last_update_time = None

    # Configuration
    self.invalid_marker_threshold = 9000000.0

  def _mocap_callback(self, pFrameOfMocapData, pUserData):
    """
    Callback invoked by SDK when new frame arrives (~60Hz).
    Extracts and stores all marker positions.
    """
    if pFrameOfMocapData is None:
      return

    frameData = pFrameOfMocapData.contents
    current_time = time.time()

    # Extract all markers
    markers = {}

    # Labeled markers
    for iMarkerSet in range(frameData.nMarkerSets):
      markerset = frameData.MocapData[iMarkerSet]
      for iMarker in range(markerset.nMarkers):
        x, y, z = markerset.Markers[iMarker]
        if x < self.invalid_marker_threshold:
          marker_id = f"labeled_{iMarkerSet}_{iMarker}"
          markers[marker_id] = {
            "xyz": [x / 1000.0, y / 1000.0, z / 1000.0],  # mm → m
            "timestamp": current_time,
            "frame": frameData.iFrame,
            "markerset": iMarkerSet,
          }

    # Unlabeled markers (OtherMarkers)
    for iOther in range(frameData.nOtherMarkers):
      x, y, z = frameData.OtherMarkers[iOther]
      if x < self.invalid_marker_threshold:
        marker_id = f"unlabeled_{iOther}"
        markers[marker_id] = {
          "xyz": [x / 1000.0, y / 1000.0, z / 1000.0],
          "timestamp": current_time,
          "frame": frameData.iFrame,
        }

    # Update global state
    with self.lock:
      self.latest_markers = markers
      self.frame_count += 1
      self.last_update_time = current_time

  def _connection_thread(self):
    """
    Background thread: connect and listen for marker data.
    """
    try:
      print(f"[XINYANG] Connecting to {self.server_ip}...")
      self.client = PySDKClient()
      ret = self.client.Initialize(bytes(self.server_ip, encoding="utf8"))

      if ret != 0:
        print(f"[XINYANG] ❌ Connection failed (error: {ret})")
        self.connected.clear()
        return

      print("[XINYANG] ✓ Connected")
      self.client.PySetDataCallback(self._mocap_callback, None)
      self.connected.set()

      # Keep thread alive while running
      while self.running.is_set():
        time.sleep(0.1)

    except Exception as e:
      print(f"[XINYANG] Exception in connection thread: {e}")
      self.connected.clear()
    finally:
      if self.client:
        self.client.Uninitialize()
        print("[XINYANG] Disconnected")

  def start(self, timeout: float = 5.0) -> bool:
    """
    Start background thread and connect to mocap system.

    Args:
      timeout: Max seconds to wait for connection

    Returns:
      True if connected, False otherwise
    """
    if self.thread and self.thread.is_alive():
      print("[XINYANG] Already running")
      return self.connected.is_set()

    self.running.set()
    self.connected.clear()
    self.thread = Thread(target=self._connection_thread, daemon=True)
    self.thread.start()

    # Wait for connection
    if self.connected.wait(timeout=timeout):
      return True
    else:
      print(f"[XINYANG] Connection timeout after {timeout}s")
      self.stop()
      return False

  def stop(self):
    """Stop background thread and disconnect."""
    self.running.clear()
    if self.thread:
      self.thread.join(timeout=2.0)
    self.connected.clear()

  def is_connected(self) -> bool:
    """Return whether currently connected."""
    return self.connected.is_set()

  def get_latest_markers(self) -> Dict:
    """
    Get all currently tracked markers.

    Returns:
      Dict[marker_id, {"xyz": [x,y,z], "timestamp": t, "frame": n}]
    """
    with self.lock:
      return dict(self.latest_markers)

  def get_marker_count(self) -> int:
    """Return number of currently tracked markers."""
    with self.lock:
      return len(self.latest_markers)

  def get_marker_position(self, marker_id: str) -> Optional[Tuple[float, float, float]]:
    """
    Get XYZ position of specific marker.

    Args:
      marker_id: e.g., "labeled_0_0" or "unlabeled_0"

    Returns:
      (x, y, z) in meters, or None if not found
    """
    with self.lock:
      if marker_id in self.latest_markers:
        xyz = self.latest_markers[marker_id]["xyz"]
        return tuple(xyz)
    return None

  def get_marker_centroid(
    self, marker_ids: List[str]
  ) -> Optional[Tuple[float, float, float]]:
    """
    Compute centroid of multiple markers (for rigid bodies).

    Args:
      marker_ids: List of marker IDs to average

    Returns:
      (x, y, z) centroid in meters, or None if any marker missing
    """
    with self.lock:
      positions = []
      for mid in marker_ids:
        if mid in self.latest_markers:
          xyz = self.latest_markers[mid]["xyz"]
          positions.append(xyz)
        else:
          return None  # Missing marker

      if positions:
        import numpy as np
        centroid = np.mean(positions, axis=0)
        return tuple(centroid.tolist())

    return None

  def get_first_marker_position(self) -> Optional[Tuple[float, float, float]]:
    """
    Get XYZ of first detected marker (simple case: single target).

    Returns:
      (x, y, z) in meters, or None if no markers
    """
    with self.lock:
      if self.latest_markers:
        first = next(iter(self.latest_markers.values()))
        xyz = first["xyz"]
        return tuple(xyz)
    return None

  def get_markers_in_region(
    self, center: Tuple[float, float, float], radius: float
  ) -> List[Tuple[str, Tuple[float, float, float]]]:
    """
    Find all markers within spherical region.

    Args:
      center: (x, y, z) region center
      radius: Sphere radius in meters

    Returns:
      List of (marker_id, (x, y, z))
    """
    import numpy as np

    with self.lock:
      result = []
      center = np.array(center)
      for mid, data in self.latest_markers.items():
        xyz = np.array(data["xyz"])
        dist = np.linalg.norm(xyz - center)
        if dist <= radius:
          result.append((mid, tuple(xyz)))
      return result

  def get_stats(self) -> Dict:
    """Return connection statistics."""
    with self.lock:
      return {
        "connected": self.is_connected(),
        "marker_count": len(self.latest_markers),
        "frame_count": self.frame_count,
        "last_update": self.last_update_time,
      }


# ============================================================================
# Simple demo/test
# ============================================================================
if __name__ == "__main__":
  print("=" * 70)
  print("XINYANG Motion Capture Reader Demo")
  print("=" * 70)

  reader = XINYINGReader("10.1.1.198")

  if not reader.start(timeout=5.0):
    print("Failed to connect to XINYANG")
    sys.exit(1)

  print("\nCapturing data for 15 seconds...")
  print("-" * 70)

  try:
    for i in range(150):
      time.sleep(0.1)

      if i % 10 == 0:  # Print every 1 second
        stats = reader.get_stats()
        print(
          f"\n[{i*0.1:5.1f}s] Markers: {stats['marker_count']:2d} | "
          f"Frames: {stats['frame_count']:4d}"
        )

        # Show first marker
        first_xyz = reader.get_first_marker_position()
        if first_xyz:
          x, y, z = first_xyz
          print(f"  First marker: X={x:.3f} Y={y:.3f} Z={z:.3f} m")

  except KeyboardInterrupt:
    print("\nInterrupted by user")
  finally:
    reader.stop()
    print("\n✓ Reader stopped")
