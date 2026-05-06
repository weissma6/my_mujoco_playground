"""
Test script to verify XINYING motion capture connectivity and marker detection.

This script:
1. Tests connection to XINYING (Nokov) server
2. Captures marker data in real-time
3. Extracts world coordinates (XYZ) for target detection
4. Demonstrates frequency-independent polling

Run with:
  python motion_capture/test_mocap_connection.py

If connection fails, check:
  - XINYING server IP (configured in SERVER_IP)
  - Network adapter: sudo ip addr add 10.1.1.51/24 dev <adapter>
  - ping 10.1.1.198
"""

import sys
import time
from threading import Lock
from collections import deque
from pathlib import Path

# Add XINYING SDK to path
sdk_path = Path(__file__).parent / "XING_Python_SDK-4.1.0.5873" / "dist" / "nokovpy-3.0.1-py3.9"
sys.path.insert(0, str(sdk_path))

try:
  from nokov.nokovsdk import PySDKClient
except ImportError as e:
  print(f"ERROR: Could not import XINYING SDK. Check path: {sdk_path}")
  print(f"Error: {e}")
  sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================
SERVER_IP = "10.1.1.198"  # XINYING server IP
MARKER_HISTORY_SIZE = 5   # Keep last N frames for averaging
INVALID_MARKER_THRESHOLD = 9000000.0


# ============================================================================
# Global State (Thread-Safe)
# ============================================================================
latest_markers = {}  # {marker_id: {"xyz": [x, y, z], "timestamp": t}}
markers_lock = Lock()
frame_count = 0
connection_time = None


# ============================================================================
# Callback: Receives data from XINYING at ~60Hz
# ============================================================================
def mocap_data_callback(pFrameOfMocapData, pUserData):
  """
  Called by XINYING SDK when new frame arrives (~60Hz).
  Extracts all markers, stores in thread-safe dict.
  """
  global latest_markers, frame_count, connection_time

  if pFrameOfMocapData is None:
    return

  frameData = pFrameOfMocapData.contents
  current_time = time.time()
  frame_count += 1

  # Extract all markers (labeled + unlabeled)
  all_markers = {}

  # Labeled markers
  for iMarkerSet in range(frameData.nMarkerSets):
    markerset = frameData.MocapData[iMarkerSet]
    for iMarker in range(markerset.nMarkers):
      x, y, z = markerset.Markers[iMarker]
      if x < INVALID_MARKER_THRESHOLD:
        marker_id = f"labeled_{iMarkerSet}_{iMarker}"
        all_markers[marker_id] = {
          "xyz": [x / 1000.0, y / 1000.0, z / 1000.0],  # mm → m
          "timestamp": current_time,
          "frame": frameData.iFrame,
        }

  # Unlabeled markers
  for iOther in range(frameData.nOtherMarkers):
    x, y, z = frameData.OtherMarkers[iOther]
    if x < INVALID_MARKER_THRESHOLD:
      marker_id = f"unlabeled_{iOther}"
      all_markers[marker_id] = {
        "xyz": [x / 1000.0, y / 1000.0, z / 1000.0],
        "timestamp": current_time,
        "frame": frameData.iFrame,
      }

  # Update global state (thread-safe)
  with markers_lock:
    latest_markers = all_markers
    if connection_time is None:
      connection_time = current_time

  # Print progress
  if frame_count % 60 == 0:  # Print every ~1 second at 60Hz
    marker_count = len(all_markers)
    elapsed = current_time - connection_time if connection_time else 0
    print(f"[{elapsed:6.1f}s] Frame #{frameData.iFrame} | "
          f"{marker_count} markers detected")


# ============================================================================
# Public API: Frequency-independent marker access
# ============================================================================
def get_latest_markers():
  """
  Thread-safe read of latest marker positions.
  Returns dict: {marker_id: {"xyz": [x, y, z], "timestamp": t, "frame": n}}
  """
  with markers_lock:
    return dict(latest_markers)  # Return copy to avoid race conditions


def get_marker_by_id(marker_id):
  """
  Get specific marker position. Returns {"xyz": [...], "timestamp": t} or None.
  """
  with markers_lock:
    if marker_id in latest_markers:
      return dict(latest_markers[marker_id])
  return None


def get_marker_centroid(marker_ids):
  """
  Compute centroid of multiple markers (useful for rigid bodies).
  marker_ids: list of marker IDs
  Returns: {"xyz": [x, y, z], "timestamp": t} or None if any marker missing
  """
  with markers_lock:
    positions = []
    timestamps = []
    for mid in marker_ids:
      if mid in latest_markers:
        positions.append(latest_markers[mid]["xyz"])
        timestamps.append(latest_markers[mid]["timestamp"])
      else:
        return None  # Missing marker
    if positions:
      import numpy as np
      centroid = np.mean(positions, axis=0).tolist()
      avg_time = sum(timestamps) / len(timestamps)
      return {"xyz": centroid, "timestamp": avg_time}
  return None


def get_marker_count():
  """Return number of currently tracked markers."""
  with markers_lock:
    return len(latest_markers)


# ============================================================================
# Main: Test and Demo
# ============================================================================
def main():
  print("=" * 70)
  print("XINYING Motion Capture Connection Test")
  print("=" * 70)
  print(f"Server IP: {SERVER_IP}")
  print(f"Invalid marker threshold: {INVALID_MARKER_THRESHOLD}")
  print()

  # Initialize SDK
  print("Initializing XINYING SDK...")
  client = PySDKClient()
  ret = client.Initialize(bytes(SERVER_IP, encoding="utf8"))

  if ret != 0:
    print(f"❌ Connection Failed (error code: {ret})")
    print("\nTroubleshooting:")
    print("  1. Check server IP is correct: 10.1.1.198")
    print("  2. Configure network adapter (macOS/Linux):")
    print("     sudo ip addr add 10.1.1.51/24 dev enx00e04c449bd3")
    print("  3. Test connectivity:")
    print("     ping 10.1.1.198")
    return False

  print("✓ Connected to XINYANG server")

  # Register callback
  client.PySetDataCallback(mocap_data_callback, None)
  print("✓ Callback registered, waiting for marker data...\n")

  # Capture data for demo
  try:
    print("Running for 10 seconds. Move objects in front of cameras...")
    print("-" * 70)

    for i in range(100):
      time.sleep(0.1)

      # Demonstrate frequency-independent polling
      marker_count = get_marker_count()
      if marker_count > 0:
        markers = get_latest_markers()

        if i % 10 == 0:  # Print every 1 second
          print(f"\n[{i*0.1:.1f}s] {marker_count} markers detected:")

          # Show first 3 markers
          for j, (mid, data) in enumerate(list(markers.items())[:3]):
            x, y, z = data["xyz"]
            print(f"  {mid:20s} → X:{x:7.3f} Y:{y:7.3f} Z:{z:7.3f} m")

          # Example: Get centroid of first N markers (simulating rigid body)
          if marker_count >= 3:
            marker_ids = list(markers.keys())[:3]
            centroid = get_marker_centroid(marker_ids)
            if centroid:
              x, y, z = centroid["xyz"]
              print(f"  [Centroid]           → X:{x:7.3f} Y:{y:7.3f} Z:{z:7.3f} m")

  except KeyboardInterrupt:
    print("\n\nUser interrupted.")
  except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

  # Cleanup
  finally:
    print("\nCleaning up...")
    client.Uninitialize()
    print("✓ Disconnected from XINYANG server")

  print("\n" + "=" * 70)
  print(f"Test Summary: Captured {frame_count} frames")
  print("=" * 70)
  return True


if __name__ == "__main__":
  success = main()
  sys.exit(0 if success else 1)
