"""Minimal Nokov mocap connection test: connect, grab one frame, print, exit.

Imports `py_data_func` from the upstream example at
  motion_capture/nokov_python_sdk-master/examples/Nokov_SDK_Client.py
and calls it on a single frame fetched via `PyGetLastFrameOfMocapData()`.
No streaming loop, no duplicated print logic.

The SDK itself is the pip-installed `nokovpy` wheel from
  motion_capture/nokov_python_sdk-master/dist/nokovpy-3.0.1-py3-none-any.whl

Usage:
  python motion_capture/mymocap/test_mocap_connection.py
  python motion_capture/mymocap/test_mocap_connection.py 10.1.1.198
"""

import sys
import time
from pathlib import Path

# Make the upstream example importable as a regular module.
_EXAMPLES = (
    Path(__file__).resolve().parents[1]
    / "nokov_python_sdk-master"
    / "examples"
)
sys.path.insert(0, str(_EXAMPLES))

import Nokov_SDK_Client as nsc  # noqa: E402
from nokov.nokovsdk import PySDKClient  # noqa: E402

SERVER_IP = sys.argv[1] if len(sys.argv) > 1 else "10.1.1.198"

client = PySDKClient()
ret = client.Initialize(bytes(SERVER_IP, "utf8"))
if ret != 0:
  sys.exit(f"FAIL: Initialize({SERVER_IP}) returned {ret}")
print(f"OK: connected to {SERVER_IP}")

# py_data_func reads `client` as a module-level global for PyTimecodeStringify.
nsc.client = client

# PyGetLastFrameOfMocapData() returns None until the server has pushed a frame.
frame = None
for _ in range(50):
  frame = client.PyGetLastFrameOfMocapData()
  if frame:
    break
  time.sleep(0.1)

if not frame:
  sys.exit("FAIL: no frame received within 5 s")

try:
  nsc.py_data_func(frame, client)
finally:
  client.PyNokovFreeFrame(frame)
