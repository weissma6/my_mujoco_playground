"""Bare-metal smoke test: connect the UR3e arm via the External Control URCap.

No UR3RealRobotPick, no wrapper — just raw ur_rtde + the FLAG_USE_EXT_UR_CAP
flag, to confirm the arm connects on PolyScope X. No motion is commanded.

Before running: on the pendant, an External Control program (Host IP = this PC,
port = UR_CAP_PORT) must be PLAYING, with the robot in Remote Control. The control
connect BLOCKS until that program is running, then times out and raises if not.

    cd robots/UR3e && python smoke_connect_arm.py
"""

import rtde_control
import rtde_receive

ROBOT_IP = "192.168.1.4"   # UR3e on PolyScope X 10.12
UR_CAP_PORT = 50002        # External Control URCap port

# --- 1. Receive interface (no pendant program needed) ---
print(f"[1] connecting rtde_receive to {ROBOT_IP} ...")
recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("    receive connected:", recv.isConnected())
print("    q =", [round(v, 4) for v in recv.getActualQ()])

# --- 2. Control interface via the External Control URCap flag ---
# Positional args: (hostname, frequency, flags, ur_cap_port). frequency=-1.0 = default.
print(f"[2] connecting rtde_control with FLAG_USE_EXT_UR_CAP on port {UR_CAP_PORT} ...")
print("    -> press PLAY on the pendant now (this blocks until the program runs)")
ctrl = rtde_control.RTDEControlInterface(
    ROBOT_IP,
    -1.0,
    rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
    UR_CAP_PORT,
)
print("    control connected:", ctrl.isConnected())
print("    program running: ", ctrl.isProgramRunning())

# --- 3. Teardown ---
ctrl.stopScript()
ctrl.disconnect()
recv.disconnect()
print("[3] disconnected. Arm connect smoke test OK.")
