"""READ-ONLY diagnostic: compare the live TCP against the live mocap box pose.

Answers "is there a constant offset between where the policy thinks the cube is
and where the gripper actually is". Physically place the gripper so the fingers
straddle the cube centre, then run this. Nothing is commanded -- it opens ONLY
rtde_receive.RTDEReceiveInterface (there is no RTDEControlInterface anywhere in
this file) plus the VRPN reader, samples for a few seconds, and prints.

Reports, all in the policy/base frame the obs is built in:
  * FK TCP        -- compute_tcp_pos(q), the MuJoCo "tcp" site. THIS is what
                     the obs uses (USE_FK_TCP=True in the pick loop).
  * RTDE TCP      -- the controller's own actual_TCP_pose, which depends on the
                     TCP offset configured in PolyScope. A gap between this and
                     the FK TCP is a sim-vs-robot flange/TCP definition
                     mismatch and would bias every rollout identically.
  * mocap box     -- get_rigid_body_xyz() through mocap_pos_to_base(), i.e.
                     exactly the number that reaches the obs.
  * box - FK TCP  -- the vector the policy sees as "where is the cube from
                     here". If the gripper is really at the cube centre this
                     should be ~0; whatever it reads instead IS the bias.

Usage:  .venv/bin/python robots/UR3e/probe_tcp_vs_mocap.py [seconds]
"""

import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../.."))

import rtde_receive  # noqa: E402

from ur3_realrobot_dependencies import UR3RealRobotPick  # noqa: E402
from motion_capture.mymocap.vrpn_dependencies import VRPNRigidBodyReader  # noqa: E402

ROBOT_IP = "192.168.1.4"
MOCAP_SERVER_IP = "10.1.1.198"
MOCAP_RIGID_BODY_NAME = "CubeInCube2"
MODEL_PATH = os.path.join(
    SCRIPT_DIR, "../../mujoco_playground/_src/manipulation/my_ur3/xmls/"
    "mjx_single_cube_position_ur3.xml")

# Sim geometry, for interpreting the numbers (half-extents 0.015/0.015/0.02).
BOX_HALF = np.array([0.015, 0.015, 0.020])

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0


def main():
    print("READ-ONLY probe: rtde_receive + VRPN. No motion is commanded.\n")

    # --- FK + calibration (no robot connection involved) --------------------
    robot = UR3RealRobotPick(host=ROBOT_IP)          # __init__ does NOT connect
    robot.init_fk_model(MODEL_PATH)
    has_cal = robot.load_base_calibration()
    print(f"base-frame calibration loaded: {has_cal}\n")

    # --- Mocap --------------------------------------------------------------
    mocap = VRPNRigidBodyReader(
        server_ip=MOCAP_SERVER_IP, rigid_body_name=MOCAP_RIGID_BODY_NAME)
    if not mocap.start(timeout=8.0):
        raise SystemExit("FATAL: could not connect to the mocap server.")
    if not mocap.wait_for_data(timeout=5.0):
        raise SystemExit(
            f"FATAL: connected, but no data for '{MOCAP_RIGID_BODY_NAME}'. "
            "Is the rigid body in view?")

    # --- RTDE receive ONLY --------------------------------------------------
    rcv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    samples = []
    t0 = time.time()
    while time.time() - t0 < DURATION_S:
        q = np.asarray(rcv.getActualQ(), dtype=float)
        rtde_tcp = np.asarray(rcv.getActualTCPPose(), dtype=float)[:3]
        box_world = mocap.get_rigid_body_xyz()
        if box_world is None:
            time.sleep(0.02)
            continue
        fk_tcp = robot.compute_tcp_pos(q)
        box_base = np.asarray(robot.mocap_pos_to_base(box_world), dtype=float)
        samples.append((q, fk_tcp, rtde_tcp, box_base, np.asarray(box_world)))
        time.sleep(0.02)

    mocap.stop()
    try:
        rcv.disconnect()
    except Exception:
        pass

    if not samples:
        raise SystemExit("FATAL: no samples captured.")

    n = len(samples)
    q = np.mean([s[0] for s in samples], axis=0)
    fk = np.mean([s[1] for s in samples], axis=0)
    rt = np.mean([s[2] for s in samples], axis=0)
    bb = np.mean([s[3] for s in samples], axis=0)
    bw = np.mean([s[4] for s in samples], axis=0)
    jit = np.std([s[3] for s in samples], axis=0)

    f = lambda v: "[" + ", ".join(f"{x:+8.4f}" for x in v) + "]"
    print(f"samples: {n} over {DURATION_S:.1f}s\n")
    print(f"  joint q (rad)      {f(q)}")
    print(f"  mocap box (world)  {f(bw)}")
    print(f"  mocap box (BASE)   {f(bb)}   <- what the obs sees")
    print(f"  mocap jitter (mm)  {f(jit * 1000)}")
    print(f"  FK  TCP   (BASE)   {f(fk)}   <- what the obs sees")
    print(f"  RTDE TCP  (BASE)   {f(rt)}")
    print()

    d_fk_rtde = fk - rt
    print(f"  FK TCP - RTDE TCP  {f(d_fk_rtde)}  |d| = "
          f"{1000 * np.linalg.norm(d_fk_rtde):6.1f} mm")
    print("      (sim tcp site vs the controller's configured TCP; a large")
    print("       constant here biases every rollout the same way)")
    print()

    d = bb - fk
    print(f"  >>> box - FK TCP   {f(d)}  |d| = {1000 * np.linalg.norm(d):6.1f} mm")
    print("      This is the policy's 'where is the cube' vector. With the")
    print("      fingers straddling the cube centre it should be ~0.")
    print()
    for ax, val in zip("xyz", d):
        bar = "within" if abs(val) <= BOX_HALF["xyz".index(ax)] else "OUTSIDE"
        print(f"      {ax}: {1000 * val:+7.1f} mm   "
              f"(cube half-extent {1000 * BOX_HALF['xyz'.index(ax)]:.0f} mm -> {bar})")


if __name__ == "__main__":
    main()
