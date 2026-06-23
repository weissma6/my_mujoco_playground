"""Live UR3e base-frame calibration in one go.

Two joint sweeps + two orientation-fixed probe moves -> base frame {B} (origin
p0, rotation R0) in the mocap world frame, written to base_frame_calibration.json.

Run at the robot: External Control PLAYING on the pendant (connect() blocks until
Play), and `calibrig4` camera-visible across both full sweeps. The step-by-step /
dry-run version is base_calibration_interactive.ipynb.
"""

import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))           # robots/UR3e/calibration
_UR3E = os.path.dirname(_HERE)                               # robots/UR3e
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # repo root
for _p in (_HERE, _UR3E, os.path.join(_ROOT, "motion_capture", "mymocap")):
    sys.path.insert(0, _p)

import base_calibration_dependencies as cal
from ur3_realrobot_dependencies import UR3RealRobotPick
from vrpn_dependencies import VRPNRigidBodyReader

# --- CONFIG ---
ROBOT_IP        = "192.168.1.4"     # UR3e on PolyScope X
UR_CAP_PORT     = 50002             # External Control URCapX go-gate
MOCAP_SERVER_IP = "10.1.1.198"
RIGID_BODY      = "calibrig4"       # tracked rigid body on the arm
# Safe, self-collision-free start (rad), rounded from a measured q and verified
# collision-free in the MuJoCo UR3 model. Each sweep moves ONE joint to an absolute
# target, holding the rest: base -1.5 -> +1.5, then shoulder 0.0 -> -3.0 (base held).
Q_SAFE_START      = np.array([-1.5, 0.0, 0.0, -1.7, 0.0, 0.0])
BASE_SWEEP_TO     = 1.5            # joint 0 (base) sweep end (rad), ~172 deg arc
SHOULDER_SWEEP_TO = -3.0           # joint 1 (shoulder) sweep end (rad), ~172 deg arc
Q_TASK_HOME       = np.array([0.0, -2.0, 1.6, -1.6, -1.5, 0.0])  # final park (XML keyframe)
MOVE_A          = 0.2            # rad/s^2, shared MoveJ acceleration (reduced after a robot warning)
START_V         = 0.6            # rad/s, move-to-position velocity
SWEEP_V         = 0.4            # rad/s, sweep velocity
PROBE_L         = 0.10            # m, orientation-fixed probe length
PROBE_V, PROBE_A = 0.03, 0.1     # m/s, m/s^2
OUT_JSON        = os.path.join(_HERE, "base_frame_calibration.json")
OUT_PNG         = os.path.join(_HERE, "base_frame_calibration.png")


def sweep_joint(robot, reader, joint_idx, q_start, target, hz=30.0, timeout=60.0):
    """Move one joint to `target` (rad) async, logging mocap xyz while it travels."""
    q_goal = np.asarray(q_start, float).copy()
    q_goal[joint_idx] = target
    robot.send_movej(q_goal, a=MOVE_A, v=SWEEP_V)
    pts, t0 = [], time.perf_counter()
    while True:
        q = np.array(robot.receive_feedback()["q"])
        xyz = reader.get_rigid_body_xyz()
        if xyz is not None:
            pts.append(xyz)
        if np.linalg.norm(q - q_goal) < 0.01 or time.perf_counter() - t0 > timeout:
            break
        time.sleep(1.0 / hz)
    return np.array(pts)


def probe_axis(robot, reader, axis_idx, settle=0.6):
    """Orientation-fixed +axis move of PROBE_L; return the mocap displacement."""
    r = robot.connect()
    p_before = reader.get_rigid_body_xyz().copy()
    pose0 = r.getActualTCPPose()              # RAW base pose (not the negated obs frame)
    pose1 = list(pose0)
    pose1[axis_idx] += PROBE_L
    robot._control.moveL(pose1, PROBE_V, PROBE_A)
    time.sleep(settle)
    p_after = reader.get_rigid_body_xyz().copy()
    robot._control.moveL(list(pose0), PROBE_V, PROBE_A)
    return p_after - p_before


def plot_calibration(pts1, pts2, p0, tcp_world, mocap_world, out_png):
    """3D plot: sweep trajectories, fitted axes, shoulder, p0, and the
    RTDE TCP vs mocap calibrig4 (both in the robot world frame) + their gap."""
    c1f, a1f, _, _ = cal.fit_circle_3d(pts1)
    c2f, a2f, _, _ = cal.fit_circle_3d(pts2)
    shoulder, _, gap = cal.closest_point_two_lines(c1f, a1f, c2f, a2f)
    dist_mm = np.linalg.norm(tcp_world - mocap_world) * 1e3

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*pts1.T, s=8, label="joint-1 sweep")
    ax.scatter(*pts2.T, s=8, label="joint-2 sweep")
    for c, a, col in [(c1f, a1f, "C0"), (c2f, a2f, "C1")]:
        seg = np.stack([c - 0.3 * a, c + 0.3 * a])
        ax.plot(*seg.T, col, lw=2)
    ax.scatter(*shoulder, c="k", s=60, marker="x",
               label=f"shoulder (gap {gap * 1e3:.2f} mm)")
    ax.scatter(*p0, c="r", s=90, marker="*", label="p0 (base origin)")
    ax.scatter(*tcp_world, c="m", s=70, marker="^", label="Robot RTDE TCP")
    ax.scatter(*mocap_world, c="g", s=70, marker="o", label="mocap calibrig4")
    gap_seg = np.stack([tcp_world, mocap_world])
    ax.plot(*gap_seg.T, "k--", lw=1.5, label=f"|TCP - body| = {dist_mm:.1f} mm")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(fontsize=8)
    ax.set_title("UR3e base-frame calibration (robot world frame)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)
    plt.show()


def print_axis_quality(R0, dp_x, dp_y, res):
    """Report how far the recovered base axes are from ideal.

    The probes measure base +X and +Y directly; ideally those two directions are
    orthogonal (90 deg) and their cross product (the base Z) matches the joint-1
    rotation axis from the sweep (0 deg). The deviation from 90 deg is exactly the
    correction Gram-Schmidt applied to make R0 orthonormal.
    """
    x = dp_x / np.linalg.norm(dp_x)
    y = dp_y / np.linalg.norm(dp_y)
    ang_xy = np.degrees(np.arccos(np.clip(x @ y, -1.0, 1.0)))   # measured X^Y angle
    ortho_err = ang_xy - 90.0                                   # ideal 0
    z_vs_axis = res["z0_vs_a1_deg"]                             # probe Z vs joint-1 axis, ideal 0
    orthonormality = np.linalg.norm(R0.T @ R0 - np.eye(3))      # ideal 0
    det = np.linalg.det(R0)                                     # ideal +1
    print("Axis calibration error (recovered vs ideal):")
    print(f"  base X^Y angle    = {ang_xy:7.3f} deg  (ideal 90.000 -> err {ortho_err:+.3f} deg)")
    print(f"  base Z vs joint-1 = {z_vs_axis:7.3f} deg  (ideal  0.000)")
    print(f"  R0 orthonormality = {orthonormality:.2e}      (ideal 0; det = {det:.4f}, ideal 1)")


def main():
    reader = VRPNRigidBodyReader(MOCAP_SERVER_IP, rigid_body_name=RIGID_BODY,
                                 names=[RIGID_BODY])
    assert reader.start(timeout=8.0) and reader.wait_for_data(timeout=5.0), \
        f"no VRPN reports for '{RIGID_BODY}'"
    assert np.linalg.norm(reader.get_rigid_body_xyz()) < 10.0, "xyz looks like mm"

    robot = UR3RealRobotPick(host=ROBOT_IP, use_ext_urcap=True, ur_cap_port=UR_CAP_PORT)
    print("Press PLAY on the pendant (External Control) to begin ...")
    robot.connect()

    try:
        robot.send_movej(Q_SAFE_START, a=MOVE_A, v=START_V, asynchronous=False)
        assert reader.get_rigid_body_xyz() is not None, "calibrig4 not visible"
        pts1 = sweep_joint(robot, reader, 0, Q_SAFE_START, BASE_SWEEP_TO)

        # Joint-2 sweep with the base held at the end of the joint-1 sweep (+1.5).
        q2_base = Q_SAFE_START.copy()
        q2_base[0] = BASE_SWEEP_TO
        robot.send_movej(q2_base, a=MOVE_A, v=START_V, asynchronous=False)
        pts2 = sweep_joint(robot, reader, 1, q2_base, SHOULDER_SWEEP_TO)

        # Blocking move to Q_TASK_HOME (elbow bent -> non-singular) so the arm is
        # fully stopped before the Cartesian probes. This is also the final rest
        # pose: probe_axis returns here after each probe, so it is the only
        # end-of-run go-to move.
        robot.send_movej(Q_TASK_HOME, a=MOVE_A, v=START_V, asynchronous=False)
        dp_x = probe_axis(robot, reader, 0)
        dp_y = probe_axis(robot, reader, 1)

        # Solve now (robot still up) so we can cross-check the calibration live.
        p0, R0, res = cal.solve(pts1, pts2, dp_x, dp_y, PROBE_L, cal.UR3E_D1)

        # TCP cross-check: map the RTDE TCP (raw base frame) into the mocap world
        # via the fresh calibration and compare to the tracked body. The residual is
        # the fixed calibrig4->TCP mount offset (NOT zero), but should be small and
        # stable if the calibration is good.
        tcp_base = np.array(robot.connect().getActualTCPPose()[:3])
        tcp_world = p0 + R0 @ tcp_base
        mocap_world = reader.get_rigid_body_xyz()
        print("TCP cross-check (mocap world frame):")
        print(f"  RTDE TCP  -> world = {np.round(tcp_world, 4)} m")
        print(f"  mocap calibrig4    = {np.round(mocap_world, 4)} m")
        print(f"  |TCP - body| = {np.linalg.norm(tcp_world - mocap_world) * 1e3:.1f} mm"
              "  (= calibrig4->TCP mount offset)")
        # Already parked at Q_TASK_HOME (probe_axis returned the arm there).
    finally:
        robot.disconnect()
        reader.stop()

    out = cal.build_output_dict(
        p0, R0, res, len(pts1), len(pts2), RIGID_BODY, cal.UR3E_D1,
        datetime.now().isoformat(timespec="seconds"))
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"p0 = {np.round(p0, 4)} m | axis_gap = {res['axis_gap_h_mm']:.2f} mm")
    print("wrote", OUT_JSON)

    print_axis_quality(R0, dp_x, dp_y, res)
    plot_calibration(pts1, pts2, p0, tcp_world, mocap_world, OUT_PNG)


if __name__ == "__main__":
    main()
