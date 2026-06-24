"""Live UR3e base-frame calibration in one go.

Two joint sweeps + two orientation-fixed probe moves -> base frame {B} (origin
p0, rotation R0) in the mocap world frame, written to base_frame_calibration.json.

Run at the robot: External Control PLAYING on the pendant (connect() blocks until
Play), and `calibrig4` camera-visible across both full sweeps. The step-by-step /
dry-run version is base_calibration_interactive.ipynb.
"""

import csv
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
# Second sweep set: independent base-orientation check from the LAST joint
# (wrist 3). Its rotation axis is the tool-flange Z, which we also read in the
# base frame from the RTDE flange orientation -> a probe-independent R0 check.
WRIST_JOINT         = 5          # last joint (wrist 3)
WRIST_SWEEP_TO      = 2.0        # rad, joint-5 sweep end (from Q_TASK_HOME[5]=0)
MIN_WRIST_RADIUS_MM = 10.0       # below this the marker is ~on the wrist axis -> skip
ORIENT_WARN_DEG     = 3.0        # warn if the independent check disagrees with R0
# Independent forward-TCP sanity check: sweep wrist 2 (joint 4) as well, intersect
# its axis with the wrist-3 axis -> wrist center, push L_NOMINAL along tool Z ->
# tcp_forward (carries NO p0/R0). Compare to the RTDE-based tcp_world.
WRIST2_JOINT        = 4          # wrist 2
WRIST2_SWEEP_TO     = 0.0        # rad, joint-4 sweep end (from Q_TASK_HOME[4]=-1.5, ~1.5 rad arc)
                                 # TODO-verify collision-free in the MuJoCo UR3 model and on the
                                 # first real run at reduced speed (cf. how Q_SAFE_START was set).
TCP_Z_OFFSET        = 0.0        # m, PolyScope-configured tool Z offset -- MUST match the pendant
                                 # TCP (flange TCP here -> 0.0; getActualTCPPose includes the TCP).
L_NOMINAL           = cal.UR3E_D6 + TCP_Z_OFFSET  # m, wrist-center -> TCP nominal distance
OUT_JSON        = os.path.join(_HERE, "base_frame_calibration.json")
OUT_PNG         = os.path.join(_HERE, "base_frame_calibration.png")
ACCURACY_DIR    = os.path.join(_HERE, "calibration_accuracy")     # repeatability tooling + reports
OUT_LOG         = os.path.join(ACCURACY_DIR, "calibration_history.csv")  # appended per run


def sweep_joint(robot, reader, joint_idx, q_start, target, move_a, sweep_v,
                hz=30.0, timeout=60.0):
    """Move one joint to `target` (rad) async, logging mocap xyz while it travels."""
    q_goal = np.asarray(q_start, float).copy()
    q_goal[joint_idx] = target
    robot.send_movej(q_goal, a=move_a, v=sweep_v)
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


def probe_axis(robot, reader, axis_idx, probe_v, probe_a, settle=0.6):
    """Orientation-fixed +axis move of PROBE_L; return the mocap displacement."""
    r = robot.connect()
    p_before = reader.get_rigid_body_xyz().copy()
    pose0 = r.getActualTCPPose()              # RAW base pose (not the negated obs frame)
    pose1 = list(pose0)
    pose1[axis_idx] += PROBE_L
    robot._control.moveL(pose1, probe_v, probe_a)
    time.sleep(settle)
    p_after = reader.get_rigid_body_xyz().copy()
    robot._control.moveL(list(pose0), probe_v, probe_a)
    return p_after - p_before


def wrist_orientation_check(robot, reader, R0, move_a, sweep_v, start_v):
    """FK-free base-orientation check from the last joint (wrist 3).

    The wrist-3 rotation axis IS the tool-flange Z. We measure it in the world
    frame (circle fit of the marker during a wrist-3 sweep) and also read it in
    the base frame from the RTDE flange orientation. A correct R0 maps one to the
    other, so the angle between (R0 @ flange_Z_base) and the measured flange_Z in
    world is an independent orientation error (deg) using data the probes never
    saw. Returns a dict (or with radius below MIN_WRIST_RADIUS_MM, still returns
    it so the caller can warn/skip). Leaves the arm back at Q_TASK_HOME.
    """
    r = robot.connect()
    try:
        flange_pose = r.getActualToolFlangePose()  # raw base frame [xyz, rotvec]
    except AttributeError:
        flange_pose = r.getActualTCPPose()         # fallback (assumes no rot TCP offset)
    fz_base = cal.rotvec_to_matrix(flange_pose[3:6])[:, 2]
    predicted_z = R0 @ fz_base

    pts_w = sweep_joint(robot, reader, WRIST_JOINT, Q_TASK_HOME, WRIST_SWEEP_TO,
                        move_a, sweep_v)
    robot.send_movej(Q_TASK_HOME, a=move_a, v=start_v, asynchronous=False)

    center, axis, radius, rms = cal.fit_circle_3d(pts_w)
    measured_z = axis / np.linalg.norm(axis)
    if measured_z @ predicted_z < 0.0:            # circle-fit axis sign is arbitrary
        measured_z = -measured_z
    error_deg = float(np.degrees(np.arccos(np.clip(measured_z @ predicted_z, -1.0, 1.0))))
    return {
        "pts": pts_w, "center": center, "measured_z": measured_z,
        "predicted_z": predicted_z, "error_deg": error_deg,
        "radius_mm": radius * 1e3, "rms_mm": rms * 1e3, "n": len(pts_w),
    }


def forward_tcp_check(robot, reader, wrist, tcp_world, move_a, sweep_v, start_v):
    """Independent forward-TCP estimate from the last two joint axes.

    Sweeps wrist 2 (joint 4), reusing the wrist-3 axis already fitted in `wrist`
    (the joint-5 sweep). Intersect the two axis lines -> wrist center; push
    L_NOMINAL along the measured tool Z -> tcp_forward. This uses ONLY mocap
    circle fits (no p0/R0), so |tcp_world - tcp_forward| is an outside-check on
    the base calibration. Leaves the arm back at Q_TASK_HOME. Returns a dict, or
    None if either wrist arc is too small to fit a reliable axis.
    """
    if wrist is None or wrist["radius_mm"] < MIN_WRIST_RADIUS_MM:
        print("TCP forward-estimate sanity check:")
        print("  SKIPPED: wrist-3 axis unusable (joint-5 arc too small) -- "
              "cannot form the wrist center.")
        return None

    pts4 = sweep_joint(robot, reader, WRIST2_JOINT, Q_TASK_HOME, WRIST2_SWEEP_TO,
                       move_a, sweep_v)
    robot.send_movej(Q_TASK_HOME, a=move_a, v=start_v, asynchronous=False)

    c4, a4, r4, rms4 = cal.fit_circle_3d(pts4)
    if r4 * 1e3 < MIN_WRIST_RADIUS_MM:
        print("TCP forward-estimate sanity check:")
        print(f"  SKIPPED: wrist-2 circle radius {r4 * 1e3:.1f} mm < "
              f"{MIN_WRIST_RADIUS_MM:.0f} mm -- marker too close to the wrist-2 axis.")
        return None

    tcp_forward, wrist_center, axis_gap = cal.forward_tcp(
        c4, a4, wrist["center"], wrist["measured_z"], L_NOMINAL,
        ref_dir=tcp_world - 0.5 * (c4 + wrist["center"]))
    return {
        "pts4": pts4, "c4": c4, "a4": a4,
        "r4_mm": r4 * 1e3, "rms4_mm": rms4 * 1e3, "axis_gap_mm": axis_gap * 1e3,
        "wrist_center": wrist_center, "tcp_forward": tcp_forward,
    }


def append_run_record(log_path, result):
    """Append one row capturing ALL computed positions of a calibration run to the
    long-lived history CSV under calibration_accuracy/. Header is written on first use;
    None fields (e.g. a skipped forward check) become empty cells. Both single-run
    main() and every batch run append here, so it is an append-only audit log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    res = result["residuals"]
    fwd = result["forward"]

    def xyz(v):
        return [round(float(x), 6) for x in v] if v is not None else ["", "", ""]

    def num(v, nd=3):
        return "" if v is None else round(float(v), nd)

    header = [
        "timestamp", "run_index",
        "p0_x", "p0_y", "p0_z",
        "tcp_world_x", "tcp_world_y", "tcp_world_z",
        "tcp_forward_x", "tcp_forward_y", "tcp_forward_z",
        "wrist_center_x", "wrist_center_y", "wrist_center_z",
        "d_world_mm", "d_forward_mm", "dist_world_to_forward_mm",
        "axis_gap_h_mm", "circle1_rms_mm", "circle2_rms_mm",
        "wrist_axis_gap_mm", "wrist2_rms_mm",
    ]
    row = [
        result["timestamp"],
        "" if result["run_index"] is None else result["run_index"],
        *xyz(result["p0"]),
        *xyz(result["tcp_world"]),
        *xyz(result["tcp_forward"]),
        *xyz(result["wrist_center"]),
        num(result["d_world_mm"], 2),
        num(result["d_forward_mm"], 2),
        num(result["d_resid_mm"], 2),
        num(res.get("axis_gap_h_mm")),
        num(res.get("circle1_rms_mm")),
        num(res.get("circle2_rms_mm")),
        num(fwd["axis_gap_mm"] if fwd is not None else None),
        num(fwd["rms4_mm"] if fwd is not None else None),
    ]
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)


def plot_calibration(pts1, pts2, p0, out_png, wrist=None, forward=None):
    """Decluttered 3D plot. The first two (base, shoulder) sweeps are drawn as
    points + fitted axes + their intersection (shoulder) + p0. The second set of
    sweeps (the wrist sweeps) are reduced to ONLY their fitted axes and their
    intersection (the wrist center) -- no scatter, no orientation/TCP overlays.
    The TCP vs forward-TCP distances live in the CSV history log instead."""
    c1f, a1f, _, _ = cal.fit_circle_3d(pts1)
    c2f, a2f, _, _ = cal.fit_circle_3d(pts2)
    shoulder, _, gap = cal.closest_point_two_lines(c1f, a1f, c2f, a2f)

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
    # Second set of sweeps (wrist 3, wrist 2): only the fitted axes + intersection.
    if wrist is not None:
        c5 = wrist["center"]
        seg = np.stack([c5 - 0.15 * wrist["measured_z"],
                        c5 + 0.15 * wrist["measured_z"]])
        ax.plot(*seg.T, "C3", lw=2, label="wrist-3 axis (tool Z)")
    if forward is not None:
        c4, a4 = forward["c4"], forward["a4"]
        seg = np.stack([c4 - 0.15 * a4, c4 + 0.15 * a4])
        ax.plot(*seg.T, "C5", lw=2, label="wrist-2 axis")
        ax.scatter(*forward["wrist_center"], c="C6", s=70, marker="P",
                   label="wrist center (axis intersection)")
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


def result_to_json(result):
    """Assemble the single-run base_frame_calibration.json payload from a
    run_calibration() result dict (reproduces the original schema: the base frame
    plus optional wrist_validation / tcp_forward_check blocks)."""
    out = cal.build_output_dict(
        result["p0"], result["R0"], result["residuals"],
        result["n_points"]["joint1"], result["n_points"]["joint2"],
        RIGID_BODY, cal.UR3E_D1, result["timestamp"])
    wrist = result["wrist"]
    if wrist is not None:
        out["wrist_validation"] = {
            "flange_z_error_deg": wrist["error_deg"],
            "circle_radius_mm": wrist["radius_mm"],
            "circle_rms_mm": wrist["rms_mm"],
            "n_points": wrist["n"],
        }
    fwd = result["forward"]
    if fwd is not None:
        out["tcp_forward_check"] = {
            "L_nominal_m": L_NOMINAL,
            "tcp_z_offset_m": TCP_Z_OFFSET,
            "wrist_center_m": fwd["wrist_center"].tolist(),
            "tcp_forward_m": fwd["tcp_forward"].tolist(),
            "tcp_world_m": fwd["tcp_world"].tolist(),
            "dist_world_to_body_mm": fwd["d_world_mm"],
            "dist_forward_to_body_mm": fwd["d_forward_mm"],
            "dist_world_to_forward_mm": fwd["d_resid_mm"],
            "wrist2_radius_mm": fwd["r4_mm"],
            "wrist2_rms_mm": fwd["rms4_mm"],
            "wrist_axis_gap_mm": fwd["axis_gap_mm"],
            "n_points_wrist2": len(fwd["pts4"]),
        }
    return out


def run_calibration(
    sweep_v=SWEEP_V, move_a=MOVE_A, start_v=START_V, probe_v=PROBE_V, probe_a=PROBE_A,
    robot=None, reader=None,
    write_json=True, append_hist=True, plot=True,
    verbose=True, run_index=None,
):
    """Run ONE full calibration and return an aggregation-ready result dict.

    Two joint sweeps + two probes -> base frame (p0, R0) via cal.solve, then the
    wrist-3 orientation check and the independent forward-TCP check. Motion speeds
    (sweep_v / move_a / start_v / probe_v / probe_a) are parameters so a batch can
    slow the arm down. File I/O is decoupled from compute via the flags:
    write_json -> base_frame_calibration.json, plot -> the PNG, append_hist -> one
    row in calibration_accuracy/calibration_history.csv.

    Pass an already-connected robot and/or reader to reuse one connection across
    runs; whatever is created here is torn down here, whatever is passed in is left
    untouched (the caller owns its lifecycle).

    Returns a dict with: ok, run_index, timestamp, p0, R0, quat_wxyz, tcp_world,
    tcp_forward, wrist_center, mocap_world, d_world_mm, d_forward_mm, d_resid_mm,
    residuals, wrist, forward, n_points.
    """
    def say(*a):
        if verbose:
            print(*a)

    owns_reader = reader is None
    owns_robot = robot is None
    if owns_reader:
        reader = VRPNRigidBodyReader(MOCAP_SERVER_IP, rigid_body_name=RIGID_BODY,
                                     names=[RIGID_BODY])
        assert reader.start(timeout=8.0) and reader.wait_for_data(timeout=5.0), \
            f"no VRPN reports for '{RIGID_BODY}'"
        assert np.linalg.norm(reader.get_rigid_body_xyz()) < 10.0, "xyz looks like mm"
    if owns_robot:
        robot = UR3RealRobotPick(host=ROBOT_IP, use_ext_urcap=True,
                                 ur_cap_port=UR_CAP_PORT)
        say("Press PLAY on the pendant (External Control) to begin ...")
        robot.connect()

    wrist = None
    fwd = None
    try:
        robot.send_movej(Q_SAFE_START, a=move_a, v=start_v, asynchronous=False)
        assert reader.get_rigid_body_xyz() is not None, "calibrig4 not visible"
        pts1 = sweep_joint(robot, reader, 0, Q_SAFE_START, BASE_SWEEP_TO,
                           move_a, sweep_v)

        # Joint-2 sweep with the base held at the end of the joint-1 sweep (+1.5).
        q2_base = Q_SAFE_START.copy()
        q2_base[0] = BASE_SWEEP_TO
        robot.send_movej(q2_base, a=move_a, v=start_v, asynchronous=False)
        pts2 = sweep_joint(robot, reader, 1, q2_base, SHOULDER_SWEEP_TO,
                           move_a, sweep_v)

        # Blocking move to Q_TASK_HOME (elbow bent -> non-singular) so the arm is
        # fully stopped before the Cartesian probes. This is also the final rest
        # pose: probe_axis returns here after each probe, so it is the only
        # end-of-run go-to move.
        robot.send_movej(Q_TASK_HOME, a=move_a, v=start_v, asynchronous=False)
        dp_x = probe_axis(robot, reader, 0, probe_v, probe_a)
        dp_y = probe_axis(robot, reader, 1, probe_v, probe_a)

        # Solve now (robot still up) so we can cross-check the calibration live.
        p0, R0, res = cal.solve(pts1, pts2, dp_x, dp_y, PROBE_L, cal.UR3E_D1)

        # TCP cross-check: map the RTDE TCP (raw base frame) into the mocap world
        # via the fresh calibration and compare to the tracked body. The residual is
        # the fixed calibrig4->TCP mount offset (NOT zero), but should be small and
        # stable if the calibration is good.
        tcp_base = np.array(robot.connect().getActualTCPPose()[:3])
        tcp_world = p0 + R0 @ tcp_base
        mocap_world = reader.get_rigid_body_xyz()
        say("TCP cross-check #1 (after probes, at Q_TASK_HOME):")
        say(f"  RTDE TCP  -> world = {np.round(tcp_world, 4)} m")
        say(f"  mocap calibrig4    = {np.round(mocap_world, 4)} m")
        say(f"  |TCP - body| = {np.linalg.norm(tcp_world - mocap_world) * 1e3:.1f} mm"
            "  (= calibrig4->TCP mount offset)")

        # --- Second sweep set: independent orientation check from the last joint.
        # Sweep wrist 3, return to Q_TASK_HOME, then repeat the checks.
        wrist = wrist_orientation_check(robot, reader, R0, move_a, sweep_v, start_v)
        tcp_world = p0 + R0 @ np.array(robot.connect().getActualTCPPose()[:3])
        mocap_world = reader.get_rigid_body_xyz()
        say("TCP cross-check #2 (after wrist sweep, back at Q_TASK_HOME):")
        say(f"  |TCP - body| = {np.linalg.norm(tcp_world - mocap_world) * 1e3:.1f} mm"
            "  (should match #1 -> pose repeatability)")
        say("Wrist-3 orientation check (independent of the probes):")
        if wrist["radius_mm"] < MIN_WRIST_RADIUS_MM:
            say(f"  SKIPPED: circle radius {wrist['radius_mm']:.1f} mm < "
                f"{MIN_WRIST_RADIUS_MM:.0f} mm -- marker too close to / proximal of "
                "the wrist-3 axis (no usable arc).")
            wrist = None
        else:
            say(f"  flange-Z error = {wrist['error_deg']:.3f} deg  "
                "(ideal 0; measured-in-world vs R0-predicted)")
            say(f"  circle radius = {wrist['radius_mm']:.1f} mm, "
                f"rms = {wrist['rms_mm']:.2f} mm, n = {wrist['n']}")
            if wrist["error_deg"] > ORIENT_WARN_DEG:
                say(f"  WARNING: independent check disagrees with the probe-based R0 by "
                    f"{wrist['error_deg']:.2f} deg (> {ORIENT_WARN_DEG:.0f} deg).")
                say("    Likely: rotational TCP offset, marker near the wrist axis, "
                    "near-singular wrist pose, or occlusion. Re-run, or use the "
                    "FK-based both-wrist option to refine R0.")

        # --- Third check: independent forward-TCP estimate from the last two
        # joint axes + nominal tool length (carries NO p0/R0). Sweeps wrist 2,
        # reuses the wrist-3 axis, returns to Q_TASK_HOME.
        fwd = forward_tcp_check(robot, reader, wrist, tcp_world,
                                move_a, sweep_v, start_v)
        if fwd is not None:
            mocap_world = reader.get_rigid_body_xyz()
            fwd["d_world_mm"] = np.linalg.norm(tcp_world - mocap_world) * 1e3
            fwd["d_forward_mm"] = np.linalg.norm(fwd["tcp_forward"] - mocap_world) * 1e3
            fwd["d_resid_mm"] = np.linalg.norm(tcp_world - fwd["tcp_forward"]) * 1e3
            fwd["tcp_world"] = tcp_world
            say("TCP forward-estimate sanity check (independent of p0/R0):")
            say(f"  |RTDE TCP    - body| = {fwd['d_world_mm']:7.1f} mm  (= mount offset)")
            say(f"  |forward TCP - body| = {fwd['d_forward_mm']:7.1f} mm  (axis + nominal length)")
            say(f"  |RTDE - forward TCP| = {fwd['d_resid_mm']:7.1f} mm  "
                "(<- sanity residual; ideal a few mm)")
            say(f"  wrist axis gap = {fwd['axis_gap_mm']:.2f} mm, wrist-2 radius = "
                f"{fwd['r4_mm']:.1f} mm, rms = {fwd['rms4_mm']:.2f} mm")
            say("  A residual along tool Z -> wrong L_NOMINAL / p0 height; "
                "a rotational residual -> R0 error.")
        # Already parked at Q_TASK_HOME (sweeps/probes returned the arm there).
    finally:
        if owns_robot:
            robot.disconnect()
        if owns_reader:
            reader.stop()

    ts = datetime.now().isoformat(timespec="seconds")
    d_world_mm = (fwd["d_world_mm"] if fwd is not None else
                  (float(np.linalg.norm(tcp_world - mocap_world) * 1e3)
                   if mocap_world is not None else None))
    result = {
        "ok": fwd is not None,
        "run_index": run_index,
        "timestamp": ts,
        "p0": p0,
        "R0": R0,
        "quat_wxyz": cal.rotation_matrix_to_quat_wxyz(R0),
        "tcp_world": tcp_world,
        "tcp_forward": fwd["tcp_forward"] if fwd is not None else None,
        "wrist_center": fwd["wrist_center"] if fwd is not None else None,
        "mocap_world": mocap_world,
        "d_world_mm": d_world_mm,
        "d_forward_mm": fwd["d_forward_mm"] if fwd is not None else None,
        "d_resid_mm": fwd["d_resid_mm"] if fwd is not None else None,
        "residuals": res,
        "wrist": wrist,
        "forward": fwd,
        "n_points": {"joint1": len(pts1), "joint2": len(pts2)},
    }

    if write_json:
        with open(OUT_JSON, "w") as f:
            json.dump(result_to_json(result), f, indent=2)
        say(f"p0 = {np.round(p0, 4)} m | axis_gap = {res['axis_gap_h_mm']:.2f} mm")
        say("wrote", OUT_JSON)
    if append_hist:
        append_run_record(OUT_LOG, result)
        say("appended history row ->", OUT_LOG)
    if verbose:
        print_axis_quality(R0, dp_x, dp_y, res)
    if plot:
        plot_calibration(pts1, pts2, p0, OUT_PNG, wrist=wrist, forward=fwd)
    return result


def main():
    run_calibration()


if __name__ == "__main__":
    main()
