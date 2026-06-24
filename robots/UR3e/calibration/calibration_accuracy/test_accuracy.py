"""UR3e base-frame calibration repeatability harness.

Runs run_calibration() N times over ONE robot+mocap connection, collects the base
origin p0, the RTDE TCP (tcp_world), the geometric forward TCP (tcp_forward) and
the TCP-to-TCP distance per run, then:
  * writes the per-run snapshot + a human-readable summary into this folder
    ("calibration_accuracy/", next to this script),
  * writes the AVERAGED calibration (mean p0 + mean R0) to base_frame_calibration.json
    in the parent calibration/ folder, so the committed calibration is the mean of
    N runs, not one noisy run.

Translation averages linearly; rotation is averaged on SO(3) via SVD
(cal.average_rotations). Fail-soft: one bad run is logged and skipped, the batch
continues. Each run parks the arm at Q_TASK_HOME and the next run's blocking move
to Q_SAFE_START repositions it, so back-to-back runs start cleanly.

Run at the robot with External Control PLAYING (connect() blocks until Play, once):
    cd robots/UR3e/calibration/calibration_accuracy
    python test_accuracy.py --runs 2      # smoke first
    python test_accuracy.py --runs 10     # full repeatability batch
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))           # .../calibration/calibration_accuracy
_CAL = os.path.dirname(_HERE)                                # robots/UR3e/calibration (has base_calibration*)
_UR3E = os.path.dirname(_CAL)                                # robots/UR3e
_ROOT = os.path.abspath(os.path.join(_UR3E, "..", ".."))     # repo root
for _p in (_HERE, _CAL, _UR3E, os.path.join(_ROOT, "motion_capture", "mymocap")):
    sys.path.insert(0, _p)

import base_calibration as bc
import base_calibration_dependencies as cal

N_RUNS = 10  # default batch size (override via --runs)


def pos_stats(arr):
    """Per-axis mean/std/min/max + 3D spread (RMS distance from the mean), meters."""
    mean = arr.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((arr - mean) ** 2, axis=1))))
    return mean, arr.std(axis=0), arr.min(axis=0), arr.max(axis=0), spread


def scalar_stats(a):
    """mean/std/min/max of a 1D array as plain floats."""
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


def axis_table(title, mean, std, mn, mx, spread_mm):
    lines = [f"### {title} (m)", "",
             "| axis | mean | std | min | max |",
             "|---|---|---|---|---|"]
    for i, ax in enumerate("xyz"):
        lines.append(f"| {ax} | {mean[i]:.5f} | {std[i]:.5f} | {mn[i]:.5f} | {mx[i]:.5f} |")
    lines += ["", f"3D spread (RMS distance from mean): **{spread_mm:.3f} mm**", ""]
    return "\n".join(lines)


def run_batch(n):
    """Connect once, run n calibrations, return the list of successful result dicts."""
    reader = bc.VRPNRigidBodyReader(bc.MOCAP_SERVER_IP, rigid_body_name=bc.RIGID_BODY,
                                    names=[bc.RIGID_BODY])
    assert reader.start(timeout=8.0) and reader.wait_for_data(timeout=5.0), \
        f"no VRPN reports for '{bc.RIGID_BODY}'"
    assert np.linalg.norm(reader.get_rigid_body_xyz()) < 10.0, "xyz looks like mm"

    robot = bc.UR3RealRobotPick(host=bc.ROBOT_IP, use_ext_urcap=True,
                                ur_cap_port=bc.UR_CAP_PORT)
    print("Press PLAY on the pendant (External Control) to begin the batch ...")
    robot.connect()

    results = []
    try:
        for i in range(1, n + 1):
            print(f"\n=== calibration run {i}/{n} ===")
            try:
                # Reuse the one connection; don't write the single-run JSON/PNG
                # (the batch writes the averaged JSON at the end). Each run still
                # appends one row to the long-lived history CSV.
                res = bc.run_calibration(
                    robot=robot, reader=reader,
                    write_json=False, append_hist=True, plot=False,
                    verbose=False, run_index=i)
                dr = res["d_resid_mm"]
                dr_s = f"{dr:.1f}" if dr is not None else "n/a"
                print(f"  run {i}: p0={np.round(res['p0'], 4)}  "
                      f"d_resid={dr_s} mm  ok={res['ok']}")
                results.append(res)
            except Exception as e:  # FAIL-SOFT: never let one run abort the batch
                print(f"  run {i} FAILED: {e!r} -- skipping")
    finally:
        robot.disconnect()
        reader.stop()
    return results


def summarize_and_write(results, n_requested):
    """Aggregate the runs, write the snapshot CSV + summary md, and overwrite
    base_frame_calibration.json with the averaged calibration."""
    os.makedirs(bc.ACCURACY_DIR, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    stamp = ts.replace(":", "-")

    # --- Per-run snapshot CSV (reuses the history schema; fresh file -> writes header).
    batch_csv = os.path.join(bc.ACCURACY_DIR, f"accuracy_report_{stamp}.csv")
    for r in results:
        bc.append_run_record(batch_csv, r)

    # --- Histogram figure of that report (hardware-free plotter in this folder).
    from plot_accuracy import load_report, plot_histograms
    hist_png = os.path.join(bc.ACCURACY_DIR, f"accuracy_histograms_{stamp}.png")
    hcols, hn = load_report(batch_csv)
    plot_histograms(hcols, hn, hist_png, source=os.path.basename(batch_csv))

    # --- Aggregate (filter None so forward-skipped runs still feed p0/tcp_world).
    p0_arr = np.array([r["p0"] for r in results])
    tcpw_arr = np.array([r["tcp_world"] for r in results if r["tcp_world"] is not None])
    tcpf_arr = np.array([r["tcp_forward"] for r in results if r["tcp_forward"] is not None])
    dresid = np.array([r["d_resid_mm"] for r in results if r["d_resid_mm"] is not None])

    p0_mean, p0_std, p0_min, p0_max, p0_spread = pos_stats(p0_arr)
    tcpw_mean, tcpw_std, tcpw_min, tcpw_max, tcpw_spread = pos_stats(tcpw_arr)
    has_fwd = len(tcpf_arr) > 0
    if has_fwd:
        tcpf_mean, tcpf_std, tcpf_min, tcpf_max, tcpf_spread = pos_stats(tcpf_arr)

    R0s = [r["R0"] for r in results]
    R0_mean = cal.average_rotations(R0s)
    disp = np.array([cal.rotation_geodesic_deg(R, R0_mean) for R in R0s])

    res_keys = results[0]["residuals"].keys()
    res_mean = {k: float(np.mean([r["residuals"][k] for r in results])) for k in res_keys}
    nj1 = int(round(np.mean([r["n_points"]["joint1"] for r in results])))
    nj2 = int(round(np.mean([r["n_points"]["joint2"] for r in results])))

    # --- Mean calibration -> base_frame_calibration.json (existing keys preserved;
    # averaging block added). Rotation averaged via SVD on SO(3), translation linearly.
    out = cal.build_output_dict(p0_mean, R0_mean, res_mean, nj1, nj2,
                                bc.RIGID_BODY, cal.UR3E_D1, ts)
    out["averaging"] = {
        "n_runs_requested": n_requested,
        "n_runs_succeeded": len(results),
        "n_runs_with_forward": int(len(tcpf_arr)),
        "rotation_average_method": "svd_orthonormalization",
        "p0_std_m": p0_std.tolist(),
        "p0_spread_mm": round(p0_spread * 1e3, 3),
        "tcp_world_std_m": tcpw_std.tolist(),
        "tcp_world_spread_mm": round(tcpw_spread * 1e3, 3),
        "tcp_forward_std_m": tcpf_std.tolist() if has_fwd else None,
        "tcp_forward_spread_mm": round(tcpf_spread * 1e3, 3) if has_fwd else None,
        "d_resid_mm": (dict(zip(("mean", "std", "min", "max"), scalar_stats(dresid)))
                       if len(dresid) else None),
        "rotation_dispersion_deg": {"mean": float(disp.mean()), "max": float(disp.max())},
    }
    with open(bc.OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # --- Human-readable summary (.md).
    parts = [
        f"# UR3e base-frame calibration repeatability -- {ts}", "",
        f"Runs: requested **{n_requested}**, succeeded **{len(results)}**, "
        f"failed **{n_requested - len(results)}**, with forward-TCP **{len(tcpf_arr)}**.",
        "",
        axis_table("Base origin p0", p0_mean, p0_std, p0_min, p0_max, p0_spread * 1e3),
        axis_table("RTDE TCP (tcp_world)", tcpw_mean, tcpw_std, tcpw_min, tcpw_max,
                   tcpw_spread * 1e3),
    ]
    if has_fwd:
        parts.append(axis_table("Forward TCP (tcp_forward)", tcpf_mean, tcpf_std,
                                tcpf_min, tcpf_max, tcpf_spread * 1e3))
    if len(dresid):
        dm, ds, dmn, dmx = scalar_stats(dresid)
        parts += ["### TCP-to-TCP distance |tcp_world - tcp_forward| (mm)", "",
                  f"mean **{dm:.2f}**, std {ds:.2f}, min {dmn:.2f}, max {dmx:.2f}", ""]
    parts += [
        "### Rotation R0 dispersion (deg, geodesic from mean)", "",
        f"mean **{disp.mean():.4f}**, max {disp.max():.4f}", "",
        "### Mean calibration written to base_frame_calibration.json", "",
        f"- p0_mean = {np.round(p0_mean, 5).tolist()} m",
        f"- quat_wxyz = {np.round(out['rotation_quat_wxyz'], 6)}",
        f"- rotation average: {out['averaging']['rotation_average_method']}", "",
    ]
    summary_md = os.path.join(bc.ACCURACY_DIR, f"accuracy_summary_{stamp}.md")
    with open(summary_md, "w") as f:
        f.write("\n".join(parts))

    print("\n--- repeatability summary ---")
    print(f"  p0 3D spread        = {p0_spread * 1e3:.3f} mm")
    print(f"  tcp_world 3D spread = {tcpw_spread * 1e3:.3f} mm")
    if has_fwd:
        print(f"  tcp_forward spread  = {tcpf_spread * 1e3:.3f} mm")
    if len(dresid):
        print(f"  |TCP-TCP| dist      = {dresid.mean():.2f} +/- {dresid.std():.2f} mm")
    print(f"  R0 dispersion       = {disp.mean():.4f} deg (max {disp.max():.4f})")
    print(f"wrote {batch_csv}")
    print(f"wrote {hist_png}")
    print(f"wrote {summary_md}")
    print(f"wrote mean calibration -> {bc.OUT_JSON}")


def main():
    ap = argparse.ArgumentParser(description="UR3e calibration repeatability batch")
    ap.add_argument("--runs", type=int, default=N_RUNS,
                    help=f"number of calibration runs (default {N_RUNS})")
    args = ap.parse_args()

    results = run_batch(args.runs)
    if not results:
        print("\nNo successful runs -- base_frame_calibration.json left unchanged.")
        return
    summarize_and_write(results, args.runs)


if __name__ == "__main__":
    main()
