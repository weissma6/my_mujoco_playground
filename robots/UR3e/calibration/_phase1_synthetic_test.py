"""Phase 1 offline check: recover a KNOWN base frame from synthetic sweeps.

Builds two circles that share the shoulder point (so the joint axes truly
intersect), adds noise, and asserts solve() recovers p0 < 1 mm and that the
3-point circumcenter agrees with the full SVD+Kasa fit. No hardware, no mocap.
"""

import numpy as np

import base_calibration_dependencies as cal


def _rot_about(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _plane_basis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = a - (a @ n) * n
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    return v1, v2


def main(seed=0, noise_m=0.0005):
    rng = np.random.default_rng(seed)
    d1 = cal.UR3E_D1

    # --- Ground-truth base frame in the world (mocap) frame ---
    p0_true = np.array([0.612, -0.345, 0.018])
    R0_true = _rot_about([0, 0, 1], np.deg2rad(37.0)) @ _rot_about(
        [1, 0, 0], np.deg2rad(2.0)
    )
    z0_true = R0_true[:, 2]                 # base +Z (joint-1 axis, up)
    a2_true = R0_true[:, 1]                 # base +Y (joint-2 / shoulder axis)
    shoulder = p0_true + d1 * z0_true       # where the two joint axes cross

    # --- Joint-1 circle: axis z0_true through the shoulder/base line ---
    c1_true = shoulder + 0.22 * z0_true     # tracked body orbits above shoulder
    r1 = 0.18
    v1a, v1b = _plane_basis(z0_true)
    th1 = np.deg2rad(np.linspace(-85, 85, 40))   # ~170 deg half-circle
    pts1 = (c1_true + r1 * (np.cos(th1)[:, None] * v1a
                            + np.sin(th1)[:, None] * v1b))
    pts1 = pts1 + rng.normal(0, noise_m, pts1.shape)

    # --- Joint-2 circle: axis a2_true through the SAME shoulder point ---
    c2_true = shoulder + 0.10 * a2_true
    r2 = 0.25
    v2a, v2b = _plane_basis(a2_true)
    th2 = np.deg2rad(np.linspace(-80, 80, 40))
    pts2 = (c2_true + r2 * (np.cos(th2)[:, None] * v2a
                            + np.sin(th2)[:, None] * v2b))
    pts2 = pts2 + rng.normal(0, noise_m, pts2.shape)

    # --- Probe moves: +X, +Y in base frame, length L (orientation fixed) ---
    L = 0.10
    dp_x = R0_true @ np.array([L, 0, 0]) + rng.normal(0, noise_m, 3)
    dp_y = R0_true @ np.array([0, L, 0]) + rng.normal(0, noise_m, 3)

    # --- Solve ---
    p0, R0, res = cal.solve(pts1, pts2, dp_x, dp_y, L, d1)

    p0_err_mm = np.linalg.norm(p0 - p0_true) * 1e3
    # Frame error: geodesic angle between R0 and R0_true.
    ang = np.degrees(np.arccos(
        np.clip((np.trace(R0.T @ R0_true) - 1) / 2, -1, 1)))

    # --- Full-fit vs 3-point circumcenter cross-check (circle 1) ---
    c_fit, n_fit, _, _ = cal.fit_circle_3d(pts1)
    idx = [0, len(pts1) // 2, len(pts1) - 1]
    c_3pt, n_3pt = cal.circumcenter_3pt(*[pts1[i] for i in idx])
    fit_vs_3pt_mm = np.linalg.norm(c_fit - c_3pt) * 1e3

    print(f"noise = {noise_m*1e3:.2f} mm/point")
    print(f"p0_true  = {np.round(p0_true, 4)}")
    print(f"p0_recov = {np.round(p0, 4)}")
    print(f"p0 error             = {p0_err_mm:.3f} mm")
    print(f"R0 geodesic error    = {ang:.3f} deg")
    print(f"full-fit vs 3-point  = {fit_vs_3pt_mm:.3f} mm")
    print("residuals:")
    for k, v in res.items():
        print(f"  {k:22s} = {v:8.4f}")

    assert p0_err_mm < 1.0, f"p0 error {p0_err_mm:.3f} mm exceeds 1 mm"
    assert ang < 1.0, f"R0 error {ang:.3f} deg too large"
    assert fit_vs_3pt_mm < 5.0, "full-fit and 3-point circumcenter disagree"
    print("\nPHASE 1 PASSED: p0 recovered < 1 mm; fit methods agree.")
    return p0_err_mm


if __name__ == "__main__":
    # Exact (no noise) must be ~0; then a realistic 0.5 mm/point noise level.
    print("=== noise-free ===")
    main(noise_m=0.0)
    print("\n=== 0.5 mm/point noise ===")
    main(noise_m=0.0005)
