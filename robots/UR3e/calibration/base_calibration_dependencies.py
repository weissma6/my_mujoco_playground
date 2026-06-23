"""Closed-form base-frame calibration for the UR3e from motion-capture sweeps.

Goal: recover the UR3e base frame {B} (origin p0 + orientation R0) expressed in
the mocap world frame {W}, from data that the cameras can see directly.

Method (all closed form, NO iterative regression):
  1. Sweep joint 1 alone -> the tracked rigid body traces a circle whose axis is
     the joint-1 rotation axis (vertical, "shoulder up"). Fit it.
  2. Sweep joint 2 alone -> a second circle whose axis is the joint-2 axis. Fit.
  3. The two joint axes intersect at the shoulder point. Intersect the two lines
     (closest point) -> c_int.
  4. p0 = c_int - d1 * z0, where z0 is the joint-1 axis (up) and d1 the nominal
     shoulder height (0.15185 m for the UR3e).
  5. Two small base-frame probe moves (+X, +Y, orientation fixed) give the base
     X and Y directions in {W}; orthonormalize -> R0 (Option A).

A point in the base frame maps to the world frame by  p_W = p0 + R0 @ p_B
(so R0's columns are the base axes expressed in {W}).

Everything here is pure numpy: no RTDE, no mocap, no MuJoCo imports. The notebook
feeds it recorded points / displacements (all in METERS) and writes the JSON.
"""

from typing import Dict, Tuple

import numpy as np

UR3E_D1 = 0.15185  # nominal UR3e shoulder height (base origin -> shoulder), meters


# ---------------------------------------------------------------------------
# Circle fitting
# ---------------------------------------------------------------------------
def fit_circle_3d(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a circle to 3D points lying (approximately) on a common plane.

    Closed form: SVD for the plane + Kasa linear circle fit in that plane.

    Args:
      points: (N, 3) array of world-frame points, N >= 3.

    Returns:
      center: (3,) circle center in world coordinates.
      axis:   (3,) unit normal of the circle plane, oriented so axis . z_world > 0
              (i.e. pointing "up"; for a near-horizontal axis the sign is
              arbitrary but harmless — only the line it defines is used).
      radius: fitted circle radius.
      rms:    std of (per-point radius - radius), in the same units as points.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {p.shape}")
    if p.shape[0] < 3:
        raise ValueError(f"need >= 3 points to fit a circle; got {p.shape[0]}")

    pbar = p.mean(axis=0)
    q = p - pbar  # centered

    # Plane fit: the circle lies in the plane spanned by the two dominant
    # singular vectors; the normal is the smallest-singular-value direction.
    _u, _s, vt = np.linalg.svd(q, full_matrices=False)
    v1 = vt[0]  # in-plane basis
    v2 = vt[1]  # in-plane basis
    n = vt[2]   # plane normal (= circle axis)
    if n[2] < 0.0:  # orient "up"
        n = -n

    # Project centered points into the (v1, v2) plane.
    u = q @ v1
    w = q @ v2

    # Kasa circle fit (linear least squares) in the 2D plane:
    #   u^2 + w^2 = 2*uc*u + 2*wc*w + (r^2 - uc^2 - wc^2)
    # Solve A @ [uc, wc, c3] = b, then r = sqrt(c3 + uc^2 + wc^2).
    a_mat = np.column_stack([2.0 * u, 2.0 * w, np.ones_like(u)])
    b_vec = u * u + w * w
    sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    uc, wc, c3 = sol
    radius = float(np.sqrt(max(c3 + uc * uc + wc * wc, 0.0)))

    center = pbar + uc * v1 + wc * v2

    radii = np.sqrt((u - uc) ** 2 + (w - wc) ** 2)
    rms = float(np.std(radii - radius))
    return center, n, radius, rms


def circumcenter_3pt(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact circle through 3 points (closed form), as a cross-check on the fit.

    Returns (center, axis) with axis the unit plane normal oriented up.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    u = p2 - p1
    v = p3 - p1
    n = np.cross(u, v)
    n2 = float(n @ n)
    if n2 < 1e-18:
        raise ValueError("circumcenter_3pt: points are collinear")
    center = p1 + (np.cross((u @ u) * v - (v @ v) * u, n)) / (2.0 * n2)
    axis = n / np.sqrt(n2)
    if axis[2] < 0.0:
        axis = -axis
    return center, axis


# ---------------------------------------------------------------------------
# Axis intersection
# ---------------------------------------------------------------------------
def closest_point_two_lines(
    c1: np.ndarray, a1: np.ndarray, c2: np.ndarray, a2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Closest points of two 3D lines L1 = c1 + lam*a1, L2 = c2 + mu*a2.

    a1, a2 are normalized internally. Returns (P1, P2, gap_h) where P1 is on L1,
    P2 on L2, and gap_h = ||P1 - P2|| (0 for truly intersecting axes; a few mm of
    gap is the practical measure of fit quality). Raises if the axes are parallel.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)

    w0 = c1 - c2
    b = float(a1 @ a2)
    d = float(a1 @ w0)
    e = float(a2 @ w0)
    den = 1.0 - b * b
    if abs(den) < 1e-12:
        raise ValueError("closest_point_two_lines: axes are (near) parallel")
    lam = (b * e - d) / den
    mu = (e - b * d) / den
    p1 = c1 + lam * a1
    p2 = c2 + mu * a2
    gap_h = float(np.linalg.norm(p1 - p2))
    return p1, p2, gap_h


def base_origin(c_int: np.ndarray, z0_up: np.ndarray, d1: float) -> np.ndarray:
    """Base origin p0 = shoulder point - d1 along the (up) joint-1 axis."""
    z0_up = np.asarray(z0_up, dtype=np.float64)
    z0_up = z0_up / np.linalg.norm(z0_up)
    return np.asarray(c_int, dtype=np.float64) - float(d1) * z0_up


# ---------------------------------------------------------------------------
# Orientation from probe moves (Option A)
# ---------------------------------------------------------------------------
def frame_from_probe(
    dp_x: np.ndarray, dp_y: np.ndarray, L: float, z0: np.ndarray
) -> np.ndarray:
    """Base orientation R0 from two orientation-fixed base-frame probe moves.

    A +X base move of length L (orientation locked) translates the tracked body
    by dp_x = R0 @ [L,0,0] in {W}, so dp_x's direction IS the base +X axis in
    {W}; likewise dp_y for +Y. Gram-Schmidt -> a right-handed orthonormal R0
    whose columns are the base X/Y/Z axes in world coordinates.

    z0 (the joint-1 axis from the circle fit) is NOT forced into the result; it
    is only used to orient the probe-derived Z consistently "up" (the agreement
    between the two is reported separately as a residual). L is unused for the
    direction (kept in the signature so callers pass the commanded length and can
    compare it to ||dp_x|| / ||dp_y|| as an independent sanity check).
    """
    dp_x = np.asarray(dp_x, dtype=np.float64)
    dp_y = np.asarray(dp_y, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)

    x = dp_x / np.linalg.norm(dp_x)
    y_raw = dp_y / np.linalg.norm(dp_y)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    # Keep Z pointing the same way as the measured joint-1 axis. Flipping Z then
    # rebuilding Y = Z x X preserves right-handedness AND a right-handed X/Y
    # probe pair, so this only corrects an overall sign, never the geometry.
    if float(z @ (z0 / np.linalg.norm(z0))) < 0.0:
        z = -z
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    r0 = np.column_stack([x, y, z])
    return r0


# ---------------------------------------------------------------------------
# Full solve
# ---------------------------------------------------------------------------
def solve(
    points_j1: np.ndarray,
    points_j2: np.ndarray,
    dp_x: np.ndarray,
    dp_y: np.ndarray,
    L: float,
    d1: float = UR3E_D1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Full base-frame solve from two joint sweeps + two probe moves.

    Returns (p0, R0, residuals) where residuals carries the fit-quality numbers
    (circle RMS, axis gap, probe-vs-axis angle, probe length errors), all in mm /
    deg for human reading.
    """
    c1, a1, r1, rms1 = fit_circle_3d(points_j1)
    c2, a2, r2, rms2 = fit_circle_3d(points_j2)

    z0 = a1 / np.linalg.norm(a1)  # joint-1 axis, oriented up
    p1, p2, gap_h = closest_point_two_lines(c1, a1, c2, a2)
    c_int = p1  # point on the joint-1 axis closest to the joint-2 axis

    p0 = base_origin(c_int, z0, d1)
    r0 = frame_from_probe(dp_x, dp_y, L, z0)

    z_probe = r0[:, 2]
    z0_vs_a1_deg = float(
        np.degrees(np.arccos(np.clip(z_probe @ z0, -1.0, 1.0)))
    )
    residuals = {
        "circle1_rms_mm": rms1 * 1e3,
        "circle2_rms_mm": rms2 * 1e3,
        "circle1_radius_mm": r1 * 1e3,
        "circle2_radius_mm": r2 * 1e3,
        "axis_gap_h_mm": gap_h * 1e3,
        "z0_vs_a1_deg": z0_vs_a1_deg,
        "probe_x_len_err_mm": (float(np.linalg.norm(dp_x)) - L) * 1e3,
        "probe_y_len_err_mm": (float(np.linalg.norm(dp_y)) - L) * 1e3,
    }
    return p0, r0, residuals


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def rotation_matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) unit quaternion."""
    r = np.asarray(r, dtype=np.float64)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def build_output_dict(
    p0: np.ndarray,
    r0: np.ndarray,
    residuals: Dict[str, float],
    n_points_j1: int,
    n_points_j2: int,
    rigid_body: str,
    d1: float,
    timestamp: str,
) -> dict:
    """Assemble the base_frame_calibration.json payload."""
    quat = rotation_matrix_to_quat_wxyz(r0)
    return {
        "translation_m": np.asarray(p0, dtype=float).tolist(),
        "rotation_quat_wxyz": quat.tolist(),
        "rotation_matrix": np.asarray(r0, dtype=float).tolist(),
        "rigid_body": rigid_body,
        "world_frame": "mocap",
        "d1_used_m": float(d1),
        "residuals": {k: float(v) for k, v in residuals.items()},
        "n_points": {"joint1": int(n_points_j1), "joint2": int(n_points_j2)},
        "timestamp": timestamp,
    }
