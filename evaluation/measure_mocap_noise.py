"""Static-capture mocap sensor-noise measurement (D9's "sensor-noise check").

Plan §D9: "a static mocap capture (cube untouched, 30-60s) gives the pure box
pos/quat measurement sigma directly -- validates the C5 assumption that
box_pos noise is sub-mm (the DR jitter was set to 2mm *deliberately above*
it). Convert quats to a geodesic angle from the mean before taking sigma;
don't std raw quaternions. This is NOT the noise floor -- it is the
sensor-level input to it."

This is a separate, cheaper measurement than gap_metrics.noise_floor()'s D9
paired-gap floor: this script only asks "how noisy is the MEASUREMENT",
with the cube sitting still and nothing else varying (no policy, no
placement, no arm motion) -- it feeds C5's obs-noise design, it does not
replace the paired-gap floor.

What it computes, given a static-capture log (box untouched on the table,
pose logged every tick for ~30-60s):
  - box POSITION measurement noise: std(x), std(y), std(z) in meters --
    the raw robot-base-frame position columns are already Euclidean, so a
    plain per-axis sample std is the right statistic.
  - box ORIENTATION measurement noise: convert every sample's quaternion to
    a GEODESIC ANGLE FROM THE MEAN quaternion (never std the raw quaternion
    components -- w/x/y/z are not a Euclidean space and taking their
    componentwise std is not a rotation-noise magnitude). The mean itself is
    computed with Markley et al.'s eigenvector method (see `mean_quaternion`
    below), which is naturally invariant to the q vs -q sign ambiguity
    (q q^T == (-q)(-q)^T -- the ambiguity cancels inside the outer product,
    no explicit sign-flip pass needed); `geodesic_angles_deg` additionally
    takes |dot(q_i, mean_q)| per sample so a lone sign-flipped sample can
    never inflate its own distance either. Reports the std of those
    per-sample geodesic angles, in degrees.

Column names -- read from the ACTUAL logging code, not guessed: the real
pickloop logs box pose as box_x/box_y/box_z (meters, base frame) and
box_qw/box_qx/box_qy/box_qz (MuJoCo w,x,y,z convention, base frame) --
see robots/UR3e/ur3_realrobot_dependencies.py's run_policy_loop row dict
(around "box_x": box_pos[0], ... "box_qw": box_quat[0], ...) and
evaluation/ur3_reward_replay.py's replay_dataframe, which reads the exact
same names. A static-capture log produced by the same logging path (or any
CSV with those column names) works with this script's defaults; --box_pos_cols
/ --box_quat_cols override them if a capture script ever names them
differently.

Local usage (plain numpy/pandas -- no MJX, no mocap hardware, safe to run
locally under CLAUDE.md's hard rules):
    python evaluation/measure_mocap_noise.py --csv static_capture.csv
    python evaluation/measure_mocap_noise.py --selftest
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

# D9 / robots/UR3e/ur3_realrobot_dependencies.py's run_policy_loop row dict +
# evaluation/ur3_reward_replay.py's replay_dataframe -- the ACTUAL column
# names this repo's logs use, not guessed. box_quat is MuJoCo's (w, x, y, z)
# convention, base frame.
BOX_POS_COLS = ("box_x", "box_y", "box_z")
BOX_QUAT_COLS = ("box_qw", "box_qx", "box_qy", "box_qz")


# ===========================================================================
# Position noise -- plain per-axis sample std (Euclidean, no subtlety).
# ===========================================================================


def position_noise(df: pd.DataFrame, cols=BOX_POS_COLS) -> dict:
    """{col: std(col)} in meters, ddof=1 (sample std). Raises if a column is
    missing rather than silently skipping it -- a missing axis would
    silently under-report the noise, not just drop a column.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"position_noise: missing column(s) {missing} -- "
                         f"available columns: {list(df.columns)}")
    return {c: float(df[c].astype(float).std(ddof=1)) for c in cols}


# ===========================================================================
# Orientation noise -- geodesic angle from the mean quaternion.
# ===========================================================================


def _normalize_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """Mean orientation of an (n, 4) array of (w, x, y, z) quaternions, via
    Markley, Cheng, Crassidis & Oshman's eigenvector method (JGCD 2007,
    "Averaging Quaternions"): build M = sum_i q_i q_i^T (4x4) and take the
    eigenvector of M's LARGEST eigenvalue.

    This is the closed-form least-squares mean on SO(3) (minimizes the sum
    of squared chordal -- equivalently, for unit quaternions, a monotone
    function of geodesic -- distances) and is AUTOMATICALLY invariant to the
    q vs -q sign ambiguity: q q^T == (-q)(-q)^T identically, so a sample
    entering the sum as -q contributes exactly the same M as if it had
    entered as q. No separate "pick a hemisphere and flip" pass is needed
    before calling this.
    """
    q = _normalize_quats(np.asarray(quats, dtype=float))
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"mean_quaternion: expected (n, 4), got {q.shape}")
    m = q.T @ q  # (4, 4)
    eigvals, eigvecs = np.linalg.eigh(m)
    mean_q = eigvecs[:, int(np.argmax(eigvals))]
    if mean_q[0] < 0:  # canonicalize w >= 0 -- cosmetic, same rotation either way
        mean_q = -mean_q
    return mean_q


def geodesic_angles_deg(quats: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    """Geodesic (shortest-path-on-SO(3)) angle, in degrees, from each row of
    `quats` to the single reference quaternion `ref_q`.

    theta = 2 * arccos(|q_i . ref_q|) -- the absolute value handles the sign
    ambiguity PER SAMPLE (q_i and -q_i are the same rotation and must give
    the same distance to ref_q; without abs(), a sign-flipped sample would
    read back as the SUPPLEMENTARY angle pi - theta instead of theta).
    """
    q = _normalize_quats(np.asarray(quats, dtype=float))
    ref = np.asarray(ref_q, dtype=float)
    ref = ref / np.linalg.norm(ref)
    dot = np.abs(q @ ref)
    dot = np.clip(dot, -1.0, 1.0)
    theta = 2.0 * np.arccos(dot)
    return np.degrees(theta)


def orientation_noise(df: pd.DataFrame, cols=BOX_QUAT_COLS) -> dict:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"orientation_noise: missing column(s) {missing} -- "
                         f"available columns: {list(df.columns)}")
    quats = df[list(cols)].to_numpy(dtype=float)
    mean_q = mean_quaternion(quats)
    angles = geodesic_angles_deg(quats, mean_q)
    return {
        "mean_quat_wxyz": mean_q.tolist(),
        "geodesic_angle_std_deg": float(np.std(angles, ddof=1)),
        "geodesic_angle_mean_deg": float(np.mean(angles)),
        "geodesic_angle_max_deg": float(np.max(angles)),
        "n_samples": int(len(df)),
    }


# ===========================================================================
# Combined report.
# ===========================================================================


def measure(csv_path: str, pos_cols=BOX_POS_COLS, quat_cols=BOX_QUAT_COLS) -> dict:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{csv_path}: empty CSV, nothing to measure.")
    return {
        "csv_path": csv_path,
        "n_samples": int(len(df)),
        "position_m": position_noise(df, pos_cols),
        "orientation_deg": orientation_noise(df, quat_cols),
    }


def _print_report(report: dict) -> None:
    print(f"\nStatic-capture mocap noise -- {report['csv_path']} "
          f"({report['n_samples']} samples)")
    print("Position (m, per-axis sample std):")
    for c, v in report["position_m"].items():
        print(f"  {c}: {v:.6f} m  ({v * 1000:.3f} mm)")
    ori = report["orientation_deg"]
    print("Orientation (geodesic angle from the mean quaternion):")
    print(f"  std  = {ori['geodesic_angle_std_deg']:.4f} deg")
    print(f"  mean = {ori['geodesic_angle_mean_deg']:.4f} deg")
    print(f"  max  = {ori['geodesic_angle_max_deg']:.4f} deg")
    print(f"  mean_quat (w,x,y,z) = {[round(x, 5) for x in ori['mean_quat_wxyz']]}")
    print(
        "\nD9 context: this is the SENSOR-LEVEL input, not the noise floor "
        "itself -- compare box_pos std against the C5 obs-noise jitter "
        "(2mm, deliberately conservative) and the D9 paired-gap floor from "
        "gap_metrics.noise_floor(), not against each other directly."
    )


# ===========================================================================
# --selftest -- synthetic data, no mocap hardware, no MJX.
# ===========================================================================


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _axis_angle_to_quat(axis: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    """axis: (n,3) unit vectors, angle_rad: (n,) -> (n,4) (w,x,y,z) quats."""
    half = angle_rad / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)[:, None]
    return np.concatenate([w[:, None], xyz], axis=1)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, (w,x,y,z) convention, broadcasts over leading dims."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def _selftest():
    rng = np.random.default_rng(0)

    print("(a) position noise: recovered per-axis std matches the injected "
          "sigma ...", end=" ")
    n = 5000
    sigma_pos = {"box_x": 0.0004, "box_y": 0.0006, "box_z": 0.00025}  # meters
    centers = {"box_x": 0.30, "box_y": 0.0, "box_z": 0.02}
    df_pos = pd.DataFrame({
        c: rng.normal(centers[c], s, n) for c, s in sigma_pos.items()
    })
    pos = position_noise(df_pos)
    for c, s in sigma_pos.items():
        rel_err = abs(pos[c] - s) / s
        assert rel_err < 0.08, (c, pos[c], s, rel_err)
    print(f"OK  ({ {c: round(v, 6) for c, v in pos.items()} })")

    # A fixed, non-trivial reference orientation (not identity) -- the
    # "known reference" the plan's own D9 note describes.
    q0 = _axis_angle_to_quat(np.array([[0.0, 0.0, 1.0]]), np.array([0.7]))[0]

    print("(b) mean_quaternion recovers the true reference, and is unaffected "
          "by injecting the q/-q sign ambiguity ...", end=" ")
    n = 3000
    axes = _random_unit_vectors(n, rng)
    sigma_deg = 2.0
    theta_deg = np.abs(rng.normal(0.0, sigma_deg, n))  # injected angular noise
    theta_rad = np.radians(theta_deg)
    deltas = _axis_angle_to_quat(axes, theta_rad)
    samples = _quat_multiply(np.broadcast_to(q0, (n, 4)), deltas)
    # Sign-flip a random half to -q (the SAME rotation) -- mean_quaternion
    # and geodesic_angles_deg must be unaffected by this.
    flip = rng.random(n) < 0.5
    samples = samples.copy()
    samples[flip] *= -1.0

    mean_q = mean_quaternion(samples)
    angle_to_true = float(geodesic_angles_deg(mean_q[None, :], q0)[0])
    assert angle_to_true < 0.3, angle_to_true
    print(f"OK  (recovered mean is {angle_to_true:.4f} deg from the true "
          f"reference, despite {int(flip.sum())}/{n} samples sign-flipped)")

    print("(c) geodesic-angle std matches the injected angular-noise "
          "magnitude ...", end=" ")
    # Ground truth: the actual std of the injected theta_deg magnitudes
    # (computed from the SAME array used to build the samples) -- not a
    # theoretical distribution constant, so this is an honest end-to-end
    # check of mean estimation + geodesic recovery together.
    expected_std_deg = float(np.std(theta_deg, ddof=1))
    recovered = geodesic_angles_deg(samples, mean_q)
    recovered_std_deg = float(np.std(recovered, ddof=1))
    rel_err = abs(recovered_std_deg - expected_std_deg) / expected_std_deg
    assert rel_err < 0.10, (recovered_std_deg, expected_std_deg, rel_err)
    print(f"OK  (recovered={recovered_std_deg:.4f} deg, "
          f"injected={expected_std_deg:.4f} deg, rel_err={rel_err:.2%})")

    print("(d) raw-quaternion-component std would NOT equal the geodesic "
          "std (guards against std'ing raw components instead) ...", end=" ")
    raw_component_std_deg = float(
        np.degrees(np.std(samples[:, 1:], ddof=1))  # nonsense on purpose
    )
    assert abs(raw_component_std_deg - recovered_std_deg) > 1.0, (
        raw_component_std_deg, recovered_std_deg)
    print("OK  (confirmed these are NOT the same quantity)")

    print("\nSELFTEST OK (pure numpy/pandas, no MJX, no mocap hardware).")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="static-capture log (box untouched, ~30-60s of ticks)")
    ap.add_argument("--box_pos_cols", nargs=3, default=list(BOX_POS_COLS),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--box_quat_cols", nargs=4, default=list(BOX_QUAT_COLS),
                    metavar=("QW", "QX", "QY", "QZ"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.csv:
        _selftest()
        return

    report = measure(args.csv, tuple(args.box_pos_cols), tuple(args.box_quat_cols))
    _print_report(report)


if __name__ == "__main__":
    _cli()
