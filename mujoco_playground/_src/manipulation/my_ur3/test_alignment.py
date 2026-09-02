"""Tests for alignment.frame_alignment (CPU only, no mjx/brax, no env)."""
import numpy as np
import pytest

from mujoco_playground._src.manipulation.my_ur3.alignment import frame_alignment


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


IDENTITY_JAW = np.array([1.0, 0.0, 0.0])
IDENTITY_APP = np.array([0.0, 0.0, 1.0])
IDENTITY_BOX = np.eye(3)


def _scores(jaw_axis, app_axis, box_axes):
    out = frame_alignment(jaw_axis, app_axis, box_axes)
    return {k: float(v) for k, v in out.items()}


def test_identity_frame_identity_box():
    s = _scores(IDENTITY_JAW, IDENTITY_APP, IDENTITY_BOX)
    assert s["jaw"] == pytest.approx(1.0, abs=1e-6)
    assert s["app"] == pytest.approx(1.0, abs=1e-6)
    assert s["third"] == pytest.approx(1.0, abs=1e-6)
    assert s["face"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("angle", [np.pi / 2, np.pi, 3 * np.pi / 2])
def test_box_yawed_multiples_of_90(angle):
    box_axes = rot([0, 0, 1], angle)
    s = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes)
    for k in ("jaw", "app", "third", "face"):
        assert s[k] == pytest.approx(1.0, abs=1e-6)


def test_box_yawed_45_degrees():
    box_axes = rot([0, 0, 1], np.pi / 4)
    s = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes)
    expected = (np.cos(np.pi / 4) - 0.5) / 0.5
    assert s["jaw"] == pytest.approx(expected, abs=1e-6)
    assert s["third"] == pytest.approx(expected, abs=1e-6)
    assert s["app"] == pytest.approx(1.0, abs=1e-6)
    assert s["face"] == pytest.approx(s["jaw"] * s["third"], abs=1e-6)


def test_tilt_cotilted_gripper_and_box():
    r = rot([1, 0, 0], 0.12)
    box_axes = r
    jaw_axis = r @ IDENTITY_JAW
    app_axis = r @ IDENTITY_APP
    s = _scores(jaw_axis, app_axis, box_axes)
    for k in ("jaw", "app", "third", "face"):
        assert s[k] == pytest.approx(1.0, abs=1e-6)


def test_tilt_untilted_gripper_vs_tilted_box():
    r = rot([1, 0, 0], 0.12)
    box_axes = r

    s_tilted = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes)
    expected_app = (np.cos(0.12) - 0.5) / 0.5
    assert s_tilted["app"] == pytest.approx(expected_app, abs=1e-6)
    assert s_tilted["app"] < 1.0

    jaw_axis = r @ IDENTITY_JAW
    app_axis = r @ IDENTITY_APP
    s_cotilted = _scores(jaw_axis, app_axis, box_axes)

    assert s_tilted["face"] < s_cotilted["face"]


def test_side_grasp_full_cube_symmetry():
    r = rot([0, 1, 0], np.pi / 2)
    jaw_axis = r @ IDENTITY_JAW
    app_axis = r @ IDENTITY_APP
    s = _scores(jaw_axis, app_axis, IDENTITY_BOX)
    for k in ("jaw", "app", "third", "face"):
        assert s[k] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("col", [0, 1, 2])
def test_sign_invariance_identity_box(col):
    box_axes = IDENTITY_BOX.copy()
    box_axes[:, col] *= -1
    s = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes)
    baseline = _scores(IDENTITY_JAW, IDENTITY_APP, IDENTITY_BOX)
    for k in ("jaw", "app", "third", "face"):
        assert s[k] == pytest.approx(baseline[k], abs=1e-6)


@pytest.mark.parametrize("col", [0, 1, 2])
def test_sign_invariance_yawed_box(col):
    box_axes = rot([0, 0, 1], np.pi / 4)
    baseline = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes)
    box_axes_flipped = box_axes.copy()
    box_axes_flipped[:, col] *= -1
    s = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes_flipped)
    for k in ("jaw", "app", "third", "face"):
        assert s[k] == pytest.approx(baseline[k], abs=1e-6)


def test_no_dead_zone_random_rotations():
    rng = np.random.default_rng(0)
    n = 2000
    floor = (1 / np.sqrt(3) - 0.5) / 0.5 - 1e-6
    min_min_score = np.inf
    for _ in range(n):
        m = rng.normal(size=(3, 3))
        q, r_ = np.linalg.qr(m)
        d = np.sign(np.diag(r_))
        d[d == 0] = 1.0
        q = q * d
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        jaw_axis = q @ IDENTITY_JAW
        app_axis = q @ IDENTITY_APP
        s = _scores(jaw_axis, app_axis, IDENTITY_BOX)
        assert s["face"] > 0.0
        min_min_score = min(min_min_score, s["jaw"], s["app"], s["third"])
    assert min_min_score >= floor


def test_non_unit_inputs():
    box_axes = rot([0, 0, 1], np.pi / 4)
    s_unit = _scores(IDENTITY_JAW, IDENTITY_APP, box_axes)
    s_scaled = _scores(IDENTITY_JAW * 3.7, IDENTITY_APP * 0.2, box_axes)
    for k in ("jaw", "app", "third", "face"):
        assert s_scaled[k] == pytest.approx(s_unit[k], abs=1e-6)


def test_cos_bound_param_shrinks_45_deg_score_to_zero():
    # At 45 deg misalignment, cos(45)=0.7071. With cos_bound=0.9 the score's
    # numerator (0.7071 - 0.9) is negative, so it must clip to exactly 0 --
    # not just "smaller than the cos_bound=0.5 case".
    box_axes = rot([0, 0, 1], np.pi / 4)
    out = frame_alignment(IDENTITY_JAW, IDENTITY_APP, box_axes, cos_bound=0.9)
    s = {k: float(v) for k, v in out.items()}
    assert s["jaw"] == pytest.approx(0.0, abs=1e-6)
    assert s["third"] == pytest.approx(0.0, abs=1e-6)
    assert s["face"] == pytest.approx(0.0, abs=1e-6)
    # app is unaffected by the box yaw (still axis-aligned) so it must stay 1.
    assert s["app"] == pytest.approx(1.0, abs=1e-6)


def test_zero_jaw_vector_no_nan():
    zero = np.array([0.0, 0.0, 0.0])
    out = frame_alignment(zero, IDENTITY_APP, IDENTITY_BOX)
    s = {k: float(v) for k, v in out.items()}
    for k in ("jaw", "app", "third", "face"):
        assert np.isfinite(s[k]), f"{k} is not finite: {s[k]}"
    # A degenerate (zero-norm) jaw axis must not accidentally score as
    # perfectly aligned.
    assert s["jaw"] == pytest.approx(0.0, abs=1e-6)
    assert s["face"] == pytest.approx(0.0, abs=1e-6)


def test_zero_jaw_and_app_vectors_no_nan():
    zero = np.array([0.0, 0.0, 0.0])
    out = frame_alignment(zero, zero, IDENTITY_BOX)
    s = {k: float(v) for k, v in out.items()}
    for k in ("jaw", "app", "third", "face"):
        assert np.isfinite(s[k]), f"{k} is not finite: {s[k]}"


def test_face_equals_product_of_the_three_scores_random_rotations():
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = rng.normal(size=(3, 3))
        q, r_ = np.linalg.qr(m)
        d = np.sign(np.diag(r_))
        d[d == 0] = 1.0
        q = q * d
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        jaw_axis = q @ IDENTITY_JAW
        app_axis = q[:, [1, 0, 2]] @ IDENTITY_APP  # not necessarily orthogonal
        box_axes = rng.normal(size=(3, 3))
        s = _scores(jaw_axis, app_axis, box_axes)
        assert s["face"] == pytest.approx(
            s["jaw"] * s["app"] * s["third"], abs=1e-6
        )


def test_jit_compatible():
    # frame_alignment is called from inside a jitted reward function -- it
    # must trace cleanly under jax.jit and return the same values as eager.
    import jax

    box_axes = rot([0, 0, 1], np.pi / 4)
    jitted = jax.jit(frame_alignment)
    eager = frame_alignment(IDENTITY_JAW, IDENTITY_APP, box_axes)
    traced = jitted(IDENTITY_JAW, IDENTITY_APP, box_axes)
    for k in ("jaw", "app", "third", "face"):
        assert float(traced[k]) == pytest.approx(float(eager[k]), abs=1e-6)


def _random_rotation(rng):
    # QR of a random Gaussian matrix, sign-fixed to a proper (det=+1)
    # rotation -- same construction as test_no_dead_zone_random_rotations /
    # test_face_equals_product_of_the_three_scores_random_rotations above.
    m = rng.normal(size=(3, 3))
    q, r_ = np.linalg.qr(m)
    d = np.sign(np.diag(r_))
    d[d == 0] = 1.0
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_frame_alignment_np_matches_jax_random_rotations():
    """Parity: evaluation.ur3_reward_replay.frame_alignment_np (numpy port,
    runs inside the real-robot 50 Hz loop, no jax import) must agree with
    alignment.frame_alignment (jax) to abs=1e-6 on all four scores, for 300
    seeded random gripper-frame rotations against random box rotations.
    """
    # ur3_reward_replay.py imports only matplotlib/numpy/pandas plus
    # mujoco_playground's own ur3_pick module (checked: no rtde_control or
    # other robot-only dependency) -- it imported cleanly on this machine.
    # importorskip is kept anyway as a guard in case that ever changes.
    replay = pytest.importorskip("evaluation.ur3_reward_replay")
    frame_alignment_np = replay.frame_alignment_np

    rng = np.random.default_rng(42)
    for _ in range(300):
        q_gripper = _random_rotation(rng)
        q_box = _random_rotation(rng)
        jaw_axis = q_gripper @ IDENTITY_JAW
        app_axis = q_gripper @ IDENTITY_APP
        expected = _scores(jaw_axis, app_axis, q_box)
        got = frame_alignment_np(jaw_axis, app_axis, q_box)
        got = {k: float(v) for k, v in got.items()}
        for k in ("jaw", "app", "third", "face"):
            assert got[k] == pytest.approx(expected[k], abs=1e-6)


def test_frame_alignment_np_zero_jaw_vector_matches_jax():
    """Degenerate case: a zero-norm jaw axis must not NaN and must match the
    jax implementation's clamp-to-zero behaviour exactly."""
    replay = pytest.importorskip("evaluation.ur3_reward_replay")
    frame_alignment_np = replay.frame_alignment_np

    zero = np.array([0.0, 0.0, 0.0])
    expected = _scores(zero, IDENTITY_APP, IDENTITY_BOX)
    got = frame_alignment_np(zero, IDENTITY_APP, IDENTITY_BOX)
    got = {k: float(v) for k, v in got.items()}
    for k in ("jaw", "app", "third", "face"):
        assert np.isfinite(got[k]), f"{k} is not finite: {got[k]}"
        assert got[k] == pytest.approx(expected[k], abs=1e-6)
