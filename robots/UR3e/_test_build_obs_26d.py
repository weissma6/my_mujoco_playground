"""Regression test: UR3RealRobotPick.build_obs_from_feedback's 26D path.

Asserts the real-robot obs builder produces the SAME grasp-frame
orientation features (jaw_proj/app_proj) and last_action slot as the
training env's UR3Pick._get_obs, by cross-checking against
evaluation/ur3_reward_replay.SimFK -- an independent MuJoCo FK
implementation built from the same scene XML. Also regression-checks that
the 13D (legacy) path is unaffected.

Pure MuJoCo FK only -- NO MJX, NO robot connection (UR3RealRobotPick's
constructor does not connect to hardware). Safe to run locally:

    python robots/UR3e/_test_build_obs_26d.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))

from ur3_realrobot_dependencies import UR3RealRobotPick  # noqa: E402
from ur3_reward_replay import SimFK  # noqa: E402

XML = os.path.join(
    _REPO_ROOT, "mujoco_playground", "_src", "manipulation", "my_ur3", "xmls",
    "mjx_single_cube_position_ur3.xml",
)


def _run():
    r = UR3RealRobotPick()
    r.init_fk_model(XML)
    r._finger_pos_est = 0.012

    q = np.array([0.1, -2.0, 1.6, -1.6, -1.5, 0.3])
    box = np.array([0.31, 0.02, 0.05])
    theta = np.radians(30)
    quat = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    fb = {"q": q.tolist(), "tcp_xyz": [0.0, 0.0, 0.0]}
    target = np.array([0.30, 0.0, 0.20])
    last_action = np.arange(7, dtype=float)

    # --- 26D path ---
    obs = r.build_obs_from_feedback(
        fb, box, target, box_quat=quat, last_action=last_action, obs_dim=26,
    )
    assert obs.shape == (1, 26), f"expected (1, 26), got {obs.shape}"
    jaw_proj, app_proj = obs[0, 13:16], obs[0, 16:19]

    # Sanity: box-frame projections of a unit axis through an orthonormal
    # rotation are unit vectors. +-1e-3 (not 1e-6) because the underlying
    # axis normalization uses a +1e-6 epsilon (matches ur3_pick.py exactly),
    # so it is not bit-exact unit length.
    assert abs(np.linalg.norm(jaw_proj) - 1.0) < 1e-3, jaw_proj
    assert abs(np.linalg.norm(app_proj) - 1.0) < 1e-3, app_proj
    assert np.allclose(obs[0, 19:26], last_action), "last_action passthrough"

    # Strict bit-parity against the independent SimFK oracle (same scene
    # XML, computed via a completely separate code path).
    geo = SimFK(XML).geom(q, 0.012, box, quat)
    jaw_axis = geo["rgt"] - geo["lft"]
    jaw_axis = jaw_axis / (np.linalg.norm(jaw_axis) + 1e-6)
    app_axis = 0.5 * (geo["lft"] + geo["rgt"]) - geo["tcp"]
    app_axis = app_axis / (np.linalg.norm(app_axis) + 1e-6)
    assert np.allclose(jaw_proj, jaw_axis @ geo["box_axes"], atol=1e-6), (
        "jaw_proj does not match the independent SimFK oracle"
    )
    assert np.allclose(app_proj, app_axis @ geo["box_axes"], atol=1e-6), (
        "app_proj does not match the independent SimFK oracle"
    )

    # --- 13D path: byte-identical to the pre-26D formula ---
    obs13 = r.build_obs_from_feedback(fb, box, target, obs_dim=13)
    assert obs13.shape == (1, 13), f"expected (1, 13), got {obs13.shape}"
    arm_q = np.array(fb["q"], dtype=np.float32)
    gripper = np.array(
        [2.0 * np.clip(r._finger_pos_est, 0.0, r._finger_hi)], dtype=np.float32
    )
    tcp = np.array(fb["tcp_xyz"], dtype=np.float32)
    b = np.array(box, dtype=np.float32)
    t = np.array(target, dtype=np.float32)
    manual13 = np.concatenate([arm_q, gripper, b - tcp, t - b])[None, :]
    assert np.allclose(obs13, manual13), "13D path drifted from the original formula"

    # --- Default obs_dim: no policy loaded -> falls back to 13D, never raises. ---
    obs_default = r.build_obs_from_feedback(fb, box, target)
    assert obs_default.shape == (1, 13)

    # --- Unsupported obs_dim raises clearly. ---
    try:
        r.build_obs_from_feedback(fb, box, target, obs_dim=19)
        raise AssertionError("expected ValueError for obs_dim=19")
    except ValueError:
        pass

    print("BUILD_OBS 26D PARITY OK")


if __name__ == "__main__":
    _run()
