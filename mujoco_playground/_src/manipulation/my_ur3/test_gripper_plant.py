"""Local (pure-MuJoCo, no MJX, no env.step) verification of the Hand-E gripper
plant: ctrlrange tightening, step/ramp response timing, and squeeze force.

Per the vault's hard rule, this does NOT instantiate UR3Pick or call
reset()/step() -- those go through mjx.Data + brax. Only plain `mujoco`
(MjModel/MjData/mj_forward/mj_step/mj_resetDataKeyframe) is used here, same
category as test_addvelocity_local.py.

Cases 1-5 are expected RED before the gripper-plant fix lands (ctrlrange
still [0, 0.05], keyframes/behavior tuned to that). Case 6 (T2 regression
guards) is expected GREEN today and after.

Run: python -m pytest mujoco_playground/_src/manipulation/my_ur3/test_gripper_plant.py -q
"""
import os
import sys

import mujoco
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from mujoco_playground._src.manipulation.my_ur3 import ur3_base  # noqa: E402

XML_DIR = os.path.join(os.path.dirname(__file__), "xmls")
PICK_XML = "mjx_single_cube_position_ur3.xml"
PICKNPLACE_XML = "mjx_single_cube_position_ur3_picknplace.xml"
BOTH_SCENES = pytest.mark.parametrize(
    "xml_name",
    [
        PICK_XML,
        pytest.param(
            PICKNPLACE_XML,
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason=(
                    "pre-existing: ur3_pick_sensor.xml references geom "
                    "'lifter', absent from the picknplace scene, so the "
                    "model does not compile"
                ),
            ),
        ),
    ],
    ids=["pick", "picknplace"],
)

GRIPPER_ACT = "hande_fingers_actuator"
LEFT_FINGER_JOINT = "hande_left_finger_joint"
RIGHT_FINGER_JOINT = "hande_right_finger_joint"
JAW_CLOSED_SUM = 0.05  # left + right qpos when both fingers sit at their 0.025 closed limit


def _load_model(xml_name):
    xml_path = os.path.join(XML_DIR, xml_name)
    assets = ur3_base.get_assets()
    xml = open(xml_path).read()
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    model.opt.timestep = 0.005
    return model


def _finger_qposadr(model):
    return (
        model.jnt_qposadr[model.joint(LEFT_FINGER_JOINT).id],
        model.jnt_qposadr[model.joint(RIGHT_FINGER_JOINT).id],
    )


def _jaw_sum(model, data):
    l_adr, r_adr = _finger_qposadr(model)
    return data.qpos[l_adr] + data.qpos[r_adr]


def _box_qposadr(model):
    return model.jnt_qposadr[model.body("box").jntadr[0]]


def _park_cube(model, data):
    qadr = _box_qposadr(model)
    data.qpos[qadr : qadr + 3] = [1.5, 1.5, 0.5]


def _fresh_parked_state(model):
    data = mujoco.MjData(model)
    _park_cube(model, data)
    mujoco.mj_forward(model, data)
    return data


def _cube_between_pads(model, data):
    """Places the box between the open pads at the mount's TCP, matches the
    env's grasp geometry, and parks the table ("lifter") mocap out of the way.
    Returns the tcp world position the box was placed at.
    """
    mujoco.mj_resetDataKeyframe(model, data, model.key("task_home").id)
    l_adr, r_adr = _finger_qposadr(model)
    data.qpos[l_adr] = 0.0
    data.qpos[r_adr] = 0.0
    mujoco.mj_forward(model, data)

    tcp_pos = data.site_xpos[model.site("tcp").id].copy()
    mount_id = model.body("robotiq_hande_mount").id
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.xmat[mount_id])

    qadr = _box_qposadr(model)
    data.qpos[qadr : qadr + 3] = tcp_pos
    data.qpos[qadr + 3 : qadr + 7] = quat

    lifter_mocapid = model.body("lifter").mocapid[0]
    data.mocap_pos[lifter_mocapid] = [1.5, 1.5, -0.5]
    mujoco.mj_forward(model, data)
    return tcp_pos


# ---------------------------------------------------------------------------
# 1. ctrlrange must match the achievable per-joint range (0.025 per finger),
#    not the current 0.05 -- ctrl above 0.025 over-commands the tendon target
#    beyond what either finger joint can reach.
# ---------------------------------------------------------------------------
@BOTH_SCENES
def test_ctrlrange_tightened_to_finger_range(xml_name):
    model = _load_model(xml_name)
    aid = model.actuator(GRIPPER_ACT).id
    np.testing.assert_allclose(model.actuator_ctrlrange[aid], [0.0, 0.025])


# ---------------------------------------------------------------------------
# 2. Every keyframe's gripper ctrl must be compatible with the tightened
#    range above (0.025), independent of whatever actuator_ctrlrange happens
#    to be set to at the time this runs.
# ---------------------------------------------------------------------------
@BOTH_SCENES
def test_keyframes_respect_tightened_ctrlrange(xml_name):
    model = _load_model(xml_name)
    aid = model.actuator(GRIPPER_ACT).id
    tightened_hi = 0.025
    offenders = [
        (model.key(k).name, float(model.key_ctrl[k][aid]))
        for k in range(model.nkey)
        if model.key_ctrl[k][aid] > tightened_hi
    ]
    assert not offenders, f"keyframes exceeding ctrlrange hi={tightened_hi}: {offenders}"


# ---------------------------------------------------------------------------
# 3. Free (unloaded) step response: ctrl slammed to ctrlrange hi at t=0.
#    Must settle to 95% closed within [0.18, 0.26] s and never overshoot the
#    physical 0.05 m jaw opening by more than 0.5 mm.
# ---------------------------------------------------------------------------
def test_free_step_response_pick_scene():
    model = _load_model(PICK_XML)
    data = _fresh_parked_state(model)
    aid = model.actuator(GRIPPER_ACT).id
    hi = model.actuator_ctrlrange[aid][1]
    data.ctrl[aid] = hi

    settle_time = None
    max_jaw = 0.0
    for _ in range(100):
        for _ in range(4):
            mujoco.mj_step(model, data)
        jaw = _jaw_sum(model, data)
        max_jaw = max(max_jaw, jaw)
        if settle_time is None and jaw >= 0.95 * JAW_CLOSED_SUM:
            settle_time = data.time

    assert settle_time is not None, "jaw never reached 95% closed within 100 control steps"
    assert 0.18 <= settle_time <= 0.26, f"settle time {settle_time} s outside [0.18, 0.26]"
    assert max_jaw <= 0.0505, f"jaw overshot to {max_jaw} m"


# ---------------------------------------------------------------------------
# 4. Ramp response: ctrl rises 0.001/control-step (clipped). Must settle to
#    95% closed within [0.50, 0.65] s, and the jaw must never lag the doubled
#    setpoint (tendon->jaw-sum factor) by more than 8 mm.
# ---------------------------------------------------------------------------
def test_ramp_response_pick_scene():
    model = _load_model(PICK_XML)
    data = _fresh_parked_state(model)
    aid = model.actuator(GRIPPER_ACT).id
    lo, hi = model.actuator_ctrlrange[aid]

    ctrl_val = 0.0
    settle_time = None
    max_lag = 0.0
    for _ in range(100):
        data.ctrl[aid] = np.clip(ctrl_val, lo, hi)
        for _ in range(4):
            mujoco.mj_step(model, data)
        jaw = _jaw_sum(model, data)
        max_lag = max(max_lag, 2 * data.ctrl[aid] - jaw)
        if settle_time is None and jaw >= 0.95 * JAW_CLOSED_SUM:
            settle_time = data.time
        ctrl_val += 0.001

    assert settle_time is not None, "jaw never reached 95% closed within 100 control steps"
    assert 0.50 <= settle_time <= 0.65, f"settle time {settle_time} s outside [0.50, 0.65]"
    assert max_lag <= 0.008, f"jaw lagged doubled setpoint by {max_lag} m"


# ---------------------------------------------------------------------------
# 5. Squeeze: box placed between the open pads, ctrl slammed to hi, held for
#    200 raw steps. Force must land in a sane grasp band (4-6.5 N), and the
#    box must not be shoved out from between the pads.
# ---------------------------------------------------------------------------
def test_squeeze_force_and_hold_pick_scene():
    model = _load_model(PICK_XML)
    data = mujoco.MjData(model)
    tcp_pos = _cube_between_pads(model, data)
    aid = model.actuator(GRIPPER_ACT).id
    hi = model.actuator_ctrlrange[aid][1]
    data.ctrl[aid] = hi

    qadr = _box_qposadr(model)
    max_disp = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data)
        max_disp = max(
            max_disp, float(np.linalg.norm(data.qpos[qadr : qadr + 3] - tcp_pos))
        )

    force = abs(data.actuator_force[aid])
    assert 4.0 <= force <= 6.5, f"squeeze force {force} N outside [4.0, 6.5]"
    assert max_disp <= 0.02, f"box displaced {max_disp} m from tcp (held position)"


# ---------------------------------------------------------------------------
# 6. T2 regression guards -- these encode the plant as it already is and must
#    stay green through the ctrlrange/response fix above.
# ---------------------------------------------------------------------------
@BOTH_SCENES
def test_t2_jaw_opening_from_pad_geoms(xml_name):
    model = _load_model(xml_name)
    left = model.geom("left_finger_collision")
    right = model.geom("right_finger_collision")
    opening = (right.pos[0] - right.size[0]) - (left.pos[0] + left.size[0])
    assert opening == pytest.approx(0.0499, abs=1e-4)


@BOTH_SCENES
def test_t2_finger_joint_range(xml_name):
    model = _load_model(xml_name)
    for jname in (LEFT_FINGER_JOINT, RIGHT_FINGER_JOINT):
        np.testing.assert_allclose(model.joint(jname).range, [0.0, 0.025])


@BOTH_SCENES
def test_t2_finger_dof_damping_and_armature(xml_name):
    model = _load_model(xml_name)
    for jname in (LEFT_FINGER_JOINT, RIGHT_FINGER_JOINT):
        dofadr = model.jnt_dofadr[model.joint(jname).id]
        assert model.dof_damping[dofadr] == 15
        assert model.dof_armature[dofadr] == 0.02


@BOTH_SCENES
def test_t2_gripper_actuator_gain_bias_force(xml_name):
    model = _load_model(xml_name)
    aid = model.actuator(GRIPPER_ACT).id
    assert model.actuator_gainprm[aid][0] == 400
    assert model.actuator_biasprm[aid][1] == -400
    assert model.actuator_biasprm[aid][2] == -1
    np.testing.assert_allclose(model.actuator_forcerange[aid], [-130, 130])


@BOTH_SCENES
def test_t2_arm_actuator_gains(xml_name):
    model = _load_model(xml_name)
    arm_actuators = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
    ]
    gains = [model.actuator_gainprm[model.actuator(a).id][0] for a in arm_actuators]
    np.testing.assert_allclose(gains, [1000, 1000, 500, 200, 200, 200])


def test_t2_pick_scene_home_keyframes():
    model = _load_model(PICK_XML)
    expected = {
        "task_home": [
            0, -2.2, 1.6, -1.6, -1.5, 0, 0.0125, 0.0125,
            0.32, 0, 0.115,
            1, 0, 0, 0,
        ],
        "low_home": [
            0, -1.7, 1.6, -1.6, -1.6, -1.5, 0, 0,
            0.32, 0, 0.115,
            1, 0, 0, 0,
        ],
        "tucked": [
            0, -2.45, 2.04, -1.26, -1.57, 0, 0, 0,
            0.32, 0, 0.115,
            1, 0, 0, 0,
        ],
    }
    for key_name, qpos in expected.items():
        np.testing.assert_allclose(
            model.key(key_name).qpos, qpos, atol=1e-9
        )


# ---------------------------------------------------------------------------
# Reviewer-added adversarial case: MuJoCo's own ctrllimited clip must be what
# is doing the work, not just the declared ctrlrange number. If ctrllimited
# ever regressed to "false" (or a caller sends an out-of-range ctrl -- e.g.
# an old checkpoint that still outputs actions scaled to [0, 0.05]), an
# in-range ctrl and an out-of-range ctrl above hi must settle to the exact
# same force: the pre-WP1 bug was precisely that ctrl above the physical
# stroke kept adding squeeze force (a hidden wind-up integrator) instead of
# being clipped away.
# ---------------------------------------------------------------------------
def test_ctrl_above_tightened_range_is_clipped_not_wound_up():
    model = _load_model(PICK_XML)
    aid = model.actuator(GRIPPER_ACT).id
    hi = model.actuator_ctrlrange[aid][1]

    # Sanity: 0.05 is genuinely out of range now (it was in-range pre-WP1).
    assert 0.05 > hi
    assert bool(model.actuator_ctrllimited[aid])

    def _settled_force(ctrl_value):
        data = mujoco.MjData(model)
        _cube_between_pads(model, data)
        data.ctrl[aid] = ctrl_value
        for _ in range(200):
            mujoco.mj_step(model, data)
        return abs(data.actuator_force[aid])

    force_at_hi = _settled_force(hi)
    force_over_range = _settled_force(0.05)

    assert force_over_range == pytest.approx(force_at_hi, abs=1e-6), (
        f"ctrl=0.05 (out of range) settled to {force_over_range} N but "
        f"ctrl={hi} (range hi) settled to {force_at_hi} N -- ctrllimited is "
        f"not clipping the out-of-range command, the pre-WP1 squeeze-force "
        f"wind-up is still reachable"
    )
    assert 4.0 <= force_at_hi <= 6.5, f"squeeze force {force_at_hi} N outside [4.0, 6.5]"
