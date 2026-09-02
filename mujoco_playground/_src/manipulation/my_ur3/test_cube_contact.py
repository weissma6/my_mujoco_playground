"""Local (pure-MuJoCo, no MJX, no env.step) verification of the cube-contact
stiffening (box solimp, contact-point budget, squeeze penetration).

Per the vault's hard rule, this does NOT instantiate UR3Pick or call
reset()/step() -- those go through mjx.Data + brax. Only plain `mujoco`
(MjModel/MjData/mj_forward/mj_step/mj_resetDataKeyframe) is used here, same
category as test_gripper_plant.py.

Cases 1-3 are expected RED before the cube-contact fix lands (box solimp
still the MuJoCo default, max_contact_points still 12, air-squeeze
penetration still ~0.6 mm). Cases 4-6 are expected GREEN today and after.

Run: python -m pytest mujoco_playground/_src/manipulation/my_ur3/test_cube_contact.py -q
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
PAD_GEOMS = ("left_finger_collision", "right_finger_collision")
BOX_SOLIMP = [0.99, 0.995, 0.001, 0.5, 2.0]


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


def _box_qposadr(model):
    return model.jnt_qposadr[model.body("box").jntadr[0]]


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


def _cube_on_table_between_pads(model, data):
    """As _cube_between_pads, but the table is raised under the cube's
    resting face instead of parked out of the way: table top (2.5 mm
    half-thickness) sits under the cube's bottom face (2 cm half-height).
    """
    tcp_pos = _cube_between_pads(model, data)
    lifter_mocapid = model.body("lifter").mocapid[0]
    data.mocap_pos[lifter_mocapid] = [0.4, 0.0, tcp_pos[2] - 0.0225]
    data.mocap_quat[lifter_mocapid] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return tcp_pos


def _pad_box_penetration(model, data):
    """Deepest pad-box penetration among active contacts (0 if none)."""
    box_id = model.geom("box").id
    pad_ids = {model.geom(g).id for g in PAD_GEOMS}
    deepest = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if box_id in pair and pair & pad_ids:
            deepest = max(deepest, -c.dist)
    return deepest


def _box_contact_count(model, data):
    box_id = model.geom("box").id
    return sum(
        1
        for i in range(data.ncon)
        if box_id in (data.contact[i].geom1, data.contact[i].geom2)
    )


def _sensor_value(model, data, name):
    adr = model.sensor_adr[model.sensor(name).id]
    return data.sensordata[adr]


# ---------------------------------------------------------------------------
# 1. Box solimp must be stiffened to match the pads (default MuJoCo solimp
#    lets the cube burrow into the pads under squeeze).
# ---------------------------------------------------------------------------
@BOTH_SCENES
def test_box_solimp_matches_pads(xml_name):
    model = _load_model(xml_name)
    box_solimp = model.geom("box").solimp
    np.testing.assert_allclose(box_solimp, BOX_SOLIMP)
    for pad in PAD_GEOMS:
        np.testing.assert_allclose(model.geom(pad).solimp, BOX_SOLIMP)


# ---------------------------------------------------------------------------
# 2. Contact-point budget: stiffened solimp needs more contact points per
#    pair to resolve the box-pad interpenetration without popping through.
# ---------------------------------------------------------------------------
def test_max_contact_points_numeric_pick_scene():
    model = _load_model(PICK_XML)
    numeric_id = model.numeric("max_contact_points").id
    adr = model.numeric_adr[numeric_id]
    assert model.numeric_data[adr] == 24


# ---------------------------------------------------------------------------
# 3. Air squeeze: cube between the pads (no table), ctrl slammed to hi, held
#    for 200 raw steps. Penetration must settle low and never blow past a
#    generous transient bound; the box must stay held in place.
# ---------------------------------------------------------------------------
def test_air_squeeze_penetration_pick_scene():
    model = _load_model(PICK_XML)
    data = mujoco.MjData(model)
    tcp_pos = _cube_between_pads(model, data)
    aid = model.actuator(GRIPPER_ACT).id
    hi = model.actuator_ctrlrange[aid][1]
    data.ctrl[aid] = hi

    qadr = _box_qposadr(model)
    penetrations = []
    max_disp = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data)
        penetrations.append(_pad_box_penetration(model, data))
        max_disp = max(
            max_disp, float(np.linalg.norm(data.qpos[qadr : qadr + 3] - tcp_pos))
        )

    steady_max = max(penetrations[-20:])
    overall_max = max(penetrations)
    assert steady_max <= 0.0003, f"steady-state penetration {steady_max} m > 0.3 mm"
    assert overall_max <= 0.0020, f"peak penetration {overall_max} m > 2 mm"
    assert max_disp <= 0.02, f"box displaced {max_disp} m from tcp placement"


# ---------------------------------------------------------------------------
# 4. Table grasp census: cube resting on the table between the pads, ctrl
#    slammed to hi, held for 200 raw steps. Both finger-box contact sensors
#    must fire, and the box-geom contact count must stay within budget.
# ---------------------------------------------------------------------------
def test_table_grasp_contact_census_pick_scene():
    model = _load_model(PICK_XML)
    data = mujoco.MjData(model)
    _cube_on_table_between_pads(model, data)
    aid = model.actuator(GRIPPER_ACT).id
    hi = model.actuator_ctrlrange[aid][1]
    data.ctrl[aid] = hi

    for _ in range(200):
        mujoco.mj_step(model, data)

    left = _sensor_value(model, data, "left_finger_box_contact")
    right = _sensor_value(model, data, "right_finger_box_contact")
    assert left > 0, "left_finger_box_contact sensor did not fire"
    assert right > 0, "right_finger_box_contact sensor did not fire"
    n_box_contacts = _box_contact_count(model, data)
    assert n_box_contacts <= 24, f"{n_box_contacts} box contacts exceed budget of 24"


# ---------------------------------------------------------------------------
# 5. Resting cube: task_home keyframe, run 100 raw steps at the keyframe's
#    own ctrl. The box must stay put on the table (no jitter, no sinking).
# ---------------------------------------------------------------------------
def test_resting_cube_stays_put_pick_scene():
    model = _load_model(PICK_XML)
    data = mujoco.MjData(model)
    key_id = model.key("task_home").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    aid = model.actuator(GRIPPER_ACT).id
    data.ctrl[:] = model.key_ctrl[key_id]
    mujoco.mj_forward(model, data)

    qadr = _box_qposadr(model)
    expected = np.array([0.32, 0.0, 0.115])
    for _ in range(100):
        mujoco.mj_step(model, data)
        np.testing.assert_allclose(
            data.qpos[qadr : qadr + 3], expected, atol=1e-3
        )


# ---------------------------------------------------------------------------
# 6. T2 regression guards -- these encode the cube/contact setup as it
#    already is and must stay green through the solimp/budget fix above.
# ---------------------------------------------------------------------------
@BOTH_SCENES
def test_t2_box_material_and_contact_props(xml_name):
    model = _load_model(xml_name)
    box = model.geom("box")
    np.testing.assert_allclose(box.friction, [1, 0.03, 0.003])
    np.testing.assert_allclose(box.solref, [0.01, 1])
    assert box.condim == 4
    assert box.contype == 1
    assert box.conaffinity == 3


@BOTH_SCENES
def test_t2_pad_solimp_unchanged(xml_name):
    model = _load_model(xml_name)
    for pad in PAD_GEOMS:
        np.testing.assert_allclose(model.geom(pad).solimp, BOX_SOLIMP)


def test_t2_pick_scene_box_and_lifter_layout():
    model = _load_model(PICK_XML)
    np.testing.assert_allclose(model.geom("box").size, [0.015, 0.015, 0.02])
    assert model.body("box").mass[0] == pytest.approx(0.036, abs=1e-6)
    np.testing.assert_allclose(model.body("lifter").pos, [0.4, 0, 0.0925])
    np.testing.assert_allclose(model.geom("lifter").size, [0.30, 0.50, 0.0025])


def test_t2_pick_scene_sensors_resolve():
    model = _load_model(PICK_XML)
    names = [
        "left_finger_pad_floor_found",
        "right_finger_pad_floor_found",
        "hand_capsule_floor_found",
        "left_finger_pad_lifter_found",
        "right_finger_pad_lifter_found",
        "hand_capsule_lifter_found",
        "box_hand_found",
        "left_finger_box_contact",
        "right_finger_box_contact",
        "tcp_position",
        "box_position",
    ]
    for name in names:
        assert model.sensor(name).id >= 0


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "pre-existing: ur3_pick_sensor.xml references geom 'lifter', "
        "absent from the picknplace scene, so the model does not compile"
    ),
)
def test_t2_picknplace_box_size():
    model = _load_model(PICKNPLACE_XML)
    np.testing.assert_allclose(model.geom("box").size, [0.02, 0.02, 0.02])


# ---------------------------------------------------------------------------
# REVIEWER (WP2) adversarial additions -- appended, existing tests above are
# unmodified. Plain MuJoCo only, no MJX/env.step, per the vault hard rule.
# ---------------------------------------------------------------------------
def _box_table_penetration(model, data):
    """Deepest box-lifter (table) penetration among active contacts."""
    box_id = model.geom("box").id
    lifter_id = model.geom("lifter").id
    deepest = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if pair == {box_id, lifter_id}:
            deepest = max(deepest, -c.dist)
    return deepest


def test_box_table_resting_penetration_not_worse_pick_scene():
    """The box solimp stiffening (spec item 1) only touches the box geom, not
    the table ('lifter'), which keeps the MuJoCo-default solimp. Guard that
    resting the (now stiffer) box on the (still-default) table does not
    regress: steady-state penetration must stay well under the 1.9/0.6 mm
    figures quoted for the old pad-cube softness, comfortably under 0.5 mm.
    """
    model = _load_model(PICK_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("task_home").id)
    # Open the fingers fully so they don't grip -- isolates box-table contact.
    l_adr, r_adr = _finger_qposadr(model)
    data.qpos[l_adr] = model.jnt_range[model.joint(LEFT_FINGER_JOINT).id][1]
    data.qpos[r_adr] = model.jnt_range[model.joint(RIGHT_FINGER_JOINT).id][1]
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    penetrations = []
    for _ in range(300):
        mujoco.mj_step(model, data)
        penetrations.append(_box_table_penetration(model, data))

    steady_max = max(penetrations[-20:])
    assert steady_max <= 0.0005, (
        f"steady-state box-table penetration {steady_max} m > 0.5 mm"
    )


def test_pad_box_friction_still_max_rule_pick_scene():
    """Friction attributes on box/pads are untouched by this diff (spec item
    1 only adds solimp); confirm the contact-level friction MuJoCo derives
    via the componentwise-max mixing rule is still dominated by the pad's
    2.0 sliding friction, unaffected by the box's new stiffer solimp.
    """
    model = _load_model(PICK_XML)
    data = mujoco.MjData(model)
    tcp_pos = _cube_between_pads(model, data)
    del tcp_pos
    aid = model.actuator(GRIPPER_ACT).id
    hi = model.actuator_ctrlrange[aid][1]
    data.ctrl[aid] = hi
    for _ in range(50):
        mujoco.mj_step(model, data)

    box_id = model.geom("box").id
    pad_ids = {model.geom(g).id for g in PAD_GEOMS}
    found = False
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if box_id in pair and pair & pad_ids:
            found = True
            assert c.friction[0] == pytest.approx(2.0), (
                f"pad-box sliding friction {c.friction[0]} != 2.0 (max rule)"
            )
    assert found, "no pad-box contact formed to check friction on"
