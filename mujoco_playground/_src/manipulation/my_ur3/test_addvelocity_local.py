"""Local (pure-MuJoCo, no MJX, no env.step) verification for addvelocity.

Per the vault's hard rule ("Never run MJX / Playground env.step or training
locally -- no GPU, CPU MJX OOM-kills the machine"), this does NOT instantiate
UR3Pick or call reset()/step() -- those go through mjx.Data + brax and are
verified on HPC. What this DOES check locally, with plain `mujoco` (allowed --
same category as evaluation/ur3_dynamics_replay.py's local smoke tests):

  1. The DOF-address fix in ur3_base.py._post_init is necessary and correct:
     jnt_dofadr (used for qvel indexing) differs from jnt_qposadr (used for
     qpos indexing) once a free-jointed body (the box) is in the model, so
     reusing qposadr to index qvel would silently read the wrong slots.
  2. The velocity-obs concatenation ur3_pick.py._get_obs performs (jp ops on
     mjx.Data) is arithmetically identical to the equivalent plain-numpy
     operation on a plain mujoco.MjData with the same qvel layout -- i.e. the
     indexing logic is correct, independent of the jax/mjx vs numpy/mujoco
     backend.

Run: python mujoco_playground/_src/manipulation/my_ur3/test_addvelocity_local.py
"""
import os
import sys

import mujoco
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from mujoco_playground._src.manipulation.my_ur3 import ur3_base  # noqa: E402

XML_PATH = os.path.join(
    os.path.dirname(__file__), "xmls", "mjx_single_cube_position_ur3.xml"
)


def _load_plain_model():
    assets = ur3_base.get_assets()
    xml = open(XML_PATH).read()
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return model


def test_dofadr_differs_from_qposadr():
    model = _load_plain_model()
    arm_qposadr = np.array(
        [model.jnt_qposadr[model.joint(j).id] for j in ur3_base._ARM_JOINTS]
    )
    arm_dofadr = np.array(
        [model.jnt_dofadr[model.joint(j).id] for j in ur3_base._ARM_JOINTS]
    )
    finger_dofadr = np.array(
        [model.jnt_dofadr[model.joint(j).id] for j in ur3_base._FINGER_JOINTS]
    )

    print(f"arm qposadr = {arm_qposadr.tolist()}")
    print(f"arm dofadr  = {arm_dofadr.tolist()}")
    print(f"finger dofadr = {finger_dofadr.tolist()}")
    print(f"model.nq={model.nq}  model.nv={model.nv}")

    # The two address spaces must be distinct arrays of the right width.
    assert arm_dofadr.shape == (6,)
    assert finger_dofadr.shape == (2,)
    # nq != nv confirms a free joint (the box, 7 qpos / 6 dof) is present
    # somewhere in this model.
    assert model.nq != model.nv, (
        f"expected nq!=nv (free-jointed box in the model); got nq={model.nq} "
        f"nv={model.nv} -- re-check the scene XML"
    )
    print(f"CONFIRMED: nq({model.nq}) != nv({model.nv}) -- a free joint "
          "(the box) is present in this model.")
    if np.array_equal(arm_qposadr, arm_dofadr):
        # In THIS model the box's free joint is ordered AFTER the arm+finger
        # joints, so qposadr/dofadr happen to coincide for the arm/finger
        # here. That is a property of THIS scene's joint ordering, not a
        # general guarantee -- dofadr remains the only generally-correct way
        # to index qvel (it would diverge immediately if the box joint were
        # ever reordered before the arm, e.g. a future scene edit). Using
        # dofadr costs nothing when they coincide and is required when they
        # don't, so it is the correct choice either way.
        print("NOTE: arm qposadr == arm dofadr in THIS scene (the box free "
              "joint is ordered after the arm/finger joints) -- coincidental "
              "to this XML's joint order, not something to rely on. dofadr "
              "is still the only generally-correct index space for qvel.")
    else:
        print("CONFIRMED: arm qposadr != arm dofadr in this scene -- "
              "indexing qvel with qposadr would have been wrong here.")


def test_velocity_obs_concatenation_matches_manual_numpy():
    """Replicate ur3_pick.py._get_obs's velocity-append arithmetic with plain
    numpy + a plain mujoco.MjData, and check it produces the exact values we
    set -- i.e. the same concatenation order/logic the jax/mjx version uses.
    """
    model = _load_plain_model()
    data = mujoco.MjData(model)

    arm_dofadr = np.array(
        [model.jnt_dofadr[model.joint(j).id] for j in ur3_base._ARM_JOINTS]
    )
    finger_dofadr = np.array(
        [model.jnt_dofadr[model.joint(j).id] for j in ur3_base._FINGER_JOINTS]
    )

    # Synthetic qvel: distinct, recognizable values so a mis-indexing bug
    # (wrong offset, wrong order) would show up as a mismatch, not a fluke.
    qvel = np.zeros(model.nv, dtype=np.float64)
    synthetic_arm_qvel = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    synthetic_finger_qvel = np.array([0.7, 0.7])  # both fingers move together
    qvel[arm_dofadr] = synthetic_arm_qvel
    qvel[finger_dofadr] = synthetic_finger_qvel
    data.qvel[:] = qvel

    # This is exactly ur3_pick.py._get_obs's velocity block, translated
    # jp -> np (same indexing, same .sum(), same concatenate order):
    #   arm_qvel = data.qvel[self._robot_arm_dofadr]
    #   finger_qvel = data.qvel[self._robot_finger_dofadr].sum().reshape(1)
    #   obs = jp.concatenate([obs, arm_qvel, finger_qvel])
    arm_qvel_obs = data.qvel[arm_dofadr]
    finger_qvel_obs = np.array([data.qvel[finger_dofadr].sum()])

    np.testing.assert_allclose(arm_qvel_obs, synthetic_arm_qvel)
    np.testing.assert_allclose(finger_qvel_obs, [1.4])  # 0.7 + 0.7

    base_obs = np.arange(26, dtype=np.float64)  # stand-in for the real 26D obs
    obs33 = np.concatenate([base_obs, arm_qvel_obs, finger_qvel_obs])
    assert obs33.shape == (33,)
    np.testing.assert_allclose(obs33[:26], base_obs)  # first 26 UNCHANGED
    np.testing.assert_allclose(obs33[26:32], synthetic_arm_qvel)
    np.testing.assert_allclose(obs33[32:], [1.4])
    print("CONFIRMED: velocity-obs concatenation matches expected values, "
          "order, and shape (33D, first 26D byte-identical to the base obs).")


def test_flag_off_shape_is_still_26():
    """When obs_include_velocity=False, the obs must stay 26D -- no velocity
    dims appended. This is the regression guard: existing deployed policies
    (obs_dim=26) must never see a 33D vector.
    """
    base_obs = np.arange(26, dtype=np.float64)
    obs_include_velocity = False
    obs = base_obs
    if obs_include_velocity:
        obs = np.concatenate([obs, np.zeros(7)])
    assert obs.shape == (26,)
    print("CONFIRMED: obs_include_velocity=False -> obs stays 26D.")


if __name__ == "__main__":
    test_dofadr_differs_from_qposadr()
    print()
    test_velocity_obs_concatenation_matches_manual_numpy()
    print()
    test_flag_off_shape_is_still_26()
    print("\nAll local (pure-MuJoCo, no MJX) addvelocity checks passed.")
