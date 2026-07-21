"""Dynamics-replay sim-to-real error for logged UR3 pick runs (Aljalbout Sec. 5.1).

    E_replay = (1/T) * sum_t || s_t^sim - s_t^real ||^2

Two replay modes, both plain MuJoCo (mj_step on CPU -- NO MJX; per CLAUDE.md
MJX/env.step must never run locally, but mj_step forward dynamics is cheap and
expected):

  * ONE-STEP: for each tick t, set MjData to the REAL state at t, apply the
    logged command for t, integrate for the logged real loop duration
    loop_dt_true_s[t] (NOT a fixed 20 ms), and compare the predicted state to
    the real state at t+1. Measures single-step model fidelity, re-synced every
    tick.
  * OPEN-LOOP: set MjData to the real init once, then drive the WHOLE logged
    command sequence with no re-sync. Error grows with t; we report the error
    vs step and the TIME-TO-DIVERGENCE.

Time-to-divergence (not a standard term -- defined here explicitly): the first
tick t at which the open-loop TCP position error first exceeds
`DIVERGENCE_TCP_TOL_M` (1 cm). It is a threshold-crossing index, reported as
(step, seconds); NaN if the threshold is never crossed within the log.

Mirroring real servoJ (documented assumption)
---------------------------------------------
On the real robot each policy tick sends ONE servoJ position target and the
low-level controller drives toward it until the next tick. If a tick stalls
(loop_dt_true_s large), the arm keeps driving to that SAME last target -- it
does not interpolate to a new one. We mirror this exactly: the arm position-
actuator ctrl (and the gripper tendon ctrl) is HELD CONSTANT across every
MuJoCo substep of the tick's integration window. We never interpolate a target
within a tick.

Which command is applied (documented assumption)
------------------------------------------------
The state CSV logs, per tick, both the raw policy `action0..6` and the realized
servoJ target `ctrl0..5` (+ the gripper tendon target `gripper_ctrl`). The
default replay applies the LOGGED ctrl -- that is precisely what servoJ
received, with the deploy controller's action_scale, alpha-blend and velocity
clamp already baked in, so it needs no re-derivation and cannot drift from the
controller. ("Apply the logged action" is thus realized through the logged
servoJ target the action produced.) `command_mode="action"` instead
reconstructs the ARM target as clip(q_real[t] + action_scale*action[t]) with
action_scale read from the run's meta.json (never hardcoded); the gripper is an
integrator on the real robot (gripper_ctrl += gripper_action_scale*action6),
not reconstructible from a single action, so it always uses the logged
gripper_ctrl.

Joint space vs task space
-------------------------
The headline E_replay is in ARM JOINT space (rad^2) -- the 6 controlled joints
are the cleanest apples-to-apples comparison (both sides are UR joint angles;
the deploy assumes joint-space identity between the robot and this XML). We also
report TCP position RMS (m) and box position RMS (m). TCP for BOTH sides is read
through evaluation/ur3_reward_replay.SimFK (a single FK definition -- no FK is
re-implemented here): the real TCP from the real logged joints, the sim TCP from
the stepped MjData's own `tcp` site (same site id, same frame). Box error is
scored only where the mocap box reading is valid.

Local usage:
    python evaluation/ur3_dynamics_replay.py                 # smoke test
    python evaluation/ur3_dynamics_replay.py --run_dir \
        robots/UR3e/real_robot_results/DR_cube_mass_light_1636c989/20260717_150159
"""

import argparse
import dataclasses
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse SimFK (and the default scene path) -- do NOT duplicate forward kinematics.
from evaluation.ur3_reward_replay import (  # noqa: E402
    SimFK,
    _ARM_JOINTS,
    _FINGER_JOINTS,
    _DEFAULT_XML,
)

DIVERGENCE_TCP_TOL_M = 0.01  # 1 cm -- open-loop time-to-divergence threshold.
_NOMINAL_SIM_DT = 0.005  # ur3_pick default_config sim_dt; substep granularity.


@dataclasses.dataclass
class ReplayTrajectory:
    """A logged run reduced to the fields the replay needs. All arrays length T."""

    arm_q: np.ndarray       # (T, 6) joint angles, rad
    finger: np.ndarray      # (T,)   per-finger opening, m in [0, 0.025]
    box_pos: np.ndarray     # (T, 3) world m
    box_quat: np.ndarray    # (T, 4) (w, x, y, z)
    arm_ctrl: np.ndarray    # (T, 6) servoJ target joint angles for tick t
    gripper_ctrl: np.ndarray  # (T,)  tendon target, m in [0, 0.05]
    dt: np.ndarray          # (T,)   real loop duration for tick t (loop_dt_true_s)
    box_valid: np.ndarray   # (T,)   bool: mocap box reading usable this tick
    action: Optional[np.ndarray] = None  # (T, 7) raw policy action (command_mode="action")
    qvel: Optional[np.ndarray] = None    # (T, nv) full velocity (self-replay only)

    def __len__(self) -> int:
        return len(self.arm_q)


class DynamicsReplay:
    """Plain-MuJoCo (mj_step) one-step and open-loop replay of a logged run."""

    def __init__(self, xml_path: str = None):
        import mujoco  # local import: keep mujoco off the module import path.

        self._mujoco = mujoco
        self.xml_path = xml_path or _DEFAULT_XML
        # SimFK owns the single FK definition (TCP/finger sites); we borrow it
        # for the REAL-side TCP. A SEPARATE model/data does the mj_step
        # integration so stepping never fights SimFK's per-call mj_forward.
        self.fk = SimFK(self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.tcp_site = self.model.site("tcp").id
        self.box_body = self.model.body("box").id

        self.arm_qadr = np.array(
            [self.model.jnt_qposadr[self.model.joint(j).id] for j in _ARM_JOINTS]
        )
        self.arm_dofadr = np.array(
            [self.model.jnt_dofadr[self.model.joint(j).id] for j in _ARM_JOINTS]
        )
        self.finger_qadr = np.array(
            [self.model.jnt_qposadr[self.model.joint(j).id] for j in _FINGER_JOINTS]
        )
        box_jntadr = self.model.body("box").jntadr[0]
        self.box_qadr = int(self.model.jnt_qposadr[box_jntadr])
        self.qpos0 = self.model.qpos0.copy()
        # Arm actuators are indices 0..5, the gripper tendon actuator is 6
        # (verified from the scene: nu=7, hande_fingers_actuator last).
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

    # -- state I/O ---------------------------------------------------------

    def _set_state(self, arm_q, finger_m, box_pos, box_quat, qvel=None):
        d = self.data
        d.qpos[:] = self.qpos0
        d.qpos[self.arm_qadr] = np.asarray(arm_q, dtype=float)
        d.qpos[self.finger_qadr] = float(finger_m)
        d.qpos[self.box_qadr : self.box_qadr + 3] = np.asarray(box_pos, dtype=float)
        q = np.asarray(box_quat, dtype=float)
        n = np.linalg.norm(q)
        d.qpos[self.box_qadr + 3 : self.box_qadr + 7] = (
            q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
        )
        # Real logs carry NO joint velocity -> reset to 0 (documented limitation:
        # one-step real error then also absorbs the unknown initial velocity).
        # Self-replay passes the true qvel so its one-step error is ~0.
        d.qvel[:] = 0.0 if qvel is None else np.asarray(qvel, dtype=float)
        d.ctrl[:] = 0.0  # ctrl is set per tick in _apply_and_step.
        self._mujoco.mj_forward(self.model, d)

    def _apply_and_step(self, arm_ctrl, gripper_ctrl, dt):
        """Hold the tick's ctrl constant and integrate exactly `dt` seconds.

        Substep granularity ~ _NOMINAL_SIM_DT; the effective timestep is dt/nsub
        so the integration window matches the REAL loop duration rather than a
        fixed 20 ms. servoJ mirror: ctrl is held across all nsub substeps.
        """
        d = self.data
        ctrl = np.empty(self.model.nu)
        ctrl[:6] = np.asarray(arm_ctrl, dtype=float)
        ctrl[6] = float(gripper_ctrl)
        ctrl = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        d.ctrl[:] = ctrl
        nsub = max(1, int(round(float(dt) / _NOMINAL_SIM_DT)))
        self.model.opt.timestep = float(dt) / nsub
        for _ in range(nsub):
            self._mujoco.mj_step(self.model, d)
        # After mj_step, site_xpos/xpos are stale (kinematics are computed at the
        # START of a step, before integration). Re-sync so the tcp/box we read
        # match the final qpos rather than lagging one substep.
        self._mujoco.mj_forward(self.model, d)

    def _read_sim(self) -> dict:
        d = self.data
        return {
            "arm_q": d.qpos[self.arm_qadr].copy(),
            "finger": float(d.qpos[self.finger_qadr[0]]),
            "box_pos": d.xpos[self.box_body].copy(),
            "tcp": d.site_xpos[self.tcp_site].copy(),
        }

    def _real_tcp(self, arm_q, finger_m, box_pos, box_quat) -> np.ndarray:
        # Real-side TCP through SimFK -> identical site/frame as the sim side.
        return self.fk.geom(arm_q, finger_m, box_pos, box_quat)["tcp"]

    # -- replays -----------------------------------------------------------

    def one_step(self, traj: ReplayTrajectory) -> pd.DataFrame:
        """Re-synced single-step prediction error at every tick."""
        rows = []
        T = len(traj)
        for t in range(T - 1):
            qvel = None if traj.qvel is None else traj.qvel[t]
            self._set_state(
                traj.arm_q[t], traj.finger[t], traj.box_pos[t], traj.box_quat[t],
                qvel=qvel,
            )
            self._apply_and_step(traj.arm_ctrl[t], traj.gripper_ctrl[t], traj.dt[t])
            sim = self._read_sim()

            joint_err = sim["arm_q"] - traj.arm_q[t + 1]
            real_tcp = self._real_tcp(
                traj.arm_q[t + 1], traj.finger[t + 1],
                traj.box_pos[t + 1], traj.box_quat[t + 1],
            )
            tcp_err = float(np.linalg.norm(sim["tcp"] - real_tcp))
            box_ok = bool(traj.box_valid[t] and traj.box_valid[t + 1])
            box_err = (
                float(np.linalg.norm(sim["box_pos"] - traj.box_pos[t + 1]))
                if box_ok else np.nan
            )
            rows.append({
                "step": t + 1,
                "joint_sq_err": float(np.dot(joint_err, joint_err)),
                "joint_rms": float(np.sqrt(np.mean(joint_err ** 2))),
                "tcp_err_m": tcp_err,
                "box_err_m": box_err,
            })
        return pd.DataFrame(rows)

    def open_loop(self, traj: ReplayTrajectory) -> pd.DataFrame:
        """Free-running prediction error from the init, no re-sync."""
        qvel0 = None if traj.qvel is None else traj.qvel[0]
        self._set_state(
            traj.arm_q[0], traj.finger[0], traj.box_pos[0], traj.box_quat[0],
            qvel=qvel0,
        )
        rows = []
        T = len(traj)
        for t in range(T - 1):
            self._apply_and_step(traj.arm_ctrl[t], traj.gripper_ctrl[t], traj.dt[t])
            sim = self._read_sim()

            joint_err = sim["arm_q"] - traj.arm_q[t + 1]
            real_tcp = self._real_tcp(
                traj.arm_q[t + 1], traj.finger[t + 1],
                traj.box_pos[t + 1], traj.box_quat[t + 1],
            )
            tcp_err = float(np.linalg.norm(sim["tcp"] - real_tcp))
            box_ok = bool(traj.box_valid[t + 1])
            box_err = (
                float(np.linalg.norm(sim["box_pos"] - traj.box_pos[t + 1]))
                if box_ok else np.nan
            )
            rows.append({
                "step": t + 1,
                "joint_sq_err": float(np.dot(joint_err, joint_err)),
                "joint_rms": float(np.sqrt(np.mean(joint_err ** 2))),
                "tcp_err_m": tcp_err,
                "box_err_m": box_err,
            })
        return pd.DataFrame(rows)


def e_replay(df: pd.DataFrame, column: str = "joint_sq_err") -> float:
    """E_replay = mean over ticks of the per-step squared error (Aljalbout 5.1).

    Default column is the arm-joint squared error (rad^2). NaNs (masked box
    ticks) are ignored so box E_replay is over valid ticks only.
    """
    return float(np.nanmean(df[column].to_numpy()))


def time_to_divergence(df: pd.DataFrame, tol_m: float = DIVERGENCE_TCP_TOL_M):
    """First tick whose open-loop TCP error exceeds `tol_m` (a threshold crossing).

    Returns (step, None) with step=NaN if the threshold is never crossed.
    """
    over = df.index[df["tcp_err_m"].to_numpy() > tol_m]
    if len(over) == 0:
        return float("nan")
    return int(df["step"].iloc[over[0]])


# ===========================================================================
# Real-log loader
# ===========================================================================


def _box_valid_mask(box_pos: np.ndarray, box_quat: np.ndarray) -> np.ndarray:
    """Mocap box reading usable: all box fields finite AND quat not all-zero.

    run_policy_loop carries the last good mocap value forward on a dropped
    frame, so an all-zero / non-finite quaternion is the only in-band signal
    that no valid orientation was ever seen; a finite carried-forward pose is
    treated as valid (it is the best estimate available for that tick).
    """
    finite = np.isfinite(box_pos).all(1) & np.isfinite(box_quat).all(1)
    quat_nonzero = np.linalg.norm(box_quat, axis=1) > 1e-6
    return finite & quat_nonzero


def load_real_trajectory(path: str, meta: dict = None) -> ReplayTrajectory:
    """Load a logged real run (ur3_pick_states.csv, or its containing folder)."""
    csv_path = path
    if os.path.isdir(path):
        csv_path = os.path.join(path, "ur3_pick_states.csv")
    df = pd.read_csv(csv_path)

    arm_q = df[[f"q{i}" for i in range(6)]].to_numpy(float)
    finger = df["finger_pos_est"].to_numpy(float)
    box_pos = df[["box_x", "box_y", "box_z"]].to_numpy(float)
    box_quat = df[["box_qw", "box_qx", "box_qy", "box_qz"]].to_numpy(float)
    arm_ctrl = df[[f"ctrl{i}" for i in range(6)]].to_numpy(float)
    gripper_ctrl = df["gripper_ctrl"].to_numpy(float)
    dt = df["loop_dt_true_s"].to_numpy(float)
    action = (
        df[[f"action{i}" for i in range(7)]].to_numpy(float)
        if all(f"action{i}" in df.columns for i in range(7)) else None
    )
    return ReplayTrajectory(
        arm_q=arm_q, finger=finger, box_pos=box_pos, box_quat=box_quat,
        arm_ctrl=arm_ctrl, gripper_ctrl=gripper_ctrl, dt=dt,
        box_valid=_box_valid_mask(box_pos, box_quat), action=action, qvel=None,
    )


def apply_action_command_mode(
    traj: ReplayTrajectory, action_scale: float, ctrl_lo, ctrl_hi
) -> ReplayTrajectory:
    """Rebuild arm_ctrl from the raw action: clip(q_real[t] + action_scale*action).

    The gripper stays on the logged gripper_ctrl (it is an integrator on the
    real robot, not reconstructible from one action). action_scale MUST come
    from the run's meta.json -- never hardcode it.
    """
    if traj.action is None:
        raise ValueError("command_mode='action' needs action0..6 columns.")
    arm_ctrl = np.clip(
        traj.arm_q + action_scale * traj.action[:, :6], ctrl_lo[:6], ctrl_hi[:6]
    )
    return dataclasses.replace(traj, arm_ctrl=arm_ctrl)


def _load_meta(run_dir: str) -> dict:
    for name in ("ur3_pick_meta.json",):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def replay_real_run(
    run_dir: str, command_mode: str = "ctrl", xml_path: str = None
) -> dict:
    """One-step + open-loop replay of one real run folder. Returns a summary."""
    rep = DynamicsReplay(xml_path)
    traj = load_real_trajectory(run_dir)
    meta = _load_meta(run_dir) if os.path.isdir(run_dir) else {}

    if command_mode == "action":
        action_scale = meta.get("action_scale", None)
        if action_scale is None:
            raise ValueError(
                "command_mode='action' requires 'action_scale' in the run's "
                "ur3_pick_meta.json (never hardcoded)."
            )
        traj = apply_action_command_mode(
            traj, float(action_scale), rep.ctrl_lo, rep.ctrl_hi
        )

    os_df = rep.one_step(traj)
    ol_df = rep.open_loop(traj)
    ttd = time_to_divergence(ol_df)
    dt_mean = float(np.mean(traj.dt))
    return {
        "run_dir": run_dir,
        "n_ticks": len(traj),
        "command_mode": command_mode,
        "one_step": os_df,
        "open_loop": ol_df,
        "E_replay_onestep_joint_rad2": e_replay(os_df, "joint_sq_err"),
        "E_replay_openloop_joint_rad2": e_replay(ol_df, "joint_sq_err"),
        "onestep_tcp_rms_m": float(np.sqrt(np.nanmean(os_df["tcp_err_m"] ** 2))),
        "onestep_box_rms_m": float(np.sqrt(np.nanmean(os_df["box_err_m"] ** 2))),
        "time_to_divergence_step": ttd,
        "time_to_divergence_s": (
            float("nan") if ttd != ttd else ttd * dt_mean
        ),
    }


# ===========================================================================
# Smoke test -- generate a sim trajectory, replay it against ITSELF. One-step
# error must be ~0. Pure MuJoCo mj_step, no MJX, no robot. Safe locally.
# ===========================================================================


def _generate_self_trajectory(rep: DynamicsReplay, n: int = 40) -> ReplayTrajectory:
    """Integrate the scene from task_home with a gentle command sequence,
    logging the full state each tick so a self-replay can reproduce it exactly.
    """
    import mujoco

    model, data = rep.model, rep.data
    key = model.key("task_home")
    home_arm = key.qpos[rep.arm_qadr].copy()
    home_finger = float(key.qpos[rep.finger_qadr[0]])

    # Init exactly at the keyframe.
    data.qpos[:] = key.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[:6] = home_arm
    data.ctrl[6] = home_finger
    mujoco.mj_forward(model, data)

    arm_q, finger, box_pos, box_quat = [], [], [], []
    arm_ctrl, gripper_ctrl, dts, qvels = [], [], [], []
    dt = 0.02  # nominal 50 Hz tick (-> 4 substeps at 0.005)
    for t in range(n):
        # Log the state at the START of tick t (this is s_t).
        arm_q.append(data.qpos[rep.arm_qadr].copy())
        finger.append(float(data.qpos[rep.finger_qadr[0]]))
        box_pos.append(data.xpos[rep.box_body].copy())
        box_quat.append(data.qpos[rep.box_qadr + 3 : rep.box_qadr + 7].copy())
        qvels.append(data.qvel.copy())
        # A small, smooth per-joint command wiggle + a slow finger close.
        cmd = home_arm + 0.05 * np.sin(0.3 * t + np.arange(6))
        g = min(0.05, 0.001 * t)
        arm_ctrl.append(cmd.copy())
        gripper_ctrl.append(g)
        dts.append(dt)
        rep._apply_and_step(cmd, g, dt)
    # One trailing sample so tick n-1 has a t+1 to compare against.
    arm_q.append(data.qpos[rep.arm_qadr].copy())
    finger.append(float(data.qpos[rep.finger_qadr[0]]))
    box_pos.append(data.xpos[rep.box_body].copy())
    box_quat.append(data.qpos[rep.box_qadr + 3 : rep.box_qadr + 7].copy())
    qvels.append(data.qvel.copy())
    arm_ctrl.append(arm_ctrl[-1].copy())
    gripper_ctrl.append(gripper_ctrl[-1])
    dts.append(dt)

    box_pos = np.array(box_pos)
    box_quat = np.array(box_quat)
    return ReplayTrajectory(
        arm_q=np.array(arm_q), finger=np.array(finger), box_pos=box_pos,
        box_quat=box_quat, arm_ctrl=np.array(arm_ctrl),
        gripper_ctrl=np.array(gripper_ctrl), dt=np.array(dts),
        box_valid=_box_valid_mask(box_pos, box_quat),
        action=None, qvel=np.array(qvels),
    )


def _smoke_test():
    rep = DynamicsReplay()
    traj = _generate_self_trajectory(rep, n=40)
    print(f"generated self-trajectory: {len(traj)} ticks, dt={traj.dt[0]}s")

    os_df = rep.one_step(traj)
    ol_df = rep.open_loop(traj)

    e_os = e_replay(os_df, "joint_sq_err")
    max_os_tcp = float(np.nanmax(os_df["tcp_err_m"]))
    e_ol = e_replay(ol_df, "joint_sq_err")
    ttd = time_to_divergence(ol_df)

    print("\n--- ONE-STEP self-replay (must be ~0) ---")
    print(f"E_replay (joint, rad^2) = {e_os:.3e}")
    print(f"max one-step TCP err (m) = {max_os_tcp:.3e}")
    print("\n--- OPEN-LOOP self-replay (also ~0: identical model/integrator) ---")
    print(f"E_replay (joint, rad^2) = {e_ol:.3e}")
    print(f"time-to-divergence (>1cm TCP) = {ttd} (NaN => never diverged)")

    # Self-replay re-runs the identical integrator, so error is ~0 up to the
    # constraint-solver warmstart reset set_state forces (~1e-12 rad^2, well
    # below any real sim-to-real gap). Not bit-exact, but "one-step error ~ 0".
    assert e_os < 1e-6, f"one-step self-replay not ~0: {e_os}"
    assert max_os_tcp < 1e-4, f"one-step TCP err not ~0: {max_os_tcp}"
    assert e_ol < 1e-6, f"open-loop self-replay not ~0: {e_ol}"
    assert ttd != ttd, f"self-replay should never diverge, got ttd={ttd}"
    print("\nSMOKE TEST OK (pure MuJoCo mj_step, no MJX, no robot).")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default=None,
                    help="a real run folder (or its ur3_pick_states.csv). Omit "
                         "to run the self-replay smoke test.")
    ap.add_argument("--command_mode", choices=["ctrl", "action"], default="ctrl")
    ap.add_argument("--glob", default=None,
                    help="glob of run folders to batch (e.g. "
                         "'robots/UR3e/real_robot_results/gap_protocol/*/*/ep*_rep*')")
    ap.add_argument("--out_csv", default=None,
                    help="write per-step one-step+open-loop errors for --run_dir here")
    args = ap.parse_args()

    if not args.run_dir and not args.glob:
        _smoke_test()
        return

    run_dirs = sorted(glob.glob(args.glob)) if args.glob else [args.run_dir]
    for rd in run_dirs:
        summ = replay_real_run(rd, command_mode=args.command_mode)
        print(f"\n{rd}  ({summ['n_ticks']} ticks, mode={summ['command_mode']})")
        print(f"  E_replay one-step  (joint rad^2) = "
              f"{summ['E_replay_onestep_joint_rad2']:.4e}")
        print(f"  E_replay open-loop (joint rad^2) = "
              f"{summ['E_replay_openloop_joint_rad2']:.4e}")
        print(f"  one-step TCP RMS (m) = {summ['onestep_tcp_rms_m']:.4e}   "
              f"box RMS (m) = {summ['onestep_box_rms_m']:.4e}")
        print(f"  time-to-divergence = step {summ['time_to_divergence_step']} "
              f"(~{summ['time_to_divergence_s']:.3f} s)")
        if args.out_csv and not args.glob:
            os_df = summ["one_step"].add_prefix("onestep_")
            ol_df = summ["open_loop"].add_prefix("openloop_")
            merged = pd.concat([os_df, ol_df], axis=1)
            merged.to_csv(args.out_csv, index=False)
            print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    _cli()
