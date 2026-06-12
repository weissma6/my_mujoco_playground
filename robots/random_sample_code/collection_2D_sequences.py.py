#!/usr/bin/env python3
"""
DLO Sim-to-Real Data Collection
────────────────────────────────
Active sequences:
  seq01_static_baseline           — sit still 30s (gravity equilibrium)
  seq02_random_walk_slow_X        — random walk on X axis only, slow
  seq03_random_walk_fast_X        — random walk on X axis only, fast
  seq04_random_walk_slow_Z        — random walk on Z axis only, slow
  seq05_random_walk_fast_Z        — random walk on Z axis only, fast
  seq09_random_walk_slow_XZ       — random walk in XZ plane, slow
  seq10_random_walk_fast_XZ       — random walk in XZ plane, fast
  seq11_sinusoid_xz_XZ            — sinusoid in XZ plane
  seq12_step_inputs_xz_XZ         — step inputs on X and Z
  seq13_random_walk_max_reach_XZ  — explores max reachable workspace in XZ plane

Records, for every sequence:
  • commanded EE pose (target)         — robot base frame
  • actual    EE pose (RTDE)           — robot base frame
  • actual    EE velocity (RTDE)       — robot base frame
  • robot joint config q (RTDE)        — radians
  • robot joint velocity qd (RTDE)     — rad/s
  • TCP wrench [Fx,Fy,Fz,Tx,Ty,Tz]     — robot base frame
  • mocap markers (raw)                — mocap world frame
  • mocap markers (transformed)        — robot base frame
  • timestamps from monotonic clock
"""

import json
import math
import pickle
import threading
import time
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import rtde_control
# pyrefly: ignore [missing-import]
import rtde_receive
# pyrefly: ignore [missing-import]
from nokov.nokovsdk import PySDKClient


# ════════════════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR  = SCRIPT_DIR / "UR_RTDE_CALIBRATION"
OUT_DIR    = SCRIPT_DIR / "dlo_dataset"
OUT_DIR.mkdir(exist_ok=True)

# ROBOT_IP        = "192.168.1.4"
ROBOT_IP        = "192.168.56.101" # Simulation
MOCAP_SERVER_IP = "10.1.1.198"
MOCAP_TARGET    = "PUBeam_11"

# ── Hardcoded start pose / joints (from earlier session) ─────────────────────
START_POSE = [-0.129948, -0.226872, 0.437462,
               1.570874, -0.565643, 0.521968]
START_Q    = [ 0.033055, -1.506314,  1.088276,
               1.538399,  0.078626, -0.430448]

# ── Payload ──────────────────────────────────────────────────────────────────
PAYLOAD_MASS_KG = 0.9
PAYLOAD_COG_M   = [0.0, 0.0, 0.05]

# ── Cube around start (m) ────────────────────────────────────────────────────
CENTRE_OFFSET = np.array([0.0, 0.0, 0.0])
DX_POS = 0.4
DX_NEG = 0.23
DY_POS = 0.125
DY_NEG = 0.125
DZ_POS = 0.2
DZ_NEG = 0.2

# ── Streaming / control ──────────────────────────────────────────────────────
SAMPLE_HZ        = 125
SERVOJ_LOOKAHEAD = 0.1
SERVOJ_GAIN      = 300

SEQUENCE_DURATION_S = 30.0

# ── Safety ───────────────────────────────────────────────────────────────────
F_ABORT             = 17.0
HOME_SPEED          = 0.4
GO_TO_CENTRE_SPEED  = 0.02
GO_TO_CENTRE_ACCEL  = 0.1

# Software Joint Limits (Radians)
# [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
# Shoulder restricted to mostly forward/up. Elbow restricted to prevent folding flat.
JOINT_LIMITS_MIN = [-math.pi, -math.pi,   0.2, -2*math.pi, -2*math.pi, -2*math.pi]
JOINT_LIMITS_MAX = [ math.pi,      0.0,   2.8,  2*math.pi,  2*math.pi,  2*math.pi]


def clamp_to_cube(pos_offset):
    """Clamp a (dx, dy, dz) offset from centre to the cube limits."""
    return np.array([
        max(-DX_NEG, min(DX_POS, pos_offset[0])),
        max(-DY_NEG, min(DY_POS, pos_offset[1])),
        max(-DZ_NEG, min(DZ_POS, pos_offset[2])),
    ])


# ════════════════════════════════════════════════════════════════════════════
#  Mocap listener — records all markers of MOCAP_TARGET, timestamped
# ════════════════════════════════════════════════════════════════════════════

class MocapStream:
    def __init__(self, server_ip, target_name):
        self.target = target_name
        self.lock = threading.Lock()
        self.latest = None
        self.collecting = False
        self.buffer = []

        self.client = PySDKClient()
        self.client.Initialize(bytes(server_ip, encoding="utf8"))
        self.client.PySetDataCallback(self._cb, None)

    def _cb(self, pFrame, _):
        t_host = time.monotonic()
        if pFrame is None: return
        frame = pFrame.contents

        for j in range(int(frame.nMarkerSets)):
            ms = frame.MocapData[j]
            if self.target not in ms.szName.decode("utf-8"):
                continue
            n = int(ms.nMarkers)
            if n == 0:
                return
            pts = np.array(
                [[ms.Markers[k][0], ms.Markers[k][1], ms.Markers[k][2]]
                 for k in range(n)], float
            ) / 1000.0

            with self.lock:
                self.latest = (t_host, pts)
                if self.collecting:
                    self.buffer.append((t_host, pts))
            return

    def latest_markers(self):
        with self.lock:
            if self.latest is None:
                return None
            t, pts = self.latest
            return (t, pts.copy())

    def start_collect(self):
        with self.lock:
            self.buffer = []
            self.collecting = True

    def stop_collect(self):
        with self.lock:
            self.collecting = False
            return list(self.buffer)


# ════════════════════════════════════════════════════════════════════════════
#  Trajectory generators
# ════════════════════════════════════════════════════════════════════════════

def n_samples(duration_s):
    return int(round(duration_s * SAMPLE_HZ))


def traj_static(duration_s, **kwargs):
    """Hold at centre. Best signal for gravity equilibrium of rope."""
    n = n_samples(duration_s)
    return np.zeros((n, 3))


def _destination_walk(duration_s, max_vel, hold_time_s, axes_active=(0,1,2),
                      seed=0, fill_fraction=0.85):
    """
    Exploratory motion: pick uniform-random destinations inside the cube.
    """
    n = n_samples(duration_s)
    dt = 1.0 / SAMPLE_HZ
    rng = np.random.default_rng(seed)

    reach = np.array([
        min(DX_POS, DX_NEG) * fill_fraction,
        min(DY_POS, DY_NEG) * fill_fraction,
        min(DZ_POS, DZ_NEG) * fill_fraction,
    ])

    def new_destination():
        d = np.zeros(3)
        for ax in axes_active:
            d[ax] = rng.uniform(-reach[ax], reach[ax])
        return d

    pos = np.zeros((n, 3))
    p   = np.zeros(3)
    dest = new_destination()
    hold_until = -1.0

    max_step = max_vel * dt

    for i in range(n):
        t = i * dt
        if hold_until > 0:
            if t >= hold_until:
                dest = new_destination()
                hold_until = -1.0
        else:
            delta = dest - p
            dist = np.linalg.norm(delta)
            if dist <= max_step:
                p = dest.copy()
                hold_until = t + hold_time_s
            else:
                p = p + delta * (max_step / dist)
        pos[i] = p

    return pos


# Single-axis variants
def traj_random_walk_slow_x(duration_s, **kwargs):
    return _destination_walk(duration_s, max_vel=0.04, hold_time_s=1.5,
                              axes_active=(0,), seed=21)

def traj_random_walk_fast_x(duration_s, **kwargs):
    return _destination_walk(duration_s, max_vel=0.12, hold_time_s=0.4,
                              axes_active=(0,), seed=22)

def traj_random_walk_slow_z(duration_s, **kwargs):
    return _destination_walk(duration_s, max_vel=0.04, hold_time_s=1.5,
                              axes_active=(2,), seed=23)

def traj_random_walk_fast_z(duration_s, **kwargs):
    return _destination_walk(duration_s, max_vel=0.12, hold_time_s=0.4,
                              axes_active=(2,), seed=24)

# XZ plane variants
def traj_random_walk_slow_xz(duration_s, **kwargs):
    return _destination_walk(duration_s, max_vel=0.04, hold_time_s=1.5,
                              axes_active=(0,2), seed=11)

def traj_random_walk_fast_xz(duration_s, **kwargs):
    return _destination_walk(duration_s, max_vel=0.08, hold_time_s=0.4,
                              axes_active=(0,2), seed=12)


def traj_sinusoid_xz(duration_s, **kwargs):
    n = n_samples(duration_s)
    t = np.arange(n) / SAMPLE_HZ
    A = np.array([min(DX_POS, DX_NEG) * 0.9, 0.0, min(DZ_POS, DZ_NEG) * 0.9])
    f = np.array([0.30, 0.0, 0.45])
    pos = np.column_stack([
        A[0] * np.sin(2*math.pi*f[0]*t),
        np.zeros(n),
        A[2] * np.sin(2*math.pi*f[2]*t + math.pi/3),
    ])
    return pos


def traj_step_inputs(duration_s, axes_active=(0,)):
    """Discrete steps (smoothed into fast ramps) with hold periods."""
    n = n_samples(duration_s)
    t = np.arange(n) / SAMPLE_HZ
    pos = np.zeros((n, 3))

    step_period = 3.0
    transition_s = 0.4  # Time in seconds to execute the fast step
    A = 0.85
    
    for i, ti in enumerate(t):
        phase = int(ti / step_period) % 4
        time_in_phase = ti % step_period
        
        # The four target scale levels
        levels = {0: 0.0, 1: 1.0, 2: 0.0, 3: -1.0}
        target_scale = levels[phase]
        prev_scale = levels[(phase - 1) % 4]
        
        if time_in_phase < transition_s:
            # Smooth interpolation
            progress = time_in_phase / transition_s
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            scale = prev_scale + (target_scale - prev_scale) * smooth_progress
        else:
            scale = target_scale
            
        for ax in axes_active:
            limit = [DX_POS, DY_POS, DZ_POS][ax]
            pos[i, ax] = scale * A * limit
            
    return pos


def traj_step_inputs_xz(duration_s, **kwargs):
    return traj_step_inputs(duration_s, axes_active=(0, 2))


# ── MAX REACH EXPLORER ───────────────────────────────────────────────────────
def traj_random_walk_max_reach_XZ(duration_s, centre_pose=None, **kwargs):
    """
    Ignores the DX/DZ constraints and instead explores the maximum safe
    physical workspace of the UR3e robot in the XZ plane.
    """
    n = n_samples(duration_s)
    dt = 1.0 / SAMPLE_HZ
    rng = np.random.default_rng(99)

    # Base coords of the start pose
    Y0 = centre_pose[1] if centre_pose else 0.0
    X0 = centre_pose[0] if centre_pose else 0.0
    Z0 = centre_pose[2] if centre_pose else 0.0

    # UR3e reachable radius is ~0.5m. 
    R_MAX = 0.45  # Safe outer margin
    Z_MIN = 0.10  # Safe margin above the table
    R_MIN = 0.18  # Safe margin around the robot base column

    def is_valid_position(abs_x, abs_z):
        if abs_z < Z_MIN: 
            return False
        
        r_cylinder = math.hypot(abs_x, Y0)
        if r_cylinder < R_MIN: 
            return False
            
        r_sphere = math.hypot(r_cylinder, abs_z)
        if r_sphere > R_MAX: 
            return False
            
        return True

    def new_destination():
        # Continually guess random points until we find one in the valid zone
        for _ in range(1000):
            tgt_x = rng.uniform(-R_MAX, R_MAX)
            tgt_z = rng.uniform(Z_MIN, R_MAX)
            if is_valid_position(tgt_x, tgt_z):
                # Return as an offset from the centre_pose
                return np.array([tgt_x - X0, 0.0, tgt_z - Z0])
        return np.array([0.0, 0.0, 0.0]) # Fallback

    pos = np.zeros((n, 3))
    p   = np.zeros(3)
    dest = new_destination()
    hold_until = -1.0
    
    max_vel = 0.12 # Fast walk
    max_step = max_vel * dt

    for i in range(n):
        t = i * dt
        if hold_until > 0:
            if t >= hold_until:
                dest = new_destination()
                hold_until = -1.0
        else:
            delta = dest - p
            dist = np.linalg.norm(delta)
            if dist <= max_step:
                p = dest.copy()
                hold_until = t + 0.4 # hold for 0.4s
            else:
                p = p + delta * (max_step / dist)
        pos[i] = p

    return pos


SEQUENCES = [
    ("seq01_static_baseline",          traj_static,                   "Hold at centre"),
    ("seq02_random_walk_slow_X",       traj_random_walk_slow_x,       "Random walk slow, X axis only"),
    ("seq03_random_walk_fast_X",       traj_random_walk_fast_x,       "Random walk fast, X axis only"),
    ("seq04_random_walk_slow_Z",       traj_random_walk_slow_z,       "Random walk slow, Z axis only"),
    ("seq05_random_walk_fast_Z",       traj_random_walk_fast_z,       "Random walk fast, Z axis only"),
    ("seq09_random_walk_slow_XZ",      traj_random_walk_slow_xz,      "Random walk slow, XZ-plane"),
    ("seq10_random_walk_fast_XZ",      traj_random_walk_fast_xz,      "Random walk fast, XZ-plane"),
    ("seq11_sinusoid_xz_XZ",           traj_sinusoid_xz,              "Sinusoids in XZ-plane"),
    ("seq12_step_inputs_xz_XZ",        traj_step_inputs_xz,           "Step inputs on X and Z"),
    ("seq13_random_walk_max_reach_XZ", traj_random_walk_max_reach_XZ, "Explore MAX reachable XZ workspace"),
]


# ════════════════════════════════════════════════════════════════════════════
#  IK + streaming
# ════════════════════════════════════════════════════════════════════════════

MAX_SERVOJ_JUMP_RAD = math.radians(5)


def reconnect_ctrl(ctrl):
    try:
        if not ctrl.isProgramRunning():
            print("\n   [reconnect] controller script not running, reconnecting…")
            ctrl.reuploadScript()
            time.sleep(0.5)
            return ctrl.isProgramRunning()
        return True
    except Exception as e:
        print(f"   [reconnect] exception: {e}")
        return False


def force_violation(wrench):
    F = np.array(wrench[:3])
    worst = int(np.argmax(np.abs(F)))
    if abs(F[worst]) > F_ABORT:
        return worst, float(F[worst])
    return None, float(F[worst])


def recover_to_centre(ctrl, recv, centre_pose, max_attempts=3):
    """Robust pre-sequence reset: reupload script if dead, moveJ → home, moveL → centre."""
    for attempt in range(max_attempts):
        try:
            if not ctrl.isProgramRunning():
                print(f"   [recover] controller script dead, reuploading (attempt {attempt+1})…")
                ctrl.reuploadScript()
                time.sleep(1.0)

            print(f"   moveJ → home  (attempt {attempt+1})")
            ok_j = ctrl.moveJ(START_Q, HOME_SPEED, 1.0)
            if not ok_j:
                print(f"   moveJ failed, retrying…")
                time.sleep(1.0)
                continue

            print(f"   moveL → cube centre")
            ok_l = ctrl.moveL(centre_pose, GO_TO_CENTRE_SPEED, GO_TO_CENTRE_ACCEL)
            if not ok_l:
                print(f"   moveL failed, retrying…")
                time.sleep(1.0)
                continue

            return True
        except Exception as e:
            print(f"   [recover] exception: {e}")
            time.sleep(1.0)

    return False


def execute_sequence(ctrl, recv, mocap, seq_name, traj_offsets, centre_pose,
                      T_base_mocap):
    """Stream traj_offsets via servoJ, recording everything in lockstep."""
    n = len(traj_offsets)
    dt = 1.0 / SAMPLE_HZ

    print(f"\n── {seq_name}: {n} samples, {n*dt:.1f} s ──")

    rec = {
        "t_host":      np.zeros(n),
        "target_pose": np.zeros((n, 6)),
        "actual_pose": np.zeros((n, 6)),
        "actual_vel":  np.zeros((n, 6)),
        "actual_q":    np.zeros((n, 6)),
        "actual_qd":   np.zeros((n, 6)),
        "tcp_force":   np.zeros((n, 6)),
        "ik_failed":   np.zeros(n, dtype=bool),
    }
    mocap_log = []

    q_prev = recv.getActualQ()
    abort_reason = None
    mocap.start_collect()
    t0 = time.monotonic()

    try:
        for i in range(n):
            t_target = t0 + i * dt

            target = list(centre_pose)
            target[0] += float(traj_offsets[i, 0])
            target[1] += float(traj_offsets[i, 1])
            target[2] += float(traj_offsets[i, 2])

            wrench = recv.getActualTCPForce()
            vio_axis, vio_val = force_violation(wrench)
            if vio_axis is not None:
                ctrl.servoStop()
                abort_reason = f"force_limit_axis_{vio_axis}_{vio_val:+.1f}N"
                for k in rec:
                    rec[k] = rec[k][:i]
                break

            reconnect_ctrl(ctrl)
            try:
                q_sol = ctrl.getInverseKinematics(target, q_prev)
                if q_sol is None or all(v == 0 for v in q_sol):
                    rec["ik_failed"][i] = True
                    q_sol = q_prev
                else:
                    q_sol = list(q_sol)
                    
                    # Safety 1: Validate IK solution for massive joint jumps
                    max_jump = max(abs(a - b) for a, b in zip(q_sol, q_prev))
                    
                    # Safety 2: Validate IK solution against joint limits
                    is_within_limits = True
                    for j_idx in range(6):
                        if not (JOINT_LIMITS_MIN[j_idx] <= q_sol[j_idx] <= JOINT_LIMITS_MAX[j_idx]):
                            is_within_limits = False
                            break
                            
                    if max_jump > MAX_SERVOJ_JUMP_RAD or not is_within_limits:
                        rec["ik_failed"][i] = True
                        q_sol = q_prev
                    else:
                        q_prev = q_sol
                        
            except Exception:
                reconnect_ctrl(ctrl)
                rec["ik_failed"][i] = True
                q_sol = q_prev

            ok = ctrl.servoJ(q_sol, 0.0, 0.0, dt, SERVOJ_LOOKAHEAD, SERVOJ_GAIN)
            if not ok:
                ctrl.servoStop()
                abort_reason = "servoJ_rejected"
                for k in rec:
                    rec[k] = rec[k][:i]
                break

            rec["t_host"][i]      = time.monotonic()
            rec["target_pose"][i] = target
            rec["actual_pose"][i] = recv.getActualTCPPose()
            rec["actual_vel"][i]  = recv.getActualTCPSpeed()
            rec["actual_q"][i]    = recv.getActualQ()
            rec["actual_qd"][i]   = recv.getActualQd()
            rec["tcp_force"][i]   = wrench

            if i % (SAMPLE_HZ // 2) == 0:
                pos = rec["actual_pose"][i, :3]
                print(f"  t={i*dt:5.1f}s  pos {pos*1000} mm  "
                      f"|F|={np.linalg.norm(wrench[:3]):5.1f} N  "
                      f"ik_fail={int(rec['ik_failed'][:i+1].sum())}",
                      end="\r", flush=True)

            sleep_for = t_target + dt - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        ctrl.servoStop()
        abort_reason = "operator_abort"
        for k in rec:
            rec[k] = rec[k][:max(0, i)]

    ctrl.servoStop()
    mocap_log = mocap.stop_collect()
    print()

    if mocap_log:
        t_mocap = np.array([t for t, _ in mocap_log])
        max_m = max(len(p) for _, p in mocap_log)
        markers_mocap = np.full((len(mocap_log), max_m, 3), np.nan)
        for k, (_, pts) in enumerate(mocap_log):
            markers_mocap[k, :len(pts)] = pts
        flat = markers_mocap.reshape(-1, 3)
        ones = np.ones((flat.shape[0], 1))
        flat_b = (T_base_mocap @ np.hstack([flat, ones]).T).T[:, :3]
        markers_base = flat_b.reshape(markers_mocap.shape)
    else:
        t_mocap = np.empty(0)
        markers_mocap = np.empty((0, 0, 3))
        markers_base  = np.empty((0, 0, 3))

    rec["t_mocap"]       = t_mocap
    rec["markers_mocap"] = markers_mocap
    rec["markers_base"]  = markers_base
    rec["abort_reason"]  = abort_reason
    rec["seq_name"]      = seq_name
    rec["t0_host"]       = t0
    rec["sample_hz"]     = SAMPLE_HZ
    rec["centre_pose"]   = np.array(centre_pose)

    return rec


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading calibration + DH …")
    with open(CALIB_DIR / "mocap_calibration.pkl", "rb") as f:
        calib = pickle.load(f)
    with open(CALIB_DIR / "ur3e_dh_params.json") as f:
        dh = json.load(f)["ur3e"]

    T_mocap_base = np.array(calib["T_mocap_base"])
    T_base_mocap = np.linalg.inv(T_mocap_base)
    print(f"  T_base_mocap (mocap → base):\n{np.array2string(T_base_mocap, precision=4, suppress_small=True)}")

    print(f"\nConnecting robot {ROBOT_IP} …")
    recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    ctrl = rtde_control.RTDEControlInterface(
        ROBOT_IP,
        flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
    )
    print(f"Setting payload {PAYLOAD_MASS_KG} kg, CoG {PAYLOAD_COG_M}")
    ctrl.setPayload(PAYLOAD_MASS_KG, PAYLOAD_COG_M)

    print(f"Connecting mocap {MOCAP_SERVER_IP} (target='{MOCAP_TARGET}') …")
    mocap = MocapStream(MOCAP_SERVER_IP, MOCAP_TARGET)
    t_wait = time.time()
    while mocap.latest_markers() is None:
        if time.time() - t_wait > 5.0:
            raise RuntimeError(f"No mocap data for '{MOCAP_TARGET}'")
        time.sleep(0.1)
    n_markers = mocap.latest_markers()[1].shape[0]
    print(f"  mocap streaming, {n_markers} markers")

    centre_pose = list(START_POSE)
    centre_pose[0] += CENTRE_OFFSET[0]
    centre_pose[1] += CENTRE_OFFSET[1]
    centre_pose[2] += CENTRE_OFFSET[2]

    session_meta = {
        "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "robot_ip":          ROBOT_IP,
        "mocap_server_ip":   MOCAP_SERVER_IP,
        "mocap_target":      MOCAP_TARGET,
        "start_pose":        list(START_POSE),
        "start_q":           list(START_Q),
        "centre_pose":       list(centre_pose),
        "centre_offset":     CENTRE_OFFSET.tolist(),
        "cube_limits": {
            "DX_POS": DX_POS, "DX_NEG": DX_NEG,
            "DY_POS": DY_POS, "DY_NEG": DY_NEG,
            "DZ_POS": DZ_POS, "DZ_NEG": DZ_NEG,
        },
        "payload_mass_kg":   PAYLOAD_MASS_KG,
        "payload_cog_m":     PAYLOAD_COG_M,
        "sample_hz":         SAMPLE_HZ,
        "duration_s":        SEQUENCE_DURATION_S,
        "f_abort":           F_ABORT,
        "T_mocap_base":      T_mocap_base.tolist(),
        "T_base_mocap":      T_base_mocap.tolist(),
        "calibration_full":  calib,
        "ur3e_dh":           dh,
        "n_mocap_markers":   int(n_markers),
        "sequence_list":     [(s[0], s[2]) for s in SEQUENCES],
        "joint_limits_min":  JOINT_LIMITS_MIN,
        "joint_limits_max":  JOINT_LIMITS_MAX,
    }
    with open(OUT_DIR / "session_meta.pkl", "wb") as f:
        pickle.dump(session_meta, f)
    print(f"\nSession meta → {OUT_DIR / 'session_meta.pkl'}")

    print("\nPress Enter to begin sequences …")
    input()

    print("\n── moveJ → home, moveL → cube centre ──")
    if not recover_to_centre(ctrl, recv, centre_pose):
        raise RuntimeError("Cannot reach cube centre even after recovery — check robot")

    successes, failures = [], []

    for seq_idx, (seq_name, traj_fn, descr) in enumerate(SEQUENCES):
        print(f"\n{'='*60}")
        print(f"[{seq_idx+1}/{len(SEQUENCES)}] {seq_name}")
        print(f"  {descr}")
        print(f"{'='*60}")

        if not recover_to_centre(ctrl, recv, centre_pose):
            print(f"  Could not return to centre after recovery; skipping")
            failures.append((seq_name, "no_centre_return"))
            continue

        # Generate trajectory offsets
        traj_offsets = traj_fn(SEQUENCE_DURATION_S, centre_pose=centre_pose)
        
        # Only clamp to the small cube if it is NOT the max_reach sequence
        if "max_reach" not in seq_name:
            traj_offsets = np.array([clamp_to_cube(o) for o in traj_offsets])

        # Critical safety: ensure the trajectory STARTS at the current EE position
        RAMP_S = 1.5
        n_ramp = int(RAMP_S * SAMPLE_HZ)
        if traj_offsets.shape[0] > n_ramp and np.linalg.norm(traj_offsets[0]) > 1e-4:
            ramp = np.linspace(0.0, 1.0, n_ramp)[:, None]
            ramped_start = ramp * traj_offsets[0][None, :]
            traj_offsets = np.vstack([ramped_start, traj_offsets])
            print(f"  added {RAMP_S:.1f}s ramp (start offset was "
                  f"{np.linalg.norm(traj_offsets[n_ramp])*1000:.1f} mm from centre)")

        rec = execute_sequence(
            ctrl, recv, mocap, seq_name, traj_offsets, centre_pose, T_base_mocap
        )

        out_path = OUT_DIR / f"{seq_name}.npz"
        np.savez_compressed(
            out_path,
            t_host         = rec["t_host"],
            target_pose    = rec["target_pose"],
            actual_pose    = rec["actual_pose"],
            actual_vel     = rec["actual_vel"],
            actual_q       = rec["actual_q"],
            actual_qd      = rec["actual_qd"],
            tcp_force      = rec["tcp_force"],
            ik_failed      = rec["ik_failed"],
            t_mocap        = rec["t_mocap"],
            markers_mocap  = rec["markers_mocap"],
            markers_base   = rec["markers_base"],
            t0_host        = np.float64(rec["t0_host"]),
            sample_hz      = np.int32(rec["sample_hz"]),
            centre_pose    = rec["centre_pose"],
            seq_name       = seq_name,
            description    = descr,
            abort_reason   = rec["abort_reason"] or "",
        )
        print(f"  saved → {out_path}  ({out_path.stat().st_size/1e3:.0f} kB)")

        if rec["abort_reason"]:
            failures.append((seq_name, rec["abort_reason"]))
        else:
            successes.append(seq_name)

    print("\n── returning home ──")
    try:
        if not ctrl.isProgramRunning():
            ctrl.reuploadScript()
            time.sleep(1.0)
        ctrl.moveJ(START_Q, HOME_SPEED, 1.0)
    except Exception as e:
        print(f"  return-home failed: {e}")
    ctrl.stopScript()
    recv.disconnect()

    print("\n" + "="*60)
    print(f"DONE.  {len(successes)} ok, {len(failures)} failed")
    print(f"Output: {OUT_DIR}")
    if failures:
        for name, why in failures:
            print(f"  ✗ {name}: {why}")
    print("="*60)


if __name__ == "__main__":
    main()