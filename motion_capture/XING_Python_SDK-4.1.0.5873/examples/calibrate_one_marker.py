import time
import math
import threading
import numpy as np
import rtde_control
import rtde_receive
from nokov.nokovsdk import PySDKClient

# ===========================================================================
# CONFIGURATION
# ===========================================================================
ROBOT_IP     = "192.168.1.4"
MOCAP_IP     = "10.1.1.198"
LOOP_FREQ    = 100.0

MOVE_SPEED   = 0.4    
MOVE_ACCEL   = 0.4    
SETTLE_TIME  = 2.0    
MOCAP_FRAMES = 60     
PRINT_INTERVAL = 30   

# The Required Start Pose
READY_POSE = [0.0, -math.pi/2, math.pi/2, math.pi, -math.pi/2, math.radians(143)]

def get_level_pose(j0_deg, j1_deg, j2_deg):
    j0, j1, j2 = math.radians(j0_deg), math.radians(j1_deg), math.radians(j2_deg)
    j3 = math.pi - j1 - j2
    return [j0, j1, j2, j3, -math.pi/2, math.radians(143)]

CALIB_POSES = [
    get_level_pose(  0,  -90,  90),   
    get_level_pose( 35,  -85,  75),   
    get_level_pose(-35,  -85,  75),   
    get_level_pose( 45, -115, 115),   
    get_level_pose(-45, -115, 115),   
    get_level_pose(  0,  -70,  55),   
    get_level_pose(  0, -125, 135),   
    get_level_pose( 25, -105, 105),   
    get_level_pose(-25, -105, 105),   
    get_level_pose( 15,  -75,  65),   
    get_level_pose(-15,  -75,  65),   
    get_level_pose(  0,  -95, 115),   
]

TEST_POSES = [
    get_level_pose( 20,  -90,  90),
    get_level_pose(-20,  -90,  90),
    get_level_pose( 10,  -80,  85),
    get_level_pose(-10, -110, 120),
]

# ===========================================================================
# GLOBALS & MOCAP
# ===========================================================================
_latest_marker_pos = None   
_marker_lock       = threading.Lock()
_marker_ready      = threading.Event()

def mocap_callback(pFrameOfMocapData, pUserData):
    global _latest_marker_pos
    if not pFrameOfMocapData: return
    frame = pFrameOfMocapData.contents
    unlabeled = []
    for i in range(frame.nOtherMarkers):
        x, y, z = frame.OtherMarkers[i]
        if x < 9_000_000.0: unlabeled.append([x/1000.0, y/1000.0, z/1000.0])

    if unlabeled:
        with _marker_lock:
            _latest_marker_pos = np.array(unlabeled[0])
            _marker_ready.set()

def collect_mocap_average(n_frames: int):
    global _latest_marker_pos
    samples = []
    while len(samples) < n_frames:
        with _marker_lock:
            if _latest_marker_pos is not None:
                samples.append(_latest_marker_pos.copy())
                _latest_marker_pos = None
        time.sleep(0.01)
    return np.array(samples).mean(axis=0)

# ===========================================================================
# MATH
# ===========================================================================
def svd_registration(src, dst):
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    t = dst_c - R @ src_c
    res = np.linalg.norm(((R @ src.T).T + t) - dst, axis=1)
    return R, t, res

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    np.set_printoptions(suppress=True, precision=6)
    
    print("Connecting to SDK...")
    client = PySDKClient()
    if client.Initialize(bytes(MOCAP_IP, encoding="utf-8")) != 0:
        print("Mocap Init Failed")
        return
    client.PySetDataCallback(mocap_callback, None)

    print("Connecting to Robot...")
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

    print("Moving to READY_POSE...")
    rtde_c.moveJ(READY_POSE, 0.5, 0.5)
    _marker_ready.wait(5.0)

    mocap_pts, robot_pts = [], []
    for i, joints in enumerate(CALIB_POSES):
        print(f"Pose {i+1}/{len(CALIB_POSES)}...")
        rtde_c.moveJ(joints, MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)
        mocap_pts.append(collect_mocap_average(MOCAP_FRAMES))
        robot_pts.append(np.array(rtde_r.getActualTCPPose()[:3]))

    # Solve Registration
    mocap_pts, robot_pts = np.array(mocap_pts), np.array(robot_pts)
    R, t, residuals = svd_registration(mocap_pts, robot_pts)
    
    # Validation
    test_errs = []
    for joints in TEST_POSES:
        rtde_c.moveJ(joints, MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)
        p_gt = np.array(rtde_r.getActualTCPPose()[:3])
        p_pred = R @ collect_mocap_average(MOCAP_FRAMES) + t
        test_errs.append(np.linalg.norm(p_pred - p_gt) * 1000)

    # ─── FINAL OUTPUT AND SAVING ───
    print("\n" + "="*50)
    print("FINAL CALIBRATION SUMMARY")
    print("="*50)
    print(f"Mean Calibration Residual: {np.mean(residuals)*1000:.3f} mm")
    print(f"Mean Validation Error:    {np.mean(test_errs):.3f} mm")
    print("-" * 50)
    print("Rotation Matrix (R):")
    print(R)
    print("\nTranslation Vector (t) [meters]:")
    print(t)
    
    # Create 4x4 Homogeneous Matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    print("\nFull 4x4 Transform (T):")
    print(T)
    print("="*50)

    # Save to files
    np.save("calibration_R.npy", R)
    np.save("calibration_t.npy", t)
    np.save("calibration_T.npy", T)
    
    # Save a human-readable text file
    with open("calibration_results.txt", "w") as f:
        f.write("Calibration Results\n")
        f.write(f"Mean Validation Error: {np.mean(test_errs):.3f} mm\n\n")
        f.write("Rotation Matrix R:\n" + str(R) + "\n\n")
        f.write("Translation Vector t:\n" + str(t) + "\n\n")
        f.write("4x4 Transform Matrix T:\n" + str(T) + "\n")

    print("\nResults saved to:")
    print(" - calibration_R.npy / calibration_t.npy")
    print(" - calibration_results.txt")

    # Cleanup
    rtde_c.moveJ(READY_POSE, 0.5, 0.5)
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("\nDone.")

if __name__ == "__main__":
    main()