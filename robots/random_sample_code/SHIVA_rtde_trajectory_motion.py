import numpy as np
import time

USE_ROBOT = True
ROBOT_IP = "192.168.1.2"

if USE_ROBOT:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    
    
rtde_c = None
rtde_r = None

def connect_robot(ip=ROBOT_IP):
    global rtde_c, rtde_r
    if not USE_ROBOT:
        print("USE_ROBOT=False → not connecting.")
        return
    rtde_c = RTDEControlInterface(ip)
    rtde_r = RTDEReceiveInterface(ip)
    
    print("Connected to robot at", ip)
    # return rtde_r.getActualQ()

def disconnect_robot():
    global rtde_c, rtde_r
    if rtde_c is not None:
        rtde_c.disconnect()
        rtde_c = None
    if rtde_r is not None:
        rtde_r.disconnect()
        rtde_r = None
    print("Disconnected robot.")

def get_q_actual():
    if not USE_ROBOT or rtde_r is None:
        raise RuntimeError("Robot not connected / USE_ROBOT=False")
    return np.array(rtde_r.getActualQ(), dtype=np.float64)

def move_servoJ(rtde_c, q_old, q_new, speed=0.1):
    """
    Your 'move' function: small joint change with servoJ.
    """
    assert np.max(np.abs(q_old - q_new)) < np.deg2rad(10)
    a, v = 0.5, 0.5
    dq = np.abs(q_new - q_old)
    weights = np.array([10,10,10,1,1,1])
    dq_max = np.max(np.dot(dq, weights))
    dt = (0.2 / speed) * 0.5
    add = max(0, dq_max - 0.1)
    dt += (add**2) * 0.3
    dt = min(max(0.02, dt), 1.5)
    dt *= 0.6
    lookahead = max(min(0.2, dt), 0.03)
    gain = 100
    rtde_c.servoJ(q_new.tolist(), a, v, dt, lookahead, gain)
    time.sleep(dt * 0.95)

def execute_trajectory_on_robot(q_traj, speed=0.1):
    """
    Stream a joint trajectory to the robot using servoJ.
    """
    if not USE_ROBOT or rtde_c is None:
        print("Robot execution skipped (USE_ROBOT=False or not connected).")
        return

    q_prev = get_q_actual()
    # q_prev = q_traj[0]
    for q in q_traj:
        move_servoJ(rtde_c, q_prev, q, speed=speed)
        q_prev = q