"""Quick test: connect to real robot via RTDE and print state."""
import rtde_receive

ROBOT_IP = "192.168.1.2"

r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Connected!")
print("q   =", [round(v, 4) for v in r.getActualQ()])
print("qd  =", [round(v, 4) for v in r.getActualQd()])
print("tcp =", [round(v, 4) for v in r.getActualTCPPose()[:3]])
r.disconnect()
print("Done.")
