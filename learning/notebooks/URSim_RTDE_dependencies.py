'''
Create functions an classes for the URSim RTDE control feedback demo.
'''

import rtde_receive
from typing import Dict, List, Optional



class URSimRTDEControlFeedback:
    def __init__(self, host="127.0.0.1", port_rtde=30004, port_urscript=30002, port_dashboard=29999):
        self.host = host
        self.port_rtde = port_rtde
        self.port_urscript = port_urscript
        self.port_dashboard = port_dashboard
        self._receiver: Optional[rtde_receive.RTDEReceiveInterface] = None

    def connect(self):
        if self._receiver is None:
            self._receiver = rtde_receive.RTDEReceiveInterface(self.host)
        return self._receiver

    def disconnect(self):
        if self._receiver is not None:
            self._receiver.disconnect()
            self._receiver = None

    def is_connected(self) -> bool:
        if self._receiver is None:
            return False
        return self._receiver.isConnected()

    def receive_feedback(self) -> Dict[str, List[float]]:
        """
        Returns:
            {
                "q": [q1..q6],                # joint positions [rad]
                "qd": [qd1..qd6],            # joint velocities [rad/s]
                "tau": [tau1..tau6],         # estimated joint torques [Nm]
                "tcp_xyz": [x, y, z],        # TCP position in base/world frame [m]
                "tcp_pose": [x,y,z,rx,ry,rz] # full TCP pose
                "tcp_speed": [vx, vy, vz, vrx, vry, vrz] # TCP speed in base/world frame [m/s, rad/s]
                "tcp_speed_3d": tcp_speed_3d     # TCP speed in 3D space (magnitude of linear velocity) [m/s]
            }
        """
        r = self.connect()

        q = r.getActualQ()
        qd = r.getActualQd()
        cur = r.getActualCurrent()
        tcp_pose = r.getActualTCPPose()
        tcp_xyz = tcp_pose[:3]
        tcp_speed = r.getActualTCPSpeed()
        vx, vy, vz = tcp_speed[:3]
        tcp_speed_3d = (vx**2 + vy**2 + vz**2) ** 0.5

        return {
            "q": list(q),
            "qd": list(qd),
            "cur": list(cur),
            "tcp_xyz": list(tcp_xyz),
            "tcp_pose": list(tcp_pose),
            "tcp_speed": list(tcp_speed),
            "tcp_speed_3d": list([tcp_speed_3d])
        }

    def print_feedback(self, digits: int = 4):
        fb = self.receive_feedback()
        print("q      =", [round(v, digits) for v in fb["q"]])
        print("qd     =", [round(v, digits) for v in fb["qd"]])
        print("cur    =", [round(v, digits) for v in fb["cur"]])
        print("tcp_xyz=", [round(v, digits) for v in fb["tcp_xyz"]])
        print("tcp_speed_3d=", [round(v, digits) for v in fb["tcp_speed_3d"]])
   
   
    pass

