import sys
import numpy as np
import time
import math
import threading
from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
from nokov.nokovsdk import *
import rtde_control
import rtde_receive

# ==========================================
# 1. CALIBRATION & CONFIGURATION
# ==========================================
ROBOT_IP = "192.168.1.4"
SERVER_IP = '10.1.1.198'
TARGET_MARKERSET_NAME = "Rope"

# Transformation Matrix (NOKOV -> UR3e Base)
T_MATRIX = np.array([
    [ 0.999023, -0.042866,  0.01073,  -0.918766],
    [ 0.042646,  0.998892,  0.019915,  0.049483],
    [-0.011572, -0.019438,  0.999744, -0.804454],
    [ 0.0,        0.0,        0.0,        1.0     ]
])

# Shared data between threads
latest_rope_coords = None  

# ==========================================
# 2. TRANSFORMATION LOGIC
# ==========================================
def transform_to_robot_frame(points_m):
    """
    Applies the 4x4 matrix to convert points from MoCap to Robot Base.
    points_m: numpy array of shape (N, 3) in meters.
    """
    if points_m is None or len(points_m) == 0:
        return None
    
    # 1. Add column of ones for homogeneous multiplication (N, 4)
    num_points = points_m.shape[0]
    ones = np.ones((num_points, 1))
    points_homo = np.hstack([points_m, ones]) 
    
    # 2. Transform: T * P^T -> then transpose back to (N, 4)
    transformed_homo = (T_MATRIX @ points_homo.T).T
    
    # 3. Return only X, Y, Z (N, 3)
    return transformed_homo[:, :3]

# ==========================================
# 3. NOKOV DATA CALLBACK
# ==========================================
def py_data_func(pFrameOfMocapData, pUserData):
    global latest_rope_coords
    if pFrameOfMocapData == None: return
    frameData = pFrameOfMocapData.contents

    for iMarkerSet in range(frameData.nMarkerSets):
        markerset = frameData.MocapData[iMarkerSet]
        markerset_name = markerset.szName.decode('utf-8')
        
        if TARGET_MARKERSET_NAME in markerset_name:
            coords = []
            for iMarker in range(markerset.nMarkers):
                x, y, z = markerset.Markers[iMarker]
                
                if x > 9000000.0: # Lost marker handling
                    coords.append([np.nan, np.nan, np.nan])
                else:
                    # Convert mm to meters
                    coords.append([x/1000.0, y/1000.0, z/1000.0])
            
            # Transform the entire set of markers to Robot Frame
            mocap_points = np.array(coords)
            latest_rope_coords = transform_to_robot_frame(mocap_points)
            break

# ==========================================
# 4. ROBOT MOVEMENT THREAD
# ==========================================
def robot_worker():
    LOOP_FREQ = 100.0
    DT = 1.0 / LOOP_FREQ
    READY_POSE = [0.0, -math.pi/2, math.pi/2, math.pi, -math.pi/2, math.radians(143)]
    NUM_CYCLES = 50
    CYCLE_DURATION = 1.2
    AMPLITUDE_VELOCITY = 0.4
    ACCELERATION = 1.5

    print(f"[Robot] Connecting to {ROBOT_IP}...")
    try:
        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP, 
            frequency=LOOP_FREQ,
            flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP
        )        
        rtde_c.moveJ(READY_POSE, 0.5, 0.5)
        print("[Robot] Ready position reached. Starting Sine Wave.")
        
        total_steps = int(NUM_CYCLES * CYCLE_DURATION * LOOP_FREQ)
        for i in range(total_steps):
            t_start = rtde_c.initPeriod()
            elapsed_time = i * DT
            v_z = AMPLITUDE_VELOCITY * np.sin(2 * np.pi * (elapsed_time / CYCLE_DURATION))
            
            speed_vector = [0.0, 0.0, v_z, 0.0, 0.0, 0.0]
            rtde_c.speedL(speed_vector, ACCELERATION, DT)
            rtde_c.waitPeriod(t_start)
            
        rtde_c.speedStop()
        rtde_c.stopScript()
    except Exception as e:
        print(f"[Robot] Error: {e}")

# ==========================================
# 5. VISUALIZER
# ==========================================
class RopeVisualizer:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Robot Frame Rope Visualizer (NOKOV -> UR3e)')
        self.window.resize(1000, 800)
        self.window.show()
        
        grid = gl.GLGridItem()
        grid.setSize(x=2, y=2) # Smaller grid more appropriate for robot workspace
        grid.setSpacing(x=0.1, y=0.1)
        self.window.addItem(grid)
        self.window.setCameraPosition(distance=2)

        self.line_plot = gl.GLLinePlotItem(color=(0, 1, 0.5, 1), width=4, antialias=True)
        self.window.addItem(self.line_plot)
        self.scatter_plot = gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=10)
        self.window.addItem(self.scatter_plot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphics)
        self.timer.start(16) 

    def update_graphics(self):
        global latest_rope_coords
        if latest_rope_coords is not None:
            # Filter out NaNs for the line plot to avoid drawing issues
            clean_coords = latest_rope_coords[~np.isnan(latest_rope_coords).any(axis=1)]
            if len(clean_coords) > 0:
                self.line_plot.setData(pos=clean_coords)
                self.scatter_plot.setData(pos=clean_coords)

    def run(self):
        self.app.exec()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # 1. Connect MoCap
    client = PySDKClient()
    if client.Initialize(bytes(SERVER_IP, encoding="utf8")) == 0:
        print("NOKOV Connected.")
        client.PySetDataCallback(py_data_func, None)
    else:
        print("NOKOV Connection Failed.")
        sys.exit()

    # 2. Start Robot Thread
    robot_thread = threading.Thread(target=robot_worker, daemon=True)
    robot_thread.start()

    # 3. Start GUI
    visualizer = RopeVisualizer()
    visualizer.run()