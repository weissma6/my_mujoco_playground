import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
from nokov.nokovsdk import *
from Utility import * # <--- WE NEED THIS BACK FOR THE PHYSICS MATH

# ==========================================
# 1. GLOBAL VARIABLES (The Bridge & The Data)
# ==========================================
latest_rope_coords = None   # For the visualizer (just X,Y,Z in meters)

# This is your new comprehensive data collection!
# It will store a list of dictionaries for every marker containing pos, vel, and acc.
latest_rope_physics = []    

# Caching arrays needed to calculate velocity and acceleration
velocity_arrays = {}
acceleration_arrays = {}
FRAME_RATE = 60 # Default assumed frame rate for math calculations

# ==========================================
# 2. CONFIGURATION
# ==========================================
SERVER_IP = '10.1.1.198'
TARGET_MARKERSET_NAME = "Rope"  

# ==========================================
# 3. NOKOV DATA CALLBACK (Background Thread)
# ==========================================
def py_data_func(pFrameOfMocapData, pUserData):
    global latest_rope_coords, latest_rope_physics, velocity_arrays, acceleration_arrays
    
    if pFrameOfMocapData == None:  
        return
        
    frameData = pFrameOfMocapData.contents

    for iMarkerSet in range(frameData.nMarkerSets):
        markerset = frameData.MocapData[iMarkerSet]
        markerset_name = markerset.szName.decode('utf-8')
        
        if TARGET_MARKERSET_NAME in markerset_name:
            coords_for_visuals = []
            physics_data = [] # Temporary list to hold this frame's rich data
            
            for iMarker in range(markerset.nMarkers):
                x = markerset.Markers[iMarker][0]
                y = markerset.Markers[iMarker][1]
                z = markerset.Markers[iMarker][2]
                
                # --- PHYSICS CALCULATIONS START ---
                marker_key = f"rope_{iMarker}"
                
                # Initialize caching arrays if they don't exist yet
                if marker_key not in velocity_arrays:
                    velocity_arrays[marker_key] = SlideFrameArray()
                    acceleration_arrays[marker_key] = SlideFrameArray()
                
                # Cache current point for math
                marker_point = Point(x, y, z, marker_key)
                velocity_arrays[marker_key].cache(marker_point)
                acceleration_arrays[marker_key].cache(marker_point)
                
                # Calculate
                vel_method = CalculateVelocity(FRAME_RATE, 3) 
                acc_method = CalculateAcceleration(FRAME_RATE, 3)
                
                velocity = velocity_arrays[marker_key].try_to_calculate(vel_method)
                acceleration = acceleration_arrays[marker_key].try_to_calculate(acc_method)
                # --- PHYSICS CALCULATIONS END ---

                # Bundle everything into a dictionary so you have all data saved
                marker_info = {
                    "marker_id": iMarker + 1,
                    "position_mm": {"x": x, "y": y, "z": z},
                    "velocity": velocity if velocity else "Calculating...",
                    "acceleration": acceleration if acceleration else "Calculating..."
                }
                print(marker_info)
                physics_data.append(marker_info)

                # --- VISUALIZER LOGIC START ---
                # Handle the "Lost Marker" quirk (9999999.0)
                if x > 9000000.0 or y > 9000000.0 or z > 9000000.0:
                    coords_for_visuals.append([np.nan, np.nan, np.nan])
                else:
                    # Convert to meters for the 3D visualizer
                    coords_for_visuals.append([x/1000.0, y/1000.0, z/1000.0])
                # --- VISUALIZER LOGIC END ---

            # Update the global variables
            latest_rope_coords = np.array(coords_for_visuals)
            latest_rope_physics = physics_data 
            break

# ==========================================
# 4. PYQTGRAPH VISUALIZER (Main Thread)
# ==========================================
class RopeVisualizer:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('NOKOV Real-Time Rope Visualizer')
        self.window.resize(1000, 800)
        self.window.show()
        
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10)
        grid.setSpacing(x=1, y=1)
        self.window.addItem(grid)
        
        self.window.setCameraPosition(distance=5)

        self.line_plot = gl.GLLinePlotItem(color=(0, 1, 0.5, 1), width=4, antialias=True)
        self.window.addItem(self.line_plot)
        
        self.scatter_plot = gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=10)
        self.window.addItem(self.scatter_plot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphics)
        self.timer.start(16) 

    def update_graphics(self):
        global latest_rope_coords, latest_rope_physics
        
        if latest_rope_coords is not None and len(latest_rope_coords) > 0:
            self.line_plot.setData(pos=latest_rope_coords)
            self.scatter_plot.setData(pos=latest_rope_coords)
            
            # Optional: Uncomment the line below if you want it to constantly print 
            # the raw physics data to the terminal while the window is running!
            # print(f"Marker 1 Velocity: {latest_rope_physics[0]['velocity']}")

    def run(self):
        sys.exit(self.app.exec())

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print(f"Connecting to NOKOV server at {SERVER_IP}...")
    client = PySDKClient()
    ret = client.Initialize(bytes(SERVER_IP, encoding="utf8"))
    
    if ret == 0:
        print("Connected Successfully!")
    else:
        print(f"Connection Failed with code: {ret}")
        sys.exit()

    client.PySetDataCallback(py_data_func, None)
    print("Starting visualizer... (Close the window to quit)")
    
    visualizer = RopeVisualizer()
    visualizer.run()