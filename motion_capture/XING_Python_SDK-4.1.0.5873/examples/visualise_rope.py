import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
from nokov.nokovsdk import *

# ==========================================
# 1. GLOBAL VARIABLES (The Bridge)
# ==========================================
# This acts as the bridge between the NOKOV background thread and the visualizer main thread.
# It holds the latest frame's X, Y, Z coordinates.
latest_rope_coords = None  

# ==========================================
# 2. CONFIGURATION
# ==========================================
SERVER_IP = '10.1.1.198'
# Change this to whatever you named your rope in the NOKOV software!
TARGET_MARKERSET_NAME = "Rope"  

# ==========================================
# 3. NOKOV DATA CALLBACK (Background Thread)
# ==========================================
def py_data_func(pFrameOfMocapData, pUserData):
    global latest_rope_coords
    
    if pFrameOfMocapData == None:  
        return
        
    frameData = pFrameOfMocapData.contents

    # Loop through the markersets to find our rope
    for iMarkerSet in range(frameData.nMarkerSets):
        markerset = frameData.MocapData[iMarkerSet]
        markerset_name = markerset.szName.decode('utf-8')
        
        # If we found the rope, extract its markers
        if TARGET_MARKERSET_NAME in markerset_name:
            coords = []
            for iMarker in range(markerset.nMarkers):
                x = markerset.Markers[iMarker][0]
                y = markerset.Markers[iMarker][1]
                z = markerset.Markers[iMarker][2]
                
                # Handle the "Lost Marker" quirk (9999999.0)
                if x > 9000000.0 or y > 9000000.0 or z > 9000000.0:
                    # If lost, we append NaN so the 3D line just breaks temporarily 
                    # instead of shooting off into deep space.
                    coords.append([np.nan, np.nan, np.nan])
                else:
                    # NOKOV outputs in millimeters. 
                    # We divide by 1000 to convert to meters so it fits nicely in the 3D grid.
                    coords.append([x/1000.0, y/1000.0, z/1000.0])
            
            # Update the global variable with the new numpy array
            latest_rope_coords = np.array(coords)
            break # We found the rope, no need to check other markersets

# ==========================================
# 4. PYQTGRAPH VISUALIZER (Main Thread)
# ==========================================
class RopeVisualizer:
    def __init__(self):
        # Setup the Application
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('NOKOV Real-Time Rope Visualizer')
        self.window.resize(1000, 800)
        self.window.show()
        
        # Add a floor grid
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10)
        grid.setSpacing(x=1, y=1)
        self.window.addItem(grid)
        
        # Set camera position (zoomed back to see the 3D space)
        self.window.setCameraPosition(distance=5)

        # Create the visual elements (Empty for now, updated on timer)
        # 1. The Line (Rope)
        self.line_plot = gl.GLLinePlotItem(color=(0, 1, 0.5, 1), width=4, antialias=True)
        self.window.addItem(self.line_plot)
        
        # 2. The Dots (Individual Markers)
        self.scatter_plot = gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=10)
        self.window.addItem(self.scatter_plot)

        # Start the Timer (Fires 60 times a second to update the graphics)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphics)
        self.timer.start(16) # 16ms is roughly 60 FPS

    def update_graphics(self):
        global latest_rope_coords
        # If NOKOV has successfully pushed data to our global variable
        if latest_rope_coords is not None and len(latest_rope_coords) > 0:
            # Update the 3D line and the points
            self.line_plot.setData(pos=latest_rope_coords)
            self.scatter_plot.setData(pos=latest_rope_coords)

    def run(self):
        # Start the GUI Loop (This blocks the script from exiting)
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

    # Tell NOKOV to start silently firing our callback function in the background
    client.PySetDataCallback(py_data_func, None)
    
    print("Starting visualizer... (Close the window to quit)")
    
    # Launch the GUI
    visualizer = RopeVisualizer()
    visualizer.run()