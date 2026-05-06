import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
from nokov.nokovsdk import *

# ==========================================
# 1. GLOBAL VARIABLES
# ==========================================
latest_raw_coords = None  
frame_count = 0  # To throttle printing

# ==========================================
# 2. CONFIGURATION
# ==========================================
SERVER_IP = '10.1.1.198'
PRINT_INTERVAL = 10  # Only print every 10th frame to keep terminal readable

# ==========================================
# 3. NOKOV DATA CALLBACK
# ==========================================
def py_data_func(pFrameOfMocapData, pUserData):
    global latest_raw_coords, frame_count
    
    if pFrameOfMocapData == None:  
        return
        
    frameData = pFrameOfMocapData.contents
    all_points = []
    labeled_count = 0
    unlabeled_count = 0

    # --- PART A: Labeled Markers ---
    for iMarkerSet in range(frameData.nMarkerSets):
        markerset = frameData.MocapData[iMarkerSet]
        for iMarker in range(markerset.nMarkers):
            x, y, z = markerset.Markers[iMarker]
            if x < 9000000.0:
                all_points.append([x/1000.0, y/1000.0, z/1000.0])
                labeled_count += 1

    # --- PART B: Unlabeled Markers (OtherMarkers) ---
    for iOther in range(frameData.nOtherMarkers):
        x, y, z = frameData.OtherMarkers[iOther]
        if x < 9000000.0:
            all_points.append([x/1000.0, y/1000.0, z/1000.0])
            unlabeled_count += 1
    
    # Update global coords for the visualizer
    if len(all_points) > 0:
        latest_raw_coords = np.array(all_points)
    else:
        latest_raw_coords = None

    # --- PART C: Constant Info Print ---
    frame_count += 1
    if frame_count % PRINT_INTERVAL == 0:
        total = labeled_count + unlabeled_count
        print(f"--- Frame: {frameData.iFrame} ---")
        print(f"Total Markers: {total} (Labeled: {labeled_count}, Unlabeled: {unlabeled_count})")
        
        if total > 0:
            # Print the first 3 markers as a sample
            sample_size = min(3, len(all_points))
            for i in range(sample_size):
                p = all_points[i]
                marker_type = "Labeled" if i < labeled_count else "Unlabeled"
                print(f"  [{marker_type} {i}] X:{p[0]:.3f} Y:{p[1]:.3f} Z:{p[2]:.3f}")
        else:
            print("  No markers detected in volume.")
        
        # Move cursor up or clear line if you want it to stay in one place, 
        # but scrolling is usually safer for debugging.
        print("-" * 30)

# ==========================================
# 4. PYQTGRAPH VISUALIZER
# ==========================================
class RawDataVisualizer:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
            
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('NOKOV Raw Marker Stream')
        self.window.resize(1000, 800)
        
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.window.addItem(grid)
        self.window.setCameraPosition(distance=8)
        
        # Cyan dots for all markers
        self.scatter_plot = gl.GLScatterPlotItem(color=(0, 1, 1, 1), size=12, pxMode=True)
        self.window.addItem(self.scatter_plot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphics)
        self.timer.start(16) 
        
        self.window.show()

    def update_graphics(self):
        global latest_raw_coords
        if latest_raw_coords is not None:
            self.scatter_plot.setData(pos=latest_raw_coords)

    def run(self):
        return self.app.exec()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print(f"Connecting to NOKOV at {SERVER_IP}...")
    client = PySDKClient()
    ret = client.Initialize(bytes(SERVER_IP, encoding="utf8"))
    
    if ret == 0:
        print("Connected Successfully!")
    else:
        print(f"Connection Failed: {ret}")
        sys.exit()

    client.PySetDataCallback(py_data_func, None)
    
    print("Starting visualizer and data log...")
    try:
        visualizer = RawDataVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.Uninitialize()
        print("Disconnected.")