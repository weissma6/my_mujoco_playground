import sys
import numpy as np
import time
import math
import threading
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
from nokov.nokovsdk import *
import rtde_control
import rtde_receive

# ==========================================
# 1. MANUAL OFFSETS & CONFIGURATION
# ==========================================
# Adjust these (in meters) to nudge the rope visualization
X_OFFSET = 0.92
Y_OFFSET = 0.28
Z_OFFSET = 0.08

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

latest_rope_coords = None  

# Vibrant Color Palette for the Rope "Graph"
ROPE_COLORS = np.array([
    [255, 51, 51],   # Red
    [255, 153, 51],  # Orange
    [255, 255, 51],  # Yellow
    [153, 255, 51],  # Lime
    [51, 255, 51],   # Green
    [51, 255, 153],  # Aqua
    [51, 255, 255],  # Cyan
    [51, 153, 255],  # Sky Blue
    [51, 51, 255],   # Blue
    [153, 51, 255],  # Purple
    [255, 51, 255],  # Magenta
    [255, 51, 153]   # Pink
]) 

# ==========================================
# 2. TRANSFORMATION LOGIC
# ==========================================
def transform_to_robot_frame(points_m):
    if points_m is None or len(points_m) == 0:
        return None
    ones = np.ones((points_m.shape[0], 1))
    points_homo = np.hstack([points_m, ones]) 
    transformed_homo = (T_MATRIX @ points_homo.T).T
    pts = transformed_homo[:, :3]
    
    # Apply manual offsets
    pts[:, 0] += X_OFFSET
    pts[:, 1] += Y_OFFSET
    pts[:, 2] += Z_OFFSET
    return pts

# ==========================================
# 3. NOKOV DATA CALLBACK
# ==========================================
def py_data_func(pFrameOfMocapData, pUserData):
    global latest_rope_coords
    if pFrameOfMocapData is None: return
    frameData = pFrameOfMocapData.contents

    for iMarkerSet in range(frameData.nMarkerSets):
        markerset = frameData.MocapData[iMarkerSet]
        if TARGET_MARKERSET_NAME in markerset.szName.decode('utf-8'):
            coords = []
            for iMarker in range(markerset.nMarkers):
                x, y, z = markerset.Markers[iMarker]
                if x < 9000000.0:
                    coords.append([x/1000.0, y/1000.0, z/1000.0])
            
            if coords:
                mocap_pts = np.array(coords)
                latest_rope_coords = transform_to_robot_frame(mocap_pts)
            break

# ==========================================
# 4. ROBOT MOVEMENT THREAD
# ==========================================
def robot_worker():
    LOOP_FREQ = 100.0
    DT = 1.0 / LOOP_FREQ
    READY_POSE = [0.0, -math.pi/2, math.pi/2, math.pi, -math.pi/2, math.radians(143)]
    
    NUM_CYCLES = 10
    CYCLE_DURATION = 1.2
    AMPLITUDE_VELOCITY = 0.35 
    ACCELERATION = 1.5

    try:
        print(f"[Robot] Connecting to {ROBOT_IP}...")
        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP, 
            frequency=LOOP_FREQ,
            flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP
        )
        
        print("[Robot] Moving to start position...")
        rtde_c.moveJ(READY_POSE, 0.5, 0.5)
        time.sleep(1.0)
        
        print("[Robot] Starting Sine Wave motion.")
        total_steps = int(NUM_CYCLES * CYCLE_DURATION * LOOP_FREQ)
        
        for i in range(total_steps):
            t_start = rtde_c.initPeriod()
            elapsed_time = i * DT
            v_z = AMPLITUDE_VELOCITY * np.sin(2 * np.pi * (elapsed_time / CYCLE_DURATION))
            rtde_c.speedL([0.0, 0.0, v_z, 0.0, 0.0, 0.0], ACCELERATION, DT)
            rtde_c.waitPeriod(t_start)
            
        rtde_c.speedStop()
        rtde_c.stopScript()
    except Exception as e:
        print(f"[Robot Error]: {e}")

# ==========================================
# 5. VISUALIZATION ENGINE (Viser)
# ==========================================
def start_visualization():
    global latest_rope_coords
    
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    print(f"Visualizer ready at http://localhost:8080")

    print("Loading UR3e URDF...")
    try:
        urdf = load_robot_description("ur3e_description")
        viser_urdf = viser.extras.ViserUrdf(server, urdf)
    except Exception as e:
        print(f"Failed to load robot model: {e}")
        return

    server.scene.add_grid("grid", width=2, height=2)

    # Initialize rope line with proper segment coloring support
    rope_line = server.scene.add_line_segments(
        "/rope/line",
        points=np.zeros((1, 2, 3)),
        colors=np.zeros((1, 2, 3)),
        line_width=3.0
    )
    
    marker_spheres = []
    
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except Exception as e:
        print(f"Receiver Error: {e}")
        return

    try:
        while True:
            # A. Sync Robot
            q = rtde_r.getActualQ()
            if q:
                viser_urdf.update_cfg(np.array(q))

            # B. Sync Rope Graph
            if latest_rope_coords is not None:
                pts = latest_rope_coords[~np.isnan(latest_rope_coords).any(axis=1)]
                num_pts = len(pts)
                
                if num_pts > 1:
                    palette_uint8 = ROPE_COLORS.astype(np.uint8)
                    marker_colors = palette_uint8[np.arange(num_pts) % len(palette_uint8)]
                    
                    # 1. Update Sphere Pool (Nodes)
                    for i in range(num_pts):
                        color_tuple = tuple(int(c) for c in marker_colors[i])
                        if i >= len(marker_spheres):
                            # FIXED: changed 'add_mesh_icosphere' to 'add_icosphere'
                            handle = server.scene.add_icosphere(
                                f"/rope/nodes/node_{i}",
                                radius=0.005, # Smaller, clean circular dots
                                color=color_tuple
                            )
                            marker_spheres.append(handle)
                        
                        marker_spheres[i].position = pts[i]
                        marker_spheres[i].color = color_tuple
                        marker_spheres[i].visible = True
                    
                    for i in range(num_pts, len(marker_spheres)):
                        marker_spheres[i].visible = False
                    
                    # 2. Update Lines (Edges)
                    # Shape (N-1 segments, 2 points per segment, 3 coords)
                    segments = np.stack([pts[:-1], pts[1:]], axis=1)
                    
                    # Line marker A to B uses marker A's color
                    # We create a color array of shape (N-1, 2, 3) 
                    # where both points of segment i use marker_colors[i]
                    line_colors = np.stack([marker_colors[:-1], marker_colors[:-1]], axis=1)
                    
                    rope_line.points = segments
                    rope_line.colors = line_colors

            time.sleep(0.016)
            
    except KeyboardInterrupt:
        print("Closing...")
    finally:
        rtde_r.disconnect()

# ==========================================
# 6. MAIN
# ==========================================
if __name__ == '__main__':
    client = PySDKClient()
    if client.Initialize(bytes(SERVER_IP, encoding="utf8")) == 0:
        client.PySetDataCallback(py_data_func, None)
        print("NOKOV Connected.")
    else:
        sys.exit()

    thread_robot = threading.Thread(target=robot_worker, daemon=True)
    thread_robot.start()

    start_visualization()