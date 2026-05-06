import sys
import numpy as np
import time
import math
import threading
import viser
import viser.extras
import imageio  # Ensure you have pip install imageio[ffmpeg]
from robot_descriptions.loaders.yourdfpy import load_robot_description
from nokov.nokovsdk import *
import rtde_control
import rtde_receive

# ==========================================
# 1. MANUAL OFFSETS & CONFIGURATION
# ==========================================
X_OFFSET = 0.96
Y_OFFSET = 0.08
Z_OFFSET = 0.08

ROBOT_IP = "192.168.1.4"
SERVER_IP = '10.1.1.198'
TARGET_MARKERSET_NAME = "Rope"

T_MATRIX = np.array([
    [ 0.999023, -0.042866,  0.01073,  -0.918766],
    [ 0.042646,  0.998892,  0.019915,  0.049483],
    [-0.011572, -0.019438,  0.999744, -0.804454],
    [ 0.0,        0.0,        0.0,        1.0     ]
])

latest_rope_coords = None  
is_running = True 
robot_ready_to_move = threading.Event()

ROPE_COLORS = np.array([
    [255, 51, 51], [255, 153, 51], [255, 255, 51], [153, 255, 51],
    [51, 255, 51], [51, 255, 153], [51, 255, 255], [51, 153, 255],
    [51, 51, 255], [153, 51, 255], [255, 51, 255], [255, 51, 153]
]) 

# ==========================================
# 2. TRANSFORMATION LOGIC
# ==========================================
def transform_to_robot_frame(points_m):
    if points_m is None or len(points_m) == 0: return None
    ones = np.ones((points_m.shape[0], 1))
    points_homo = np.hstack([points_m, ones]) 
    transformed_homo = (T_MATRIX @ points_homo.T).T
    pts = transformed_homo[:, :3]
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
                latest_rope_coords = transform_to_robot_frame(np.array(coords))
            break

# ==========================================
# 4. ROBOT MOVEMENT THREAD
# ==========================================
def robot_worker():
    global is_running
    LOOP_FREQ = 100.0
    DT = 1.0 / LOOP_FREQ
    READY_POSE = [0.0, -math.pi/2, math.pi/2, math.pi, -math.pi/2, math.radians(143)]
    NUM_CYCLES = 50
    CYCLE_DURATION = 1.05
    AMPLITUDE_VELOCITY = 0.3
    ACCELERATION = 1.5

    try:
        print(f"[Robot] Connecting to {ROBOT_IP}...")
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP, frequency=LOOP_FREQ, flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP)
        print("[Robot] Moving to start...")
        rtde_c.moveJ(READY_POSE, 0.5, 0.5)
        
        robot_ready_to_move.wait()
        time.sleep(1.0)
        
        print("[Robot] Starting Sine Wave motion.")
        total_steps = int(NUM_CYCLES * CYCLE_DURATION * LOOP_FREQ)
        for i in range(total_steps):
            t_start = rtde_c.initPeriod()
            v_z = AMPLITUDE_VELOCITY * np.sin(2 * np.pi * (i * DT / CYCLE_DURATION))
            rtde_c.speedL([0.0, 0.0, v_z, 0.0, 0.0, 0.0], ACCELERATION, DT)
            rtde_c.waitPeriod(t_start)
        rtde_c.speedStop()
        rtde_c.stopScript()
    except Exception as e:
        print(f"[Robot Error]: {e}")
    finally:
        time.sleep(2.0) 
        is_running = False

# ==========================================
# 5. VISUALIZATION & VIDEO ENGINE
# ==========================================
def start_visualization():
    global latest_rope_coords, is_running
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    print(f"Visualizer ready at http://localhost:8080")

    try:
        urdf = load_robot_description("ur3e_description")
        viser_urdf = viser.extras.ViserUrdf(server, urdf)
    except Exception as e:
        print(f"Failed to load robot model: {e}")
        return

    # Keep lines hidden
    rope_line = server.scene.add_line_segments("/rope/line", points=np.zeros((1, 2, 3)), colors=(255, 255, 255), line_width=3.0, visible=False)
    marker_spheres = []
    video_frames = []
    
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except Exception as e:
        print(f"Receiver Error: {e}")
        return

    print("--- ACTION REQUIRED ---")
    print("1. Open http://localhost:8080")
    print("2. Position your camera.")
    
    while True:
        clients = server.get_clients()
        if len(clients) > 0:
            print("Browser detected! Starting recording in 3s...")
            time.sleep(3)
            robot_ready_to_move.set()
            break
        time.sleep(0.5)

    try:
        while is_running:
            with server.atomic():
                # A. Update Robot
                q = rtde_r.getActualQ()
                if q: viser_urdf.update_cfg(np.array(q))

                # B. Update Blobs
                if latest_rope_coords is not None:
                    pts = latest_rope_coords[~np.isnan(latest_rope_coords).any(axis=1)]
                    num_pts = len(pts)
                    if num_pts > 0:
                        palette = (ROPE_COLORS).astype(np.uint8)
                        for i in range(num_pts):
                            color = tuple(int(c) for c in palette[i % len(palette)])
                            if i >= len(marker_spheres):
                                marker_spheres.append(server.scene.add_icosphere(f"/rope/n_{i}", radius=0.007, color=color))
                            marker_spheres[i].position = pts[i]
                            marker_spheres[i].visible = True
                        for i in range(num_pts, len(marker_spheres)):
                            marker_spheres[i].visible = False

            # C. THE SMART CAPTURE FIX: Try all clients until one works
            clients = server.get_clients()
            frame_captured = False
            for client_id, client in clients.items():
                try:
                    # Request high-quality render
                    frame = client.request_render(width=1280, height=720)
                    if frame is not None:
                        video_frames.append(frame)
                        frame_captured = True
                        break # We found an active tab, no need to check others
                except Exception:
                    continue
            
            # Record at roughly 30 FPS
            time.sleep(0.033) 
            
    except KeyboardInterrupt:
        pass
    finally:
        if video_frames:
            print(f"Saving video ({len(video_frames)} frames)...")
            imageio.mimsave("rope_experiment.mp4", video_frames, fps=30)
            print("Video saved successfully as rope_experiment.mp4")
        else:
            print("CRITICAL: No frames captured. Ensure at least one browser tab is focused and visible.")
        rtde_r.disconnect()

if __name__ == '__main__':
    client = PySDKClient()
    if client.Initialize(bytes(SERVER_IP, encoding="utf8")) == 0:
        client.PySetDataCallback(py_data_func, None)
    else:
        sys.exit()

    thread_robot = threading.Thread(target=robot_worker, daemon=True)
    thread_robot.start()
    start_visualization()