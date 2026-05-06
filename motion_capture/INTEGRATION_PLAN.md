# Motion Capture Integration Plan

**Date:** 2026-05-06  
**Status:** ✓ Test scripts and reader module created  

---

## Quick Reference

| Component | Purpose | Status |
|-----------|---------|--------|
| `test_mocap_connection.py` | Verify XINYANG connectivity, test marker detection | ✓ Created |
| `motion_capture_reader.py` | Frequency-independent marker reading (threaded) | ✓ Created |
| `URSim_RTDE_dependencies.py` | (Modify) Add motion capture target support | TODO |
| `UR10_RealRobot_Reach_ONE.py` | (Modify) Use mocap target instead of hardcoded | TODO |
| CLAUDE.md | Document mocap integration workflow | TODO |

---

## Architecture

```
XINYANG Motion Capture System (hardware)
  ↓ UDP ~60Hz
┌─────────────────────────────────────┐
│ motion_capture_reader.py            │
│ (background thread)                 │
│ - Connects to 10.1.1.198            │
│ - Receives marker frames            │
│ - Stores latest positions (thread-safe)
│ - Provides query API                │
└─────────────────────────────────────┘
  ↑ (frequency-independent polling)
┌─────────────────────────────────────┐
│ Control Loop (50Hz)                 │
│ UR10_RealRobot_Reach_ONE.py         │
│ - Queries latest marker position    │
│ - Uses as target_pos for reaching   │
│ - Runs policy at 50Hz               │
│ - Independent of mocap frequency    │
└─────────────────────────────────────┘
  ↓
Real Robot + RTDE
```

---

## Step 1: Test Motion Capture Connectivity

### Setup Network (macOS/Linux)

```bash
# Find USB Ethernet adapter
ifconfig | grep -A 5 enx

# Configure IP on that interface (example)
sudo ip addr add 10.1.1.51/24 dev enx00e04c449bd3

# Verify connectivity
ping 10.1.1.198
```

### Run Test Script

```bash
cd my_mujoco_playground/motion_capture
python test_mocap_connection.py
```

**Expected output:**
```
Connected to XINYANG server
✓ Callback registered, waiting for marker data...

[0.0s] Frame #123 | 4 markers detected
  [1.0s] 4 markers detected:
    labeled_0_0 → X:  0.345 Y: -0.123 Z:  0.789 m
    labeled_0_1 → X:  0.356 Y: -0.115 Z:  0.795 m
    ...
    [Centroid] → X:  0.350 Y: -0.119 Z:  0.792 m
```

**Troubleshooting:**
- Connection failed? Check IP, adapter, firewall
- No markers? Ensure markers are in view of cameras
- Jittery? Normal for mocap systems

---

## Step 2: Integrate into Real Robot Control

### Modify `URSim_RTDE_dependencies.py`

Add at top:
```python
from motion_capture.motion_capture_reader import XINYINGReader
```

In `__init__`:
```python
self.mocap_reader = None
```

Add method:
```python
def initialize_mocap(self, server_ip: str = "10.1.1.198", timeout: float = 5.0):
    """Start motion capture reader in background thread."""
    self.mocap_reader = XINYINGReader(server_ip)
    success = self.mocap_reader.start(timeout=timeout)
    if success:
        print(f"✓ Motion capture connected, tracking markers at ~60Hz")
    else:
        print(f"✗ Motion capture failed to connect")
    return success

def get_mocap_target(self) -> Optional[np.ndarray]:
    """Get latest target XYZ from motion capture system."""
    if self.mocap_reader and self.mocap_reader.is_connected():
        # Option 1: Single independent marker
        xyz = self.mocap_reader.get_first_marker_position()
        
        # Option 2: Centroid of rigid body (multiple markers)
        # xyz = self.mocap_reader.get_marker_centroid(["labeled_0_0", "labeled_0_1", ...])
        
        if xyz:
            return np.array(xyz)
    return None
```

### Modify `UR10_RealRobot_Reach_ONE.py`

```python
# At setup:
controller = URSimRTDESimpleReach(host=ROBOT_IP)
controller.load_policy(...)

# Initialize mocap (background thread starts here)
if not controller.initialize_mocap():
    print("Warning: Motion capture not available, using fallback")
    use_mocap = False
else:
    use_mocap = True

# In control loop:
while running:
    # Get current joint state
    q_measured = controller.receive_feedback()["q"]
    
    # Get target (mocap or fallback)
    if use_mocap:
        target_xyz = controller.get_mocap_target()
        if target_xyz is None:
            target_xyz = hardcoded_fallback  # Fallback if no markers
    else:
        target_xyz = hardcoded_fallback
    
    # Build observation (add target from mocap)
    obs = build_obs(q_measured, tcp_pos, target_xyz)
    
    # Policy inference
    action = policy(obs)
    
    # Control
    controller.send_servoj(q_measured + action * action_scale)
```

---

## Step 3: Target Definition

### Option A: Single Cube (Simplest)

Place **one marker** on cube. Script detects first marker:
```python
target_xyz = self.mocap_reader.get_first_marker_position()
```

### Option B: Rigid Body (Multiple Markers)

Attach 3+ markers to cube in known pattern. Script computes centroid:
```python
target_xyz = self.mocap_reader.get_marker_centroid([
    "labeled_0_0",  # Left corner
    "labeled_0_1",  # Right corner
    "labeled_0_2",  # Front corner
])
```

### Option C: Filtered Region

Place markers anywhere, script finds them in reach volume:
```python
# Find all markers within 0.5m of table center (1.0, 1.0, 0.05)
markers_in_region = self.mocap_reader.get_markers_in_region(
    center=(1.0, 1.0, 0.05),
    radius=0.5
)
if markers_in_region:
    target_xyz = markers_in_region[0][1]  # Use first one
```

---

## Step 4: Frequency Independence (Key Concept)

Your control loop runs at **50Hz** (0.02s dt).  
Motion capture runs at **~60Hz** (0.017s dt).

### Without frequency independence (WRONG):
```
Frame #1 arrives (60Hz)  → build obs → send command (50Hz)
                          → miss frames when 60Hz lands between 50Hz ticks
                          → inconsistent target position
```

### With frequency independence (CORRECT):
```
60Hz thread:  Frame #1  Frame #2  Frame #3  Frame #4 ...
                ↓        ↓        ↓         ↓
              (stored)  (stored)  (stored)  (stored)

50Hz loop:    Get latest  Get latest  Get latest
              (always use Frame #3)  (always use Frame #4)
              ✓ Consistent, no racing
```

The reader module handles this automatically:
- Background thread updates `latest_markers` continuously (~60Hz)
- Control loop calls `get_first_marker_position()` whenever it needs (50Hz)
- Thread-safe `Lock` prevents race conditions

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Connection refused | Network misconfiguration | Run `ping 10.1.1.198`, check adapter config |
| No markers detected | Markers not in camera view | Move objects in front of cameras |
| Jittery motion | Normal mocap noise | Add smoothing: average last N frames |
| Control loop crashes | None = no markers | Add fallback to hardcoded target |
| Frequency mismatch errors | Accessing data in wrong thread | Use provided reader module (thread-safe) |

---

## Files Overview

### `test_mocap_connection.py`
- Standalone test script
- Connects to XINYANG, displays markers in real-time
- No integration with robot yet
- **Run this first** to verify hardware setup

### `motion_capture_reader.py`
- Reusable reader module
- Background thread + thread-safe dict
- Public API: `get_first_marker_position()`, `get_marker_centroid()`, etc.
- Can be imported and used in any script

### Changes needed in robot control files
- `URSim_RTDE_dependencies.py` — add `initialize_mocap()` + `get_mocap_target()`
- `UR10_RealRobot_Reach_ONE.py` — use `get_mocap_target()` in loop

---

## Next Steps

1. **Test connectivity**
   ```bash
   python motion_capture/test_mocap_connection.py
   ```

2. **Verify marker detection**
   - Ensure markers on cube are visible
   - Check frame rate (~60Hz)

3. **Integrate into robot control**
   - Modify `URSimRTDESimpleReach` class
   - Add `initialize_mocap()` call in setup

4. **Test on real robot**
   - Run `UR10_RealRobot_Reach_ONE.py`
   - Monitor marker tracking in control loop
   - Adjust target filtering if needed

5. **Update CLAUDE.md**
   - Document motion capture workflow
   - Add troubleshooting section

---

## References

- XINYANG SDK: `motion_capture/XING_Python_SDK-4.1.0.5873/`
- Original visualizer: `motion_capture/XING_Python_SDK-4.1.0.5873/examples/visualise_raw.py`
- This plan: (this file)
