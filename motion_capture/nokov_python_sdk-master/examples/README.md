# Motion Capture Data Processing Scripts

This README provides a brief explanation of the functionality of three Python scripts designed for processing motion capture data using the Nokov SDK.

## 1. Nokov_SDK_Client.py

This script serves as a basic client for the Nokov motion capture SDK. It connects to the Nokov server, retrieves motion capture data frames, and prints out information about markersets, rigid bodies, and unidentified markers.

The script provides:
- `py_data_func` — callback function called each time a new frame of motion capture data is available
- `py_msg_func` — message callback to handle log messages from the SDK
- `py_forcePlate_func` — callback to process force plate data

## 2. Nokov_SDK_Client_With_Vel_Acc_Degree.py

This script extends the functionality of the basic Nokov SDK client to include the calculation of velocity, acceleration, and angle changes for markers. It uses the `Utility` module to perform these calculations.

The script assumes that the first markerset with three markers represents a finger, and it calculates the angle between the lines formed by the fingertip, finger middle, and finger root. It also calculates the velocity and acceleration of the fingertip marker.

### Instructions for Calculating Finger Joints

**Attachment:** Place one marker on a selected finger, one on a joint, and one on the fingertip, joint, and base of the finger, respectively.

**Action:** Finger bending

**Motion Capture Software Operation:** Create a markerset for the finger, with one markerset corresponding to one finger, containing three marker points corresponding to the fingertip, joint, and base of the finger. Multiple markersets can be created simultaneously for multiple fingers.

**Data Collection:**

To obtain the coordinates of each point in real-time, the acceleration of the fingertip, and the angle change of the joint (line connecting three points), run `Nokov_SDK_Client_With_Vel_Acc_Degree.py`. Locate line 41 of the code, as shown below. This example uses the first markerset; comment out this line to simultaneously obtain data for multiple fingers.

```python
if (0 == iMarkerSet and 3 == markerset.nMarkers):
```

## 3. Utility.py

The `Utility` module provides auxiliary functions for external calculation of speed, acceleration, and angle support. It defines:

- Classes for representing points, velocity, and acceleration in 3D space
- Methods for calculating these values based on multiple frames of motion capture data
- `SlideFrameArray` class for managing a sliding window of frames, useful for calculating velocity and acceleration over time

---

> **Note:** The scripts are designed to work with the Nokov motion capture system and require the Nokov SDK to function properly. The server IP address must be set correctly for the client scripts to connect to the Nokov server.
