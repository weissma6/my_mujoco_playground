# Files

1. **Dist** — The directory of installation package (support python3)
2. **Script** — The directory of python test scripts
3. **vcredist_x64.exe** — VS2019 runtime library. If the `msvcr140.dll` is lost, double-click to install it

# Python Runtime Environment Configuration

1. Install Python3 64-bit from [python.org](http://python.org/download/)

2. Install the Nokov module:
   - Go to the `Dist` directory and perform the following operations (Windows x64 example):
     ```bash
     pip install nokovpy-3.0.1-py3-none-any.whl
     ```

3. Test module — go to the `Script` directory:
   - Run `python Nokov_SDK_Client.py` and change the IP address of XINGYING as required:
     ```bash
     python Nokov_SDK_Client.py -s 10.1.1.198
     ```
   - Using the Python interpreter, type the relevant Python commands as described in the function definitions and comments in `nokovsdk.py` under the Python package installation path (e.g., `C:\Users\Administrator\.pyenv\pyenv-win\versions\3.9.6\Lib\site-packages\nokov\nokovsdk.py`). See also the `Nokov_SDK_Client.py` script.

# Demo Test

1. Set the PC IP address to `10.1.1.198` and subnet mask to `255.255.255.0`. Disable the firewall and network blocking software.
2. Run XINGYING motion capture software as an administrator.
3. Select the data broadcast interface, select "NIC Address", and then check "Setting" → "SDK Enabled". XINGYING sends motion capture data; the local and other computers can receive data.
4. In live mode, click run; or, in post-process mode, play back data.
5. After configuring the Python runtime environment according to the instructions, double-click the `Nokov_SDK_Client.py` script to receive SDK data.

> **Note:** In post-process mode, you need to close the client receiver before switching.

# Data Instructions

1. The coordinate system is right-handed.
2. For the Marker defined in Markerset, due to software operation problems, Marker occlusion and other reasons, the X, Y and Z coordinate values will be filled with `9999999.000000` when points are lost or cannot be identified.
3. For Skeleton Builder, Rx, Ry, Rz may exceed 360 degrees during the movement of the skeleton (e.g., for two rotations, the rotation angle is 720 degrees).
4. Euler Angle rotation order: see **Markerset Properties → Rotation Order**. The default rotation order is ZYX.
5. Bone axis orientation: see **Markerset Properties → Bone Axis**. The default bone axis is positive in the Y direction.
6. The six-degree-of-freedom information of the skeleton refers to the translation and rotation of the skeleton relative to the parent segment. When creating a Markerset, the skeleton defined by default is relative to the geodetic coordinate system.
