# nokov_python_vrpn-master

VRPN client for the lab Nokov (XINGYING) motion-capture server — the VRPN
counterpart of `../nokov_python_sdk-master`.

The Nokov server broadcasts every rigid body and marker as a **VRPN tracker** on
TCP/UDP port **3883** (no auth — only the server IP is needed, like the SDK).
This folder vendors a built `vrpn` Python module plus a tiny discovery helper so
the rest of the repo can stream poses over VRPN without compiling anything.

For the project's reusable, thread-safe reader (the analog of
`mocap_dependencies.py`), see
`../mymocap/vrpn_dependencies.py` → `VRPNRigidBodyReader`.

## Layout

```
dist/vrpn.so          hand-coded VRPN 3.x Python binding (the importable module)
bin/discover_senders  helper that lists every tracker the server publishes
src/discover_senders.cpp  source for that helper
examples/vrpn_tracker_client.py  raw-module streaming demo (no project wrappers)
```

## Quick start

```bash
# 1. What does the server publish right now?  (rigid bodies AND markers)
./bin/discover_senders 10.1.1.198

# 2. Stream specific trackers with the raw module:
python examples/vrpn_tracker_client.py 10.1.1.198 PUBeam_11 CubeInCube_Marker1

# 3. Or use the project reader (discovers + streams ALL by default):
python ../mymocap/vrpn_dependencies.py --server-ip 10.1.1.198            # all
python ../mymocap/vrpn_dependencies.py --name PUBeam_11                  # one
python ../mymocap/vrpn_dependencies.py --list                           # names
```

`import vrpn` resolves because `vrpn_dependencies.py` / the example prepend
`dist/` to `sys.path`. To make it importable everywhere instead, copy it into
the venv:

```bash
cp dist/vrpn.so "$VIRTUAL_ENV/lib/python3.10/site-packages/"
```

## Data conventions

* `vrpn.receiver.Tracker(f"{name}@{host}")` + `register_change_handler(userdata,
  callback, "position")`; drive it by calling `tracker.mainloop()` in a loop.
* The position callback receives `data["position"]` = `(x, y, z)` in **meters**
  (VRPN does not use mm, unlike the Nokov SDK) and `data["quaternion"]` =
  `(x, y, z, w)`. `VRPNRigidBodyReader` re-orders this to `(w, x, y, z)` to match
  `NokovRigidBodyReader`.
* A tracker the server *knows* but isn't currently streaming (rigid body not
  visible to the cameras) produces no reports and VRPN logs
  `No response from server for >= 3 seconds`. That is normal, not an error.

## Server prerequisites

Same as the SDK path: on the Windows mocap PC, XINGYING must have **SDK/VRPN
broadcast enabled** and at least one rigid body / markerset defined and visible
to the cameras. The host must be on the `10.1.1.0/24` mocap subnet (server at
`10.1.1.198`).

## Provenance / rebuilding

`dist/vrpn.so` and `bin/discover_senders` were built from the upstream VRPN
source (https://github.com/vrpn/vrpn, suite version 07.38) against this repo's
venv Python 3.10. `cmake` was installed as a pip wheel (`uv pip install cmake`);
no `swig` is needed (the hand-coded 3.x binding is used, not the SWIG one).

```bash
git clone --depth 1 --recursive https://github.com/vrpn/vrpn.git
cd vrpn && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DVRPN_BUILD_PYTHON_HANDCODED_3X=ON \
  -DVRPN_BUILD_CLIENTS=ON -DVRPN_BUILD_SERVERS=OFF \
  -DPython_EXECUTABLE="$(which python)" -DPython_FIND_VIRTUALENV=ONLY \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --target vrpn-python vrpn_print_devices -j"$(nproc)"
# -> build/python/vrpn.so  (copy to dist/)

# discovery helper (links the static libvrpn.a):
g++ ../discover_senders.cpp -I.. -o discover_senders ./libvrpn.a -lpthread
```

`src/discover_senders.cpp` is kept here so the helper can be rebuilt without the
full VRPN checkout (point `-I` and the static lib at a local `libvrpn.a`).
