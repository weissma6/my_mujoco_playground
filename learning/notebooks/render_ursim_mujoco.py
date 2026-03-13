from URSim_RTDE_dependencies import URSimRTDEControlFeedback
import time
import mujoco
import mujoco.viewer
from IPython.display import clear_output


robot = URSimRTDEControlFeedback()
xml_path = "mujoco_playground/_src/manipulation/my_ur10/xmls/mjx_single_cube_position.xml"


model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

joint_qpos_indices = [0, 1, 2, 3, 4, 5]

frequency_hz = 50.0
dt = 1.0 / frequency_hz
max_time_s = 10.0

start_t = time.perf_counter()

with mujoco.viewer.launch_passive(model, data) as viewer:
    next_t = time.perf_counter()

    try:
        while viewer.is_running():
            fb = robot.receive_feedback()
            q = fb["q"]
            tcp_xyz = fb["tcp_xyz"]

            for i, q_idx in enumerate(joint_qpos_indices):
                data.qpos[q_idx] = q[i]

            mujoco.mj_forward(model, data)
            viewer.sync()

            clear_output(wait=True)
            print("q      =", [round(v, 3) for v in q])
            print("tcp_xyz=", [round(v, 3) for v in tcp_xyz])

            if time.perf_counter() - start_t > max_time_s:
                break

            next_t += dt
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        robot.disconnect()