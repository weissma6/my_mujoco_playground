"""UR3e initial-pose libraries (freedrive-collected robot start configurations).

These per-level JSON files hold real UR3e start poses recorded by hand-guiding the
arm (see ``robots/UR3e/collect_init_poses.py``). The SAME files feed BOTH:
  * training  — the env reset samples a start pose for the "library" condition, and
  * evaluation — the sim and real eval harnesses reset/move to each held-out pose.

Layout (inside this env package so the SLURM cluster ships it with the code and the
loader pulls in no RTDE / robots-folder dependencies)::

    init_poses/
      <split>/<level>.json        split in {train, test}, level in {light, mid, hard}

train and test poses are collected in SEPARATE sessions → SEPARATE files; no pose
appears in both. Each file::

    {
      "metadata": {
        "robot": "UR3e", "gripper": "Robotiq HandE", "split": "train",
        "joint_order": ["shoulder_pan", "shoulder_lift", "elbow",
                        "wrist_1", "wrist_2", "wrist_3"],
        "finger_units": "sim_m[0,0.025]",
        "capture": {"duration_s": 60, "interval_s": 2, "source": "RTDE getActualQ"},
        "created": "<ISO>", "robot_ip": "<ip>", "seed": <int>
      },
      "poses": [
        {"arm_qpos": [6 floats, rad], "finger": <float, sim m in [0, 0.025]>,
         "level": "<level>", "captured_at": "<ISO>", "tcp_pose": [6 floats]?},
        ...
      ]
    }

``finger`` is in sim Hand-E per-finger meters [0, 0.025] (0 = open, 0.025 = closed),
matching ``robots/hande/HandE_dependency.py`` and the finger qpos in ``ur3_pick``.
"""

import json
import os

import numpy as np

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def load_init_poses(level: str, split: str, root: str = None) -> np.ndarray:
  """Load one init-pose library as an ``(N, 7)`` array ``[6 arm rad + 1 finger m]``.

  Args:
    level: difficulty tier — ``"light"`` | ``"mid"`` | ``"hard"`` (the file stem).
    split: ``"train"`` or ``"test"`` (the sub-directory). train and test never
      share a pose.
    root: directory holding the ``<split>/`` sub-dirs. ``None`` resolves to this
      package's own directory (cwd-independent), so the env and the eval scripts
      get the same data regardless of where they run.

  Returns:
    ``np.ndarray`` of shape ``(N, 7)``: columns 0-5 are the arm joint angles
    (rad, in ``JOINT_ORDER``), column 6 is the sim finger position (m, [0, 0.025]).

  Raises:
    FileNotFoundError: the ``<root>/<split>/<level>.json`` file does not exist.
    ValueError: the file has no poses, or a pose's ``arm_qpos`` is not length 6.
  """
  if root is None:
    root = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(root, split, f"{level}.json")
  if not os.path.exists(path):
    raise FileNotFoundError(
        f"init-pose library not found: {path} "
        f"(collect it with robots/UR3e/collect_init_poses.py "
        f"--level {level} --split {split})"
    )
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

  poses = data.get("poses", [])
  if not poses:
    raise ValueError(f"init-pose library has no poses: {path}")

  rows = []
  for i, p in enumerate(poses):
    arm = p["arm_qpos"]
    if len(arm) != 6:
      raise ValueError(
          f"pose {i} in {path} has arm_qpos length {len(arm)}, expected 6"
      )
    rows.append([float(x) for x in arm] + [float(p["finger"])])
  return np.asarray(rows, dtype=np.float64)
