"""
This file loads all the mujoco and checks for cuda of appe
"""

# @title Install pre-requisites (CPU-only for Apple Silicon)
import os
import platform
import subprocess
import sys
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import pickle
from mujoco_playground import wrapper
from mujoco_playground import registry
import mediapy as media
import matplotlib.pyplot as plt


def is_nvidia_available():
    """Check if nvidia-smi is callable (Linux/Colab with GPU)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


system = platform.system()
machine = platform.machine()

if system == "Darwin":  # macOS (Apple Silicon / Intel Mac)
    print("Detected macOS:", machine)
    # Force MuJoCo to use glfw rendering
    os.environ["MUJOCO_GL"] = "glfw"
    print("Using MUJOCO_GL=glfw for macOS")
    # Force JAX to CPU (avoid Metal/MPS backend issues)
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    print("Forcing JAX to run on CPU backend")

elif is_nvidia_available():  # Linux + NVIDIA GPU
    print("Detected NVIDIA GPU")
    os.environ["MUJOCO_GL"] = "egl"
    print("Using MUJOCO_GL=egl for GPU rendering")

    # Add missing EGL ICD config if necessary (Colab hack)
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write(
                """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""
            )
else:  # Fallback (no GPU)
    print("No NVIDIA GPU detected, running in CPU/OSMesa mode")
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

# --- Test Mujoco ---
try:
    import mujoco

    mujoco.MjModel.from_xml_string("<mujoco/>")
    print("Mujoco installation and rendering backend OK")
except Exception as e:
    print("❌ Mujoco test failed:", e)

    # Version check
import jax, mujoco, brax, flax

print("JAX:", jax.__version__)
print("MuJoCo:", mujoco.__version__)
print("Brax:", brax.__version__)
print("Flax:", flax.__version__)

# ==============================================================================
# Install ffmpeg if not already installed
# ==============================================================================
# Graphics and plotting
print("\nChecking media packages...")

# Check ffmpeg
try:
    subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
    )
    print("✓ ffmpeg available")
except:
    print("⚠ ffmpeg not found (optional for video rendering)")

# Import mediapy
try:
    import mediapy as media

    print("✓ mediapy available")
except ImportError:
    print("⚠ mediapy not installed (pip install mediapy)")
    media = None

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# =============================================================================
# ## Number of CPU Cores used
# =============================================================================
# must be done BEFORE "import jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


devices = jax.devices()
device_count = len(devices)
print(devices)
print("JAX device count:", device_count)
