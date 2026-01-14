#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

module load podman

# Optional: show which python is on host (for debugging)
which python || true
python -c "import sys; print(sys.executable)" || true

# Run INSIDE the container, not on the host
podman run --rm \
  -v "$HOME/VT1_MBRL:/workspace/VT1_MBRL:Z" \
  -w /workspace/VT1_MBRL/my_mujoco_playground \
  localhost/mujoco:env_EGL \
  bash -lc '
    python - << EOF
import sys, jax, jaxlib, importlib.util
print("exe:", sys.executable)
print("jax:", jax.__version__, jax.__file__)
print("jaxlib:", jaxlib.__version__, jaxlib.__file__)
print("mhlo:", importlib.util.find_spec("jaxlib.mlir.dialects.mhlo"))
print("devices:", jax.devices())
EOF
    python batch_runs/scripts/run_one_ur10.py
  '
