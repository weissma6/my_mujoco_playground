FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# GPU-capable base image (cluster will have NVIDIA + CUDA)

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git \
    libosmesa6-dev patchelf \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the 'python' command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Working directory inside container
WORKDIR /workspace

# Install Python deps
# jax[cuda12_pip] from SOE docs
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --upgrade "jax[cuda12]" && \
    pip install -r /tmp/requirements.txt

# MuJoCo/Brax environment
ENV MUJOCO_GL=egl
ENV WANDB_DIR=/workspace/wandb

# Default command (will be overridden by sarus run)
CMD ["bash"]
