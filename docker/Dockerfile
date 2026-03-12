# =============================================================================
# Base Image and Build Arguments
# =============================================================================

# Build arguments
ARG ros_distro=jazzy
ARG gid
ARG uid
ARG username
ARG install_cuda=false
ARG is_jetson=false
ARG BLACKWELL_GPU=false
ARG BASE_STAGE_NAME=base_amd64

# =============================================================================
# Stage 1: Platform-Specific Base Images
# =============================================================================
# For Robot (ARM64/Jetson Thor):
ARG BASE_IMAGE_ARM64=nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE_ARM64} AS base_arm64

ARG is_jetson
ARG install_cuda
ARG L4T_RELEASE=r38.2
ENV DEBIAN_FRONTEND=noninteractive

# Setup apt caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install the jetson package repository (only on actual Jetson devices)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$is_jetson" = "true" ]; then \
        apt-get update && apt-get install -y wget && \
        mkdir -p /etc/apt/keyrings && \
        wget -O /etc/apt/keyrings/jetson-ota-public.asc https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
        printf "deb [signed-by=/etc/apt/keyrings/jetson-ota-public.asc] https://repo.download.nvidia.com/jetson/common %s main\n" "$L4T_RELEASE" \
        > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
        printf "deb [signed-by=/etc/apt/keyrings/jetson-ota-public.asc] https://repo.download.nvidia.com/jetson/som %s main\n" "$L4T_RELEASE" \
        >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
        printf "deb [signed-by=/etc/apt/keyrings/jetson-ota-public.asc] https://repo.download.nvidia.com/jetson/ffmpeg %s main\n" "$L4T_RELEASE" \
        >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list; \
    fi

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$install_cuda" = "true" ] && [ "$is_jetson" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
        libnvinfer10 libnvinfer-dev python3-libnvinfer \
        vpi4-dev vpi4-samples vpi4-python-src \
        nvpl-lapack nvpl-blas nvpl-common libgomp1; \
    fi

# For Desktop (AMD64/x86_64): Use Ubuntu 24.04 with optional CUDA
FROM ubuntu:24.04 AS base_amd64

ARG install_cuda
ARG BLACKWELL_GPU
ENV DEBIAN_FRONTEND=noninteractive

# Setup apt caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$install_cuda" = "true" ]; then \
        CUDA_TOOLKIT_PKG="cuda-toolkit-12-6"; \
        if [ "$BLACKWELL_GPU" = "true" ] || (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX 50"); then \
            echo ">>> DETECTED BLACKWELL GPU (RTX 50 Series). Using CUDA 12.8"; \
            CUDA_TOOLKIT_PKG="cuda-toolkit-12-8"; \
        fi && \
        echo "Installing CUDA Toolkit: ${CUDA_TOOLKIT_PKG}" && \
        apt-get update && apt-get install -y wget gnupg2 && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        rm cuda-keyring_1.1-1_all.deb && \
        apt-get update && apt-get install -y ${CUDA_TOOLKIT_PKG} libcudnn9-cuda-12 libcudnn9-dev-cuda-12; \
    fi

# =============================================================================
# Stage 2: Select Base - Auto-detects platform during native build
# =============================================================================
ARG BASE_STAGE_NAME
FROM $BASE_STAGE_NAME AS base-selected

ARG ros_distro=jazzy
ARG gid
ARG uid
ARG username

# =============================================================================
# Stage 3: Base Setup - Locale and Essential Tools
# =============================================================================
FROM base-selected AS base-setup

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# install UV
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Install base dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential cmake git vim nano wget curl gnupg2 lsb-release \
    iproute2 software-properties-common tmux tmuxinator xz-utils libcap-dev libdw-dev

# =============================================================================
# Stage 4: ROS2 Installation (Jazzy)
# =============================================================================
FROM base-setup AS ros2-install

ARG ros_distro
ARG is_jetson

# ROS 2 Jazzy repo
RUN set -eux; \
    ROS_APT_VER="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F\" '{print $4}')"; \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_VER}/ros2-apt-source_${ROS_APT_VER}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-noble})_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb; rm /tmp/ros2-apt-source.deb

# NVIDIA Isaac ROS APT repo (Release 4.0 for Jazzy/Noble)
RUN if [ "$is_jetson" = "true" ]; then \
        set -eux; \
        curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | gpg --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg; \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.0 $(. /etc/os-release && echo ${UBUNTU_CODENAME:-noble}) main" > /etc/apt/sources.list.d/nvidia-isaac-ros.list; \
    fi

# Install ROS 2 desktop + dev tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-$ros_distro-desktop ros-dev-tools \
    alsa-utils fastdds-tools lsof libglfw3-dev libcli11-dev libhidapi-hidraw0 \
    libhidapi-libusb0 libusb-1.0-0 joystick liblua5.1-dev libgoogle-glog-dev \
    libgflags-dev libpopt-dev libmodbus-dev libgtest-dev

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$is_jetson" = "true" ]; then \
        apt-get update && \
        (apt-get install -y --no-install-recommends ros-$ros_distro-isaac-ros-common || echo "Isaac ROS not available"); \
    fi

# Initialize rosdep
RUN rosdep init && rosdep update

COPY scripts/ros2_deps.sh /tmp/ros2_deps.sh
RUN chmod +x /tmp/ros2_deps.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    /tmp/ros2_deps.sh && rm /tmp/ros2_deps.sh

# =============================================================================
# Stage 4: Display & MoveIt Dependencies
# =============================================================================
FROM ros2-install AS display-deps

# EGL, OpenGL libraries (for MuJoCo rendering and GUI)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    libegl1 libegl1-mesa-dev libgles2-mesa-dev libgles2 libapr1-dev libglvnd-dev libgl1-mesa-dev libgl1-mesa-dri mesa-utils mesa-utils-extra xorg-dev libglu1-mesa-dev

# Install Xfce4 and TurboVNC (for headless display)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    xfce4 xfce4-goodies dbus-x11

ARG TARGETARCH
RUN wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_amd64.deb -O /tmp/turbovnc.deb && \
    dpkg -i /tmp/turbovnc.deb && rm /tmp/turbovnc.deb && \
    ln -s /opt/TurboVNC/bin/* /usr/local/bin/

COPY scripts/vnc /usr/local/bin/vnc
RUN chmod +x /usr/local/bin/vnc

# MoveIt dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    libbenchmark-dev libaprutil1-dev python3-pygraphviz freeglut3-dev libglew-dev

# =============================================================================
# Stage 5: Python Setup
# =============================================================================
FROM display-deps AS python-setup

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    python3-dev python3-venv python3-pip protobuf-compiler libprotobuf-dev

# Downgrade setuptools to < 70 to avoid colcon symlink-install breakages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages setuptools==69.5.1

# =============================================================================
# Stage 6: Simulator (MuJoCo, Madrona, JAX)
# =============================================================================
FROM python-setup AS simulator

# Install system dependencies required for Madrona and MuJoCo
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev

# Install JAX for CUDA 12
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -U "jax[cuda12]<0.6.0"

# Clone and install MuJoCo Playground
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone -b v0.1.0 https://github.com/google-deepmind/mujoco_playground.git /opt/mujoco_playground && \
    cd /opt/mujoco_playground && \
    pip3 install --break-system-packages --ignore-installed scipy numpy -e .

# Clone, compile, and install Madrona MJX
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/shacklettbp/madrona_mjx.git /opt/madrona_mjx && \
    cd /opt/madrona_mjx && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake -DLOAD_VULKAN=OFF .. && \
    make -j$(nproc) && \
    cd .. && \
    pip3 install --break-system-packages --ignore-installed scipy numpy -e .

# Enforce full precision for JAX on Ampere GPUs
ENV JAX_DEFAULT_MATMUL_PRECISION=highest

# =============================================================================
# Stage 7: User Setup
# =============================================================================
FROM simulator AS final

ARG uid
ARG gid
ARG username
ARG ros_distro

ENV UID=$uid
ENV GID=$gid
ENV USERNAME=$username
ENV ROS_DISTRO=$ros_distro
ENV CONTAINER_NAME=sim_ws

# Create user and group, then configure sudo
RUN set -e && \
    getent group ${GID} || groupadd -g ${GID} ${USERNAME} && \
    if id ${UID} >/dev/null 2>&1; then \
        existing_user=$(id -nu ${UID}) && \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m $existing_user 2>/dev/null || true && \
        usermod -g ${GID} ${USERNAME} 2>/dev/null || true; \
    else \
        useradd -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME}; \
    fi && \
    mkdir -p /home/${USERNAME} && \
    chmod 777 /home/${USERNAME} && \
    (chown -R ${UID}:${GID} /home/${USERNAME} || true) && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    visudo -c && \
    usermod -aG video ${USERNAME} && \
    usermod -aG input ${USERNAME}

# Create groups matching the host's group IDs (i2c, dialout, input) for Jetson
RUN set -e && \
    if getent group i2c >/dev/null 2>&1 && getent group 116 >/dev/null 2>&1; then \
        existing_group=$(getent group 116 | cut -d: -f1) && \
        if [ "$existing_group" != "i2c" ]; then groupdel i2c && groupmod -n i2c $existing_group; fi; \
    elif getent group 116 >/dev/null 2>&1; then existing_group=$(getent group 116 | cut -d: -f1) && groupmod -n i2c $existing_group; \
    elif getent group i2c >/dev/null 2>&1; then groupmod -g 116 i2c; \
    else groupadd -g 116 i2c; fi && \
    if getent group dialout >/dev/null 2>&1 && getent group 20 >/dev/null 2>&1; then \
        existing_group=$(getent group 20 | cut -d: -f1) && \
        if [ "$existing_group" != "dialout" ]; then groupdel dialout && groupmod -n dialout $existing_group; fi; \
    elif getent group 20 >/dev/null 2>&1; then existing_group=$(getent group 20 | cut -d: -f1) && groupmod -n dialout $existing_group; \
    elif getent group dialout >/dev/null 2>&1; then groupmod -g 20 dialout; \
    else groupadd -g 20 dialout; fi && \
    if ! getent group input > /dev/null 2>&1; then groupadd -r input; fi && \
    usermod -aG i2c,dialout,input ${USERNAME}

COPY docker/bash_profile.shared /tmp/
COPY docker/bash_profile.desktop /tmp/
COPY scripts/setup_bash_profile.sh /tmp/
RUN chmod +x /tmp/setup_bash_profile.sh && \
    ARCH=$(dpkg --print-architecture) && \
    /tmp/setup_bash_profile.sh ${ARCH} ${USERNAME} ${ROS_DISTRO} ${GID} && \
    rm -f /tmp/bash_profile.shared /tmp/bash_profile.desktop /tmp/setup_bash_profile.sh

RUN echo "set -g mouse on" >> /home/${USERNAME}/.tmux.conf && \
    echo "set -g history-limit 10000" >> /home/${USERNAME}/.tmux.conf && \
    (chown ${USERNAME}:${GID} /home/${USERNAME}/.tmux.conf || true)

# Downgrade setuptools to avoid colcon symlink-install breakage in ROS 2
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages setuptools==69.5.1

# Fix cache permissions if ros cache exists
RUN mkdir -p /home/${USERNAME}/.cache && chmod -R 777 /home/${USERNAME}/.cache

USER $username
ENV HOME=/home/$username
WORKDIR /home/$username
