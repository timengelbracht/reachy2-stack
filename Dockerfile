FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------------------------
# OS deps
# -------------------------------------------------------------------------
RUN sed -i 's|archive.ubuntu.com|ch.archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
        build-essential git wget curl gnupg2 lsb-release software-properties-common \
        libeigen3-dev libopencv-dev libgl1-mesa-glx libx11-dev libglfw3-dev \
        libglew-dev libtbb-dev libjsoncpp-dev libspdlog-dev libfmt-dev \
        libjpeg-dev libpng-dev libtiff-dev libxi-dev libxxf86vm-dev \
        libxcursor-dev libxinerama-dev libc++-dev libc++abi-dev clang zstd \
        ffmpeg \
        # pyenv build deps
        make libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------
# Install pyenv + Python 3.12.12
# -------------------------------------------------------------------------
ENV PYENV_ROOT=/root/.pyenv
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

RUN curl -fsSL https://pyenv.run | bash

# Install and set global Python 3.12.12
RUN bash -lc 'pyenv install 3.12.12 && pyenv global 3.12.12 && python --version'

# Make sure "python3" also points to pyenv's python
RUN ln -sf "$(command -v python)" /usr/bin/python3 && \
    ln -sf "$(command -v python)" /usr/bin/python

# Upgrade pip for 3.12
RUN python -m pip install --upgrade pip

# Install latest CMake (3.27+)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh && \
    chmod +x cmake-3.27.9-linux-x86_64.sh && \
    ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.27.9-linux-x86_64.sh

# CUDA 12.2 installation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y cuda-toolkit-12-2

ENV PATH=/usr/local/cuda/bin:${PATH}
#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}


# ---- extra GUI deps for GLFW / Filament -------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libxkbcommon-dev libwayland-dev libxrandr-dev libxi-dev \
        libxinerama-dev libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*


# 1. Clone Open3D (cached separately)
RUN git clone --recursive https://github.com/isl-org/Open3D /open3d

# 2. Create build dir
RUN mkdir -p /open3d/build

# 3. Configure CMake
RUN cd /open3d/build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_CUDA_MODULE=ON \
      -DCUDA_ARCH_NAME=Auto \
      -DBUILD_GUI=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DPYTHON_EXECUTABLE=$(which python3)

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. Build Open3D (heavy step — cached unless sources change)
RUN cd /open3d/build && \
    make -j6

# 5. Install to /usr/local (fast)
RUN cd /open3d/build && \
    make install

# 6. Build pip wheel
RUN cd /open3d/build && \
    make pip-package -j6

# 7. Install the wheel
RUN pip install --ignore-installed /open3d/build/lib/python_package/pip_package/open3d-*.whl

ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

CMD ["/bin/bash"]
