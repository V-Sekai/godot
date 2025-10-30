FROM almalinux:9

WORKDIR /workspace

RUN dnf update -y && dnf install -y \
    gcc g++ make git \
    python3 python3-devel \
    pkg-config \
    libX11-devel libXcursor-devel libXinerama-devel libXi-devel \
    libXrandr-devel libXext-devel libxkbcommon-devel \
    alsa-lib-devel \
    mesa-libGL-devel \
    libudev-devel \
    openssl-devel \
    zlib-devel \
    unzip \
    tmux

RUN dnf install -y epel-release && \
    dnf install -y erlang erlang-erl_interface

# Install Elixir from precompiled binaries
RUN cd /tmp && \
    curl -L https://github.com/elixir-lang/elixir/releases/download/v1.17.3/elixir-otp-27.zip -o elixir.zip && \
    unzip elixir.zip && \
    mkdir -p /opt/elixir && \
    mv bin lib man /opt/elixir/ && \
    rm elixir.zip

ENV PATH="/opt/elixir/bin:${PATH}"

RUN pip3 install scons

RUN dnf install -y \
    vim nano \
    wget \
    which \
    ca-certificates

RUN useradd -m -s /bin/bash -G wheel dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER dev

ENV PATH="/home/dev/.local/bin:${PATH}"
