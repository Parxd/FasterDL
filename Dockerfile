# Base img
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL authors="parxed"

# Environment vars
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

# Tools
RUN apt-get update &&  \
    apt-get install -y \
    cmake \
    g++ \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Dir setup
WORKDIR /workspaces
# COPY . /app

# Build
# RUN mkdir build && \
#     cd build && \
#     cmake . && \
#     make
#
# Run exec
# CMD ["./app"]
