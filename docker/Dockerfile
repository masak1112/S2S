# syntax=docker/dockerfile:1.7
#
# Portable container for the S2S (subseasonal-to-seasonal) forecasting project.
#
# Built from scratch on a plain CUDA base image -- no conda, no NGC base --
# so the result is self-contained and converts cleanly to Apptainer/Singularity
# for HPC systems that do not allow Docker.
#
# Pins mirror the known-good `s2s_env` conda environment (torch 2.6.0 + cu124).
#
#   docker build -f docker/Dockerfile -t s2s:latest .
#
# apex / transformer_engine are intentionally omitted: every shipped config sets
# `use_transformer_engine: False`, both imports are lazy, and neither package is
# present in the working conda env. Use the `--build-arg WITH_MPI=true` flag if
# you need mpi4py for the data_utils/netcdf_to_h5_*.py preprocessing scripts.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG WITH_MPI=false
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System packages
#   libgeos/libproj  -> Cartopy and pyproj
#   libhdf5/libnetcdf -> h5py and netCDF4
#   libgl1/libglib   -> matplotlib backends
#   git              -> wandb records the repo commit at run time
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgeos-dev \
        libproj-dev \
        proj-bin \
        libhdf5-dev \
        libnetcdf-dev \
        libgl1 \
        libglib2.0-0 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Optional MPI layer, only for the data_utils preprocessing scripts.
RUN if [ "${WITH_MPI}" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libopenmpi-dev openmpi-bin \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Make `python`/`python3` resolve to the requested interpreter.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 && \
    python -m pip install --upgrade "pip>=24.0" "setuptools>=70.0" wheel

# ---------------------------------------------------------------------------
# PyTorch first, from the CUDA 12.4 wheel index, so the pinned CUDA build wins
# before anything else can drag in a default (CPU or cu12x) torch.
# ---------------------------------------------------------------------------
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0

# ---------------------------------------------------------------------------
# Remaining Python dependencies.
# Copied on its own so edits to source code do not invalidate this layer.
# ---------------------------------------------------------------------------
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN if [ "${WITH_MPI}" = "true" ]; then python -m pip install mpi4py==4.0.1; fi

# Fail the build now, not three hours into a training run, if the CUDA-linked
# extension modules did not import cleanly.
RUN python -c "\
import torch, h5py, netCDF4, xarray, cartopy, timm, einops, matplotlib;\
print('torch', torch.__version__, 'cuda', torch.version.cuda);\
assert torch.version.cuda.startswith('12.4'), torch.version.cuda"

# ---------------------------------------------------------------------------
# Bind-mounting over /workspace/v2.0 at run time still works for development.
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY v2.0/     /workspace/v2.0/
COPY data_utils/ /workspace/data_utils/
COPY README.md /workspace/

# The training scripts do `from utils.YParams import YParams`, i.e. they expect
# to run with v2.0 on the import path.
ENV PYTHONPATH=/workspace/v2.0:/workspace \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MPLBACKEND=Agg

# Writable, world-accessible defaults: Apptainer runs as the invoking user, who
# is not root and cannot write to a root-owned HOME.
ENV HOME=/tmp \
    WANDB_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

# The training code writes spectra_out/, gif_out/, acc_plots/ into os.getcwd()
# and resolves the relative `exp_dir: results` against it. /workspace is
# root-owned and read-only for a non-root user (always the case under
# Apptainer), so default the working directory to a writable mount point.
# The run scripts bind the host results directory here.
RUN mkdir -p /runs && chmod 1777 /runs
WORKDIR /runs

CMD ["/bin/bash"]
