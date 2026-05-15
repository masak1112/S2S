#!/bin/bash -l

#SBATCH --job-name=inference    
#SBATCH --output=dsi_inference_%x_%j.out
#SBATCH --error=dsi_inference_%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=Monsoon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 
#SBATCH --gres=gpu:h200:4
#SBATCH --mail-user=gongbing@uchicago.edu
##$SBATCH --gres=gpu:a40:4 #for development
#SBATCH --mem=1000G 

# Enable GPU support for MPI
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited
export WANDB_MODE=offline

# NCCL optimizations for H100
export NCCL_DEBUG=INFO  # Set to WARN in production
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5  # GPU Direct RDMA
export NCCL_P2P_LEVEL=5  # Enable P2P
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback

# PyTorch NCCL settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Remove in production

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Let each rank use its NUMA-local CPUs
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# Load apptainer
# module load apptainer

echo nvidia-smi
nvidia-smi

# Get GPU count from host before entering container
export NUM_GPUS=$(nvidia-smi -L | wc -l)
# export CUDA_VISIBLE_DEVICES=0
# export NUM_GPUS=1

echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES} NUM_GPUS= ${NUM_GPUS} JOB_ID= ${SLURM_JOB_ID}"

# Configuration
CONFIG_FILE=../config/exp2.yaml


# NGC credentials — set before pulling (requires NGC API key)
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD='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'

echo "--- STARTING NSYS PROFILING RUN IN NGC CONTAINER (nvcr.io/nvidia/pytorch:26.01-py3) ---"
# Verify affinity before profiling

# apptainer exec --nv pytorch_25.10.sif which nsys
apptainer exec \
    --nv \
    --bind /net/monsoon \
    /net/monsoon/bing/pytorch_25.10.sif \
    bash -c "
        pip uninstall netCDF4 -y -q 2>/dev/null; pip install ruamel.yaml ruamel.base wandb xarray cartopy h5py h5netcdf timm cftime dask seaborn cdsapi cf_xarray onnx -q --user &&
        echo 'nvidia-smi topo -m' &&
        nvidia-smi topo -m &&
        echo 'nvidia-smi topo -p2p n' &&
        nvidia-smi topo -p2p n &&
        echo 'nvidia-smi topo -p2p r' &&
        nvidia-smi topo -p2p r &&
        echo 'nvidia-smi topo -p2p w' &&
        nvidia-smi topo -p2p w &&
        echo 'nvidia-smi nvlink --status' &&
        nvidia-smi nvlink --status &&
        echo 'nvidia-smi nvlink --errorcounters' &&
        nvidia-smi nvlink --errorcounters && 
        echo 'nvidia-smi -q -d NVLINK (fallback if unsupported)' &&
        (nvidia-smi -q -d NVLINK || nvidia-smi -q | sed -n '/NVLINK/,+120p' || true) &&
        echo 'NCCL all-reduce bandwidth' &&
        all_reduce_perf -b 8M -e 8G -f 2 -g 4 &&
        PYTHONPATH=bing/Pangu_test/S2S/v2.0 \
        NCCL_IB_DISABLE=1 \
        nsys profile \
            -w true \
            -t cuda,nvtx,cudnn \
            -o /net/monsoon/bing/Pangu_test/S2S/v2.0/HPC_scripts/nsys_report_inference_%q{SLURM_JOB_ID} \
            --force-overwrite=true \
            torchrun \
            --standalone \
            --nproc_per_node=${NUM_GPUS} \
            ../inference_optimized.py \
            --yaml_config=${CONFIG_FILE} \
            --run_num=01_nsys_dsi \
            --disable_save \
            --async_save \
"


