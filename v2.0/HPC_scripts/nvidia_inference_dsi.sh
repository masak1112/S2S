#!/bin/bash -l

#SBATCH --job-name=monsoon_storage_inference  
#SBATCH --output=dsi_inference_nvme%x_%j.out
#SBATCH --error=dsi_inference_nvme%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=Monsoon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --gres=gpu:h200:4,local:disk:100G
#SBATCH --mail-user=mlotfollahi@nvidia.com
##$SBATCH --gres=gpu:a40:4 #for development
#SBATCH --mem=1000G 
echo "SLRUM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLRUM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLRUM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLRUM_NTASKS: $SLURM_NTASKS"

# Enable GPU support for MPI
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited
export WANDB_MODE=offline

# NCCL optimizations for H100
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5  # GPU Direct RDMA
export NCCL_P2P_LEVEL=5  # Enable P2P
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback

# PyTorch NCCL settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF #DETAIL

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

user="${USER}"
id="${SLURM_JOB_ID}"
export NVME_DIR="/local/scratch/${user}_${id}"
mkdir -p "${NVME_DIR}"

echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES} NUM_GPUS= ${NUM_GPUS} JOB_ID= ${SLURM_JOB_ID}"
echo "NVME_DIR= ${NVME_DIR}"

# Configuration
CONFIG_FILE=../config/exp2.yaml
CONFIG_NAME=S2S
RUN_NUM=01_nsys_dsi


# NGC credentials — set before pulling (requires NGC API key)
# export APPTAINER_DOCKER_USERNAME='$oauthtoken'
# export APPTAINER_DOCKER_PASSWORD='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'

echo "--- STARTING NSYS PROFILING RUN IN NGC CONTAINER (nvcr.io/nvidia/pytorch:26.01-py3) ---"
# Verify affinity before profiling

# apptainer exec --nv pytorch_25.10.sif which nsys
apptainer exec \
    --nv \
    --bind /net/monsoon,/local/scratch \
    /net/monsoon/bing/pytorch_25.10.sif \
    bash -c "
        pip install ruamel.yaml ruamel.base wandb xarray cartopy h5py h5netcdf timm cftime dask seaborn cdsapi cf_xarray onnx -q --user &&
        PYTHONPATH=mahsa/Pangu/S2S/v2.0 \
        NCCL_IB_DISABLE=1 \
        nsys profile \
            -w true \
            -t cuda,nvtx,cudnn \
            --sample=none \
            --cpuctxsw=none \
            -o /net/monsoon/mahsa/Pangu/S2S/v2.0/HPC_scripts/nsys_report_inference_nvme_%q{SLURM_JOB_ID} \
            --force-overwrite=true \
            torchrun \
            --standalone \
            --nproc_per_node=${NUM_GPUS} \
            /net/monsoon/mahsa/Pangu/S2S/v2.0/inference_optimized.py \
            --yaml_config=${CONFIG_FILE} \
            --run_num=${RUN_NUM} \
            --async_save \
            --enable_nvme \
            --nvme_dir=${NVME_DIR}
            " && \

mkdir -p "/net/monsoon/mahsa/Pangu/S2S/v2.0/HPC_scripts/results/${CONFIG_NAME}/${RUN_NUM}"
cp -a "${NVME_DIR}/results/${CONFIG_NAME}/${RUN_NUM}/." "/net/monsoon/mahsa/Pangu/S2S/v2.0/HPC_scripts/results/${CONFIG_NAME}/${RUN_NUM}/"
