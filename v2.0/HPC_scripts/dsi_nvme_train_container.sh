#!/bin/bash -l

#SBATCH --job-name=train_nvme_test_h100
#SBATCH --output=dsi_%x_%j.out
#SBATCH --error=dsi_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4,local:disk:200G
#SBATCH --mail-user=gongbing@uchicago.edu

user="$USER"
echo "$USER"
id="$SLURM_JOB_ID"
echo "$SLURM_JOB_ID"

dir="/local/scratch/${user}_${id}"

data_path="${dir}/2021_dataset.zip"

cp /net/monsoon/S2S/2021_dataset.zip ${data_path}
unzip ${data_path} -d ${dir}
data_dir="${dir}/net/monsoon/S2S/h5data0"
cp /net/monsoon/S2S/h5data0/*nc ${data_dir}

echo "SLRUM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLRUM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLRUM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLRUM_NTASKS: $SLURM_NTASKS"

export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited

# NCCL optimizations for H100
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=5
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# PyTorch settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo nvidia-smi
nvidia-smi

export NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES} NUM_GPUS= ${NUM_GPUS} JOB_ID= ${SLURM_JOB_ID}"

config_file=/net/monsoon/bing/S2S/v2.0/config/exp2_nvidia.yaml
SIF=/net/monsoon/bing/s2s_image.sif

echo "--- STARTING TRAINING IN CONTAINER ${SIF} ---"

apptainer exec \
    --nv \
    --bind /net/monsoon:/net/monsoon \
    --bind ${dir}:${dir} \
    --env WANDB_MODE=offline \
    --env NCCL_DEBUG=INFO \
    --env NCCL_IB_DISABLE=0 \
    --env NCCL_NET_GDR_LEVEL=5 \
    --env NCCL_P2P_LEVEL=5 \
    --env "NCCL_SOCKET_IFNAME=^lo,docker0" \
    --env NCCL_SOCKET_NTHREADS=4 \
    --env NCCL_NSOCKS_PERTHREAD=4 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --env CUDA_LAUNCH_BLOCKING=0 \
    --env TORCH_CUDNN_V8_API_ENABLED=1 \
    ${SIF} \
    torchrun --nproc_per_node=${NUM_GPUS} --standalone \
        /net/monsoon/bing/S2S/v2.0/train.py \
        --yaml_config=${config_file} \
        --local_storage=${data_dir}
