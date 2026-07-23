#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

# MPI and OpenMP settings
# Restrict to GPUs 0,1,2,3 only
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

# Launch your script using torch.distributed.launch
config_file=../config/exp16_nvidia.yaml

#train command
python -m torch.distributed.launch --master_port=29500 --nproc_per_node=$NUM_TASKS_PER_NODE ../inference.py --yaml_config=$config_file --run_num=si_c1
