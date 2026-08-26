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
config_file=../config/exp16_nvidia_v2.yaml

#finetune decoder with CRPS loss (ensemble=2, frozen SI UNet)
python -m torch.distributed.launch --master_port=29502 --nproc_per_node=$NUM_TASKS_PER_NODE ../train_finetune_decoder.py --yaml_config=$config_file --run_num=finetune_decoder_crps --epochs=10 --run_iter=1 \
    --finetune_ckpt=results/S2S/finetune_finetune_decoder_crps/training_checkpoints/finetune_ckpt_3200.tar
