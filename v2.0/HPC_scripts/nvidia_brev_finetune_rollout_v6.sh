#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

# MPI and OpenMP settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v2.yaml

# Fine-tune decoder with multi-step rollout CRPS loss.
# Rollout steps increase by 1 every 2000 iterations, up to max_rollout_steps.
# CRPS is averaged uniformly over all lead times at each iteration.
python -m torch.distributed.launch --master_port=29504 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_finetune_rollout.py \
    --yaml_config=$config_file \
    --run_num=finetune_rollout_crps_multipsteps_v3 \
    --epochs=20\
    --run_iter=1 \
    --max_rollout_steps=20 \
    --rollout_step_interval=1000 \
    --initial_rollout_steps=5 \
    --finetune_ckpt=results/S2S/rollout_finetune_finetune_rollout_crps_multipsteps_20260811/training_checkpoints/rollout_ckpt_1800.tar
bing/S2S/v2.0/config