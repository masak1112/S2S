#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

# MPI and OpenMP settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml
#finetune_ckpt=/data/bing/S2S/v2.0/HPC_scripts/results/S2S/finetune_finetune_decoder_crps/training_checkpoints/finetune_ckpt_3200.tar
#finetune_ckpt=/data/bing/S2S/v2.0/HPC_scripts/results/S2S/rollout_finetune_finetune_rollout_crps/training_checkpoints/rollout_ckpt_1000.tar
finetune_ckpt=/data/bing/S2S/v2.0/HPC_scripts/results/S2S/rollout_finetune_finetune_rollout_crps_multipsteps_v3/training_checkpoints/rollout_ckpt_2000.tar
# inference with CRPS fine-tuned decoder
python -m torch.distributed.launch --master_port=29503 --nproc_per_node=$NUM_TASKS_PER_NODE ../inference.py \
    --yaml_config=$config_file \
    --run_num=finetune_rollout_crps_multipsteps_20260815_2000steps  \
    --finetune_ckpt=$finetune_ckpt
