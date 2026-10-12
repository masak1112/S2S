#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml

# Finetune the output-space postprocessing SI using the SSTCOND base model.
#
# The SSTCOND UNet (si_c1_sstcond) is used as the frozen base instead of the
# rollout-finetuned checkpoint.  checkpoint_path_finetuned is intentionally
# absent from the S2S_SSTCOND config section — loading it would overwrite the
# SSTCOND UNet weights with the original si_c1 rollout checkpoint.
#
# We warm-start the postprocessor from the pretrained pp_si_v2 checkpoint so
# that only the conditioning-related adjustments need to be learned.
#
# --run_num=pp_si_sstcond_v1  writes to:
#   results/S2S_SSTCOND/postprocess_si_pp_si_sstcond_v1/
python -m torch.distributed.launch --master_port=29505 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S_SSTCOND \
    --run_num=pp_si_sstcond_v1 \
    --epochs=10 \
    --run_iter=1 \
    --rollout_steps=1 \
    --pp_ckpt=results/S2S/postprocess_si_pp_si_v2/training_checkpoints/pp_si_last.tar
