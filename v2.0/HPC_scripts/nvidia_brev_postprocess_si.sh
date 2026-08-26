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

config_file=../config/exp16_nvidia_v3.yaml

# Train the output-space postprocessing SI:  noise -> (ERA5 - member_i)
#
# The whole base model (encoders, latent SI, Pangu decoder) is frozen; only the
# small physical-space SI trains. The frozen base comes from
# checkpoint_path_finetuned in the config, currently the rollout finetune
# checkpoint rollout_ckpt_1400.tar (epoch 13, 1400 iters) — it carries the full
# encoder + rollout-finetuned model_det + SI unet state, so the postprocessor
# trains against exactly the model used at inference.
#
# --rollout_steps > 1 unrolls the frozen base model so the SI sees multiple lead
# times. Keep it >1 if the loader supports it: the dispersion deficit is worst
# at 8-16 days (SSR ~0.47 for z500 at day 8 vs ~0.70 at day 45), and a model
# trained only on 24 h errors will not calibrate the long leads.
python -m torch.distributed.launch --master_port=29504 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_postprocess_si.py \
    --yaml_config=$config_file \
    --run_num=pp_si_v2 \
    --epochs=10 \
    --run_iter=1 \
    --rollout_steps=1 \
    
    
    #--pp_ckpt=results/S2S/postprocess_si_pp_si_v1/training_checkpoints/pp_si_ckpt_2000.tar
