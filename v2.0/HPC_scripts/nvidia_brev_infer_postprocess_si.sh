#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml

#pp_ckpt=results/S2S/postprocess_si_pp_si_v1/training_checkpoints/pp_si_ckpt_2200.tar
pp_ckpt=results/S2S/postprocess_si_pp_si_v2/training_checkpoints/pp_si_ckpt_10800.tar
# Ensemble inference with postprocess-SI spread correction over ALL sel_dates
# (Mondays/Thursdays, May-July, 2019-2024 = 156 init dates, all 00UTC).
#
# Every rank walks the full date list and only the ensemble members are sharded,
# so 40 members on 4 GPUs is 10 members x 156 dates each.
#
# --blend should be the best_blend reported by training validation; 1.0 replaces
# each member outright and tends to overshoot when the base ensemble already
# carries part of the error.
#
# DISK: all 16 variables is ~1.1 GB per member-file, so 156 x 40 would need
# ~7.1 TB (14 TB with the control run) against ~3.8 TB free. --save_vars limits
# output to the three variables the CRPS/SSR benchmark scores, giving
# ~0.23 GB/file => ~1.4 TB per run. Drop the flag only if you have the space.
SAVE_VARS="2m_temperature total_precipitation_24hr geopotential"

python -m torch.distributed.launch --master_port=29505 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=pp_si_infer_v2 \
    --pp_ckpt=$pp_ckpt \
    --num_members=20 \
    --inference_steps=45 \
    --blend=1.0 \
    --feedback=base \
    # --save_vars $SAVE_VARS

# Control run — identical pipeline, correction disabled. Score both with
# benchmark-dev/metrics/CRPS/crps_ssr_main.py to isolate the SI's effect on
# SSR and CRPS.
# python -m torch.distributed.launch --master_port=29506 --nproc_per_node=$NUM_TASKS_PER_NODE \
#     ../inference_postprocess_si.py \
#     --yaml_config=$config_file \
#     --config=S2S \
#     --run_num=pp_si_infer_v1_base \
#     --num_members=40 \
#     --inference_steps=45 \
#     --no_correction \
#     --save_vars $SAVE_VARS

# Quick smoke test — 2019 only, 4 dates, 4 members:
#   --years 2019 --max_dates 4 --num_members 4
