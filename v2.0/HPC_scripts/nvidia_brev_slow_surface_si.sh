#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml

# Dedicated postprocess SI for the slow surface fields plus precipitation:
#   sea_surface_temperature, skin_temperature,
#   soil_temperature_level_1, volumetric_soil_water_layer_1,
#   total_precipitation_24hr   (diagnostic; corrected but never fed back)
#
# ~2.3M params vs the joint model's 34.3M -- these fields are smooth and few.
# The whole base model is frozen; only this SI trains.
#
# slow_v1 checkpoints were trained on the four slow fields only. The UNet
# width depends on the field count, so they cannot be resumed with
# precipitation in slow_fields -- this is a fresh run. To resume slow_v1
# instead, remove total_precipitation_24hr from slow_fields in the config
# and restore --slow_ckpt.
#
# --rollout_steps matters more here than for the joint model. Measured against
# ERA5, SST is well calibrated at day 1 (SSR ~1.0) and badly under-dispersed by
# day 44 (~0.40), with a cold drift reaching -1.4 K. Training only on 24 h
# errors teaches the regime that is not broken, so unroll as far as memory and
# the multi-step loader allow.
python -m torch.distributed.launch --master_port=29507 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_slow_surface_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=slow_v2_precip \
    --epochs=10 \
    --run_iter=1 \
    --rollout_steps=10
