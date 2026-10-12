#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml

#slow_ckpt=results/S2S/slow_surface_si_slow_v1/training_checkpoints/slow_si_ckpt_1600.tar
slow_ckpt=results/S2S/slow_surface_si_slow_v1/training_checkpoints/slow_si_ckpt_4800.tar
# Ensemble inference with slow-surface SI correction over ALL sel_dates
# (Mondays/Thursdays, May-July, 2019-2024 = 156 init dates, all 00UTC).
#
# Only the four slow fields are corrected -- sea_surface_temperature,
# skin_temperature, soil_temperature_level_1, volumetric_soil_water_layer_1.
# Upper air and diagnostics are written straight through from the base model.
#
# Every rank walks the full date list and only the ensemble members are sharded,
# so 20 members on 4 GPUs is 5 members x 156 dates each.
#
# --blend scales the residual; the SI models a member's *total* error, so 1.0
# makes (member + residual) a draw from p(truth | member). --spread_boost
# inflates member-to-member residual deviation, which is the knob for the
# under-dispersion at long leads (SST SSR ~0.40 by day 44) -- draws conditioned
# on similar members correlate strongly and otherwise act as a shared bias
# correction. Tune both against the training run's validation output.
#
# DISK: all variables (16 surface/diagnostic + 5 upper-air x 17 levels) is
# ~1.15 GB per member-file, measured from the pp_si_infer_v2 output. At 157
# init dates x 20 members that is ~3.6 TB, against ~4.1 TB free on /data --
# it fits, but with little headroom, and a --no_correction control run needs
# the same again. Check `df -h /data` before launching.
#
# To write only the four corrected fields instead (~0.23 GB/file, ~0.7 TB per
# run), pass:
#   --save_vars sea_surface_temperature skin_temperature \
#               soil_temperature_level_1 volumetric_soil_water_layer_1
# Omitting --save_vars entirely, as below, writes every variable.

python -m torch.distributed.launch --master_port=29508 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_slow_surface_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=slow_si_infer_v1 \
    --slow_ckpt=$slow_ckpt \
    --num_members=20 \
    --inference_steps=45 \
    --blend=1.0 \
    --spread_boost=0.0 \
    --feedback=base

# Control run — identical pipeline, correction disabled. Score both with
# benchmark-dev/metrics/CRPS/crps_ssr_main.py to isolate the SI's effect on
# SSR, CRPS and the cold SST drift.
# python -m torch.distributed.launch --master_port=29509 --nproc_per_node=$NUM_TASKS_PER_NODE \
#     ../inference_slow_surface_si.py \
#     --yaml_config=$config_file \
#     --config=S2S \
#     --run_num=slow_si_infer_v1_base \
#     --num_members=20 \
#     --inference_steps=45 \
#     --no_correction

# Quick smoke test — 2019 only, 2 dates, 4 members, short rollout:
#   --years 2019 --max_dates 2 --num_members 4 --inference_steps 5
#
# Resume: relaunch with the same --run_num; members whose .nc already exists are
# skipped before any GPU work. Use --overwrite to force regeneration.
