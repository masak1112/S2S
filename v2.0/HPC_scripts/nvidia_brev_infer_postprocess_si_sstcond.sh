#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_TASKS_PER_NODE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml
run_num=pp_si_infer_sstcond_v1

# Ensemble inference with postprocess-SI correction, ALL components from the
# S2S_SSTCOND tree (based on nvidia_brev_infer_postprocess_si.sh):
#
#   latent SI UNet + rollout-finetuned decoder
#     --base_ckpt  results/S2S_SSTCOND/rollout_finetune_finetune_rollout_crps_sstcond/
#                  training_checkpoints/rollout_ckpt_250.tar
#       full 571-entry state: encoder + decoder finetuned with rollout CRPS
#       (iters=250, epoch 2) + the SST-conditioned SI UNet it was finetuned
#       with (= si_c1_sstcond/last_ckpt.tar, gate_in 0.029). It replaces
#       checkpoint_path_finetuned (which in the yaml still points at the old
#       production rollout_ckpt_1000).
#
#   postprocess SI (output space)
#     --pp_ckpt    results/S2S_SSTCOND/postprocess_si_pp_si_sstcond_v1/
#                  training_checkpoints/pp_si_ckpt_12200.tar
#       96 fields, max_lead_steps 10; resumed from pp_si_v2 (iters 9279) and
#       trained on under --config=S2S_SSTCOND.
#
#   --config=S2S_SSTCOND  -> vae_cond_mode: mean, the setting the SSTCOND UNet
#       was trained with (the VAE and Pangu encoder checkpoints are the shared
#       ones under /data/bing/S2S/checkpoints, as in every run).
#
# Output: results/S2S_SSTCOND/pp_si_infer_sstcond_v1/predictions/
#   pangu_plasim_pp_si_infer_sstcond_v1_24h_45step_<init>_ens_<m>.nc
#
# 8 GPUs: members are sharded round-robin over ranks, so ranks 0-3 run 3
# members each and ranks 4-7 run 2 (every rank walks all init dates).
# Members/blend/feedback as in the main run: 20 members, 45 steps, blend 1.0,
# feedback base. Init dates come from the config's sel_dates list (currently
# 206 dates, 2019-01-03 .. 2024-04-29, Jan-Apr), NOT the 156 May-Jul dates of
# the original run - check the "Running N init dates" log line.
#
# Existing member files are skipped, so relaunching resumes.
#
# DISK: ~1.1 GB per member-file for all 16 variables (~3.4 TB for the full
# run). --save_vars limits output to the three CRPS/SSR benchmark variables
# (~0.23 GB/file). Check df -h /data first.
SAVE_VARS="2m_temperature total_precipitation_24hr geopotential"

base_ckpt=results/S2S_SSTCOND/rollout_finetune_finetune_rollout_crps_sstcond/training_checkpoints/rollout_ckpt_250.tar
pp_ckpt=results/S2S_SSTCOND/postprocess_si_pp_si_sstcond_v1/training_checkpoints/pp_si_ckpt_12200.tar

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.launch --master_port=29517 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S_SSTCOND \
    --run_num=$run_num \
    --base_ckpt=$base_ckpt \
    --pp_ckpt=$pp_ckpt \
    --num_members=20 \
    --inference_steps=45 \
    --blend=1.0 \
    --feedback=base \
    --base_latent=si \
    2>&1 | tee $log
    # --save_vars $SAVE_VARS
    # --latent_cond=off        # SST-conditioning ablation of THIS model (same weights, SST encoding zeroed)

# Quick smoke test — 2019 only, 4 dates, 4 members:
#   --years 2019 --max_dates 4 --num_members 4
