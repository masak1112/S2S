#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

# Latent SI + CRPS rollout finetune ensemble inference.
#   ON : latent-space SI + CRPS-finetuned Pangu decoder, both from
#        checkpoint_path_finetuned = rollout_ckpt_1000.tar (full state, loaded last)
#   (its latent SI UNet is identical to si_c1_full_state; only the decoder differs)
#   OFF: postprocess SI (--no_correction)
#   OFF: MC dropout (--mc_dropout off), so all ensemble spread comes from the latent SI
#
# Init dates: same 50 as the det_backbone test run (2019-01-03 .. 2020-02-24),
# the first 50 sel_dates; ERA5 ground truth in /data/bing/gt_era5 covers 2019-2020.
#
# Members are sharded over GPUs; every GPU walks all dates. Raising NUM_MEMBERS
# later and relaunching only computes the new members (existing files are skipped,
# and each (date, member) has its own seed).
#
# Output: results/S2S/latent_si_crps/predictions/
#   pangu_plasim_latent_si_crps_24h_45step_<init>_ens_<m>.nc   (~1.1 GB each, all vars)

export CUDA_VISIBLE_DEVICES=${GPUS:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NUM_MEMBERS=${NUM_MEMBERS:-4}

config_file=../config/exp16_nvidia_v3_latent_si_crps.yaml
run_num=latent_si_crps

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.run --master_port=29543 --nproc_per_node=$NGPU \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=$run_num \
    --base_latent=si \
    --mc_dropout=off \
    --no_correction \
    --num_members=$NUM_MEMBERS \
    --inference_steps=45 \
    --years 2019 2020 --max_dates 50 \
    2>&1 | tee $log
