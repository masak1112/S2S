#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

# Full-pipeline ensemble inference: every checkpoint on.
#   VAE c1 (SST conditioning)      checkpoint_path_vae_c1
#   Pangu backbone                 checkpoint_path_det
#   latent SI + CRPS-finetuned decoder
#                                  checkpoint_path_finetuned = rollout_ckpt_1000.tar
#   postprocess SI (output space)  --pp_ckpt pp_si_ckpt_10800.tar (96 fields)
# Postprocess settings as in the reference run pp_si_infer_v2: blend 1.0,
# feedback base (correction is applied to the saved output, the uncorrected
# state continues the rollout).
# MC dropout ON (dropout + DropPath active in the Pangu backbone at inference), as in
# the reference run pp_si_infer_v2 and the upstream default; members then differ
# through both the latent-SI noise and dropout. (The latent_si / latent_si_crps
# runs have it off.)
#
# Init dates: same 50 as the other runs (2019-01-03 .. 2020-02-24).
# Members are sharded over GPUs; relaunching skips finished member files.
#
# Output: results/S2S/full_pp_si/predictions/
#   pangu_plasim_full_pp_si_24h_45step_<init>_ens_<m>.nc   (~1.1 GB each, all vars)

export CUDA_VISIBLE_DEVICES=${GPUS:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NUM_MEMBERS=${NUM_MEMBERS:-4}

config_file=../config/exp16_nvidia_v3_full.yaml
run_num=full_pp_si
pp_ckpt=/data/bing/checkpoints_UC-S2S_V2/loaded/4_postprocess_si/pp_si_ckpt_10800.tar

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.run --master_port=29544 --nproc_per_node=$NGPU \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=$run_num \
    --pp_ckpt=$pp_ckpt \
    --blend=1.0 \
    --feedback=base \
    --base_latent=si \
    --mc_dropout=on \
    --num_members=$NUM_MEMBERS \
    --inference_steps=45 \
    --years 2019 2020 --max_dates 50 \
    2>&1 | tee $log
