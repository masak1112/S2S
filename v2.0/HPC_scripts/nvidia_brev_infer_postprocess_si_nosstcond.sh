#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml

pp_ckpt=results/S2S/postprocess_si_pp_si_v2/training_checkpoints/pp_si_ckpt_10800.tar

# ABLATION: latent SI ON, SST conditioning of the latent SI OFF.
#
# --base_latent si keeps the latent-space SI inside the base model: every
# member draws its own noise around the Pangu latent and MC dropout follows
# params.infer_mc_dropout, exactly as in the main run.
#
# --latent_cond off zeroes the conditioning tensor that the latent-SI UNet
# cross-attends to. That tensor is the VAE encoding of the SST surface
# component (params.component = 8 in the 9-channel prognostic surface input:
# 5 surface + 3 land + 1 ocean). The Pangu latent that is concatenated to the
# UNet input (state conditioning) is untouched, so the SI still sees the full
# atmospheric state — it just no longer gets a separate SST signal. The
# postprocess SI is applied on top as usual.
#
# Because the latent SI is sampled per member, the base trajectory differs
# between members and is rolled out once per member (no shared-base shortcut),
# so this run costs the same as nvidia_brev_infer_postprocess_si.sh.
#
# Score together with the other runs to separate the contributions:
#   pp_si_infer_v2            latent SI (SST cond) + postprocess SI   (nvidia_brev_infer_postprocess_si.sh)
#   pp_si_infer_v1_base       latent SI only                          (--no_correction control)
#   pp_si_infer_v2_pponly     postprocess SI only                     (nvidia_brev_infer_postprocess_si_pponly.sh)
#   pp_si_infer_v2_nosstcond  latent SI (no SST cond) + postprocess SI (this script)
#
# NOTE (checked 2026-09-23): in the base checkpoint this run loads
# (checkpoint_path_finetuned = rollout_finetune_finetune_rollout_crps/rollout_ckpt_1000.tar,
# and likewise si_c1_full_state/last_ckpt.tar) the learned gates on both
# cross-attention fuses are ~0 (cross_attn_in.gate = -4.4e-6,
# cross_attn_mid.gate = -1.6e-39, to_out weights ~1e-39). The SST conditioning
# enters the UNet ONLY through x + gate * cross_attn(x, cond), so the trained
# model already ignores it and --latent_cond off should reproduce the main run
# to within float noise. Treat a null result here as confirmation of that, and
# retrain the latent SI without the SST branch if a true no-SST model is wanted.
#
# Caveat: the latent SI was trained with the SST conditioning present, so
# zeroing it at inference is an out-of-distribution input to the UNet's
# cross-attention branch; treat any change in spread as a sensitivity to the
# SST signal, not as what an SST-free SI would have learned.
#
# DISK: same as the main run. --save_vars limits output to the three variables
# the CRPS/SSR benchmark scores (~0.23 GB/file). Drop the flag only if you have
# the space.
SAVE_VARS="2m_temperature total_precipitation_24hr geopotential"

python -m torch.distributed.launch --master_port=29508 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=pp_si_infer_v2_nosstcond \
    --pp_ckpt=$pp_ckpt \
    --num_members=20 \
    --inference_steps=45 \
    --blend=1.0 \
    --feedback=base \
    --base_latent=si \
    --latent_cond=off
    # --mc_dropout=off         # remove dropout spread so the latent SI draws are the only base-model spread
    # --save_vars $SAVE_VARS

# Quick smoke test — 2019 only, 4 dates, 4 members:
#   --years 2019 --max_dates 4 --num_members 4
