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

# ABLATION: postprocess SI ONLY.
#
# --base_latent deterministic bypasses the latent-space SI inside the base
# model (the Pangu latent is passed to the decoder unperturbed) and turns MC
# dropout off. Every member therefore shares one deterministic base
# trajectory and ALL ensemble spread comes from the postprocess-SI draws.
#
# Because that base trajectory is identical for every member (--feedback base),
# inference_postprocess_si.py rolls it out once per init date and reuses it,
# so this run costs roughly one member's worth of base-model work per date.
# The per-member seed still governs the postprocess-SI draws.
#
# Score together with the other two runs to separate the contributions:
#   pp_si_infer_v2         latent SI + postprocess SI   (nvidia_brev_infer_postprocess_si.sh)
#   pp_si_infer_v1_base    latent SI only               (--no_correction control)
#   pp_si_infer_v2_pponly  postprocess SI only          (this script)
#
# Caveat: the postprocess SI was trained on residuals of latent-SI-perturbed
# members, so applying it to a deterministic base is a mild distribution shift.
#
# DISK: same as the main run. --save_vars limits output to the three variables
# the CRPS/SSR benchmark scores (~0.23 GB/file). Drop the flag only if you have
# the space.
SAVE_VARS="2m_temperature total_precipitation_24hr geopotential"

python -m torch.distributed.launch --master_port=29507 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=pp_si_infer_v2_pponly \
    --pp_ckpt=$pp_ckpt \
    --num_members=20 \
    --inference_steps=45 \
    --blend=1.0 \
    --feedback=base \
    --base_latent=deterministic
    # --mc_dropout=on          # keep dropout spread in the base while still skipping the latent SI
    # --save_vars $SAVE_VARS

# Quick smoke test — 2019 only, 4 dates, 4 members:
#   --years 2019 --max_dates 4 --num_members 4
