#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

# Deterministic-backbone-only inference: the frozen Pangu model from
# checkpoint_path_det, rolled out autoregressively for 45 daily steps.
#   --base_latent deterministic  -> latent-space SI bypassed (Pangu latent passed
#                                   straight to the decoder), MC dropout off
#   --no_correction              -> postprocess SI not loaded or applied
#   config: checkpoint_path_finetuned = None, so the rollout-finetuned decoder is
#           NOT loaded and model_det keeps the pure backbone weights.
# The run is fully deterministic, so one member (ens_0) is all there is.
#
# With one member only rank 0 has work (members are sharded, dates are not),
# so run on a single GPU.
#
# Init dates: config sel_dates (Mon/Thu, Jan-Apr). The data in /data/bing/h5data
# ends at 2023, so 2024 is excluded.
#
# Output: results/S2S/det_backbone/predictions/
#   pangu_plasim_det_backbone_24h_45step_<init>_ens_0.nc   (~1.1 GB each, all vars)
# Existing files are skipped, so relaunching resumes.

export CUDA_VISIBLE_DEVICES=${GPU:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

config_file=../config/exp16_nvidia_v3_det_backbone.yaml
run_num=det_backbone

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.run --master_port=29532 --nproc_per_node=1 \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file \
    --config=S2S \
    --run_num=$run_num \
    --base_latent=deterministic \
    --no_correction \
    --num_members=1 \
    --inference_steps=45 \
    --years 2019 2020 2021 2022 2023 \
    2>&1 | tee $log
