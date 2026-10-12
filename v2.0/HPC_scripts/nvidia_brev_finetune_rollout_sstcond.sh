#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

# MPI and OpenMP settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

# Same yaml as the SST-conditioned SI retrain. --config=S2S_SSTCOND makes
# train_finetune_rollout.py take its SI UNet from that block's
#   checkpoint_path_diff = results/S2S_SSTCOND/si_c1_sstcond/training_checkpoints/last_ckpt.tar
# (= diff_ckpt_6001, saved 2026-09-24 03:04) and build the model with
# vae_cond_mode: mean, matching how that UNet was trained.
config_file=../config/exp16_nvidia_v3.yaml
run_num=finetune_rollout_crps_sstcond

# Fine-tune decoder with multi-step rollout CRPS loss, based on
# nvidia_brev_finetune_rollout_v6.sh. Rollout steps increase by 1 every
# rollout_step_interval iterations, from initial_rollout_steps up to
# max_rollout_steps; CRPS is averaged uniformly over all lead times.
#
# What is trained / frozen:
#   trained : model_det.{upsample, layer4, patchrecovery2d, patchrecovery3d}
#   frozen  : VAE encoder, Pangu encoder, SI UNet (from the SSTCOND checkpoint)
#
# RESUME (2026-09-25): --finetune_ckpt now points at this run's own
# rollout_ckpt_200.tar (saved 03:01, iters=200, epoch=1). Its UNet is the
# SSTCOND one, so loading the full state is safe; the optimizer state and the
# iteration counter (and hence the rollout-step schedule) continue from 200.
# For a FRESH start remove that line again.
#
# Original note: do not pass an OLD finetune checkpoint: that flag loads the FULL model state
# with strict=True (UNet included) and would overwrite the SST-conditioned UNet
# with the one inside the old finetune checkpoint. The decoder therefore starts
# from checkpoint_path_det (original Pangu), not from rollout_ckpt_1000.
# To resume THIS run after an interruption, add
#   --finetune_ckpt=results/S2S_SSTCOND/rollout_finetune_finetune_rollout_crps_sstcond/training_checkpoints/rollout_ckpt_<iters>.tar
# (its UNet is the SSTCOND one, so that is safe).
#
# Output: results/S2S_SSTCOND/rollout_finetune_finetune_rollout_crps_sstcond/
#   rollout_finetune.log, training_checkpoints/rollout_ckpt_<iters>.tar
#
# CAVEAT: in the 6001 checkpoint the SST cross-attention gate had decayed to
# 0.029 (SST changes the SI drift by ~0.04%), so this SI behaves almost
# identically to the production one. Use diff_ckpt_2001.tar (gate 0.084) via
# checkpoint_path_diff in the yaml if you want the largest surviving SST effect.

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.launch --master_port=29516 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_finetune_rollout.py \
    --yaml_config=$config_file \
    --config=S2S_SSTCOND \
    --run_num=$run_num \
    --epochs=20 \
    --run_iter=1 \
    --max_rollout_steps=20 \
    --rollout_step_interval=1000 \
    --initial_rollout_steps=5 \
    --finetune_ckpt=results/S2S_SSTCOND/rollout_finetune_finetune_rollout_crps_sstcond/training_checkpoints/rollout_ckpt_200.tar \
    2>&1 | tee $log
