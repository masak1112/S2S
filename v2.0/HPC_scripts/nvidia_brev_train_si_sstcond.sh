#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_nvidia_v3.yaml
run_num=si_c1_sstcond

# RETRAIN the latent-space SI with a working SST conditioning branch.
#
# Config block S2S_SSTCOND (see config/exp16_nvidia_v3.yaml) =
#   S2S + warm start from si_c1_full_state/last_ckpt.tar (UNet weights only)
#       + conditioning branch re-initialised, gate_init 0.1
#       + z = VAE posterior MEAN of SST (not a noisy sample)
#       + conditioning branch: weight_decay 0, lr x10
#       + gradients averaged across ranks (sync_grads)
#       + cond diagnostics every 200 iters
#
# Output: results/S2S_SSTCOND/si_c1_sstcond/
#   training_checkpoints/diff_ckpt_<iters>.tar every 2000 iters, last_ckpt.tar
#   out.log  — grep "cond diag" for gate / rel_effect. In the old runs
#              rel_effect was 0; if it stays < ~1e-3 after 2-4k iters the
#              branch is collapsing again and the run should be stopped.
#
# Cost reference: si_c1_full_state ran ~890 iters/h on 4 GPUs (~23 h for 20k),
# but that run did not sync gradients, so per-iter cost here is similar and
# each iter is worth 4x the samples.
#
# Resume: relaunching this script resumes from last_ckpt.tar automatically.
# Fresh restart: delete results/S2S_SSTCOND/si_c1_sstcond/training_checkpoints/last_ckpt.tar
#                (the warm start from si_c1_full_state then applies again).
#
# AFTER TRAINING — using the new SI downstream:
#   inference_postprocess_si.py loads checkpoint_path_finetuned LAST with
#   strict=False, which would overwrite the new UNet with the old collapsed one.
#   The decoder finetune (rollout_ckpt_1000) only touched
#   model_det.{upsample,layer4,patchrecovery2d,patchrecovery3d}, so the new UNet
#   is compatible with it; the loader just needs a "unet from ckpt A, decoder
#   from ckpt B" option (or a merged checkpoint). Not done yet.
#   The postprocess SI (pp_si_v2) was trained on members of the OLD latent SI
#   and should be re-validated / retrained against the new one.

mkdir -p logs
log=logs/train_si_sstcond_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.launch --master_port=29509 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_diffusion.py \
    --yaml_config=$config_file \
    --config=S2S_SSTCOND \
    --run_num=$run_num \
    2>&1 | tee $log

# Smoke test (few batches per year, exits quickly): add to the yaml block
#   mode: 'test'
#   test_iterations: 20
