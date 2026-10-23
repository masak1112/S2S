#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_midway_sstanom.yaml
run_num=si_c1_sstanom

# RETRAIN the latent-space SI conditioned on the persistent SST anomaly (no VAE).
# Config block S2S_SSTANOM = S2S_SSTCOND + sst_cond_anom (see config yaml).
# Warm start from si_c1_full_state (UNet only, cond branch re-initialised).
# grep "cond diag" in the log: rel_effect must become clearly non-zero.
# Output: results/S2S_SSTANOM/si_c1_sstanom/
# Downstream (rollout finetune, postprocess SI, inference) must use
# --config=S2S_SSTANOM; the base UNet changed, so the rollout finetune and
# postprocess SI need retraining against it.

mkdir -p logs
log=logs/train_si_sstanom_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

python -m torch.distributed.launch --master_port=29510 --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train_diffusion.py \
    --yaml_config=$config_file \
    --config=S2S_SSTANOM \
    --run_num=$run_num \
    2>&1 | tee $log

# Smoke test (few batches per year, exits quickly): add to the yaml block
#   mode: 'test'
#   test_iterations: 20
