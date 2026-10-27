#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=23:10:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o midway_infer_multicond_fuxi_%x_%j.out
#SBATCH -e midway_infer_multicond_fuxi_%x_%j.err
#
# Ensemble inference with the MULTI-VARIABLE anomaly-conditioned model (config S2S_MULTICOND):
#   base   : stage-2 rollout_ckpt_1050.tar (decoder rollout-finetuned with EVOLVING conditioning
#            + the stage-1 latent-SI UNet conditioned on 8 anomalies); override with BASE_CKPT=...
#   pp SI  : checkpoints_UC-S2S_V3/loaded/4_postprocess_si/pp_si_ckpt_11400.tar. It was trained
#            against the SST-anomaly base, NOT retrained for this model (override with PP_CKPT=...).
#   cond   : --sst_cond_mode evolving (default, matches stage-2 training: SST, soil water, z10/z250/
#            z500, OLR rebuilt from the fed-back forecast; snow depth and sea ice persistent);
#            SST_MODE=persistent holds the initial-condition map fixed instead.
# FuXi-S2S initial dates (--init_date_set fuxi_s2s: 104 per year, 2019-2021) instead of Mon/Thu Jan-Apr.
# First MAX_DATES (default 104 = one year) dates of YEARS (default "2019 2020 2021"), ${NUM_MEMBERS:-20} members, 45 daily steps, blend 1.0.
# One job per year: sbatch --export=ALL,YEARS=2019 ... (2019|2020|2021). 20 members x 104 dates ~ 2.3 TB per year, all variables.
# Output: results/S2S_MULTICOND/pp_si_infer_multicond_2019/predictions/ (shared with the 2019 run; existing member files are skipped, never overwritten).
# Smoke test: sbatch --export=ALL,SMOKE=1 midway_infer_multicond_fuxi.sh
# DISK: all 16 variables, ~1.1 GB per file: 20 members x 104 dates ~ 2.3 TB per year.

export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited
ml python
conda activate /project/pedramh/bing/env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/pedramh/bing/S2S/v2.0/HPC_scripts || exit 1
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_midway_sstanom.yaml
CFG=S2S_MULTICOND
run_num=pp_si_infer_multicond_2019
# SAVE_VARS: none -- all variables are saved (~1.1 GB per member file).

base_ckpt=${BASE_CKPT:-results/S2S_MULTICOND/rollout_finetune_rollout_multicond/training_checkpoints/rollout_ckpt_1050.tar}
pp_dir=/project/pedramh/bing/checkpoints_UC-S2S_V3/loaded/4_postprocess_si
pp_ckpt=${PP_CKPT:-$pp_dir/pp_si_ckpt_11400.tar}

[[ -f "$base_ckpt" ]] || { echo "ABORT: base checkpoint missing: $base_ckpt"; exit 1; }
[[ -f "$pp_ckpt"  ]] || { echo "ABORT: postprocess-SI checkpoint missing (looked in $pp_dir)"; exit 1; }
echo "base_ckpt = $base_ckpt"
echo "pp_ckpt   = $pp_ckpt"

extra="--max_dates ${MAX_DATES:-104}"
if [[ -n "$SMOKE" ]]; then
  run_num=${run_num}_smoke
  extra="--max_dates 1 --num_members 2"
fi

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

/project/pedramh/bing/env/bin/python -m torch.distributed.launch --master_port=$((29540 + ${YEARS%% *} - 2019)) \
    --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file --config=$CFG \
    --run_num=$run_num \
    --base_ckpt=$base_ckpt --pp_ckpt=$pp_ckpt \
    --init_date_set fuxi_s2s \
    --years ${YEARS:-2019 2020 2021} \
    --num_members ${NUM_MEMBERS:-20} \
    --inference_steps 45 \
    --blend 1.0 \
    --feedback base \
    --base_latent si \
    --sst_cond_mode ${SST_MODE:-evolving} \
    $extra \
    2>&1 | tee $log
