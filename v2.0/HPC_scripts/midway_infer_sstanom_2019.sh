#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=23:10:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o midway_infer_sstanom_%x_%j.out
#SBATCH -e midway_infer_sstanom_%x_%j.err
#
# Ensemble inference with the postprocess SI on the S2S_SSTANOM models (Midway).
#   base    : rollout_ckpt_650.tar  (rollout-finetuned decoder + stage-1 SI UNet that
#             was trained on the persistent SST-anomaly conditioning)
#   pp SI   : latest pp_si_ckpt_N.tar of results/S2S_SSTANOM/postprocess_si_pp_si_sstanom
#             (override with PP_CKPT=...)
#   scope   : 2019 init dates only (sel_dates Mon/Thu, Jan-Apr -> 34 dates), 4 members,
#             45 daily steps, blend 1.0, uncorrected member fed back (--feedback base).
# 4 GPUs: members are sharded round-robin over ranks, so each rank runs 1 member and
# walks all 34 init dates. The persistent SST-anomaly map is built once per init date
# by the dataloader and held fixed over the 45 steps.
#
# Output (new folder, nothing existing is touched):
#   results/S2S_SSTANOM/pp_si_infer_sstanom_2019/predictions/
#     pangu_plasim_pp_si_infer_sstanom_2019_24h_45step_<init>_ens_<m>.nc
# Existing member files are skipped, so resubmitting resumes an interrupted run.
#
# Smoke test first (1 date, 2 members): sbatch --export=ALL,SMOKE=1 midway_infer_sstanom_2019.sh
# DISK: /project is ~full (2.8 TB free). Full 16-variable files are ~1.1 GB each
# (34 dates x 4 members = ~150 GB); --save_vars keeps the 3 benchmark variables
# (~0.23 GB each, ~31 GB total).

export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited
ml python
conda activate /project/pedramh/bing/env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/pedramh/bing/S2S/v2.0/HPC_scripts || exit 1
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

config_file=../config/exp16_midway_sstanom.yaml
CFG=S2S_SSTANOM
run_num=pp_si_infer_sstanom_2019
SAVE_VARS="2m_temperature total_precipitation_24hr geopotential"

base_ckpt=results/S2S_SSTANOM/rollout_finetune_rollout_sstanom/training_checkpoints/rollout_ckpt_650.tar
pp_dir=results/S2S_SSTANOM/postprocess_si_pp_si_sstanom/training_checkpoints
pp_ckpt=${PP_CKPT:-$(ls -v $pp_dir/pp_si_ckpt_*.tar 2>/dev/null | tail -1)}

[[ -f "$base_ckpt" ]] || { echo "ABORT: base checkpoint missing: $base_ckpt"; exit 1; }
[[ -f "$pp_ckpt"  ]] || { echo "ABORT: postprocess-SI checkpoint missing (looked in $pp_dir)"; exit 1; }
echo "base_ckpt = $base_ckpt"
echo "pp_ckpt   = $pp_ckpt"

extra=""
if [[ -n "$SMOKE" ]]; then
  run_num=${run_num}_smoke
  extra="--max_dates 1 --num_members 2"
fi

mkdir -p logs
log=logs/${run_num}_$(date +%Y%m%d_%H%M).log
echo "logging to $log"

/project/pedramh/bing/env/bin/python -m torch.distributed.launch --master_port=29517 \
    --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../inference_postprocess_si.py \
    --yaml_config=$config_file --config=$CFG \
    --run_num=$run_num \
    --base_ckpt=$base_ckpt --pp_ckpt=$pp_ckpt \
    --years 2019 \
    --num_members 4 \
    --inference_steps 45 \
    --blend 1.0 \
    --feedback base \
    --base_latent si \
    --save_vars $SAVE_VARS \
    $extra \
    2>&1 | tee $log
