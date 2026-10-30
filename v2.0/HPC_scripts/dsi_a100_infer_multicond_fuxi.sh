#!/bin/bash -l

#SBATCH --job-name=mc_fuxi
#SBATCH --output=mc_fuxi_%x_%j.out
#SBATCH --error=mc_fuxi_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-user=gongbing@uchicago.edu
#SBATCH --mem=500G

# DSI port of midway_infer_multicond_fuxi.sh: ensemble inference with the 8-anomaly multicond model (S2S_MULTICOND)
# using the checkpoints in checkpoints_UC-S2S_v4_multicondi. FuXi-S2S init dates (104/yr), 45 daily steps, blend 1.0.
# Usage: sbatch dsi_a100_infer_multicond_fuxi.sh <smoke|Y2019|Y2020|Y2021|all>  (all FuXi dates, 20 members)
#   env: SST_MODE (evolving|persistent), BASE_CKPT, PP_CKPT
# All 16 variables are saved: ~1.1 GB per file, 20 members x 104 dates ~ 2.3 TB per year.
# Output: results/S2S_MULTICOND/pp_si_infer_multicond12800_<year>/predictions/ (existing member files are skipped, so resubmitting resumes).
# pp SI: pp_si_ckpt_12800 (stage-3 postprocess trained for multicond). Output run names: pp_si_infer_multicond12800_<year>.
MODE=${1:?mode: smoke|Y2019|Y2020|Y2021|all}

conda activate /net/spaces/monsoon/bing/myenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=4
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

cd /net/spaces/monsoon/bing/S2S/v2.0/HPC_scripts || exit 1
CKPT=/net/spaces/monsoon/bing/checkpoints_UC-S2S_v4_multicondi/loaded
BASE_CKPT=${BASE_CKPT:-$CKPT/3_base_full_state_rollout_finetuned_multicond/rollout_ckpt_1050.tar}
PP_CKPT=${PP_CKPT:-$CKPT/4_postprocess_si/pp_si_ckpt_12800.tar}
[[ -f "$BASE_CKPT" ]] || { echo "ABORT: base checkpoint missing: $BASE_CKPT"; exit 1; }
[[ -f "$PP_CKPT"   ]] || { echo "ABORT: pp checkpoint missing: $PP_CKPT"; exit 1; }
echo "base_ckpt=$BASE_CKPT"; echo "pp_ckpt=$PP_CKPT"

case $MODE in
  smoke)   YEARS=2019; RUN=pp_si_infer_multicond12800_2019_smoke; EXTRA="--max_dates 1 --num_members 2" ;;
  Y20[0-9][0-9]) YEARS=${MODE#Y}; RUN=pp_si_infer_multicond12800_$YEARS; EXTRA="--num_members 20" ;;
  all)     YEARS="2019 2020 2021"; RUN=pp_si_infer_multicond12800_2019; EXTRA="--num_members 20" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

mkdir -p logs
log=logs/${RUN}_$(date +%Y%m%d_%H%M).log

python -m torch.distributed.launch --master_port=$((29540 + ${YEARS%% *} - 2019)) --nproc_per_node=$NUM_GPUS \
    ../inference_postprocess_si.py \
    --yaml_config=../config/exp16_dsi_multicond.yaml --config=S2S_MULTICOND \
    --run_num=$RUN \
    --base_ckpt=$BASE_CKPT --pp_ckpt=$PP_CKPT \
    --init_date_set fuxi_s2s \
    --years $YEARS \
    --inference_steps 45 --blend 1.0 \
    --feedback base --base_latent si \
    --sst_cond_mode ${SST_MODE:-evolving} \
    $EXTRA \
    2>&1 | tee $log
