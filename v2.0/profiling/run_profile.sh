#!/bin/bash -l
# Nsight Systems profiles of the UC-S2S v2 training and inference scripts.
#
#   cd v2.0/profiling && ./run_profile.sh <target>
#   target: diffusion | decoder | rollout | inference | all
#
# Each run captures a short steady-state window (after warm-up), then exits,
# so nothing is checkpointed and no training state is touched:
#   - wandb is disabled (WANDB_MODE=disabled)
#   - outputs (logs, hyperparams, predictions) go under ./results in this dir
#   - reports go to ./reports/<target>_rank<R>.nsys-rep
#
# Overridable: GPUS, NSYS_RANKS, PROF_START, PROF_STOP, ROLLOUT_STEPS,
# NUM_MEMBERS, CONFIG_FILE.
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env  # before set -u: conda hooks use unset vars
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export S2S_NVTX=1
export NSYS_OUTDIR=reports
mkdir -p reports logs

# Current config: data in /data/bing/h5data, checkpoints in
# /data/bing/checkpoints_UC-S2S_V2 (exp16_nvidia_v2/v3.yaml point at /data/ERA5,
# which does not exist on this machine).
CONFIG_FILE=${CONFIG_FILE:-../config/exp16_nvidia_v3_full.yaml}
PP_CKPT=/data/bing/checkpoints_UC-S2S_V2/loaded/4_postprocess_si/pp_si_ckpt_10800.tar

launch() {  # launch <name> <ngpu> <port> <script> [args...]
  local name=$1 ngpu=$2 port=$3; shift 3
  export NSYS_NAME=$name
  local log=logs/${name}_$(date +%Y%m%d_%H%M%S).log
  echo "[$name] GPUs=$CUDA_VISIBLE_DEVICES window=[$S2S_PROF_START,$S2S_PROF_STOP) log=$log"
  torchrun --no-python --master_port=$port --nproc_per_node=$ngpu \
    ./nsys_rank.sh python "$@" 2>&1 | tee "$log"
}

run_diffusion() {
  # As in nvidia_brev_train_si_sstcond.sh: 4 GPUs, S2S_SSTCOND (sync_grads on).
  export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
  export S2S_PROF_START=${PROF_START:-10} S2S_PROF_STOP=${PROF_STOP:-30}
  launch diffusion 4 29611 ../train_diffusion.py \
    --yaml_config=$CONFIG_FILE --config=S2S_SSTCOND --run_num=nsys_profile
}

run_decoder() {
  # As in nvidia_brev_finetune_decoder.sh: 4 GPUs, CRPS with 2 SI members.
  export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
  export S2S_PROF_START=${PROF_START:-5} S2S_PROF_STOP=${PROF_STOP:-15}
  launch decoder 4 29612 ../train_finetune_decoder.py \
    --yaml_config=$CONFIG_FILE --config=S2S --run_num=nsys_profile --epochs=1
}

run_rollout() {
  # As in nvidia_brev_finetune_rollout_v6.sh at its starting horizon (5 steps).
  export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
  export S2S_PROF_START=${PROF_START:-3} S2S_PROF_STOP=${PROF_STOP:-8}
  launch rollout 4 29613 ../train_finetune_rollout.py \
    --yaml_config=$CONFIG_FILE --config=S2S --run_num=nsys_profile --epochs=1 \
    --max_rollout_steps=20 --rollout_step_interval=1000 \
    --initial_rollout_steps=${ROLLOUT_STEPS:-5}
}

run_inference() {
  # As in nvidia_brev_infer_full_rerun.sh (1 GPU). The window is one init date
  # (date 0 is warm-up): its sample load + NUM_MEMBERS 45-step members + saves.
  export CUDA_VISIBLE_DEVICES=${GPUS:-0}
  export S2S_PROF_START=${PROF_START:-1} S2S_PROF_STOP=${PROF_STOP:-2}
  launch inference 1 29614 ../inference_postprocess_si.py \
    --yaml_config=$CONFIG_FILE --config=S2S --run_num=nsys_profile_infer \
    --pp_ckpt=$PP_CKPT --blend=1.0 --feedback=base --base_latent=si \
    --mc_dropout=off --num_members=${NUM_MEMBERS:-2} --inference_steps=45 \
    --years 2019 --max_dates 3 --overwrite
}

case "${1:-}" in
  diffusion) run_diffusion ;;
  decoder)   run_decoder ;;
  rollout)   run_rollout ;;
  inference) run_inference ;;
  all)       run_diffusion; run_decoder; run_rollout; run_inference ;;
  *) echo "usage: $0 diffusion|decoder|rollout|inference|all"; exit 1 ;;
esac
