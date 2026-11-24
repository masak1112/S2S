#!/bin/bash -l
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /home/nvidia/miniforge3/envs/s2s_env

echo nvidia-smi

# Nsight Systems profiles of the training and inference scripts.
#
#   ./nvidia_brev_nsys_profile.sh diffusion|decoder|rollout|inference|inference4|all
#
# inference: 1 GPU, as nvidia_brev_infer_full_rerun.sh. inference4: 4 GPUs,
# one member per GPU, as nvidia_brev_infer_postprocess_si.sh; shows contention
# for the shared disk/CPU when all ranks load and save at the same time.
#
# Each training report holds training and validation together: --val_every
# is set so one validation pass falls inside the capture window.
#   diffusion: validation_diffusion() (SI loss, 30 batches) at iteration 20
#   decoder:   checkpoint save + validate_crps_ssr() (4 ICs x 4 members x
#              10 days) at iteration 8
#   rollout:   validate_crps_ssr() at iteration 5
#
# Each run captures a short steady-state window after warm-up (rank 0 under
# nsys; NSYS_RANKS=0,1,2,3 for all ranks), prints per-step wall time, and exits: nothing is checkpointed, wandb
# is disabled.
# Reports: ../profiling/reports/<target>_<tag>_rank0.nsys-rep
# Summary: python ../profiling/analyze.py <report>
#
# S2S_FAST_KERNELS=1 (default) uses the fast GroupNorm / interpolation in
# networks/diffusion.py, tag "fast"; S2S_FAST_KERNELS=0 profiles the stock
# PyTorch ops, tag "stock".

# MPI and OpenMP settings
# Restrict to GPUs 4,5,6,7 only (0-3 are used by other jobs)
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_TASKS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

# Profiling settings (utils/profiling.py)
export WANDB_MODE=disabled
export S2S_NVTX=1
export S2S_FAST_KERNELS=${S2S_FAST_KERNELS:-1}
if [ "$S2S_FAST_KERNELS" == "1" ]; then tag=fast; else tag=stock; fi
export NSYS_OUTDIR=$(realpath ../profiling)/reports
mkdir -p logs $NSYS_OUTDIR
# Every rank goes through nsys_rank.sh; the ranks in NSYS_RANKS run under nsys
# (one report per GPU: <target>_<tag>_rank<R>.nsys-rep).
export NSYS_RANKS=${NSYS_RANKS:-0}
# --use-env: ranks read LOCAL_RANK from the environment (required with --no-python)
nsys_launch="--use-env --no-python ../profiling/nsys_rank.sh python"

# Launch your script using torch.distributed.launch
# (exp16_nvidia_v3_full.yaml: data in /data/bing/h5data, checkpoints in
# /data/bing/checkpoints_UC-S2S_V2)
config_file=../config/exp16_nvidia_v3_full.yaml
pp_ckpt=/data/bing/checkpoints_UC-S2S_V2/loaded/4_postprocess_si/pp_si_ckpt_10800.tar

profile_diffusion() {
    # as nvidia_brev_train_si_sstcond.sh; capture iterations 10-30, with
    # validation_diffusion() at iteration 20
    export NSYS_NAME=diffusion_${tag} S2S_PROF_START=10 S2S_PROF_STOP=30
    python -m torch.distributed.launch --master_port=29601 --nproc_per_node=$NUM_TASKS_PER_NODE $nsys_launch \
        ../train_diffusion.py --yaml_config=$config_file --config=S2S_SSTCOND --run_num=nsys_profile_${tag} \
        --val_every=20 \
        2>&1 | tee logs/nsys_profile_diffusion_${tag}_$(date +%Y%m%d_%H%M).log
}

profile_decoder() {
    # as nvidia_brev_finetune_decoder.sh; capture iterations 5-15, which with
    # --val_every=8 include the checkpoint save + validate_crps_ssr() at iteration 8
    export NSYS_NAME=decoder_${tag} S2S_PROF_START=5 S2S_PROF_STOP=15
    python -m torch.distributed.launch --master_port=29602 --nproc_per_node=$NUM_TASKS_PER_NODE $nsys_launch \
        ../train_finetune_decoder.py --yaml_config=$config_file --run_num=nsys_profile_${tag} --epochs=1 \
        --val_every=8 \
        2>&1 | tee logs/nsys_profile_decoder_${tag}_$(date +%Y%m%d_%H%M).log
    rm -f results/S2S/finetune_nsys_profile_${tag}/training_checkpoints/*.tar
}

profile_rollout() {
    # as nvidia_brev_finetune_rollout_v6.sh at its starting horizon (5 steps);
    # capture iterations 3-8, with validate_crps_ssr() at iteration 5
    export NSYS_NAME=rollout_${tag} S2S_PROF_START=3 S2S_PROF_STOP=8
    python -m torch.distributed.launch --master_port=29603 --nproc_per_node=$NUM_TASKS_PER_NODE $nsys_launch \
        ../train_finetune_rollout.py --yaml_config=$config_file --run_num=nsys_profile_${tag} --epochs=1 \
        --max_rollout_steps=20 --rollout_step_interval=1000 --initial_rollout_steps=5 \
        --val_every=5 \
        2>&1 | tee logs/nsys_profile_rollout_${tag}_$(date +%Y%m%d_%H%M).log
}

profile_inference() {
    # as nvidia_brev_infer_full_rerun.sh on 1 GPU (GPU 4); 3 init dates x 1 member,
    # capture init date 1 (date 0 is warm-up); predictions are deleted afterwards
    export NSYS_NAME=inference_${tag} S2S_PROF_START=1 S2S_PROF_STOP=2
    CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port=29604 --nproc_per_node=1 $nsys_launch \
        ../inference_postprocess_si.py --yaml_config=$config_file --config=S2S --run_num=nsys_profile_infer_${tag} \
        --pp_ckpt=$pp_ckpt --blend=1.0 --feedback=base --base_latent=si --mc_dropout=off \
        --num_members=1 --inference_steps=45 --years 2019 --max_dates 3 --overwrite \
        2>&1 | tee logs/nsys_profile_inference_${tag}_$(date +%Y%m%d_%H%M).log
    rm -f results/S2S/nsys_profile_infer_${tag}/predictions/*.nc
}

profile_inference4() {
    # as profile_inference on GPUs 4-7: 4 members, one per rank, so all ranks
    # load each init date and write their NetCDF at the same time
    export NSYS_NAME=inference4gpu_${tag} S2S_PROF_START=1 S2S_PROF_STOP=2
    python -m torch.distributed.launch --master_port=29606 --nproc_per_node=$NUM_TASKS_PER_NODE $nsys_launch \
        ../inference_postprocess_si.py --yaml_config=$config_file --config=S2S --run_num=nsys_profile_infer4_${tag} \
        --pp_ckpt=$pp_ckpt --blend=1.0 --feedback=base --base_latent=si --mc_dropout=off \
        --num_members=$NUM_TASKS_PER_NODE --inference_steps=45 --years 2019 --max_dates 3 --overwrite \
        2>&1 | tee logs/nsys_profile_inference4_${tag}_$(date +%Y%m%d_%H%M).log
    rm -f results/S2S/nsys_profile_infer4_${tag}/predictions/*.nc
}

case "${1:-all}" in
    diffusion) profile_diffusion ;;
    decoder|decoder_val) profile_decoder ;;
    rollout)   profile_rollout ;;
    inference) profile_inference ;;
    inference4) profile_inference4 ;;
    all)       profile_diffusion; profile_decoder; profile_rollout; profile_inference ;;
    *) echo "usage: $0 diffusion|decoder|rollout|inference|inference4|all"; exit 1 ;;
esac

ls -la $NSYS_OUTDIR/*_${tag}_rank*.nsys-rep
