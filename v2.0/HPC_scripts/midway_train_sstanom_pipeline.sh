#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=23:10:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o midway_sstanom_%x_%j.out
#SBATCH -e midway_sstanom_%x_%j.err
#
# Persistent-SST-anomaly pipeline (config block S2S_SSTANOM), Midway version of
# midway_training.sh. Three stages, each reads the previous stage's output:
#
#   1  latent SI           train_diffusion.py          -> results/S2S_SSTANOM/si_c1_sstanom/
#                          (warm start from the old SI, conditioning branch re-initialised,
#                           conditioned on the interpolated IC SST anomaly, VAE not used)
#   2  CRPS decoder        train_finetune_rollout.py   -> results/S2S_SSTANOM/rollout_finetune_rollout_sstanom/
#                          (SI UNet from stage 1 frozen; only model_det upsample/layer4/patchrecovery train)
#   3  postprocess SI      train_postprocess_si.py     -> results/S2S_SSTANOM/postprocess_si_pp_si_sstanom/
#                          (frozen base = stage 2 rollout_last.tar; warm start from the old pp_si ckpt)
#
# 23 h is far too short for all three, so submit them as a dependency chain:
#   j1=$(sbatch --parsable --export=ALL,STAGE=1 midway_train_sstanom_pipeline.sh)
#   j2=$(sbatch --parsable --dependency=afterany:$j1 --export=ALL,STAGE=2 midway_train_sstanom_pipeline.sh)
#   sbatch --dependency=afterany:$j2 --export=ALL,STAGE=3 midway_train_sstanom_pipeline.sh
# Every stage resumes automatically (stage 1: last_ckpt.tar) or via the *_last/ckpt vars
# below, so a stage that hits the time limit can just be resubmitted. Run from HPC_scripts/.
# STAGE=all (default) runs 1,2,3 back to back inside one allocation.

export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited
ml python
conda activate /project/pedramh/bing/env
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/pedramh/bing/S2S/v2.0/HPC_scripts || exit 1
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
echo "NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE}"

PY=/project/pedramh/bing/env/bin/python
config_file=../config/exp16_midway_sstanom.yaml
CFG=S2S_SSTANOM
STAGE=${STAGE:-all}
R=results/$CFG

# ---- stage knobs --------------------------------------------------------------
# Stage 2: resume an interrupted rollout finetune by pointing at its last checkpoint
# (full state incl. optimizer + iteration counter; its UNet is the stage-1 one, so safe).
ROLLOUT_RESUME=${ROLLOUT_RESUME:-}     # e.g. $R/rollout_finetune_rollout_sstanom/training_checkpoints/rollout_last.tar
# Stage 3: warm start of the postprocess SI (and resume when re-submitted).
PP_INIT=/project/pedramh/bing/checkpoints_UC-S2S_V2/loaded/4_postprocess_si/pp_si_ckpt_10800.tar
PP_LAST=$R/postprocess_si_pp_si_sstanom/training_checkpoints/pp_si_last.tar
# -------------------------------------------------------------------------------

# ---- never overwrite existing checkpoints ---------------------------------------
# All three stages write ONLY under results/S2S_SSTANOM/ (new run names). Source
# checkpoints (old SI, finetuned base, pp_si_*, checkpoints_UC-S2S_V2) are read-only
# inputs. The yaml's checkpoint_path_diff (stage 1 output dir) must stay in there too.
want=/project/pedramh/bing/S2S/v2.0/HPC_scripts/results/S2S_SSTANOM/
diff_out=$(grep -A40 '^S2S_SSTANOM:' ../config/exp16_midway_sstanom.yaml | grep -m1 '^  checkpoint_path_diff:' | sed 's/.*"\(.*\)".*/\1/')
[[ "$diff_out" == $want* ]] || { echo "ABORT: S2S_SSTANOM checkpoint_path_diff ($diff_out) is outside $want"; exit 1; }

mkdir -p logs
stamp=$(date +%Y%m%d_%H%M)

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

if run_stage 1; then
  echo "=== stage 1: latent SI on persistent SST anomaly ==="
  $PY -m torch.distributed.launch --master_port=29510 --nproc_per_node=$NUM_TASKS_PER_NODE \
      ../train_diffusion.py \
      --yaml_config=$config_file --config=$CFG --run_num=si_c1_sstanom \
      2>&1 | tee logs/sstanom_si_$stamp.log
  # grep "cond diag" in the log: rel_effect must be clearly non-zero. If it stays <1e-3
  # after 2-4k iters the conditioning is not being used; stop and investigate.
  [[ -f $R/si_c1_sstanom/training_checkpoints/last_ckpt.tar ]] || { echo "stage 1 produced no last_ckpt.tar"; exit 1; }
fi

if run_stage 2; then
  echo "=== stage 2: rollout CRPS decoder finetune ==="
  # SI UNet comes from checkpoint_path_diff in the yaml (= stage-1 last_ckpt.tar).
  [[ -f $R/si_c1_sstanom/training_checkpoints/last_ckpt.tar ]] || { echo "run stage 1 first"; exit 1; }
  # A fresh rollout run would overwrite its own rollout_ckpt_N / rollout_last files,
  # so if outputs already exist demand an explicit resume instead of restarting.
  if [[ -z "$ROLLOUT_RESUME" ]] && ls $R/rollout_finetune_rollout_sstanom/training_checkpoints/rollout_*.tar >/dev/null 2>&1; then
    echo "ABORT: stage 2 outputs already exist; set ROLLOUT_RESUME=<rollout_ckpt_N.tar> to continue (or use a new run_num)."; exit 1
  fi
  resume_arg=""
  [[ -n "$ROLLOUT_RESUME" ]] && resume_arg="--finetune_ckpt=$ROLLOUT_RESUME"
  $PY -m torch.distributed.launch --master_port=29516 --nproc_per_node=$NUM_TASKS_PER_NODE \
      ../train_finetune_rollout.py \
      --yaml_config=$config_file --config=$CFG --run_num=rollout_sstanom \
      --epochs=20 --run_iter=1 \
      --max_rollout_steps=5 --rollout_step_interval=500 --initial_rollout_steps=1 \
      $resume_arg \
      2>&1 | tee logs/sstanom_rollout_$stamp.log
  [[ -f $R/rollout_finetune_rollout_sstanom/training_checkpoints/rollout_last.tar ]] || { echo "stage 2 produced no rollout_last.tar"; exit 1; }
fi

if run_stage 3; then
  echo "=== stage 3: postprocess SI finetune ==="
  # Frozen base = checkpoint_path_finetuned in the yaml (= stage-2 rollout_last.tar).
  # Base = checkpoint_path_finetuned in the yaml (a rollout_ckpt_N.tar chosen by hand).
  base_ckpt=$(grep -A60 '^S2S_SSTANOM:' ../config/exp16_midway_sstanom.yaml | grep '^  checkpoint_path_finetuned:' | tail -1 | sed 's/.*"\(.*\)".*/\1/')
  [[ -f "$base_ckpt" ]] || { echo "stage-3 base checkpoint missing: $base_ckpt"; exit 1; }
  echo "stage-3 frozen base = $base_ckpt"
  # Resume this run if it already has a checkpoint, otherwise warm start from the old pp SI.
  if [[ -f $PP_LAST ]]; then pp_ckpt=$PP_LAST; else pp_ckpt=$PP_INIT; fi
  echo "pp_ckpt = $pp_ckpt"
  # NOTE: rollout_steps=1 trains on 24 h errors only; raise it (needs the multi-step
  # loader) to calibrate the 8-16 day leads.
  $PY -m torch.distributed.launch --master_port=29505 --nproc_per_node=$NUM_TASKS_PER_NODE \
      ../train_postprocess_si.py \
      --yaml_config=$config_file --config=$CFG --run_num=pp_si_sstanom \
      --epochs=10 --run_iter=1 --rollout_steps=1 \
      --pp_ckpt=$pp_ckpt \
      2>&1 | tee logs/sstanom_pp_$stamp.log
fi
