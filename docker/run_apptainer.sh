#!/bin/bash -l
# Apptainer replacement for HPC_scripts/nvidia_brev.sh.
#
# Instead of `conda activate s2s_env`, this runs the same torchrun command
# inside the container image. Everything mutable (data, checkpoints, results)
# is bind-mounted, so the image itself stays read-only and portable.
#
#   ./docker/run_apptainer.sh train
#   ./docker/run_apptainer.sh infer
#   SIF=/scratch/$USER/s2s.sif DATA_DIR=/scratch/$USER/h5data ./docker/run_apptainer.sh train
set -euo pipefail

MODE="${1:-train}"

# --- paths (override via environment) -------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIF="${SIF:-${PROJECT_DIR}/s2s.sif}"
DATA_DIR="${DATA_DIR:-/data/ERA5}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/v2.0/HPC_scripts/results}"
# Working dir inside the container (/runs): holds results/, spectra_out/,
# gif_out/, acc_plots/ that the training code writes relative to cwd.
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/v2.0/HPC_scripts}"
CONFIG_FILE="${CONFIG_FILE:-/workspace/v2.0/config/exp16_nvidia_v2.yaml}"
RUN_NUM="${RUN_NUM:-si_c1_full_state}"

# --- distributed settings --------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_TASKS_PER_NODE="${NUM_TASKS_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${MODE}" in
  train) SCRIPT=/workspace/v2.0/train_diffusion.py; PORT="${MASTER_PORT:-29500}" ;;
  infer) SCRIPT=/workspace/v2.0/inference.py;       PORT="${MASTER_PORT:-29501}" ;;
  shell) SCRIPT=""                                                              ;;
  *) echo "usage: $0 {train|infer|shell}" >&2; exit 1 ;;
esac

[ -f "${SIF}" ] || { echo "ERROR: image not found: ${SIF}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

# Bind mutable state in; --nv exposes the host NVIDIA driver and devices.
# The shipped configs use absolute host paths (data_dir: /data/ERA5,
# checkpoint_path_*: /data/bing/S2S/...). Bind to those same paths inside the
# container so existing YAML files work unmodified.
HOST_PROJECT_PATH="${HOST_PROJECT_PATH:-/data/bing/S2S}"
BINDS=(
  --bind "${DATA_DIR}:/data/ERA5:ro"
  --bind "${CKPT_DIR}:${HOST_PROJECT_PATH}/checkpoints"
  --bind "${OUTPUT_DIR}:/runs"
)
# Let a bind-mounted source tree override the baked-in copy for development.
if [ "${DEV_MOUNT:-false}" = "true" ]; then
  BINDS+=(--bind "${PROJECT_DIR}/v2.0:/workspace/v2.0")
fi

echo "image      : ${SIF}"
echo "mode       : ${MODE}"
echo "GPUs       : ${CUDA_VISIBLE_DEVICES} (${NUM_TASKS_PER_NODE} procs)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true

if [ "${MODE}" = "shell" ]; then
  exec apptainer shell --nv --cleanenv "${BINDS[@]}" "${SIF}"
fi

# torchrun, not the deprecated torch.distributed.launch used by the conda scripts.
exec apptainer exec --nv --cleanenv "${BINDS[@]}" "${SIF}" \
  torchrun --standalone \
           --nproc_per_node="${NUM_TASKS_PER_NODE}" \
           --master_port="${PORT}" \
           "${SCRIPT}" \
           --yaml_config="${CONFIG_FILE}" \
           --run_num="${RUN_NUM}"
