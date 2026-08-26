#!/bin/bash
# Docker equivalent of run_apptainer.sh, for systems where Docker is available.
#
#   ./docker/run_docker.sh train
#   ./docker/run_docker.sh infer
#   ./docker/run_docker.sh shell
set -euo pipefail

MODE="${1:-train}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE="${IMAGE:-s2s:latest}"
HOST_PROJECT_PATH="${HOST_PROJECT_PATH:-/data/bing/S2S}"

DATA_DIR="${DATA_DIR:-/data/ERA5}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/v2.0/HPC_scripts/results}"
# Working dir inside the container (/runs): holds results/, spectra_out/,
# gif_out/, acc_plots/ that the training code writes relative to cwd.
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/v2.0/HPC_scripts}"
CONFIG_FILE="${CONFIG_FILE:-/workspace/v2.0/config/exp16_nvidia_v2.yaml}"
RUN_NUM="${RUN_NUM:-si_c1_full_state}"
NUM_TASKS_PER_NODE="${NUM_TASKS_PER_NODE:-4}"

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

case "${MODE}" in
  train) SCRIPT=/workspace/v2.0/train_diffusion.py; PORT="${MASTER_PORT:-29500}" ;;
  infer) SCRIPT=/workspace/v2.0/inference.py;       PORT="${MASTER_PORT:-29501}" ;;
  shell) SCRIPT="" ;;
  *) echo "usage: $0 {train|infer|shell}" >&2; exit 1 ;;
esac

# --shm-size is required: the default 64MB deadlocks PyTorch DataLoader workers.
DOCKER_ARGS=(
  --rm --gpus "${GPUS:-all}"
  --shm-size="${SHM_SIZE:-16g}"
  --ipc=host
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  -e WANDB_API_KEY="${WANDB_API_KEY:-}"
  -v "${DATA_DIR}:/data/ERA5:ro"
  -v "${CKPT_DIR}:${HOST_PROJECT_PATH}/checkpoints"
  -v "${OUTPUT_DIR}:/runs"
  -u "$(id -u):$(id -g)"
)
[ "${DEV_MOUNT:-false}" = "true" ] && DOCKER_ARGS+=(-v "${PROJECT_DIR}/v2.0:/workspace/v2.0")

if [ "${MODE}" = "shell" ]; then
  exec docker run -it "${DOCKER_ARGS[@]}" "${IMAGE}" /bin/bash
fi

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" \
  torchrun --standalone --nproc_per_node="${NUM_TASKS_PER_NODE}" \
           --master_port="${PORT}" \
           "${SCRIPT}" --yaml_config="${CONFIG_FILE}" --run_num="${RUN_NUM}"
