#!/bin/bash
# Per-rank wrapper for `torchrun --no-python`: runs the ranks listed in
# NSYS_RANKS (default "0") under nsys, every other rank as plain python.
# The capture window is opened/closed by utils/profiling.py via
# cudaProfilerStart/Stop (S2S_PROF_START / S2S_PROF_STOP).
#
#   torchrun --no-python --nproc_per_node=4 ./nsys_rank.sh python ../train_diffusion.py ...
set -e
NSYS_RANKS=${NSYS_RANKS:-0}
NAME=${NSYS_NAME:-profile}
OUTDIR=${NSYS_OUTDIR:-reports}
mkdir -p "$OUTDIR"
for r in ${NSYS_RANKS//,/ }; do
  if [ "$r" == "${RANK:-0}" ]; then
    exec nsys profile \
      --trace=cuda,nvtx,osrt,cublas,cudnn \
      --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      --output="${OUTDIR}/${NAME}_rank${RANK:-0}" \
      "$@"
  fi
done
exec "$@"
