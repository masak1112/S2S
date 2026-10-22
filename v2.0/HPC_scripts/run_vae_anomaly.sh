#!/bin/bash
# SST-anomaly VAE pipeline. Run from anywhere:
#   ./run_vae_anomaly.sh clim     # 1. standardize the climatology
#   ./run_vae_anomaly.sh scale    # 2. estimate sst_anom_scale (then paste into the yaml)
#   ./run_vae_anomaly.sh train    # 3. train the VAE
#   ./run_vae_anomaly.sh all      # 1 -> 2 -> (auto-set scale) -> 3
set -euo pipefail

PY=${PY:-/net/monsoon/bing/myenv/bin/python}
ROOT=/net/monsoon/bing/S2S/v2.0
DATA=/net/monsoon/S2S/h5data0
CFG=${CFG:-$ROOT/config/exp17_anom.yaml}
RUN=${RUN:-vae_anom}
EPOCHS=${EPOCHS:-20}
SCALE_BATCHES=${SCALE_BATCHES:-200}
[ -n "${SLURM_JOB_ID:-}" ] || export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$ROOT"

do_clim() {
  $PY utils/make_sst_anom_clim.py \
    --clim_in clim_00z/clim_00z_sst.npy \
    --mean_nc $DATA/pangu_s2s_1979-2018_surface_mean.nc \
    --std_nc  $DATA/pangu_s2s_1979-2018_surface_std.nc \
    --out clim_00z/clim_00z_sst_std.npy
}

do_scale() {
  $PY train_vae_anomaly.py --yaml_config "$CFG" --estimate_scale "$SCALE_BATCHES" | tee /tmp/vae_anom_scale.log
}

set_scale() {  # write the estimated scale into the yaml
  local s
  s=$(grep -o 'set sst_anom_scale: [0-9.]*' /tmp/vae_anom_scale.log | awk '{print $3}')
  [ -n "$s" ] || { echo "could not parse scale"; exit 1; }
  sed -i "s#^\(  sst_anom_scale:\) .*#\1 $s#" "$CFG"
  echo "sst_anom_scale -> $s in $CFG"
}

do_train() {
  mkdir -p logs
  $PY train_vae_anomaly.py --yaml_config "$CFG" --run_num "$RUN" --epochs "$EPOCHS" ${INIT:+--init_ckpt "$INIT"} 2>&1 | tee logs/vae_${RUN}.log
}

case "${1:-}" in
  clim)  do_clim ;;
  scale) do_scale; echo "Now set sst_anom_scale in $CFG (or use 'all')" ;;
  train) do_train ;;
  all)   do_clim; do_scale; set_scale; do_train ;;
  *) echo "usage: $0 {clim|scale|train|all}"; exit 1 ;;
esac
