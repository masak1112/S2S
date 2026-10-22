#!/bin/bash -l

#SBATCH --job-name=vae_fc3d
#SBATCH --output=vae_fc3d_%x_%j.out
#SBATCH --error=vae_fc3d_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=gongbing@uchicago.edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# SST encoder-decoder forecast t -> t+LEAD days (default 3); latent is later the SI condition.
# Usage:    sbatch dsi_a100_train_vae_forecast.sh
# Override: LEAD=3 EPOCHS=40 RUN=vae_fc3d2 sbatch --export=ALL dsi_a100_train_vae_forecast.sh
# Optional warm start from the reconstruction VAE: INIT=/path/to/vae_vae_anom/ckpt.tar
conda activate /net/monsoon/bing/myenv
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

ROOT=/net/monsoon/bing/S2S/v2.0
cd "$ROOT"
mkdir -p logs
RUN=${RUN:-vae_fc3d}
python train_vae_forecast.py \
    --yaml_config config/exp17_anom.yaml \
    --lead_days ${LEAD:-3} \
    --epochs ${EPOCHS:-20} \
    --run_num "$RUN" \
    ${INIT:+--init_ckpt "$INIT"} 2>&1 | tee logs/vae_${RUN}.log
