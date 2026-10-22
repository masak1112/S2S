#!/bin/bash -l

#SBATCH --job-name=vae_anom
#SBATCH --output=vae_anom_%x_%j.out
#SBATCH --error=vae_anom_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=gongbing@uchicago.edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# Usage: sbatch dsi_a100_train_vae.sh   (train only; losses logged to wandb + stdout)
# Override: EPOCHS=40 RUN=vae_anom2 sbatch --export=ALL dsi_a100_train_vae.sh
conda activate /net/monsoon/bing/myenv
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
/net/monsoon/bing/S2S/v2.0/HPC_scripts/run_vae_anomaly.sh train
