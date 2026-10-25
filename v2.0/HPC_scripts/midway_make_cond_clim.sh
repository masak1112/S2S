#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=04:00:00
#SBATCH -p caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH -o midway_cond_clim_%j.out
#SBATCH -e midway_cond_clim_%j.err
# Builds clim_00z/cond_clim_<v>.npy and cond_scalar_<v>.npy for utils/cond_anom_vars.REGISTRY.
ml python
conda activate /project/pedramh/bing/env
cd /project/pedramh/bing/S2S/v2.0 || exit 1
/project/pedramh/bing/env/bin/python utils/make_cond_anom_clim.py --out clim_00z --workers 16 $EXTRA
