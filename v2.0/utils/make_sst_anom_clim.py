"""Convert the raw SST daily climatology into standardized units for the anomaly VAE.

The dataset feeds the models (sst_K - mean) / std. The anomaly VAE needs the
climatology in the same units, so that  anomaly = x_std - clim_std  equals
(sst_K - clim_K) / std.  NaN (land) is preserved; the VAE builds its ocean mask
from it.

Usage:
    python utils/make_sst_anom_clim.py \
        --clim_in clim_00z/clim_00z_sst.npy \
        --mean_nc /net/monsoon/S2S/h5data0/pangu_s2s_1979-2018_surface_mean.nc \
        --std_nc  /net/monsoon/S2S/h5data0/pangu_s2s_1979-2018_surface_std.nc \
        --out clim_00z/clim_00z_sst_std.npy
"""
import argparse
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("--clim_in", required=True)
parser.add_argument("--mean_nc", required=True)
parser.add_argument("--std_nc", required=True)
parser.add_argument("--var", default="sea_surface_temperature")
parser.add_argument("--out", required=True)
args = parser.parse_args()

clim = np.load(args.clim_in).astype(np.float32)  # [366, H, W], Kelvin, NaN on land
with xr.open_dataset(args.mean_nc) as ds:
    mean = float(ds[args.var].values)
with xr.open_dataset(args.std_nc) as ds:
    std = float(ds[args.var].values)

clim_std = (clim - mean) / std
np.save(args.out, clim_std.astype(np.float32))
print(f"{args.var}: mean={mean:.3f} K, std={std:.3f} K")
print(f"saved {args.out} shape={clim_std.shape}, land(NaN) frac day0={np.isnan(clim_std[0]).mean():.3f}")
print("Next: estimate sst_anom_scale with  train_vae_anomaly.py --estimate_scale 200")
