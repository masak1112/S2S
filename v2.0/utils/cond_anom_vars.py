"""Registry + helpers for the multi-variable anomaly conditioning of the latent SI.

Every conditioning variable is turned into one channel
    (x_ic - daily_clim[doy]) / sigma_v        (land/ocean/ice masked to 0 outside its domain)
interpolated to the latent grid (sst_cond_size); variables that only exist over part of
the globe (`masked`) get one extra "valid fraction" channel. Channel layout:
    [anom_v for v in cond_anom_vars] + [valid_v for v in cond_anom_vars if masked(v)]
`doy` is the 0-based ordinal day of year (file index // 4 for 6-hourly files), the same
convention as the SST anomaly VAE (VAE._doy_index).
Climatology / scale files are written by utils/make_cond_anom_clim.py.
"""
import os

import numpy as np

# name -> (h5 dataset key, masked)
REGISTRY = {
    'sst':  ('sea_surface_temperature', True),
    'z10':  ('geopotential_10.0', False),
    'z250': ('geopotential_250.0', False),
    'z500': ('geopotential_500.0', False),
    'olr':  ('mean_top_net_long_wave_radiation_flux', False),
    'vswl': ('volumetric_soil_water_layer_1', True),
    'snow': ('snow_depth', False),
    'sic':  ('sea_ice_cover', True),
}


def cond_vars(params):
    """Configured variable names, or None for the legacy SST-only conditioning."""
    v = getattr(params, 'cond_anom_vars', None)
    if not v:
        return None
    v = [str(x) for x in v]
    bad = [x for x in v if x not in REGISTRY]
    if bad:
        raise ValueError(f"unknown cond_anom_vars {bad}; known: {sorted(REGISTRY)}")
    return v


def cond_channel_names(params):
    """Names of the conditioning channels in order ('anom_sst', ..., 'valid_sst', ...)."""
    v = cond_vars(params)
    if v is None:
        return ['anom_sst', 'valid_sst']
    return [f'anom_{x}' for x in v] + [f'valid_{x}' for x in v if REGISTRY[x][1]]


def cond_dim(params):
    return len(cond_channel_names(params))


def var_channel_groups(params):
    """{variable: [channel indices]} (anomaly channel + its valid channel if any)."""
    names = cond_channel_names(params)
    v = cond_vars(params) or ['sst']
    return {x: [i for i, n in enumerate(names) if n in (f'anom_{x}', f'valid_{x}')] for x in v}


def load_stats(params):
    """clim memmaps [366,H,W] and scalar sigma per variable (physical units)."""
    d = str(params.cond_anom_dir)
    clim, sigma = {}, {}
    for x in cond_vars(params):
        clim[x] = np.load(os.path.join(d, f'cond_clim_{x}.npy'), mmap_mode='r')
        sigma[x] = float(np.load(os.path.join(d, f'cond_scalar_{x}.npy'))[1])
    return clim, sigma
