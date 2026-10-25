"""Evolving anomaly conditioning for autoregressive rollouts (multi-variable conditioning).

The latent SI is conditioned on anomaly maps (utils/cond_anom_vars.py). By default the map built
from the initial condition is held fixed. EvolvingCond instead rebuilds, at every rollout step,
the channels of the variables the model itself predicts
    SST, soil water   <- surface state      z10 / z250 / z500 <- upper-air state     OLR <- diagnostic output
from the fed-back state (anomaly vs the climatology of that step's valid date, same sigma and the
same masked interpolation as the dataloader). Variables the model does not predict (snow depth,
sea ice) and all valid-fraction channels keep their initial-condition values.
"""
import datetime
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F

from utils.cond_anom_vars import REGISTRY, cond_channel_names, cond_vars


class EvolvingCond:
    def __init__(self, dataset, params, device):
        self.device = device
        self.vars = cond_vars(params)
        self.names = cond_channel_names(params)
        self.size = tuple(dataset.sst_cond_size)
        self.dt_hours = int(params.timedelta_hours)
        self.clim, self.sigma = dataset.cond_clim, dataset.cond_sigma
        self.doy = dataset._doy_index
        sv, uv = list(dataset.surface_variables), list(dataset.upper_air_variables)
        levels = [int(l) for l in dataset.levels]
        dv = list(dataset.diagnostic_variables)
        self.src = {}   # var -> (kind, index tuple, mean, std) in dataset standardization
        for v in self.vars:
            key = REGISTRY[v][0]
            if key in sv:
                i = sv.index(key)
                self.src[v] = ('sfc', (i,), float(dataset.surface_mean[i]), float(dataset.surface_std[i]))
            elif key.startswith('geopotential_'):
                vi, li = uv.index('geopotential'), levels.index(int(float(key.split('_')[1])))
                self.src[v] = ('ua', (vi, li), float(dataset.upper_air_mean[vi, li]),
                               float(dataset.upper_air_std[vi, li]))
            elif key in dv:
                i = dv.index(key)
                self.src[v] = ('diag', (i,), float(dataset.diagnostic_mean[i]), float(dataset.diagnostic_std[i]))
        self.evolved = [v for v in self.vars if v in self.src]
        self.anom_idx = [self.names.index(f'anom_{v}') for v in self.evolved]

    def _physical(self, v, s_cur, ua_cur, diag_prev):
        kind, idx, mean, std = self.src[v]
        if kind == 'sfc':
            x = s_cur[:, idx[0]]
        elif kind == 'ua':
            x = ua_cur[:, idx[0], idx[1]]
        else:
            x = diag_prev[:, idx[0]]
        return x.float() * std + mean

    @torch.no_grad()
    def build(self, persistent, s_cur, ua_cur, diag_prev, start_times, step):
        """persistent: [B, C, h, w] initial-condition map; returns the map for rollout step `step`."""
        if step == 0 or not self.evolved:
            return persistent
        B = s_cur.shape[0]
        dates = [datetime.datetime(*[int(t) for t in start_times[b][:4]]) + timedelta(hours=self.dt_hours * step)
                 for b in range(B)]
        anoms, valids = [], []
        for v in self.evolved:
            raw = self._physical(v, s_cur, ua_cur, diag_prev)                       # [B, H, W]
            clim = torch.stack([torch.from_numpy(np.asarray(self.clim[v][self.doy(d)], dtype=np.float32))
                                for d in dates], 0).to(raw.device)
            valid = (~torch.isnan(clim)).float()
            anoms.append(torch.nan_to_num(raw - clim, nan=0.0) * valid / self.sigma[v])
            valids.append(valid)
        n = len(self.evolved)
        lo = F.interpolate(torch.stack(anoms + valids, dim=1), size=self.size, mode='bilinear',
                           align_corners=False, antialias=True)
        frac = lo[:, n:]
        anom_lo = lo[:, :n] / frac.clamp_min(1e-3) * (frac > 1e-3).float()
        out = persistent.clone()
        out[:, self.anom_idx] = anom_lo.to(out.dtype)
        return out
