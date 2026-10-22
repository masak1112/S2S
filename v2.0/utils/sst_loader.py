"""SST-only data loader for the SST VAE scripts (train_vae_anomaly.py, train_vae_forecast.py).

utils/data_loader_multifiles.GetDataset reads every variable (~90 fields) of two
H5 files per sample and normalises all of them, to use one SST channel. This
reads only `input/sea_surface_temperature` (~260 KB) at t and t+lead_days.

Values are identical to the full loader's surface channel: land NaN filled with
mask_fill['sea_surface_temperature'] (270 K), then standardised with the
surface mean/std files. Date indexing follows GetDataset (one H5 file per
data_timedelta_hours step, named {year}_{idx:04d}.h5).

Each item: (sst_t [1,H,W], sst_t+lead [1,H,W], date [4] = year, month, day, hour of t).
With lead_days=0 the target is the input (no second read).
"""
import os
from datetime import datetime, timedelta

import h5py
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, RandomSampler

SST = 'sea_surface_temperature'


class SSTDataset(Dataset):
    def __init__(self, params, year_start, year_end, lead_days=0):
        self.data_dir = params.data_dir
        self.step_h = int(params.data_timedelta_hours)
        self.lead_days = int(lead_days)
        self.start = datetime(year_start, 1, 1)
        hours = (datetime(year_end, 1, 1) - self.start).days * 24
        # same trimming as GetDataset: stop one lead short of the end of the range
        self.n = len(np.arange(0, hours, self.step_h)) - self.lead_days * 24 // self.step_h
        mask_fill = getattr(params, 'mask_fill', None) or {}
        self.fill = float(mask_fill.get(SST, 270.))
        with xr.open_dataset(os.path.join(self.data_dir, params.surface_mean)) as ds:
            self.mean = float(ds[SST].values)
        with xr.open_dataset(os.path.join(self.data_dir, params.surface_std)) as ds:
            self.std = float(ds[SST].values)

    def __len__(self):
        return self.n

    def _read(self, t):
        idx = int((t - datetime(t.year, 1, 1)).total_seconds()) // 3600 // self.step_h
        with h5py.File(os.path.join(self.data_dir, f'{t.year}_{idx:04d}.h5'), 'r') as f:
            x = f['input'][SST][()]
        x = np.where(np.isnan(x), self.fill, x)
        return torch.from_numpy(((x - self.mean) / self.std).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, i):
        t = self.start + timedelta(hours=i * self.step_h)
        x = self._read(t)
        y = self._read(t + timedelta(days=self.lead_days)) if self.lead_days > 0 else x
        date = torch.tensor([t.year, t.month, t.day, t.hour])
        return x, y, date


def get_sst_loader(params, year_start, year_end, lead_days=0, num_workers=8, shuffle=True):
    ds = SSTDataset(params, year_start, year_end, lead_days)
    return DataLoader(ds, batch_size=int(params.batch_size), sampler=RandomSampler(ds) if shuffle else None,
                      num_workers=num_workers, drop_last=True, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
