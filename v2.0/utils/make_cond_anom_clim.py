"""Daily 00z climatologies + global anomaly std for the multi-variable conditioning.

For every variable in utils/cond_anom_vars.REGISTRY (except SST, which re-uses the existing
clim_00z_sst.npy and the std from anom_stats_00z):
  pass 1: mean over the training years of each ordinal day's 00z field, then smoothed along the
          day axis keeping the first `--harmonics` harmonics (40 samples per day is noisy);
  pass 2: area-weighted std of (field - clim) over valid points on every `--stride`-th day.
Writes <out>/cond_clim_<v>.npy [366,H,W] (physical units, NaN where the variable is undefined)
and <out>/cond_scalar_<v>.npy = [mean, std] of the anomaly.

    python utils/make_cond_anom_clim.py --out clim_00z --workers 16
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import h5py
import numpy as np

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'cond_anom_vars', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cond_anom_vars.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
REGISTRY = _mod.REGISTRY

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default='/project/pedramh/h5data/h5data')
p.add_argument('--out', default='clim_00z')
p.add_argument('--y0', type=int, default=1979)
p.add_argument('--y1', type=int, default=2018)
p.add_argument('--harmonics', type=int, default=6)
p.add_argument('--stride', type=int, default=3)
p.add_argument('--workers', type=int, default=16)
p.add_argument('--reuse_clim', action='store_true', help='skip pass 1, use the saved cond_clim_<v>.npy')
p.add_argument('--vars', nargs='+', default=[v for v in REGISTRY if v != 'sst'])
args = p.parse_args()
KEYS = [REGISTRY[v][0] for v in args.vars]
YEARS = range(args.y0, args.y1 + 1)


def read(year, doy):
    path = os.path.join(args.data_dir, f'{year}_{doy * 4:04d}.h5')
    if not os.path.isfile(path):
        return None
    with h5py.File(path, 'r') as f:
        return np.stack([f['input'][k][()].astype(np.float64) for k in KEYS], 0)


def clim_chunk(doys):
    out = {}
    for d in doys:
        s, n = 0.0, 0.0
        for y in YEARS:
            a = read(y, d)
            if a is None:
                continue
            ok = ~np.isnan(a)
            s = s + np.where(ok, a, 0.0)
            n = n + ok
        out[d] = (s, n)
    return out


_CLIM = None


def anom_chunk(doys):
    global _CLIM
    if _CLIM is None:   # memory-mapped per worker; nothing big is pickled
        _CLIM = [np.load(os.path.join(args.out, f'cond_clim_{v}.npy'), mmap_mode='r') for v in args.vars]
    clim = _CLIM
    ss, cnt, sm = np.zeros(len(KEYS)), np.zeros(len(KEYS)), np.zeros(len(KEYS))
    lat = np.linspace(89.5, -89.5, 180)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
    for d in doys:
        for y in YEARS:
            a = read(y, d)
            if a is None:
                continue
            an = a - np.stack([np.asarray(c[d], dtype=np.float64) for c in clim], 0)
            ok = ~np.isnan(an)
            for i in range(len(KEYS)):
                m = ok[i]
                sm[i] += (an[i][m] * w[m]).sum()
                ss[i] += (an[i][m] ** 2 * w[m]).sum()
                cnt[i] += w[m].sum()
    return sm, ss, cnt


def run_pass1():
    ndoy = 366
    chunks = [list(range(i, ndoy, args.workers)) for i in range(args.workers)]
    with Pool(args.workers) as pool:
        res = pool.map(clim_chunk, chunks)
    S = np.zeros((ndoy, len(KEYS), 180, 360)); N = np.zeros_like(S)
    for r in res:
        for d, (s, n) in r.items():
            S[d], N[d] = s, n
    clim = np.where(N > 0, S / np.maximum(N, 1), np.nan).astype(np.float64)   # [366,V,H,W]
    # harmonic smoothing along the day axis (cyclic), NaN stays NaN
    valid = ~np.isnan(clim[0])
    c0 = np.where(np.isnan(clim), 0.0, clim)
    F = np.fft.rfft(c0, axis=0)
    F[args.harmonics + 1:] = 0
    sm = np.fft.irfft(F, n=ndoy, axis=0)
    clim = np.where(valid[None], sm, np.nan).astype(np.float32)
    for i, v in enumerate(args.vars):
        np.save(os.path.join(args.out, f'cond_clim_{v}.npy'), clim[:, i])
        print(v, 'clim saved', clim[:, i].shape, 'nan frac', float(np.isnan(clim[0, i]).mean()), flush=True)


if __name__ == '__main__':
    os.makedirs(args.out, exist_ok=True)
    # SST: keep the existing climatology (K) and the std from anom_stats_00z
    if os.path.isfile(os.path.join(args.out, 'clim_00z_sst.npy')):
        shutil.copy(os.path.join(args.out, 'clim_00z_sst.npy'), os.path.join(args.out, 'cond_clim_sst.npy'))
        np.save(os.path.join(args.out, 'cond_scalar_sst.npy'), np.array([0.0, 0.681823]))
    ndoy = 366
    if not args.reuse_clim:
        run_pass1()
    sel = list(range(0, ndoy, args.stride))
    parts = [sel[i::args.workers] for i in range(args.workers)]
    with Pool(args.workers) as pool:
        res = pool.map(anom_chunk, parts)
    sm = sum(r[0] for r in res); ss = sum(r[1] for r in res); cnt = sum(r[2] for r in res)
    mean = sm / cnt; std = np.sqrt(ss / cnt - mean ** 2)
    for i, v in enumerate(args.vars):
        np.save(os.path.join(args.out, f'cond_scalar_{v}.npy'), np.array([mean[i], std[i]]))
        print(f'{v}: anomaly mean {mean[i]:.4g} std {std[i]:.6g}', flush=True)
    sys.exit(0)
