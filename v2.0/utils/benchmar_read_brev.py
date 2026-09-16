"""
Read benchmark: zarr vs HDF5, mimicking GetDataset.__getitem__ access pattern.

Pattern:
  shuffled -- random reads (training __getitem__), executed by N dask worker
              processes in parallel, like DataLoader(num_workers=N).
              Each read fetches a window of --steps_per_read consecutive
              timesteps starting at a random index (1 = single-step sample).
              Dask dashboard: http://<host>:35033

Reports per-read latency seen inside a worker and the aggregate throughput
(total bytes / wall time) delivered by all workers together.

Usage:
  python benchmark_read.py                         # 16 workers, 100 reads
  python benchmark_read.py --workers 1             # single-process baseline
  python benchmark_read.py --n_reads 400 --workers 8 --skip_h5
  python benchmark_read.py --steps_per_read 4     # 4 consecutive steps per read
"""
import argparse
import os
import time

import h5py
import numpy as np
import zarr
from dask.distributed import Client

DASHBOARD = ":35033"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZARR_DIR  = "/data/zarr_era5"
H5_DIR    = "/data/ERA5"
YEAR      = 1979
N_STEPS   = 1460           # timesteps in one year (6-hourly, 365 days)
DATA_TIMEDELTA_HOURS = 6

VARIABLES = [
    # upper-air  (5 vars × 13 levels = 65 keys in zarr)
    *[f"{v}_{int(lev)}.0"
      for v in ("temperature", "u_component_of_wind", "v_component_of_wind",
                "specific_humidity", "geopotential")
      for lev in (5, 10, 20, 30, 50, 70, 150, 250, 400, 500, 850, 925, 1000)],
    # surface
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "surface_pressure",
    # land / ocean
    "volumetric_soil_water_layer_1", "soil_temperature_level_1", "skin_temperature",
    "sea_surface_temperature",
    # boundary
    "toa_incident_solar_radiation",
    # diagnostic
    "total_precipitation_24hr", "mean_top_net_long_wave_radiation_flux",
]

# H5 variable list mirrors the zarr names; the h5 input group uses the same names
H5_VARIABLES = VARIABLES  # same strings stored under input/{var}


# ---------------------------------------------------------------------------
# Zarr reader
# ---------------------------------------------------------------------------
def open_zarr(year: int):
    path = os.path.join(ZARR_DIR, f"{year}.zarr")
    return zarr.open(path, mode="r")


def zarr_read_timestep(store, tidx: int, n_steps: int = 1) -> np.ndarray:
    """Read all benchmark variables for timesteps [tidx, tidx+n_steps) from a
    zarr store.  Returns (n_steps, n_vars, lat, lon)."""
    return np.stack([store[v][tidx:tidx + n_steps] for v in VARIABLES], axis=1)


# ---------------------------------------------------------------------------
# HDF5 reader  (one file per timestep: {year}_{idx:04}.h5)
# ---------------------------------------------------------------------------
def h5_path(year: int, tidx: int) -> str:
    return os.path.join(H5_DIR, f"{year}_{tidx:04}.h5")


def _h5_read_one(year: int, tidx: int) -> np.ndarray:
    """Read all benchmark variables for a single timestep from one h5 file."""
    path = h5_path(year, tidx)
    with h5py.File(path, "r") as f:
        return np.stack([f["input"][v][()] for v in H5_VARIABLES], axis=0)


def h5_read_timestep(year: int, tidx: int, n_steps: int = 1) -> np.ndarray:
    """Read timesteps [tidx, tidx+n_steps); one h5 file per step.
    Returns (n_steps, n_vars, lat, lon)."""
    return np.stack([_h5_read_one(year, t) for t in range(tidx, tidx + n_steps)],
                    axis=0)


# ---------------------------------------------------------------------------
# Worker-side tasks (run inside the pool processes)
# ---------------------------------------------------------------------------
_zstore = None   # one open zarr store per worker process (opened lazily)


def _zarr_task(tidx: int, n_steps: int = 1):
    # Lazy open: dask pickles __main__ functions by value, so a separate
    # initializer would populate a different copy of this module's globals.
    global _zstore
    if _zstore is None:
        _zstore = open_zarr(YEAR)
    t0 = time.perf_counter()
    arr = zarr_read_timestep(_zstore, tidx, n_steps)
    return time.perf_counter() - t0, arr.nbytes


def _h5_task(tidx: int, n_steps: int = 1):
    t0 = time.perf_counter()
    arr = h5_read_timestep(YEAR, tidx, n_steps)
    return time.perf_counter() - t0, arr.nbytes


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_shuffled(client, task_fn, indices, label: str, n_steps: int = 1) -> dict:
    """Shuffled window reads (training pattern) across the dask workers.
    Each read fetches n_steps consecutive timesteps starting at the index.

    Only (latency, nbytes) is returned from each worker, not the array itself,
    so the IPC cost of shipping samples to the main process is NOT included.
    """
    rng = np.random.default_rng(42)
    shuffled = [int(i) for i in rng.permutation(indices)]
    workers = len(client.scheduler_info()["workers"])

    # warm-up through the same path as the timed reads: one task per worker
    # pays store open / imports and pulls the first chunk into page cache
    client.gather(client.map(task_fn, [0] * workers, n_steps=n_steps, pure=False))

    t_wall = time.perf_counter()
    # pure=False: duplicate indices must not be deduplicated into one task
    futures = client.map(task_fn, shuffled, n_steps=n_steps, pure=False)
    results = client.gather(futures)
    wall = time.perf_counter() - t_wall

    times = np.array([r[0] for r in results])
    total_bytes = sum(r[1] for r in results)
    return _stats(times, total_bytes, wall, label, workers, n_steps)


def _stats(times, total_bytes, wall, label, workers, n_steps=1) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  workers     : {workers}")
    print(f"  reads       : {len(times)} reads x {n_steps} steps "
          f"= {len(times) * n_steps} steps")
    print(f"  total data  : {total_bytes / 1e9:.3f} GB")
    print(f"  wall time   : {wall:.2f} s")
    print(f"  aggregate   : {total_bytes / wall / 1e6:.1f} MB/s   (all workers)")
    print(f"  samples/s   : {len(times) / wall:.2f}")
    print(f"  per-read latency inside a worker:")
    print(f"    mean      : {times.mean()*1e3:.1f} ms")
    print(f"    p50       : {np.percentile(times, 50)*1e3:.1f} ms")
    print(f"    p95       : {np.percentile(times, 95)*1e3:.1f} ms")
    print(f"    p99       : {np.percentile(times, 99)*1e3:.1f} ms")
    return dict(label=label, workers=workers,
                mean_ms=times.mean()*1e3,
                p95_ms=np.percentile(times, 95)*1e3,
                agg_MBs=total_bytes / wall / 1e6,
                samples_s=len(times) / wall)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reads", type=int, default=1000,
                        help="number of reads for the shuffled benchmark")
    parser.add_argument("--steps_per_read", type=int, default=1,
                        help="consecutive timesteps fetched by each read "
                             "(1 = single-step sample)")
    parser.add_argument("--workers", type=int, default=None,
                        help="number of dask worker processes (1 thread each); "
                             "omit to use dask's default local cluster")
    parser.add_argument("--skip_h5", action="store_true",
                        help="skip HDF5 benchmark (e.g. h5data not mounted)")
    args = parser.parse_args()

    n_steps = max(1, min(args.steps_per_read, N_STEPS))
    n_reads = min(args.n_reads, N_STEPS)

    if args.workers is None:
        client = Client(dashboard_address=DASHBOARD)
    else:
        client = Client(n_workers=max(1, args.workers), threads_per_worker=1,
                        dashboard_address=DASHBOARD)
    info = client.scheduler_info()["workers"]
    workers = len(info)
    threads = sum(w["nthreads"] for w in info.values())
    print(f"dask dashboard: {client.dashboard_link}")
    print(f"dask cluster  : {workers} workers, {threads} threads total")

    rng = np.random.default_rng(0)
    # start indices such that the whole window stays inside the year
    shuffled_idxs = rng.integers(0, N_STEPS - n_steps + 1, size=n_reads)

    results = []

    # ---- Zarr ----
    print(f"\n>>> zarr: {n_reads} shuffled reads x {n_steps} steps "
          f"with {workers} workers ...")
    results.append(run_shuffled(
        client, _zarr_task, shuffled_idxs,
        f"zarr  | shuffled | {n_reads}x{n_steps} | {len(VARIABLES)} vars",
        n_steps=n_steps))

    # ---- HDF5 ----
    if not args.skip_h5:
        if not os.path.exists(h5_path(YEAR, 0)):
            print(f"\n[WARN] HDF5 files not found under {H5_DIR}, skipping h5 benchmark.")
        else:
            print(f"\n>>> hdf5: {n_reads} shuffled reads x {n_steps} steps "
                  f"with {workers} workers ...")
            results.append(run_shuffled(
                client, _h5_task, shuffled_idxs,
                f"hdf5  | shuffled | {n_reads}x{n_steps} | {len(H5_VARIABLES)} vars",
                n_steps=n_steps))

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'store+pattern':<40} {'wrk':>4} {'mean ms':>8} {'p95 ms':>8} {'MB/s':>8} {'smp/s':>7}")
    print(f"  {'-'*40} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for r in results:
        print(f"  {r['label']:<40} {r['workers']:>4} {r['mean_ms']:>8.1f} "
              f"{r['p95_ms']:>8.1f} {r['agg_MBs']:>8.1f} {r['samples_s']:>7.2f}")
    print()

    client.close()


if __name__ == "__main__":
    # Dask spawns worker processes that re-import this module, so the Client
    # must be created inside main(), never at module level.
    main()
