"""
Convert per-timestep HDF5 files to a single zarr store.

Each H5 file  {h5_dir}/{year}_{tidx:04}.h5  contains one timestep under
the 'input' group: variables shaped (180, 360).  This script stacks all
timesteps into a zarr store with shape (T, 180, 360) and configurable
chunk shape.

Usage:
  /projects/bdiu/bgong1/.venv/bin/python h5_to_zarr.py
  /projects/bdiu/bgong1/.venv/bin/python h5_to_zarr.py \
      --h5-dir  /work/hdd/bdiu/bgong1/data/h5data \
      --dst     /work/hdd/bccu/bgong1/data/zarr_data_chunck_latlon/1979.zarr \
      --year    1979 \
      --chunk-t 1 --chunk-lat 60 --chunk-lon 30
"""
import argparse
import os
import shutil
import time
import glob

import h5py
import numpy as np
import zarr
from zarr.codecs import ZstdCodec


H5_DIR_DEFAULT  = "/work/hdd/bdiu/bgong1/data/h5data"
DST_DEFAULT     = "/work/hdd/bccu/bgong1/data/zarr_data_chunck_latlon/1979.zarr"
YEAR_DEFAULT    = 1979
CHUNK_T         = 1
CHUNK_LAT       = 60
CHUNK_LON       = 30
DATA_TIMEDELTA_HOURS = 6
N_STEPS_YEAR    = 365 * (24 // DATA_TIMEDELTA_HOURS)   # 1460


def h5_path(h5_dir: str, year: int, tidx: int) -> str:
    return os.path.join(h5_dir, f"{year}_{tidx:04}.h5")


def convert(h5_dir: str, dst_path: str, year: int,
            chunk_t: int, chunk_lat: int, chunk_lon: int, n_workers: int = 8):

    # Count available timesteps
    files = sorted(
        p for p in glob.glob(os.path.join(h5_dir, f"{year}_*.h5"))
        if os.path.isfile(p) and p.endswith(".h5") and os.path.dirname(os.path.abspath(p)) == os.path.abspath(h5_dir)
    )
    n_steps = len(files)
    if n_steps == 0:
        raise FileNotFoundError(f"No H5 files found for year {year} in {h5_dir}")

    # Read variable list and spatial shape from first file
    with h5py.File(files[0], "r") as f:
        variables = sorted(k for k in f["input"].keys() if k != "time")
        sample = f["input"][variables[0]][()]
        lat_size, lon_size = sample.shape

    print(f"First 3 files found: {files[:3]}")
    print(f"H5 source  : {h5_dir}")
    print(f"Dest       : {dst_path}")
    print(f"Year       : {year}  ({n_steps} timesteps)")
    print(f"Variables  : {len(variables)}")
    print(f"Spatial    : ({lat_size}, {lon_size})")
    print(f"Chunks     : ({chunk_t}, {chunk_lat}, {chunk_lon})\n")

    os.makedirs(dst_path, exist_ok=True)
    store = zarr.storage.LocalStore(dst_path)
    # mode="a" — open existing store if present, create if not
    dst = zarr.open_group(store=store, mode="a", zarr_format=3)

    shape  = (n_steps, lat_size, lon_size)
    chunks = (chunk_t, chunk_lat, chunk_lon)
    # expected chunk count per 3-D variable
    chunks_per_var = (
        (n_steps + chunk_t - 1) // chunk_t *
        (lat_size + chunk_lat - 1) // chunk_lat *
        (lon_size + chunk_lon - 1) // chunk_lon
    )

    def _is_complete(var):
        """True if the variable directory already has all expected chunk files."""
        chunk_dir = os.path.join(dst_path, var, "c")
        if not os.path.isdir(chunk_dir):
            return False
        n = sum(1 for _, _, fs in os.walk(chunk_dir) for f in fs)
        return n >= chunks_per_var

    # Create array schemas for any variable not yet present
    print("Checking / creating array schemas ...")
    zarr_arrays = {}
    for var in variables:
        if var in dst:
            zarr_arrays[var] = dst[var]
        else:
            zarr_arrays[var] = dst.create_array(
                var,
                shape=shape,
                chunks=chunks,
                dtype=np.float32,
                compressors=ZstdCodec(level=0),
            )

    # Time array — write only if missing
    if "time" not in dst:
        import datetime
        base = datetime.datetime(year, 1, 1)
        time_strs = [
            (base + datetime.timedelta(hours=i * DATA_TIMEDELTA_HOURS)).strftime(
                "%Y-%m-%dT%H:%M:%S.000000000"
            )
            for i in range(n_steps)
        ]
        dst.create_array(
            "time",
            shape=(n_steps,),
            chunks=(n_steps,),
            dtype=str,
            compressors=ZstdCodec(level=0),
        )[:] = np.array(time_strs, dtype=str)

    # Identify which variables still need writing
    todo = [var for var in variables if not _is_complete(var)]
    skipped = len(variables) - len(todo)
    print(f"  {skipped} variables already complete, {len(todo)} remaining.")

    if not todo:
        print("All variables already written — nothing to do.")
        return

    # Parallel write: each worker reads one variable from source zarr and
    # writes it rechunked to the destination. Threads release the GIL during
    # zarr I/O so multiple variables are compressed and written concurrently.
    src_zarr = zarr.open("/work/hdd/bccu/bgong1/data/zarr_data/1979.zarr", mode="r")
    n_workers = min(n_workers, len(todo))
    print(f"Converting variables in parallel (workers={n_workers}) ...")

    completed = [0]
    t_total = time.perf_counter()

    def process_var(args):
        i, var = args
        t0 = time.perf_counter()
        buf = np.array(src_zarr[var][:], dtype=np.float32)
        zarr_arrays[var][:] = buf
        elapsed = time.perf_counter() - t0
        completed[0] += 1
        rate = completed[0] / (time.perf_counter() - t_total)
        eta  = (len(todo) - completed[0]) / rate if rate > 0 else 0
        print(f"  [{completed[0]:3d}/{len(todo)}] {var:50s}  {elapsed:.1f}s  ETA {eta/60:.1f} min",
              flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(process_var, (i, var)) for i, var in enumerate(todo)]
        for f in as_completed(futures):
            f.result()  # re-raise any exception

    total = time.perf_counter() - t_total
    print(f"\nDone in {total/60:.1f} min  →  {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir",    default=H5_DIR_DEFAULT)
    parser.add_argument("--dst",       default=DST_DEFAULT)
    parser.add_argument("--year",      type=int, default=YEAR_DEFAULT)
    parser.add_argument("--chunk-t",   type=int, default=CHUNK_T)
    parser.add_argument("--chunk-lat", type=int, default=CHUNK_LAT)
    parser.add_argument("--chunk-lon", type=int, default=CHUNK_LON)
    parser.add_argument("--workers",   type=int, default=8)
    args = parser.parse_args()
    convert(args.h5_dir, args.dst, args.year,
            args.chunk_t, args.chunk_lat, args.chunk_lon, args.workers)


if __name__ == "__main__":
    main()
