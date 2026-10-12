"""
Convert per-timestep HDF5 files to a single zarr store, reading from H5.

Each H5 file  {h5_dir}/{year}_{tidx:04}.h5  contains one timestep under
the 'input' group: variables shaped (180, 360).  This script stacks all
timesteps into a zarr store with shape (T, 180, 360) and configurable
chunk shape.

This is the H5-sourced counterpart to h5_to_zarr.py, which rechunks from an
existing zarr store instead.  Variables keep their flat `{name}_{level}` names
so utils/data_loader_multifiles.py can index them unchanged.

Timesteps are read in blocks by a process pool; with chunk_t == 1 each output
chunk belongs to exactly one block, so workers never write the same chunk.

Usage:
  python h5_to_zarr_from_h5.py \
      --h5-dir  /data/ERA5 \
      --dst     /data/zarr_era5/1979.zarr \
      --year    1979 \
      --chunk-t 1 --chunk-lat 60 --chunk-lon 30
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import zarr
from zarr.codecs import ZstdCodec


H5_DIR_DEFAULT = "/data/ERA5"
DST_DEFAULT    = "/data/zarr_era5/1979.zarr"
YEAR_DEFAULT   = 1979
CHUNK_T        = 1
CHUNK_LAT      = 60
CHUNK_LON      = 30
DATA_TIMEDELTA_HOURS = 6
BLOCK_DEFAULT  = 20
DIMS           = ("time", "lat", "lon")


def h5_path(h5_dir: str, year: int, tidx: int) -> str:
    return os.path.join(h5_dir, f"{year}_{tidx:04}.h5")


def scan_year(h5_dir: str, year: int):
    """Return (sorted file list, variable names, (lat, lon)) for a year."""
    files = sorted(
        p for p in glob.glob(os.path.join(h5_dir, f"{year}_*.h5"))
        if os.path.isfile(p)
        and os.path.dirname(os.path.abspath(p)) == os.path.abspath(h5_dir)
    )
    if not files:
        raise FileNotFoundError(f"No H5 files found for year {year} in {h5_dir}")

    with h5py.File(files[0], "r") as f:
        variables = sorted(k for k in f["input"].keys() if k != "time")
        lat_size, lon_size = f["input"][variables[0]].shape
    return files, variables, (lat_size, lon_size)


def read_times(files):
    """Read the scalar 'time' string out of each timestep file."""
    out = []
    for p in files:
        with h5py.File(p, "r") as f:
            raw = f["input/time"][()]
            out.append(raw.decode() if isinstance(raw, bytes) else str(raw))
    return out


def _write_block(args):
    """Copy timesteps [start, stop) from H5 into the destination store."""
    dst_path, files, variables, start, stop = args
    dst = zarr.open_group(store=zarr.storage.LocalStore(dst_path), mode="r+")
    n = stop - start
    # Buffer the whole block per variable so each region is written once.
    bufs = {}
    for j, p in enumerate(files[start:stop]):
        with h5py.File(p, "r") as f:
            grp = f["input"]
            for var in variables:
                if var not in bufs:
                    bufs[var] = np.empty((n,) + grp[var].shape, dtype=np.float32)
                bufs[var][j] = grp[var][()]
    for var in variables:
        dst[var][start:stop] = bufs[var]
    return n


def convert(h5_dir: str, dst_path: str, year: int,
            chunk_t: int, chunk_lat: int, chunk_lon: int,
            n_workers: int = 8, block: int = BLOCK_DEFAULT):

    files, variables, (lat_size, lon_size) = scan_year(h5_dir, year)
    n_steps = len(files)

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

    shape = (n_steps, lat_size, lon_size)
    chunks = (chunk_t, chunk_lat, chunk_lon)

    print("Checking / creating array schemas ...")
    for var in variables:
        if var not in dst:
            arr = dst.create_array(
                var,
                shape=shape,
                chunks=chunks,
                dtype=np.float32,
                compressors=ZstdCodec(level=0),
                dimension_names=DIMS,
            )
            # zarr v3 carries dims in dimension_names; _ARRAY_DIMENSIONS is kept
            # as well so v2-style readers still resolve the dimensions.
            arr.attrs["_ARRAY_DIMENSIONS"] = list(DIMS)

    # Time array — taken from the files themselves, not assumed from a calendar,
    # so any gap in the source shows up rather than being silently regularised.
    if "time" not in dst:
        time_strs = read_times(files)
        # Stored as int64 nanoseconds with CF units rather than as strings: a
        # variable-length string array reports a 2-element chunks tuple for a
        # 1-D array, which breaks xarray.open_zarr's dimension mapping.
        times = np.array(time_strs, dtype="datetime64[ns]")
        t = dst.create_array(
            "time",
            shape=(n_steps,),
            chunks=(n_steps,),
            dtype=np.int64,
            compressors=ZstdCodec(level=0),
            dimension_names=("time",),
        )
        t[:] = times.astype(np.int64)
        t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        t.attrs["units"] = "nanoseconds since 1970-01-01"
        t.attrs["calendar"] = "proleptic_gregorian"
        dst.attrs["time_start"] = time_strs[0]
        dst.attrs["time_end"] = time_strs[-1]

    dst.attrs["year"] = year
    dst.attrs["source"] = h5_dir
    dst.attrs["chunks"] = list(chunks)

    blocks = [
        (dst_path, files, variables, s, min(s + block, n_steps))
        for s in range(0, n_steps, block)
    ]
    n_workers = min(n_workers, len(blocks))
    print(f"Converting {n_steps} timesteps in parallel (workers={n_workers}) ...")

    done = 0
    t_total = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for n in pool.map(_write_block, blocks):
            done += n
            elapsed = time.perf_counter() - t_total
            rate = done / elapsed if elapsed else 0
            eta = (n_steps - done) / rate if rate else 0
            print(f"  [{done:4d}/{n_steps}] {rate:5.1f} steps/s  ETA {eta/60:.1f} min",
                  flush=True)

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
    parser.add_argument("--block",     type=int, default=BLOCK_DEFAULT,
                        help="timesteps per worker task")
    args = parser.parse_args()
    convert(args.h5_dir, args.dst, args.year,
            args.chunk_t, args.chunk_lat, args.chunk_lon,
            args.workers, args.block)


if __name__ == "__main__":
    main()
