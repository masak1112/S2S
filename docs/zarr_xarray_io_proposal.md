# Proposal: Moving I/O from per-timestep HDF5 to Zarr + xarray

## Why we're looking at this

The current data pipeline (`v2.0/utils/data_loader_multifiles.py`, fed by
`data_utils/netcdf_to_h5_2.py` / `netcdf_to_h5_better.py`) stores each
timestep as its own HDF5 file: `{year}_{idx:04}.h5`, e.g. one file per 6h
step per year, each holding an `input` group of per-variable arrays at
721x1440 resolution.

`GetDataset._get_data()` opens and closes an `h5py.File` on **every**
`__getitem__` call — for `surface_t`, `surface_t_1` (target), and, in
rollout/lead-time modes, one open per boundary/target step. With
`num_data_workers > 1` and `DistributedSampler`, that's many worker
processes independently doing small, uncached opens against a directory of
hundreds of thousands of small files. On our HPC filesystems (Lustre-backed
on Derecho/Stampede3, GPFS-like on Midway) this pattern is known to be slow:
metadata/open overhead dominates, per-file locking limits concurrent reader
scaling, and small-file counts are themselves an operational problem (inode
exhaustion, slow `find`/backup, filesystem team complaints).

Switching the on-disk format to **Zarr**, read through **xarray**, is aimed
at:
- Fewer, larger objects instead of one file per timestep (fewer opens, less
  metadata-server load).
- Chunking aligned to actual access pattern (contiguous time reads for
  training, whole-field reads for inference) instead of one-file-per-step.
- Compression to cut bytes moved (fields are smooth, compress well).
- Native support for concurrent multi-process/multi-worker reads without
  the file-open contention h5py has under `DistributedSampler` + many
  `num_data_workers`.
- Optional lazy/dask-backed access so we don't have to hand-roll index →
  file-path → group → variable logic in `_get_data`/`get_out_path`.

This doc lays out approaches to *try and benchmark*, not a final decision.
Nothing here has been implemented yet — see "Suggested next steps."

## Current shape of the data, for reference

- Grid: 721 x 1440 (lat/lon), pressure levels subset (13 levels available),
  surface + upper-air + boundary + diagnostic variable groups.
- Cadence: 6-hourly, ~1460 steps/year.
- Access pattern in training: read `data_in` at `t`, `data_out` at `t+timedelta`
  for every sample, so consecutive-in-time reads dominate, but
  `DistributedSampler(shuffle=True)` randomizes access order across the
  whole date range — this matters a lot for chunk selection (see below).
- Access pattern in validation/inference (rollout): one input read + a
  sequence of boundary reads out to `max_lead_time`, i.e. genuinely
  sequential-in-time reads.
- Per-file read today is small: one timestep, subset of ~5-18 variables,
  each `(721, 1440)` or `(levels, 721, 1440)`.

## Approaches to evaluate

### 1. Single consolidated Zarr store per split, time as outer chunk dim
Write one Zarr store per split (train/val/test), one array per variable (or
a small number of grouped arrays — e.g. all surface vars stacked on a
`variable` dim, all upper-air vars stacked on `variable`+`level` dims),
chunked as `(chunk_time, ...)` with `chunk_time` on the order of 8-64 steps
tuned against read pattern.

- Pros: closest analog to current per-year h5 files but far fewer chunks
  overall; straightforward `xr.open_zarr(..., consolidated=True)` +
  `.isel(time=...)` reads; plays well with `dask` for parallel prefetch.
- Cons: random-shuffle training access will pull entire chunks for a single
  timestep unless chunk_time is tuned small, which then re-introduces
  many-small-chunks overhead. Need to actually test chunk_time against our
  real sampler (shuffled per-epoch indices), not assume.
- This is the natural "just replace h5py with zarr" option and should be
  the first one benchmarked.

### 2. Zarr v3 with sharding
Same layout as (1), but using Zarr v3's **sharded** storage, which packs
many logical chunks into fewer physical files/objects. This directly
targets the small-file-count problem: you can keep small logical chunks
(good for random single-timestep access) while keeping the *physical*
object count low (good for Lustre/GPFS metadata).

- Pros: decouples "chunk size for read pattern" from "file count for
  filesystem happiness" — probably the best fit for our shuffled,
  single-timestep access pattern on a parallel HPC filesystem.
- Cons: newer feature (needs `zarr-python >= 3`), less battle-tested in the
  climate/weather ML ecosystem than plain Zarr v2; need to confirm current
  xarray/dask versions in `v2.0/environment.yml` and `docker/requirements.txt`
  support it before committing.

### 3. Kerchunk / VirtualiZarr reference layer over existing files
Instead of rewriting data, generate a Zarr-compatible virtual index
(`kerchunk` or the newer `virtualizarr`) pointing at either the *existing*
NetCDF source (`/eagle/.../6h_721_1440_with_poles`) or even the current HDF5
files, so xarray can open them through the zarr/dask machinery without a
full data rewrite.

- Pros: near-zero storage cost, fast to prototype, lets us test whether
  xarray+zarr *access patterns* (lazy dask arrays, `.isel`, chunked reads)
  help even before committing to a full re-conversion.
- Cons: read performance is bounded by the underlying per-file format
  (still opening many small HDF5/NetCDF files under the hood), so it won't
  fix the small-file/open-overhead problem — only useful as a fast way to
  A/B the *API* change, not the *storage* change. Good as a cheap early
  experiment, not a destination.

### 4. Rechunk to larger time windows + reshape access pattern instead of pure format swap
A variant worth calling out: part of our current slowness may be the
per-timestep *access* pattern (opening a file per sample) more than the
format itself. We could pair the Zarr move with a dataloader change that
reads a whole `chunk_time`-sized block per worker fetch and serves
individual timesteps out of an in-process cache, instead of one
open+read per `__getitem__`. This changes `GetDataset` more substantially
(closer to an `IterableDataset` or a block-caching wrapper) but could get
most of the win even before finishing a full Zarr migration, and is
required to actually benefit from any chunk_time > 1 in options 1/2.

- Pros: addresses the real bottleneck (call overhead + no caching) directly;
  works with either h5 or zarr as backing store, so it's a fair baseline to
  benchmark *against* the format switch.
- Cons: more invasive dataloader rewrite; interacts with
  `DistributedSampler(shuffle=True)` — need a chunk-aware or shuffled-block
  sampler so workers don't each end up re-reading the same chunk redundantly
  or thrashing the cache.

### 5. Cloud object store (S3-compatible) + Zarr
Only relevant if we expect to move data off HPC parallel filesystems onto
object storage (e.g. for cloud-bursting or NVIDIA hackathon GPU instances
that don't mount Midway/Derecho/Stampede3 filesystems directly). Same Zarr
layout as (1)/(2), accessed via `fsspec`/`s3fs`.

- Pros: matches the `docker/` containerized workflow already added in this
  branch for cloud/hackathon GPU environments; avoids depending on HPC-only
  mount paths.
- Cons: separate concern from the HPC I/O bottleneck — network bandwidth
  and egress cost become the new constraint; only pursue if we actually
  plan to train off cluster-native storage.

## Suggested comparison matrix to actually run

Benchmark candidates 1, 2, and 4 (3 is a cheap sanity check to run first,
5 only if cloud training is in scope) against the **current h5py loader**
using:

- Wall-clock time per epoch and per-`__getitem__` (the code already has
  `nvtx.range_push`/`logging.debug` timing around `h5_open`/`h5_read` in
  `_get_data` — mirror that instrumentation for the zarr path so the
  comparison is apples-to-apples).
- Behavior under `DistributedSampler(shuffle=True)` specifically, since
  shuffled multi-worker access is our actual training pattern, not
  sequential scan.
- Multi-node scaling (does read throughput hold up as `NUM_TASKS_PER_NODE`
  and node count increase, given `docker/run_apptainer.sh`'s
  `NUM_TASKS_PER_NODE` knob).
- File/object count on disk (operationally relevant — this is what's
  currently drawing complaints from HPC center filesystem teams).
- Storage footprint with a couple of compressor choices (`blosc:zstd`,
  `blosc:lz4`) vs current uncompressed h5.

## Suggested next steps

1. Pick one year of one split (e.g. `val/2019`) as a small testbed.
2. Convert it via options 1 and 2 (and try option 3 as a zero-copy
   baseline) using `xarray.Dataset.to_zarr`.
3. Write a minimal read-benchmark script that mimics `GetDataset._get_data`'s
   access pattern (shuffled single-timestep reads + sequential rollout
   reads) against each store.
4. Compare against current h5 numbers on the same filesystem/node.
5. Only after that, decide whether to invest in a full `GetDataset` rewrite
   (needed regardless of which storage layout wins, to replace
   `h5py.File`/`get_out_path` with `xr.open_zarr`/`.isel`) and a full
   dataset re-conversion pipeline (replacing `netcdf_to_h5_2.py` /
   `netcdf_to_h5_better.py`).

Open question to resolve with the team before committing engineering time:
do we want this to also solve the "move off HPC-mounted paths" problem
(option 5), or is the HPC small-file/open-overhead problem (options 1/2/4)
the whole scope for now?
