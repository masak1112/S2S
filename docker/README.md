# S2S Container

Self-contained container for the S2S forecasting project, built from a plain
CUDA base image (no conda). Runs under Docker locally and converts to
Apptainer/Singularity for HPC systems that don't permit Docker.

Pins mirror the known-good `s2s_env` conda environment: **Python 3.11, torch
2.6.0+cu124**.

## Build

```bash
./docker/build.sh                # docker image (s2s:latest)
./docker/build.sh --sif          # also produce s2s.sif for HPC
./docker/build.sh --sif --mpi    # add mpi4py for data_utils preprocessing
```

## Run

```bash
./docker/run_docker.sh train     # or: infer | shell
./docker/run_apptainer.sh train  # same, via Apptainer
```

Override any path with environment variables:

```bash
SIF=/scratch/$USER/s2s.sif \
DATA_DIR=/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/h5data \
CKPT_DIR=/scratch/$USER/checkpoints \
OUTPUT_DIR=/scratch/$USER/runs \
CONFIG_FILE=/workspace/v2.0/config/exp16_nvidia_v2.yaml \
NUM_TASKS_PER_NODE=4 \
  ./docker/run_apptainer.sh train
```

## Moving to another system

No registry required:

```bash
./docker/build.sh --sif                    # if apptainer is installed locally
scp s2s.sif user@hpc:/scratch/user/        # single portable file
```

Without local Apptainer, `build.sh --sif` writes `s2s-docker.tar` instead; on
the target system run:

```bash
apptainer build s2s.sif docker-archive://s2s-docker.tar
```

## Layout and mounts

| Host | Container | Notes |
|---|---|---|
| `DATA_DIR` | `/data/ERA5` | read-only; matches `data_dir` in the configs |
| `CKPT_DIR` | `/data/bing/S2S/checkpoints` | configs use absolute paths |
| `OUTPUT_DIR` | `/runs` | working dir; `results/`, `spectra_out/`, `gif_out/`, `acc_plots/` |

The source is baked into `/workspace/v2.0` with `PYTHONPATH` set so
`from utils.YParams import YParams` resolves. Set `DEV_MOUNT=true` to
bind-mount your working tree over it instead of rebuilding.

Working directory is `/runs`, not the source tree: the training code writes
`spectra_out/` etc. relative to `cwd` and resolves the relative `exp_dir:
results` there. `/workspace` is root-owned, and Apptainer always runs as the
invoking user, so cwd must be a writable mount.

## Weights & Biases

```bash
WANDB_API_KEY=xxxx ./docker/run_docker.sh train   # online
WANDB_MODE=offline ./docker/run_docker.sh train   # offline, sync later
```

## Notes

- **apex / transformer_engine are omitted.** Every shipped config sets
  `use_transformer_engine: False`, both imports are lazy, and neither package
  is installed in the working conda env. Adding them means a multi-hour source
  build for no current benefit.
- **mpi4py is optional** (`--mpi`), needed only by `data_utils/netcdf_to_h5_*.py`.
- `utils/weighted_acc_rmse.py` fails to import due to a pre-existing
  TorchScript error. It reproduces in the host conda env and is unrelated to
  the container; nothing else imports the module.
- `--shm-size=16g` / `--ipc=host` are set for Docker: the 64MB default
  deadlocks PyTorch DataLoader workers.
