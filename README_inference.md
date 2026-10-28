# Running Inference (S2S v2.0)


Ensemble sub-seasonal forecasts: a base model (Pangu-style encoder/decoder + latent-space stochastic interpolant, SI) rolled out autoregressively, optionally followed by an output-space **postprocess SI** that corrects ensemble spread.

The current pipeline is `inference_postprocess_si.py`, configured by the multi-variable anomaly-conditioned model (`S2S_MULTICOND` in `v2.0/config/exp16_midway_sstanom.yaml`).

## 1. Code and environment

### Get the code and switch branch

```bash
git clone git@github.com:masak1112/S2S.git
cd S2S
git fetch origin
git checkout uc-s2s_v3_multicond      # first time: git checkout --track origin/uc-s2s_v3_multicond
git branch --show-current             # should print uc-s2s_v3_multicond
```


### Python environment

Use the shared environment where possible:

```bash
# Midway (UChicago RCC)
ml python
conda activate /project/pedramh/bing/env
```

Other clusters (from the main [README.md](README.md)):

```bash
# Derecho (NCAR)
conda activate /glade/work/zand/anaconda/py311
# Stampede3 (TACC)
conda activate /home1/10786/bgong1/stampede3/env
```

Or build your own from the repo (env name `s2s`):

```bash
conda env create -f v2.0/environment.yml --prefix /path/to/myenv
conda activate /path/to/myenv
```

Check that it works on a GPU node:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

The inference does not need a wandb login (`log_to_wandb` is forced off). A multi-GPU node is expected, since the script shards ensemble members across ranks via `torch.distributed`.

### Data

Inference reads the ERA5 `.h5` data (Midway: `/project/pedramh/h5data/h5data`; other clusters are listed in [README.md](README.md)). Set the data paths in the YAML config for your cluster.

## 2. What you need

| Item | Where (default) |
|---|---|
| Config | `v2.0/config/exp16_midway_sstanom.yaml`, config name `S2S_MULTICOND` |
| Base checkpoint (stage 2 rollout-finetuned + latent SI) | `v2.0/HPC_scripts/results/S2S_MULTICOND/rollout_finetune_rollout_multicond/training_checkpoints/rollout_ckpt_1050.tar` |
| Postprocess-SI checkpoint | `/project/pedramh/bing/checkpoints_UC-S2S_V3/loaded/4_postprocess_si/pp_si_ckpt_11400.tar` |
| Conditioning climatology | `v2.0/clim_00z/` (`cond_anom_dir` in the config of checkpoints directories) |
| Input data / normalization stats | paths set in the YAML config |

Checkpoints are set with `--base_ckpt` / `--pp_ckpt`; these override the paths in the YAML.

## 3. Run on SLURM (recommended)

```bash
cd v2.0/HPC_scripts

# Smoke test: 1 init date, 2 members
sbatch --export=ALL,SMOKE=1 midway_infer_multicond_fuxi.sh

# One year of FuXi-S2S init dates (104 dates), 20 members
sbatch --export=ALL,YEARS=2019 midway_infer_multicond_fuxi.sh
sbatch --export=ALL,YEARS=2020 midway_infer_multicond_fuxi.sh
sbatch --export=ALL,YEARS=2021 midway_infer_multicond_fuxi.sh
```

Environment variables the script understands:

| Variable | Default | Meaning |
|---|---|---|
| `YEARS` | `2019 2020 2021` | Init years |
| `MAX_DATES` | `104` | Cap on init dates |
| `NUM_MEMBERS` | `20` | Ensemble size |
| `BASE_CKPT` | stage-2 `rollout_ckpt_1050.tar` | Base checkpoint |
| `PP_CKPT` | `pp_si_ckpt_11400.tar` | Postprocess-SI checkpoint |
| `SST_MODE` | `evolving` | `evolving` or `persistent` (see below) |
| `SMOKE` | unset | If set: 1 date, 2 members, separate `_smoke` output folder |

Related scripts: `midway_infer_multicond_ens20.sh` (single year, saves only the 3 benchmark variables), `midway_infer_multicond_2019.sh`, `midway_infer_sstanom_2019*.sh` (older SST-anomaly model).

## 4. Run directly

Run from `v2.0/HPC_scripts/` (paths in the config are relative to it):

```bash
cd v2.0/HPC_scripts
python -m torch.distributed.launch --master_port=29540 --nproc_per_node=$(nvidia-smi -L | wc -l) \
    ../inference_postprocess_si.py \
    --yaml_config=../config/exp16_midway_sstanom.yaml --config=S2S_MULTICOND \
    --run_num=my_run \
    --base_ckpt=results/S2S_MULTICOND/rollout_finetune_rollout_multicond/training_checkpoints/rollout_ckpt_1050.tar \
    --pp_ckpt=/project/pedramh/bing/checkpoints_UC-S2S_V3/loaded/4_postprocess_si/pp_si_ckpt_11400.tar \
    --init_date_set fuxi_s2s --years 2019 \
    --num_members 20 --inference_steps 45 \
    --blend 1.0 --feedback base --base_latent si --sst_cond_mode evolving
```

### Main options (`inference_postprocess_si.py`)

| Flag | Meaning |
|---|---|
| `--yaml_config`, `--config` | Config file and the section in it (`S2S_MULTICOND`) |
| `--run_num` | Name of the output folder |
| `--base_ckpt`, `--pp_ckpt` | Base-model and postprocess-SI checkpoints |
| `--num_members` | Ensemble size (sharded across GPUs) |
| `--inference_steps` | Number of daily steps (45) |
| `--years`, `--init_date_set fuxi_s2s`, `--max_dates` | Choose init dates (`fuxi_s2s`: 104/yr for 2019–2021; default is Mon/Thu, Jan–Apr) |
| `--blend` | Weight of the postprocess-SI residual (1.0) |
| `--feedback {base,corrected}` | Which field continues the rollout. `base` (default) keeps the postprocess SI as a pure diagnostic |
| `--no_correction` | Base model only (control run) |
| `--base_latent {si,deterministic}` | `deterministic` bypasses the latent SI so all spread comes from the postprocess SI |
| `--latent_cond {sst,off}` | `off` removes SST conditioning from the latent SI |
| `--mc_dropout {on,off}` | Force MC dropout on/off |
| `--sst_cond_mode {persistent,evolving}` | `evolving` rebuilds SST, soil water, z10/z250/z500, OLR anomalies each step from the forecast (matches stage-2 training); `persistent` holds the initial map fixed |
| `--save_vars ...` | Write only these variables (e.g. `2m_temperature total_precipitation_24hr geopotential`) |
| `--overwrite` | Regenerate existing member files (default: skip, so interrupted runs can be resumed by relaunching) |

## 5. Choosing other initial dates

Init dates are not passed as a list on the command line. They come from one of two fixed calendars (all at 00 UTC), then get filtered by `--years` and `--max_dates`. The date logic is in `v2.0/utils/data_loader_multifiles.py` and `_inference_date_indices` in `v2.0/inference_postprocess_si.py`.

| Date set | Flag | Dates | Years available |
|---|---|---|---|
| FuXi-S2S | `--init_date_set fuxi_s2s` | 104 per year, the same calendar days every year (e.g. Jan 3, 6, 10, 13, 17, 20, 24, 27, 31, ...) | 2019, 2020, 2021 |
| Default | (no flag) | Every Monday and Thursday, Jan to Apr (`get_initialization_dates`) | 2019 to 2024 |

Examples (run from `v2.0/HPC_scripts`, with the same other flags as in section 4):

```bash
# FuXi-S2S dates for 2020 only
--init_date_set fuxi_s2s --years 2020

# FuXi-S2S dates for 2020 and 2021
--init_date_set fuxi_s2s --years 2020 2021

# Default Mon/Thu Jan-Apr dates for 2022 (no --init_date_set)
--years 2022

# Only the first 5 dates of that year (quick test)
--years 2022 --max_dates 5
```

With SLURM, `--years` is set through `YEARS`, and `--max_dates` through `MAX_DATES`:

```bash
sbatch --export=ALL,YEARS=2020 midway_infer_multicond_fuxi.sh
sbatch --export=ALL,YEARS=2021,MAX_DATES=10 midway_infer_multicond_fuxi.sh
```

The script uses `--init_date_set fuxi_s2s`, so only 2019 to 2021 work with it. To use 2022 to 2024 or the Mon/Thu dates, copy the script and delete the `--init_date_set fuxi_s2s` line.

On start-up rank 0 logs what was selected, so check it before a long run:

```
Running 104 init dates from 312 sel_dates (years [2020])
  first: 2020-01-03 00:00:00   last: 2020-12-29 00:00:00
```

Notes:
- `--max_dates` keeps the **first** N dates in time order. You cannot pick dates from the middle of the year with it. To forecast a specific date, change the calendar (below).
- Each init date needs ERA5 data for the initial state and for the next 45 days (to verify against). Check that `data_dir` has the years you ask for.
- Output file names carry the init date (`..._2020010300_ens_<k>.nc`), so runs for different dates can share one `--run_num`. Existing files are skipped unless `--overwrite` is given.

### Forecasting a specific date

There is no `--init_dates` flag. To run your own dates, edit the calendar in `v2.0/utils/data_loader_multifiles.py`:

- FuXi-S2S set: change `_FUXI_S2S_DAYS` (month to list of days), for example `7: (4,)` and empty the other months to run only 4 July. Run with `--init_date_set fuxi_s2s`.
- Default set: change `start_month` / `end_month` in `get_initialization_dates`, for example `end_month=12` for Mon/Thu across the whole year.

Keep dates at 00 UTC. Other hours are skipped.

## 6. Output

```
v2.0/HPC_scripts/results/S2S_MULTICOND/<run_num>/
├── predictions/
│   └── pangu_plasim_<run_num>_24h_45step_<YYYYMMDDHH>_ens_<k>.nc   # one file per init date and member
├── inference_postprocess_si.log
└── <config copy>.yaml
```

- Fields are inverse-transformed to physical units.
- **Disk**: all 16 variables is ~1.1 GB per file (~2.3 TB per year at 20 members × 104 dates). Use `--save_vars` to cut this to ~0.23 GB per file.
- Logs from the SLURM scripts go to `v2.0/HPC_scripts/logs/`.

## 7. Other inference scripts

| Script | Purpose |
|---|---|
| `inference.py` | Base model only (`--run_num`, `--yaml_config`); see `v2.0/HPC_scripts/nvidia_brev_infer.sh` |
| `inference_slow_surface_si.py` | Slow-surface SI model; see `nvidia_brev_infer_slow_surface_si.sh` |
| `inference_postprocess_si.py` | Base model + postprocess SI (this guide) |

The `nvidia_brev_*` scripts are the equivalents for the NVIDIA Brev machine (different conda env and config files).

## Troubleshooting

- `ABORT: base checkpoint missing` / `postprocess-SI checkpoint missing`: the checkpoint path is wrong. Set `BASE_CKPT` / `PP_CKPT`.
- Out of GPU memory: reduce members per rank, or keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set (the scripts do this).
- Port already in use: change `--master_port` (the SLURM script offsets it by year).
