# Run guide: postprocess-SI inference with `--feedback corrected`

For the Claude Code session that will run this on a new cluster. Read it all
before launching anything.

## 1. What this run is and why

`inference_postprocess_si.py` rolls out a frozen base model for 45 daily steps
and adds a draw from a trained "postprocess SI" (PP-SI) to every step's output:

```
corrected = member + blend * residual,   residual ~ p(ERA5 - member | member, lead)
```

The `--feedback` flag chooses which state goes into the **next** step:

| | `--feedback base` (already done, run `pp_si_infer_v2`) | `--feedback corrected` (**this run**) |
|---|---|---|
| Next-step input | raw base output `p` | corrected output `c = p + residual` |
| PP-SI changes the trajectory? | no (pure postprocessing) | yes (learned stochastic model-error term in the loop) |

The aim is an ensemble in which the PP-SI is **part of the forecast model**
rather than a postprocessing step, so it can be compared with raw FuXi-S2S /
ECMWF S2S ensembles. Only the flag differs. Everything else must match the
`pp_si_infer_v2` run exactly, so the two runs can be compared.

Code path (no changes needed for run A): `inference_postprocess_si.py`, the
rollout loop at roughly lines 537-572. With `corrected`, `s_cur, ua_cur = c_sfc, c_ua`.
Note: only surface + upper-air go into the next step. Diagnostics
(`total_precipitation_24hr`, etc.) are never inputs, so precip is still
corrected only as an output.

## 2. Exact settings of the reference run (`pp_si_infer_v2`), which must be matched

Taken from its log and launch script on the original machine:

| Item | Value |
|---|---|
| Config | `config/exp16_nvidia_v3.yaml`, `--config=S2S` |
| Base model checkpoint | `results/S2S/rollout_finetune_finetune_rollout_crps/training_checkpoints/rollout_ckpt_1000.tar` (epoch 12, iters 1000) |
| Encoder / det checkpoints | `checkpoint_path_vae_c1` = `checkpoints/c1/ckpt.tar`, `checkpoint_path_det` = `checkpoints/determinstic/ckpt.tar` |
| PP-SI checkpoint | `results/S2S/postprocess_si_pp_si_v2/training_checkpoints/pp_si_ckpt_10800.tar` (96 fields, `max_lead_steps=10`) |
| Members | 20 |
| Steps | 45 (`timedelta_hours: 24`, so day 1 to day 45) |
| Blend | 1.0 |
| Base latent | `si` (default), `latent_cond sst` (default), MC dropout from config |
| `base_temperature` | 1.0 (config) |
| Init dates | `sel_dates: True`: Mondays/Thursdays, May-July, 2019-2024, 00 UTC = 156 dates |
| Expected log line | `residual_std range: 0.01095 .. 0.3982` |

**IMPORTANT: base checkpoint mismatch in the config.** The `S2S` block of
`exp16_nvidia_v3.yaml` now sets `checkpoint_path_finetuned` to the
**SSTCOND** `rollout_ckpt_250.tar`, **not** the `rollout_ckpt_1000.tar` that
the PP-SI was trained against and that `pp_si_infer_v2` used. You **must** pass
`--base_ckpt <path>/rollout_ckpt_1000.tar` explicitly. Then confirm that the log line
`Base model loaded from ... rollout_ckpt_1000.tar (epoch=12, iters=1000)` appears.

## 3. What to copy to the cluster

1. **Code**: the full `v2.0/` tree (`inference_postprocess_si.py`, `train.py`,
   `networks/`, `utils/`, `config/`, `HPC_scripts/`). Python env: torch 2.6
   (cu124 on the original), xarray, cf_xarray, h5py, numpy, wandb (imported
   but disabled at inference).
2. **Checkpoints** (4 files): `c1/ckpt.tar`, `determinstic/ckpt.tar`,
   `rollout_ckpt_1000.tar`, `pp_si_ckpt_10800.tar`.
3. **Data** (`data_dir`, `/data/ERA5` on the original, ~1.9 TB in total): at least the
   2019-2024 h5 files (plus early 2025 if the loader reads targets past
   end-of-July + 45 days), and every normalisation/climatology/statistics file the config
   refers to (`surface_mean`, `*_std`, `1979-2018_*_climatology.nc`, constant
   boundary / land-mask files). Grep the config for `.nc` / `.npy` / `.h5`
   filenames and check that each one exists.
4. **Benchmark** (for scoring): `benchmark-dev/metrics/CRPS/crps_ssr_main.py`
   and its `config.yaml`.

Then edit the paths in the `S2S` block of the config (`data_dir`, `exp_dir`,
`checkpoint_path_vae_c1`, `checkpoint_path_det`, and any statistics paths) for the new
cluster. Do **not** edit any other values.

## 4. Smoke test first

```bash
cd v2.0/HPC_scripts
python -m torch.distributed.launch --master_port=29507 --nproc_per_node=<NGPU> \
    ../inference_postprocess_si.py \
    --yaml_config=../config/exp16_nvidia_v3.yaml --config=S2S \
    --run_num=pp_si_infer_v2_fbcorr_smoke \
    --base_ckpt=<path>/rollout_ckpt_1000.tar \
    --pp_ckpt=<path>/pp_si_ckpt_10800.tar \
    --num_members=4 --inference_steps=45 --blend=1.0 \
    --feedback=corrected \
    --years 2019 --max_dates 2 \
    --save_vars 2m_temperature total_precipitation_24hr geopotential
```

Check in `results/S2S/pp_si_infer_v2_fbcorr_smoke/inference_postprocess_si.log`:
- `feedback=corrected`
- base model loaded from `rollout_ckpt_1000.tar`, PP-SI from `pp_si_ckpt_10800.tar`
- `residual_std range: 0.01095 .. 0.3982`

Then open one output NetCDF and check:
- **No NaN/Inf**, and values stay physically plausible over all 45 days (t2m roughly
  180-340 K, z500 roughly 4.6e4-6.0e4 m²/s², precip >= 0). Feeding the correction back
  can make the run blow up or drift, and this check catches that.
- Print the global-mean t2m and z500 per lead for each member. A steady
  monotonic drift over the 45 days is a red flag. Report it.
- Ensemble spread (std across the 4 members) per lead. Note whether it keeps
  growing without bound.
- Time per member per date, used to estimate the full run's wall time.

## 5. Full runs

Run A is required. Run B is recommended (see section 6).

```bash
python -m torch.distributed.launch --master_port=29507 --nproc_per_node=<NGPU> \
    ../inference_postprocess_si.py \
    --yaml_config=../config/exp16_nvidia_v3.yaml --config=S2S \
    --run_num=pp_si_infer_v2_fbcorr \
    --base_ckpt=<path>/rollout_ckpt_1000.tar \
    --pp_ckpt=<path>/pp_si_ckpt_10800.tar \
    --num_members=20 --inference_steps=45 --blend=1.0 \
    --feedback=corrected \
    --save_vars 2m_temperature total_precipitation_24hr geopotential
```

- Disk: with `--save_vars`, about 0.23 GB per file, so 156 × 20 ≈ **0.7 TB**. Without it,
  about 3.4 TB.
- The job can be resumed: relaunching with the same `--run_num` skips member files
  that already exist. Do not pass `--overwrite` unless you mean to regenerate.
- Members are split across ranks; every rank goes through all 156 dates.
- If there is no raw control run on the original machine, also run one here
  (same command, `--no_correction`, `--run_num=pp_si_infer_v1_base`,
  **`--num_members=20`**, drop `--feedback`). This is the raw-vs-raw baseline.

## 6. Known problem: lead conditioning was never trained beyond lead 1

`pp_si_v2` was trained with `--rollout_steps=1`, so during training the lead
input was always `lead_frac = 1/10 = 0.1`, and the SI learned only the
**1-day** error. At inference, `_apply_correction` passes
`lead_frac = (step+1)/10`, which reaches **4.5** at day 45: far outside training.
(The `feedback base` run has the same problem.)

For `feedback corrected`, the clean interpretation is "add a one-step model error
at every step and carry it forward", so the lead input should stay at its
training value. Add an option for this (run **B**):

In `inference_postprocess_si.py`:

1. `__init__` signature: add `fixed_lead_step: int | None = None`, and store
   `self.fixed_lead_step = fixed_lead_step`.
2. In `_apply_correction`, replace the `lead_frac` numerator:
   ```python
   lead_step = self.fixed_lead_step if self.fixed_lead_step is not None else step + 1
   lead_frac = torch.full((member.shape[0],), lead_step / self.max_lead_steps,
                          device=member.device, dtype=member.dtype)
   ```
3. argparse: `parser.add_argument('--fixed_lead_step', default=None, type=int,
   help='Hold the PP-SI lead conditioning at this step (1 = as trained by pp_si_v2).')`
   and pass `fixed_lead_step=args.fixed_lead_step` to `PostprocessInference(...)`.
4. Add `fixed_lead_step` to the `Base model: ...` logging line so the log records it.

Run B = run A command + `--fixed_lead_step 1`, `--run_num=pp_si_infer_v2_fbcorr_lead1`.
Smoke-test it the same way first.

## 7. Sanity checks before scoring

1. **Day 1 must match `pp_si_infer_v2`.** Seeds are per member
   (`seed = 100000*batch_idx + ens_id`) and step 1 is computed identically in both
   modes, so day-1 fields of `fbcorr` and `v2` should be the same up to GPU
   nondeterminism (cudnn.benchmark is on, so expect agreement to within about 1e-3, not
   bitwise). Days 2+ should differ. If day 1 differs a lot, the base or PP-SI checkpoint
   or the config does not match. Stop and report.
2. No NaN/Inf in any file. File count = 156 × 20 per run.
3. Check `pp_si_ckpt_10800.tar` and `rollout_ckpt_1000.tar` against the original
   (e.g. `md5sum`).

## 8. Scoring

Use `benchmark-dev/metrics/CRPS/crps_ssr_main.py`. Add entries under `models:` in
its config for `pp_si_infer_v2_fbcorr` (and `_lead1`, and the control run) that
point to the new `predictions/` directories, in the same format as the existing
`pp_si_infer_v2` entry.

Score t2m, z500 and total precipitation and report, per lead day (at least
days 1, 3, 5, 7, 10, 14, 21, 28, 35, 42, plus week 3-4 means):
- CRPS
- SSR (spread/RMSE; note whether the finite-ensemble factor sqrt((M+1)/M) is
  applied)
- **RMSE of the ensemble mean**. Feeding the correction back must not make this worse
  than the raw base control. If it does, the run is harming the forecast.

Report three columns side by side: control (raw), `feedback base` (v2),
`feedback corrected` (A, and B if run).

## 9. Evaluation caveat to keep in mind

The benchmark scores the first 4 dates (2019-05-02, 05-06, 05-09, 05-13), which
are also `pp_val_indices` used for validation during PP-SI training. Report
scores both with all 156 dates and with those 4 removed.

## 10. What to report back

- Exact commands run, and the cluster/GPU type.
- The log header lines listed in section 4 for each run.
- The results of the sanity checks in section 7.
- The score table from section 8.
- Any blow-up or drift seen, with the lead day where it starts.
