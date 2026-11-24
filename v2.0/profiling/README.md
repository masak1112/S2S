# Nsight Systems profiling of UC-S2S v2

Tools to profile the training and inference scripts with Nsight Systems
(`nsys`) and summarise the reports. Findings: [PROFILING_REPORT.md](PROFILING_REPORT.md).

## What was added to the code

All of this is inactive unless the environment variables below are set, so normal
runs are unchanged.

- `utils/profiling.py`
  - NVTX ranges: `nvtx_range`, `nvtx_push`/`nvtx_pop`, `nvtx_wrap`
  - `timed_iter`, which marks the time spent waiting on the DataLoader
  - `ProfileWindow`, which opens and closes the capture with
    `cudaProfilerStart`/`cudaProfilerStop` at chosen step counts, prints the
    wall time of each step, and exits.
- NVTX ranges in:
  - `train_diffusion.py`, `train_finetune_decoder.py` and
    `train_finetune_rollout.py`: `data_wait`, `h2d`, `forward`, `backward`,
    `grad_allreduce`, `optimizer`, `rollout_step`, `empty_cache`,
    `checkpoint_save`
  - `inference_postprocess_si.py`: `load_date_sample`, `member`, `lead_step`,
    `base_step`, `pp_correction`, `d2h_inv_transform`, `save_prediction`,
    `to_netcdf`
  - `networks/stochastic_interpolant.py`: `encode*`, `unet_fwd_x2`,
    `si_sample`, `decode`, `crps_loss`
  - `utils/data_loader_multifiles.py`: `dataset_getitem` and `h5_read`, which
    run in the DataLoader worker processes

| variable | meaning |
|---|---|
| `S2S_NVTX=1` | emit the NVTX ranges |
| `S2S_PROF_START=N`, `S2S_PROF_STOP=M` | capture steps `[N, M)` (iterations for training, init dates for inference) |
| `S2S_PROF_EXIT=0` | keep running after the window closes (default: exit) |

## Running

The main launcher is `HPC_scripts/nvidia_brev_nsys_profile.sh`. It runs on
GPUs 4–7 and puts training and validation in one report per script; see the
comments at its top. One-page summary for sharing:
`UC-S2S_v2_Nsight_Profiling_Report.docx`.

```bash
cd v2.0/HPC_scripts
./nvidia_brev_nsys_profile.sh diffusion|decoder|rollout|inference|inference4|all
```

`run_profile.sh` in this folder is the first-round launcher. It produced the
stock-kernel reports on GPUs 0–3, training only:

```bash
cd v2.0/profiling
./run_profile.sh diffusion     # 4 GPUs, S2S_SSTCOND, iterations 10-30
./run_profile.sh decoder       # 4 GPUs, iterations 5-15
./run_profile.sh rollout       # 4 GPUs, 5 rollout steps, iterations 3-8
GPUS=4 NUM_MEMBERS=1 ./run_profile.sh inference   # 1 GPU, init date 1 of 3
python analyze.py reports/diffusion_rank0.nsys-rep
```

- `nsys_rank.sh` wraps each rank for `torchrun --no-python` and profiles only
  the ranks in `NSYS_RANKS` (default `0`).
- Runs use `config/exp16_nvidia_v3_full.yaml`, the config whose data and
  checkpoint paths exist on this machine.
- wandb is disabled.
- Nothing is checkpointed, because each process exits when the window closes.
- Logs, hyperparameters and predictions go under this folder (git-ignored).

Open a report in the Nsight Systems GUI to see the timeline:
`nsys-ui reports/<name>_rank0.nsys-rep`.

Not available on this VM without root: CPU/Python sampling, because
`kernel.perf_event_paranoid=4`, and GPU hardware metrics, because profiling
counters are restricted to admin users.

## Micro-benchmarks

- `bench_unet_kernels.py`: one SI training step of `ConUNet_1degV2`, stock vs
  `FastGroupNorm` vs `FastGroupNorm` + matmul interpolation, with a check that
  the outputs agree.
- `bench_unet_sampling.py`: the 14-step SI sampling loop at the finetune shape
  (batch 8, fp16) and the inference shape (batch 1, fp32).
- `bench_unet_compile.py`: the same loop with `torch.compile`.
