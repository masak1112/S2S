# UC-S2S v2: Nsight Systems profiling report

Profiled on 2026-11-24, branch `bing_uc_s2s_v2_profiling`.

- Machine: 8× H100 80GB.
- Config: `config/exp16_nvidia_v3_full.yaml`.
- Tools: Nsight Systems 2025.5 tracing CUDA, NVTX, cuDNN/cuBLAS and OS runtime,
  plus the micro-benchmarks in this folder.
- How to reproduce: [README.md](README.md).

## Summary

All four scripts are **GPU-bound**: the GPU is busy 95–99.5 % of the time.
They are not data-bound at today's speed. The GPU time goes mostly to two
inefficient PyTorch kernels inside `ConUNet_1degV2`, the UNet used by both the
latent SI and the postprocess SI:

1. **`nn.GroupNorm` statistics** (`RowwiseMomentsCUDAKernel`): 42–87 % of GPU
   time in every script.
   - PyTorch's GroupNorm kernel runs one thread block per (sample, group).
   - With 4 groups and batch 1–8, that is 4–32 blocks on a 132-SM GPU, each
     reducing about 31M values from a 64×4050×480 activation.
   - Each call takes 9–19 ms where it should take well under 1 ms.
2. **Bilinear-upsample backward** (`upsample_bilinear2d_backward_out_frame`):
   26 % of GPU time in `train_diffusion.py`.
   - It mostly comes from `CrossAttentionConditionFuse`, which upsamples an 8×8
     map to 4050×480 with `F.interpolate`.
   - The backward pass accumulates gradients for those 64 input pixels with
     atomic adds, millions of them per pixel.

Both can be replaced by mathematically identical code that keeps the same
parameters, `state_dict` and checkpoints. Measured on the real UNet:

| script | today | main cost | expected with fixes 1–3 below |
|---|---|---|---|
| `train_diffusion.py` (4 GPUs, batch 1 per GPU) | 4.08 s/iter, about 16 h/epoch | GroupNorm 54 %, upsample backward 26 % | about 1.0 s/iter, **~4×** (UNet step measured 3.90 → 0.96 s) |
| `train_finetune_decoder.py` (4 GPUs, 4×2 members) | 21.8 s/iter, plus validation | frozen SI sampling 90 % | about 11–16 s/iter, **1.4–2×** |
| `train_finetune_rollout.py` (5 rollout steps) | 114 s/iter | SI sampling 86 %, `empty_cache` 9 % | about 50–80 s/iter, **1.5–2.3×** |
| `inference_postprocess_si.py` (1 GPU) | 550 s per 45-day member | GroupNorm **86.5 %** | about 2–2.5 min per member, **~4×** (SI sample measured 10.2 → 1.85 s) |

Wall-clock implications of today's numbers:
- One epoch of `train_diffusion.py` takes about 16 h (14,200 iterations × 4.08 s).
- `nvidia_brev_infer_full_rerun.sh` (50 dates × 4 members on 1 GPU) takes
  about 30 h; with fix 1 it would take about 7 h.

## Recommended changes, in priority order

1. **Replace `nn.GroupNorm` in `networks/diffusion.py` with a version that
   computes its statistics with `torch.var_mean`.** This helps all four
   scripts.
   - Same parameters and output; agrees with the stock layer to 2e-6 relative
     in strict fp32.
   - Reference implementation: `FastGroupNorm` in `bench_unet_kernels.py`.
   - Measured gain on SI sampling: 5.5× at batch 1 (inference) and 1.36× at
     batch 8 (finetunes).
   - Measured gain on a training step: 1.95×.
2. **Compute the bilinear interpolations in the UNet as two small matmuls**
   (`out = A_h @ x @ A_w^T`). This affects training.
   - It covers the `F.interpolate` in `CrossAttentionConditionFuse` and the
     `nn.Upsample` in `Upsample_1deg`.
   - Weights are identical to `F.interpolate`; results agree to 4e-7–1e-5
     relative, for values and gradients.
   - Combined with fix 1, a training step goes from 3.90 s to 0.96 s.
   - Reference implementation: `matmul_interpolate` in `bench_unet_kernels.py`.
3. **`torch.compile` the UNet where it is sampled at batch 8** (decoder and
   rollout finetune).
   - Measured: 14.9 s → 9.2 s per 14-step sample after fix 1 (20.3 s stock).
   - Cost: about 75 s of compilation at startup.
   - No gain at batch 1 (inference: 2.2 s vs 1.85 s), so don't compile there.
4. **Rollout finetune: remove the two `torch.cuda.empty_cache()` calls per
   rollout step** (`train_finetune_rollout.py` lines 302 and 402 on this branch).
   - Each call takes about 1 s, about 10 s per iteration (8.9 %).
   - With `expandable_segments` every call unmaps memory that the next
     allocation maps back: 185k `cuMemCreate`/`cuMemRelease` calls in 5
     iterations, and 4.3 % GPU idle.
5. **Rollout finetune data: load only the targets the rollout uses.** This
   becomes the bottleneck after fixes 1–4.
   - The `validate=True` loader reads `forecast_lead_times[-1]` = 45 targets
     plus 45 boundary files per sample, but the rollout uses
     `num_rollout` (5–20) of them.
   - Measured: 8.4–10.3 s per sample on uncached files. With batch 4 and
     `num_data_workers: 1`, that is about 38 s of loading per iteration,
     hidden today only by the 114 s of compute.
   - Each iteration also copies 4.6 GB host→device from non-pinned memory
     (`pin_memory` is off for `validate` loaders; 0.77 s per iteration).
   - Fix: limit lead times to `max_rollout_steps`, raise `num_data_workers`
     to 4 or more, and turn on `pin_memory`.
6. **Decoder finetune: periodic `validate_crps_ssr()` adds about 38 % wall
   time.** This is estimated, not profiled.
   - Every 200 iterations it runs 4 ICs × 4 members × 10 days = 160
     base-model steps at batch 1 in fp32.
   - At the 10.5 s per base step measured in inference, that is about 28 min
     of validation per 73 min of training.
   - It runs on all 4 ranks, though only rank 0 logs.
   - Fix 1 cuts it to roughly 7 min. Running it on rank 0 only (or batching
     the 4 members) saves more.
7. **`train_diffusion.py` gradient sync: use one bucketed all-reduce.**
   - `sync_grads` issues 208 separate `all_reduce` + `div_` calls per
     iteration, 41 ms of CPU time.
   - That's 1 % today but about 4 % after fix 1+2.
   - Fix: flatten into one buffer, or wrap the UNet in DDP.
8. **Inference: build the dataset with `load_targets=False`.**
   - Each init date loads 45 target states (9.6 s) that `predict()` discards.
   - That's 1.8 % today, more after fix 1, and amortised over members.

### Not bottlenecks

- **Data loading in `train_diffusion.py` and decoder finetune:** the main
  process waits about 0.1 ms per batch.
  - After the ~4× speed-up, `train_diffusion` will need about 0.45 s of
    uncached reads per 1 s iteration on 1 worker. Raising
    `num_data_workers` to 2–4 is cheap insurance.
- **Host↔device copies, checkpoint saves, NetCDF writing:** 4.2 s per
  inference member, 0.8 %.
- **The 8 host syncs per training iteration** (`x.std().item()`,
  `assert not isnan().any()`, GradScaler): GPU idle is 0.5 % today.
  - Revisit after the kernel fixes, when iterations are short.

## Other issues found while profiling

These are not speed problems, but they affect what the runs produce.

- **No gradient sync in either finetune script.**
  - In `train_finetune_decoder.py` and `train_finetune_rollout.py`,
    `diff_model` is not DDP-wrapped and nothing all-reduces its gradients.
  - Each of the 4 ranks trains an independent decoder on its own quarter of
    the data, and only rank 0's copy is saved, so 3 of 4 GPUs' work is
    discarded.
  - This is the same problem `sync_grads` fixed in `train_diffusion.py`.
- **The SI UNet samples in train mode during finetuning.**
  - In both finetune scripts the frozen SI UNet stays in `train()` mode while
    it samples: the profile shows its dropout kernels firing on every forward.
  - Inference samples with the UNet in `eval()` mode, so the decoder is
    finetuned on a different member distribution than the one it sees at
    inference.
- **Stale data paths in the launch configs.**
  - `exp16_nvidia_v2.yaml` and `exp16_nvidia_v3.yaml` point `data_dir` at
    `/data/ERA5`, which doesn't exist on this machine.
  - So `nvidia_brev.sh`, `nvidia_brev_finetune_*.sh` and
    `nvidia_brev_train_si_sstcond.sh` won't start here as written.
- **Startup does redundant work.**
  - Every trainer builds the datasets twice: once in `Trainer.__init__`, then
    again in the subclass.
  - `train_diffusion.py` opens an extra DataLoader iterator per epoch
    (`data_iter = iter(...)`; `next(data_iter)`) and throws away its batch.
  - `StochasticInterpolant.prediction()` computes `patchembed2d/3d` and then
    recomputes them inside `_encode_det`.

## Per-script detail

Numbers are for rank 0 during the capture window, after warm-up. The other
ranks' step times agree within 1 %.

### `train_diffusion.py`

Setup: `S2S_SSTCOND` config, 4 GPUs, iterations 10–30.

- **Step time:** 4.084 s, very steady.
- **GPU:** busy 99.5 %.
- **CPU time per range:** `forward` 0.47 s, `backward` 2.82 s, `optimizer`
  0.75 s, `grad_allreduce` 0.04 s, `data_wait` 0.1 ms.
  - Most of the `optimizer` time is the GradScaler's host sync waiting for the
    backward kernels to drain.
- **GPU time:** UNet forward ×2 is 1.48 s per iteration. Backward, run on the
  autograd thread and including the activation-checkpoint recompute, is about
  2.4 s. Pangu encodes are 0.14 s.
- **Kernels:**
  - GroupNorm statistics: 54.2 %, 132 calls per iteration, 16.7 ms each.
  - Bilinear upsample backward: 26.0 %, 8 calls per iteration, 132 ms each.
  - cuDNN NCHW↔NHWC transposes: 2.6 %.

### `train_finetune_decoder.py`

Setup: 4 GPUs, per-GPU batch 4, `crps_ensemble_size` 2, iterations 5–15.

- **Step time:** 21.8 s. `si_sample` is 19.5 s of it (89.5 %); decode, loss,
  backward and optimizer total about 2 s.
- **GPU:** busy 99.8 %.
- **Kernels:**
  - GroupNorm statistics: 41.6 %.
  - NCHW↔NHWC transposes: 8.5 %.
  - The rest is mostly fp32 elementwise kernels (SiLU, dropout, add, copies)
    on 8×64×4050×480 tensors. These are limited by memory bandwidth, which is
    why `torch.compile` fusion helps at this batch size.
- **Per-iteration counts:** 52 host syncs; the per-step CRPS logging alone
  calls `.item()` 5 times.

### `train_finetune_rollout.py`

Setup: 4 GPUs, per-GPU batch 4, 2 members, 5 rollout steps, iterations 3–8.

- **Step time:** 114 s, 22.6 s per rollout step.
- **GPU:** busy 95.7 %.
- **CPU time per rollout step:** `si_sample` 19.6 s, `empty_cache` 2 × 1.0 s,
  `decode_and_loss` 0.31 s, `item_sync` 0.24 s, `backward` 0.15 s,
  `encode` 0.07 s.
- **Precision:** SI sampling runs outside `autocast`, so its kernels run in
  fp32 with TF32 convs. Benchmarked at batch 8: stock 20.4 s, FastGroupNorm
  15.0 s, FastGroupNorm with fp16 autocast 14.9 s.
- **Data:** `data_wait` is 3 ms today. The cost of a rollout sample is in the
  detail under recommendation 5.
- **Scaling:** at 20 rollout steps the iteration is about 4× longer
  (about 450 s).

### `inference_postprocess_si.py`

Setup: as `nvidia_brev_infer_full_rerun.sh` (latent SI + CRPS decoder +
postprocess SI, MC dropout off), 1 GPU, one init date with 1 member.

- **Per member:** 550 s.
  - 45 lead steps × 11.9 s each.
  - Loading the date: 9.6 s.
  - Writing the NetCDF: 4.2 s, of which `to_netcdf` is 3.2 s.
- **Per lead step:**
  - `si_sample`: 9.66 s (79 %)
  - `pp_correction` (20-step postprocess SI): 1.27 s (10 %)
  - `decode`: 0.75 s of CPU time but only 25 ms of GPU work; it is mostly
    waiting for the SI sample
  - device-to-host copy and inverse transform: 0.14 s
- **GPU:** busy 97 %.
- **Kernels:** GroupNorm statistics are 86.5 % of GPU time, 1,164 calls per
  lead step at 8.8 ms average.
- **Batching members:** after fix 1 this does not help (8 members at batch 8:
  14.9 s ≈ 8 × 1.85 s). Only batching *with* `torch.compile` gains a little
  (1.16 s per member-sample).

## Method and caveats

- **What was profiled:** one rank per run (rank 0), inside a
  `cudaProfilerStart`/`Stop` window after warm-up. The window sizes are in
  [README.md](README.md).
  - Profiling overhead is negligible: rank 0 under nsys ran at 4.084 s per
    iteration vs 4.08 s for the unprofiled ranks.
- **Weights:** the SI UNet weights in the diffusion and decoder runs were
  freshly initialised, because the referenced UNet checkpoints are absent or
  misspelled (`checkpoint_paht_diff`). Kernel timings do not depend on the
  weights.
- **Storage and page cache:** `/data` is a network persistent disk.
  - Uncached reads: about 240 MB/s per stream, about 250 ms per 26 MB sample
    file through h5py.
  - Cached reads: about 19 ms per file.
  - The 1.7 TB dataset is partly in the 1.8 TB page cache, so loader timings
    depend on what was read recently. The per-sample loader numbers above
    were measured on uncached files.
- **What wasn't available:** CPU/Python sampling (needs
  `kernel.perf_event_paranoid` ≤ 2) and GPU hardware metrics (need admin
  profiling permissions).
  - Worker-side NVTX ranges were not captured inside the
    `cudaProfilerApi` window, so loader costs were measured directly instead.
- **Estimates:** the speed-ups in the summary table combine measured profile
  times with measured micro-benchmarks of the changed parts. They are
  estimates until the fixes are applied and re-profiled. The validation cost
  in recommendation 6 is an estimate from the measured base-step time.
- **Concurrency:** the inference and rollout profiles ran at the same time on
  different GPUs (4 and 0–3).
