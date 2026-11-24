#!/usr/bin/env python
"""Time the SI sampling loop (14 UNet forwards, no grad) as used by
train_finetune_decoder.py / train_finetune_rollout.py (batch 4 x 2 members,
fp16 autocast) and inference_postprocess_si.py (batch 1, fp32), with the
stock UNet and with FastGroupNorm (+ channels_last).

    CUDA_VISIBLE_DEVICES=7 python bench_unet_sampling.py
"""
import time

import torch

import bench_unet_kernels as b

dev = "cuda"
torch.backends.cudnn.benchmark = True


def sample_loop(unet, x0, xp, z, n_steps=14, amp=True):
    ts = torch.linspace(0, 1, n_steps + 1, device=dev)
    x = x0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for i in range(n_steps):
            t = ts[i].expand(x.shape[0])
            drift = unet(torch.cat([x, xp], dim=1), z, t)
            x = x + (ts[i + 1] - ts[i]) * drift.float()
    return x


def bench(label, unet, args, amp, n=3, channels_last=False):
    unet.eval()
    if channels_last:
        unet = unet.to(memory_format=torch.channels_last)
        args = tuple(a.contiguous(memory_format=torch.channels_last) if a.dim() == 4 else a
                     for a in args)
    sample_loop(unet, *args, amp=amp)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        sample_loop(unet, *args, amp=amp)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n
    print(f"  {label:40s} {dt * 1e3:9.1f} ms per 14-step SI sample", flush=True)
    return dt


def main():
    for bs, amp, name in [(8, True, "finetune (4 ICs x 2 members, fp16 autocast)"),
                          (1, False, "inference (1 member, fp32)")]:
        print(name)
        g = torch.Generator(device=dev).manual_seed(0)
        x0 = torch.randn(bs, 10, 4050, 480, device=dev, generator=g)
        xp = torch.randn(bs, 10, 4050, 480, device=dev, generator=g)
        z = torch.randn(bs, 2, 45, 90, device=dev, generator=g)
        args = (x0, xp, z)
        t0 = bench("stock UNet", b.make_unet(), args, amp)
        t1 = bench("FastGroupNorm", b.patch_unet(b.make_unet(), upsample=False), args, amp)
        t2 = bench("FastGroupNorm + channels_last", b.patch_unet(b.make_unet(), upsample=False),
                   args, amp, channels_last=True)
        print(f"  speed-up {t0 / t1:.2f}x (GroupNorm), {t0 / t2:.2f}x (+ channels_last)")
        del x0, xp, z, args
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
