#!/usr/bin/env python
"""SI sampling loop with FastGroupNorm + torch.compile on the UNet.

    CUDA_VISIBLE_DEVICES=7 python bench_unet_compile.py
"""
import time
import torch
import bench_unet_kernels as b
from bench_unet_sampling import sample_loop

dev = "cuda"
torch.backends.cudnn.benchmark = True

for bs, amp, name in [(8, True, "finetune (4 ICs x 2 members, fp16 autocast)"),
                      (1, False, "inference (1 member, fp32)")]:
    g = torch.Generator(device=dev).manual_seed(0)
    args = (torch.randn(bs, 10, 4050, 480, device=dev, generator=g),
            torch.randn(bs, 10, 4050, 480, device=dev, generator=g),
            torch.randn(bs, 2, 45, 90, device=dev, generator=g))
    unet = b.patch_unet(b.make_unet(), upsample=False).eval()
    cu = torch.compile(unet)
    t = time.perf_counter()
    sample_loop(cu, *args, amp=amp)
    torch.cuda.synchronize()
    print(f"{name}: compile + first sample {time.perf_counter() - t:.1f} s", flush=True)
    t = time.perf_counter()
    for _ in range(3):
        sample_loop(cu, *args, amp=amp)
    torch.cuda.synchronize()
    print(f"  FastGroupNorm + torch.compile  {(time.perf_counter() - t) / 3 * 1e3:9.1f} ms per 14-step SI sample", flush=True)
    del args, unet, cu
    torch.cuda.empty_cache()
