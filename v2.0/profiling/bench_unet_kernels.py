#!/usr/bin/env python
"""Micro-benchmark for the two kernels that dominate train_diffusion.py.

The nsys profile shows ~80 % of GPU time in
  * RowwiseMomentsCUDAKernel  -- nn.GroupNorm statistics. With batch 1 per GPU
    and 4 groups the kernel launches N*G = 4 blocks on a 132-SM H100.
  * upsample_bilinear2d_backward_out_frame -- bilinear upsample backward
    (atomics), mostly from CrossAttentionConditionFuse's 8x8 -> full-res
    F.interpolate.

This script times one SI training step of the real ConUNet_1degV2 (two UNet
forwards + backward, fp16 autocast, activation checkpointing as in the config)
with the stock modules and with drop-in replacements that compute the same
function:
  * FastGroupNorm: same parameters/state_dict, statistics via torch.var_mean
  * matmul_interpolate: bilinear interpolation as two small matmuls
    (out = A_h @ x @ A_w^T), identical weights to F.interpolate

    CUDA_VISIBLE_DEVICES=7 python bench_unet_kernels.py
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import networks.diffusion as D  # noqa: E402

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"


# ----------------------------------------------------------------------------
# Drop-in replacements
# ----------------------------------------------------------------------------
class FastGroupNorm(nn.GroupNorm):
    """nn.GroupNorm with the statistics computed by torch.var_mean.

    PyTorch's GroupNorm kernel parallelises over N*G rows only; var_mean's
    reduction splits each row across many blocks. Output is float32 like
    GroupNorm under autocast.
    """

    def forward(self, x):
        n, c = x.shape[:2]
        xf = x.float().reshape(n, self.num_groups, -1)
        var, mean = torch.var_mean(xf, dim=-1, keepdim=True, correction=0)
        y = ((xf - mean) * torch.rsqrt(var + self.eps)).reshape(x.shape)
        if self.affine:
            shape = (1, c) + (1,) * (x.dim() - 2)
            y = y * self.weight.reshape(shape) + self.bias.reshape(shape)
        return y


_mat_cache = {}


def _interp_matrix(n_in, n_out, align_corners, device):
    """(n_out, n_in) matrix of 1-D linear interpolation weights, same
    coordinate convention as F.interpolate(mode='bilinear')."""
    key = (n_in, n_out, align_corners, device)
    if key in _mat_cache:
        return _mat_cache[key]
    dst = torch.arange(n_out, dtype=torch.float64, device=device)
    if align_corners:
        src = dst * ((n_in - 1) / (n_out - 1)) if n_out > 1 else torch.zeros_like(dst)
    else:
        src = ((dst + 0.5) * (n_in / n_out) - 0.5).clamp(min=0)
    i0 = src.floor().long().clamp(max=n_in - 1)
    i1 = (i0 + 1).clamp(max=n_in - 1)
    w1 = src - i0
    w0 = 1 - w1
    m = torch.zeros(n_out, n_in, dtype=torch.float64, device=device)
    m.scatter_add_(1, i0[:, None], w0[:, None])
    m.scatter_add_(1, i1[:, None], w1[:, None])
    m = m.float()
    _mat_cache[key] = m
    return m


def matmul_interpolate(x, size, align_corners):
    h, w = x.shape[-2:]
    a_h = _interp_matrix(h, size[0], align_corners, x.device)
    a_w = _interp_matrix(w, size[1], align_corners, x.device)
    x = x.float()
    return torch.matmul(a_h, torch.matmul(x, a_w.t()))


class MatmulUpsample(nn.Module):
    def __init__(self, scale, align_corners):
        super().__init__()
        self.scale, self.align_corners = scale, align_corners

    def forward(self, x):
        h, w = x.shape[-2:]
        return matmul_interpolate(x, (h * self.scale, w * self.scale), self.align_corners)


def patch_unet(unet, groupnorm=True, upsample=True):
    for name, mod in list(unet.named_modules()):
        for cname, child in list(mod.named_children()):
            if groupnorm and type(child) is nn.GroupNorm:
                new = FastGroupNorm(child.num_groups, child.num_channels, child.eps, child.affine)
                new.load_state_dict(child.state_dict())
                setattr(mod, cname, new.to(child.weight.device))
            if upsample and isinstance(child, nn.Upsample):
                setattr(mod, cname, MatmulUpsample(int(child.scale_factor), child.align_corners))
    return unet


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------
def make_unet():
    torch.manual_seed(0)
    return D.ConUNet_1degV2(dim_in=20, dim_cond=2, dim_out=10, c=64, c_mults=(1, 2, 2, 4),
                            scale=[2, 2, 2], resnet_block_groups=4, checkpointing=2,
                            use_reentrant=False, gate_init=0.1).to(dev)


def step(unet, xp, xm, z, ts):
    with torch.autocast("cuda", dtype=torch.float16):
        dp = unet(xp, z, ts)
        dm = unet(xm, z, ts)
        loss = dp.float().pow(2).mean() + dm.float().pow(2).mean()
    loss.backward()
    return loss


def bench(label, unet, inputs, n=5):
    unet.train()
    for _ in range(2):
        step(unet, *inputs)
        unet.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        step(unet, *inputs)
        unet.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n
    print(f"{label:42s} {dt * 1e3:8.1f} ms / training step", flush=True)
    return dt


def main():
    # Shapes as in StochasticInterpolant.training_step with batch 1 per GPU:
    # Pangu latent (1, 10, 45*90, 480), UNet input = cat(path point, latent).
    B, C, H, W = 1, 10, 4050, 480
    g = torch.Generator(device=dev).manual_seed(0)
    xp = torch.randn(B, 2 * C, H, W, device=dev, generator=g)
    xm = torch.randn(B, 2 * C, H, W, device=dev, generator=g)
    z = torch.randn(B, 2, 45, 90, device=dev, generator=g)
    ts = torch.rand(B, device=dev, generator=g)
    inputs = (xp, xm, z, ts)

    # Correctness: replacements vs stock, same weights, fp32, eval (no dropout).
    base = make_unet().eval()
    fast = patch_unet(make_unet().eval())
    fast.load_state_dict(base.state_dict())
    orig_interp = F.interpolate

    def interp_hook(x, size=None, scale_factor=None, mode="nearest", align_corners=None, **kw):
        if mode == "bilinear" and size is not None and x.dim() == 4:
            return matmul_interpolate(x, size, bool(align_corners))
        return orig_interp(x, size=size, scale_factor=scale_factor, mode=mode,
                           align_corners=align_corners, **kw)

    with torch.no_grad():
        ref = base(xp, z, ts)
        D.F.interpolate = interp_hook
        out = fast(xp, z, ts)
        D.F.interpolate = orig_interp
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"max |diff| {(out - ref).abs().max().item():.3e}   rel L2 diff {rel:.3e}  (fp32 forward)")
    del base, fast, ref, out
    torch.cuda.empty_cache()

    t_base = bench("stock UNet", make_unet(), inputs)
    t_gn = bench("+ FastGroupNorm", patch_unet(make_unet(), upsample=False), inputs)
    D.F.interpolate = interp_hook
    t_gn_up = bench("+ FastGroupNorm + matmul interpolation", patch_unet(make_unet()), inputs)
    D.F.interpolate = orig_interp
    print(f"speed-up: GroupNorm alone {t_base / t_gn:.2f}x, both {t_base / t_gn_up:.2f}x")
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
