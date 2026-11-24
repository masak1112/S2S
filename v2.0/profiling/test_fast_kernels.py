#!/usr/bin/env python
"""Check networks/diffusion.py's fast kernels against the stock PyTorch ops.

    CUDA_VISIBLE_DEVICES=4 python test_fast_kernels.py

1. FastGroupNorm vs nn.GroupNorm: output and gradients (x, weight, bias),
   strict fp32 and fp16 autocast, contiguous and sliced inputs.
2. interpolate_bilinear vs F.interpolate: values and gradients.
3. ConUNet_1degV2 with the trained latent-SI weights (checkpoint_path_finetuned):
   stock vs fast forward in fp32 eval, and one SI training step (fp16
   autocast, two forwards + backward) compared on loss and gradients.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import networks.diffusion as D  # noqa: E402

dev = "cuda"
CKPT = "/data/bing/checkpoints_UC-S2S_V2/loaded/3_base_full_state_rollout_finetuned/rollout_ckpt_1000.tar"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def check(name, value, tol):
    ok = value <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:58s} {value:.2e} (tol {tol:.0e})")
    return ok


def strict_fp32(on):
    torch.backends.cuda.matmul.allow_tf32 = not on
    torch.backends.cudnn.allow_tf32 = not on


def test_groupnorm():
    print("FastGroupNorm vs nn.GroupNorm")
    strict_fp32(True)
    ok = True
    g = torch.Generator(device=dev).manual_seed(0)
    for groups, ch, shape in [(4, 64, (2, 64, 405, 48)), (1, 64, (1, 64, 4050, 480)),
                              (4, 128, (8, 128, 203, 24))]:
        ref = torch.nn.GroupNorm(groups, ch).to(dev)
        torch.nn.init.normal_(ref.weight, generator=None)
        torch.nn.init.normal_(ref.bias)
        fast = D.FastGroupNorm(groups, ch).to(dev)
        fast.load_state_dict(ref.state_dict())
        base = torch.randn(shape[0], shape[1], shape[2] + 3, shape[3], device=dev, generator=g) * 3 + 1
        for sliced in (False, True):
            x0 = base[:, :, :shape[2]] if sliced else base[:, :, 3:].contiguous()
            for amp in (False, True):
                xa = x0.clone().requires_grad_(True)
                xb = x0.clone().requires_grad_(True)
                gy = torch.randn(shape, device=dev, generator=g)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    inp_a = xa.half() if amp else xa
                    inp_b = xb.half() if amp else xb
                    ya, yb = ref(inp_a), fast(inp_b)
                tag = f"G={groups} {tuple(shape)} {'sliced' if sliced else 'contig'} {'fp16-amp' if amp else 'fp32'}"
                if amp:
                    ok &= check(f"out  {tag}", rel(yb, ya), 1e-4)
                else:
                    # Judge both against fp64: on huge groups the stock kernel's
                    # single-block fp32 sum is the less accurate of the two.
                    exact = F.group_norm(x0.double(), groups, ref.weight.double(),
                                         ref.bias.double(), ref.eps)
                    e_stock, e_fast = rel(ya, exact), rel(yb, exact)
                    ok &= check(f"out  {tag} (err vs fp64; stock {e_stock:.1e})",
                                e_fast, max(1e-5, e_stock))
                ok &= (ya.dtype == yb.dtype)
                ref.zero_grad(); fast.zero_grad()
                (ya.float() * gy).sum().backward()
                (yb.float() * gy).sum().backward()
                ok &= check(f"dx   {tag}", rel(xb.grad, xa.grad), 1e-4 if not amp else 5e-3)
                ok &= check(f"dw   {tag}", rel(fast.weight.grad, ref.weight.grad), 1e-4)
                ok &= check(f"db   {tag}", rel(fast.bias.grad, ref.bias.grad), 1e-4)
    return ok


def test_interp():
    print("interpolate_bilinear vs F.interpolate")
    ok = True
    g = torch.Generator(device=dev).manual_seed(1)
    for strict in (True, False):
        strict_fp32(strict)
        for shape, size, ac in [((1, 64, 8, 8), (4050, 480), False),
                                ((1, 256, 8, 8), (507, 60), False),
                                ((1, 256, 507, 60), (1014, 120), True),
                                ((1, 128, 1013, 120), (2026, 240), True)]:
            x = torch.randn(shape, device=dev, generator=g)
            xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
            ya = F.interpolate(xa, size=size, mode="bilinear", align_corners=ac)
            yb = D.interpolate_bilinear(xb, size, ac)
            gy = torch.randn_like(ya)
            (ya * gy).sum().backward()
            (yb * gy).sum().backward()
            tag = f"{tuple(shape)}->{size} ac={ac} {'fp32' if strict else 'tf32'}"
            if strict:
                # fp64 reference: stock F.interpolate computes source
                # coordinates in fp32, which is its own error source.
                a_h = D._interp_matrix(shape[2], size[0], ac, dev).double()
                a_w = D._interp_matrix(shape[3], size[1], ac, dev).double()
                exact = a_h @ (x.double() @ a_w.t())
                e_stock, e_fast = rel(ya, exact), rel(yb, exact)
                ok &= check(f"out  {tag} (err vs fp64; stock {e_stock:.1e})", e_fast, max(2e-5, e_stock))
                ok &= check(f"grad {tag} (vs stock)", rel(xb.grad, xa.grad), max(2e-5, 2 * e_stock))
            else:
                ok &= check(f"out  {tag} (vs stock)", rel(yb, ya), 2e-3)
                ok &= check(f"grad {tag} (vs stock)", rel(xb.grad, xa.grad), 2e-3)
    return ok


def make_unet(fast):
    D.FAST_KERNELS = fast
    torch.manual_seed(0)
    unet = D.ConUNet_1degV2(dim_in=20, dim_cond=2, dim_out=10, c=64, c_mults=(1, 2, 2, 4),
                            scale=[2, 2, 2], resnet_block_groups=4, checkpointing=2,
                            use_reentrant=False, gate_init=0.1).to(dev)
    state = torch.load(CKPT, map_location=dev, weights_only=False)["model_state"]
    state = {k[len("unet."):]: v for k, v in state.items() if k.startswith("unet.")}
    unet.load_state_dict(state, strict=True)
    return unet


def test_unet():
    print(f"ConUNet_1degV2 with trained weights ({os.path.basename(CKPT)})")
    ok = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator(device=dev).manual_seed(2)
    xp = torch.randn(1, 20, 4050, 480, device=dev, generator=g)
    xm = torch.randn(1, 20, 4050, 480, device=dev, generator=g)
    z = torch.randn(1, 2, 45, 90, device=dev, generator=g)
    ts = torch.rand(1, device=dev, generator=g)

    stock, fast = make_unet(False).eval(), make_unet(True).eval()
    n_fast = sum(isinstance(m, D.FastGroupNorm) for m in fast.modules())
    print(f"  fast UNet has {n_fast} FastGroupNorm modules, stock has "
          f"{sum(isinstance(m, D.FastGroupNorm) for m in stock.modules())}")

    strict_fp32(True)
    with torch.no_grad():
        D.FAST_KERNELS = False
        a = stock(xp, z, ts)
        D.FAST_KERNELS = True
        b = fast(xp, z, ts)
    ok &= check("forward, fp32 eval (sampling / inference path)", rel(b, a), 1e-4)

    # Training step as in StochasticInterpolant.training_step, with the
    # codebase's precision settings (TF32 on, fp16 autocast). eval() turns
    # dropout off so both models see the same function.
    strict_fp32(False)
    stock.train(); fast.train()
    for m in list(stock.modules()) + list(fast.modules()):
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    def train_step(unet, flag):
        D.FAST_KERNELS = flag
        unet.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = unet(xp, z, ts).float().pow(2).mean() + unet(xm, z, ts).float().pow(2).mean()
        loss.backward()
        return loss.detach(), torch.cat([p.grad.flatten() for p in unet.parameters()])

    la, ga = train_step(stock, False)
    la2, ga2 = train_step(stock, False)
    lb, gb = train_step(fast, True)
    noise = rel(ga2, ga)
    print(f"  stock vs stock (run-to-run noise): loss {rel(la2, la):.2e}, grads {noise:.2e}")
    ok &= check("training-step loss, fp16 autocast", rel(lb, la), 1e-3)
    ok &= check("training-step gradients (all params), fp16 autocast", rel(gb, ga), max(1e-2, 10 * noise))
    return ok


if __name__ == "__main__":
    results = [test_groupnorm(), test_interp(), test_unet()]
    print("ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
