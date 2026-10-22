"""Train the SST-anomaly VAE (single GPU).

The VAE encodes  (sst - daily climatology) / sst_anom_scale  over ocean points
(see networks/vae.py, VAE.prepare_input). The resulting latent is what the
diffusion/SI UNet cross-attends to.

Required config keys (in addition to the usual data keys; data are read by utils/sst_loader.py):
    sst_anomaly: True
    sst_clim_path: ".../clim_00z_sst_std.npy"   # from utils/make_sst_anom_clim.py
    sst_anom_scale: <float>                     # from --estimate_scale
    component, base_channels, z_channels, vae_loss_weight

Usage:
    python train_vae_anomaly.py --yaml_config config/exp17_anom.yaml --estimate_scale 200
    python train_vae_anomaly.py --yaml_config config/exp17_anom.yaml --run_num vae_anom --epochs 20
"""
import os
import argparse
import torch
import wandb

from utils.YParams import YParams
from utils.sst_loader import get_sst_loader
from networks.vae import VAE


def masked_recon_loss(x_recon, x, valid):
    valid = valid.to(x.dtype)
    return ((x_recon - x) ** 2 * valid).sum() / valid.sum().clamp(min=1)


def kl_loss(mean, logvar):
    return -0.5 * torch.mean(1 + logvar - mean ** 2 - logvar.exp())


def make_loader(params, num_workers, train=True):
    """SST-only loader (utils/sst_loader.py): (sst_t, sst_t, date); val years just change the range."""
    return get_sst_loader(
        params,
        year_start=params.train_year_start if train else params.val_year_start,
        year_end=params.train_year_end if train else params.val_year_end,
        lead_days=0, num_workers=num_workers, shuffle=True,
    )


@torch.no_grad()
def estimate_scale(model, loader, device, n_batches):
    """Std of the raw (scale=1) masked anomaly over ocean points -> set as sst_anom_scale."""
    model.anom_scale = 1.0
    s = s2 = n = 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x, valid = model.prepare_input(batch[0].to(device), batch[-1].to(device))
        v = valid.to(x.dtype)
        s += (x * v).sum().item()
        s2 += (x ** 2 * v).sum().item()
        n += v.sum().item()
    mean = s / n
    std = (s2 / n - mean ** 2) ** 0.5
    print(f"SST anomaly (standardized units): mean={mean:.4f} std={std:.4f}  ->  set sst_anom_scale: {std:.4f}")


@torch.no_grad()
def validate(model, loader, device, kl_w, n_batches=50):
    model.eval()
    tot_rec = tot_kl = 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        surface, date = batch[0].to(device), batch[-1].to(device)
        x, valid = model.prepare_input(surface, date)
        z, mean, logvar = model.encode(x)
        rec = model.decode(z, target_shape=x.shape[-2:])
        tot_rec += masked_recon_loss(rec, x, valid).item()
        tot_kl += kl_loss(mean, logvar).item()
    n = max(1, min(n_batches, i + 1))
    model.train()
    return tot_rec / n, tot_kl / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True)
    parser.add_argument("--config", default="S2S")
    parser.add_argument("--run_num", default="vae_anom")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--init_ckpt", default=None,
                        help="Optional ckpt.tar to warm start the model weights from (optimizer/epoch restart)")
    parser.add_argument("--num_workers", default=8, type=int, help="DataLoader workers (match --cpus-per-task)")
    parser.add_argument("--estimate_scale", default=0, type=int,
                        help="Only estimate sst_anom_scale from this many batches, then exit")
    parser.add_argument("--raw", action="store_true",
                        help="Train on the raw standardized SST instead of the anomaly "
                             "(no climatology needed; --estimate_scale does not apply)")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['sst_anomaly'] = not args.raw
    if args.raw and args.estimate_scale > 0:
        parser.error("--estimate_scale only applies to anomaly mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(params).to(device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device)['model_state'])
        print(f"warm start from {args.init_ckpt}")
    loader = make_loader(params, args.num_workers, train=True)

    if args.estimate_scale > 0:
        estimate_scale(model, loader, device, args.estimate_scale)
        return

    val_loader = make_loader(params, args.num_workers, train=False)
    kl_w = float(getattr(params, 'vae_loss_weight', 0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(params, 'vae_lr', 1e-4)))
    run_num = f"{args.run_num}_raw" if args.raw and args.run_num == "vae_anom" else args.run_num
    out_dir = os.path.join(getattr(params, 'exp_dir', 'results'), 'S2S', f'vae_{run_num}')
    os.makedirs(out_dir, exist_ok=True)

    use_wandb = bool(getattr(params, 'log_to_wandb', False))
    if use_wandb:
        if getattr(params, 'wandb_offline', False):
            os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project=getattr(params, 'project', 'Pangu-S2S'),
                   entity=getattr(params, 'entity', None),
                   group=getattr(params, 'group', None),
                   name=f"vae_{run_num}", config=params.params if hasattr(params, 'params') else None,
                   dir=out_dir)
        wandb.define_metric('iters')
        wandb.define_metric('train/*', step_metric='iters')
        wandb.define_metric('val/*', step_metric='epoch')

    model.train()
    iters = 0
    for epoch in range(args.epochs):
        for it, batch in enumerate(loader):
            surface, date = batch[0].to(device), batch[-1].to(device)
            x, valid = model.prepare_input(surface, date)
            z, mean, logvar = model.encode(x)
            rec = model.decode(z, target_shape=x.shape[-2:])
            loss_rec = masked_recon_loss(rec, x, valid)
            loss_kl = kl_loss(mean, logvar)
            loss = loss_rec + kl_w * loss_kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            iters += 1
            if it % 100 == 0:
                print(f"ep {epoch} it {it}  loss {loss.item():.4f}  rec {loss_rec.item():.4f}  kl {loss_kl.item():.4f}", flush=True)
                if use_wandb:
                    wandb.log({'iters': iters, 'epoch': epoch, 'train/loss': loss.item(),
                               'train/rec': loss_rec.item(), 'train/kl': loss_kl.item()})
        rec_v, kl_v = validate(model, val_loader, device, kl_w)
        print(f"== epoch {epoch}  val rec {rec_v:.4f}  val kl {kl_v:.4f}", flush=True)
        if use_wandb:
            wandb.log({'iters': iters, 'epoch': epoch, 'val/rec': rec_v, 'val/kl': kl_v,
                       'val/loss': rec_v + kl_w * kl_v})
        torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'iters': it},
                   os.path.join(out_dir, 'ckpt.tar'))
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
