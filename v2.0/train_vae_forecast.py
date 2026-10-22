"""Train a simple encoder-decoder that forecasts SST at t+LEAD days (single GPU).

Same VAE as train_vae_anomaly.py (networks/vae.py), but the decoder is trained
to reconstruct the SST anomaly at t+lead_days from the encoding of the SST
anomaly at t, instead of reconstructing the input. The encoder latent is then
a forecast-aware SST representation that the latent SI / diffusion UNet can
cross-attend to exactly like the reconstruction VAE (same class, same
prepare_input/encode API; point checkpoint_path_vae_c1-style loading at this
run's ckpt.tar, or use VAE.encoder weights directly).

Anomaly = (sst - daily climatology) / sst_anom_scale, computed with the
climatology of the *input* date for x(t) and of the *target* date for y(t+lead).
Ocean mask is the target-date mask, minus sea-ice points (daily climatological SST < --ice_thresh_k,
default 271.5 K, at the input or target date); the loss is a cos(lat)-weighted masked MSE over the remaining points
(validation MSE and the persistence baseline use the same weights).

Validation also reports the persistence baseline (predict anomaly(t) as
anomaly(t+lead)); the model should beat it, otherwise the latent carries no
forecast information beyond the current state.

Config: reuse config/exp17_anom.yaml (sst_anomaly, sst_clim_path, sst_anom_scale,
base_channels, z_channels, vae_loss_weight, batch_size, data dir / mean-std files).
Data come from utils/sst_loader.py (SST only; --lead_days sets the target offset).

Usage:
    python train_vae_forecast.py --yaml_config config/exp17_anom.yaml --lead_days 3 --run_num vae_fc3d --epochs 20
"""
import os
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import torch
import wandb

from utils.YParams import YParams
from utils.sst_loader import get_sst_loader
from networks.vae import VAE


def lat_weights(lat_deg, device):
    """cos(lat) weights normalised to mean 1 over latitude (same as utils/losses.py), shape [1,1,H,1]."""
    w = torch.cos(torch.deg2rad(torch.as_tensor(lat_deg, dtype=torch.float32, device=device)))
    return (w.numel() * w / w.sum()).view(1, 1, -1, 1)


def masked_mse(pred, target, valid, w=None):
    """Latitude-weighted masked MSE: sum(w*valid*err^2) / sum(w*valid). w=None -> unweighted."""
    weight = valid.to(pred.dtype)
    if w is not None:
        weight = weight * w
    return ((pred - target) ** 2 * weight).sum() / weight.sum().clamp(min=1e-8)


def kl_loss(mean, logvar):
    return -0.5 * torch.mean(1 + logvar - mean ** 2 - logvar.exp())


def shift_date(date, days):
    """date: [B, >=4] (year, month, day, hour) -> same layout, `days` later."""
    out = date.clone()
    for i, (y, m, d, h) in enumerate(date[:, :4].tolist()):
        t = datetime(int(y), int(m), int(d), int(h)) + timedelta(days=days)
        out[i, :4] = torch.tensor([t.year, t.month, t.day, t.hour], dtype=date.dtype)
    return out


def make_loader(params, lead_days, num_workers, train=True):
    """SST-only loader (utils/sst_loader.py): (sst_t, sst_t+lead, date). Val years just change the range."""
    return get_sst_loader(
        params,
        year_start=params.train_year_start if train else params.val_year_start,
        year_end=params.train_year_end if train else params.val_year_end,
        lead_days=lead_days, num_workers=num_workers, shuffle=True,
    )


def make_ice_mask(clim_raw_path, thresh_k, device):
    """[366,H,W] bool, True where the daily Kelvin climatology is below thresh_k (sea-ice / near-freezing)."""
    clim = np.load(clim_raw_path)
    return torch.from_numpy(clim < thresh_k).to(device)   # NaN (land) -> False


def forecast_step(model, batch, device, lead_days, ice=None):
    """Returns (pred, target, valid, mean, logvar, persistence) in anomaly units.

    valid = target-date ocean mask, minus climatological sea-ice points if `ice` is given.
    """
    surface_in, surface_tgt, date = batch[0].to(device), batch[1].to(device), batch[-1].to(device)
    x, _ = model.prepare_input(surface_in, date)
    date_tgt = shift_date(date, lead_days)
    y, valid = model.prepare_input(surface_tgt, date_tgt)
    if ice is not None:
        # mask if ice at either the input or the target date (ice edge moves over the lead)
        valid = valid & ~ice[model._doy_index(date_tgt)].unsqueeze(1) & ~ice[model._doy_index(date)].unsqueeze(1)
    z, mean, logvar = model.encode(x)
    pred = model.decode(z, target_shape=y.shape[-2:])
    return pred, y, valid, mean, logvar, x


@torch.no_grad()
def validate(model, loader, device, lead_days, ice=None, w=None, n_batches=50):
    model.eval()
    mse = kl = pers = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        pred, y, valid, mean, logvar, x = forecast_step(model, batch, device, lead_days, ice)
        mse += masked_mse(pred, y, valid, w).item()
        pers += masked_mse(x, y, valid, w).item()
        kl += kl_loss(mean, logvar).item()
        n += 1
    model.train()
    n = max(n, 1)
    return mse / n, kl / n, pers / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True)
    parser.add_argument("--config", default="S2S")
    parser.add_argument("--run_num", default="vae_fc3d")
    parser.add_argument("--lead_days", default=3, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--num_workers", default=8, type=int, help="DataLoader workers (match --cpus-per-task)")
    parser.add_argument("--save_every", default=1000, type=int,
                        help="Also save ckpt_<iters>.tar every N iterations (0 disables)")
    parser.add_argument("--init_ckpt", default=None,
                        help="Optional VAE ckpt.tar (e.g. the anomaly-reconstruction VAE) to warm start from")
    parser.add_argument("--clim_raw", default="clim_00z/clim_00z_sst.npy",
                        help="[366,H,W] daily SST climatology in Kelvin (NaN on land) used for the ice mask")
    parser.add_argument("--ice_thresh_k", default=271.5, type=float,
                        help="Exclude points whose daily climatological SST is below this (K); <=0 disables")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['sst_anomaly'] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(params).to(device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device)['model_state'])
        print(f"warm start from {args.init_ckpt}")
    ice = make_ice_mask(args.clim_raw, args.ice_thresh_k, device) if args.ice_thresh_k > 0 else None
    if ice is not None:
        print(f"ice mask (clim < {args.ice_thresh_k} K): {ice.float().mean().item():.3%} of points on average, "
              f"{ice.float().mean((1, 2)).max().item():.3%} on the iciest day")
    w = lat_weights(params.lat, device)   # cos(lat) area weighting for the forecast loss and validation
    loader = make_loader(params, args.lead_days, args.num_workers, train=True)
    val_loader = make_loader(params, args.lead_days, args.num_workers, train=False)

    kl_w = float(getattr(params, 'vae_loss_weight', 0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(params, 'vae_lr', 1e-4)))
    out_dir = os.path.join(getattr(params, 'exp_dir', 'results'), 'S2S', f'vae_{args.run_num}')
    os.makedirs(out_dir, exist_ok=True)

    use_wandb = bool(getattr(params, 'log_to_wandb', False))
    if use_wandb:
        if getattr(params, 'wandb_offline', False):
            os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project=getattr(params, 'project', 'Pangu-S2S'),
                   entity=getattr(params, 'entity', None),
                   group=getattr(params, 'group', None),
                   name=f"vae_{args.run_num}", config=params.params if hasattr(params, 'params') else None,
                   dir=out_dir)
        wandb.define_metric('iters')
        wandb.define_metric('train/*', step_metric='iters')
        wandb.define_metric('val/*', step_metric='epoch')

    model.train()
    iters = 0
    t_mark = time.time()
    for epoch in range(args.epochs):
        for it, batch in enumerate(loader):
            pred, y, valid, mean, logvar, _ = forecast_step(model, batch, device, args.lead_days, ice)
            loss_fc = masked_mse(pred, y, valid, w)
            loss_kl = kl_loss(mean, logvar)
            loss = loss_fc + kl_w * loss_kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            iters += 1
            if args.save_every > 0 and iters % args.save_every == 0:
                state = {'model_state': model.state_dict(), 'epoch': epoch, 'iters': iters, 'lead_days': args.lead_days}
                torch.save(state, os.path.join(out_dir, f'ckpt_{iters}.tar'))
                torch.save(state, os.path.join(out_dir, 'ckpt.tar'))
            if it % 100 == 0:
                sec_it = (time.time() - t_mark) / (100 if it > 0 else 1)   # mean over the last 100 iters
                t_mark = time.time()
                print(f"ep {epoch} it {it}  loss {loss.item():.4f}  fc {loss_fc.item():.4f}  kl {loss_kl.item():.4f}  {sec_it:.3f} s/iter", flush=True)
                if use_wandb:
                    wandb.log({'iters': iters, 'epoch': epoch, 'train/loss': loss.item(),
                               'train/fc': loss_fc.item(), 'train/kl': loss_kl.item(), 'train/sec_per_iter': sec_it})
        fc_v, kl_v, pers_v = validate(model, val_loader, device, args.lead_days, ice, w)
        print(f"== epoch {epoch}  val fc {fc_v:.4f}  persistence {pers_v:.4f}  skill {1 - fc_v / pers_v:.3f}  val kl {kl_v:.4f}", flush=True)
        if use_wandb:
            wandb.log({'iters': iters, 'epoch': epoch, 'val/fc': fc_v, 'val/persistence': pers_v,
                       'val/skill_vs_persistence': 1 - fc_v / pers_v, 'val/kl': kl_v})
        torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'iters': iters, 'lead_days': args.lead_days},
                   os.path.join(out_dir, 'ckpt.tar'))
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
