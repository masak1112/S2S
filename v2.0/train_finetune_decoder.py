"""Fine-tune the Pangu decoder with CRPS loss using a frozen SI ensemble.

Only the decoder layers (upsample, layer4, patchrecovery2d, patchrecovery3d)
are trainable. The SI UNet and all encoders stay frozen.

Usage:
    torchrun --nproc_per_node=<N> train_finetune_decoder.py \
        --yaml_config config/PANGU_S2S.yaml --config S2S \
        --epochs 10
"""

import argparse
import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.amp import GradScaler, autocast
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from utils import logging_utils
from utils.YParams import YParams

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()


class DecoderFinetuner(Trainer):
    def __init__(self, params, world_rank):
        super().__init__(params, world_rank)

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        for p in self.model_vae.parameters():
            p.requires_grad = False

        self.diff_model = self._build_si_model()
        self._load_si_checkpoint()

        # Freeze SI UNet — decoder will be unfrozen below
        for p in self.diff_model.unet.parameters():
            p.requires_grad = False

        # Unfreeze decoder layers only
        decoder_modules = [
            self.diff_model.model_det.upsample,
            self.diff_model.model_det.layer4,
            self.diff_model.model_det.patchrecovery2d,
            self.diff_model.model_det.patchrecovery3d,
        ]
        for mod in decoder_modules:
            for p in mod.parameters():
                p.requires_grad = True

        decoder_params = [p for mod in decoder_modules for p in mod.parameters()]
        n_decoder = sum(p.numel() for p in decoder_params)
        n_total = sum(p.numel() for p in self.diff_model.parameters())
        print(f"Decoder params: {n_decoder:,} / {n_total:,} total")

        self.optimizer = torch.optim.Adam(
            decoder_params,
            lr=float(getattr(params, 'finetune_lr', params.lr)),
            weight_decay=params.weight_decay,
        )

        self.get_dataset()
        self.scaler = GradScaler()
        self.wandb_enabled = bool(getattr(params, 'log_to_wandb', False) and wandb.run is not None)
        if self.wandb_enabled and world_rank == 0:
            wandb.define_metric('iters')
            for key in ('train/crps_loss', 'train/crps_surface', 'train/crps_upper_air',
                        'train/crps_diagnostic', 'train/rmse_diagnostic', 'lr'):
                wandb.define_metric(key, step_metric='iters')
            for var in ('t2m', 'precip', 'z500'):
                for lead in (1, 5, 10):
                    for metric in ('crps', 'ssr'):
                        wandb.define_metric(f'val/{var}_lead{lead}d_{metric}', step_metric='iters')

    def _build_si_model(self):
        model = StochasticInterpolant(
            T=1000,
            VAEEncoder=self.model_vae.module,
            DETEncoder=self.model_det.module,
            params=self.params,
        ).to(device)
        return model

    def _load_si_checkpoint(self):
        """Load pretrained SI checkpoint then freeze encoders."""
        def _load(path, module):
            ckpt = torch.load(path, map_location=f'cuda:{self.params.local_rank}', weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            module.load_state_dict(state, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)

        # Load SI UNet weights
        si_ckpt_path = getattr(self.params, 'checkpoint_path_diff', None)
        if si_ckpt_path and os.path.isfile(si_ckpt_path):
            ckpt = torch.load(si_ckpt_path, map_location=f'cuda:{self.params.local_rank}', weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            unet_state = {k: v for k, v in state.items() if k.startswith('unet.')}
            self.diff_model.load_state_dict(unet_state, strict=False)
            print(f"Loaded SI UNet weights from {si_ckpt_path}")
        else:
            print("No SI checkpoint found — using randomly initialised UNet")

        self.diff_model.freeze_encoder()
        self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()
        print("Encoders frozen for CRPS fine-tuning")

    def _get_latitudes(self):
        """Return latitude tensor from params.lat (same source used by Trainer)."""
        return torch.from_numpy(np.array(self.params.lat)).float()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _crps_scores(self, ensemble: torch.Tensor, target: torch.Tensor, lat_weights: torch.Tensor):
        """Compute latitude-weighted CRPS and its skill/spread components.

        Args:
            ensemble: [M, ...spatial] — M ensemble members for a single IC / lead time
            target:   [...spatial]    — ground-truth field
            lat_weights: [H] latitude weights, broadcastable to spatial dims

        Returns:
            crps_mean (scalar), skill (scalar), spread (scalar)
        """
        M = ensemble.shape[0]
        # Reshape lat weights to broadcast over (H, W)
        w = lat_weights.to(ensemble.device).view(-1, 1)  # [H, 1]

        # Skill: mean over members of |member - target|
        skill = (ensemble - target.unsqueeze(0)).abs()  # [M, H, W]
        skill = (skill * w).sum(dim=(-2, -1)) / w.sum()  # [M]
        skill = skill.mean()

        # Spread: (1/M^2) * sum_{i,j} |xi - xj|  (i != j term = 1/(M*(M-1)))
        spread = torch.zeros(1, device=ensemble.device)
        for i in range(M):
            for j in range(i + 1, M):
                diff = (ensemble[i] - ensemble[j]).abs()
                spread = spread + (diff * w).sum() / w.sum()
        spread = spread * 2.0 / (M * (M - 1))

        crps = skill - 0.5 * spread
        return crps, skill, spread

    @torch.no_grad()
    def validate_crps_ssr(self):
        """Run validation on 4 ICs × 4 ensemble members.

        Reports CRPS and Spread-Skill Ratio (SSR = spread / RMSE) at lead
        times 1, 5, 10 days for 2m_temperature, total_precipitation_24hr,
        and geopotential@500 hPa.  Only rank-0 logs results.
        """
        NUM_IC = 4
        NUM_ENS = 4
        LEAD_DAYS = [1, 5, 10]
        timedelta_h = int(getattr(self.params, 'timedelta_hours', 24))
        lead_steps = [d * 24 // timedelta_h for d in LEAD_DAYS]
        max_steps = max(lead_steps)

        latitudes = self._get_latitudes()
        lat_weights = torch.cos(np.pi / 180.0 * latitudes)
        lat_weights = lat_weights / lat_weights.sum() * len(lat_weights)

        # Identify variable indices
        surf_vars = list(self.valid_dataset.surface_variables)
        ua_vars = list(self.valid_dataset.upper_air_variables)
        diag_vars = list(self.valid_dataset.diagnostic_variables) if self.params.has_diagnostic else []
        levels = list(self.params.levels)

        t2m_idx_sfc = surf_vars.index('2m_temperature') if '2m_temperature' in surf_vars else None
        precip_idx_diag = diag_vars.index('total_precipitation_24hr') if 'total_precipitation_24hr' in diag_vars else None
        z500_ua_idx = ua_vars.index('geopotential') if 'geopotential' in ua_vars else None
        z500_lev_idx = levels.index(500) if 500 in levels else None

        self.diff_model.eval()
        device = self.device

        # Collect 4 ICs from the validation loader.
        # The validate=True loader returns:
        #   (surf, ua, tgt_surf_steps, tgt_ua_steps, [tgt_diag_steps,] vbc, start_time)
        # vbc shape: [B, max_lead+1, C_vbc, H, W]
        ic_list = []
        for data in self.valid_data_loader:
            if self.params.has_diagnostic:
                surf, ua, _tgt_surf, _tgt_ua, _tgt_diag, vbc, _t = data
            else:
                surf, ua, _tgt_surf, _tgt_ua, vbc, _t = data
            surf = surf.to(device, dtype=torch.float32, non_blocking=True)
            ua = ua.to(device, dtype=torch.float32, non_blocking=True)
            vbc = vbc.to(device, dtype=torch.float32, non_blocking=True)
            for b in range(surf.shape[0]):
                ic_list.append({
                    'surf': surf[b:b+1],
                    'ua': ua[b:b+1],
                    'vbc': vbc[b:b+1],  # [1, max_lead+1, C_vbc, H, W]
                })
                if len(ic_list) >= NUM_IC:
                    break
            if len(ic_list) >= NUM_IC:
                break

        # Storage: results[var_label][lead_day] = list of (crps, skill, spread, rmse) per IC
        results = {}
        var_specs = []
        if t2m_idx_sfc is not None:
            var_specs.append(('t2m', 'surface', t2m_idx_sfc))
        if precip_idx_diag is not None:
            var_specs.append(('precip', 'diagnostic', precip_idx_diag))
        if z500_ua_idx is not None and z500_lev_idx is not None:
            var_specs.append(('z500', 'upper_air', (z500_ua_idx, z500_lev_idx)))
        for label, *_ in var_specs:
            results[label] = {d: [] for d in LEAD_DAYS}

        for ic in ic_list:
            surf0 = ic['surf']
            ua0 = ic['ua']
            vbc = ic['vbc']

            # Run NUM_ENS members autoregressively up to max_steps
            ens_surf = []   # each: [max_steps, 1, C_sfc, H, W] in standardised space
            ens_ua = []
            ens_diag = []

            for ens_id in range(NUM_ENS):
                s_cur = surf0.clone()
                ua_cur = ua0.clone()
                traj_s, traj_ua, traj_d = [], [], []
                for step in range(max_steps):
                    out_s, out_ua, out_d = self.diff_model.prediction(
                        surface_in=s_cur,
                        constant_boundary=self.constant_boundary_data[:1],
                        varying_boundary=vbc[:, step],
                        upper_air_in=ua_cur,
                        device=device,
                        mc_dropout=True,
                        temperature=1.5,
                    )
                    traj_s.append(out_s)
                    traj_ua.append(out_ua)
                    traj_d.append(out_d)
                    s_cur, ua_cur = out_s, out_ua

                ens_surf.append(torch.stack(traj_s, dim=1))    # [1, steps, C_sfc, H, W]
                ens_ua.append(torch.stack(traj_ua, dim=1))     # [1, steps, C_ua, P, H, W]
                ens_diag.append(torch.stack(traj_d, dim=1))    # [1, steps, C_diag, H, W]

            # ens_surf: list of [1, steps, ...] → stack → [M, steps, ...]
            ens_surf_t = torch.cat(ens_surf, dim=0)   # [M, steps, C, H, W]
            ens_ua_t = torch.cat(ens_ua, dim=0)
            ens_diag_t = torch.cat(ens_diag, dim=0)

            for lead_day, step in zip(LEAD_DAYS, lead_steps):
                s_idx = step - 1  # step=1 → index 0 (we stored step 1..max_steps)
                for label, vtype, vidx in var_specs:
                    if vtype == 'surface':
                        ens_field = ens_surf_t[:, s_idx, vidx]  # [M, H, W]
                    elif vtype == 'diagnostic':
                        ens_field = ens_diag_t[:, s_idx, vidx]
                    else:
                        ua_idx, lev_idx = vidx
                        ens_field = ens_ua_t[:, s_idx, ua_idx, lev_idx]

                    # Compute ensemble mean as reference for SSR
                    ens_mean = ens_field.mean(dim=0)  # [H, W]
                    w = lat_weights.to(device).view(-1, 1)

                    # CRPS
                    crps_val, skill_val, spread_val = self._crps_scores(ens_field, ens_mean, lat_weights)

                    # RMSE of ensemble mean vs. each member (proxy for true error)
                    se = ((ens_field - ens_mean.unsqueeze(0)) ** 2 * w).sum(dim=(-2, -1)) / w.sum()
                    rmse_proxy = se.mean().sqrt()
                    # SSR = spread / RMSE (using ensemble std as spread, RMSE proxy)
                    ens_std = ens_field.std(dim=0)
                    spread_std = (ens_std * w).sum() / w.sum()
                    ssr = spread_std / (rmse_proxy + 1e-8)

                    results[label][lead_day].append({
                        'crps': crps_val.item(),
                        'skill': skill_val.item(),
                        'spread': spread_val.item(),
                        'ssr': ssr.item(),
                    })

        if world_rank == 0:
            log_lines = [f"[val@{self.iters}] CRPS/SSR validation ({NUM_IC} ICs, {NUM_ENS} members)"]
            for label, _, _ in var_specs:
                for lead_day in LEAD_DAYS:
                    recs = results[label][lead_day]
                    if not recs:
                        continue
                    crps_m = np.mean([r['crps'] for r in recs])
                    ssr_m = np.mean([r['ssr'] for r in recs])
                    log_lines.append(
                        f"  {label:6s}  lead={lead_day:2d}d  CRPS={crps_m:.4f}  SSR={ssr_m:.4f}"
                    )
            logging.info('\n'.join(log_lines))

            if self.wandb_enabled:
                wlog = {}
                for label, _, _ in var_specs:
                    for lead_day in LEAD_DAYS:
                        recs = results[label][lead_day]
                        if not recs:
                            continue
                        wlog[f'val/{label}_lead{lead_day}d_crps'] = np.mean([r['crps'] for r in recs])
                        wlog[f'val/{label}_lead{lead_day}d_ssr'] = np.mean([r['ssr'] for r in recs])
                wandb.log(wlog, step=self.iters)

        self.diff_model.train()
        return results

    def train_one_epoch_crps(self):
        self.epoch += 1
        latitudes = self._get_latitudes()
        loss_val = 0.0
        num_samples = int(getattr(self.params, 'crps_ensemble_size', 2))

        for year_idx, loader in enumerate(self.train_data_loaders):
            for i, data in enumerate(loader):
                if self.params.mode == 'test' and i >= self.params.test_iterations:
                    break

                self.iters += 1
                input_surface, input_upper_air, target_surface, target_upper_air, \
                    target_diagnostic, varying_boundary_data = self._prepare_inputs_batch(data)

                self.optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16):
                    loss, crps_sfc, crps_ua, crps_diag, precip_rmse = self.diff_model.crps_finetune_step(
                        surface_in=input_surface,
                        constant_boundary=self.constant_boundary_data,
                        varying_boundary=varying_boundary_data,
                        upper_air_in=input_upper_air,
                        target_surface=target_surface,
                        target_upper_air=target_upper_air,
                        latitudes=latitudes,
                        num_samples=num_samples,
                        target_diagnostic=target_diagnostic if self.params.has_diagnostic else None,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.diff_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_val = loss.item()
                if i % 100 == 0:
                    logging.info(f"[finetune] epoch={self.epoch} iter={self.iters} loss={loss_val:.4f}")

                if self.wandb_enabled and world_rank == 0:
                    wandb.log({
                        'train/crps_loss': loss_val,
                        'train/crps_surface': crps_sfc.item(),
                        'train/crps_upper_air': crps_ua.item(),
                        'train/crps_diagnostic': crps_diag.item(),
                        'train/rmse_diagnostic': precip_rmse.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                    }, step=self.iters)

                if self.iters % 200 == 0:
                    self._save(f"finetune_ckpt_{self.iters}.tar")
                    self.validate_crps_ssr()

        return {'train_crps_loss': loss_val, 'epoch': self.epoch}

    def _save(self, name):
        if world_rank != 0:
            return
        out_dir = os.path.join(self.params.experiment_dir, 'training_checkpoints')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        torch.save({
            'model_state': self.diff_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iters': self.iters,
        }, path)
        logging.info(f"Saved finetune checkpoint: {path}")

    def _resume(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{self.params.local_rank}', weights_only=False)
        state = ckpt['model_state']
        if any(k.startswith('module.') for k in state):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        self.diff_model.load_state_dict(state, strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.iters = ckpt['iters']
        logging.info(f"Resumed from {ckpt_path} (epoch={self.epoch}, iters={self.iters})")

    def run(self, epochs=10):
        for _ in range(epochs):
            logs = self.train_one_epoch_crps()
            if self.wandb_enabled and world_rank == 0:
                wandb.log(logs, step=self.epoch)
            self._save('finetune_last.tar')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', default='0200', type=str)
    parser.add_argument('--yaml_config', default='v2.0/config/PANGU_S2S.yaml', type=str)
    parser.add_argument('--config', default='S2S', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--run_iter', default=1, type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--finetune_ckpt', default=None, type=str,
                        help='Path to a finetune checkpoint to resume from')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0:
        params['has_diagnostic'] = True
    else:
        params['has_diagnostic'] = False

    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1

    if getattr(params, 'wandb_offline', False):
        os.environ['WANDB_MODE'] = 'offline'

    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
    else:
        params['world_size'] = torch.cuda.device_count()

    if params['world_size'] > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
    else:
        world_rank = 0
        local_rank = 0

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    params['local_rank'] = local_rank

    expDir = os.path.join(params.exp_dir, args.config, f"finetune_{args.run_num}")
    if world_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)
        logging_utils.log_to_file(logger_name=None,
                                   log_filename=os.path.join(expDir, 'finetune.log'))
        logging_utils.log_versions()
        params.log()

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['run_iter'] = args.run_iter
    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']

    trainer = DecoderFinetuner(params, world_rank)
    if args.finetune_ckpt:
        trainer._resume(args.finetune_ckpt)
    trainer.run(epochs=args.epochs)
    logging.info('Fine-tuning DONE — rank %d' % world_rank)
