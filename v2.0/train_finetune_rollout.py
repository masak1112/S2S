"""Fine-tune the Pangu decoder with multi-step rollout CRPS loss.

The number of autoregressive rollout steps grows by 1 every `rollout_step_interval`
iterations (default 2000), starting from 1.  At each rollout step the CRPS is
computed and the loss is averaged uniformly over all lead times.

Only the decoder layers (upsample, layer4, patchrecovery2d, patchrecovery3d)
are trainable.  The SI UNet and all encoders stay frozen.

Usage:
    torchrun --nproc_per_node=<N> train_finetune_rollout.py \\
        --yaml_config config/PANGU_S2S.yaml --config S2S \\
        --epochs 10 --max_rollout_steps 10 --rollout_step_interval 2000
"""

import argparse
import logging
import os
from collections import OrderedDict

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
from utils.losses import Latitude_weighted_CRPSLoss

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()


class RolloutFinetuner(Trainer):
    def __init__(self, params, world_rank, max_rollout_steps=10, rollout_step_interval=2000, initial_rollout_steps=1):
        super().__init__(params, world_rank)

        self.max_rollout_steps = max_rollout_steps
        self.rollout_step_interval = rollout_step_interval
        # Offset so the schedule starts at initial_rollout_steps from iteration 0.
        # current_rollout_steps = 1 + (iters + iters_offset) // interval
        self.iters_offset = (initial_rollout_steps - 1) * rollout_step_interval

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        for p in self.model_vae.parameters():
            p.requires_grad = False

        self.diff_model = self._build_si_model()
        self._load_si_checkpoint()

        # Freeze SI UNet; decoder unfrozen below
        for p in self.diff_model.unet.parameters():
            p.requires_grad = False

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

        self.get_dataset_rollout()
        self.scaler = GradScaler()
        self.wandb_enabled = bool(getattr(params, 'log_to_wandb', False) and wandb.run is not None)
        if self.wandb_enabled and world_rank == 0:
            wandb.define_metric('iters')
            for key in ('train/crps_loss', 'train/crps_surface', 'train/crps_upper_air',
                        'train/crps_diagnostic', 'train/rmse_diagnostic', 'train/rollout_steps', 'lr'):
                wandb.define_metric(key, step_metric='iters')

    # ------------------------------------------------------------------
    # Current rollout horizon
    # ------------------------------------------------------------------

    @property
    def current_rollout_steps(self):
        """Steps increase by 1 every rollout_step_interval iterations, capped at max.
        iters_offset shifts the schedule so training begins at initial_rollout_steps."""
        steps = 1 + (self.iters + self.iters_offset) // self.rollout_step_interval
        return min(steps, self.max_rollout_steps)

    # ------------------------------------------------------------------
    # Dataset — use validate=True loaders so multi-step targets are returned
    # ------------------------------------------------------------------

    def get_dataset_rollout(self):
        """Build training loaders with validate=True so the dataset returns
        multi-step targets alongside the boundary conditions."""
        from utils.data_loader_multifiles import get_data_loader
        logging.info('rank %d, begin rollout data loader init' % self.world_rank)

        def _make_loader(year_start, year_end):
            # train=False + validate=True is required: when train=True the dataset
            # always sets self.validate=False, suppressing multi-step target output.
            # shuffle=True gives each DDP rank a disjoint shard and re-shuffles each epoch.
            return get_data_loader(
                self.params,
                self.params.data_dir,
                dist.is_initialized(),
                year_start=year_start,
                year_end=year_end,
                train=False,
                validate=True,
                shuffle=True,
            )

        # sel_dates restricts __len__ to fixed S2S init dates (inference use only).
        # Fine-tuning uses the full continuous date range for both train and val loaders.
        # The dataset holds a live reference to self.params so the flag must stay False.
        self.params['sel_dates'] = False

        if self.params.train_year_to_year:
            self.train_data_loaders, self.train_datasets, self.train_samplers = [], [], []
            for year_start in range(self.params.train_year_start, self.params.train_year_end):
                loader, dataset, sampler = _make_loader(year_start, year_start + 1)
                self.train_data_loaders.append(loader)
                self.train_datasets.append(dataset)
                self.train_samplers.append(sampler)
        else:
            loader, dataset, sampler = _make_loader(
                self.params.train_year_start, self.params.train_year_end)
            self.train_data_loaders = [loader]
            self.train_datasets    = [dataset]
            self.train_samplers    = [sampler]

        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            self.params, self.params.data_dir, dist.is_initialized(),
            year_start=self.params.val_year_start,
            year_end=self.params.val_year_end,
            train=False,
            num_inferences=self.params.num_inferences,
            validate=True,
        )

    def _prepare_inputs_batch_rollout(self, data):
        """Unpack the validate+lead_times tuple from the dataloader.

        Dataloader returns (train=False, validate=True, lead_times set):
          with diagnostics:    (surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc, start_time)  — 7 items
          without diagnostics: (surf, ua, tgt_sfc, tgt_ua, vbc, start_time)             — 6 items
        Shapes after collation:
          surf:     (B, C_sfc, H, W)
          ua:       (B, C_ua, P, H, W)
          tgt_sfc:  (B, max_steps, C_sfc, H, W)
          tgt_ua:   (B, max_steps, C_ua, P, H, W)
          tgt_diag: (B, max_steps, C_diag, H, W)  — only when has_diagnostic
          vbc:      (B, max_steps+1, C_vbc, H, W)
          start_time dropped.
        """
        dev = self.device
        f32 = torch.float32

        if self.params.has_diagnostic:
            surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc, _t = data
            tgt_diag = tgt_diag.to(dev, dtype=f32, non_blocking=True)
        else:
            surf, ua, tgt_sfc, tgt_ua, vbc, _t = data
            tgt_diag = None

        surf    = surf.to(dev,    dtype=f32, non_blocking=True)
        ua      = ua.to(dev,      dtype=f32, non_blocking=True)
        tgt_sfc = tgt_sfc.to(dev, dtype=f32, non_blocking=True)
        tgt_ua  = tgt_ua.to(dev,  dtype=f32, non_blocking=True)
        vbc     = vbc.to(dev,     dtype=f32, non_blocking=True)

        return surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc

    # ------------------------------------------------------------------
    # Model construction / checkpoint helpers (same as DecoderFinetuner)
    # ------------------------------------------------------------------

    def _build_si_model(self):
        model = StochasticInterpolant(
            T=1000,
            VAEEncoder=self.model_vae.module,
            DETEncoder=self.model_det.module,
            params=self.params,
        ).to(device)
        return model

    def _load_si_checkpoint(self):
        def _load(path, module):
            ckpt = torch.load(path, map_location=f'cuda:{self.params.local_rank}', weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            module.load_state_dict(state, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)

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
        print("Encoders frozen for rollout CRPS fine-tuning")

    def _get_latitudes(self):
        return torch.from_numpy(np.array(self.params.lat)).float()

    # ------------------------------------------------------------------
    # Single rollout CRPS step (averaged over lead times)
    # ------------------------------------------------------------------

    def _rollout_crps_step(self, surface_in, upper_air_in, varying_boundary_data,
                           target_surface_steps, target_upper_air_steps,
                           target_diagnostic_steps, num_rollout, num_samples):
        """Run `num_rollout` autoregressive steps; backward per step to free activation memory.

        Gradients are accumulated via per-step scaled backward() calls so that each
        step's decoder activation graph is freed before the next step's SI sampling
        runs.  The caller must NOT call backward() or scaler.step() again — this
        method handles both.  Only Python float scalars are returned for logging.

        Args:
            surface_in:            (B, C_sfc, H, W)
            upper_air_in:          (B, C_ua, P, H, W)
            varying_boundary_data: (B, max_steps+1, C_vbc, H, W)
            target_surface_steps:  (B, max_steps, C_sfc, H, W)
            target_upper_air_steps:(B, max_steps, C_ua, P, H, W)
            target_diagnostic_steps: (B, max_steps, C_diag, H, W) or None
            num_rollout:           int — how many steps to unroll
            num_samples:           int — ensemble size for CRPS
        """
        latitudes = self._get_latitudes().to(self.device)
        crps_loss_fn = Latitude_weighted_CRPSLoss(latitudes, num_ensemble_members=num_samples)

        s_cur = surface_in       # (B, C_sfc, H, W)
        ua_cur = upper_air_in    # (B, C_ua, P, H, W)

        total_crps_sfc  = 0.0
        total_crps_ua   = 0.0
        total_crps_diag = 0.0
        total_precip_rmse = 0.0
        total_loss = 0.0

        for step in range(num_rollout):
            vbc_step = varying_boundary_data[:, step]   # (B, C_vbc, H, W)
            tgt_sfc  = target_surface_steps[:, step]    # (B, C_sfc, H, W)
            tgt_ua   = target_upper_air_steps[:, step]  # (B, C_ua, P, H, W)
            tgt_diag = target_diagnostic_steps[:, step] if target_diagnostic_steps is not None else None

            # Encode (frozen, no grad through SI)
            with torch.no_grad():
                surface_cat = self.diff_model._prepare_surface(
                    s_cur, self.constant_boundary_data, vbc_step)
                z = self.diff_model._encode_vae(surface_cat, ua_cur)
                x, skip = self.diff_model._encode_det(surface_cat, ua_cur, train=False)

                source_sigma = float(getattr(self.params, 'source_sigma', x.std().item())) * 1.5
                x_si = self.diff_model.generate(
                    model_diff=self.diff_model.unet,
                    z=z,
                    x=x,
                    num_samples=num_samples,
                    sample_shape=x.shape,
                    device=self.device,
                    temperature=1.5,
                    source_sigma=source_sigma,
                )  # (B*num_samples, Pl, seq, embed*scale)

            # Free encoder/SI intermediates before the decoder forward pass.
            torch.cuda.empty_cache()

            BS = x_si.shape[0]

            # Decoder — gradients flow here
            with autocast(device_type='cuda', dtype=torch.float16):
                Pl_est, Lat_est, Lon_est = self.diff_model.model_det.EST_input_resolution
                x_dec = x_si.reshape(BS, -1, 240 * self.params.updown_scale_factor)
                x_dec = self.diff_model.model_det.upsample(x_dec)
                x_dec = self.diff_model.model_det.layer4(x_dec, train=False)
                skip_expanded = skip.repeat_interleave(num_samples, dim=0)
                output = torch.cat([x_dec, skip_expanded], dim=-1)
                output = output.transpose(1, 2).reshape(BS, -1, Pl_est, Lat_est, Lon_est)

                output_surface  = output[:, :, -1, :, :]
                output_upper_air = output[:, :, :-1, :, :]
                output_2D = self.diff_model.model_det.patchrecovery2d(output_surface)
                pred_sfc  = output_2D[:, self.diff_model.surface_prognostic_idxs]
                pred_ua   = self.diff_model.model_det.patchrecovery3d(output_upper_air)
                pred_diag = output_2D[
                    :,
                    self.diff_model.num_surface_vars:
                    self.diff_model.num_surface_vars + self.diff_model.num_diagnostic_vars
                ].reshape(BS, -1, pred_sfc.shape[-2], pred_sfc.shape[-1])

                # CRPS targets repeated for ensemble dimension
                tgt_sfc_rep = tgt_sfc.repeat_interleave(num_samples, dim=0)
                tgt_ua_rep  = tgt_ua.repeat_interleave(num_samples, dim=0)

                crps_sfc = crps_loss_fn(pred_sfc, tgt_sfc_rep)
                crps_ua  = crps_loss_fn(pred_ua,  tgt_ua_rep)
                step_loss = 0.2 * (crps_sfc + crps_ua)

                crps_diag    = torch.tensor(0.0, device=self.device)
                precip_rmse  = torch.tensor(0.0, device=self.device)

                precip_idxs = [i for i, v in enumerate(self.params.diagnostic_variables)
                                   if 'precipitation' in v]
                
                if tgt_diag is not None and self.diff_model.num_diagnostic_vars > 0:
                    tgt_diag_rep = tgt_diag.repeat_interleave(num_samples, dim=0)
                    #Before checkpoint 1000, we use the following 
                    #crps_diag = crps_loss_fn(pred_diag, tgt_diag_rep)
                    #step_loss = step_loss + 0.8 * crps_diag
                    #After checkpoitn 1000  only oprimtize the precpitaiton
                    crps_diag = crps_loss_fn(pred_diag[:, precip_idxs], tgt_diag_rep[:, precip_idxs])
                    step_loss = crps_diag


                    if precip_idxs:
                        import math
                        lat_w = torch.cos(torch.tensor(self.params.lat, dtype=torch.float32,
                                                       device=self.device) * math.pi / 180.0)
                        lat_w = (lat_w / lat_w.mean()).view(1, 1, -1, 1)
                        for m in range(num_samples):
                            pred_precip = pred_diag[m::num_samples][:, precip_idxs]
                            tgt_precip  = tgt_diag[:, precip_idxs]
                            precip_rmse = precip_rmse + torch.sqrt(
                                (lat_w * (pred_precip - tgt_precip) ** 2).mean())
                        #before checkpoint 1000
                        
                        #step_loss = step_loss + 0.1 * precip_rmse
                        #after checkpoit 1000
                        step_loss = step_loss

            # Scale and backward for this step; accumulate gradients across steps.
            # Dividing by num_rollout here matches the averaging done at the end.
            self.scaler.scale(step_loss / num_rollout).backward()

            # Accumulate scalar logging values only (no graph retained).
            total_crps_sfc    += crps_sfc.item()
            total_crps_ua     += crps_ua.item()
            total_crps_diag   += crps_diag.item()
            total_precip_rmse += precip_rmse.item()
            total_loss        += step_loss.item()

            # Advance state: use ensemble mean as the deterministic next IC.
            # Cast to float32 because the encode path (_encode_det, _encode_vae)
            # runs outside autocast and expects float32 inputs.
            with torch.no_grad():
                pred_sfc_mean = pred_sfc.reshape(
                    -1, num_samples, *pred_sfc.shape[1:]).mean(dim=1)
                pred_ua_mean  = pred_ua.reshape(
                    -1, num_samples, *pred_ua.shape[1:]).mean(dim=1)
            s_cur  = pred_sfc_mean.detach().float()
            ua_cur = pred_ua_mean.detach().float()

            # Free this step's decoder graph before next step's SI sampling.
            del x_si, x_dec, skip, skip_expanded, output, output_surface, output_upper_air
            del output_2D, pred_sfc, pred_ua, pred_diag, step_loss, crps_sfc, crps_ua
            del crps_diag, precip_rmse
            torch.cuda.empty_cache()

        return (total_loss        / num_rollout,
                total_crps_sfc    / num_rollout,
                total_crps_ua     / num_rollout,
                total_crps_diag   / num_rollout,
                total_precip_rmse / num_rollout)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_one_epoch_rollout(self):
        self.epoch += 1
        num_samples = int(getattr(self.params, 'crps_ensemble_size', 2))
        loss_val = 0.0

        for year_idx, loader in enumerate(self.train_data_loaders):
            sampler = self.train_samplers[year_idx]
            if sampler is not None:
                sampler.set_epoch(self.epoch)
            for i, data in enumerate(loader):
                if self.params.mode == 'test' and i >= self.params.test_iterations:
                    break

                self.iters += 1
                num_rollout = self.current_rollout_steps

                # Unpack: vbc (B, max_steps+1, C, H, W), targets (B, max_steps, C, ...)
                input_surface, input_upper_air, target_surface, target_upper_air, \
                    target_diagnostic, varying_boundary_data = self._prepare_inputs_batch_rollout(data)

                # Slice to current rollout horizon
                target_surface   = target_surface[:, :num_rollout]
                target_upper_air = target_upper_air[:, :num_rollout]
                if target_diagnostic is not None:
                    target_diagnostic = target_diagnostic[:, :num_rollout]
                # vbc step i is the boundary valid at the start of step i
                vbc = varying_boundary_data[:, :num_rollout]

                self.optimizer.zero_grad()

                # _rollout_crps_step calls scaler.scale(step_loss).backward() per
                # step to free each step's activation graph before the next SI
                # sampling run.  It returns Python floats, not tensors.
                loss_val, crps_sfc, crps_ua, crps_diag, precip_rmse = self._rollout_crps_step(
                    surface_in=input_surface,
                    upper_air_in=input_upper_air,
                    varying_boundary_data=vbc,
                    target_surface_steps=target_surface,
                    target_upper_air_steps=target_upper_air,
                    target_diagnostic_steps=target_diagnostic,
                    num_rollout=num_rollout,
                    num_samples=num_samples,
                )

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.diff_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if i % 100 == 0:
                    logging.info(
                        f"[rollout-finetune] epoch={self.epoch} iter={self.iters} "
                        f"rollout_steps={num_rollout} loss={loss_val:.4f}"
                    )

                if self.wandb_enabled and world_rank == 0:
                    wandb.log({
                        'train/crps_loss':        loss_val,
                        'train/crps_surface':     crps_sfc,
                        'train/crps_upper_air':   crps_ua,
                        'train/crps_diagnostic':  crps_diag,
                        'train/rmse_diagnostic':  precip_rmse,
                        'train/rollout_steps':    num_rollout,
                        'lr': self.optimizer.param_groups[0]['lr'],
                    }, step=self.iters)

                if self.iters % 200 == 0:
                    self._save(f"rollout_ckpt_{self.iters}.tar")

        return {'train_crps_loss': loss_val, 'epoch': self.epoch}

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

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
        logging.info(f"Saved rollout checkpoint: {path}")

    def _resume(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{self.params.local_rank}', weights_only=False)
        state = ckpt['model_state']
        if any(k.startswith('module.') for k in state):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        self.diff_model.load_state_dict(state, strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.iters = ckpt['iters']
        logging.info(f"Resumed from {ckpt_path} (epoch={self.epoch}, iters={self.iters}, "
                     f"rollout_steps={self.current_rollout_steps})")

    def run(self, epochs=10):
        while self.epoch < epochs:
            logs = self.train_one_epoch_rollout()
            if self.wandb_enabled and world_rank == 0:
                wandb.log(logs, step=self.epoch)
            self._save('rollout_last.tar')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num',              default='0300', type=str)
    parser.add_argument('--yaml_config',          default='v2.0/config/PANGU_S2S.yaml', type=str)
    parser.add_argument('--config',               default='S2S', type=str)
    parser.add_argument('--epochs',               default=10, type=int)
    parser.add_argument('--run_iter',             default=1, type=int)
    parser.add_argument('--local-rank',           type=int)
    parser.add_argument('--max_rollout_steps',    default=10, type=int,
                        help='Maximum number of autoregressive rollout steps')
    parser.add_argument('--rollout_step_interval', default=2000, type=int,
                        help='Increase rollout steps by 1 every N iterations')
    parser.add_argument('--initial_rollout_steps', default=1, type=int,
                        help='Rollout step count to start from (default 1)')
    parser.add_argument('--finetune_ckpt',        default=None, type=str,
                        help='Path to a rollout checkpoint to resume from')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0:
        params['has_diagnostic'] = True
    else:
        params['has_diagnostic'] = False

    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1

    # Dataset must return multi-step targets: set forecast_lead_times accordingly
    if not hasattr(params, 'forecast_lead_times') or params.forecast_lead_times is None:
        params['forecast_lead_times'] = list(range(1, args.max_rollout_steps + 1))

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

    expDir = os.path.join(params.exp_dir, args.config, f"rollout_finetune_{args.run_num}")
    if world_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)
        logging_utils.log_to_file(logger_name=None,
                                   log_filename=os.path.join(expDir, 'rollout_finetune.log'))
        logging_utils.log_versions()
        params.log()

    params['experiment_dir']        = os.path.abspath(expDir)
    params['checkpoint_path']       = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path']  = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming']              = False
    params['run_iter']              = args.run_iter
    params['log_to_wandb']         = (world_rank == 0) and params['log_to_wandb']

    trainer = RolloutFinetuner(
        params, world_rank,
        max_rollout_steps=args.max_rollout_steps,
        rollout_step_interval=args.rollout_step_interval,
        initial_rollout_steps=args.initial_rollout_steps,
    )
    if args.finetune_ckpt:
        ckpt = torch.load(args.finetune_ckpt, map_location=f'cuda:{params.local_rank}', weights_only=False)
        state = ckpt['model_state']
        if any(k.startswith('module.') for k in state):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        trainer.diff_model.load_state_dict(state, strict=True)
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.epoch = ckpt.get('epoch', 0)
        trainer.iters = ckpt.get('iters', 0)
        logging.info(f"Resumed from {args.finetune_ckpt} "
                     f"(epoch={trainer.epoch}, iters={trainer.iters})")
    trainer.run(epochs=trainer.epoch + args.epochs)
    logging.info('Rollout fine-tuning DONE — rank %d' % world_rank)
