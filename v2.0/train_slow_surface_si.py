"""Train a dedicated postprocess SI for the slow surface fields.

Corrects sea_surface_temperature, skin_temperature, soil_temperature_level_1
and volumetric_soil_water_layer_1 -- the components that drive S2S skill
through slow boundary forcing and that the joint 96-field postprocessor
handles poorly (see networks/slow_surface_si.py for the measurements) --
and optionally total_precipitation_24hr, the headline S2S target that
responds to those slow fields.

Fields may come from the surface tensor or the diagnostic tensor; the
checkpoint records the source of each so inference scatters residuals back
into the right tensor. Diagnostic fields (precipitation) are never fed back
into the rollout, so correcting them is a pure postprocess.

The whole base model (VAE, deterministic encoder, latent SI, Pangu decoder)
is frozen; only this small SI trains, learning

    noise -> (ERA5 - member_i)   conditioned on (member_i, lead)

Usage:
    torchrun --nproc_per_node=<N> train_slow_surface_si.py \
        --yaml_config config/exp16_nvidia_v3.yaml --config S2S \
        --run_num slow_v1 --epochs 10 --rollout_steps 10

    `--rollout_steps` matters more here than for the joint model: SST is
    well calibrated at day 1 (SSR ~1.0) and badly under-dispersed by day 44
    (~0.40), so training only on 24 h errors teaches the wrong regime.
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

from networks.slow_surface_si import SlowSurfaceSI
from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from utils import logging_utils
from utils.YParams import YParams
from utils.sst_cond import pop_sst_cond

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()

DEFAULT_FIELDS = ['sea_surface_temperature', 'skin_temperature',
                  'soil_temperature_level_1', 'volumetric_soil_water_layer_1']

# Diagnostic fields that are physically bounded below by zero. Their corrected
# values are floored (in normalised units) at the value that maps to 0.
NONNEG_FIELDS = {'total_precipitation_24hr', 'total_precipitation_6hr',
                 'total_precipitation'}


class SlowSurfaceTrainer(Trainer):
    """Trains SlowSurfaceSI on top of a fully frozen base forecast model."""

    def __init__(self, params, world_rank, rollout_steps: int = 1):
        super().__init__(params, world_rank)

        self.rollout_steps = max(1, int(rollout_steps))
        # Lead conditioning spans the full forecast range rather than being
        # clamped near 10 steps, so day 1 and day 45 are distinguishable.
        self.max_lead_steps = int(getattr(params, 'slow_max_lead_steps',
                                          max(self.rollout_steps, 45)))

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        self.diff_model = self._build_si_model()
        self._load_base_checkpoint()
        for p in self.diff_model.parameters():
            p.requires_grad = False
        self.diff_model.eval()

        self.get_dataset_rollout()
        self._setup_fields()

        self.pp_model = SlowSurfaceSI(
            n_fields=self.n_fields,
            base_channels=int(getattr(params, 'slow_base_channels', 32)),
            c_mults=tuple(getattr(params, 'slow_c_mults', (1, 2, 4))),
            scale=tuple(getattr(params, 'slow_scale', (2, 2))),
            gamma_type=str(getattr(params, 'slow_gamma_type', 'brownian')),
            dropout=float(getattr(params, 'slow_dropout', 0.1)),
            field_masks=self.field_masks,
        ).to(self.device)

        if world_rank == 0:
            n = sum(p.numel() for p in self.pp_model.parameters())
            print(f"Slow-surface SI: {n:,} params over {self.n_fields} fields")
            for (name, src, idx), m in zip(self.field_spec, self.field_masks):
                pct = 100.0 * m.float().mean().item()
                print(f"  {name:32s} {src:10s} ch {idx:2d}  valid on {pct:5.1f}% of grid")
            print(f"  lead conditioning normalised by {self.max_lead_steps} steps")

        self.optimizer = torch.optim.Adam(
            self.pp_model.parameters(),
            lr=float(getattr(params, 'slow_lr', 2e-4)),
            weight_decay=params.weight_decay,
        )
        self.scaler = GradScaler()
        self._residual_std_ready = False

        self.wandb_enabled = bool(getattr(params, 'log_to_wandb', False)
                                  and wandb.run is not None)
        if self.wandb_enabled and world_rank == 0:
            wandb.define_metric('iters')
            for k in ('train/slow_si_loss', 'lr'):
                wandb.define_metric(k, step_metric='iters')

    # ------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------

    def _build_si_model(self):
        return StochasticInterpolant(
            T=1000, VAEEncoder=self.model_vae.module,
            DETEncoder=self.model_det.module, params=self.params).to(self.device)

    def _load_base_checkpoint(self):
        def _load(path, module):
            ck = torch.load(path, map_location=f'cuda:{self.params.local_rank}',
                            weights_only=False)
            st = ck['model_state']
            if any(k.startswith('module.') for k in st):
                st = OrderedDict((k[7:], v) for k, v in st.items())
            module.load_state_dict(st, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)

        base = (getattr(self.params, 'checkpoint_path_finetuned', None)
                or getattr(self.params, 'checkpoint_path_diff', None))
        if base and os.path.isfile(base):
            ck = torch.load(base, map_location=f'cuda:{self.params.local_rank}',
                            weights_only=False)
            st = ck['model_state']
            if any(k.startswith('module.') for k in st):
                st = OrderedDict((k[7:], v) for k, v in st.items())
            self.diff_model.load_state_dict(st, strict=False)
            if world_rank == 0:
                print(f"Base model from {base} (epoch={ck.get('epoch')}, "
                      f"iters={ck.get('iters')})")
        elif world_rank == 0:
            print("WARNING: no base checkpoint — frozen base is random init.")
        self.diff_model.freeze_encoder()

    # ------------------------------------------------------------------
    # Dataset — validate=True loaders so multi-step targets are returned
    # ------------------------------------------------------------------

    def get_dataset_rollout(self):
        """Build training loaders that return multi-step targets.

        The base Trainer.get_dataset() passes train=True, and in that mode the
        dataset forces validate=False and yields a single 24 h target, so
        rollout_steps > 1 has nothing to unroll against. Mirrors
        train_finetune_rollout.get_dataset_rollout().
        """
        from utils.data_loader_multifiles import get_data_loader
        logging.info('rank %d, begin slow-si rollout data loader init' % self.world_rank)

        def _make_loader(year_start, year_end):
            # train=False + validate=True is required: train=True suppresses
            # multi-step target output. shuffle=True gives each DDP rank a
            # disjoint shard and re-shuffles every epoch.
            return get_data_loader(
                self.params, self.params.data_dir, dist.is_initialized(),
                year_start=year_start, year_end=year_end,
                train=False, validate=True, shuffle=True)

        # sel_dates restricts __len__ to fixed S2S init dates (inference only);
        # training wants the full continuous range. The dataset holds a live
        # reference to self.params, so the flag must stay False.
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
            self.train_datasets = [dataset]
            self.train_samplers = [sampler]

        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            self.params, self.params.data_dir, dist.is_initialized(),
            year_start=self.params.val_year_start,
            year_end=self.params.val_year_end,
            train=False, num_inferences=self.params.num_inferences,
            validate=True)

    def _prepare_inputs_batch_rollout(self, data):
        """Unpack the validate+lead_times tuple from the dataloader.

          with diagnostics:    (surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc, start_time)
          without diagnostics: (surf, ua, tgt_sfc, tgt_ua, vbc, start_time)
        Shapes after collation:
          surf (B, C_sfc, H, W); ua (B, C_ua, P, H, W)
          tgt_sfc (B, max_steps, C_sfc, H, W); tgt_ua (B, max_steps, C_ua, P, H, W)
          vbc (B, max_steps+1, C_vbc, H, W); start_time dropped.
        """
        dev, f32 = self.device, torch.float32
        # predict_delta=True makes the dataset append delta targets, shifting vbc
        # and start_time; the positional unpack below would silently mis-assign.
        if getattr(self.params, 'predict_delta', False):
            raise NotImplementedError(
                "slow-surface SI rollout does not support predict_delta=True")
        data, self.sst_cond_batch = pop_sst_cond(data, self.params, dev)
        if self.params.has_diagnostic:
            surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc, _t = data
            tgt_diag = tgt_diag.to(dev, dtype=f32, non_blocking=True)
        else:
            surf, ua, tgt_sfc, tgt_ua, vbc, _t = data
            tgt_diag = None
        surf = surf.to(dev, dtype=f32, non_blocking=True)
        ua = ua.to(dev, dtype=f32, non_blocking=True)
        tgt_sfc = tgt_sfc.to(dev, dtype=f32, non_blocking=True)
        tgt_ua = tgt_ua.to(dev, dtype=f32, non_blocking=True)
        vbc = vbc.to(dev, dtype=f32, non_blocking=True)
        return surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc

    # ------------------------------------------------------------------
    # Fields and masks
    # ------------------------------------------------------------------

    def _setup_fields(self):
        """Resolve each field to (name, source, channel) and build its mask.

        The slow fields live in the surface tensor, whose channel order is the
        dataset's `surface_variables` (config surface vars followed by the land
        and ocean vars). Precipitation lives in the diagnostic tensor, ordered
        as `diagnostic_variables`. A name is looked up in the surface list
        first, then the diagnostic list.
        """
        ds = self.train_datasets[0]
        surf = list(ds.surface_variables)
        diag = list(getattr(ds, 'diagnostic_variables', []) or [])
        requested = list(getattr(self.params, 'slow_fields', DEFAULT_FIELDS))

        spec, missing = [], []
        for name in requested:
            if name in surf:
                spec.append((name, 'surface', surf.index(name)))
            elif name in diag:
                spec.append((name, 'diagnostic', diag.index(name)))
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"slow_fields not in dataset surface_variables {surf} or "
                f"diagnostic_variables {diag}: {missing}")

        self.field_spec = spec
        self.field_names = [s[0] for s in spec]
        self.field_sources = [s[1] for s in spec]
        self.field_idxs = [s[2] for s in spec]
        self.n_fields = len(spec)
        self.has_diag_fields = any(src == 'diagnostic' for src in self.field_sources)

        # Normalised-space floor for non-negative diagnostic fields. The
        # dataset applies (x - mean) / std, so physical 0 maps to -mean/std.
        # None for fields with no lower bound.
        self.field_floors = []
        for name, src, idx in spec:
            if src == 'diagnostic' and name in NONNEG_FIELDS:
                mu = float(ds.diagnostic_mean.reshape(-1)[idx])
                sd = float(ds.diagnostic_std.reshape(-1)[idx])
                self.field_floors.append(-mu / sd)
            else:
                self.field_floors.append(None)

        # land_mask is the raw land_sea_mask plane (1 = land).
        lm = self.train_datasets[0].land_mask
        lm = lm[0] if lm.dim() == 3 else lm
        land = (lm > 0.5).to(self.device)

        land_vars = set(getattr(self.params, 'land_variables', []))
        ocean_vars = set(getattr(self.params, 'ocean_variables', []))

        masks = []
        for name, src, _ in spec:
            if src == 'diagnostic':
                masks.append(torch.ones_like(land))   # precipitation: global
            elif name in ocean_vars:
                masks.append(~land)          # SST: valid over ocean only
            elif name in land_vars and name != 'skin_temperature':
                masks.append(land)           # soil fields: land only
            else:
                # skin_temperature is defined everywhere -- the data confirms
                # it carries no fill value.
                masks.append(torch.ones_like(land))
        self.field_masks = torch.stack(masks).to(self.device)

    def _gather(self, pred_sfc: torch.Tensor,
                pred_diag: torch.Tensor | None = None) -> torch.Tensor:
        """Stack the configured fields into [B, n_fields, H, W]."""
        planes = []
        for name, src, idx in self.field_spec:
            if src == 'surface':
                planes.append(pred_sfc[:, idx])
            else:
                if pred_diag is None:
                    raise ValueError(
                        f"field '{name}' is diagnostic but no diagnostic tensor "
                        f"was supplied; is has_diagnostic set?")
                planes.append(pred_diag[:, idx])
        return torch.stack(planes, dim=1)

    def _apply_floors(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp bounded fields (precipitation >= 0) in normalised units.

        Mirrors the physical-space clip in inference_slow_surface_si so the
        validation scores describe what actually gets written to disk.
        """
        if all(f is None for f in self.field_floors):
            return x
        x = x.clone()
        for i, f in enumerate(self.field_floors):
            if f is not None:
                x[:, i] = x[:, i].clamp(min=f)
        return x

    # ------------------------------------------------------------------
    # Residual normalisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _calibrate_residual_std(self, n_batches: int = 8):
        """Per-field residual std over valid cells only."""
        sums = torch.zeros(self.n_fields, device=self.device)
        counts = 0
        for i, data in enumerate(self.train_data_loaders[0]):
            if i >= n_batches:
                break
            got = self._forward_base(data)
            if got is None:
                continue
            member, target, _ = got
            r = (target - member) ** 2
            m = self.field_masks.unsqueeze(0).to(r.dtype)
            sums += (r * m).sum(dim=(0, 2, 3)) / (m.sum(dim=(0, 2, 3)) *
                                                  r.shape[0] + 1e-8)
            counts += 1
        if counts == 0:
            logging.warning("[slow-si] no batches for std calibration.")
            return
        std = torch.sqrt(sums / counts).clamp(min=1e-8)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(std, op=dist.ReduceOp.SUM)
            std /= dist.get_world_size()
        self.pp_model.residual_std.copy_(std.view(1, -1, 1, 1))
        self._residual_std_ready = True
        if world_rank == 0:
            pretty = '  '.join(f'{n}={v:.4g}' for n, v in zip(self.field_names,
                                                              std.tolist()))
            logging.info(f"[slow-si] residual std: {pretty}")

    # ------------------------------------------------------------------
    # Base forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_base(self, data):
        """Roll the frozen base model to a random lead; return (member, target, lead)."""
        (in_sfc, in_ua, tgt_sfc, tgt_ua, tgt_diag, vbc) = self._prepare_inputs_batch_rollout(data)

        multi = tgt_sfc.dim() == 5
        if self.rollout_steps > 1 and not multi:
            raise RuntimeError(
                "rollout_steps > 1 needs a multi-step dataloader; use 1 otherwise.")

        n_steps = min(self.rollout_steps, tgt_sfc.shape[1]) if multi else 1
        if n_steps < 1:
            return None
        step = int(torch.randint(0, n_steps, (1,)).item())

        s_cur, ua_cur = in_sfc, in_ua
        cb = self.constant_boundary_data[:s_cur.shape[0]]
        for k in range(step + 1):
            vb = vbc[:, k] if vbc.dim() == 5 else vbc
            p_sfc, p_ua, p_diag = self.diff_model.prediction(
                surface_in=s_cur, constant_boundary=cb, varying_boundary=vb,
                upper_air_in=ua_cur, num_samples=1, device=self.device,
                temperature=float(getattr(self.params, 'base_temperature', 1.5)),
                sst_cond=self.sst_cond_batch)
            if k < step:
                s_cur, ua_cur = p_sfc.detach().float(), p_ua.detach().float()

        tgt = tgt_sfc[:, step] if multi else tgt_sfc
        tgt_d = None
        if tgt_diag is not None:
            tgt_d = tgt_diag[:, step] if tgt_diag.dim() == 5 else tgt_diag
        member = self._gather(p_sfc, p_diag)
        target = self._gather(tgt, tgt_d)
        lead = torch.full((member.shape[0],), (step + 1) / self.max_lead_steps,
                          device=self.device, dtype=member.dtype)
        return member, target, lead

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self):
        self.epoch += 1
        self.pp_model.train()
        loss_val = 0.0

        if not self._residual_std_ready:
            self._calibrate_residual_std()

        for year_idx, loader in enumerate(self.train_data_loaders):
            sampler = self.train_samplers[year_idx]
            if sampler is not None:
                sampler.set_epoch(self.epoch)
            for i, data in enumerate(loader):
                if self.params.mode == 'test' and i >= self.params.test_iterations:
                    break
                got = self._forward_base(data)
                if got is None:
                    continue
                member, target, lead = got

                self.iters += 1
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = self.pp_model.training_step(member=member, target=target,
                                                       lead_frac=lead)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pp_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_val = loss.item()
                if i % 100 == 0:
                    logging.info(f"[slow-si] epoch={self.epoch} iter={self.iters} "
                                 f"loss={loss_val:.5f}")
                if self.wandb_enabled and world_rank == 0:
                    wandb.log({'train/slow_si_loss': loss_val,
                               'lr': self.optimizer.param_groups[0]['lr']},
                              step=self.iters)

                if self.iters % int(getattr(self.params, 'slow_ckpt_freq', 200)) == 0:
                    self._save(f"slow_si_ckpt_{self.iters}.tar")
                if self.iters % int(getattr(self.params, 'slow_val_freq', 5000)) == 0:
                    self.validate()
                    self.pp_model.train()

        return {'train_slow_si_loss': loss_val, 'epoch': self.epoch}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _scores(ens, truth, w, mask):
        """Latitude-weighted CRPS/SSR/bias over valid cells of one field."""
        M = ens.shape[0]
        wm = (w * mask.to(w.dtype))
        tot = wm.sum() + 1e-8

        skill = (((ens - truth.unsqueeze(0)).abs()) * wm).sum(dim=(-2, -1)) / tot
        skill = skill.mean()
        if M > 1:
            srt, _ = torch.sort(ens, dim=0)
            coef = (2 * torch.arange(1, M + 1, device=ens.device,
                                     dtype=ens.dtype) - M - 1).view(-1, 1, 1)
            disp = ((srt * coef).sum(dim=0) * (2.0 / (M * (M - 1))) * wm).sum() / tot
            spread2 = ((ens.var(dim=0, unbiased=True)) * wm).sum() / tot
        else:
            disp = torch.zeros((), device=ens.device, dtype=ens.dtype)
            spread2 = disp
        em = ens.mean(dim=0)
        err2 = (((em - truth) ** 2) * wm).sum() / tot
        bias = ((em - truth) * wm).sum() / tot
        return (skill - 0.5 * disp), torch.sqrt(spread2 / (err2 + 1e-12)), \
            torch.sqrt(err2), bias

    @torch.no_grad()
    def validate(self):
        """Base vs corrected, verified against the truth at several leads."""
        self.pp_model.eval()
        num_ic = int(getattr(self.params, 'slow_val_ics', 4))
        num_ens = int(getattr(self.params, 'slow_val_members', 4))
        lead_days = list(getattr(self.params, 'slow_val_lead_days', [1, 5, 10]))
        blend = float(getattr(self.params, 'slow_blend', 1.0))
        boost = float(getattr(self.params, 'slow_spread_boost', 0.0))
        td = int(getattr(self.params, 'timedelta_hours', 24))
        steps = sorted({max(1, d * 24 // td) for d in lead_days})
        max_steps = max(steps)

        lat = torch.from_numpy(np.array(self.params.lat)).float().to(self.device)
        w = torch.cos(np.pi / 180.0 * lat).view(-1, 1)

        res = {n: {s: {k: [] for k in
                       ('b_crps', 'b_ssr', 'b_rmse', 'b_bias',
                        'p_crps', 'p_ssr', 'p_rmse', 'p_bias')}
                   for s in steps} for n in self.field_names}
        seen = 0

        from torch.utils.data import default_collate
        ds = self.valid_dataset
        idxs = list(getattr(self.params, 'slow_val_indices', range(num_ic)))[:num_ic]

        for di in idxs:
            if di >= len(ds):
                break
            data = default_collate([ds[di]])
            data, sst_c = pop_sst_cond(data, self.params, self.device)
            if self.params.has_diagnostic:
                in_sfc, in_ua, tgt_sfc, _tu, tgt_diag, vbc, _t = data
                tgt_diag = tgt_diag.to(self.device, dtype=torch.float32)
            else:
                in_sfc, in_ua, tgt_sfc, _tu, vbc, _t = data
                tgt_diag = None
            in_sfc, in_ua, tgt_sfc, vbc = (
                x.to(self.device, dtype=torch.float32) for x in
                (in_sfc, in_ua, tgt_sfc, vbc))
            if tgt_sfc.dim() != 5 or tgt_sfc.shape[1] < max_steps:
                logging.warning("[slow-si] validation needs a multi-step loader "
                                "with enough leads; skipping.")
                break
            seen += 1

            per = {s: [] for s in steps}
            for _ in range(num_ens):
                s_cur, ua_cur = in_sfc, in_ua
                for k in range(max_steps):
                    vb = vbc[:, k] if vbc.dim() == 5 else vbc
                    p_sfc, p_ua, p_diag = self.diff_model.prediction(
                        surface_in=s_cur, constant_boundary=self.constant_boundary_data[:1],
                        varying_boundary=vb, upper_air_in=ua_cur, num_samples=1,
                        device=self.device,
                        temperature=float(getattr(self.params, 'base_temperature', 1.5)),
                        sst_cond=sst_c)
                    if (k + 1) in per:
                        per[k + 1].append(self._gather(p_sfc, p_diag))
                    s_cur, ua_cur = p_sfc, p_ua

            for s in steps:
                base_raw = torch.cat(per[s], dim=0)
                td_s = tgt_diag[:, s - 1] if tgt_diag is not None else None
                truth = self._gather(tgt_sfc[:, s - 1], td_s)[0]
                lf = torch.full((base_raw.shape[0],), s / self.max_lead_steps,
                                device=self.device, dtype=base_raw.dtype)
                # The SI conditions on the raw member, exactly as at inference.
                pp_raw = self.pp_model.postprocess(
                    base_raw, lead_frac=lf, draws_per_member=1,
                    n_steps=int(getattr(self.params, 'slow_sample_steps', 20)),
                    blend=blend, spread_boost=boost)
                # Scores are computed on floored fields so they describe what
                # inference writes (it clips precipitation at 0 for both).
                base_ens, pp_ens = self._apply_floors(base_raw), self._apply_floors(pp_raw)
                for fi, name in enumerate(self.field_names):
                    m = self.field_masks[fi]
                    bc, bs, br, bb = self._scores(base_ens[:, fi], truth[fi], w, m)
                    pc, ps, pr, pb = self._scores(pp_ens[:, fi], truth[fi], w, m)
                    r = res[name][s]
                    for k, v in (('b_crps', bc), ('b_ssr', bs), ('b_rmse', br),
                                 ('b_bias', bb), ('p_crps', pc), ('p_ssr', ps),
                                 ('p_rmse', pr), ('p_bias', pb)):
                        r[k].append(v.item())

        if world_rank == 0:
            lines = [f"[slow-si val@{self.iters}] {seen} ICs x {num_ens} members, "
                     f"blend={blend:.2f} boost={boost:.2f}, vs truth"]
            wlog = {}
            for name in self.field_names:
                for s in steps:
                    r = res[name][s]
                    if not r['b_crps']:
                        continue
                    d = {k: float(np.mean(v)) for k, v in r.items()}
                    day = s * td // 24
                    lines.append(
                        f"  {name:30s} lead={day:2d}d  "
                        f"SSR {d['b_ssr']:.3f}->{d['p_ssr']:.3f}  "
                        f"RMSE {d['b_rmse']:.4f}->{d['p_rmse']:.4f}  "
                        f"CRPS {d['b_crps']:.4f}->{d['p_crps']:.4f}  "
                        f"bias {d['b_bias']:+.3f}->{d['p_bias']:+.3f}")
                    tag = name[:12]
                    for k, v in d.items():
                        wlog[f'val/{tag}_lead{day}d_{k}'] = v
            logging.info('\n'.join(lines))
            if self.wandb_enabled:
                wandb.log(wlog, step=self.iters)
        return res

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save(self, name):
        if world_rank != 0:
            return
        out = os.path.join(self.params.experiment_dir, 'training_checkpoints')
        os.makedirs(out, exist_ok=True)
        torch.save({'model_state': self.pp_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.epoch, 'iters': self.iters,
                    'field_names': self.field_names,
                    'field_sources': self.field_sources,
                    'field_idxs': self.field_idxs,
                    'max_lead_steps': self.max_lead_steps},
                   os.path.join(out, name))
        logging.info(f"Saved slow-surface checkpoint: {os.path.join(out, name)}")

    def _resume(self, path):
        ck = torch.load(path, map_location=f'cuda:{self.params.local_rank}',
                        weights_only=False)
        # A checkpoint trained on a different field list (e.g. the four slow
        # fields without precipitation) has a different UNet width; fail with
        # the field lists rather than a shape-mismatch traceback.
        ck_fields = list(ck.get('field_names', []))
        if ck_fields and ck_fields != self.field_names:
            raise ValueError(
                f"Cannot resume {path}: checkpoint fields {ck_fields} != "
                f"configured slow_fields {self.field_names}. Start a new run "
                f"(drop --slow_ckpt) or restore the old slow_fields.")
        self.pp_model.load_state_dict(ck['model_state'], strict=True)
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.epoch, self.iters = ck['epoch'], ck['iters']
        self._residual_std_ready = True
        logging.info(f"Resumed {path} (epoch={self.epoch}, iters={self.iters})")

    def run(self, epochs=10):
        for _ in range(epochs):
            logs = self.train_one_epoch()
            if self.wandb_enabled and world_rank == 0:
                wandb.log(logs, step=self.epoch)
            self._save('slow_si_last.tar')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_num', default='slow_v1', type=str)
    ap.add_argument('--yaml_config', default='config/exp16_nvidia_v3.yaml', type=str)
    ap.add_argument('--config', default='S2S', type=str)
    ap.add_argument('--epochs', default=10, type=int)
    ap.add_argument('--run_iter', default=1, type=int)
    ap.add_argument('--local-rank', type=int)
    ap.add_argument('--rollout_steps', default=1, type=int,
                    help='Unroll this many steps so the SI sees long leads.')
    ap.add_argument('--slow_ckpt', default=None, type=str)
    args = ap.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['config_filepath'] = os.path.abspath(args.yaml_config)
    params['has_diagnostic'] = bool(
        hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0)
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1
    # The dataset returns one target per entry in forecast_lead_times, so it must
    # cover every step the base model is unrolled through. The config value is a
    # sparse inference list ([1, 2, 10, 15, 45]); rollout needs contiguous steps.
    params['forecast_lead_times'] = list(range(1, max(1, args.rollout_steps) + 1))
    if getattr(params, 'wandb_offline', False):
        os.environ['WANDB_MODE'] = 'offline'

    params['world_size'] = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    if params['world_size'] > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
    else:
        world_rank = 0
        local_rank = 0

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    params['local_rank'] = local_rank

    expDir = os.path.join(params.exp_dir, args.config, f"slow_surface_si_{args.run_num}")
    if world_rank == 0:
        os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)
        logging_utils.log_to_file(logger_name=None,
                                  log_filename=os.path.join(expDir, 'slow_surface_si.log'))
        logging_utils.log_versions()
        params.log()

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['run_iter'] = args.run_iter
    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']

    tr = SlowSurfaceTrainer(params, world_rank, rollout_steps=args.rollout_steps)
    if args.slow_ckpt:
        tr._resume(args.slow_ckpt)
    tr.run(epochs=args.epochs)
    logging.info('Slow-surface SI training DONE — rank %d' % world_rank)
