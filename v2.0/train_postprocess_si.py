"""Train an output-space SI that postprocesses forecast ensembles for spread.

The base model (VAE, deterministic encoder, latent SI, Pangu decoder) is
loaded frozen and used only to generate forecast members.  A small SI in
physical space then learns

    noise  ->  (ERA5 - member_i)      conditioned on (member_i, lead)

so that at inference each base member can be perturbed by a draw from the
learned error distribution, lifting the spread-skill ratio toward 1 without
touching the base model's skill.

Usage:
    torchrun --nproc_per_node=<N> train_postprocess_si.py \
        --yaml_config config/exp16_nvidia_v3.yaml --config S2S \
        --run_num 0001 --epochs 10

    # multi-lead training via short rollouts (recommended -- the dispersion
    # deficit is worst at 8-16 day leads, and lead-1 training alone will not
    # calibrate them):
    torchrun --nproc_per_node=<N> train_postprocess_si.py \
        --yaml_config config/exp16_nvidia_v3.yaml --config S2S \
        --rollout_steps 8
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

from networks.postprocess_si import PostprocessSI
from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from utils import logging_utils
from utils.YParams import YParams

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()


class PostprocessSITrainer(Trainer):
    """Trains `PostprocessSI` on top of a fully frozen base forecast model."""

    def __init__(self, params, world_rank, rollout_steps: int = 1):
        super().__init__(params, world_rank)

        self.rollout_steps = max(1, int(rollout_steps))
        # Lead conditioning is normalised by the longest lead we ever train or
        # validate on, so the [0,1] range means the same thing in both.
        self.max_lead_steps = max(self.rollout_steps, 10)

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        self.diff_model = self._build_si_model()
        self._load_si_checkpoint()

        # Freeze the entire base model — encoders, latent SI and decoder.
        # Unlike DecoderFinetuner, nothing upstream trains here.
        for p in self.diff_model.parameters():
            p.requires_grad = False
        self.diff_model.eval()

        self.get_dataset()

        # Fields the postprocessor corrects. Surface + diagnostic 2D fields
        # are the ones the CRPS/SSR benchmark scores; upper-air z500 is added
        # as a single level so the benchmark's third variable is covered.
        self._setup_field_spec()

        self.pp_model = PostprocessSI(
            n_fields=self.n_fields,
            base_channels=int(getattr(params, 'pp_base_channels', 64)),
            c_mults=tuple(getattr(params, 'pp_c_mults', (1, 2, 4))),
            scale=tuple(getattr(params, 'pp_scale', (2, 2))),
            gamma_type=str(getattr(params, 'pp_gamma_type', 'brownian')),
            dropout=float(getattr(params, 'pp_dropout', 0.1)),
        ).to(self.device)

        n_pp = sum(p.numel() for p in self.pp_model.parameters())
        n_base = sum(p.numel() for p in self.diff_model.parameters())
        if world_rank == 0:
            print(f"Postprocess SI params: {n_pp:,}  (frozen base: {n_base:,})")
            print(f"Correcting {self.n_fields} fields: {self.field_names}")

        self.optimizer = torch.optim.Adam(
            self.pp_model.parameters(),
            lr=float(getattr(params, 'pp_lr', getattr(params, 'finetune_lr', params.lr))),
            weight_decay=params.weight_decay,
        )

        self.scaler = GradScaler()
        self._residual_std_ready = False

        self.wandb_enabled = bool(getattr(params, 'log_to_wandb', False) and wandb.run is not None)
        if self.wandb_enabled and world_rank == 0:
            wandb.define_metric('iters')
            for key in ('train/pp_si_loss', 'lr'):
                wandb.define_metric(key, step_metric='iters')
            val_leads = list(getattr(params, 'pp_val_lead_days', [1, 3, 5]))
            _tag = lambda f: (f.replace('total_precipitation_24hr', 'precip')
                               .replace('2m_temperature', 't2m')
                               .replace('geopotential_500', 'z500'))
            for var in (_tag(f) for f in self.headline_fields):
                for lead in val_leads:
                    for kind in ('base', 'pp'):
                        for metric in ('crps', 'ssr'):
                            wandb.define_metric(f'val/{var}_lead{lead}d_{kind}_{metric}',
                                                step_metric='iters')

    # ------------------------------------------------------------------
    # Base model construction (mirrors DecoderFinetuner)
    # ------------------------------------------------------------------

    def _build_si_model(self):
        return StochasticInterpolant(
            T=1000,
            VAEEncoder=self.model_vae.module,
            DETEncoder=self.model_det.module,
            params=self.params,
        ).to(self.device)

    def _load_si_checkpoint(self):
        def _load(path, module):
            ckpt = torch.load(path, map_location=f'cuda:{self.params.local_rank}',
                              weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            module.load_state_dict(state, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)

        # Overlay the finetuned base model last so its trained decoder wins over
        # the pristine deterministic checkpoint loaded above. A rollout/decoder
        # finetune checkpoint carries the full encoder + model_det + unet state,
        # so this supersedes both earlier loads.
        base_ckpt = (getattr(self.params, 'checkpoint_path_finetuned', None)
                     or getattr(self.params, 'checkpoint_path_diff', None))
        if base_ckpt and os.path.isfile(base_ckpt):
            ckpt = torch.load(base_ckpt, map_location=f'cuda:{self.params.local_rank}',
                              weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            missing, unexpected = self.diff_model.load_state_dict(state, strict=False)
            if world_rank == 0:
                loaded = {}
                for k in state:
                    loaded[k.split('.')[0]] = loaded.get(k.split('.')[0], 0) + 1
                print(f"Loaded base model from {base_ckpt}")
                print(f"  epoch={ckpt.get('epoch')} iters={ckpt.get('iters')} "
                      f"submodules={loaded}")
                if unexpected:
                    print(f"  WARNING: {len(unexpected)} unexpected keys, "
                          f"e.g. {list(unexpected)[:3]}")
                # `missing` is expected to be non-empty only for buffers the
                # checkpoint does not carry; flag it if a whole submodule is
                # absent, which would mean part of the base model is untrained.
                for mod in ('encoder', 'model_det', 'unet'):
                    if not any(k.startswith(mod + '.') for k in state):
                        print(f"  WARNING: checkpoint has no '{mod}.*' weights — "
                              f"that submodule keeps its earlier/random init.")
        elif world_rank == 0:
            print("WARNING: no base SI/finetune checkpoint found — the frozen base "
                  "model is randomly initialised and the postprocessor will learn "
                  "nothing useful. Set checkpoint_path_finetuned in the config.")

        self.diff_model.freeze_encoder()

    def _get_latitudes(self):
        return torch.from_numpy(np.array(self.params.lat)).float()

    # ------------------------------------------------------------------
    # Field selection
    # ------------------------------------------------------------------

    def _setup_field_spec(self):
        """Decide which 2D fields the postprocessor corrects.

        Each entry is (label, source, index) where source is one of
        'surface' | 'diagnostic' | 'upper_air'.  Upper-air entries carry a
        (var_idx, level_idx) pair, so every pressure level is its own channel.

        ``params.pp_fields`` accepts:
          * ``'all'``            -- every surface, diagnostic and upper-air
                                    level (the default): 92 planes for the
                                    standard 5 sfc + 2 diag + 5 ua x 17 levels.
          * a list of names      -- ``'2m_temperature'``, ``'geopotential_500'``
                                    (any ``<upper_air_var>_<level>``), or a bare
                                    upper-air name like ``'geopotential'`` to
                                    take all of its levels.
        """
        surf_vars = list(self.train_datasets[0].surface_variables)
        diag_vars = (list(self.train_datasets[0].diagnostic_variables)
                     if self.params.has_diagnostic else [])
        ua_vars = list(self.train_datasets[0].upper_air_variables)
        levels = list(self.params.levels)

        requested = getattr(self.params, 'pp_fields', 'all')
        if isinstance(requested, str):
            requested = [requested]
        requested = list(requested)

        spec = []

        def _add_all():
            for i, v in enumerate(surf_vars):
                spec.append((v, 'surface', i))
            for i, v in enumerate(diag_vars):
                spec.append((v, 'diagnostic', i))
            for vi, v in enumerate(ua_vars):
                for li, lev in enumerate(levels):
                    spec.append((f'{v}_{lev}', 'upper_air', (vi, li)))

        if len(requested) == 1 and str(requested[0]).lower() == 'all':
            _add_all()
        else:
            for name in requested:
                name = str(name)
                if name in surf_vars:
                    spec.append((name, 'surface', surf_vars.index(name)))
                elif name in diag_vars:
                    spec.append((name, 'diagnostic', diag_vars.index(name)))
                elif name in ua_vars:
                    # Bare upper-air variable: expand to all levels.
                    vi = ua_vars.index(name)
                    for li, lev in enumerate(levels):
                        spec.append((f'{name}_{lev}', 'upper_air', (vi, li)))
                elif '_' in name and name.rsplit('_', 1)[0] in ua_vars:
                    base, lev_s = name.rsplit('_', 1)
                    try:
                        lev = int(lev_s)
                    except ValueError:
                        if world_rank == 0:
                            print(f"  skipping '{name}': level '{lev_s}' is not an int")
                        continue
                    if lev in levels:
                        spec.append((name, 'upper_air',
                                     (ua_vars.index(base), levels.index(lev))))
                    elif world_rank == 0:
                        print(f"  skipping {name}: level {lev} not in config levels")
                elif world_rank == 0:
                    print(f"  skipping unknown pp_field '{name}'")

        if not spec:
            raise ValueError("No valid pp_fields resolved; check params.pp_fields.")

        self.field_spec = spec
        self.field_names = [s[0] for s in spec]
        self.n_fields = len(spec)

        # Fields validation actually scores. Training still learns all
        # `pp_fields`; this only limits what CRPS/SSR is computed and reported
        # for, since scoring all 92 channels costs a calibrate_blend sweep each.
        headline = list(getattr(self.params, 'pp_headline_fields',
                                ['2m_temperature', 'total_precipitation_24hr']))
        self.headline_fields = [f for f in headline if f in self.field_names]
        if not self.headline_fields:
            raise ValueError(
                f"None of pp_headline_fields={headline} are in pp_fields; "
                f"validation would score nothing.")
        # Channel indices of the scored fields, for slicing during validation.
        self.headline_idxs = [self.field_names.index(f) for f in self.headline_fields]
        if world_rank == 0:
            missing = [f for f in headline if f not in self.field_names]
            if missing:
                print(f"  NOTE: headline fields not in pp_fields, not scored: {missing}")

    def _gather_fields(self, pred_sfc, pred_ua, pred_diag) -> torch.Tensor:
        """Stack the configured fields into [B, n_fields, H, W]."""
        planes = []
        for name, source, idx in self.field_spec:
            if source == 'surface':
                planes.append(pred_sfc[:, idx])
            elif source == 'diagnostic':
                if pred_diag is None:
                    raise ValueError(
                        f"pp_field '{name}' is diagnostic but no diagnostic "
                        f"tensor was supplied.")
                planes.append(pred_diag[:, idx])
            else:
                var_idx, lev_idx = idx
                planes.append(pred_ua[:, var_idx, lev_idx])
        return torch.stack(planes, dim=1)

    # ------------------------------------------------------------------
    # Residual normalisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _calibrate_residual_std(self, n_batches: int = 8):
        """Estimate per-field residual std so all fields share one UNet.

        z500 residuals are O(100) and precipitation residuals O(1e-3); without
        per-field scaling the loss is dominated by one variable.
        """
        loader = self.train_data_loaders[0]
        sums, counts = torch.zeros(self.n_fields, device=self.device), 0

        for i, data in enumerate(loader):
            if i >= n_batches:
                break
            batch = self._forward_base(data, num_samples=1)
            if batch is None:
                continue
            member, target, _ = batch
            sums += (target - member).pow(2).mean(dim=(0, 2, 3))
            counts += 1

        if counts == 0:
            logging.warning("Residual std calibration saw no batches; using unit scale.")
            return

        std = torch.sqrt(sums / counts).clamp(min=1e-8)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(std, op=dist.ReduceOp.SUM)
            std /= dist.get_world_size()

        self.pp_model.residual_std.copy_(std.view(1, -1, 1, 1))
        self._residual_std_ready = True
        if world_rank == 0:
            pretty = '  '.join(f'{n}={v:.4g}' for n, v in zip(self.field_names, std.tolist()))
            logging.info(f"[pp-si] residual std calibrated: {pretty}")

    # ------------------------------------------------------------------
    # Base-model forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_base(self, data, num_samples: int = 1):
        """Run the frozen base model and return (member, target, lead_frac).

        With rollout_steps > 1 the model is unrolled and one step is sampled
        uniformly, so the postprocessor sees the full range of lead times
        rather than only lead 1.

        Returns None when the batch cannot supply the requested rollout depth.
        """
        (input_surface, input_upper_air, target_surface, target_upper_air,
         target_diagnostic, varying_boundary_data) = self._prepare_inputs_batch(data)

        multi_step = target_surface.dim() == 5
        if self.rollout_steps > 1 and not multi_step:
            raise RuntimeError(
                "rollout_steps > 1 requires a multi-step dataloader "
                "(target_surface with a step dimension). Use rollout_steps=1 "
                "with the single-step loader.")

        n_steps = self.rollout_steps
        if multi_step:
            n_steps = min(n_steps, target_surface.shape[1])
            if n_steps < 1:
                return None
        # Train on one randomly chosen lead per batch: unrolling every step and
        # keeping them all would multiply memory for no extra independent
        # information, since consecutive leads are highly correlated.
        target_step = int(torch.randint(0, n_steps, (1,)).item())

        s_cur, ua_cur = input_surface, input_upper_air
        cb = self.constant_boundary_data[:s_cur.shape[0]]

        for step in range(target_step + 1):
            vbc = (varying_boundary_data[:, step] if varying_boundary_data.dim() == 5
                   else varying_boundary_data)
            pred_sfc, pred_ua, pred_diag = self.diff_model.prediction(
                surface_in=s_cur,
                constant_boundary=cb,
                varying_boundary=vbc,
                upper_air_in=ua_cur,
                num_samples=num_samples,
                device=self.device,
                temperature=float(getattr(self.params, 'base_temperature', 1.0)),
            )
            if step < target_step:
                s_cur = pred_sfc.detach().float()
                ua_cur = pred_ua.detach().float()

        if multi_step:
            tgt_sfc = target_surface[:, target_step]
            tgt_ua = target_upper_air[:, target_step]
            tgt_diag = (target_diagnostic[:, target_step]
                        if self.params.has_diagnostic else None)
        else:
            tgt_sfc, tgt_ua = target_surface, target_upper_air
            tgt_diag = target_diagnostic if self.params.has_diagnostic else None

        member = self._gather_fields(pred_sfc, pred_ua, pred_diag)
        target = self._gather_fields(tgt_sfc, tgt_ua, tgt_diag)

        # prediction() repeats the batch when num_samples > 1; align targets.
        if member.shape[0] != target.shape[0]:
            target = target.repeat_interleave(num_samples, dim=0)

        lead_frac = torch.full((member.shape[0],),
                               (target_step + 1) / self.max_lead_steps,
                               device=self.device, dtype=member.dtype)
        return member, target, lead_frac

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_one_epoch(self):
        self.epoch += 1
        self.pp_model.train()
        loss_val = 0.0

        if not self._residual_std_ready:
            self._calibrate_residual_std()

        for loader in self.train_data_loaders:
            for i, data in enumerate(loader):
                if self.params.mode == 'test' and i >= self.params.test_iterations:
                    break

                batch = self._forward_base(data, num_samples=1)
                if batch is None:
                    continue
                member, target, lead_frac = batch

                self.iters += 1
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = self.pp_model.training_step(
                        member=member, target=target, lead_frac=lead_frac)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pp_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_val = loss.item()
                if i % 100 == 0:
                    logging.info(f"[pp-si] epoch={self.epoch} iter={self.iters} "
                                 f"loss={loss_val:.5f}")

                if self.wandb_enabled and world_rank == 0:
                    wandb.log({'train/pp_si_loss': loss_val,
                               'lr': self.optimizer.param_groups[0]['lr']},
                              step=self.iters)

                # Checkpointing and validation are on separate cadences:
                # validation runs a full ensemble rollout and is far more
                # expensive, so it is much less frequent than saving.
                if self.iters % int(getattr(self.params, 'pp_ckpt_freq', 200)) == 0:
                    self._save(f"pp_si_ckpt_{self.iters}.tar")

                if self.iters % int(getattr(self.params, 'pp_val_freq', 5000)) == 0:
                    self.validate_spread()
                    self.pp_model.train()

        return {'train_pp_si_loss': loss_val, 'epoch': self.epoch}

    # ------------------------------------------------------------------
    # Validation: base vs postprocessed, scored against the real target
    # ------------------------------------------------------------------

    def _validation_batches(self, num_ic: int):
        """Yield exactly `num_ic` validation samples from fixed init dates.

        Iterating `valid_data_loader` would give whichever samples the sampler
        happens to shard to this rank, so the validation set would drift
        between runs and between ranks.  When `params.sel_dates` is on, the
        dataset exposes `dates_all` (Mondays/Thursdays, May-July, 2019+) and we
        index it directly instead.

        With the default 2019 start those first four dates are 2019-05-02,
        05-06, 05-09 and 05-13 -- the same init dates as `_common_init_dates`
        in the offline CRPS/SSR benchmark config, so in-training numbers line
        up with the benchmark.
        """
        ds = self.valid_dataset
        idxs = list(getattr(self.params, 'pp_val_indices', range(num_ic)))[:num_ic]

        if not getattr(self.params, 'sel_dates', False) or not hasattr(ds, 'dates_all'):
            # No fixed-date list available; fall back to the loader and just
            # take the first batches it produces.
            if world_rank == 0:
                logging.warning("[pp-si] sel_dates is off — validation dates are "
                                "whatever the loader yields and will vary by run.")
            yield from self.valid_data_loader
            return

        if world_rank == 0:
            picked = [str(ds.dates_all[i]) for i in idxs if i < len(ds.dates_all)]
            logging.info(f"[pp-si] validation init dates: {picked}")

        from torch.utils.data import default_collate
        for i in idxs:
            if i >= len(ds):
                logging.warning(f"[pp-si] validation index {i} out of range "
                                f"({len(ds)} samples); stopping early.")
                return
            # Collate a single sample so downstream code sees the usual
            # leading batch dimension.
            yield default_collate([ds[i]])

    @staticmethod
    def _crps_and_ssr(ensemble: torch.Tensor, truth: torch.Tensor, w: torch.Tensor):
        """Latitude-weighted CRPS and SSR of an ensemble against the truth.

        SSR uses the variance form, sqrt(mean(spread^2)/mean(err^2)), matching
        the offline benchmark in benchmark-dev/metrics/CRPS/crps_ssr_main.py.

        Args:
            ensemble: [M, H, W]
            truth:    [H, W]
            w:        [H, 1] latitude weights
        """
        M = ensemble.shape[0]
        wsum = w.sum() * ensemble.shape[-1]

        skill = ((ensemble - truth.unsqueeze(0)).abs() * w).sum(dim=(-2, -1)) / wsum
        skill = skill.mean()

        if M > 1:
            # Sorted-form pairwise dispersion: O(M log M) instead of O(M^2).
            srt, _ = torch.sort(ensemble, dim=0)
            coef = (2 * torch.arange(1, M + 1, device=ensemble.device,
                                     dtype=ensemble.dtype) - M - 1).view(-1, 1, 1)
            disp = (srt * coef).sum(dim=0) * (2.0 / (M * (M - 1)))
            disp = (disp * w).sum() / wsum
        else:
            disp = torch.zeros((), device=ensemble.device, dtype=ensemble.dtype)

        crps = skill - 0.5 * disp

        ens_mean = ensemble.mean(dim=0)
        err2 = (((ens_mean - truth) ** 2) * w).sum() / wsum
        if M > 1:
            var = ensemble.var(dim=0, unbiased=True)
            spread2 = (var * w).sum() / wsum
        else:
            spread2 = torch.zeros((), device=ensemble.device, dtype=ensemble.dtype)
        ssr = torch.sqrt(spread2 / (err2 + 1e-12))

        return crps, ssr

    @torch.no_grad()
    def validate_spread(self):
        """Compare base vs postprocessed ensembles against the verifying truth.

        Note this differs from DecoderFinetuner.validate_crps_ssr, which scores
        the ensemble against its own mean and therefore reports SSR ~ 1 no
        matter how under-dispersive the model is.  Here the target field is the
        truth, so the number is a real calibration measurement.
        """
        self.pp_model.eval()

        num_ic = int(getattr(self.params, 'pp_val_ics', 4))
        num_ens = int(getattr(self.params, 'pp_val_members', 8))
        draws = int(getattr(self.params, 'pp_val_draws', 1))
        lead_days = list(getattr(self.params, 'pp_val_lead_days', [1, 3, 5]))
        timedelta_h = int(getattr(self.params, 'timedelta_hours', 24))
        lead_steps = sorted({max(1, d * 24 // timedelta_h) for d in lead_days})

        # Validation cost is num_ic x num_ens x max_steps base-model forwards,
        # so cap the rollout length and drop any lead beyond it rather than
        # silently paying for a longer unroll.
        cap = int(getattr(self.params, 'pp_val_max_steps', 5))
        dropped = [d for d, s in zip(sorted(lead_days), lead_steps) if s > cap]
        lead_steps = [s for s in lead_steps if s <= cap]
        if not lead_steps:
            lead_steps = [cap]
        lead_days = [s * timedelta_h // 24 for s in lead_steps]
        if dropped and world_rank == 0:
            logging.info(f"[pp-si] validation capped at {cap} steps; "
                         f"skipping lead days {dropped}")
        max_steps = max(lead_steps)

        lat = self._get_latitudes().to(self.device)
        w = torch.cos(np.pi / 180.0 * lat).view(-1, 1)

        # Fixed blend used for the reported postprocessed scores. The SI learns
        # the *total* member error, so blend=1 replaces the member outright and
        # tends to overshoot when the base ensemble already carries part of the
        # error; this is the knob that lands SSR near 1.
        blend = float(getattr(self.params, 'pp_blend', 1.0))

        results = {name: {d: {'base_crps': [], 'base_ssr': [], 'pp_crps': [],
                              'pp_ssr': [], 'fit_blend': []}
                          for d in lead_days} for name in self.headline_fields}
        n_seen = 0

        for data in self._validation_batches(num_ic):
            if n_seen >= num_ic:
                break
            # NB: this is the *validation* loader, whose layout differs from the
            # training one _prepare_inputs_batch expects: the targets are
            # multi-step stacks and a trailing start_time tensor is appended
            # (see GetDataset.__getitem__, `validate and lead_times` branch).
            if self.params.has_diagnostic:
                (in_sfc, in_ua, tgt_sfc, tgt_ua, tgt_diag,
                 vbc, _times) = data
            else:
                (in_sfc, in_ua, tgt_sfc, tgt_ua, vbc, _times) = data
                tgt_diag = None
            in_sfc, in_ua, tgt_sfc, tgt_ua, vbc = map(
                lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True),
                (in_sfc, in_ua, tgt_sfc, tgt_ua, vbc))
            if tgt_diag is not None:
                tgt_diag = tgt_diag.to(self.device, dtype=torch.float32,
                                       non_blocking=True)
            if tgt_sfc.dim() != 5:
                logging.warning("[pp-si] validation needs a multi-step loader; skipping.")
                break
            if tgt_sfc.shape[1] < max_steps:
                logging.warning(
                    f"[pp-si] validation loader supplies {tgt_sfc.shape[1]} steps "
                    f"but lead {max_steps} was requested; skipping.")
                break

            for b in range(in_sfc.shape[0]):
                if n_seen >= num_ic:
                    break
                n_seen += 1

                # Roll each member forward independently, keeping the fields
                # we score at the requested lead times.
                per_lead = {s: [] for s in lead_steps}
                for _ in range(num_ens):
                    s_cur = in_sfc[b:b + 1]
                    ua_cur = in_ua[b:b + 1]
                    for step in range(max_steps):
                        vbc_step = vbc[b:b + 1, step] if vbc.dim() == 5 else vbc[b:b + 1]
                        p_sfc, p_ua, p_diag = self.diff_model.prediction(
                            surface_in=s_cur,
                            constant_boundary=self.constant_boundary_data[:1],
                            varying_boundary=vbc_step,
                            upper_air_in=ua_cur,
                            num_samples=1,
                            device=self.device,
                            temperature=float(getattr(self.params, 'base_temperature', 1)),
                        )
                        if (step + 1) in per_lead:
                            per_lead[step + 1].append(
                                self._gather_fields(p_sfc, p_ua, p_diag))
                        s_cur, ua_cur = p_sfc, p_ua

                for lead_day, step in zip(lead_days, lead_steps):
                    base_ens = torch.cat(per_lead[step], dim=0)   # [M, n_fields, H, W]
                    truth = self._gather_fields(
                        tgt_sfc[b:b + 1, step - 1],
                        tgt_ua[b:b + 1, step - 1],
                        tgt_diag[b:b + 1, step - 1] if self.params.has_diagnostic else None,
                    )[0]

                    lf = torch.full((base_ens.shape[0],), step / self.max_lead_steps,
                                    device=self.device, dtype=base_ens.dtype)
                    residual = self.pp_model.sample_residual(
                        base_ens, lead_frac=lf, num_samples=draws,
                        n_steps=int(getattr(self.params, 'pp_sample_steps', 20)),
                    )
                    base_rep = base_ens.repeat_interleave(draws, dim=0)
                    pp_ens = base_rep + blend * residual

                    # Score only the headline fields: the SI still generates
                    # residuals for all channels above, but CRPS/SSR and the
                    # blend sweep run on just these.
                    for name, fi in zip(self.headline_fields, self.headline_idxs):
                        bc, bs = self._crps_and_ssr(base_ens[:, fi], truth[fi], w)
                        pc, ps = self._crps_and_ssr(pp_ens[:, fi], truth[fi], w)
                        # Best achievable blend for this field/lead, reported so
                        # the fixed pp_blend can be tuned against it.
                        fit = self.pp_model.calibrate_blend(
                            base_rep[:, fi], residual[:, fi], truth[fi], w)
                        r = results[name][lead_day]
                        r['base_crps'].append(bc.item())
                        r['base_ssr'].append(bs.item())
                        r['pp_crps'].append(pc.item())
                        r['pp_ssr'].append(ps.item())
                        r['fit_blend'].append(fit)

        if world_rank == 0:
            scored = ', '.join(self.headline_fields)
            lines = [f"[pp-si val@{self.iters}] base vs postprocessed "
                     f"({n_seen} ICs x {num_ens} members, blend={blend:.2f}, "
                     f"vs truth; {self.n_fields} fields trained, scoring {scored})"]
            wlog = {}
            for name in self.headline_fields:
                for lead_day in lead_days:
                    r = results[name][lead_day]
                    if not r['base_crps']:
                        continue
                    bc, bs = np.mean(r['base_crps']), np.mean(r['base_ssr'])
                    pc, ps = np.mean(r['pp_crps']), np.mean(r['pp_ssr'])
                    fb = np.mean(r['fit_blend'])
                    lines.append(
                        f"  {name:26s} lead={lead_day:2d}d  "
                        f"CRPS {bc:.4f} -> {pc:.4f}   SSR {bs:.3f} -> {ps:.3f}"
                        f"   best_blend={fb:.2f}")
                    tag = name.replace('total_precipitation_24hr', 'precip') \
                              .replace('2m_temperature', 't2m') \
                              .replace('geopotential_500', 'z500')
                    wlog[f'val/{tag}_lead{lead_day}d_base_crps'] = bc
                    wlog[f'val/{tag}_lead{lead_day}d_base_ssr'] = bs
                    wlog[f'val/{tag}_lead{lead_day}d_pp_crps'] = pc
                    wlog[f'val/{tag}_lead{lead_day}d_pp_ssr'] = ps
                    wlog[f'val/{tag}_lead{lead_day}d_fit_blend'] = fb

            logging.info('\n'.join(lines))
            if self.wandb_enabled:
                wandb.log(wlog, step=self.iters)

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, name):
        if world_rank != 0:
            return
        out_dir = os.path.join(self.params.experiment_dir, 'training_checkpoints')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        torch.save({
            'model_state': self.pp_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iters': self.iters,
            'field_names': self.field_names,
            'max_lead_steps': self.max_lead_steps,
        }, path)
        logging.info(f"Saved postprocess-SI checkpoint: {path}")

    def _resume(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{self.params.local_rank}',
                          weights_only=False)
        self.pp_model.load_state_dict(ckpt['model_state'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.iters = ckpt['iters']
        self._residual_std_ready = True
        logging.info(f"Resumed from {ckpt_path} (epoch={self.epoch}, iters={self.iters})")

    def run(self, epochs=10):
        for _ in range(epochs):
            logs = self.train_one_epoch()
            if self.wandb_enabled and world_rank == 0:
                wandb.log(logs, step=self.epoch)
            self._save('pp_si_last.tar')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', default='0001', type=str)
    parser.add_argument('--yaml_config', default='config/exp16_nvidia_v3.yaml', type=str)
    parser.add_argument('--config', default='S2S', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--run_iter', default=1, type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--rollout_steps', default=1, type=int,
                        help='Unroll this many steps so the SI sees multiple lead '
                             'times. Requires a multi-step dataloader.')
    parser.add_argument('--pp_ckpt', default=None, type=str,
                        help='Path to a postprocess-SI checkpoint to resume from')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    params['has_diagnostic'] = bool(
        hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0)

    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1

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

    expDir = os.path.join(params.exp_dir, args.config, f"postprocess_si_{args.run_num}")
    if world_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, 'postprocess_si.log'))
        logging_utils.log_versions()
        params.log()

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['run_iter'] = args.run_iter
    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']

    trainer = PostprocessSITrainer(params, world_rank, rollout_steps=args.rollout_steps)
    if args.pp_ckpt:
        trainer._resume(args.pp_ckpt)
    trainer.run(epochs=args.epochs)
    logging.info('Postprocess-SI training DONE — rank %d' % world_rank)
