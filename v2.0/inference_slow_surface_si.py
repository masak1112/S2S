"""Ensemble inference with slow-surface SI correction.

Runs the frozen base model autoregressively to produce ensemble members, then
corrects only the four slow surface fields at each lead time with a draw from
the trained slow-surface SI (see networks/slow_surface_si.py and
train_slow_surface_si.py):

    corrected_field = field + blend * residual,   residual ~ p(ERA5 - member)

Only sea_surface_temperature, skin_temperature, soil_temperature_level_1 and
volumetric_soil_water_layer_1 are touched. Every other variable is written
through from the base model unchanged, so this output can be compared directly
against a plain inference run to isolate the slow-field effect.

The correction is applied in *normalised* space -- the space the SI was trained
in -- and only then inverse-transformed to physical units.

Masking
-------
The SI zeroes its residual outside each field's valid region (SST over ocean,
soil fields over land; see the field_masks buffer carried in the checkpoint).
Because the correction is additive, masked-out cells pass through untouched,
so land cells of SST keep whatever fill value the base model produced.

Prognostic feedback
-------------------
By default the corrected fields are written out but the *uncorrected* member
continues the rollout (``--feedback base``), keeping the SI a pure diagnostic
postprocess. ``--feedback corrected`` feeds the corrected slow fields back as
the next step's initial condition, which lets the correction influence the
trajectory -- physically interesting for slow boundary forcing, but it iterates
a correction the SI was not trained to compound.

Usage:
    python inference_slow_surface_si.py \
        --yaml_config config/exp16_nvidia_v3.yaml --config S2S \
        --run_num slow_si_infer_v1 \
        --slow_ckpt results/S2S/slow_surface_si_slow_v1/training_checkpoints/slow_si_ckpt_1600.tar \
        --num_members 20 --inference_steps 45 --blend 1.0

Output layout matches inference.py so the CRPS/SSR benchmark's format_b loader
reads it unchanged:
    {experiment_dir}/predictions/{nettype}_{run_num}_{dt}h_{steps}step_{YYYYMMDDHH}_ens_{N}.nc
"""

import argparse
import logging
import os
import shutil
import time
from collections import OrderedDict
from datetime import timedelta

import cf_xarray  # noqa: F401  (registers the .cf accessor used below)
import numpy as np
import torch
import torch.distributed as dist
import xarray as xr

from networks.slow_surface_si import SlowSurfaceSI
from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from utils import logging_utils
from utils.YParams import YParams

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()
world_size = dist.get_world_size()

DEFAULT_FIELDS = ['sea_surface_temperature', 'skin_temperature',
                  'soil_temperature_level_1', 'volumetric_soil_water_layer_1']


class SlowSurfaceInference(Trainer):
    """Base-model ensemble inference with slow-surface SI correction."""

    def __init__(self, params, world_rank, slow_ckpt: str,
                 num_members: int = 20, blend: float = 1.0,
                 spread_boost: float = 0.0, feedback: str = 'base',
                 apply_correction: bool = True,
                 years: list | None = None, max_dates: int | None = None,
                 save_vars: list | None = None, overwrite: bool = False):
        super().__init__(params, world_rank)

        self.num_members = int(num_members)
        self.blend = float(blend)
        self.spread_boost = float(spread_boost)
        self.feedback = feedback
        self.apply_correction = bool(apply_correction)
        self.years = set(years) if years else None
        self.max_dates = max_dates
        self.save_vars = set(save_vars) if save_vars else None
        # When False, members whose output file already exists are skipped.
        self.overwrite = bool(overwrite)
        # Trainer does not define this (inference.py's Stepper does), so set it
        # here for the output-array allocation below.
        self.num_diagnostic_vars = (len(self.params.diagnostic_variables)
                                    if self.params.has_diagnostic else 0)

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        self.diff_model = self._build_si_model()
        self._load_base_checkpoint()
        for p in self.diff_model.parameters():
            p.requires_grad = False
        self.diff_model.eval()

        self.get_dataset()

        self.pp_model = None
        # max_lead_steps is overwritten from the checkpoint when a correction is
        # loaded; the default only matters for --no_correction runs.
        self.max_lead_steps = int(getattr(params, 'slow_max_lead_steps', 45))
        if self.apply_correction:
            self._load_slow_checkpoint(slow_ckpt)

    # ------------------------------------------------------------------
    # Model / checkpoint setup (mirrors train_slow_surface_si.py)
    # ------------------------------------------------------------------

    def _build_si_model(self):
        return StochasticInterpolant(
            T=1000,
            VAEEncoder=self.model_vae.module,
            DETEncoder=self.model_det.module,
            params=self.params,
        ).to(self.device)

    def _load_base_checkpoint(self):
        def _load(path, module):
            ckpt = torch.load(path, map_location=f'cuda:{self.params.local_rank}',
                              weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            module.load_state_dict(state, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)

        base_ckpt = (getattr(self.params, 'checkpoint_path_finetuned', None)
                     or getattr(self.params, 'checkpoint_path_diff', None))
        if base_ckpt and os.path.isfile(base_ckpt):
            ckpt = torch.load(base_ckpt, map_location=f'cuda:{self.params.local_rank}',
                              weights_only=False)
            state = ckpt['model_state']
            if any(k.startswith('module.') for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            self.diff_model.load_state_dict(state, strict=False)
            if world_rank == 0:
                logging.info(f"Base model loaded from {base_ckpt} "
                             f"(epoch={ckpt.get('epoch')}, iters={ckpt.get('iters')})")
        elif world_rank == 0:
            logging.warning("No base checkpoint found — base model is randomly "
                            "initialised. Set checkpoint_path_finetuned.")
        self.diff_model.freeze_encoder()

    def _load_slow_checkpoint(self, slow_ckpt: str):
        if not slow_ckpt or not os.path.isfile(slow_ckpt):
            raise FileNotFoundError(
                f"Slow-surface SI checkpoint not found: {slow_ckpt}. Pass "
                f"--slow_ckpt, or --no_correction for plain base-model inference.")

        ckpt = torch.load(slow_ckpt, map_location=f'cuda:{self.params.local_rank}',
                          weights_only=False)

        # The checkpoint records the field names, which tensor each came from
        # ('surface' or 'diagnostic') and the channel index within it. Trusting
        # the checkpoint rather than re-deriving means a reordered variable
        # list cannot silently send residuals to the wrong channels.
        # Checkpoints written before diagnostic fields were supported carry no
        # 'field_sources'; every field in them is a surface channel.
        self.field_names = list(ckpt['field_names'])
        self.field_idxs = list(ckpt['field_idxs'])
        self.field_sources = list(ckpt.get('field_sources',
                                           ['surface'] * len(self.field_names)))
        self.n_fields = len(self.field_names)
        self.field_spec = list(zip(self.field_names, self.field_sources,
                                   self.field_idxs))
        self.has_diag_fields = 'diagnostic' in self.field_sources

        surf_vars = list(self.valid_dataset.surface_variables)
        diag_vars = list(getattr(self.valid_dataset, 'diagnostic_variables', []) or [])
        mismatch = []
        for n, src, i in self.field_spec:
            pool = surf_vars if src == 'surface' else diag_vars
            if i >= len(pool) or pool[i] != n:
                mismatch.append((n, src, i))
        if mismatch:
            raise ValueError(
                f"Checkpoint field/channel map does not match the dataset. "
                f"Offending (name, source, channel): {mismatch}. Dataset surface: "
                f"{surf_vars}; diagnostic: {diag_vars}. The config's variable "
                f"lists must match the training run.")
        if self.has_diag_fields and not self.params.has_diagnostic:
            raise ValueError(
                "Checkpoint corrects diagnostic fields but the config has no "
                "diagnostic_variables; the base model would produce nothing to correct.")

        # field_masks is a registered buffer, so its shape is fixed at
        # construction and must already match the checkpoint's [n_fields, H, W]
        # for strict loading; the constructor's default is a [n_fields, 1, 1]
        # placeholder. The values below are overwritten by load_state_dict —
        # only the shape matters here.
        ck_masks = ckpt['model_state']['field_masks']
        self.pp_model = SlowSurfaceSI(
            n_fields=self.n_fields,
            base_channels=int(getattr(self.params, 'slow_base_channels', 32)),
            c_mults=tuple(getattr(self.params, 'slow_c_mults', (1, 2, 4))),
            scale=tuple(getattr(self.params, 'slow_scale', (2, 2))),
            gamma_type=str(getattr(self.params, 'slow_gamma_type', 'brownian')),
            dropout=float(getattr(self.params, 'slow_dropout', 0.1)),
            field_masks=torch.zeros_like(ck_masks[0], dtype=torch.bool),
        ).to(self.device)
        # strict=True: residual_std and field_masks are buffers in the
        # checkpoint, so the calibrated std and the masks come along with the
        # weights and must not be silently replaced by constructor defaults.
        self.pp_model.load_state_dict(ckpt['model_state'], strict=True)
        self.pp_model.eval()
        for p in self.pp_model.parameters():
            p.requires_grad = False

        # Lead normalisation must match training, else the lead conditioning
        # plane means something different at inference than it did in training.
        self.max_lead_steps = int(ckpt.get('max_lead_steps', 45))

        if world_rank == 0:
            logging.info(
                f"Slow-surface SI loaded from {slow_ckpt} "
                f"(epoch={ckpt.get('epoch')}, iters={ckpt.get('iters')}, "
                f"max_lead_steps={self.max_lead_steps}, blend={self.blend}, "
                f"spread_boost={self.spread_boost})")
            std = self.pp_model.residual_std.flatten().tolist()
            masks = self.pp_model.field_masks[0]
            for (name, src, ch), s, m in zip(self.field_spec, std, masks):
                logging.info(f"  {name:32s} {src:10s} ch {ch:2d}  "
                             f"residual_std={s:.4g}  "
                             f"valid on {100.0 * m.float().mean().item():5.1f}% of grid")

    # ------------------------------------------------------------------
    # Correction: gather -> SI residual -> scatter back
    # ------------------------------------------------------------------

    def _gather(self, sfc: torch.Tensor,
                diag: torch.Tensor | None = None) -> torch.Tensor:
        """Pull the corrected fields out of their tensors, in checkpoint order."""
        planes = []
        for name, src, i in self.field_spec:
            if src == 'surface':
                planes.append(sfc[:, i])
            else:
                if diag is None:
                    raise ValueError(f"field '{name}' is diagnostic but the base "
                                     f"model returned no diagnostic tensor.")
                planes.append(diag[:, i])
        return torch.stack(planes, dim=1)

    @torch.no_grad()
    def _apply_correction(self, sfc: torch.Tensor, diag: torch.Tensor | None,
                          step: int):
        """Correct one member's fields; returns new (surface, diagnostic) tensors.

        Only the channels the checkpoint names are replaced. Diagnostics come
        back unchanged (or None) when the checkpoint has no diagnostic fields.

        `step` is 0-based, so lead step `step + 1` matches the training
        convention in train_slow_surface_si._forward_base.
        """
        if self.pp_model is None:
            return sfc, diag

        member = self._gather(sfc, diag)
        lead_frac = torch.full((member.shape[0],),
                               (step + 1) / self.max_lead_steps,
                               device=member.device, dtype=member.dtype)
        # postprocess() returns member + blend * residual with the residual
        # zeroed outside each field's valid mask, so invalid cells are
        # unchanged rather than zeroed.
        corrected = self.pp_model.postprocess(
            member, lead_frac=lead_frac, draws_per_member=1,
            n_steps=int(getattr(self.params, 'slow_sample_steps', 20)),
            blend=self.blend, spread_boost=self.spread_boost)

        sfc_out = sfc.clone()
        diag_out = diag.clone() if (diag is not None and self.has_diag_fields) else diag
        for ci, (_n, src, ch) in enumerate(self.field_spec):
            if src == 'surface':
                sfc_out[:, ch] = corrected[:, ci]
            else:
                diag_out[:, ch] = corrected[:, ci]
        return sfc_out, diag_out

    # ------------------------------------------------------------------
    # Date selection
    # ------------------------------------------------------------------

    def _inference_date_indices(self):
        """Indices into valid_dataset covering every init date to forecast.

        With `sel_dates` on, the dataset's `__len__` is `len(dates_all)` --
        Mondays/Thursdays, May-July, for 2019 onward -- and `ds[i]` maps
        straight to `dates_all[i]`. All of them are run unless restricted by
        --years or --max_dates.
        """
        ds = self.valid_dataset
        n = len(ds)
        idxs = list(range(n))

        dates_all = getattr(ds, 'dates_all', None)
        if dates_all is None:
            if world_rank == 0:
                logging.info(f"sel_dates off — running all {n} dataset samples.")
            return idxs

        # Restrict to requested years, and skip inits that save_prediction
        # would discard anyway (it only writes 00UTC).
        keep = []
        skipped_hour = 0
        for i in idxs:
            d = dates_all[i]
            if self.years and d.year not in self.years:
                continue
            if getattr(d, 'hour', 0) != 0:
                skipped_hour += 1
                continue
            keep.append(i)

        if self.max_dates is not None:
            keep = keep[:self.max_dates]

        if world_rank == 0:
            yrs = sorted({dates_all[i].year for i in keep})
            logging.info(
                f"Running {len(keep)} init dates from {len(dates_all)} sel_dates "
                f"(years {yrs}"
                + (f", {skipped_hour} non-00UTC skipped" if skipped_hour else "") + ")")
            if keep:
                logging.info(f"  first: {dates_all[keep[0]]}   "
                             f"last: {dates_all[keep[-1]]}")
        return keep

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self):
        steps = int(self.params['inference_steps'])
        # The loader only fetches boundary conditions out to
        # forecast_lead_times[-1] steps, so a longer rollout would index past
        # the end of `vbc`.
        max_avail = int(self.params.forecast_lead_times[-1])
        if steps > max_avail:
            if world_rank == 0:
                logging.warning(
                    f"inference_steps={steps} exceeds forecast_lead_times[-1]="
                    f"{max_avail}; clipping. Extend forecast_lead_times to go further.")
            steps = max_avail
        # Keep params in sync — save_prediction uses it for the filename and
        # the valid-time range.
        self.params['inference_steps'] = steps

        if world_rank == 0 and steps > self.max_lead_steps:
            logging.warning(
                f"inference_steps={steps} exceeds the SI's max_lead_steps="
                f"{self.max_lead_steps}; lead_frac will exceed 1.0 beyond step "
                f"{self.max_lead_steps}, which is outside the trained range.")

        nonneg_vars = {'total_precipitation_24hr', 'total_precipitation_6hr',
                       'mean_top_net_long_wave_radiation_flux'}

        # Every rank walks the FULL date list and shards only the ensemble
        # members. Iterating valid_data_loader instead would apply
        # DistributedSampler on top of the member split -- each rank would then
        # cover only its own dates AND its own members, so most (date, member)
        # pairs would never be computed. Its drop_last=True would also silently
        # discard the tail of the date list.
        my_members = list(range(world_rank, self.num_members, world_size))
        date_idxs = self._inference_date_indices()
        logging.info(f"rank {world_rank}: {len(date_idxs)} init dates x "
                     f"{len(my_members)} members {my_members}")

        from torch.utils.data import default_collate

        t0 = time.time()
        for batch_idx in date_idxs:
            data = default_collate([self.valid_dataset[batch_idx]])
            if self.params.has_diagnostic:
                in_sfc0, in_ua0, _ts, _tu, _td, vbc, times = data
            else:
                in_sfc0, in_ua0, _ts, _tu, vbc, times = data
            in_sfc0, in_ua0, vbc = (
                x.to(self.device, dtype=torch.float32, non_blocking=True)
                for x in (in_sfc0, in_ua0, vbc))

            start_times = [
                self.valid_dataset.datetime_class(
                    times[i, 0].item(), times[i, 1].item(), times[i, 2].item(),
                    hour=times[i, 3].item())
                for i in range(times.shape[0])
            ]

            B = in_sfc0.shape[0]
            cb = self.constant_boundary_data[:B]

            for ens_id in my_members:
                # Resume support: a member already written by an earlier run is
                # skipped before any GPU work, so an interrupted job can be
                # relaunched with the same --run_num and will only fill gaps.
                # save_prediction only writes samples whose init hour is 00, so
                # a date is complete when every 00UTC sample has its file.
                if not self.overwrite:
                    done = [self._prediction_path(st, ens_id)
                            for st in start_times if st.strftime('%H') == '00']
                    if done and all(os.path.exists(f) for f in done):
                        logging.info(
                            f"rank {world_rank}: skipping existing "
                            f"{os.path.basename(done[0])}")
                        continue

                # Seed per (init date, member) so members differ, runs are
                # reproducible, and no two pairs collide.
                seed = 100_000 * batch_idx + ens_id
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                out_sfc = np.zeros((B, steps + 1, in_sfc0.shape[1],
                                    in_sfc0.shape[2], in_sfc0.shape[3]), np.float32)
                out_ua = np.zeros((B, steps + 1, in_ua0.shape[1], in_ua0.shape[2],
                                   in_ua0.shape[3], in_ua0.shape[4]), np.float32)
                out_diag = (np.zeros((B, steps + 1, self.num_diagnostic_vars,
                                      in_sfc0.shape[2], in_sfc0.shape[3]), np.float32)
                            if self.params.has_diagnostic else None)

                out_sfc[:, 0] = self.valid_dataset.surface_inv_transform(
                    in_sfc0.cpu()).numpy()
                out_ua[:, 0] = self.valid_dataset.upper_air_inv_transform(
                    in_ua0.cpu()).numpy()

                s_cur, ua_cur = in_sfc0, in_ua0

                for step in range(steps):
                    p_sfc, p_ua, p_diag = self.diff_model.prediction(
                        surface_in=s_cur,
                        constant_boundary=cb,
                        varying_boundary=vbc[:, step],
                        upper_air_in=ua_cur,
                        num_samples=1,
                        device=self.device,
                        temperature=float(getattr(self.params, 'base_temperature', 1.5)),
                        mc_dropout=bool(getattr(self.params, 'infer_mc_dropout', True)),
                    )

                    # Correct in normalised space, before inv_transform. The
                    # surface tensor and (if the checkpoint includes
                    # precipitation) the diagnostic tensor are touched; upper
                    # air passes through from the base model. Diagnostics are
                    # never fed back, so their correction is a pure postprocess.
                    c_sfc, c_diag = self._apply_correction(p_sfc, p_diag, step)

                    # Saved output is the corrected field; the trajectory
                    # continues from whichever `--feedback` selects.
                    out_sfc[:, step + 1] = self.valid_dataset.surface_inv_transform(
                        c_sfc.cpu()).numpy()
                    out_ua[:, step + 1] = self.valid_dataset.upper_air_inv_transform(
                        p_ua.cpu()).numpy()
                    if out_diag is not None and c_diag is not None:
                        # nonneg clip below also bounds the corrected precipitation.
                        diag_phys = self.valid_dataset.diagnostic_inv_transform(
                            c_diag.cpu()).numpy()
                        for vi, vname in enumerate(self.valid_dataset.diagnostic_variables):
                            if vname in nonneg_vars:
                                diag_phys[:, vi] = np.clip(diag_phys[:, vi], 0.0, None)
                        out_diag[:, step + 1] = diag_phys

                    if self.feedback == 'corrected':
                        s_cur, ua_cur = c_sfc, p_ua
                    else:
                        s_cur, ua_cur = p_sfc, p_ua

                self.save_prediction(out_sfc, out_ua, start_times,
                                     diagnostic_prediction=out_diag, ens_id=ens_id)

            logging.info(f"rank {world_rank}: batch {batch_idx} done "
                         f"({time.time() - t0:.0f}s elapsed)")

        if dist.is_initialized():
            dist.barrier()
        if world_rank == 0:
            logging.info(f"All members written in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------
    # Saving (format matches inference.py / benchmark format_b loader)
    # ------------------------------------------------------------------

    def _prediction_path(self, start_time, ens_id):
        """Output path for one (init date, member), matching save_prediction.

        Kept in sync with the filename built in save_prediction below; both
        depend only on values known before the rollout starts, which is what
        makes the existence check in predict() possible.
        """
        savedir = os.path.join(self.params['experiment_dir'], 'predictions')
        filename = '%s_%s_%dh_%dstep_%s_ens_%s.nc' % (
            self.params.nettype, self.params.run_num,
            self.params['timedelta_hours'], self.params['inference_steps'],
            start_time.strftime('%Y%m%d%H'), ens_id)
        return os.path.join(savedir, filename)

    def save_prediction(self, surface_prediction, upper_air_prediction, start_times,
                        diagnostic_prediction=None, ens_id=None):
        savedir = os.path.join(self.params['experiment_dir'], 'predictions')
        os.makedirs(savedir, exist_ok=True)

        cfg_src = getattr(self.params, 'config_filepath', None)
        if cfg_src and os.path.isfile(cfg_src):
            dst = os.path.join(self.params['experiment_dir'], os.path.basename(cfg_src))
            if not os.path.exists(dst):
                shutil.copy(cfg_src, dst)

        dt_h = self.params['timedelta_hours']
        steps = self.params['inference_steps']

        for sample in range(surface_prediction.shape[0]):
            if start_times[sample].strftime('%H') != '00':
                continue

            # Valid times run from the init time out to init + steps*dt.
            time_range = xr.cftime_range(
                start_times[sample],
                start_times[sample] + timedelta(hours=dt_h * steps),
                freq="%dh" % dt_h, inclusive="both")

            coordinates = {'time': time_range,
                           'level': self.params.levels,
                           'latitude': self.params.lat,
                           'longitude': self.params.lon}

            filename = '%s_%s_%dh_%dstep_%s_ens_%s.nc' % (
                self.params.nettype, self.params.run_num, dt_h, steps,
                start_times[sample].strftime('%Y%m%d%H'), ens_id)

            if self.pp_model is not None:
                tag = ('slow-surface-SI corrected on '
                       + ','.join(self.field_names))
            else:
                tag = 'base only'
            dataset = xr.Dataset(
                data_vars=dict(), coords=coordinates,
                attrs=dict(description=(
                    f"Prediction from {self.params.nettype} run {self.params.run_num} "
                    f"({tag}, blend={self.blend}, "
                    f"spread_boost={self.spread_boost}, feedback={self.feedback})")))
            dataset["level"].attrs['axis'] = 'Z'
            dataset['latitude'].attrs['axis'] = 'Y'
            dataset['longitude'].attrs['axis'] = 'X'
            dataset["level"].attrs['positive'] = 'down'
            dataset = dataset.cf.guess_coord_axis()

            keep = self.save_vars  # None => write everything

            for idx, var in enumerate(self.valid_dataset.surface_variables):
                if keep is not None and var not in keep:
                    continue
                dataset[var] = xr.DataArray(
                    data=surface_prediction[sample, :, idx],
                    dims=["time", "latitude", "longitude"],
                    coords={'time': time_range,
                            'latitude': dataset.latitude.values,
                            'longitude': dataset.longitude.values})
            for idx, var in enumerate(self.valid_dataset.upper_air_variables):
                if keep is not None and var not in keep:
                    continue
                dataset[var] = xr.DataArray(
                    data=upper_air_prediction[sample, :, idx],
                    dims=["time", "level", "latitude", "longitude"],
                    coords=coordinates)
            if self.params.has_diagnostic and diagnostic_prediction is not None:
                for idx, var in enumerate(self.valid_dataset.diagnostic_variables):
                    if keep is not None and var not in keep:
                        continue
                    dataset[var] = xr.DataArray(
                        data=diagnostic_prediction[sample, :, idx],
                        dims=["time", "latitude", "longitude"],
                        coords={'time': time_range,
                                'latitude': dataset.latitude.values,
                                'longitude': dataset.longitude.values})

            dataset["latitude"] = dataset["latitude"].astype('float32').assign_attrs(
                {'long_name': 'Latitude', 'unit': 'degrees_north'})
            dataset["longitude"] = dataset["longitude"].astype('float32').assign_attrs(
                {'long_name': 'Longitude', 'unit': 'degrees_east'})
            dataset["time"] = dataset["time"].assign_attrs(
                {'long_name': "Forecast Valid Time"})
            dataset["level"] = dataset["level"].astype('float32').assign_attrs(
                {'long_name': 'Level', 'unit': 'hPa'})
            dataset = dataset.chunk({'time': 1, "level": 1})
            dataset.to_netcdf(os.path.join(savedir, filename), 'w')
            logging.info(f"saved {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', default='slow_si_infer', type=str)
    parser.add_argument('--yaml_config', default='config/exp16_nvidia_v3.yaml', type=str)
    parser.add_argument('--config', default='S2S', type=str)
    parser.add_argument('--run_iter', default=1, type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--slow_ckpt', default=None, type=str,
                        help='Trained slow-surface-SI checkpoint.')
    parser.add_argument('--num_members', default=20, type=int,
                        help='Ensemble size; sharded across ranks.')
    parser.add_argument('--inference_steps', default=None, type=int,
                        help='Override params.inference_steps.')
    parser.add_argument('--blend', default=None, type=float,
                        help='Residual weight. Defaults to params.slow_blend.')
    parser.add_argument('--spread_boost', default=None, type=float,
                        help='Inflate member-to-member residual deviation. '
                             'Defaults to params.slow_spread_boost.')
    parser.add_argument('--feedback', default='base', choices=('base', 'corrected'),
                        help="Which surface field continues the rollout. 'base' "
                             "(default) keeps the SI a pure diagnostic postprocess.")
    parser.add_argument('--no_correction', action='store_true',
                        help='Run base-model inference only — the control run to '
                             'compare corrected output against.')
    parser.add_argument('--years', default=None, type=int, nargs='+',
                        help='Restrict sel_dates to these init years (default: all).')
    parser.add_argument('--max_dates', default=None, type=int,
                        help='Cap the number of init dates (default: all).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate members whose output file already '
                             'exists. Default is to skip them, so an '
                             'interrupted run can be resumed by relaunching '
                             'with the same --run_num.')
    parser.add_argument('--save_vars', default=None, type=str, nargs='+',
                        help='Only write these variables. Writing all variables '
                             'costs ~1.1 GB per member-file; the slow-field '
                             'benchmark only needs the four corrected fields.')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['config_filepath'] = os.path.abspath(args.yaml_config)

    params['has_diagnostic'] = bool(
        hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0)
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1
    params['log_to_wandb'] = False

    if args.inference_steps is not None:
        params['inference_steps'] = args.inference_steps
    elif 'inference_steps' not in params:
        params['inference_steps'] = 45

    params['world_size'] = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    if params['world_size'] > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
    else:
        world_rank = 0
        local_rank = 0

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    params['local_rank'] = local_rank
    params['run_num'] = args.run_num

    expDir = os.path.join(params.exp_dir, args.config, args.run_num)
    if world_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'predictions'), exist_ok=True)
        logging_utils.log_to_file(
            logger_name=None,
            log_filename=os.path.join(expDir, 'inference_slow_surface_si.log'))
        logging_utils.log_versions()
        params.log()

    params['experiment_dir'] = os.path.abspath(expDir)
    params['resuming'] = False
    params['run_iter'] = args.run_iter

    blend = args.blend if args.blend is not None else float(
        getattr(params, 'slow_blend', 1.0))
    spread_boost = args.spread_boost if args.spread_boost is not None else float(
        getattr(params, 'slow_spread_boost', 0.0))

    runner = SlowSurfaceInference(
        params, world_rank,
        slow_ckpt=args.slow_ckpt,
        num_members=args.num_members,
        blend=blend,
        spread_boost=spread_boost,
        feedback=args.feedback,
        apply_correction=not args.no_correction,
        years=args.years,
        max_dates=args.max_dates,
        save_vars=args.save_vars,
        overwrite=args.overwrite,
    )
    runner.predict()
    logging.info('Slow-surface SI inference DONE — rank %d' % world_rank)
