"""Ensemble inference with output-space postprocess-SI spread correction.

Runs the frozen base model autoregressively to produce ensemble members, then
adds a draw from the trained postprocess SI (see networks/postprocess_si.py and
train_postprocess_si.py) to each member at each lead time:

    corrected_member = member + blend * residual,   residual ~ p(ERA5 - member)

The correction is applied in *normalised* space -- the same space the SI was
trained in -- and only then inverse-transformed to physical units, so the saved
NetCDF files are directly comparable to plain inference output.

Prognostic feedback
-------------------
By default the corrected fields are written out but the *uncorrected* member is
fed back as the next step's initial condition (``--feedback base``). The SI is
trained as a single-step error model, so feeding its output back would compound
a correction it was never trained to iterate on. ``--feedback corrected``
enables the alternative if you want the perturbation to influence the
trajectory.

Usage:
    python inference_postprocess_si.py \
        --yaml_config config/exp16_nvidia_v3.yaml --config S2S \
        --run_num pp_si_infer_v1 \
        --pp_ckpt results/S2S/postprocess_si_pp_si_v1/training_checkpoints/pp_si_last.tar \
        --num_members 40 --inference_steps 45 --blend 1.0

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

from networks.postprocess_si import PostprocessSI
from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from utils import logging_utils
from utils.YParams import YParams

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()
world_size = dist.get_world_size()


class PostprocessInference(Trainer):
    """Base-model ensemble inference with postprocess-SI spread correction."""

    def __init__(self, params, world_rank, pp_ckpt: str,
                 num_members: int = 40, blend: float = 1.0,
                 feedback: str = 'base', apply_correction: bool = True,
                 years: list | None = None, max_dates: int | None = None,
                 save_vars: list | None = None, overwrite: bool = False):
        super().__init__(params, world_rank)

        self.num_members = int(num_members)
        self.blend = float(blend)
        self.feedback = feedback
        self.apply_correction = bool(apply_correction)
        self.years = set(years) if years else None
        self.max_dates = max_dates
        # Variables written to disk. All 16 variables at 45 steps is ~1.1 GB
        # per member-file, so a full sel_dates x 40-member run would need
        # several TB; restricting to the scored variables keeps it tractable.
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
        self._setup_field_spec()

        self.pp_model = None
        if self.apply_correction:
            self._load_pp_checkpoint(pp_ckpt)

    # ------------------------------------------------------------------
    # Model / checkpoint setup (mirrors train_postprocess_si.py)
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

    def _load_pp_checkpoint(self, pp_ckpt: str):
        if not pp_ckpt or not os.path.isfile(pp_ckpt):
            raise FileNotFoundError(
                f"Postprocess-SI checkpoint not found: {pp_ckpt}. Pass --pp_ckpt, "
                f"or --no_correction to run plain base-model inference.")

        ckpt = torch.load(pp_ckpt, map_location=f'cuda:{self.params.local_rank}',
                          weights_only=False)

        # The checkpoint records the field list it was trained on; the current
        # config must resolve to the same channels or the residuals would be
        # added to the wrong variables.
        ckpt_fields = ckpt.get('field_names')
        if ckpt_fields is not None and list(ckpt_fields) != list(self.field_names):
            raise ValueError(
                f"Field mismatch: checkpoint has {len(ckpt_fields)} fields, config "
                f"resolves {len(self.field_names)}. First difference at "
                f"{next((i for i, (a, b) in enumerate(zip(ckpt_fields, self.field_names)) if a != b), 'end')}. "
                f"Set pp_fields to match the training run.")

        self.pp_model = PostprocessSI(
            n_fields=self.n_fields,
            base_channels=int(getattr(self.params, 'pp_base_channels', 64)),
            c_mults=tuple(getattr(self.params, 'pp_c_mults', (1, 2, 4))),
            scale=tuple(getattr(self.params, 'pp_scale', (2, 2))),
            gamma_type=str(getattr(self.params, 'pp_gamma_type', 'brownian')),
            dropout=float(getattr(self.params, 'pp_dropout', 0.1)),
        ).to(self.device)
        self.pp_model.load_state_dict(ckpt['model_state'], strict=True)
        self.pp_model.eval()
        for p in self.pp_model.parameters():
            p.requires_grad = False

        # Lead normalisation must match training, else the lead conditioning
        # plane means something different at inference than it did in training.
        self.max_lead_steps = int(ckpt.get('max_lead_steps', 10))

        if world_rank == 0:
            logging.info(
                f"Postprocess SI loaded from {pp_ckpt} "
                f"(iters={ckpt.get('iters')}, {self.n_fields} fields, "
                f"max_lead_steps={self.max_lead_steps}, blend={self.blend})")
            std = self.pp_model.residual_std.flatten()
            logging.info(f"  residual_std range: {std.min():.4g} .. {std.max():.4g}")

    # ------------------------------------------------------------------
    # Field spec (must match the training run)
    # ------------------------------------------------------------------

    def _setup_field_spec(self):
        surf_vars = list(self.valid_dataset.surface_variables)
        diag_vars = (list(self.valid_dataset.diagnostic_variables)
                     if self.params.has_diagnostic else [])
        ua_vars = list(self.valid_dataset.upper_air_variables)
        levels = list(self.params.levels)

        requested = getattr(self.params, 'pp_fields', 'all')
        if isinstance(requested, str):
            requested = [requested]
        requested = list(requested)

        spec = []
        if len(requested) == 1 and str(requested[0]).lower() == 'all':
            for i, v in enumerate(surf_vars):
                spec.append((v, 'surface', i))
            for i, v in enumerate(diag_vars):
                spec.append((v, 'diagnostic', i))
            for vi, v in enumerate(ua_vars):
                for li, lev in enumerate(levels):
                    spec.append((f'{v}_{lev}', 'upper_air', (vi, li)))
        else:
            for name in requested:
                name = str(name)
                if name in surf_vars:
                    spec.append((name, 'surface', surf_vars.index(name)))
                elif name in diag_vars:
                    spec.append((name, 'diagnostic', diag_vars.index(name)))
                elif name in ua_vars:
                    vi = ua_vars.index(name)
                    for li, lev in enumerate(levels):
                        spec.append((f'{name}_{lev}', 'upper_air', (vi, li)))
                elif '_' in name and name.rsplit('_', 1)[0] in ua_vars:
                    base, lev_s = name.rsplit('_', 1)
                    try:
                        lev = int(lev_s)
                    except ValueError:
                        continue
                    if lev in levels:
                        spec.append((name, 'upper_air',
                                     (ua_vars.index(base), levels.index(lev))))

        if not spec:
            raise ValueError("No valid pp_fields resolved; check params.pp_fields.")

        self.field_spec = spec
        self.field_names = [s[0] for s in spec]
        self.n_fields = len(spec)

    # ------------------------------------------------------------------
    # Correction: gather -> SI residual -> scatter back
    # ------------------------------------------------------------------

    def _gather_fields(self, sfc, ua, diag) -> torch.Tensor:
        planes = []
        for name, source, idx in self.field_spec:
            if source == 'surface':
                planes.append(sfc[:, idx])
            elif source == 'diagnostic':
                if diag is None:
                    raise ValueError(f"field '{name}' is diagnostic but none supplied")
                planes.append(diag[:, idx])
            else:
                vi, li = idx
                planes.append(ua[:, vi, li])
        return torch.stack(planes, dim=1)

    def _scatter_fields(self, corrected: torch.Tensor, sfc, ua, diag):
        """Write corrected planes back into clones of the model outputs."""
        sfc_out = sfc.clone()
        ua_out = ua.clone()
        diag_out = diag.clone() if diag is not None else None

        for ci, (name, source, idx) in enumerate(self.field_spec):
            plane = corrected[:, ci]
            if source == 'surface':
                sfc_out[:, idx] = plane
            elif source == 'diagnostic':
                if diag_out is not None:
                    diag_out[:, idx] = plane
            else:
                vi, li = idx
                ua_out[:, vi, li] = plane
        return sfc_out, ua_out, diag_out

    @torch.no_grad()
    def _apply_correction(self, sfc, ua, diag, step: int):
        """Add one SI residual draw to each member, in normalised space."""
        if self.pp_model is None:
            return sfc, ua, diag

        member = self._gather_fields(sfc, ua, diag)
        lead_frac = torch.full((member.shape[0],),
                               (step + 1) / self.max_lead_steps,
                               device=member.device, dtype=member.dtype)
        residual = self.pp_model.sample_residual(
            member, lead_frac=lead_frac, num_samples=1,
            n_steps=int(getattr(self.params, 'pp_sample_steps', 20)),
        )
        corrected = member + self.blend * residual
        return self._scatter_fields(corrected, sfc, ua, diag)

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

                    # Correct in normalised space, before inv_transform.
                    c_sfc, c_ua, c_diag = self._apply_correction(
                        p_sfc, p_ua, p_diag, step)

                    # Saved output is the corrected field; the trajectory
                    # continues from whichever `--feedback` selects.
                    out_sfc[:, step + 1] = self.valid_dataset.surface_inv_transform(
                        c_sfc.cpu()).numpy()
                    out_ua[:, step + 1] = self.valid_dataset.upper_air_inv_transform(
                        c_ua.cpu()).numpy()
                    if out_diag is not None and c_diag is not None:
                        diag_phys = self.valid_dataset.diagnostic_inv_transform(
                            c_diag.cpu()).numpy()
                        for vi, vname in enumerate(self.valid_dataset.diagnostic_variables):
                            if vname in nonneg_vars:
                                diag_phys[:, vi] = np.clip(diag_phys[:, vi], 0.0, None)
                        out_diag[:, step + 1] = diag_phys

                    if self.feedback == 'corrected':
                        s_cur, ua_cur = c_sfc, c_ua
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
            # (inference.py adds a `dt_h * sample` offset here, which is only
            # correct when consecutive samples in a batch are consecutive init
            # times; this script collates one init date at a time.)
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

            tag = 'postprocess-SI corrected' if self.pp_model is not None else 'base only'
            dataset = xr.Dataset(
                data_vars=dict(), coords=coordinates,
                attrs=dict(description=(
                    f"Prediction from {self.params.nettype} run {self.params.run_num} "
                    f"({tag}, blend={self.blend}, feedback={self.feedback})")))
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
    parser.add_argument('--run_num', default='pp_si_infer', type=str)
    parser.add_argument('--yaml_config', default='config/exp16_nvidia_v3.yaml', type=str)
    parser.add_argument('--config', default='S2S', type=str)
    parser.add_argument('--run_iter', default=1, type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--pp_ckpt', default=None, type=str,
                        help='Trained postprocess-SI checkpoint.')
    parser.add_argument('--num_members', default=40, type=int,
                        help='Ensemble size; sharded across ranks.')
    parser.add_argument('--inference_steps', default=None, type=int,
                        help='Override params.inference_steps.')
    parser.add_argument('--blend', default=None, type=float,
                        help='Residual weight. Defaults to params.pp_blend. Use the '
                             'best_blend reported by validation.')
    parser.add_argument('--feedback', default='base', choices=('base', 'corrected'),
                        help="Which field continues the rollout. 'base' (default) "
                             "keeps the SI as a pure diagnostic postprocess.")
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
                        help='Only write these variables. All 16 vars is ~1.1 GB '
                             'per member-file, so a full 156-date x 40-member run '
                             'needs ~7 TB. The benchmark scores only '
                             '2m_temperature, geopotential and '
                             'total_precipitation_24hr.')
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
            log_filename=os.path.join(expDir, 'inference_postprocess_si.log'))
        logging_utils.log_versions()
        params.log()

    params['experiment_dir'] = os.path.abspath(expDir)
    params['resuming'] = False
    params['run_iter'] = args.run_iter

    blend = args.blend if args.blend is not None else float(
        getattr(params, 'pp_blend', 1.0))

    runner = PostprocessInference(
        params, world_rank,
        pp_ckpt=args.pp_ckpt,
        num_members=args.num_members,
        blend=blend,
        feedback=args.feedback,
        apply_correction=not args.no_correction,
        years=args.years,
        max_dates=args.max_dates,
        save_vars=args.save_vars,
        overwrite=args.overwrite,
    )
    runner.predict()
    logging.info('Postprocess-SI inference DONE — rank %d' % world_rank)
