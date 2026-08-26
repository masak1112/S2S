from networks.pangu import PanguModel_Plasim
from networks.pangu_vae import PanguModel_Plasim_VAE
from networks.diffusion import ConditionalDiffusionModel
from networks.stochastic_interpolant import StochasticInterpolant
from tqdm import tqdm
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import wandb
from utils.data_loader_multifiles import get_data_loader, get_infer_data
from utils.YParams import YParams
import os, shutil
import time
import numpy as np
import math
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from pathlib import Path
import dask
import xarray as xr
import cf_xarray as cfxr
from datetime import timedelta
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
from utils.integrate import Integrator, forward_euler
from train import Trainer
from networks.vae import VAE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

dask.config.set(scheduler='synchronous')
torch._dynamo.config.optimize_ddp = False
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache() 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

class Stepper(Trainer):
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank, async_save=False):

        self.params = params
        self.world_rank = world_rank
        self.async_save = async_save
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        if self.async_save:
            logging.info('Asynchronous Saving')
        else: 
            logging.info('Synchronous Saving')
        self.run_uuid = str(uuid.uuid4())
        self.has_land = False
        self.has_ocean = False
        self.mask_output = False
        if hasattr(self.params, 'land_variables'):
            if len(self.params.land_variables) > 0:
                self.has_land = True
        else:
            self.params['land_variables'] = []
        if hasattr(self.params, 'ocean_variables'):
            if len(self.params.ocean_variables) > 0:
                self.has_ocean = True
        else:
            self.params['ocean_variables'] = []
        if hasattr(self.params, 'mask_output'):
            self.mask_output = params.mask_output
        
        self.num_diagnostic_vars = len(self.params.diagnostic_variables) if self.params.has_diagnostic else 0
        logging.info('rank %d, begin data loader init' % world_rank)
        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(params, params.data_dir, dist.is_initialized(),
                                                                     year_start=params.val_year_start,
                                                                     year_end=params.val_year_end, train=False,
                                                                     num_inferences = params.num_inferences, validate = True)
        print(f'Valid dataset length: {len(self.valid_dataset)}')
        self.constant_boundary_data = self.valid_dataset.constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device)
        logging.info('rank %d, data loader initialized' % world_rank)

        if params.nettype == 'pangu_plasim':
            if (self.has_land or self.has_ocean) and self.mask_output:
                land_mask = torch.clone(self.valid_dataset.land_mask.detach()).to(self.device)
                print(f'Land Mask shape: {land_mask.shape}')
                mask_bool = []
                for var in self.params.surface_variables:
                    if var in self.params.land_variables:
                        mask_bool.append(torch.clone(land_mask).to(torch.bool))
                    elif var in self.params.ocean_variables:
                        mask_bool.append(torch.logical_not(torch.clone(land_mask).to(torch.bool)))
                    else:
                        mask_bool.append(torch.ones(land_mask.shape, device=self.device, dtype=torch.bool))
                mask_bool = torch.stack(mask_bool)
            else:
                land_mask = None
            
            self.model_vae = VAE(self.params).to(self.device)
            self.model_det = PanguModel_Plasim(self.params, land_mask = land_mask, 
                                               mask_fill = self.params.mask_fill).to(self.device)

        else:
            raise Exception("not implemented")

        if params.diffusion_model_type == "SI":
            self.diff_model = StochasticInterpolant(VAEEncoder=self.model_vae, 
                                DETEncoder = self.model_det,
                                params = self.params).to(self.device)
        else:
            self.diff_model = ConditionalDiffusionModel(
                                T=1000,
                                VAEEncoder=self.model_vae, 
                                DETEncoder = self.model_det,
                                params = self.params# Use default simple encoder
                            ).to(self. device)
        self.model_diff = self.diff_model

        self.restore_diff_checkpoint(params.checkpoint_path_diff)
        self._reload_vae_checkpoint(
            checkpoint_path_vae=params.checkpoint_path_vae_c1,
            checkpoint_path_det=params.checkpoint_path_det,
        )
        finetune_ckpt = getattr(params, 'checkpoint_path_finetune', None)
        if finetune_ckpt and os.path.isfile(finetune_ckpt):
            self.restore_finetune_checkpoint(finetune_ckpt)
            print(f"Fine-tuned decoder loaded from {finetune_ckpt}")

        # Background writer: overlaps NetCDF serialization with GPU rollout of the
        # next ensemble member. Saves are submitted per (batch, ens_id) and drained
        # at the end of validate_one_epoch.
        self._save_executor = ThreadPoolExecutor(max_workers=2) if self.async_save else None
        self._pending_saves = []
        # HDF5/netCDF4 is not thread-safe in this environment: concurrent to_netcdf()
        # calls segfault the process. Dataset construction still overlaps with GPU
        # work; only the actual file write is serialized behind this lock.
        self._netcdf_write_lock = threading.Lock()
        # Pinned CPU staging buffers for async D2H copies; lazily allocated on
        # first use and reused across ensemble members.
        self._surf_cpu = None
        self._upper_cpu = None
        self._diag_cpu = None

    def restore_diff_checkpoint(self, checkpoint_path_diff):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path_diff, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
        raw_state = checkpoint['model_state']
        # Strip DDP 'module.' prefix if present
        if any(k.startswith('module.') for k in raw_state):
            raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
        # Exclude frozen encoder/model_det weights — those must come from their
        # own checkpoints (loaded via restore_checkpoint), not from the diff ckpt.
        # Loading them from here would silently corrupt the canonical encoder weights.
        unet_state = {k: v for k, v in raw_state.items()
                      if not k.startswith('encoder.') and not k.startswith('model_det.')}
        missing, unexpected = self.diff_model.load_state_dict(unet_state, strict=False)
        encoder_keys_skipped = [k for k in raw_state if k.startswith('encoder.') or k.startswith('model_det.')]
        if encoder_keys_skipped:
            print(f"restore_diff_checkpoint: skipped {len(encoder_keys_skipped)} frozen encoder keys")
        encoder_unexpected = [k for k in unexpected if not k.startswith('encoder.') and not k.startswith('model_det.')]
        if encoder_unexpected:
            print(f"restore_diff_checkpoint: unexpected non-encoder keys: {encoder_unexpected}")
        # Refresh snapshot so assert_encoder_frozen() baselines from the correct weights
        #self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']

  
    def restore_finetune_checkpoint(self, finetune_ckpt_path):
        """Load fine-tuned decoder weights (upsample, layer4, patchrecovery*) from a
        CRPS fine-tune checkpoint, overriding the frozen decoder loaded earlier.
        Only model_det.* keys are applied; UNet and encoder keys are ignored.
        """
        ckpt = torch.load(finetune_ckpt_path,
                          map_location=f'cuda:{self.params.local_rank}',
                          weights_only=False)
        state = ckpt['model_state']
        if any(k.startswith('module.') for k in state):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        decoder_state = {k: v for k, v in state.items() if k.startswith('model_det.')}
        missing, unexpected = self.diff_model.load_state_dict(decoder_state, strict=False)
        print(f"restore_finetune_checkpoint: loaded {len(decoder_state)} decoder keys "
              f"from {finetune_ckpt_path}")
        if missing:
            print(f"  missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    def _reload_vae_checkpoint(self, checkpoint_path_vae=None, checkpoint_path_det=None):
        """Explicitly load VAE and deterministic checkpoint weights into
        diff_model.encoder and diff_model.model_det respectively.
        Guarantees both always use checkpoint_path_vae / checkpoint_path_det,
        regardless of what any diff checkpoint may contain."""

        def _load(path, module):
            ckpt = torch.load(path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
            raw_state = ckpt['model_state']
            if any(k.startswith('module.') for k in raw_state):
                raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
            module.load_state_dict(raw_state, strict=True)

        _load(checkpoint_path_vae, self.diff_model.encoder)
        print("Loaded VAE weights from checkpoint_path_vae into diff_model.encoder")
        _load(checkpoint_path_det, self.diff_model.model_det)
        print("Loaded det weights from checkpoint_path_det into diff_model.model_det")
        self.model_vae = self.diff_model.encoder
        self.model_det = self.diff_model.model_det
        self.diff_model.freeze_encoder()
        
    def _get_era5_t2m(self, year):
        if not hasattr(self, '_era5_cache'):
            self._era5_cache = {}
        if year not in self._era5_cache:
            era5_dir = getattr(self.params, 'era5_dir', '/project/pedramh/bing/era5')
            fpath = os.path.join(era5_dir, f'{year}_180x360.nc')
            if os.path.exists(fpath):
                self._era5_cache[year] = xr.open_dataset(fpath, engine='netcdf4')
            else:
                self._era5_cache[year] = None
                logging.warning(f'ERA5 file not found: {fpath}')
        return self._era5_cache[year]

    def _make_diagnostic_plots(self, diag_samples, savedir):
        """Save one figure per lead time comparing GT vs two ensemble members for 3 ICs."""
        var_name = '2m_temperature'
        surf_vars = list(self.valid_dataset.surface_variables)
        if var_name not in surf_vars:
            logging.warning(f'{var_name} not in surface_variables, skipping diagnostic plots')
            return

        var_idx = surf_vars.index(var_name)
        lead_days = [1, 5, 10, 15, 30]
        timedelta_h = int(self.params['timedelta_hours'])

        for lead_day in lead_days:
            step = (lead_day * 24) // timedelta_h
            if step > self.params['inference_steps']:
                logging.info(f'Skipping diagnostic lead day {lead_day}: step {step} > inference_steps')
                continue

            panel_data = []
            for sample in diag_samples:
                start_time = sample['start_time']
                ens0_arr = sample.get('ens0')
                ens1_arr = sample.get('ens1')
                if ens0_arr is None or ens1_arr is None:
                    continue

                pred_ens0 = ens0_arr[step, var_idx]
                pred_ens1 = ens1_arr[step, var_idx]

                # Load ERA5 ground truth at the valid time
                valid_dt = start_time + timedelta(hours=step * timedelta_h)
                year = valid_dt.year
                era5_ds = self._get_era5_t2m(year)
                if era5_ds is not None:
                    ts = pd.Timestamp(valid_dt.year, valid_dt.month, valid_dt.day, valid_dt.hour)
                    lat_name = 'lat' if 'lat' in era5_ds.dims else 'latitude'
                    lon_name = 'lon' if 'lon' in era5_ds.dims else 'longitude'
                    gt_arr = (era5_ds['2m_temperature']
                              .sel(time=ts, method='nearest')
                              .values)
                    # Align ERA5 lat order with model output (both S→N)
                    era5_lats = era5_ds[lat_name].values
                    if era5_lats[0] > era5_lats[-1]:
                        gt_arr = gt_arr[::-1]
                else:
                    gt_arr = np.full(pred_ens0.shape, np.nan)

                ic_label = f"IC {start_time.strftime('%Y-%m-%d')}\nLead {lead_day}d"
                panel_data.append((gt_arr, pred_ens0, pred_ens1, ic_label))

            if not panel_data:
                continue

            all_vals = np.concatenate([
                a.ravel() for triple in panel_data for a in triple[:3]
                if not np.all(np.isnan(a))
            ])
            vmin = np.nanpercentile(all_vals, 2)
            vmax = np.nanpercentile(all_vals, 98)

            n_cols = len(panel_data)
            fig = plt.figure(figsize=(5 * n_cols, 11))
            gs = gridspec.GridSpec(4, n_cols, height_ratios=[1, 1, 1, 0.08],
                                   hspace=0.4, wspace=0.05)
            row_labels = ['Ground Truth', 'Ensemble Member 1', 'Ensemble Member 2']
            im = None
            for col, (gt_arr, pred_ens0, pred_ens1, ic_label) in enumerate(panel_data):
                for row, arr in enumerate([gt_arr, pred_ens0, pred_ens1]):
                    ax = fig.add_subplot(gs[row, col])
                    im = ax.imshow(arr, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                                   aspect='auto', origin='lower')
                    if row == 0:
                        ax.set_title(ic_label, fontsize=9)
                    ax.axis('off')
                    if col == 0:
                        ax.text(-0.08, 0.5, row_labels[row], transform=ax.transAxes,
                                fontsize=10, va='center', ha='right', rotation=90)

            if im is not None:
                cbar_ax = fig.add_subplot(gs[3, :])
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                cbar.set_label(f'{var_name} (K)', fontsize=10)

            fig.suptitle(f'{var_name}  —  Lead {lead_day} days', fontsize=12)
            fname = os.path.join(savedir, f'diag_{var_name}_lead{lead_day:02d}d.png')
            fig.savefig(fname, dpi=120, bbox_inches='tight')
            plt.close(fig)
            logging.info(f'Saved diagnostic plot: {fname}')

    def predict(self):
        if self.params.log_to_screen:
            logging.info("Starting Model Inference Loop...")
        valid_time = self.validate_one_epoch()
        if self.world_rank == 0:
            logging.info("Validation loop wall time (seconds): %.3f", valid_time)


    def validate_one_epoch(self):
        self.model_diff.eval()
        total_start = time.time()

        # Use the first 3 initial conditions (first 3 samples of the first batch)
        diag_selected = {(0, 0), (0, 1), (0, 2)}
        # diag_store: (batch_idx, sample_idx) -> {start_time, ens0, ens1}
        diag_store = {}

        # Accumulate all member outputs for batch 0 to measure spread growth per step.
        spread_tracker = []  # list of (steps+1, vars, lat, lon) arrays, one per member

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for batch_idx, data in enumerate(self.valid_data_loader, 0):
                # Host->device once per batch (not once per member): the 50 ensemble
                # members all start from the same IC, so re-uploading it 50x was pure
                # overhead. Each member clones the GPU copy instead.
                val_input_surface_batch, val_input_upper_air_batch, _, _, _, val_varying_boundary_data_batch, times = map(
                        lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)

                # Single D2H for the timestamps, then build start_times once per batch.
                times_np = times.cpu().numpy().astype(int)
                start_times = [
                    self.valid_dataset.datetime_class(
                        times_np[i, 0], times_np[i, 1], times_np[i, 2], hour=times_np[i, 3])
                    for i in range(times_np.shape[0])
                ]

                for ens_id in list(range(0, 50)):
                    val_input_surface = val_input_surface_batch.clone()
                    val_input_upper_air = val_input_upper_air_batch.clone()
                    val_varying_boundary_data = val_varying_boundary_data_batch

                    ic_noise_std = getattr(self.params, 'ic_noise_std', 0.2)

                    # t=0 IC perturbation, scaled per-variable (member 0 = control)
                    # if ens_id > 0 and ic_noise_std > 0.0:
                    #     surf_scale = val_input_surface.std(dim=(-2, -1), keepdim=True)
                    #     ua_scale   = val_input_upper_air.std(dim=(-2, -1), keepdim=True)
                    #     val_input_surface   = val_input_surface   + ic_noise_std * surf_scale * torch.randn_like(val_input_surface)
                    #     val_input_upper_air = val_input_upper_air + ic_noise_std * ua_scale   * torch.randn_like(val_input_upper_air)
                    temperature = 1
                    base_temperature      = getattr(self.params, 'base_temperature',      1.5)
                    temperature_growth   = getattr(self.params, 'temperature_growth',   0.1)
                    step_noise_std       = getattr(self.params, 'step_noise_std',       0.03)
                    noise_growth_exp     = getattr(self.params, 'noise_growth_exp',     0.5)
                    use_gaussian_latent  = getattr(self.params, 'use_gaussian_latent',  False)
                    gaussian_latent_std  = getattr(self.params, 'gaussian_latent_std',  1.0)

                    # Keep the whole rollout on-GPU and do a single bulk D2H after the
                    # loop, instead of 3 device->host copies per step. Index 0 is the IC.
                    _surf_steps_gpu = [val_input_surface.detach()]
                    _upper_steps_gpu = [val_input_upper_air.detach()]
                    _diag_steps_gpu = []
                    if self.params.has_diagnostic:
                        # t=0 has no diagnostic output. Placeholder only, so indices line up
                        # with the surface/upper-air arrays; it is re-zeroed AFTER the inverse
                        # transform below (inv_transform(0) == mean, not 0).
                        _diag_steps_gpu.append(torch.zeros(
                            (val_input_surface.shape[0], self.num_diagnostic_vars,
                             val_input_surface.shape[2], val_input_surface.shape[3]),
                            dtype=torch.float32, device=self.device))

                    for time_step in range(self.params['inference_steps']):
                        # # Approach 1: temperature grows linearly with lead time
                        # temperature = base_temperature * (1.0 + temperature_growth * time_step)
                        # Approach 2: exponential growth — accelerates spread at longer leads
                        # temperature = base_temperature * math.exp(temperature_growth * time_step)

                        val_out_surface, val_out_upper_air, val_out_diagnostic = self.diff_model.prediction(surface_in=val_input_surface, constant_boundary=self.constant_boundary_data,
                                                                    varying_boundary=val_varying_boundary_data[:,time_step],
                                                                    upper_air_in=val_input_upper_air, device=self.device,
                                                                    mc_dropout=True, temperature=temperature,
                                                                    use_gaussian_latent=(use_gaussian_latent and ens_id > 0),
                                                                    gaussian_latent_std=gaussian_latent_std)

                        # Approach 2: per-step additive state noise with amplitude that grows with lead time.
                        # Constant noise reaches an equilibrium with the model's attractor and spread plateaus;
                        # growing noise (step+1)^noise_growth_exp overcomes the contraction at long leads.
                        # if ens_id > 0 and step_noise_std > 0.0:
                        #     noise_scale = step_noise_std * math.pow(time_step + 1, noise_growth_exp)
                        #     surf_scale = val_out_surface.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                        #     ua_scale   = val_out_upper_air.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                        #     val_out_surface   = val_out_surface   + noise_scale * surf_scale * torch.randn_like(val_out_surface)
                        #     val_out_upper_air = val_out_upper_air + noise_scale * ua_scale   * torch.randn_like(val_out_upper_air)
                        #     # Diagnostic variables (e.g. precipitation) are not fed back as inputs so they
                        #     # receive no spread from the prognostic noise chain. Perturb them directly here
                        #     # in normalized space; non-negative variables are clamped after inv_transform.
                        #     if self.params.has_diagnostic:
                        #         diag_scale = val_out_diagnostic.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                        #         val_out_diagnostic = val_out_diagnostic + noise_scale * diag_scale * torch.randn_like(val_out_diagnostic)

                        # Stay on GPU: inverse-transform and clamping are applied in bulk
                        # after the rollout completes.
                        if self.params.has_diagnostic:
                            _diag_steps_gpu.append(val_out_diagnostic.detach())
                        val_input_surface, val_input_upper_air = val_out_surface, val_out_upper_air

                        _surf_steps_gpu.append(val_out_surface.detach())
                        _upper_steps_gpu.append(val_out_upper_air.detach())

                    # ---- Bulk GPU -> CPU transfer (one sync point per member) ----
                    _surf_gpu = torch.stack(_surf_steps_gpu, dim=1)    # (B, T, V, H, W)
                    _upper_gpu = torch.stack(_upper_steps_gpu, dim=1)  # (B, T, V, L, H, W)
                    B, T = _surf_gpu.shape[:2]

                    # inv_transform expects a leading batch dim, so fold time into it.
                    surf_phys = self.valid_dataset.surface_inv_transform(
                        _surf_gpu.reshape(B * T, *_surf_gpu.shape[2:]))
                    upper_phys = self.valid_dataset.upper_air_inv_transform(
                        _upper_gpu.reshape(B * T, *_upper_gpu.shape[2:]))

                    diag_phys = None
                    if self.params.has_diagnostic:
                        _diag_gpu = torch.stack(_diag_steps_gpu, dim=1)
                        diag_phys = self.valid_dataset.diagnostic_inv_transform(
                            _diag_gpu.reshape(B * T, *_diag_gpu.shape[2:]))
                        # Clamp non-negative diagnostic variables (precipitation) to zero.
                        # Done on GPU over all timesteps at once; the t=0 zero slot is
                        # unaffected since inv_transform of 0 is clamped the same as before.
                        nonneg_vars = {'total_precipitation_24hr', 'total_precipitation_6hr',
                                       'mean_top_net_long_wave_radiation_flux'}
                        for vi, vname in enumerate(self.valid_dataset.diagnostic_variables):
                            if vname in nonneg_vars:
                                diag_phys[:, vi].clamp_(min=0.0)
                        # Restore the t=0 slot to literal zeros. inv_transform maps the
                        # normalized-zero placeholder to `mean`, but the previous code left
                        # index 0 untouched at 0.0 — keep the saved files identical.
                        diag_phys = diag_phys.reshape(B, T, *diag_phys.shape[1:])
                        diag_phys[:, 0].zero_()
                        diag_phys = diag_phys.reshape(B * T, *diag_phys.shape[2:])

                    # Lazily allocate pinned staging buffers so each D2H is async and
                    # cudaHostAlloc happens once, not once per ensemble member.
                    if self._surf_cpu is None or self._surf_cpu.shape != surf_phys.shape:
                        self._surf_cpu = torch.empty(surf_phys.shape, dtype=torch.float32, pin_memory=True)
                    if self._upper_cpu is None or self._upper_cpu.shape != upper_phys.shape:
                        self._upper_cpu = torch.empty(upper_phys.shape, dtype=torch.float32, pin_memory=True)
                    self._surf_cpu.copy_(surf_phys, non_blocking=True)
                    self._upper_cpu.copy_(upper_phys, non_blocking=True)
                    if diag_phys is not None:
                        if self._diag_cpu is None or self._diag_cpu.shape != diag_phys.shape:
                            self._diag_cpu = torch.empty(diag_phys.shape, dtype=torch.float32, pin_memory=True)
                        self._diag_cpu.copy_(diag_phys, non_blocking=True)

                    # Single sync covering every copy queued above.
                    torch.cuda.synchronize()

                    # .copy() detaches from the pinned buffers so they can be reused for
                    # the next member while the save thread still holds these arrays.
                    val_output_surface = self._surf_cpu.numpy().reshape(B, T, *_surf_gpu.shape[2:]).copy()
                    val_output_upper_air = self._upper_cpu.numpy().reshape(B, T, *_upper_gpu.shape[2:]).copy()
                    val_output_diagnostic = (
                        self._diag_cpu.numpy().reshape(B, T, *_diag_gpu.shape[2:]).copy()
                        if diag_phys is not None else None
                    )

                    # val_output_* are freshly allocated per ens_id, so the worker thread
                    # owns them outright — no buffer reuse race with the next member.
                    _diag_out = val_output_diagnostic
                    if self._save_executor is not None:
                        self._pending_saves.append(self._save_executor.submit(
                            self.save_prediction,
                            val_output_surface, val_output_upper_air, list(start_times),
                            _diag_out, ens_id))
                    else:
                        self.save_prediction(val_output_surface, val_output_upper_air, start_times,
                                             diagnostic_prediction=_diag_out,
                                             ens_id=ens_id)

                    # Track spread across members for the first sample of batch 0.
                    if batch_idx == 0:
                        spread_tracker.append(val_output_surface[0])  # (steps+1, vars, lat, lon)

                    # Collect predictions for diagnostic plots (ens_id 0 and 1 only)
                    if ens_id in (0, 1):
                        for si in range(val_output_surface.shape[0]):
                            key = (batch_idx, si)
                            if key in diag_selected:
                                if key not in diag_store:
                                    diag_store[key] = {'start_time': start_times[si]}
                                ens_key = f'ens{ens_id}'
                                diag_store[key][ens_key] = val_output_surface[si].copy()

                        # Generate plots as soon as both ensemble members are ready (after ens_id=1, first batch)
                        if ens_id == 1 and batch_idx == 0:
                            savedir = os.path.join(self.params['experiment_dir'], 'predictions')
                            os.makedirs(savedir, exist_ok=True)
                            ready = [v for v in diag_store.values() if 'ens0' in v and 'ens1' in v]
                            if ready:
                                self._make_diagnostic_plots(ready, savedir)

                # After all members done for batch 0: print spread per step per variable.
                if batch_idx == 0 and len(spread_tracker) > 1:
                    ens_array = np.stack(spread_tracker, axis=0)  # (n_members, steps+1, vars, lat, lon)
                    surf_vars = list(self.valid_dataset.surface_variables)
                    print("\n[Spread diagnostic] Ensemble std per lead step (sample 0, batch 0):")
                    print(f"  {'Step':>5}  " + "  ".join(f"{v[:12]:>12}" for v in surf_vars))
                    for step in range(ens_array.shape[1]):
                        std_per_var = ens_array[:, step].std(axis=0).mean(axis=(-2, -1))  # (vars,)
                        print(f"  {step:>5}  " + "  ".join(f"{s:>12.4f}" for s in std_per_var))
                    print()

        # Block until every queued save has landed on disk. .result() re-raises any
        # exception from the worker thread, which would otherwise be swallowed.
        if self._pending_saves:
            logging.info('Waiting for %d background save(s) to finish...', len(self._pending_saves))
            _save_wait_start = time.time()
            for f in self._pending_saves:
                f.result()
            self._pending_saves.clear()
            logging.info('Background saves drained in %.2fs', time.time() - _save_wait_start)
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None

        total_time = time.time() - total_start
        return total_time
    
    

    def save_prediction(self, surface_prediction, upper_air_prediction, start_times, diagnostic_prediction = None, ens_id=None):
        print("Saving predictions...")
        
        if ens_id == 0 :
            print("_____________________________________________")
            print("start times for the first ensemble member:", start_times)
            print("_____________________________________________")
        inference_results_dir = self.params['experiment_dir']
        savedir = os.path.join(inference_results_dir, 'predictions')
        
        os.makedirs(savedir, exist_ok=True)
            
        pred_config = os.path.join(self.params['experiment_dir'], os.path.basename(params['config_filepath']))
        if not os.path.exists(pred_config):
            shutil.copy(params['config_filepath'], pred_config)
            
        for sample in range(surface_prediction.shape[0]):
     
            time_range = xr.cftime_range(start_times[sample]+  timedelta(hours = self.params['timedelta_hours'] * sample) , 
                                         start_times[sample] + timedelta(hours = self.params['timedelta_hours'] * (sample + self.params['inference_steps'])),
                                         freq = "%dh" % self.params['timedelta_hours'], inclusive = "both") #
          
            coordinates = {'time': time_range,
                               'level': self.params.levels, 
                               'latitude': self.params.lat,
                               'longitude': self.params.lon}
            
            if start_times[sample].strftime('%H')=='00' :
                #and (start_times[sample].strftime('%m')=='05'or start_times[sample].strftime('%m')=='06'or start_times[sample].strftime('%m')=='07')
                filename = '%s_%s_%dh_%dstep_%s_ens_%s.nc' % (self.params.nettype, self.params.run_num, self.params['timedelta_hours'],
                                                        self.params['inference_steps'], start_times[sample].strftime('%Y%m%d%H'), ens_id)

                print(f"filenmae for start times:",start_times[sample] )
                print("_____________________________________________")
                dataset = xr.Dataset(data_vars = dict(),
                                    coords = coordinates,
                                    attrs = dict(description = f"Prediction from {self.params.nettype} model run {self.params.run_num}"))
                # print("Adding attributes to coordinates")
                dataset["level"].attrs['axis'] = 'Z'
                dataset['latitude'].attrs['axis'] = 'Y'
                dataset['longitude'].attrs['axis'] = 'X'
                dataset["level"].attrs['positive'] = 'down' # this litle line cost me half a day of work. It's for guess_coord_axis to work properly.
                dataset = dataset.cf.guess_coord_axis()
                for idx, var in enumerate(self.valid_dataset.surface_variables):
                    da = xr.DataArray(data = surface_prediction[sample, :, idx],
                                    dims=["time", "latitude", "longitude"],
                                    coords = {'time': time_range,
                                                    'latitude': dataset.latitude.values,
                                                    'longitude': dataset.longitude.values
                                                        })
                    #da = da.assign_attrs(self.valid_dataset.data_dss[0][var].attrs)
                    dataset[var] = da
                for idx, var in enumerate(self.valid_dataset.upper_air_variables):
                    da = xr.DataArray(data = upper_air_prediction[sample, :, idx],
                                    dims=["time", "level", "latitude", "longitude"],
                                    coords = coordinates)
                    #da = da.assign_attrs(self.valid_dataset.data_dss[0][var].attrs)
                    dataset[var] = da
                if self.params.has_diagnostic and diagnostic_prediction is not None:
                    for idx, var in enumerate(self.valid_dataset.diagnostic_variables):
                        da = xr.DataArray(data = diagnostic_prediction[sample, :, idx],
                                        dims=["time", "latitude", "longitude"],
                                        coords = {'time': time_range,
                                                        'latitude': dataset.latitude.values,
                                                        'longitude': dataset.longitude.values
                                                            })
                        #da = da.assign_attrs(self.valid_dataset.data_dss[0][var].attrs)
                        dataset[var] = da

                print("Added all variables to dataset") 
                dataset["latitude"] = dataset["latitude"].astype('float32').assign_attrs({'long_name': 'Latitude', 'unit': 'degrees_north'})  
                dataset["longitude"] = dataset["longitude"].astype('float32').assign_attrs({'long_name': 'Longitude', 'unit': 'degrees_east'})  
                dataset["time"] = dataset["time"].assign_attrs({'long_name': "Forecast Valid Time"}) 
                dataset["level"] = dataset["level"].astype('float32').assign_attrs({'long_name': 'Level', 'unit': 'hPa'})        
                #filename = f'{self.params.nettype}_{self.params.run_num}_{self.params['timedelta_hours']}h_{self.params['inference_steps']}step_{self.params.val_start_year}_{batch_idx * self.params.batch_size + sample}.nc'
                # Serialized: concurrent HDF5 writes segfault (h5netcdf is unavailable
                # in this env, so the netCDF4 engine is the only option).
                with self._netcdf_write_lock:
                    dataset.to_netcdf(os.path.join(savedir, filename), 'w')
                print('Done saving to directiory: ', os.path.join(savedir, filename))
            else:
                print(f"Skipping saving for start time {start_times[sample]} since it's not 00UTC")
            



            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0189', type=str)
    parser.add_argument("--yaml_config", default='config/PANGU_NEW_0189.yaml', type=str)
    parser.add_argument("--config", default='S2S', type=str)
    parser.add_argument("--enable_amp", default=True, action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--run_iter", default=1, type=int)
    # Defaults ON (matches the async-save branch), but stays switchable via
    # --no-async_save; `default=True` with store_true would pin it on permanently.
    parser.add_argument("--async_save", default=True, action=argparse.BooleanOptionalAction,
                        help="Enable asynchronous (background-thread) saving")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--finetune_ckpt", default=None, type=str,
                        help="Path to CRPS fine-tuned decoder checkpoint; overrides decoder weights after normal loading")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['epsilon_factor'] = args.epsilon_factor
    params['run_iter'] = args.run_iter
    if args.finetune_ckpt:
        params['checkpoint_path_finetune'] = args.finetune_ckpt
    if hasattr(params, 'diagnostic_variables'):
        if len(params.diagnostic_variables) > 0:
            params['has_diagnostic'] = True
        else:
            params['has_diagnostic'] = False
    else:
        params['has_diagnostic'] = False
    print(f'Has diagnostic: {params.has_diagnostic}')
    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
        print(params['world_size'])
    else:
        params['world_size'] = torch.cuda.device_count()
        print(params['world_size'])

    #params['world_size'] = 1
    '''if torch.cuda.device_count() == 1:
        world_rank = 0
        local_rank = 0
        params['batch_size'] = params['batch_size']//4'''
    
    if params['world_size'] > 1:
        #dist.init_process_group(backend='nccl', init_method='env://')
        if 'derecho' in str(Path(__file__)):
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])

        args.gpu = local_rank
        world_rank = dist.get_rank()
        print("##########WORLD RANK: TESTING ", world_rank)

        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
    else:
        world_rank = 0
        local_rank = 0

    if not hasattr(params, 'forecast_lead_times'):
        params['inference_steps'] = (24 * 15) // params.timedelta_hours
    else:
        params['inference_steps'] = max(params.forecast_lead_times)

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    expDir = os.path.join(os.getcwd(), 'results', args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    ckpt_path = 'training_checkpoints/ckpt.tar'
    best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
    params['checkpoint_path'] = os.path.join(expDir, ckpt_path)
    params['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)
    params['config_filepath'] = os.path.join(os.getcwd(), args.yaml_config)
    params['run_num'] = args.run_num

    # Do not comment this line out please:
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False

    params['resuming'] = False
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "Pangu_plasim_" + args.config  
    params['project'] = "Pangu"  
    params['entity'] = "proj-ai-weather"
    if world_rank == 0:
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(os.getcwd(), 'logs', log_file))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile)
    
    inference = Stepper(params, world_rank, args.async_save)
    inference.predict()
    logging.info('DONE ---- rank %d' % world_rank)
