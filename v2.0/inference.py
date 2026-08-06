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
        self._first_batch_loaded = False
        self._first_forward_done = False
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
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.data_dir, dist.is_initialized(), 
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

        self._save_executor = ThreadPoolExecutor(max_workers=2) if self.async_save else None
        self._pending_saves = []
        # Pinned CPU buffers for async D2H transfers; lazily allocated on first use
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
        """One figure per variable (t2m, precip): rows=GT/Ens1/Ens2, cols=lead days."""
        timedelta_h = int(self.params['timedelta_hours'])
        lead_days = [1, 5, 10, 15, 30]

        surf_vars = list(self.valid_dataset.surface_variables)
        diag_vars = (list(self.valid_dataset.diagnostic_variables)
                     if self.params.has_diagnostic else [])

        # Identify index and source array key for each plot variable
        plot_vars = []
        if '2m_temperature' in surf_vars:
            plot_vars.append(dict(name='2m_temperature', unit='K', cmap='RdBu_r',
                                  src='surf', idx=surf_vars.index('2m_temperature')))
        else:
            logging.warning('2m_temperature not in surface_variables, skipping in diagnostic plots')

        precip_name = next((v for v in diag_vars if 'precip' in v.lower()), None)
        if precip_name is not None:
            plot_vars.append(dict(name=precip_name, unit='mm', cmap='Blues',
                                  src='diag', idx=diag_vars.index(precip_name)))
        else:
            logging.warning('No precipitation variable found in diagnostic_variables')

        row_labels = ['Ground Truth', 'Ensemble 1', 'Ensemble 2']

        for pv in plot_vars:
            var_name = pv['name']
            var_idx  = pv['idx']
            src      = pv['src']

            # valid lead days given inference_steps
            valid_leads = [d for d in lead_days
                           if (d * 24) // timedelta_h <= self.params['inference_steps']]
            if not valid_leads:
                continue

            n_cols = len(valid_leads)
            fig, axes = plt.subplots(3, n_cols,
                                     figsize=(4 * n_cols, 9),
                                     gridspec_kw={'hspace': 0.35, 'wspace': 0.05})
            # Ensure axes is always 2-D
            if n_cols == 1:
                axes = axes.reshape(3, 1)

            # Collect vmin/vmax across all panels for a shared colorscale
            all_vals = []
            for lead_day in valid_leads:
                step = (lead_day * 24) // timedelta_h
                for sample in diag_samples:
                    for ens_key in ('ens0', 'ens1'):
                        arr_src = sample.get(ens_key if src == 'surf' else f'diag_{ens_key}')
                        if arr_src is not None:
                            all_vals.append(arr_src[step, var_idx].ravel())
            vmin = np.nanpercentile(np.concatenate(all_vals), 2)  if all_vals else 0
            vmax = np.nanpercentile(np.concatenate(all_vals), 98) if all_vals else 1

            im = None
            for col, lead_day in enumerate(valid_leads):
                step = (lead_day * 24) // timedelta_h

                # Use first available sample for this lead
                sample = diag_samples[0]
                start_time = sample['start_time']
                valid_dt   = start_time + timedelta(hours=step * timedelta_h)

                # Ground-truth row
                if src == 'surf' and var_name == '2m_temperature':
                    era5_ds = self._get_era5_t2m(valid_dt.year)
                    if era5_ds is not None:
                        ts = pd.Timestamp(valid_dt.year, valid_dt.month,
                                          valid_dt.day, valid_dt.hour)
                        lat_name = 'lat' if 'lat' in era5_ds.dims else 'latitude'
                        gt_arr = era5_ds['2m_temperature'].sel(time=ts, method='nearest').values
                        if era5_ds[lat_name].values[0] > era5_ds[lat_name].values[-1]:
                            gt_arr = gt_arr[::-1]
                    else:
                        ens0_src = sample.get('ens0' if src == 'surf' else 'diag_ens0')
                        gt_arr = np.full(ens0_src[step, var_idx].shape, np.nan)
                else:
                    ens0_src = sample.get('ens0' if src == 'surf' else 'diag_ens0')
                    gt_arr = np.full(ens0_src[step, var_idx].shape, np.nan)

                ens0_src = sample.get('ens0' if src == 'surf' else 'diag_ens0')
                ens1_src = sample.get('ens1' if src == 'surf' else 'diag_ens1')
                pred_ens0 = ens0_src[step, var_idx] if ens0_src is not None else np.full_like(gt_arr, np.nan)
                pred_ens1 = ens1_src[step, var_idx] if ens1_src is not None else np.full_like(gt_arr, np.nan)

                for row, arr in enumerate([gt_arr, pred_ens0, pred_ens1]):
                    ax = axes[row, col]
                    im = ax.imshow(arr, cmap=pv['cmap'], vmin=vmin, vmax=vmax,
                                   aspect='auto', origin='lower')
                    ax.axis('off')
                    if row == 0:
                        ax.set_title(f'Lead {lead_day}d', fontsize=9, pad=3)
                    if col == 0:
                        ax.set_ylabel(row_labels[row], fontsize=9)
                        ax.yaxis.set_visible(True)
                        ax.tick_params(left=False, labelleft=False)
                        ax.spines['left'].set_visible(False)
                        ax.text(-0.12, 0.5, row_labels[row], transform=ax.transAxes,
                                fontsize=10, va='center', ha='right', rotation=90)

            # Shared colorbar below the grid
            if im is not None:
                fig.subplots_adjust(bottom=0.12)
                cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                cbar.set_label(f'{var_name} ({pv["unit"]})', fontsize=10)

            ic_str = diag_samples[0]['start_time'].strftime('%Y-%m-%d')
            fig.suptitle(f'{var_name}  —  IC {ic_str}', fontsize=12, y=0.98)
            fname = os.path.join(savedir, f'diag_{var_name}.png')
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
                # Unpack once per batch; move to GPU with non_blocking for overlap
                val_input_surface_batch, val_input_upper_air_batch, _, _, _, val_varying_boundary_data_batch, times = map(
                        lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                self._first_batch_loaded = True

                times_np = times.cpu().numpy().astype(int)
                start_times = [
                    self.valid_dataset.datetime_class(
                        times_np[idx, 0], times_np[idx, 1], times_np[idx, 2], hour=times_np[idx, 3])
                    for idx in range(times_np.shape[0])
                ]

                for ens_id in range(50):
                    val_input_surface = val_input_surface_batch.clone()
                    val_input_upper_air = val_input_upper_air_batch.clone()
                    val_varying_boundary_data = val_varying_boundary_data_batch

                    ic_noise_std = getattr(self.params, 'ic_noise_std', 0.2)

                    # t=0 IC perturbation, scaled per-variable (member 0 = control)
                    if ens_id > 0 and ic_noise_std > 0.0:
                        surf_scale = val_input_surface.std(dim=(-2, -1), keepdim=True)
                        ua_scale   = val_input_upper_air.std(dim=(-2, -1), keepdim=True)
                        val_input_surface   = val_input_surface   + ic_noise_std * surf_scale * torch.randn_like(val_input_surface)
                        val_input_upper_air = val_input_upper_air + ic_noise_std * ua_scale   * torch.randn_like(val_input_upper_air)

                    # Accumulate GPU tensors across rollout to avoid per-step D2H transfers
                    _val_output_surface_gpu = [val_input_surface.detach()]
                    _val_output_upper_air_gpu = [val_input_upper_air.detach()]
                    _val_output_diagnostic_gpu = []
                    if self.params.has_diagnostic:
                        val_output_diagnostic_init = torch.zeros(
                            (val_input_surface.shape[0], self.num_diagnostic_vars,
                             val_input_surface.shape[2], val_input_surface.shape[3]),
                            dtype=torch.float32, device=self.device)
                        _val_output_diagnostic_gpu.append(val_output_diagnostic_init)

                    for time_step in range(self.params['inference_steps']):
                        val_out_surface, val_out_upper_air, val_out_diagnostic = self.diff_model.prediction(
                            surface_in=val_input_surface,
                            constant_boundary=self.constant_boundary_data,
                            varying_boundary=val_varying_boundary_data[:, time_step],
                            upper_air_in=val_input_upper_air,
                            device=self.device,
                            mc_dropout=True, temperature=1.5)
                        _val_output_surface_gpu.append(val_out_surface.detach())
                        _val_output_upper_air_gpu.append(val_out_upper_air.detach())
                        if self.params.has_diagnostic:
                            _val_output_diagnostic_gpu.append(val_out_diagnostic.detach())
                        val_input_surface, val_input_upper_air = val_out_surface, val_out_upper_air
                        if not self._first_forward_done:
                            self._first_forward_done = True

                    # Stack all timesteps on GPU: (B, T, ...)
                    _surf_gpu = torch.stack(_val_output_surface_gpu, dim=1)
                    _upper_gpu = torch.stack(_val_output_upper_air_gpu, dim=1)
                    B, T = _surf_gpu.shape[:2]

                    # Apply inverse transforms on GPU before D2H
                    surf_t = self.valid_dataset.surface_inv_transform(
                        _surf_gpu.view(B * T, *_surf_gpu.shape[2:]))
                    upper_t = self.valid_dataset.upper_air_inv_transform(
                        _upper_gpu.view(B * T, *_upper_gpu.shape[2:]))

                    # Lazy-allocate pinned CPU buffers to avoid repeated cudaHostAlloc
                    if self._surf_cpu is None or self._surf_cpu.shape != surf_t.shape:
                        self._surf_cpu = torch.empty(surf_t.shape, dtype=torch.float32, pin_memory=True)
                    if self._upper_cpu is None or self._upper_cpu.shape != upper_t.shape:
                        self._upper_cpu = torch.empty(upper_t.shape, dtype=torch.float32, pin_memory=True)

                    # Async D2H into pinned buffers
                    self._surf_cpu.copy_(surf_t, non_blocking=True)
                    self._upper_cpu.copy_(upper_t, non_blocking=True)

                    if self.params.has_diagnostic:
                        _diag_gpu = torch.stack(_val_output_diagnostic_gpu, dim=1)
                        diag_t = self.valid_dataset.diagnostic_inv_transform(
                            _diag_gpu.view(B * T, *_diag_gpu.shape[2:]))
                        if self._diag_cpu is None or self._diag_cpu.shape != diag_t.shape:
                            self._diag_cpu = torch.empty(diag_t.shape, dtype=torch.float32, pin_memory=True)
                        self._diag_cpu.copy_(diag_t, non_blocking=True)

                    # Single sync point for all D2H transfers
                    torch.cuda.synchronize()

                    # .copy() makes independent numpy arrays so pinned buffers can be reused
                    val_output_surface = self._surf_cpu.numpy().reshape(B, T, *_surf_gpu.shape[2:]).copy()
                    val_output_upper_air = self._upper_cpu.numpy().reshape(B, T, *_upper_gpu.shape[2:]).copy()
                    val_output_diagnostic = (
                        self._diag_cpu.numpy().reshape(B, T, *_diag_gpu.shape[2:]).copy()
                        if self.params.has_diagnostic else None
                    )

                    # Submit save asynchronously so GPU can start next member immediately
                    if self._save_executor is not None:
                        f = self._save_executor.submit(
                            self.save_prediction,
                            val_output_surface, val_output_upper_air,
                            list(start_times), val_output_diagnostic, ens_id)
                        self._pending_saves.append(f)
                    else:
                        self.save_prediction(val_output_surface, val_output_upper_air,
                                             start_times, val_output_diagnostic, ens_id=ens_id)

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
                                if val_output_diagnostic is not None:
                                    diag_store[key][f'diag_{ens_key}'] = val_output_diagnostic[si].copy()

                        # Generate plots after both ensemble members collected for first batch
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

        # Wait for all background saves to finish
        for f in self._pending_saves:
            f.result()
        self._pending_saves.clear()
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
                dataset.to_netcdf(os.path.join(savedir, filename), mode='w', engine='h5netcdf')
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
    parser.add_argument("--async_save", default=True, action="store_true", help="Enable asynchronous saving")
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
