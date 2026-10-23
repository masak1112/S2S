#Train diffusion model##

"""
Training script for diffusion model.
Author: Bing Gong
Date: February 18, 2026
"""

from pathlib import Path
import logging
import time
import torch
from networks.diffusion import ConditionalDiffusionModel, is_cond_branch_param
from networks.stochastic_interpolant import StochasticInterpolant

from train import Trainer
import tqdm
from collections import OrderedDict
import wandb 
import numpy as np
from ruamel.yaml import YAML
import os
from utils import logging_utils
from ruamel.yaml.comments import CommentedMap as ruamelDict
import argparse
from utils.YParams import YParams
import torch.distributed as dist
from torch.amp import autocast, GradScaler


if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

world_rank = dist.get_rank()
print(f"World rank: {world_rank}")

class DiffusionTrainer(Trainer):
    def __init__(self, params,world_rank):
        super().__init__(params,world_rank)
       
        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()
        #load model weights 
        #self.restore_checkpoint(self.params.checkpoint_path_vae,self.params.checkpoint_path_det, optimizer=False)
                
        # freeze all params in self.model
        for p in self.model_vae.parameters():
            p.requires_grad = False

        self.diff_model= self.get_diffusion_model()
        self.temp_path = os.path.split(self.params.checkpoint_path_diff)[0]
        self.optimizer = self._build_optimizer()
        # Average UNet gradients across ranks. The SI model is not wrapped in
        # DDP, so without this every rank trains its own copy on its own data
        # shard and only rank 0's copy is ever saved.
        self.sync_grads = bool(getattr(self.params, 'sync_grads', False)) and dist.is_initialized() \
            and dist.get_world_size() > 1
        self.cond_diag_freq = int(getattr(self.params, 'cond_diag_freq', 0))
        init_ckpt = getattr(self.params, 'checkpoint_path_diff_init', None)
        if self.params.checkpoint_path_diff and os.path.isfile(self.params.checkpoint_path_diff):
            self.restore_diff_checkpoint(self.params.checkpoint_path_diff)
            self.setup_scheduler(restart=False)
        elif init_ckpt and os.path.isfile(init_ckpt):
            self._warm_start_from(init_ckpt)
            self.setup_scheduler(restart=True)
        else:
            self.setup_scheduler(restart=True)
        # Explicitly ensure diff_model.encoder uses checkpoint_path_vae weights,
        # overriding anything that restore_diff_checkpoint may have loaded.
        self._reload_vae_checkpoint()
        
        self.get_dataset()
        self.scaler = GradScaler()
        self.params = params
        
        self.wandb_enabled = bool(getattr(self.params, "log_to_wandb", False) and wandb.run is not None)
        if getattr(self.params, "log_to_wandb", False) and not self.wandb_enabled and self.world_rank == 0:
            logging.warning("W&B logging is enabled in config, but wandb.init() is not active. Skipping wandb.log calls.")
        
 
    def _build_optimizer(self):
        """Adam over the SI model. If cond_branch_weight_decay or
        cond_branch_lr_mult is set, the conditioning branch (cross-attention
        fuses, gates, cond projections) gets its own param group so it can be
        exempted from weight decay and/or given a larger lr. Otherwise a single
        group, identical to the original runs (keeps their optimizer state
        loadable)."""
        wd_cond = getattr(self.params, 'cond_branch_weight_decay', None)
        lr_mult = getattr(self.params, 'cond_branch_lr_mult', None)
        if wd_cond is None and lr_mult is None:
            return torch.optim.Adam(self.diff_model.parameters(), lr=self.params.lr,
                                    weight_decay=self.params.weight_decay)
        wd_cond = self.params.weight_decay if wd_cond is None else float(wd_cond)
        lr_mult = 1.0 if lr_mult is None else float(lr_mult)
        cond, rest = [], []
        for name, p in self.diff_model.named_parameters():
            if not p.requires_grad:
                continue
            (cond if is_cond_branch_param(name) else rest).append(p)
        if self.world_rank == 0:
            print(f"Optimizer: {sum(p.numel() for p in rest):,} params @ lr={self.params.lr}, "
                  f"wd={self.params.weight_decay}; conditioning branch "
                  f"{sum(p.numel() for p in cond):,} params @ lr={self.params.lr * lr_mult}, wd={wd_cond}")
        return torch.optim.Adam([
            {'params': rest, 'lr': self.params.lr, 'weight_decay': self.params.weight_decay},
            {'params': cond, 'lr': self.params.lr * lr_mult, 'weight_decay': wd_cond},
        ])

    def _warm_start_from(self, path):
        """Initialise the UNet from another run's checkpoint without resuming it.

        Loads only UNet weights (encoder/model_det come from their own
        checkpoints), skips the conditioning branch when reinit_cond_branch is
        set (its weights in the si_c1 runs are denormals — worthless), and does
        not touch the optimizer, iteration counter or epoch."""
        checkpoint = torch.load(path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
        raw_state = checkpoint['model_state']
        if any(k.startswith('module.') for k in raw_state):
            raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
        reinit = bool(getattr(self.params, 'reinit_cond_branch', True))
        state, skipped = OrderedDict(), []
        for k, v in raw_state.items():
            if k.startswith('encoder.') or k.startswith('model_det.'):
                continue
            if reinit and is_cond_branch_param(k):
                skipped.append(k)
                continue
            state[k] = v
        missing, unexpected = self.diff_model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not (k.startswith('encoder.') or k.startswith('model_det.'))
                       and not (reinit and is_cond_branch_param(k))]
        if bad_missing or unexpected:
            raise RuntimeError(f"warm start: unexpected keys {unexpected}, missing {bad_missing}")
        if self.world_rank == 0:
            print(f"Warm start from {path} (iters={checkpoint.get('iters')}): loaded {len(state)} UNet tensors, "
                  f"re-initialised {len(skipped)} conditioning-branch tensors "
                  f"(gate_init={getattr(self.params, 'cross_attn_gate_init', 0.0)}, "
                  f"vae_cond_mode={getattr(self.params, 'vae_cond_mode', 'sample')})")
        self.iters, self.startEpoch, self.epoch = 0, 0, 0

    def _log_cond_diagnostics(self, input_surface, varying_boundary_data, input_upper_air, loss=None,
                              sst_cond=None):
        # Runs in float32: under fp16 autocast the UNet output can overflow to
        # inf on some batches (the same overflow the GradScaler skips), which
        # turned the relative effect into NaN.
        with torch.autocast(device_type='cuda', enabled=False):
            d = self.diff_model.cond_sensitivity(surface_in=input_surface.float(),
                                                 constant_boundary=self.constant_boundary_data.float(),
                                                 varying_boundary=varying_boundary_data.float(),
                                                 upper_air_in=input_upper_air.float(),
                                                 sst_cond=sst_cond)
        if loss is not None:
            d['train_loss'] = float(loss)
        logging.info("cond diag iter %d: loss=%.4g  rel_effect=%.4f  gate_in=%.4g gate_mid=%.4g  "
                     "to_out_in=%.3g to_out_mid=%.3g  |z|=%.3g" % (
                         self.iters, d.get('train_loss', float('nan')), d['cond_rel_effect'],
                         d['cross_attn_in_gate'], d['cross_attn_mid_gate'],
                         d['cross_attn_in_to_out_absmax'], d['cross_attn_mid_to_out_absmax'], d['cond_z_absmean']))
        if self.wandb_enabled:
            wandb.log({f"cond/{k}": v for k, v in d.items()}, step=self.iters)

    def _reload_vae_checkpoint(self):
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

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        print("Loaded VAE weights from checkpoint_path_vae into diff_model.encoder")
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)
        print("Loaded det weights from checkpoint_path_det into diff_model.model_det")
        self.diff_model.freeze_encoder()
        self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()


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
        self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        print('START EPOCH:', self.startEpoch)
        # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Restored diffusion checkpoint from epoch %d, iters %d" % (self.epoch, self.iters))
        
    def training_one_epoch_diffusion(self) -> dict[str, torch.Tensor]:
        """
        Single training step.
        Returns dict with 'loss', 'diffusion_loss', 'kl_loss'.
        """
        
        self.epoch += 1
        total_iterations = sum(len(loader) for loader in self.train_data_loaders)
        diagnostic_logs = {}
        loss = 0

        logging.info(f"Expected total batches: {total_iterations}")
        if not self.train_data_loaders:
            logging.warning("No training data loaders available.")
            return 0, 0, {"train_loss": 0.0}

        # self.model.eval()

        # pbar = tqdm(total=total_iterations, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

        for year_idx, train_data_loader in enumerate(self.train_data_loaders):
            logging.debug(f"Processing year idx {year_idx}")
            
            current_dataset = self.train_datasets[year_idx]
            if self.params.train_year_to_year:
                logging.debug(f"Processing year {self.params.train_year_start + year_idx}")
            else:
                logging.debug(f"Processing years {self.params.train_year_start} to {self.params.train_year_end}")
      
            data_iter = iter(train_data_loader)
            data = next(data_iter)
            for i, data in enumerate(train_data_loader):
                if i % 100 == 0:
                    logging.info("training on batch %d of year %d" % (i, self.params.train_year_start + year_idx))
                    
                if self.params.mode == "test" and i >= self.params.test_iterations:
                    logging.info("Test mode: only processing first batches")
                    # pbar.update(total_iterations - self.iters)
                    break  
                else:
                
                    self.iters += 1
                    input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = self._prepare_inputs_batch(data)
                    self.optimizer.zero_grad()

                    do_scatter = (self.iters % 1000 == 0) and (self.world_rank == 0)
                    _scatter_dir = os.path.dirname(self.params.checkpoint_path_diff)
                    scatter_path = os.path.join(_scatter_dir, "scatter_tmp.png")

                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        loss = self.diff_model.training_step(surface_in = input_surface,
                                                             constant_boundary = self.constant_boundary_data,
                                                             varying_boundary = varying_boundary_data,
                                                             upper_air_in = input_upper_air,
                                                             target_surface_in = target_surface,
                                                             target_upper_air = target_upper_air,
                                                             plot_freq = 200, iter = self.iters,
                                                             plot_scatter = do_scatter, scatter_path = scatter_path,
                                                             sst_cond = self.sst_cond_batch)

                    self.scaler.scale(loss).backward()
                    if self.sync_grads:
                        # Average the still-SCALED grads BEFORE unscale_: the
                        # scaler records found_inf during unscale_, so reducing
                        # afterwards lets a rank with finite local grads apply
                        # a step with inf pulled in from another rank (NaN
                        # weights). Reducing first, every rank sees the same
                        # grads, the same found_inf and the same scale history.
                        ws = dist.get_world_size()
                        for p in self.diff_model.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                                p.grad.div_(ws)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.diff_model.parameters(), max_norm=1.0)
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    if scale_after < scale_before:
                        print("overflow happened")

                    if self.params.scheduler == 'OneCycleLR':
                        self.scheduler.step()

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    diagnostic_logs = {"loss": loss, "lr": current_lr}


                    if self.world_rank == 0 and self.cond_diag_freq > 0 and self.iters % self.cond_diag_freq == 0:
                        self._log_cond_diagnostics(input_surface, varying_boundary_data, input_upper_air,
                                                   loss=loss.detach(), sst_cond=self.sst_cond_batch)

                    if self.world_rank == 0 and self.wandb_enabled:
                        wandb.log(diagnostic_logs, step= self.iters)
                        if do_scatter and self.wandb_enabled and os.path.isfile(scatter_path):
                            wandb.log({"scatter_pred_vs_gt": wandb.Image(scatter_path)}, step=self.iters)
                    if i >= 2000 and i % 2000 == 0 and self.world_rank == 0:
                        temp_path = os.path.split(self.params.checkpoint_path_diff)[0]
                        diff_path = os.path.join(temp_path, f"diff_ckpt_{self.iters}.tar")
                        last_path = os.path.join(temp_path, "last_ckpt.tar")
                        logging.info(f"Year {self.params.train_year_start + year_idx}, Loss: {diagnostic_logs['loss']:.4f}")
                        self.save_checkpoint(diff_path, self.diff_model)
                        self.save_checkpoint(last_path, self.diff_model)
                        logging.info(f"Updated last checkpoint: {last_path}")
                        if self.wandb_enabled:
                            wandb.save(last_path, base_path=temp_path)
                        
        # pbar.close()
        # pbar.updac te(1)
        logs ={"train_loss": loss, "epoch": self.epoch}
        return logs
        
    def train_diff(self, epochs = 50):
        for epoch in range(epochs):
            logs = self.training_one_epoch_diffusion()
            if self.wandb_enabled:
                wandb.log(logs, step = self.epoch)  

            if self.world_rank == 0:
                temp_path = os.path.split(self.params.checkpoint_path_diff)[0]
                last_path = os.path.join(temp_path, "diff_ckpt.tar")
                # Also write the resume file so an interruption before the
                # first in-loop save (every 2000 iters) can still resume.
                self.save_checkpoint(self.params.checkpoint_path_diff, self.diff_model)
                self.save_checkpoint(last_path, self.diff_model)
                logging.info("Saved latest checkpoint: %s", self.params.checkpoint_path_diff)
                logging.info("Saved last checkpoint alias: %s", last_path)
                if self.wandb_enabled:
                    wandb.save(last_path, base_path=temp_path)
            
            # validation_interval = int(getattr(self.params, "validation_interval", 1))
            # if validation_interval > 0 and (self.epoch % validation_interval == 0):
            #     self.validation_diffusion()


    def validation_diffusion(self):
        self.diff_model.eval()
        valid_start = time.time()

        if not hasattr(self, "valid_data_loader") or self.valid_data_loader is None:
            logging.warning("No valid_data_loader available. Skipping diffusion validation.")
            self.diff_model.train()
            return {"valid_loss": float("nan")}

        max_valid_batches = int(getattr(self.params, "valid_max_batches", 30))
        total_valid_batches = len(self.valid_data_loader)
        if max_valid_batches > 0:
            total_valid_batches = min(total_valid_batches, max_valid_batches)

        loss_sum = torch.zeros(1, dtype=torch.float32, device=self.device)
        step_count = torch.zeros(1, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for i, data in tqdm.tqdm(
                enumerate(self.valid_data_loader, 0),
                total=total_valid_batches,
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'
            ):
                if max_valid_batches > 0 and i >= max_valid_batches:
                    break

                input_surface, input_upper_air, target_surface, target_upper_air, _,  varying_boundary_data = self._prepare_inputs_batch(data)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    val_loss = self.diff_model.training_step(
                        surface_in=input_surface,
                        constant_boundary=self.constant_boundary_data,
                        varying_boundary=varying_boundary_data,
                        upper_air_in=input_upper_air,
                        target_surface_in=target_surface,
                        target_upper_air=target_upper_air,
                        train=False,
                        plot_freq=0,
                        iter=self.iters,
                        sst_cond=self.sst_cond_batch,
                    )

                loss_sum += val_loss.detach().float()
                step_count += 1.0

        if dist.is_initialized():
            dist.all_reduce(loss_sum)
            dist.all_reduce(step_count)

        valid_loss = (loss_sum / torch.clamp_min(step_count, 1.0)).item()
        valid_time = time.time() - valid_start
        logs = {
            "epoch": self.epoch,
            "valid_loss": valid_loss,
            "valid_time_sec": valid_time,
        }

        if self.world_rank == 0:
            logging.info(
                "Validation diffusion | epoch=%d valid_loss=%.6f steps=%d time=%.2fs",
                self.epoch,
                valid_loss,
                int(step_count.item()),
                valid_time,
            )
            if self.wandb_enabled:
                wandb.log(logs, step=self.epoch)

        self.diff_model.train()
        return logs
        
        
    def get_diffusion_model(self):
        if self.params.diffusion_model_type == "diffusion":
            self.diff_model =  ConditionalDiffusionModel(
                                T=1000,
                                VAEEncoder=self.model_vae.module, 
                                DETEncoder = self.model_det.module,
                                params = self.params# Use default simple encoder
                            ).to(device)
        elif self.params.diffusion_model_type == "SI":
            self.diff_model = StochasticInterpolant(
                                T=1000,
                                VAEEncoder=self.model_vae.module, 
                                DETEncoder = self.model_det.module,
                                params = self.params# Use default simple encoder
                            ).to(device)
                # Count parameters
        n_params = sum(p.numel() for p in self.diff_model.parameters())
        print(f"Total parameters: {n_params:,}")
        return self.diff_model


            
            
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0100', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_S2S.yaml', type=str)
    parser.add_argument("--config", default='S2S', type=str) 
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--run_iter", default=1, type=int)
    # parser.add_argument("--num_inferences", type = int)
    # parser.add_argument("--window_size", default = '2,2,2', type = str)
    parser.add_argument("--fresh_start", default=False, action="store_true", help="Start training from scratch, ignoring existing checkpoints")
    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--year", default=1979, type=int, help="if enbale_nvme is true, then use the year data")
    parser.add_argument("--enable_nvme", action="store_true", default=False, help="Save predictions to NVMe/local scratch if available.")
    parser.add_argument("--nvme_dir", type=str, default="", help="Directory on NVMe/local scratch to save predictions if --enable_nvme is set.")
    #######
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

        
    print("This is the starting point f")
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['epsilon_factor'] = args.epsilon_factor
    params['run_iter'] = args.run_iter

    if hasattr(params, 'diagnostic_variables'):
        if len(params.diagnostic_variables) > 0:
            params['has_diagnostic'] = True
        else:
            params['has_diagnostic'] = False
    else:
        params['has_diagnostic'] = False

    print(f'Has diagnostic: {params.has_diagnostic}')
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1

    if hasattr(params, "wandb_offline"):
        if params.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())


    ##Check GPU memory 
    print(torch.cuda.get_device_name(0))
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")


    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
        print(params['world_size'])
    else:
        params['world_size'] = torch.cuda.device_count()
        print(params['world_size'])

     
    if params['world_size'] > 1:
        
        if 'derecho' in str(Path(__file__)):
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])

        args.gpu = local_rank
        
        # print("##########WORLD RANK: TESTING ", world_rank)
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
    else:
        world_rank = 0
        local_rank = 0

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    if args.enable_nvme:
        params["data_dir"] = args.nvme_dir
        params["train_year_start"] = args.year
        params["train_year_end"] = args.year + 1
        
    
    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    ckpt_path = 'training_checkpoints/ckpt.tar'
    best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
    params['checkpoint_path'] = os.path.join(expDir, ckpt_path)
    params['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)

    checkpoint_exists = os.path.isfile(params.checkpoint_path)

    # Determine whether to resume or start fresh
    if params.fresh_start or args.fresh_start:
        params['resuming'] = False
        if checkpoint_exists and world_rank == 0:
            logging.info("Fresh start requested. Ignoring existing checkpoint.")

    elif checkpoint_exists:
        params['resuming'] = True
        if world_rank == 0:
            logging.info("Resuming from existing checkpoint.")
    else:
        params['resuming'] = False
        if world_rank == 0:
            logging.info("No checkpoint found. Starting fresh training run.")

    # # Do not comment this line out please:
    # # args.resuming = True if os.path.isfile(params.checkpoint_path) else False
    # args.resuming = False
    # params['resuming'] = args.resuming

    params['local_rank'] = local_rank

    # Add indicator for precision method and engine
    if params['use_transformer_engine']:
        print("Using Transformer Engine")
    else:
        print("Using PyTorch native")

    if world_rank == 0:
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, log_file))
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

    trainer = DiffusionTrainer(params, world_rank)
    # trainer.setup_model()
    trainer.train_diff()
    logging.info('DONE ---- rank %d' % world_rank)


    
    



