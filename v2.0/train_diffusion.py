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
from networks.diffusion import ConditionalDiffusionModel
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
        self.optimizer = torch.optim.Adam(self.diff_model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        if self.params.checkpoint_path_diff and os.path.isfile(self.params.checkpoint_path_diff):
            self.restore_diff_checkpoint(self.params.checkpoint_path_diff)
            self.setup_scheduler(restart=False)
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

        _load(self.params.checkpoint_path_vae, self.diff_model.encoder)
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
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        self.optimizer.zero_grad()
                        loss = self.diff_model.training_step(surface_in = input_surface, 
                                                             constant_boundary = self.constant_boundary_data, 
                                                             varying_boundary = varying_boundary_data, 
                                                             upper_air_in = input_upper_air)   

                        
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.diff_model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        
                    if self.params.scheduler == 'OneCycleLR':
                        self.scheduler.step()

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    diagnostic_logs = {"loss": loss, "lr": current_lr}
                        
                    if self.world_rank == 0 and self.wandb_enabled:
                        #wandb.log(diagnostic_logs, step=(self.epoch-1) * total_iterations + self.iters)
                        wandb.log(diagnostic_logs, step= self.iters)
                    if i % 2000 == 0:
                        temp_path = os.path.split(self.params.checkpoint_path_diff)[0]
                        diff_path = os.path.join(temp_path, f"diff_ckpt_{self.iters}.tar")
                        logging.info(f"Year {self.params.train_year_start + year_idx}, Loss: {diagnostic_logs['loss']:.4f}")
                        self.save_checkpoint(diff_path, self.diff_model)
        # pbar.close()
        # pbar.updac te(1)
        logs ={"train_loss": loss, "epoch": self.epoch}
        return logs
        
    def train_diff(self, epochs = 50):
        for epoch in range(epochs):
            logs = self.training_one_epoch_diffusion()
            if self.wandb_enabled:
                wandb.log(logs, step=self.epoch)
            # if epoch % self.params.validation_interval == 0:
            #     self.validation_diffusion()


    # def validation_diffusion(self):
    #     self.diff_model_.eval()
    #     #n_valid_batches = 50  # do validation on first 50 images, just for LR scheduler
    #     # define the lead times to evaluate (in time steps)
    #     lead_times_steps = self.params.forecast_lead_times
    #     with torch.no_grad():
    #             latitudes = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)

    #     # Initialize validation loss variables
    #     valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, valid_steps, \
    #     valid_surface_lwrmse, valid_upper_air_lwrmse, valid_diagnostic_lwrmse, \
    #     multi_step_losses, multi_step_rmse = self.inti_valid_loss(lead_times_steps)
        
    #     valid_start = time.time()
    #     nb = len(self.valid_data_loader)

    #     diagnostic_logs = {}

    #     sample_idx = np.random.randint(len(self.valid_data_loader))

    #     all_predictions = []
    #     all_ground_truths = []
    #     acc_predictions = []
    #     acc_ground_truths = []

    #     # with torch.inference_mode():
    #     with torch.no_grad():
    #         for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
         
    #     return None
        
        
    def get_diffusion_model(self):
        self.diff_model =  ConditionalDiffusionModel(
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
    parser.add_argument("--local_storage", default=False, type=str)
    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.local_storage:
        params["data_dir"] = args.local_storage
        print("using the local storage:",params["data_dir"])
        
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


    
    



