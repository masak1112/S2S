from networks.pangu import PanguModel_Plasim
from networks.pangu_vae import PanguModel_Plasim_VAE
from networks.diffusion import ConditionalDiffusionModel
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
                self.has_land = True
        else:
            self.params['ocean_variables'] = []
        if hasattr(self.params, 'mask_output'):
            self.mask_output = params.mask_output
        
        self.num_diagnostic_vars = len(self.params.diagnostic_variables) if self.params.has_diagnostic else 0

        # if params.log_to_wandb:
        #     wandb.init(config=params, name=params.name, group=params.group, project=params.project,
        #                entity=params.entity, settings=wandb.Settings(_disable_stats=True) )

        logging.info('rank %d, begin data loader init' % world_rank)
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                     year_start=params.val_year_start, 
                                                                     year_end=params.val_year_end, train=False,
                                                                     num_inferences = params.num_inferences, validate = True)
        print(f'Valid dataset length: {len(self.valid_dataset)}')
        # self.infer_data_loader, self.infer_dataset = get_infer_data(params, params.data_dir, dist.is_initialized(),
        #                                                              year_start=params.val_year_start,
        #                                                              year_end=params.val_year_end, step=1462,
        #                                                              num_inferences = params.num_inferences, validate = True)

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
            
            self.model_vae = PanguModel_Plasim_VAE(self.params, land_mask = land_mask, mask_fill = self.params.mask_fill).to(self.device)
            self.model_det = PanguModel_Plasim(self.params, land_mask = land_mask, 
                                               mask_fill = self.params.mask_fill).to(self.device)

            # self.model = torch.compile(self.model, mode = 'default')
        else:
            raise Exception("not implemented")

 
        self.diff_model = ConditionalDiffusionModel(
                            T=1000,
                            VAEEncoder=self.model_vae, 
                            DETEncoder = self.model_det,
                            params = self.params# Use default simple encoder
                         ).to(self. device)
        self.model_diff = self.diff_model
  
        self.restore_diff_checkpoint(params.checkpoint_path_diff)
        self._reload_vae_checkpoint(
            checkpoint_path_vae=params.checkpoint_path_vae,
            checkpoint_path_det=params.checkpoint_path_det,
        )

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

  
    def _reload_vae_checkpoint(self, checkpoint_path_vae=None, checkpoint_path_det=None):
        """Explicitly load VAE and deterministic checkpoint weights into
        diff_model.encoder and diff_model.model_det respectively.
        Guarantees both always use checkpoint_path_vae / checkpoint_path_det,
        regardless of what any diff checkpoint may contain."""
        checkpoint_path_vae = checkpoint_path_vae or self.params.checkpoint_path_vae
        checkpoint_path_det = checkpoint_path_det or self.params.checkpoint_path_det

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
        
    def predict(self):
        if self.params.log_to_screen:
            logging.info("Starting Model Inference Loop...")
        valid_time, valid_logs = self.validate_one_epoch()
        
        


    def validate_one_epoch(self):
        self.model_diff.eval()
        total_start = time.time()

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for i, data in enumerate(self.valid_data_loader, 0):
                for ens_id in list(range(3)):
     
                    if self.params.has_diagnostic:
                        val_input_surface, val_input_upper_air, _, _, _, val_varying_boundary_data, times = map(
                            lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    else:
                        val_input_surface, val_input_upper_air, _, _, _, times = map(
                            lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    
                    start_times = []
                    for i in range(times.shape[0]):  # Iterate over all samples in the batch
                        start_time = self.valid_dataset.datetime_class(times[i,0].item(), times[i,1].item(), times[i,2].item(), hour=times[i,3].item())
                        start_times.append(start_time)

                
                    val_output_surface = np.zeros((val_input_surface.shape[0], self.params['inference_steps']+1,
                                                    val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]),
                                                    dtype = np.float32)
                    val_output_upper_air = np.zeros((val_input_upper_air.shape[0], self.params['inference_steps']+1,
                                                    val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                                        val_input_upper_air.shape[3], val_input_upper_air.shape[4]),
                                                    dtype = np.float32)
                    if self.params.has_diagnostic:
                        val_output_diagnostic = np.zeros((val_input_surface.shape[0], self.params['inference_steps']+1,
                                                        self.num_diagnostic_vars, val_input_surface.shape[2], val_input_surface.shape[3]),
                                                        dtype = np.float32)
                    
                    val_output_surface[:,0] = self.valid_dataset.surface_inv_transform(val_input_surface.to('cpu')).numpy()
                    val_output_upper_air[:,0] = self.valid_dataset.upper_air_inv_transform(val_input_upper_air.to('cpu')).numpy()
                

                    for time_step in range(self.params['inference_steps']):
                        if self.params.has_diagnostic:
                            
                            val_out_surface, val_out_upper_air, val_out_diagnostic = self.diff_model.prediction(surface_in=val_input_surface, constant_boundary=self.constant_boundary_data, 
                                                                    varying_boundary=val_varying_boundary_data[:,time_step], 
                                                                    upper_air_in=val_input_upper_air, device=self.device)
                            # val_out_surface, val_out_upper_air, val_out_diagnostic, _, _ = self.model(val_input_surface, 
                            #                                                                     self.constant_boundary_data, 
                            #                                                                     val_varying_boundary_data[:,time_step],
                            #                                                                     val_input_upper_air)
                            
                            val_output_diagnostic[:, time_step + 1] = self.valid_dataset.diagnostic_inv_transform(val_out_diagnostic.to('cpu')).numpy()
  

                        else:
                            val_out_surface, val_out_upper_air = self.model(val_input_surface, self.constant_boundary_data, 
                                                                                val_varying_boundary_data[:,time_step], val_input_upper_air)
                        if self.params.predict_delta:
                            val_input_surface, val_input_upper_air = self.integrator(val_input_surface, val_input_upper_air, val_out_surface, val_out_upper_air)
                        else:
                            val_input_surface, val_input_upper_air = val_out_surface, val_out_upper_air
                    
                        val_output_surface[:,time_step + 1] = self.valid_dataset.surface_inv_transform(val_input_surface.to('cpu')).numpy()
                        val_output_upper_air[:,time_step + 1] = self.valid_dataset.upper_air_inv_transform(val_input_upper_air.to('cpu')).numpy()
                        
                        
                    
    
                    if self.params.has_diagnostic:
                        self.save_prediction(val_output_surface, val_output_upper_air , start_times, val_output_diagnostic, ens_id=ens_id)
                    else:
                        self.save_prediction(val_output_surface, val_output_upper_air, start_times, ens_id=ens_id)
        
                
        total_time = time.time() - total_start

        # if self.params.log_to_wandb:
        #     wandb.log(logs, step=self.epoch)
        return total_time
    

    def save_prediction(self, surface_prediction, upper_air_prediction, start_times, diagnostic_prediction = None, ens_id=None):
        print("Saving predictions...")
        
        if ens_id == 0 :
            print("_____________________________________________")
            print("start times for the first ensemble member:", start_times)
            print("_____________________________________________")
        inference_results_dir = self.params['experiment_dir']
        savedir = os.path.join(inference_results_dir, 'predictions')
        
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
            
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
                dataset = dataset.chunk({'time': 1, "level": 1})
                #filename = f'{self.params.nettype}_{self.params.run_num}_{self.params['timedelta_hours']}h_{self.params['inference_steps']}step_{self.params.val_start_year}_{batch_idx * self.params.batch_size + sample}.nc'
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
    parser.add_argument("--async_save", default = False, action="store_true", help="Enable asynchronous saving")
    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
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
