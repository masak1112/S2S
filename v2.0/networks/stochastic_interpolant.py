import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from tqdm.auto import tqdm
from torch import nn, einsum, optim
from torch.nn import functional as F
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks.diffusion import ConditionalDiffusionModel
from networks.diffusion import ConUNet_1degV2
from networks.vae import VAE



# class DriftScheduler(nn.Module):
#     def __init__(self,
#                  num_refinement_steps,  # this corresponds to physical time steps
#                  num_train_steps=None,  # number of training steps
#                  integrator='em', 
#                  sigma_coef=1.0,  
#                  beta_fn = "t",
#                  use_gF = False,
#                  antithetic_sampling=True,
#                  sigma_sample=None,
#                  ndim=2,
#                  ):
#         super(DriftScheduler, self).__init__()
        

class StochasticInterpolant(ConditionalDiffusionModel):
    def __init__(self, path='linear', gamma_type='zero', **kwargs):
        super(StochasticInterpolant, self).__init__(**kwargs)
        self.path = path
        self.gamma_type = gamma_type
   

    def alpha(self, t):
        return 1-t
    
    def alpha_dot(self, t):
        return -1.0
    
    
    def beta(self, t):
        return t
    
    def beta_dot(self, t):
        return 1.0
    
    def gamma(self, t): 
        if self.gamma_type == 'brownian':
            return torch.sqrt(t*(1-t))
        elif self.gamma_type == 'a-brownian':
            return torch.sqrt(self.aval*t*(1-t))
        elif self.gamma_type == 'zero':
            return torch.zeros_like(t)
        elif self.gamma_type == 'bsquared':
            return t*(1-t)
        elif self.gamma_type == 'sinesquared':
            return torch.sin(math.pi * t)**2
        elif self.gamma_type == 'sigmoid':  
            f = torch.tensor(10.0)
            return torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f*(t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
        elif self.gamma_type == None:
            return torch.zeros(1) ### no gamma
        else:
            raise NotImplementedError("The gamma you specified is not implemented.")
    
    
    def gamma_dot(self, t):
        if self.gamma_type == 'brownian':
            denom = torch.clamp(torch.sqrt(t*(1-t)), min=1e-3)
            return denom
        elif self.gamma_type == 'a-brownian':
            return (1/(2*torch.sqrt(self.aval*t*(1-t)))) * self.aval*(1 -2*t)
        elif self.gamma_type == 'zero':
            return torch.zeros_like(t)
        elif self.gamma_type == 'bsquared':
            return 1 -2*t
        elif self.gamma_type == 'sinesquared':
            return 2*math.pi*torch.sin(math.pi * t)*torch.cos(math.pi*t)
        elif self.gamma_type == 'sigmoid':  
            f = torch.tensor(10.0)
            return (-f)*( 1 - torch.sigmoid(-1 + f*(t - (1/2))) )*torch.sigmoid(-1 + f*(t - (1/2)))  + f*(1 - torch.sigmoid(1 + f*(t - (1/2)))  )*torch.sigmoid(1 + f*(t - (1/2)))
        elif self.gamma_type == None:
            return torch.zeros(1) ### no gamma
        else:
            raise NotImplementedError("The gamma you specified is not implemented.")
        
    # def image_sq_norm(self, x):
    #     return x.pow(2).sum(-1).sum(-1).sum(-1)
    
    def image_sq_norm(self, x):
        return x.pow(2).mean(-1).mean(-1).mean(-1)

    def I(self, x0, x1, t):
        return self.alpha(t) * x0 + self.beta(t) * x1
    
    def dIdt(self, x0, x1, t):
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1       
        
    
    def source_distribution(self, x: torch.Tensor):
        """Return a standard Gaussian distribution N(0, I) with the same shape as `x`.

        Args:
            x: tensor used only to determine shape, device, and dtype (B, ...).
            t: unused (kept for API compatibility).
            sigma: unused (kept for API compatibility).

        Returns:
            A torch.distributions.Independent Normal distribution with mean 0
            and unit variance over all non-batch dimensions.
        """
        normal = torch.distributions.Normal(
            loc=torch.zeros_like(x),
            scale=torch.ones_like(x),
        )
        dist = torch.distributions.Independent(normal, reinterpreted_batch_ndims=x.dim() - 1)
        return dist

    def sample_from_source(self, x: torch.Tensor, 
                           n_samples: int = 1,
                           reparam: bool = True) -> torch.Tensor:
        
        """Draw samples from the source distribution centered at `x`.

        Args:
            x: latent tensor (B, ...)
            t: optional time(s) to derive scale via `gamma(t)`
            sigma: optional scale override
            n_samples: number of independent samples to draw per batch item
            reparam: if True use `rsample` (reparameterized); otherwise use `sample`.

        Returns:
            Tensor of shape `(n_samples, B, ...)` if `n_samples>1`, else `(B, ...)`.
        """
        dist = self.source_distribution(x)
        if n_samples is None or n_samples <= 1:
            return dist.rsample() if reparam else dist.sample()
        samples = dist.rsample((n_samples,)) if reparam else dist.sample((n_samples,))
        return samples
        
    
    def training_step(self, surface_in, constant_boundary, varying_boundary, 
                      upper_air_in, train = True, plot_freq = 0, 
                      plot_path = "noise_comparison.png", iter=0,lower_upper: tuple = (0.0001, 0.9999)):
        
        """Single diffusion training step. Returns RMSE loss."""
        lower, upper = lower_upper[0], lower_upper[1]
        
        
        surface = self._prepare_surface(surface_in, constant_boundary, varying_boundary)
        B = surface.size(0)
        device = surface.device
        ts  = lower + (upper - lower)*torch.rand(size=(B,), device=device)

        
        # Encode stochastic condition (VAE) and deterministic features
        z = self._encode_vae(surface, upper_air_in)
        x = self._encode_det(surface, upper_air_in, train)
        
    
        #source and target distributions
        base = self.sample_from_source(x, n_samples=B, reparam=True)
        assert base.shape == x.shape, f"Expected shape of base noise to match x. Got {base.shape} and {x.shape}."
        print("base range from ", base.min().item(), " to ", base.max().item())
        
        target = x  
        print("target range from ", target.min().item(), " to ", target.max().item())
        
        It = self.I(x0=base, x1=target, t=ts)
        print("It range from ", It.min().item(), " to ", It.max().item())
        dIdt = self.dIdt(x0=base, x1=target, t=ts)
        print("dIdt range from ", dIdt.min().item(), " to ", dIdt.max().item())
        It_p = It + self.gamma(ts)*torch.randn_like(It).to(device)
        It_m = It - self.gamma(ts)*torch.randn_like(It).to(device)
        print("It_p range from ", It_p.min().item(), " to ", It_p.max().item())
        print("It_m range from ", It_m.min().item(), " to ", It_m.max().item())
        assert not torch.isnan(It_p).any(), f"It_p has NaN"
        assert not torch.isnan(z).any(), f"z has NaN"
        
        noise   = torch.randn_like(base).to(device)
        drift_p  = self.unet(It_p, z, ts)
        drift_m  = self.unet(It_m, z, ts)
        print("drift_p range from ", drift_p.min().item(), " to ", drift_p.max().item())
        target_p = dIdt - noise * self.gamma_dot(ts) 
        target_m = dIdt + noise * self.gamma_dot(ts)
        print("target_p range from ", target_p.min().item(), " to ", target_p.max().item())
        print("target_m range from ", target_m.min().item(), " to ", target_m.max().item())
    
        
        loss_p= self.image_sq_norm(drift_p - target_p).mean()
        loss_m= self.image_sq_norm(drift_m - target_m).mean()
        
        loss = loss_p + loss_m
        return loss
    
    
    def step_forward(self, x, drift, dt, g, eps = torch.tensor(0.5)):
        """
        Perform one step of the forward SDE: x_{t+dt} = x_t + drift*dt + squire(dt)*gamma(t)*eps, where dW ~ N(0, dt).
        """
        dW = torch.sqrt(dt)*torch.randn_like(x, device=x.device)
        #This is from Anthony
        #x = x + dt*drift + g*dW
        #This is from 
        x = x + dt*drift + torch.sqrt(2*eps) * dW
        return x
    
    
        
        
    @torch.no_grad()
    def generate(
        self,
        model_diff = None,
        z: torch.Tensor= None,
        num_samples: int = 1,
        sample_shape: tuple | None = None,
        device =  None, 
        x = None,
    ) -> torch.Tensor:
        """
        Generate images conditioned on a source image's VAE encoding.

        Args:
            z: conditional information
            num_samples:     how many samples to draw per condition
            use_mean:        if True use mu (deterministic), else sample z
            x: the latent space from deterministic encoder
        """
        # Repeat condition for num_samples
        z = z.repeat_interleave(num_samples, dim=0)
        assert x.shape == sample_shape
        if sample_shape is None:
            C, H, W = z.shape[1:]
        else:
            if len(sample_shape) == 4:
                C, H, W = sample_shape[1:]
            elif len(sample_shape) == 3:
                C, H, W = sample_shape
            else:
                raise ValueError(f"sample_shape must have 3 or 4 dims, got {sample_shape}")

        shape = (z.size(0), C, H, W)
    
        output = self.sample(model = model_diff, shape = shape, x = x, cond = z, device = device)
        return output


    @torch.no_grad()
    def sample(
        self,
        model: ConUNet_1degV2,
        shape: tuple,
        cond: torch.Tensor,
        device: torch.device,
        T: int = 20,
        start_end  = (0, 1),
        x = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Full reverse diffusion loop."""
        shape = x.shape
        x = self.sample_from_source(x, n_samples=shape[0], reparam=True).to(device)
        print("x after sample from source")
        self.start, self.end = start_end[0], start_end[1]
        self.ts = torch.linspace(self.start, self.end, T)
        
        
        for ii, t in enumerate(self.ts[:-1]):
                t_current = self.ts[ii]
                t_next = self.ts[ii+1] 
                dt = t_next - t_current 
                x = self.p_sample(model = model, x_t = x, t = t*1000, cond =cond, dt = dt)
                
        return x

    @torch.no_grad()
    def p_sample(
        self,
        model: ConUNet_1degV2,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        
        # Ensure time and dt tensors are on the same device as x_t
        t = t.to(x_t.device)
        dt = dt.to(x_t.device)
        drift = model(x_t, cond, t)
        y  = self.step_forward(x_t, drift, dt, self.gamma(t))

        return y




    def prediction(self, surface_in, constant_boundary, 
                   varying_boundary, upper_air_in, 
                   num_samples = 1, device = None):
        
        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        surface_in = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)

        B = surface_in.size(0)
        device = surface_in.device
        self.scheduler_diff.to(device)
        ###############encoder 2 start ########################
        # 1. Encode condition
        #######VAE ENCODER START ########
        surface_vae = self.encoder.patchembed2d(surface_in)
        upper_air_vae = self.encoder.patchembed3d(upper_air_in)
        x = torch.concat([upper_air_vae, surface_vae.unsqueeze(2)], dim=2)

        B_vae, C_vae, Pl_vae, _, _ = x.shape

        x_vae = x.reshape(B_vae, C_vae, -1).transpose(1, 2)
        x_vae = self.encoder.layer1(x_vae)
        # skip = x_vae
        x_vae = self.encoder.downsample(x_vae) #8, 10350, 384
        x_vae = self.encoder.layer2(x_vae)
        x_vae = self.encoder.layer3(x_vae)
        x_vae = x_vae.reshape(B, self.downscale_resolution[0], self.downscale_resolution[1], self.downscale_resolution[2], -1).permute(0, 4, 1, 2, 3)
        mu = self.encoder.layer_mu(x_vae) # 
        sigma = self.encoder.layer_sigma(x_vae) 
        norm = self.encoder.reparameterize(mu, sigma) #1, 192, 10, 23, 45
    
        z = norm.permute(0, 2, 3,4, 1).reshape(B_vae, Pl_vae, -1, 192 * self.params.updown_scale_factor) #8, 10350, 384
        # print("x shape after VAE reparameterize ", z.shape) # 2, 10, 1035, 384
        
        ###############encoder 1 start (deterministic) ########################
        surface_det = self.model_det.patchembed2d(surface_in)
        upper_air_det = self.model_det.patchembed3d(upper_air_in)
        x = torch.concat([upper_air_det, surface_det.unsqueeze(2)], dim=2)
   
        B_det, C_det, Pl_det, Lat, Lon = x.shape
        #print("x_det shape before reshape ", x.shape)  #torch.Size([2, 192, 10, 45, 90])

        x_det = x.reshape(B_det, C_det, -1).transpose(1, 2)
        x_det = self.model_det.layer1(x_det, train=False)
        skip = x_det
        x = self.model_det.downsample(x_det)
        x = self.model_det.layer2(x, train=False)
        x = self.model_det.layer3(x, train=False)
        x = x.reshape(B, Pl_det, -1,240*self.params.updown_scale_factor)
        target_latent_shape = x.shape
        
        if torch.isnan(x).any():
            print(f"[NaN check] x before diffusion has NaN: {torch.isnan(x).sum().item()} NaNs")
        else:
            print("[NaN check] x before diffusion : no NaN")     
            
        x = self.generate(
            model_diff=self.unet,
            z=z,
            x=x,
            num_samples=num_samples,
            sample_shape=target_latent_shape,
            device=device,
        )
        
        if torch.isnan(x).any():
            print(f"[NaN check] x has NaN: {torch.isnan(x).sum().item()} NaNs")
        else:
            print("[NaN check] x : no NaN")
        print("x shape after diffusion",x.shape)        
        x = x.reshape(B, -1 ,240*self.params.updown_scale_factor)
        
        ######## DETERMINISTIC DECODER START ######
        x = self.model_det.upsample(x)
        x = self.model_det.layer4(x, train=False)
        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl_det, Lat, Lon)
        output_surface = output[:, :, -1, :, :]
        output_upper_air = output[:, :, :-1, :, :]
        output_2D = self.model_det.patchrecovery2d(output_surface)
        output_surface = output_2D[:, self.surface_prognostic_idxs]
        if torch.isnan(output_surface).any():
            print(f"[NaN check] output_surface (after patchrecovery2d) has NaN: {torch.isnan(output_surface).sum().item()} NaNs")
        else:
            print("[NaN check] output_surface (after patchrecovery2d): no NaN")

        output_upper_air = self.model_det.patchrecovery3d(output_upper_air)
        output_diagnostic = output_2D[:, self.num_surface_vars:self.num_surface_vars + self.num_diagnostic_vars].reshape(
            output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1])
        
        
        return output_surface, output_upper_air, output_diagnostic 
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        