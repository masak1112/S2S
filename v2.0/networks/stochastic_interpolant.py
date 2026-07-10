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
                      upper_air_in, target_surface_in=None, target_upper_air=None,
                      train = True, plot_freq = 0,
                      plot_path = "noise_comparison.png", iter=0, lower_upper: tuple = (0.0001, 0.9999),
                      plot_scatter: bool = False, scatter_path: str = "scatter_pred_gt.png"):

        """Single diffusion training step. Returns RMSE loss.

        The SI is trained on the residual between the ground-truth t+1 latent
        (encoded by the frozen deterministic Pangu encoder) and Pangu's own
        smoothed mean-latent prediction, i.e. what Pangu's encoder misses.
        """
        lower, upper = lower_upper[0], lower_upper[1]

        if target_surface_in is None or target_upper_air is None:
            raise ValueError(
                "training_step requires target_surface_in/target_upper_air (ground-truth "
                "t+1 fields) to form the Pangu-latent residual target."
            )

        surface = self._prepare_surface(surface_in, constant_boundary, varying_boundary)
        target_surface = self._prepare_surface(target_surface_in, constant_boundary, varying_boundary)
        B = surface.size(0)
        device = surface.device
        ts  = lower + (upper - lower)*torch.rand(size=(B,), device=device)

        # Encode stochastic condition (VAE) and deterministic features
        z = self._encode_vae(surface, upper_air_in)
        x, skip = self._encode_det(surface, upper_air_in, train=True)

        # Ground-truth t+1 latent, encoded through the same frozen Pangu path.
        with torch.no_grad():
            x_true, _ = self._encode_det(target_surface, target_upper_air, train=False)

        # What Pangu's mean-latent prediction misses.
        residual = x_true - x

        #source and target distributions
        base = self.sample_from_source(residual, n_samples=B, reparam=True)
        assert base.shape == residual.shape, f"Expected shape of base noise to match residual. Got {base.shape} and {residual.shape}."


        target = residual
        It = self.I(x0=base, x1=target, t=ts)
        dIdt = self.dIdt(x0=base, x1=target, t=ts)
        It_p = It + self.gamma(ts)*torch.randn_like(It).to(device)
        It_m = It - self.gamma(ts)*torch.randn_like(It).to(device)

        assert not torch.isnan(It_p).any(), f"It_p has NaN"
        assert not torch.isnan(z).any(), f"z has NaN"

        noise   = torch.randn_like(base).to(device)
        drift_p  = self.unet(It_p, z, ts)
        drift_m  = self.unet(It_m, z, ts)

        target_p = dIdt - noise * self.gamma_dot(ts)
        target_m = dIdt + noise * self.gamma_dot(ts)

        loss_p= self.image_sq_norm(drift_p - target_p).mean()
        loss_m= self.image_sq_norm(drift_m - target_m).mean()

        loss = loss_p + loss_m

        if plot_scatter:
            self._plot_scatter(drift_p, target_p, iter=iter, save_path=scatter_path)

        return loss

    def _plot_scatter(self, pred: torch.Tensor, target: torch.Tensor,
                      iter: int = 0, save_path: str = "scatter_pred_gt.png",
                      max_points: int = 4096) -> None:
        """Scatter plot of predicted vs ground-truth drift for the first batch item."""
        pred_np   = pred[0].detach().cpu().float().flatten().numpy()
        target_np = target[0].detach().cpu().float().flatten().numpy()

        if len(pred_np) > max_points:
            idx = np.random.default_rng(iter).choice(len(pred_np), max_points, replace=False)
            pred_np, target_np = pred_np[idx], target_np[idx]

        corr = float(np.corrcoef(pred_np, target_np)[0, 1]) if len(pred_np) > 1 else float('nan')

        lim = max(float(np.abs(target_np).max()), float(np.abs(pred_np).max())) * 1.05
        lim = lim if lim > 0 else 1.0

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(target_np, pred_np, s=2, alpha=0.25, linewidths=0, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.2, label='y = x')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Ground truth drift")
        ax.set_ylabel("Predicted drift")
        ax.set_title(f"Pred vs GT drift  iter={iter}  r={corr:.3f}")
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    
    def step_forward(self, x, drift, dt, g, eps = torch.tensor(0.99)):
        """
        Perform one step of the forward SDE: x_{t+dt} = x_t + drift*dt + squire(dt)*gamma(t)*eps, where dW ~ N(0, dt).
        """
        dW = torch.sqrt(dt)*torch.randn_like(x, device=x.device)
        #This is from Anthony
        #x = x + dt*drift + g*dW
    
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
        T: int = 15,
        start_end  = (0, 1),
        x = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Full reverse diffusion loop."""
        shape = x.shape
        x = self.sample_from_source(x, n_samples=shape[0], reparam=True).to(device)
       
        self.start, self.end = start_end[0], start_end[1]
        self.ts = torch.linspace(self.start, self.end, T)
        
        
        for ii, t in enumerate(self.ts[:-1]):
                t_current = self.ts[ii]
                t_next = self.ts[ii+1] 
                dt = t_next - t_current 
                x = self.p_sample(model = model, x_t = x, t = t, cond =cond, dt = dt)
                
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
                   num_samples = 1, device = None, seed = None):
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        surface_in = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)

        B = surface_in.size(0)
        device = surface_in.device
        self.scheduler_diff.to(device)
        
        ###############encoder 2 start ########################
        # 1. Encode condition
        #######VAE ENCODER START ########

    
        z  = self._encode_vae(surface_in, upper_air_in)    

        ###############encoder 1 start (deterministic) ########################
        surface_det = self.model_det.patchembed2d(surface_in)
        upper_air_det = self.model_det.patchembed3d(upper_air_in)
        x = torch.concat([upper_air_det, surface_det.unsqueeze(2)], dim=2)
        _, _, Pl_det, Lat, Lon = x.shape
        
        x, skip = self._encode_det(surface_in, upper_air_in, train=False)
        target_latent_shape = x.shape

        # if torch.isnan(x).any():
        #     print(f"[NaN check] x before diffusion has NaN: {torch.isnan(x).sum().item()} NaNs")
        # else:
        #     print("[NaN check] x before diffusion : no NaN")

        residual = self.generate(
            model_diff=self.unet,
            z=z,
            x=x,
            num_samples=num_samples,
            sample_shape=target_latent_shape,
            device=device,
        )

        # Add the SI-sampled residual back onto Pangu's mean-latent prediction
        # before decoding, since the SI was trained on x_true - x_pangu.
        x = x.repeat_interleave(num_samples, dim=0) + residual

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
        # if torch.isnan(output_surface).any():
        #     print(f"[NaN check] output_surface (after patchrecovery2d) has NaN: {torch.isnan(output_surface).sum().item()} NaNs")
        # else:
        #     print("[NaN check] output_surface (after patchrecovery2d): no NaN")

        output_upper_air = self.model_det.patchrecovery3d(output_upper_air)
        output_diagnostic = output_2D[:, self.num_surface_vars:self.num_surface_vars + self.num_diagnostic_vars].reshape(
            output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1])

        
        return output_surface, output_upper_air, output_diagnostic 
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        