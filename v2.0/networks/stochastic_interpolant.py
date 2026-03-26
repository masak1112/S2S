import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from functools import partial
from tqdm.auto import tqdm
from torch import nn, einsum, optim
from torch.nn import functional as F
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks.diffusion import ConditionalDiffusionModel



def make_It(path='linear', gamma = None, gamma_dot = None, gg_dot = None):
    """gamma function must be specified if using the trigonometric interpolant"""

    if path == 'linear':
        
        
        a      = lambda t: (1-t)
        adot   = lambda t: -1.0
        b      = lambda t: t
        bdot   = lambda t: 1.0
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'trig':
        if gamma == None:
            raise TypeError("Gamma function must be provided for trigonometric interpolant!")
        a    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)
        b    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        adot = lambda t: -gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t) \
                                - 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        bdot = lambda t: -gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t) \
                                + 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)

        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'encoding-decoding':

        a    = lambda t: torch.where(t <= 0.5, torch.cos(math.pi*t)**2, torch.tensor(0.))
        adot = lambda t: torch.where(t <= 0.5, -2*math.pi*torch.cos(math.pi*t)*torch.sin(math.pi*t), torch.tensor(0.))
        b    = lambda t: torch.where(t > 0.5,  torch.cos(math.pi*t)**2, 0.)
        bdot = lambda t: torch.where(t > 0.5,  -2*math.pi*torch.cos(math.pi*t)*torch.sin(math.pi*t), torch.tensor(0.))
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
    
    elif path == 'one-sided-linear':

        a      = lambda t: (1-t)
        adot   = lambda t: -1.0
        b      = lambda t: t
        bdot   = lambda t: 1.0
        
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1

    elif path == 'one-sided-trig':

        a      = lambda t: torch.cos(0.5*math.pi*t)
        adot   = lambda t: -0.5*math.pi*torch.sin(0.5*math.pi*t)
        b      = lambda t: torch.sin(0.5*math.pi*t)
        bdot   = lambda t: 0.5*math.pi*torch.cos(0.5*math.pi*t)

        
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'mirror':
        if gamma == None:
            raise TypeError("Gamma function must be provided for mirror interpolant!")
        
        a     = lambda t: gamma(t)
        adot  = lambda t: gamma_dot(t)
        b     = lambda t: torch.tensor(1.0)
        bdot  = lambda t: torch.tensor(0.0)
        
        It    = lambda t, x0, x1: b(t)*x1 + a(t)*x0
        dtIt  = lambda t, x0, x1: adot(t)*x0
        
    elif path == 'custom':
        return None, None, None

    else:
        raise NotImplementedError("The interpolant you specified is not implemented.")

    
    return It, dtIt, (a, adot, b, bdot)


def make_gamma(gamma_type = 'brownian', aval = None):
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    if gamma_type == 'brownian':
        gamma = lambda t: torch.sqrt(t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
        gg_dot = lambda t: (1/2)*(1-2*t)
        
    elif gamma_type == 'a-brownian':
        gamma = lambda t: torch.sqrt(a*t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(a*t*(1-t)))) * a*(1 -2*t)
        gg_dot = lambda t: (a/2)*(1-2*t)
        
    elif gamma_type == 'zero':
        gamma = gamma_dot = gg_dot = lambda t: torch.zeros_like(t)

    elif gamma_type == 'bsquared':
        gamma = lambda t: t*(1-t)
        gamma_dot = lambda t: 1 -2*t
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sinesquared':
        gamma = lambda t: torch.sin(math.pi * t)**2
        gamma_dot = lambda t: 2*math.pi*torch.sin(math.pi * t)*torch.cos(math.pi*t)
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sigmoid':
        f = torch.tensor(10.0)
        gamma = lambda t: torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f*(t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
        gamma_dot = lambda t: (-f)*( 1 - torch.sigmoid(-1 + f*(t - (1/2))) )*torch.sigmoid(-1 + f*(t - (1/2)))  + f*(1 - torch.sigmoid(1 + f*(t - (1/2)))  )*torch.sigmoid(1 + f*(t - (1/2)))
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == None:
        gamma     = lambda t: torch.zeros(1) ### no gamma
        gamma_dot = lambda t: torch.zeros(1) ### no gamma
        gg_dot    = lambda t: torch.zeros(1) ### no gamma
        
    else:
        raise NotImplementedError("The gamma you specified is not implemented.")
        
                
    return gamma, gamma_dot, gg_dot


#### here ye we define all the possible losses! For b, v, s, eta

def loss_per_sample_b(
    b: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    dtIt        = interpolant.dtIt(t, x0, x1)
    gamma_dot   = interpolant.gamma_dot(t)
    btp         = b(xtp, t)
    btm         = b(xtm, t)
    loss        = 0.5*torch.sum(btp**2) - torch.sum((dtIt + gamma_dot*z) * btp)
    loss       += 0.5*torch.sum(btm**2) - torch.sum((dtIt - gamma_dot*z) * btm)
    
    return loss
    
def loss_per_sample_s(
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    stp         = s(xtp, t)
    stm         = s(xtm, t)
    loss      = 0.5*torch.sum(stp**2) + (1 / interpolant.gamma(t))*torch.sum(stp*z)
    loss     += 0.5*torch.sum(stm**2) - (1 / interpolant.gamma(t))*torch.sum(stm*z)
    
    return loss


def loss_per_sample_eta(
    eta: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z   = interpolant.calc_xt(t, x0, x1)
    xt, t   = xt.unsqueeze(0), t.unsqueeze(0)
    eta_val = eta(xt, t)
    return 0.5*torch.sum(eta_val**2) + torch.sum(eta_val*z) 
    

def loss_per_sample_v(
    v: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt  = interpolant.dtIt(t, x0, x1)
    v_val = v(xt, t)
    
    return 0.5*torch.sum(v_val**2) - torch.sum(dtIt * v_val)



def loss_per_sample_one_sided_b(
    b: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample."""
    xt  = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt        = interpolant.dtIt(t, x0, x1)
    # gamma_dot   = interpolant.gamma_dot(t)
    bt          = b(xt, t)
    loss        = 0.5*torch.sum(bt**2) - torch.sum((dtIt) * bt)
    
    return loss

def loss_per_sample_one_sided_v(
    v: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample."""
    xt    = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt  = interpolant.dtIt(t, x0, x1)
    vt = v(xt, t)
    loss  = 0.5*torch.sum(vt**2) - torch.sum((dtIt) * vt)
    
    return loss



def loss_per_sample_one_sided_s(
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic samples for x_t = sqrt(1-t)z + sqrt(t) x1 where z=x0.
    """
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    stp         = s(xtp, t)
    stm         = s(xtm, t)
    alpha       = interpolant.a(t)
    
    loss      = 0.5*torch.sum(stp**2) + (1 / (alpha))*torch.sum(stp*x0)
    loss     += 0.5*torch.sum(stm**2) - (1 / (alpha))*torch.sum(stm*x0)
    
    return loss


def loss_per_sample_one_sided_eta(
    eta: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via samples for x_t = alpha(t)z + beta(t) x1 where z=x0.
    """
    xt         = interpolant.calc_xt(t, x0, x1)
    xt, t      = xt.unsqueeze(0), t.unsqueeze(0)
    etat         = eta(xt, t)
    loss      = 0.5*torch.sum(etat**2) + torch.sum(etat*x0)
    
    return loss


def loss_per_sample_mirror(
    s: Score,
    x0: Sample,
    x1: Sample,
    t: torch.tensor,
    interpolant: Interpolant
) -> torch.tensor:
    """Compute the loss on an individual sample via antithetic sampling."""
    xt        = interpolant.calc_xt(t, x0, x1)
    xt, t     = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt      = interpolant.dtIt(t, x0, x1)
    st        = s(xt, t)

    loss      = 0.5*torch.sum(st**2) + (1 / interpolant.gamma(t))*torch.sum(st*x0)
    
    return loss

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
    def __init__(self, path='linear', gamma_type='brownian', aval=None, **kwargs):
        super(StochasticInterpolant, self).__init__(**kwargs)
   

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
            return (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
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
        
    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)
    
    def I(self, x0, x1, t):
        return self.alpha(t) * x0 + self.beta(t) * x1
    
    def dIdt(self, x0, x1, t):
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1       
        
    
    def source_distribution(self, x: torch.Tensor, t: torch.Tensor | None = None, sigma: float | torch.Tensor | None = None):
        """Return a (diagonal) Gaussian distribution centered at latent `x`.

        Args:
            x: latent tensor with leading batch dimension (B, ...).
            t: optional time(s) used to derive a scale from `gamma(t)`; may be a
               scalar tensor of shape (B,) or a single float.
            sigma: optional fixed scale (float or tensor). If omitted and `t`
               is provided, uses `abs(gamma(t)) + eps` as scale. If both are
               omitted, a small default scale is used.

        Returns:
            A torch.distributions.Independent Normal distribution whose mean
            equals `x` and which has diagonal covariance given by `sigma**2`.
        """
        eps = 1e-6
        if sigma is None:
            if t is None:
                s = torch.tensor(1e-3, device=x.device, dtype=x.dtype)
            else:
                # allow scalar or per-batch t
                t_t = t
                if not torch.is_tensor(t_t):
                    t_t = torch.tensor(t, device=x.device, dtype=x.dtype)
                s = self.gamma(t_t).to(device=x.device, dtype=x.dtype)
                # ensure positivity and avoid exact zero
                s = s.abs() + eps
        else:
            s = sigma
            if not torch.is_tensor(s):
                s = torch.tensor(float(s), device=x.device, dtype=x.dtype)
            else:
                s = s.to(device=x.device, dtype=x.dtype)

        # Broadcast per-sample scales to the shape of x
        if s.dim() == 1 and x.dim() >= 2:
            view = [s.shape[0]] + [1] * (x.dim() - 1)
            s = s.view(*view)

        normal = torch.distributions.Normal(loc=x, scale=s)
        # treat all non-batch dims as event dims
        dist = torch.distributions.Independent(normal, reinterpreted_batch_ndims=x.dim() - 1)
        return dist

    def sample_from_source(self, x: torch.Tensor, t: torch.Tensor | None = None, sigma: float | torch.Tensor | None = None, n_samples: int = 1, reparam: bool = True) -> torch.Tensor:
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
        dist = self.source_distribution(x, t=t, sigma=sigma)
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
        ts  = lower + (upper - lower)*torch.rand(size=(B,))

        
        # Encode stochastic condition (VAE) and deterministic features
        z = self._encode_vae(surface, upper_air_in)
        x = self._encode_det(surface, upper_air_in, train)
        
    
        #source and target distributions
        base = self.sample_from_source(x, t=ts, sigma=None, n_samples=B, reparam=True)
        target = x  
        
        It = self.I(x0=base, x1=target, t=ts)
        dIdt = self.dIdt(x0=base, x1=target, t=ts)
        It_p = It + self.gamma(ts)*torch.randn_like(It).to(device)
        It_m = It - self.gamma(ts)*torch.randn_like(It).to(device)
        
        noise   = torch.randn(base).to(device)
        drift_p  = self.unet(It_p, z, ts)
        drift_m  = self.unet(It_m, z, ts)
        target_p = dIdt - noise * self.gamma_dot(ts) 
        target_m = dIdt + noise * self.gamma_dot(ts)
    
        
        loss_p= self.image_sq_norm(drift_p - target_p).mean()
        loss_m= self.image_sq_norm(drift_m - target_m).mean()
        
        loss = loss_p + loss_m
        return loss
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        