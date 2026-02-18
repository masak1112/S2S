
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Diffusion UNet building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepCondResBlock(nn.Module):
    """
    Residual block that accepts:
      - timestep embedding (t_emb)
      - conditioning vector (cond) — e.g. VAE latent z
    """
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch * 2))
        self.c_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch * 2))

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # Timestep conditioning via scale-shift (AdaGN style)
        t_scale, t_shift = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + t_scale) + t_shift

        # VAE conditioning via scale-shift
        c_scale, c_shift = self.c_proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + c_scale) + c_shift

        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight spatial self-attention for UNet bottleneck."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


# ---------------------------------------------------------------------------
# Conditional UNet
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    UNet denoiser conditioned on:
      1. Diffusion timestep t
      2. VAE latent z (encoder output)

    x_t -> predicted noise (or x_0, depending on parameterisation).
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 4, 8),
        latent_dim: int = 256,
        t_emb_dim: int = 512,
    ):
        super().__init__()

        # --- timestep embedding ---
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        channels = [base_channels * m for m in channel_mults]

        # --- encoder (down path) ---
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = channels[0]
        skip_channels = [in_ch]

        for out_ch in channels[:-1]:
            self.down_blocks.append(
                nn.ModuleList([
                    TimestepCondResBlock(in_ch, out_ch, t_emb_dim, latent_dim),
                    TimestepCondResBlock(out_ch, out_ch, t_emb_dim, latent_dim),
                ])
            )
            self.down_samples.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            skip_channels.append(out_ch)
            in_ch = out_ch

        # --- bottleneck ---
        mid_ch = channels[-1]
        self.mid_res1 = TimestepCondResBlock(in_ch, mid_ch, t_emb_dim, latent_dim)
        self.mid_attn = SelfAttention2d(mid_ch)
        self.mid_res2 = TimestepCondResBlock(mid_ch, mid_ch, t_emb_dim, latent_dim)

        # --- decoder (up path) ---
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        in_ch = mid_ch

        for out_ch in reversed(channels[:-1]):
            skip_ch = skip_channels.pop()
            self.up_samples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                )
            )
            self.up_blocks.append(
                nn.ModuleList([
                    TimestepCondResBlock(out_ch + skip_ch, out_ch, t_emb_dim, latent_dim),
                    TimestepCondResBlock(out_ch, out_ch, t_emb_dim, latent_dim),
                ])
            )
            in_ch = out_ch

        # --- output ---
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:   noisy image   (B, C, H, W)
            t:     timestep      (B,)  integers in [0, T)
            cond:  VAE latent z  (B, latent_dim)
        Returns:
            predicted noise      (B, C, H, W)
        """
        t_emb = self.t_emb(t)                  # (B, t_emb_dim)

        h = self.init_conv(x_t)
        skips = [h]

        for (rb1, rb2), ds in zip(self.down_blocks, self.down_samples):
            h = rb1(h, t_emb, cond)
            h = rb2(h, t_emb, cond)
            skips.append(h)
            h = ds(h)

        h = self.mid_res1(h, t_emb, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb, cond)

        for (rb1, rb2), us in zip(self.up_blocks, self.up_samples):
            h = us(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = rb1(h, t_emb, cond)
            h = rb2(h, t_emb, cond)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """
    Linear beta schedule and forward / reverse diffusion utilities.
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_bar", alphas_bar)
        self.register("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())

    def register(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    def to(self, device):
        for attr in ["betas", "alphas", "alphas_bar",
                     "sqrt_alphas_bar", "sqrt_one_minus_alphas_bar"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t given x_0 and t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: ConditionalUNet,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: sample x_{t-1} given x_t."""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_recip_a = (1.0 / self.alphas[t].sqrt()).view(-1, 1, 1, 1)

        eps_pred = model(x_t, t, cond)
        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_a * (x_t - betas_t / sqrt_1mab * eps_pred)

        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + betas_t.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: ConditionalUNet,
        shape: tuple,
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Full reverse diffusion loop."""
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            x = self.p_sample(model, x, t, cond)
        return x


# ---------------------------------------------------------------------------
# Combined model + training step
# ---------------------------------------------------------------------------

class ConditionalDiffusionModel(nn.Module):
    """
    Full pipeline:
      encoder (VAE) -> latent z -> diffusion UNet conditioned on z.
    """
    def __init__(
        self,
        img_channels: int = 3,
        vae_base_ch: int = 64,
        vae_ch_mults: tuple = (1, 2, 4),
        latent_dim: int = 256,
        unet_base_ch: int = 128,
        unet_ch_mults: tuple = (1, 2, 4, 8),
        t_emb_dim: int = 512,
        T: int = 1000,
        kl_weight: float = 1e-4,
        VAEEncoder = None,
        params = None,
        
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.params = params
        self.downscale_resolution = (VAEEncoder.patchembed3d.output_size[0]+1+1*self.params.upper_air_boundary,
                            (VAEEncoder.patchembed2d.output_size[0] - VAEEncoder.patchembed2d.output_size[0] % self.params.updown_scale_factor) \
                            // self.params.updown_scale_factor + VAEEncoder.patchembed2d.output_size[0] % self.params.updown_scale_factor,
                            (VAEEncoder.patchembed2d.output_size[1] - VAEEncoder.patchembed2d.output_size[1] % self.params.updown_scale_factor) \
                            // self.params.updown_scale_factor + VAEEncoder.patchembed2d.output_size[1] % self.params.updown_scale_factor)
        
        self.num_surface_vars = len(params.surface_variables)
        self.num_diagnostic_vars = len(params.diagnostic_variables)
        self.num_land_vars = len(params.land_variables)
        self.num_ocean_vars = len(params.ocean_variables)
        self.surface_prognostic_idxs = torch.cat((torch.arange(self.num_surface_vars).long(),
                                                  torch.arange(self.num_surface_vars + self.num_diagnostic_vars, self.num_surface_vars + self.num_diagnostic_vars + self.num_land_vars + self.num_ocean_vars).long()))
        #if self.predict_delta:
        self.encoder = VAEEncoder
        self.unet = ConditionalUNet(img_channels, unet_base_ch, unet_ch_mults, latent_dim, t_emb_dim)
        self.scheduler_diff = DDPMScheduler(T=T)

    def encode(self, x: torch.Tensor):
        """Encode image to VAE latent."""
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar

    def training_step(self, surface_in, constant_boundary, varying_boundary, upper_air_in, train = False) -> dict[str, torch.Tensor]:
        """
        Single training step.
        Returns dict with 'loss', 'diffusion_loss', 'kl_loss'.
        """
        
        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        surface_in = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)

        B = surface_in.size(0)
        device = surface_in.device
        self.scheduler_diff.to(device)
        # 1. Encode condition
        #######VAE ENCODER START ########
        surface_vae = self.encoder.patchembed2d(surface_in)
        upper_air_vae = self.encoder.patchembed3d(upper_air_in)
        x_vae = torch.concat([upper_air_vae, surface_vae.unsqueeze(2)], dim=2)
        
        B_vae, C_vae, Pl_vae, _, _ = x_vae.shape
        
        x_vae = x_vae.reshape(B_vae, C_vae, -1).transpose(1, 2)
        x_vae = self.encoder.layer1(x_vae)
        # skip = x_vae
        x_vae = self.encoder.downsample(x_vae) #8, 10350, 384
        x_vae = self.encoder.layer2(x_vae)
        x_vae = self.encoder.layer3(x_vae)
        print("x_vae shape before VAE in model fusion", x_vae.shape) # 1, 10350, 384

        x_vae = x_vae.reshape(B, self.downscale_resolution[0], self.downscale_resolution[1], self.downscale_resolution[2], -1).permute(0, 4, 1, 2, 3)
        print("x_vae reshaped  reshape ", x_vae.shape) 
        # ###########VAE Enocer 1#################
        mu = self.encoder.layer_mu(x_vae) # 
        print( "mu shape after VAE ", mu.shape)
        sigma = self.encoder.layer_sigma(x_vae) 
        print("sigma shape after VAE ", sigma.shape)
        norm = self.encoder.reparameterize(mu, sigma) #1, 192, 10, 23, 45
        print("norm shape after VAE ", norm.shape)
        z = norm.permute(0, 2, 3,4, 1).reshape(B_vae, -1, 192 * self.params.updown_scale_factor) #8, 10350, 384
        print("x shape after VAE reparameterize ", z.shape) # 2, 10350, 384
        #######VAE ENCODER END######## 

        # 2. Sample random timestep
        t = torch.randint(0, self.scheduler_diff.T, (B,), device=device)

        # 3. Forward diffusion
        z_t, noise = self.scheduler_diff.q_sample(z, t)

        # 4. Predict noise
        noise_pred = self.unet(z_t, t, z)

        # 5. Losses
        loss = F.mse_loss(noise_pred, noise)


        return loss 

    @torch.no_grad()
    def generate(
        self,
        condition_image: torch.Tensor,
        num_samples: int = 1,
        use_mean: bool = True,
    ) -> torch.Tensor:
        """
        Generate images conditioned on a source image's VAE encoding.

        Args:
            condition_image: (1, C, H, W) or (B, C, H, W) reference image
            num_samples:     how many samples to draw per condition
            use_mean:        if True use mu (deterministic), else sample z
        """
        device = condition_image.device
        mu, logvar = self.encoder(condition_image)
        z = mu if use_mean else VAEEncoder.reparameterize(mu, logvar)
        # Repeat condition for num_samples
        z = z.repeat_interleave(num_samples, dim=0)

        C, H, W = condition_image.shape[1:]
        shape = (z.size(0), C, H, W)
        return self.scheduler.sample(self.unet, shape, z, device)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
