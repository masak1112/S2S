"""Stochastic Interpolant for slow-surface fields (SST, skin/soil, soil moisture).

Why a dedicated model
---------------------
The joint 96-field postprocessor treats these four fields as 4 channels out of
96, where 85 upper-air channels dominate the loss. Measured against ERA5 they
behave nothing like the atmosphere:

  * error grows slowly and smoothly, over ocean-basin scales;
  * the dominant defect at long lead is a systematic drift, not spread
    (SST bias reaches -1.4 K by day 44 while RMSE is 3.3 K);
  * spread-skill ratio decays monotonically (SST SSR 1.04 at d1 -> 0.40 at d44)
    rather than partially recovering the way t2m and z500 do.

Three things this fixes relative to the joint model:

  1. Lead conditioning spans the full forecast range. The joint model clamps
     `lead_frac` at 10 steps, so days 10-44 are indistinguishable to it and one
     averaged correction gets smeared across all of them -- which is roughly
     right near day 30 and badly wrong at day 1 (it drove SST SSR to 0.19 and
     doubled CRPS there).
  2. The loss is masked per field. SST is land-filled at a constant 270 K over
     34% of cells and the soil fields are ocean-filled over 66%; training on
     those constants wastes capacity and leaks corrections across coastlines.
     `skin_temperature` is global and takes no mask.
  3. Members can be decorrelated at sampling time, so draws add ensemble
     dispersion instead of a common shift.

The interpolant itself is the same construction as networks/postprocess_si.py:
a linear path with the two-sided (antithetic) drift-matching objective,
learning `noise -> (ERA5 - member)` conditioned on the member and lead.
"""

import math

import torch
import torch.nn as nn

from networks.diffusion import ConUNet_1degV2


class SlowSurfaceSI(nn.Module):
    """SI over physical-space residuals for a handful of slow surface fields.

    Args:
        n_fields:      number of corrected fields (channels).
        field_masks:   [n_fields, H, W] bool tensor, True where a field is
                       valid. Stored as a buffer so it travels with the
                       checkpoint and is applied to both loss and samples.
        base_channels: UNet width. These fields are smooth and few, so this can
                       be far smaller than the joint model's.
        residual_std:  per-field normalisation for the residual target.
    """

    def __init__(
        self,
        n_fields: int,
        base_channels: int = 32,
        c_mults: tuple = (1, 2, 4),
        scale: tuple = (2, 2),
        gamma_type: str = 'brownian',
        dropout: float = 0.1,
        field_masks: torch.Tensor | None = None,
        residual_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.gamma_type = gamma_type

        # UNet input: [interpolant point, member]; conditioning: member + lead.
        self.unet = ConUNet_1degV2(
            dim_in=2 * n_fields,
            dim_cond=n_fields + 1,
            dim_out=n_fields,
            c=base_channels,
            c_mults=c_mults,
            scale=list(scale),
            dropout=dropout,
        )

        if residual_std is None:
            residual_std = torch.ones(n_fields)
        self.register_buffer('residual_std', residual_std.view(1, -1, 1, 1).clone())

        if field_masks is None:
            field_masks = torch.ones(n_fields, 1, 1, dtype=torch.bool)
        self.register_buffer('field_masks', field_masks.unsqueeze(0).clone())

    # ------------------------------------------------------------------
    # Interpolant schedule (linear path, gamma-perturbed)
    # ------------------------------------------------------------------

    @staticmethod
    def alpha(t):
        return 1.0 - t

    @staticmethod
    def alpha_dot(t):
        return -torch.ones_like(t)

    @staticmethod
    def beta(t):
        return t

    @staticmethod
    def beta_dot(t):
        return torch.ones_like(t)

    def gamma(self, t):
        if self.gamma_type == 'brownian':
            return torch.sqrt(torch.clamp(t * (1 - t), min=0.0))
        if self.gamma_type == 'bsquared':
            return t * (1 - t)
        if self.gamma_type in ('zero', None):
            return torch.zeros_like(t)
        raise NotImplementedError(f"gamma_type '{self.gamma_type}' not implemented.")

    def gamma_dot(self, t):
        if self.gamma_type == 'brownian':
            return (1 - 2 * t) / (2 * torch.clamp(torch.sqrt(t * (1 - t)), min=1e-3))
        if self.gamma_type == 'bsquared':
            return 1 - 2 * t
        if self.gamma_type in ('zero', None):
            return torch.zeros_like(t)
        raise NotImplementedError(f"gamma_type '{self.gamma_type}' not implemented.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise(self, r):
        return r / self.residual_std

    def _denormalise(self, r):
        return r * self.residual_std

    def _build_cond(self, member, lead_frac):
        if lead_frac is None:
            lead_frac = torch.zeros(member.shape[0], device=member.device,
                                    dtype=member.dtype)
        plane = lead_frac.view(-1, 1, 1, 1).expand(
            member.shape[0], 1, member.shape[-2], member.shape[-1])
        return torch.cat([member, plane], dim=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, member, target, lead_frac=None,
                      lower_upper: tuple = (1e-4, 1 - 1e-4)) -> torch.Tensor:
        """Masked two-sided drift-matching loss on (target - member).

        Args:
            member:    [B, n_fields, H, W] forecast member (normalised units).
            target:    [B, n_fields, H, W] verifying truth, same normalisation.
            lead_frac: [B] lead time scaled to [0, 1].
        """
        residual = self._normalise(target - member)
        B = residual.shape[0]
        device = residual.device

        lo, hi = lower_upper
        ts = lo + (hi - lo) * torch.rand(B, device=device, dtype=residual.dtype)
        t_b = ts.view(-1, 1, 1, 1)

        base = torch.randn_like(residual)
        It = self.alpha(t_b) * base + self.beta(t_b) * residual
        dIdt = self.alpha_dot(t_b) * base + self.beta_dot(t_b) * residual

        g, g_dot = self.gamma(t_b), self.gamma_dot(t_b)
        noise = torch.randn_like(residual)

        cond = self._build_cond(member, lead_frac)
        drift_p = self.unet(torch.cat([It + g * noise, member], dim=1), cond, ts)
        drift_m = self.unet(torch.cat([It - g * noise, member], dim=1), cond, ts)

        target_p = dIdt - noise * g_dot
        target_m = dIdt + noise * g_dot

        # Only score valid cells: SST is land-filled with a constant over 34% of
        # the grid and the soil fields are ocean-filled over 66%. Averaging over
        # those constants would let the model spend capacity predicting fill.
        mask = self.field_masks.to(drift_p.dtype)
        denom = mask.sum() * B / self.n_fields + 1e-8
        loss_p = (((drift_p - target_p) ** 2) * mask).sum() / (denom * self.n_fields)
        loss_m = (((drift_m - target_m) ** 2) * mask).sum() / (denom * self.n_fields)
        return 0.5 * (loss_p + loss_m)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_residual(self, member, lead_frac=None, num_samples: int = 1,
                        n_steps: int = 20, temperature: float = 1.0,
                        eps: float = 0.0) -> torch.Tensor:
        """Integrate the flow from noise to a residual field.

        Returns [B*num_samples, n_fields, H, W] in physical units, zeroed
        outside each field's valid mask.
        """
        mem = member.repeat_interleave(num_samples, dim=0)
        lf = (lead_frac.repeat_interleave(num_samples, dim=0)
              if lead_frac is not None else None)
        cond = self._build_cond(mem, lf)

        x_t = torch.randn(mem.shape[0], self.n_fields, mem.shape[-2], mem.shape[-1],
                          device=mem.device, dtype=mem.dtype) * temperature

        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=mem.device, dtype=mem.dtype)
        for i in range(n_steps):
            t, dt = ts[i], ts[i + 1] - ts[i]
            drift = self.unet(torch.cat([x_t, mem], dim=1), cond, t.expand(mem.shape[0]))
            x_t = x_t + drift * dt
            if eps > 0:
                x_t = x_t + math.sqrt(2 * eps) * torch.sqrt(dt) * torch.randn_like(x_t)

        out = self._denormalise(x_t)
        return out * self.field_masks.to(out.dtype)

    @torch.no_grad()
    def postprocess(self, members, lead_frac=None, draws_per_member: int = 1,
                    n_steps: int = 20, blend: float = 1.0,
                    temperature: float = 1.0,
                    spread_boost: float = 0.0) -> torch.Tensor:
        """Correct a base ensemble.

        `blend` scales the residual as in the joint model: the SI models a
        member's *total* error, so blend=1 makes `member + residual` a draw from
        p(truth | member) rather than a perturbation layered on top.

        `spread_boost` additionally inflates the member-to-member *deviation* of
        the residuals. Draws conditioned on similar members come out strongly
        correlated -- they act like a shared bias correction, which fixes drift
        but leaves the ensemble under-dispersed. Re-scaling the deviation about
        the residual mean adds dispersion without disturbing the mean
        correction. 0 keeps raw draws.
        """
        residual = self.sample_residual(
            members, lead_frac=lead_frac, num_samples=draws_per_member,
            n_steps=n_steps, temperature=temperature)

        if spread_boost > 0:
            m = residual.mean(dim=0, keepdim=True)
            residual = m + (1.0 + spread_boost) * (residual - m)
            residual = residual * self.field_masks.to(residual.dtype)

        base = members.repeat_interleave(draws_per_member, dim=0)
        return base + blend * residual
