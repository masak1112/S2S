"""Output-space Stochastic Interpolant for ensemble spread postprocessing.

Motivation
----------
The rollout/decoder model produces an ensemble that is systematically
under-dispersive (SSR ~0.65-0.74 against ERA5, worst at 8-16 day leads).
The spread it does have is generated in *latent* space, so its magnitude is
only indirectly related to the actual forecast error.

This module trains a small SI in *physical output space* that learns the
conditional error distribution

    noise  ->  (ERA5 - member_i)      conditioned on (member_i, lead)

i.e. given one deterministic forecast member, generate plausible error
fields.  At inference each base member gets one (or more) residual draws
added to it, so the postprocessed ensemble spread reflects the true error
distribution rather than the base model's internal disagreement.

Why residual and not full-field
-------------------------------
Targeting the residual means the base forecast skill is preserved by
construction -- worst case the SI contributes nothing and the members are
unchanged.  Targeting the full field would require the SI to relearn the
atmospheric state from scratch before it could improve calibration.

Conditioning on lead time
-------------------------
The dispersion deficit is strongly lead-dependent (SSR 0.47 at day 8 vs
0.70 at day 45), so lead is a conditioning input.  A lead-blind model
averages those regimes and calibrates neither.

Everything upstream (VAE, deterministic encoder, latent SI, Pangu decoder)
stays frozen; only this module trains.
"""

import math

import torch
import torch.nn as nn

from networks.diffusion import ConUNet_1degV2


class PostprocessSI(nn.Module):
    """Stochastic interpolant over physical-space forecast residuals.

    The interpolant path is linear, ``I(t) = (1-t)*x0 + t*x1``, with
    ``x0 ~ N(0, sigma)`` and ``x1`` the (normalised) residual field.  The
    drift is learned with the standard two-sided (antithetic) SI objective,
    matching `networks.stochastic_interpolant.StochasticInterpolant`.

    Args:
        n_fields:      number of physical fields being corrected (channels).
        cond_channels: extra conditioning channels appended to the UNet input.
                       The member itself is always concatenated; lead time is
                       supplied as a broadcast plane when ``use_lead_plane``.
        base_channels: UNet width.
        gamma_type:    latent-noise schedule; 'brownian' matches the latent SI.
        residual_std:  per-field normalisation for the residual target. Stored
                       as a buffer so it travels with the checkpoint.
    """

    def __init__(
        self,
        n_fields: int,
        base_channels: int = 64,
        c_mults: tuple = (1, 2, 4),
        scale: tuple = (2, 2),
        gamma_type: str = 'brownian',
        dropout: float = 0.1,
        use_lead_plane: bool = True,
        residual_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.gamma_type = gamma_type
        self.use_lead_plane = use_lead_plane

        # UNet input: [interpolant point (n_fields), member (n_fields)]
        dim_in = 2 * n_fields
        # Conditioning tensor: member field + optional lead plane.
        dim_cond = n_fields + (1 if use_lead_plane else 0)

        self.unet = ConUNet_1degV2(
            dim_in=dim_in,
            dim_cond=dim_cond,
            dim_out=n_fields,
            c=base_channels,
            c_mults=c_mults,
            scale=list(scale),
            dropout=dropout,
        )

        # Per-field std used to normalise the residual target into ~unit
        # scale.  Without this, fields with very different magnitudes
        # (z500 ~ 500 m2/s2 vs precip ~ 1e-3 m) cannot share one UNet.
        if residual_std is None:
            residual_std = torch.ones(n_fields)
        self.register_buffer('residual_std', residual_std.view(1, -1, 1, 1).clone())

    # ------------------------------------------------------------------
    # Interpolant schedule
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
        if self.gamma_type == 'sinesquared':
            return torch.sin(math.pi * t) ** 2
        if self.gamma_type in ('zero', None):
            return torch.zeros_like(t)
        raise NotImplementedError(f"gamma_type '{self.gamma_type}' is not implemented.")

    def gamma_dot(self, t):
        if self.gamma_type == 'brownian':
            # Clamped to avoid the 1/0 blow-up at the path endpoints.
            return (1 - 2 * t) / (2 * torch.clamp(torch.sqrt(t * (1 - t)), min=1e-3))
        if self.gamma_type == 'bsquared':
            return 1 - 2 * t
        if self.gamma_type == 'sinesquared':
            return 2 * math.pi * torch.sin(math.pi * t) * torch.cos(math.pi * t)
        if self.gamma_type in ('zero', None):
            return torch.zeros_like(t)
        raise NotImplementedError(f"gamma_type '{self.gamma_type}' is not implemented.")

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def _normalise(self, residual: torch.Tensor) -> torch.Tensor:
        return residual / self.residual_std

    def _denormalise(self, residual: torch.Tensor) -> torch.Tensor:
        return residual * self.residual_std

    def _build_cond(self, member: torch.Tensor, lead_frac: torch.Tensor | None) -> torch.Tensor:
        """Assemble the conditioning tensor: member field (+ lead plane)."""
        cond = member
        if self.use_lead_plane:
            if lead_frac is None:
                lead_frac = torch.zeros(member.shape[0], device=member.device,
                                        dtype=member.dtype)
            plane = lead_frac.view(-1, 1, 1, 1).expand(
                member.shape[0], 1, member.shape[-2], member.shape[-1])
            cond = torch.cat([cond, plane], dim=1)
        return cond

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        member: torch.Tensor,
        target: torch.Tensor,
        lead_frac: torch.Tensor | None = None,
        lower_upper: tuple = (1e-4, 1 - 1e-4),
    ) -> torch.Tensor:
        """Two-sided SI drift-matching loss on the residual (target - member).

        Args:
            member:    [B, n_fields, H, W] one forecast member (normalised units).
            target:    [B, n_fields, H, W] verifying ERA5 at the same valid time.
            lead_frac: [B] lead time scaled to roughly [0, 1] for conditioning.

        Returns:
            Scalar loss.
        """
        residual = self._normalise(target - member)
        B = residual.shape[0]
        device = residual.device

        lo, hi = lower_upper
        ts = lo + (hi - lo) * torch.rand(B, device=device, dtype=residual.dtype)
        t_b = ts.view(-1, 1, 1, 1)

        # Source is unit Gaussian because the target is already normalised.
        base = torch.randn_like(residual)

        It = self.alpha(t_b) * base + self.beta(t_b) * residual
        dIdt = self.alpha_dot(t_b) * base + self.beta_dot(t_b) * residual

        g = self.gamma(t_b)
        g_dot = self.gamma_dot(t_b)
        noise = torch.randn_like(residual)

        It_p = It + g * noise
        It_m = It - g * noise

        cond = self._build_cond(member, lead_frac)
        drift_p = self.unet(torch.cat([It_p, member], dim=1), cond, ts)
        drift_m = self.unet(torch.cat([It_m, member], dim=1), cond, ts)

        target_p = dIdt - noise * g_dot
        target_m = dIdt + noise * g_dot

        loss_p = (drift_p - target_p).pow(2).mean()
        loss_m = (drift_m - target_m).pow(2).mean()
        return 0.5 * (loss_p + loss_m)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_residual(
        self,
        member: torch.Tensor,
        lead_frac: torch.Tensor | None = None,
        num_samples: int = 1,
        n_steps: int = 20,
        eps: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Integrate the probability-flow / SDE path to draw residual fields.

        Args:
            member:      [B, n_fields, H, W]
            lead_frac:   [B] lead conditioning
            num_samples: draws per member
            n_steps:     Euler steps along t in [0, 1]
            eps:         extra SDE diffusion; 0 gives the deterministic flow,
                         which is the calibrated choice once the SI is trained.
            temperature: scale on the initial noise; >1 inflates spread.

        Returns:
            [B*num_samples, n_fields, H, W] residuals in physical units.
        """
        mem = member.repeat_interleave(num_samples, dim=0)
        lf = (lead_frac.repeat_interleave(num_samples, dim=0)
              if lead_frac is not None else None)
        cond = self._build_cond(mem, lf)

        x_t = torch.randn(
            mem.shape[0], self.n_fields, mem.shape[-2], mem.shape[-1],
            device=mem.device, dtype=mem.dtype,
        ) * temperature

        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=mem.device, dtype=mem.dtype)
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - t
            t_batch = t.expand(mem.shape[0])
            drift = self.unet(torch.cat([x_t, mem], dim=1), cond, t_batch)
            x_t = x_t + drift * dt
            if eps > 0:
                x_t = x_t + math.sqrt(2 * eps) * torch.sqrt(dt) * torch.randn_like(x_t)

        return self._denormalise(x_t)

    @torch.no_grad()
    def postprocess(
        self,
        members: torch.Tensor,
        lead_frac: torch.Tensor | None = None,
        draws_per_member: int = 1,
        n_steps: int = 20,
        temperature: float = 1.0,
        blend: float = 1.0,
    ) -> torch.Tensor:
        """Turn a base ensemble into a calibrated one via SI residual draws.

        The SI models ``p(truth - member_i)``, i.e. the *total* error of one
        member.  So ``member_i + residual_draw`` is itself a sample from
        ``p(truth | member_i)`` and stands in for the member -- it is not a
        perturbation to be added on top of it.  Adding a full-magnitude draw
        to a member that already carries its own error would compound the two
        and roughly double the ensemble variance (SSR overshoots ~2x).

        `blend` interpolates between the two readings:

            blend = 1.0  ->  member + residual        (replacement; calibrated)
            blend = 0.0  ->  member                   (base ensemble unchanged)

        Intermediate values shrink the correction toward the base member,
        which is useful when the base ensemble already carries part of the
        error and only needs topping up.  ``blend`` therefore acts as the
        calibration knob: tune it on validation so SSR lands near 1.

        Args:
            members: [M, n_fields, H, W] base ensemble for one IC/lead.

        Returns:
            [M*draws_per_member, n_fields, H, W] corrected ensemble.
        """
        residual = self.sample_residual(
            members, lead_frac=lead_frac, num_samples=draws_per_member,
            n_steps=n_steps, temperature=temperature,
        )
        base = members.repeat_interleave(draws_per_member, dim=0)
        return base + blend * residual

    @staticmethod
    def calibrate_blend(
        base_ens: torch.Tensor,
        residual: torch.Tensor,
        truth: torch.Tensor,
        lat_weights: torch.Tensor | None = None,
        grid: tuple = (0.0, 1.5, 31),
    ) -> float:
        """Pick the blend weight whose ensemble spread best matches its error.

        Scans candidate blend values and returns the one whose spread-skill
        ratio is closest to 1.  This is a one-parameter fit on held-out data,
        so it calibrates without retraining -- and because SSR varies strongly
        with lead time, it is worth fitting per lead.

        Args:
            base_ens: [M, H, W] base ensemble for one field / IC / lead.
            residual: [M, H, W] SI residual draws aligned with base_ens.
            truth:    [H, W] verifying field.
            lat_weights: [H, 1] latitude weights, or None for unweighted.

        Returns:
            The blend value in the scanned grid with SSR closest to 1.
        """
        w = (torch.ones(base_ens.shape[-2], 1, device=base_ens.device,
                        dtype=base_ens.dtype) if lat_weights is None else lat_weights)
        wsum = w.sum() * base_ens.shape[-1]

        best_b, best_gap = 1.0, float('inf')
        lo, hi, n = grid
        for i in range(int(n)):
            b = lo + (hi - lo) * i / max(1, int(n) - 1)
            ens = base_ens + b * residual
            err2 = ((((ens.mean(dim=0) - truth) ** 2) * w).sum() / wsum)
            spread2 = ((ens.var(dim=0, unbiased=True) * w).sum() / wsum)
            ssr = torch.sqrt(spread2 / (err2 + 1e-12)).item()
            gap = abs(ssr - 1.0)
            if gap < best_gap:
                best_gap, best_b = gap, b
        return best_b
