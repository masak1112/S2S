"""Helpers for the persistent SST-anomaly conditioning of the latent SI.

When ``sst_cond_anom`` is on, the dataloader appends the conditioning map
[B, 2, h, w] as the LAST element of every batch. Consumers call
``pop_sst_cond`` first so their fixed-size tuple unpacking stays unchanged.
"""
import torch


def pop_sst_cond(data, params, device=None):
    """Return (data_without_cond, sst_cond_or_None)."""
    if not bool(getattr(params, 'sst_cond_anom', False)):
        return data, None
    data = list(data)
    cond = data.pop()
    if device is not None:
        cond = cond.to(device, dtype=torch.float32, non_blocking=True)
    return data, cond
