# Conditional Layer Normalization — Implementation Summary

Reference: https://arxiv.org/pdf/2103.00993

## Formulation

```
CLN(x, c) = γ(c) · (x − μ) / σ + β(c)
```

where γ(c) and β(c) are produced by two separate linear layers from a conditioning vector `c` (shape `(B, cond_dim)`), rather than being fixed learned parameters as in standard LayerNorm.

## New class: `ConditionalLayerNorm`

Location: `pangu.py:76`

```python
class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim):
        # nn.LayerNorm with elementwise_affine=False (no fixed γ/β)
        # gamma_proj: Linear(cond_dim → normalized_shape), init weights=1, bias=0
        # beta_proj:  Linear(cond_dim → normalized_shape), init weights=0, bias=0

    def forward(self, x, c):
        # x: (B, N, C),  c: (B, cond_dim)
        # returns γ(c) * norm(x) + β(c)
```

Initialisation ensures γ starts as identity (scale=1) and β starts as zero, so the module is a drop-in for plain LayerNorm at the start of training.

## Modified classes

All changes are **backward compatible**: passing `cond_dim=None` (the default) keeps the original `nn.LayerNorm` behaviour.

| Class | What changed |
|---|---|
| `EarthSpecificBlock` | `norm1`, `norm2` → `ConditionalLayerNorm`; `forward(x, c=None)` |
| `EarthSpecificLayer` | Passes `cond_dim` to each block; `forward(x, train, c=None)` |
| `DownSample` | `norm` → `ConditionalLayerNorm`; `forward(x, c=None)` |
| `UpSample` | `norm` → `ConditionalLayerNorm`; `forward(x, c=None)` |
| `PanguModel_Plasim` | `__init__` accepts `cond_dim=None`; `forward(..., cond=None)` threads `cond` to all sub-modules |

## Usage

```python
# Instantiate with a conditioning dimension
model = PanguModel_Plasim(params, cond_dim=128)

# At forward time, pass a (B, cond_dim) conditioning tensor
cond = some_embedding(lead_time_or_climate_index)  # (B, 128)
output_surface, output_upper_air = model(
    surface_in, constant_boundary, varying_boundary, upper_air_in,
    train=True, cond=cond
)
```

When `cond=None` (or `cond_dim=None` at construction), the model falls back to standard LayerNorm with no change in behaviour.
