"""Phase-2 encoder: horizon-banked bidirectional Mamba-2, then memory pooling.

Two arms share this module and differ only in how the decoder memory is thinned:

- mamba_pool keeps a frame when the short-horizon bank's cumulative Δ crosses
  one sign of integrated time. The decoder no longer cross-attends to every frame.
- mamba_uniform keeps every Nth frame for the same N. Same shortening, no Δ.

The backward scan is reversed inside each clip's own length. The phase-1 encoder
is unchanged and remains the baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba2

from .scope import apply_horizon_banks, delta_pool, flip_valid, short_head_delta, uniform_pool


class ScopeMambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, headdim: int, d_conv: int,
                 dropout: float, short_heads: int, mid_heads: int, dt_weight_scale: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv)
        self.short_dt = apply_horizon_banks(self.fwd, short_heads, mid_heads, dt_weight_scale)
        apply_horizon_banks(self.bwd, short_heads, mid_heads, dt_weight_scale)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None,
                return_short_delta: bool = False):
        hidden = self.norm(x)
        if pad_mask is not None:
            hidden = hidden.masked_fill(pad_mask.unsqueeze(-1), 0)
        forward = self.fwd(hidden)
        backward = flip_valid(self.bwd(flip_valid(hidden, pad_mask)), pad_mask)
        merged = self.proj(torch.cat([forward, backward], dim=-1))
        out = x + self.dropout(merged)
        if pad_mask is not None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0)
        if return_short_delta:
            return out, hidden
        return out


class ScopePoolEncoder(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 5, d_state: int = 64, expand: int = 2,
                 headdim: int = 64, d_conv: int = 4, dropout: float = 0.1,
                 pool_mode: str = "delta", pool_every_frames: int = 16,
                 short_heads: int = 8, mid_heads: int = 4, dt_weight_scale: float = 0.05):
        super().__init__()
        if pool_mode not in ("delta", "uniform", "none"):
            raise ValueError(f"unknown pool_mode: {pool_mode!r}")
        self.pool_mode = pool_mode
        self.pool_every_frames = pool_every_frames
        self.short_heads = short_heads
        self.layers = nn.ModuleList([
            ScopeMambaLayer(
                d_model, d_state, expand, headdim, d_conv, dropout,
                short_heads, mid_heads, dt_weight_scale,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # One short-sign of integrated Δ. short_dt is identical for every layer.
        short_dt = self.layers[0].short_dt
        self.register_buffer(
            "pool_threshold",
            torch.tensor(short_dt * pool_every_frames, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        pad = src_key_padding_mask
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0)
        last_hidden = None
        last_layer = self.layers[-1]
        for layer in self.layers:
            if layer is last_layer and self.pool_mode == "delta":
                x, last_hidden = layer(x, pad, return_short_delta=True)
            else:
                x = layer(x, pad)
        x = self.norm(x)
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0)
        if self.pool_mode == "none":
            return x, pad
        if self.pool_mode == "uniform":
            return uniform_pool(x, pad, self.pool_every_frames)
        with torch.no_grad():
            delta = short_head_delta(last_layer.fwd, last_hidden, self.short_heads)
        return delta_pool(x, delta, pad, float(self.pool_threshold))
