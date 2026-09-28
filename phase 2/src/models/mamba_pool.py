"""Phase-2 encoder: horizon-banked bidirectional Mamba-2, then memory pooling.

The phase-2 gate arms all use this module (see configs/model/ and scripts/run_phase2_gate.sh):

- mamba_padfix: horizon banks off, no pooling. The phase-1 encoder with only the padding
  fix, so the gate has a baseline that differs from the others in nothing else.
- mamba_banks: horizon banks on, no pooling. Isolates the effect of the bank init.
- mamba_uniform: banks on, keeps every Nth frame. Shorter memory, no Δ.
- mamba_pool_matched: banks on, keeps frames at equal quantiles of the short bank's
  cumulative Δ, with exactly as many frames per clip as mamba_uniform. Only the
  positions differ, so this is the clean test of whether Δ places the memory well.
- mamba_pool: banks on, keeps a frame whenever cumulative Δ crosses a fixed threshold
  (one sign at init). Free rate: the frame count drifts with Δ during training, so it is
  not matched to the uniform control. Watch the logged kept_frac.
- mamba_uniform_avg: banks on, each memory token is the plain mean of 16 consecutive frames.
- mamba_pool_avg: banks on, CIF-style. Segments end at equal quantiles of cumulative Δ (same
  count as uniform_avg) and each token is the Δ-weighted mean of its segment. Δ is the weight,
  so the decoder loss trains it directly; the pick-one-frame arms cannot.

The backward scan is reversed inside each clip's own length. The phase-1 encoder
is unchanged and remains the baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba2

from .scope import (
    apply_horizon_banks,
    delta_avg_pool,
    delta_pool,
    flip_valid,
    matched_delta_pool,
    short_head_delta,
    uniform_avg_pool,
    uniform_pool,
)


class ScopeMambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, headdim: int, d_conv: int,
                 dropout: float, short_heads: int, mid_heads: int, dt_weight_scale: float,
                 horizon_banks: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv)
        self.short_dt = None
        if horizon_banks:
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
                 short_heads: int = 8, mid_heads: int = 4, dt_weight_scale: float = 0.05,
                 horizon_banks: bool = True):
        super().__init__()
        if pool_mode not in ("delta", "delta_matched", "delta_avg", "uniform", "uniform_avg", "none"):
            raise ValueError(f"unknown pool_mode: {pool_mode!r}")
        if pool_mode.startswith("delta") and not horizon_banks:
            raise ValueError("Δ pooling reads the short bank, so it needs horizon_banks=True")
        self.pool_mode = pool_mode
        self.pool_every_frames = pool_every_frames
        self.short_heads = short_heads
        self.layers = nn.ModuleList([
            ScopeMambaLayer(
                d_model, d_state, expand, headdim, d_conv, dropout,
                short_heads, mid_heads, dt_weight_scale, horizon_banks,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # One short-sign of integrated Δ. short_dt is identical for every layer.
        short_dt = self.layers[0].short_dt or 0.0
        self.register_buffer(
            "pool_threshold",
            torch.tensor(short_dt * pool_every_frames, dtype=torch.float32),
            persistent=True,
        )
        self.reset_pool_stats()

    def reset_pool_stats(self) -> None:
        self._n_clips, self._n_frames, self._n_kept = 0, 0, 0

    def pop_pool_stats(self) -> dict:
        """Memory-length stats since the last call: the fraction of real frames the
        decoder sees, and the mean memory tokens per clip. Resets the counters."""
        stats = {}
        if self._n_clips:
            stats = {"kept_frac": self._n_kept / max(1, self._n_frames),
                     "mem_tokens": self._n_kept / self._n_clips}
        self.reset_pool_stats()
        return stats

    def _count(self, pad, mem_pad, B: int, T: int):
        self._n_clips += B
        self._n_frames += int((~pad).sum()) if pad is not None else B * T
        self._n_kept += int((~mem_pad).sum()) if mem_pad is not None else B * T

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        pad = src_key_padding_mask
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0)
        last_hidden = None
        last_layer = self.layers[-1]
        for layer in self.layers:
            if layer is last_layer and self.pool_mode.startswith("delta"):
                x, last_hidden = layer(x, pad, return_short_delta=True)
            else:
                x = layer(x, pad)
        x = self.norm(x)
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0)
        B, T = x.shape[:2]
        if self.pool_mode == "none":
            memory, mem_pad = x, pad
        elif self.pool_mode == "uniform":
            memory, mem_pad = uniform_pool(x, pad, self.pool_every_frames)
        elif self.pool_mode == "uniform_avg":
            memory, mem_pad = uniform_avg_pool(x, pad, self.pool_every_frames)
        elif self.pool_mode == "delta_avg":
            # With grad: Δ is the averaging weight, so the loss reaches the dt projection.
            delta = short_head_delta(last_layer.fwd, last_hidden, self.short_heads)
            memory, mem_pad = delta_avg_pool(x, delta, pad, self.pool_every_frames)
        else:
            with torch.no_grad():
                delta = short_head_delta(last_layer.fwd, last_hidden, self.short_heads)
            if self.pool_mode == "delta_matched":
                memory, mem_pad = matched_delta_pool(x, delta, pad, self.pool_every_frames)
            else:
                memory, mem_pad = delta_pool(x, delta, pad, float(self.pool_threshold))
        self._count(pad, mem_pad, B, T)
        return memory, mem_pad
