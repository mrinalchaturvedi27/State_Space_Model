"""A2/A2b encoder: bidirectional Mamba-2, no positional encoding (BENCHMARK_PLAN.md §3).

Bidirectional because the Transformer encoder (A1) is bidirectional -- a unidirectional-SSM
encoder would be a confounded comparison (it's ablation 2 instead, see §7). Each layer runs
a forward and backward Mamba-2 scan over the same pre-LN input and concatenates+projects,
mirroring a pre-LN residual block.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class BiMamba2Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 256, expand: int = 2, headdim: int = 64,
                d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, d_conv=d_conv)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        b = self.bwd(h.flip(dims=[1])).flip(dims=[1])
        merged = self.proj(torch.cat([f, b], dim=-1))
        return x + self.dropout(merged)


class BiMambaEncoder(nn.Module):
    """No positional encoding -- the recurrence is inherently ordered; this is the disclosed
    mechanism difference from the Transformer arm (§3), not an oversight.

    Padding handling: padded positions are zeroed before/after every layer rather than packed
    with cu_seqlens/varlen kernels. This is an approximation (the d_conv=4 local convolution
    mixes a few pad positions at the real/pad boundary) but matches common public Mamba-seq2seq
    baselines and is simple to keep byte-identical across seeds; revisit with varlen packing if
    the padding fraction turns out to matter at T_max=512 batch composition.
    """

    def __init__(self, d_model: int = 512, n_layers: int = 5, d_state: int = 256, expand: int = 2,
                headdim: int = 64, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMamba2Layer(d_model, d_state, expand, headdim, d_conv, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        def zero_pad(t: torch.Tensor) -> torch.Tensor:
            if src_key_padding_mask is None:
                return t
            return t.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        x = zero_pad(x)
        for layer in self.layers:
            x = zero_pad(layer(x))
        return self.norm(x)
