"""A1 encoder: 6-layer pre-LN Transformer with sinusoidal PE (BENCHMARK_PLAN.md §3)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.pos_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        # Final norm required with norm_first=True (see build_decoder's docstring in common.py for
        # why); also keeps parity with BiMambaEncoder, which already ends with its own LayerNorm.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_dropout(self.pos(x))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
