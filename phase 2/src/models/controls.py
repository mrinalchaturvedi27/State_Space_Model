"""Control encoders for the diagnostic program (RESEARCH_PROGRAM_2026-10-08.md step 4).

The diagnostics showed the Transformer is nearly order-blind (shuffling every frame costs it -0.35
chrF2 on iSign vs -4.21 for Mamba). Mamba-2 contains a short depthwise temporal convolution in every
layer; the Transformer has none and uses absolute sinusoidal positions. These arms test which missing
ingredient explains Mamba's advantage. Each is parameter-matched to the Transformer (+-2%):

  transformer_convstem  the phase-1 Transformer + a convolutional stem: 2 residual depthwise-separable
                        temporal conv blocks (kernel 5) before the positional encoding.
                        Receptive field of the stem alone: 9 frames.
  transformer_relpos    the phase-1 Transformer with absolute positions replaced by ALiBi-style
                        symmetric relative biases (-slope_h * |i - j| per head); no extra parameters.
  tcn                   no attention: residual blocks of a depthwise dilated temporal conv
                        (kernel 5, dilations 1,2,4,8 repeated) + a pointwise FFN. Receptive field
                        1 + sum((k - 1) * d) frames (121 frames for 8 layers), printed by receptive_field().
                        tcn_wide (config only): dilations 1,2,...,64,1 -> 513 frames, the whole clip.
  transformer_local     the phase-1 Transformer with attention restricted to |i - j| <= local_half_width
                        (default 10) in every layer: receptive field 1 + 2 * 10 * 6 = 121 frames,
                        matched to the TCN, with no convolution.
Padded frames are zeroed before every convolution, so a clip's encoding does not depend on padding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEncoding


def _zero_pad(x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
    return x if pad is None else x.masked_fill(pad.unsqueeze(-1), 0.0)


class ConvBlock(nn.Module):
    """Pre-LN residual depthwise-separable temporal conv: LN -> depthwise conv -> GELU -> pointwise."""

    def __init__(self, d_model: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=dilation * (kernel - 1) // 2,
                            dilation=dilation, groups=d_model)
        self.pw = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad=None):
        h = _zero_pad(self.norm(x), pad)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(self.pw(nn.functional.gelu(h)))


def _layer(d_model, n_heads, ffn, dropout):
    return nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
                                      activation="gelu", batch_first=True, norm_first=True)


class ConvStemTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8, dim_feedforward=2048, dropout=0.1,
                 max_len=1024, stem_blocks=2, stem_kernel=5):
        super().__init__()
        self.stem = nn.ModuleList([ConvBlock(d_model, stem_kernel, 1, dropout) for _ in range(stem_blocks)])
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.pos_dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(_layer(d_model, n_heads, dim_feedforward, dropout), n_layers,
                                             norm=nn.LayerNorm(d_model))
        self.stem_receptive_field = 1 + stem_blocks * (stem_kernel - 1)

    def forward(self, x, src_key_padding_mask=None):
        x = _zero_pad(x, src_key_padding_mask)
        for blk in self.stem:
            x = _zero_pad(blk(x, src_key_padding_mask), src_key_padding_mask)
        x = self.pos_dropout(self.pos(x))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes 2^(-8/n), 2^(-16/n), ... (Press et al., ALiBi)."""
    start = 2 ** (-8.0 / n_heads)
    return torch.tensor([start ** (h + 1) for h in range(n_heads)])


class RelPosTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)
        self.in_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_layer(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def bias(self, B: int, T: int, pad: torch.Tensor | None, device, dtype) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        dist = (pos[None, :] - pos[:, None]).abs().to(dtype)
        b = -self.slopes.to(device=device, dtype=dtype)[:, None, None] * dist          # (H, T, T)
        b = b.unsqueeze(0).expand(B, -1, -1, -1)
        if pad is not None:                                                             # padding as -inf keys
            b = b.masked_fill(pad[:, None, None, :], float("-inf"))
        return b.reshape(B * self.n_heads, T, T)

    def forward(self, x, src_key_padding_mask=None):
        B, T, _ = x.shape
        mask = self.bias(B, T, src_key_padding_mask, x.device, torch.float32)
        x = self.in_dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x)


class TCNEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=8, kernel=5, dilations=(1, 2, 4, 8), dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.dil = [dilations[i % len(dilations)] for i in range(n_layers)]
        self.convs = nn.ModuleList([ConvBlock(d_model, kernel, d, dropout) for d in self.dil])
        self.ffns = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
                                   for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.kernel = kernel

    def receptive_field(self) -> int:
        return 1 + sum((self.kernel - 1) * d for d in self.dil)

    def forward(self, x, src_key_padding_mask=None):
        pad = src_key_padding_mask
        x = _zero_pad(x, pad)
        for conv, ffn in zip(self.convs, self.ffns):
            x = _zero_pad(conv(x, pad), pad)
            x = _zero_pad(x + ffn(x), pad)
        return self.norm(x)


class LocalAttnTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8, dim_feedforward=2048, dropout=0.1, max_len=1024,
                 half_width=10):
        super().__init__()
        self.n_heads, self.half_width = n_heads, half_width
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.pos_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_layer(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def receptive_field(self) -> int:
        return 1 + 2 * self.half_width * len(self.layers)

    def mask(self, B, T, pad, device):
        pos = torch.arange(T, device=device)
        far = (pos[None, :] - pos[:, None]).abs() > self.half_width                     # (T, T)
        m = far.unsqueeze(0).expand(B, -1, -1)
        if pad is not None:
            m = m | pad[:, None, :]
        m = m & ~torch.eye(T, dtype=torch.bool, device=device)  # every query sees itself: no empty rows
        bias = torch.zeros(B, T, T, device=device).masked_fill(m, float("-inf"))
        return bias.repeat_interleave(self.n_heads, dim=0)

    def forward(self, x, src_key_padding_mask=None):
        B, T, _ = x.shape
        mask = self.mask(B, T, src_key_padding_mask, x.device)
        x = self.pos_dropout(self.pos(x))
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x)
