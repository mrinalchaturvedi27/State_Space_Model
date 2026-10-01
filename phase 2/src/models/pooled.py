"""Encoder-agnostic memory pooling: any encoder, then the phase-2 plain-average pooling.

Used for transformer_uniform_avg -- the control for "is 15x memory compression something
special about Mamba, or does any encoder survive it?". No parameters are added, so the
Transformer arm keeps its phase-1 parameter count.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .scope import uniform_avg_pool


class UniformAvgPooledEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, stride: int = 16):
        super().__init__()
        self.encoder = encoder
        self.stride = stride
        self.reset_pool_stats()

    def reset_pool_stats(self) -> None:
        self._n_clips, self._n_frames, self._n_kept = 0, 0, 0

    def pop_pool_stats(self) -> dict:
        stats = {}
        if self._n_clips:
            stats = {"kept_frac": self._n_kept / max(1, self._n_frames),
                     "mem_tokens": self._n_kept / self._n_clips}
        self.reset_pool_stats()
        return stats

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        if isinstance(out, tuple):
            out, src_key_padding_mask = out
        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
        memory, mem_pad = uniform_avg_pool(out, src_key_padding_mask, self.stride)
        B, T = x.shape[:2]
        self._n_clips += B
        self._n_frames += int((~src_key_padding_mask).sum()) if src_key_padding_mask is not None else B * T
        self._n_kept += int((~mem_pad).sum())
        return memory, mem_pad
