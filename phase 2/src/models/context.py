"""C1 model: the SSM encoder reads earlier clips of the same video, the decoder does not.

Input frames carry two flag channels after the d_in pose features (see
src/data.py:ContextPoseTextDataset): is_context and is_start. A learned type embedding marks
context frames and a learned boundary embedding marks each clip's first frame. The encoder runs
over [context ..., current]; the decoder cross-attends to the current clip's frames only, so
context can reach the translation only through the encoder -- the same one-variable-changed
discipline as phase 1 (identical decoder, identical memory length).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import PoseToTextModel
from .scope import pack_kept


class ContextPoseToTextModel(PoseToTextModel):
    def __init__(self, *args, d_in: int = 356, **kwargs):
        super().__init__(*args, d_in=d_in, **kwargs)
        self.d_in = d_in
        d_model = self.d_model
        self.context_embed = nn.Parameter(torch.zeros(d_model))
        self.boundary_embed = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.context_embed, std=0.02)
        nn.init.normal_(self.boundary_embed, std=0.02)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        feat = src[..., : self.d_in]
        is_context = src[..., self.d_in]
        is_start = src[..., self.d_in + 1]
        x = self.front_end(feat)
        x = (x + is_context.unsqueeze(-1).to(x.dtype) * self.context_embed
             + is_start.unsqueeze(-1).to(x.dtype) * self.boundary_embed)
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        memory = out[0] if isinstance(out, tuple) else out
        current = is_context < 0.5
        if src_key_padding_mask is not None:
            current = current & ~src_key_padding_mask
        return pack_kept(memory, current)
