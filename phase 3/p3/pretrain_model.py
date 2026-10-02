"""Masked-frame pretraining wrapper around the translation model's own front end + encoder.

The front end and encoder are taken from phase 2's build_model(...) for the same model config, so
their parameter names and shapes are exactly those of PoseToTextModel; encoder_state_dict() returns
them with the 'front_end.' / 'encoder.' prefixes that phase 2's train.py --init-encoder loads.
The mask token and reconstruction head are pretraining-only and are dropped.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from p3 import paths  # noqa: F401
from p3.masking import HAND_DIMS
from src.models import build_model


class MaskedPoseModel(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        base = build_model(model_cfg, vocab_size=8)
        self.front_end, self.encoder = base.front_end, base.encoder
        d, d_in = model_cfg["d_model"], model_cfg.get("d_in", 356)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_in))

    def forward(self, feat, pad, frame_mask, hand_mask):
        """L1 reconstruction loss on the hidden values of real (non-pad) frames."""
        hand_sel = torch.zeros_like(feat, dtype=torch.bool)
        hand_sel[..., HAND_DIMS] = hand_mask.unsqueeze(-1)
        target_sel = (frame_mask.unsqueeze(-1).expand_as(feat) | hand_sel) & ~pad.unsqueeze(-1)
        x_in = feat.masked_fill(frame_mask.unsqueeze(-1), 0.0).masked_fill(hand_sel, 0.0)
        x = self.front_end(x_in)
        x = torch.where(frame_mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        out = self.encoder(x, src_key_padding_mask=pad)
        enc = out[0] if isinstance(out, tuple) else out
        pred = self.head(enc).float()
        n = target_sel.sum()
        loss = (pred - feat.float()).abs()[target_sel].sum() / n.clamp_min(1)
        return loss, int(n)

    def encoder_state_dict(self) -> dict:
        sd = {f"front_end.{k}": v.detach().cpu() for k, v in self.front_end.state_dict().items()}
        sd.update({f"encoder.{k}": v.detach().cpu() for k, v in self.encoder.state_dict().items()})
        return sd
