"""Horizon banks and Δ-synchronous pooling for the phase-2 Mamba encoder.

Sign clips mix three timescales: a manual sign (~0.5 s), a phrase, and a
non-manual or discourse span. Mamba-2 already has a per-head step size Δ.
This module biases each head onto one of those timescales, then keeps a
decoder memory token only when the short-horizon bank has integrated about
one sign of time. A uniform stride of the same width is the control: it
shortens the memory the same way and ignores Δ.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def softplus_inv(dt: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus, the transform Mamba-2 uses to store dt_bias."""
    return dt + torch.log(-torch.expm1(-dt))


def horizon_specs(nheads: int, short_heads: int = 8, mid_heads: int = 4):
    """(start, end, A, tau_frames) for the short, mid, and long banks.

    Retention is τ = 1 / (Δ · A), so the stored step size is Δ = 1 / (τ · A).
    """
    if short_heads < 1 or mid_heads < 1 or short_heads + mid_heads >= nheads:
        raise ValueError(
            f"need at least one head in each bank, got nheads={nheads} "
            f"short={short_heads} mid={mid_heads}"
        )
    long_heads = nheads - short_heads - mid_heads
    bands = (
        (short_heads, 8.0, 14.0),   # ~0.5 s at 25 fps
        (mid_heads, 4.0, 75.0),     # a short phrase
        (long_heads, 2.0, 500.0),   # clause / non-manual scope
    )
    specs = []
    cursor = 0
    for count, a_value, tau in bands:
        specs.append((cursor, cursor + count, a_value, tau))
        cursor += count
    return specs


def apply_horizon_banks(mamba, short_heads: int = 8, mid_heads: int = 4,
                        dt_weight_scale: float = 0.05) -> float:
    """Re-init one Mamba2 module so each head starts at its bank's timescale.

    The dt rows of in_proj are scaled down so the bias, not a random projection,
    sets Δ at the start of training. Returns the short bank's Δ, which the
    pooling threshold is measured in.
    """
    specs = horizon_specs(mamba.nheads, short_heads, mid_heads)
    short_dt = 1.0 / (specs[0][3] * specs[0][2])
    with torch.no_grad():
        mamba.in_proj.weight[-mamba.nheads :].mul_(dt_weight_scale)
        if mamba.in_proj.bias is not None:
            mamba.in_proj.bias[-mamba.nheads :].zero_()
        for start, end, a_value, tau in specs:
            dt = 1.0 / (tau * a_value)
            mamba.A_log.data[start:end].fill_(math.log(a_value))
            mamba.dt_bias.data[start:end].fill_(float(softplus_inv(torch.tensor(dt))))
    return short_dt


def short_head_delta(mamba, hidden: torch.Tensor, short_heads: int) -> torch.Tensor:
    """Per-frame mean Δ of the short-horizon heads. Shape (B, T), float32.

    Mamba2's fused scan consumes the same quantity: the last `nheads` channels
    of in_proj are the raw dt, and Δ = softplus(dt + dt_bias).
    """
    nheads = mamba.nheads
    weight = mamba.in_proj.weight[-nheads:]
    bias = None if mamba.in_proj.bias is None else mamba.in_proj.bias[-nheads:]
    dt_raw = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    dt = F.softplus(dt_raw + mamba.dt_bias.float())
    return dt[..., :short_heads].mean(dim=-1)


def flip_valid(hidden: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Reverse each sequence over its real frames. Pads stay at the right.

    A full-tensor flip puts the padded tail in front of the backward scan, so a
    clip's encoding depends on how much padding the batch gave it.
    """
    if pad_mask is None:
        return hidden.flip(1)
    lengths = (~pad_mask).sum(dim=1)
    T = hidden.size(1)
    index = torch.arange(T, device=hidden.device).unsqueeze(0)
    src = (lengths.unsqueeze(1) - 1 - index).clamp(min=0, max=T - 1)
    reversed_h = torch.gather(hidden, 1, src.unsqueeze(-1).expand_as(hidden))
    return reversed_h.masked_fill(pad_mask.unsqueeze(-1), 0)


def pack_kept(memory: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack kept frames to the left. keep is a bool mask of shape (B, T)."""
    B, _, D = memory.shape
    counts = keep.sum(dim=1)
    width = int(counts.max().item()) if B else 1
    width = max(width, 1)
    packed = memory.new_zeros(B, width, D)
    packed_pad = torch.ones(B, width, dtype=torch.bool, device=memory.device)
    if not bool(keep.any()):
        return packed, packed_pad
    position = torch.cumsum(keep.long(), dim=1) - 1
    batch_idx, time_idx = keep.nonzero(as_tuple=True)
    packed[batch_idx, position[batch_idx, time_idx]] = memory[batch_idx, time_idx]
    packed_pad[batch_idx, position[batch_idx, time_idx]] = False
    return packed, packed_pad


def delta_pool(memory: torch.Tensor, delta: torch.Tensor, pad_mask: torch.Tensor | None,
               threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the last frame of each bucket of cumulative short-bank Δ.

    Also keeps the last real frame, so a clip whose Δ never crosses the
    threshold still produces a token. Pads contribute nothing to the sum.
    """
    B, T, _ = memory.shape
    if pad_mask is None:
        pad_mask = torch.zeros(B, T, dtype=torch.bool, device=memory.device)
    valid = ~pad_mask
    step = delta.float().clamp_min(0).masked_fill(~valid, 0)
    if threshold <= 0:
        keep = valid
    else:
        buckets = torch.floor(torch.cumsum(step, dim=1) / threshold).long()
        buckets = buckets.masked_fill(~valid, -1)
        nxt = torch.cat(
            [buckets[:, 1:], torch.full((B, 1), -2, dtype=buckets.dtype, device=buckets.device)],
            dim=1,
        )
        lengths = valid.sum(dim=1)
        last = (lengths - 1).clamp_min(0)
        is_last = torch.arange(T, device=memory.device).unsqueeze(0) == last.unsqueeze(1)
        keep = valid & ((buckets != nxt) | is_last)
    return pack_kept(memory, keep)


def uniform_pool(memory: torch.Tensor, pad_mask: torch.Tensor | None,
                 stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep every `stride`-th real frame plus the last one. The no-Δ control."""
    B, T, _ = memory.shape
    if pad_mask is None:
        pad_mask = torch.zeros(B, T, dtype=torch.bool, device=memory.device)
    valid = ~pad_mask
    lengths = valid.sum(dim=1)
    index = torch.arange(T, device=memory.device).unsqueeze(0)
    last = (lengths - 1).clamp_min(0)
    is_last = index == last.unsqueeze(1)
    # Real frames are a prefix (collate right-pads), so the absolute index is
    # the position within the clip.
    keep = valid & ((index % max(1, stride) == 0) | is_last)
    return pack_kept(memory, keep)
