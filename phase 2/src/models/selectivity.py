"""Selectivity controls (diagnostic program, after the step-4 controls).

The TCN (dilated conv, receptive field 121 frames) matched Mamba overall on iSign, but Mamba kept a
lead on long clips (>256 frames: +0.44 chrF2, long-minus-short interaction +0.67). These arms ask
whether that residual comes from Mamba's selective (input-dependent) long-range memory:

  mamba_nonselective  the padfix encoder with B, C and Δ made input-independent: their in_proj
                      channels are replaced by learned constants, so every scan is a linear
                      time-invariant recurrence (S4D-like) with the same state size and layout.
                      The x and z paths stay input-dependent. The replaced in_proj rows remain in
                      the module unused, so loading/param bookkeeping matches mamba_padfix.
  mamba_window        the padfix encoder run on non-overlapping windows of window_frames frames
                      (default 128, 5.1 s, close to the TCN's 121): no information crosses a window
                      edge in any layer, so the receptive field is the window. Same parameters as
                      mamba_padfix.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def make_nonselective(mamba: nn.Module) -> nn.Module:
    """Replace the B, C and Δ channels of a Mamba2's in_proj output by learned constants.

    Mamba2's in_proj output is [z, x, B, C, dt] (no MLP channels by default): B and C are the
    2*ngroups*d_state channels after 2*d_inner, dt the last nheads. B and C still pass through the
    depthwise conv + SiLU; on a constant input that is constant (apart from the conv's zero-padded
    first frames), so Δ = softplus(const + dt_bias), B and C do not depend on the signing.
    """
    out_features = mamba.in_proj.out_features
    start = 2 * mamba.d_inner
    n_bc = out_features - mamba.nheads - start
    if hasattr(mamba, "ngroups") and hasattr(mamba, "d_state"):
        assert n_bc == 2 * mamba.ngroups * mamba.d_state, "unexpected Mamba2 in_proj layout (d_mlp > 0?)"
    mamba.register_parameter("const_bc", nn.Parameter(torch.randn(n_bc) * 0.5))
    mamba.register_parameter("const_dt", nn.Parameter(torch.zeros(mamba.nheads)))  # Δ = softplus(dt_bias) at init

    def hook(module, inputs, out):
        const = torch.cat([module.const_bc, module.const_dt]).to(out.dtype)
        return torch.cat([out[..., :start], const.expand(*out.shape[:-1], -1)], dim=-1)

    mamba.in_proj.register_forward_hook(lambda m, i, o: hook(mamba, i, o))
    return mamba


class WindowedEncoder(nn.Module):
    """Runs an encoder on non-overlapping windows of `window` frames and stitches the outputs back.
    Windows that are entirely padding are dropped; the inner encoder handles partial windows with
    its own padding mask."""

    def __init__(self, inner: nn.Module, window: int = 128):
        super().__init__()
        self.inner = inner
        self.window = window

    def pop_pool_stats(self) -> dict:
        self.inner.reset_pool_stats()  # per-window counts would read as pooling; there is none
        return {}

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        B, T, D = x.shape
        W = self.window
        n = -(-T // W)
        pad = src_key_padding_mask if src_key_padding_mask is not None else torch.zeros(B, T, dtype=torch.bool, device=x.device)
        xp = torch.zeros(B, n * W, D, dtype=x.dtype, device=x.device)
        xp[:, :T] = x
        pp = torch.ones(B, n * W, dtype=torch.bool, device=x.device)
        pp[:, :T] = pad
        xw, pw = xp.view(B * n, W, D), pp.view(B * n, W)
        keep = ~pw.all(1)
        out = self.inner(xw[keep], pw[keep])
        mem = out[0] if isinstance(out, tuple) else out
        full = torch.zeros(B * n, W, mem.size(-1), dtype=mem.dtype, device=mem.device)
        full[keep] = mem
        return full.view(B, n * W, -1)[:, :T], src_key_padding_mask
