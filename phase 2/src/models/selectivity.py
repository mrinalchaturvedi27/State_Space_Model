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
                      mamba_padfix. window_mode "overlap" separates context from boundaries
                      (2026-10-12 review): windows of W frames at stride W/2, each frame's output taken
                      from the window in whose central half it lies, so every frame still sees at most
                      W frames but no frame (except the clip ends) sits next to a window edge.
                      window_offset shifts the block grid (the first block has `offset` frames).
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
    """Runs an encoder on windows of `window` frames and stitches the outputs back.
    mode "block": non-overlapping windows starting at `offset` (0: the grid 0, W, 2W, ...).
    mode "overlap": windows [kH - H/2, kH + 3H/2) with H = W/2, keeping outputs for [kH, (k+1)H).
    Windows that are entirely padding are skipped; the inner encoder handles partial windows with
    its own padding mask (padding is always trailing, so a window slice stays trailing-padded)."""

    def __init__(self, inner: nn.Module, window: int = 128, mode: str = "block", offset: int = 0):
        super().__init__()
        if mode not in ("block", "overlap"):
            raise ValueError(f"unknown window mode {mode!r}")
        self.inner = inner
        self.window = window
        self.mode = mode
        self.offset = offset

    def spans(self, T: int):
        """[(start, end, keep_start, keep_end)] covering 0..T, each frame kept exactly once."""
        W = self.window
        if self.mode == "block":
            cuts = sorted({0, T} | set(range(self.offset % W or W, T, W)))
            return [(a, b, a, b) for a, b in zip(cuts[:-1], cuts[1:])]
        H = W // 2
        out = []
        for k in range(-(-T // H)):
            ks, ke = k * H, min(T, (k + 1) * H)
            out.append((max(0, ks - H // 2), min(T, ks + W - H // 2), ks, ke))
        return out

    def pop_pool_stats(self) -> dict:
        self.inner.reset_pool_stats()  # per-window counts would read as pooling; there is none
        return {}

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        if self.mode == "block" and self.offset == 0:
            return self._block_fast(x, src_key_padding_mask)
        B, T, _ = x.shape
        pad = src_key_padding_mask if src_key_padding_mask is not None else torch.zeros(B, T, dtype=torch.bool, device=x.device)
        full = None
        for s, e, ks, ke in self.spans(T):
            ps = pad[:, s:e]
            rows = ~ps.all(1)
            if not rows.any():
                continue
            out = self.inner(x[rows, s:e], ps[rows])
            mem = out[0] if isinstance(out, tuple) else out
            if full is None:
                full = torch.zeros(B, T, mem.size(-1), dtype=mem.dtype, device=mem.device)
            full[rows, ks:ke] = mem[:, ks - s:ke - s]
        return full, src_key_padding_mask

    def _block_fast(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None):
        """The grid 0, W, 2W, ...: all windows of the batch in one call (the trained configuration)."""
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
