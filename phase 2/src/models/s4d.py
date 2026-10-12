"""S4D encoder: an independently parameterised linear time-invariant SSM baseline (2026-10-12 review).

The non-selective Mamba arm keeps Mamba-2's layout, so with ngroups = 1 and a scalar A per head its
long-range filter is one decaying exponential per head. S4D (Gu et al. 2022, arXiv:2206.11893) is the
standard diagonal LTI SSM: per channel, N/2 complex modes with their own decay AND oscillation,
discretised by zero-order hold, applied as a convolution kernel computed in closed form:
    K_h(l) = 2 Re( sum_n C_hn (exp(dt_h A_hn) - 1) / A_hn * exp(dt_h A_hn l) ),  A = -exp(a) + i b
S4D-Lin init: A_n = -1/2 + i pi n; log dt ~ U[log 1e-3, log 1e-1]; C ~ CN(0, 1).
Each layer, pre-LN residual: bidirectional S4D (separate forward / backward kernels, each causal over
the clip's own frames via per-clip reversal) + D skip -> GELU -> Linear(d, 2d) -> GLU, then a
pre-LN FFN. No input-dependent gating inside the SSM. Padded frames are zeroed before every layer, and
both convolutions are causal over a trailing-padded sequence, so a clip's encoding ignores padding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scope import flip_valid


class S4DKernel(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        n = d_state // 2
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n).float().repeat(d_model, 1))
        self.C = nn.Parameter(torch.view_as_real(torch.randn(d_model, n, dtype=torch.cfloat) * 0.5 ** 0.5))

    def forward(self, L: int) -> torch.Tensor:
        """(d_model, L) real kernel."""
        dt = self.log_dt.exp().unsqueeze(-1)                                    # (H, 1)
        A = -self.log_A_real.exp() + 1j * self.A_imag                           # (H, N)
        dtA = dt * A
        C = torch.view_as_complex(self.C) * (torch.exp(dtA) - 1.0) / A
        V = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=A.device))     # (H, N, L)
        return 2 * torch.einsum("hn,hnl->hl", C, V).real

    def retention_frames(self) -> torch.Tensor:
        """Per mode, frames for the kernel envelope to decay by 1/e: 1 / (dt * |Re A|)."""
        return 1.0 / (self.log_dt.exp().unsqueeze(-1) * self.log_A_real.exp())


def causal_conv(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """u (B, T, H), k (H, T) -> (B, T, H), y_t = sum_{s<=t} k_{t-s} u_s, via FFT in fp32."""
    T = u.size(1)
    uf = torch.fft.rfft(u.float().transpose(1, 2), n=2 * T)
    kf = torch.fft.rfft(k.float(), n=2 * T)
    return torch.fft.irfft(uf * kf, n=2 * T)[..., :T].transpose(1, 2).to(u.dtype)


class S4DLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd, self.bwd = S4DKernel(d_model, d_state), S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.glu = nn.Linear(d_model, 2 * d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        T = x.size(1)
        h = self.norm(x)
        if pad is not None:
            h = h.masked_fill(pad.unsqueeze(-1), 0)
        y = causal_conv(h, self.fwd(T))
        y = y + flip_valid(causal_conv(flip_valid(h, pad), self.bwd(T)), pad)
        y = F.gelu(y + h * self.D)
        x = x + self.drop(F.glu(self.glu(y), dim=-1))
        x = x + self.ffn(x)
        return x if pad is None else x.masked_fill(pad.unsqueeze(-1), 0)


class S4DEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=7, d_state=64, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([S4DLayer(d_model, d_state, dim_feedforward, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        pad = src_key_padding_mask
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0)
        for layer in self.layers:
            x = layer(x, pad)
        x = self.norm(x)
        return x if pad is None else x.masked_fill(pad.unsqueeze(-1), 0)
