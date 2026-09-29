"""Per-token vector quantizers for the decoder memory (post-hoc, no training).

turbo      TurboQuant-style MSE quantizer: store the vector's norm (fp16), rotate the unit
           vector by a fixed random orthogonal matrix so every coordinate is ~N(0, 1/d), then
           quantize each coordinate with the Lloyd-Max codebook for a Gaussian at b bits.
turbo_prod TurboQuant-style inner-product quantizer: turbo at b-1 bits, plus a 1-bit QJL sketch
           of the residual (sign(S r) and ||r||) that makes <q, x_hat> unbiased for any q.
naive      baseline: per-token asymmetric min-max uniform quantization at b bits (min and max
           stored in fp16).

Reference to verify before citing: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (Google Research, 2025). This is a faithful-in-spirit
reimplementation for d = 512 (Gaussian approximation of the rotated coordinates), not their code.

bits_per_vector() gives the storage cost used for the memory-vs-quality plots.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=None)
def lloyd_max_gaussian(bits: int, iters: int = 300) -> np.ndarray:
    """MSE-optimal scalar codebook (sorted centroids) for N(0, 1) with 2**bits levels."""
    levels = 2 ** bits
    grid = np.linspace(-9.0, 9.0, 400_001)
    pdf = np.exp(-0.5 * grid ** 2)
    # start from equal-probability cell midpoints
    cdf = np.cumsum(pdf) / pdf.sum()
    centroids = np.interp((np.arange(levels) + 0.5) / levels, cdf, grid)
    for _ in range(iters):
        edges = np.concatenate([[-np.inf], (centroids[1:] + centroids[:-1]) / 2, [np.inf]])
        cell = np.searchsorted(edges, grid) - 1
        mass = np.bincount(cell, weights=pdf, minlength=levels)
        first = np.bincount(cell, weights=pdf * grid, minlength=levels)
        centroids = first / np.maximum(mass, 1e-300)
    return (centroids - centroids[::-1]) / 2  # N(0,1) is symmetric; remove grid asymmetry


def random_orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # Haar-distributed
    return q.float()


class VectorQuantizer:
    """Quantize-dequantize rows of a (..., d) tensor. Deterministic given (d, seed)."""

    def __init__(self, method: str, bits: int, d: int, seed: int = 0):
        if method not in ("turbo", "turbo_prod", "naive"):
            raise ValueError(f"unknown method: {method!r}")
        if method == "turbo_prod" and bits < 2:
            raise ValueError("turbo_prod needs >= 2 bits (b-1 for MSE + 1 for QJL)")
        self.method, self.bits, self.d = method, bits, d
        mse_bits = bits - 1 if method == "turbo_prod" else bits
        self.rotation = random_orthogonal(d, seed)
        self.codebook = torch.tensor(lloyd_max_gaussian(mse_bits), dtype=torch.float32) if method != "naive" else None
        if method == "turbo_prod":
            g = torch.Generator().manual_seed(seed + 1)
            self.sketch = torch.randn(d, d, generator=g)

    def bits_per_vector(self) -> int:
        if self.method == "naive":
            return self.d * self.bits + 32          # codes + fp16 min, max
        if self.method == "turbo":
            return self.d * self.bits + 16          # codes + fp16 norm
        return self.d * self.bits + 32              # (b-1)-bit codes + 1-bit sketch + fp16 norm, ||r||

    def _turbo_mse(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).half().float()
        unit = x / norm.clamp_min(1e-12)
        y = unit @ self.rotation.T.to(x.device) * math.sqrt(self.d)       # coords ~ N(0, 1)
        cb = self.codebook.to(x.device)
        edges = (cb[1:] + cb[:-1]) / 2
        yq = cb[torch.bucketize(y, edges)]
        return (yq / math.sqrt(self.d)) @ self.rotation.to(x.device) * norm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        if self.method == "naive":
            lo = x.amin(dim=-1, keepdim=True).half().float()
            hi = x.amax(dim=-1, keepdim=True).half().float()
            scale = ((hi - lo) / (2 ** self.bits - 1)).clamp_min(1e-12)
            out = torch.round((x - lo) / scale).clamp(0, 2 ** self.bits - 1) * scale + lo
        elif self.method == "turbo":
            out = self._turbo_mse(x)
        else:
            base = self._turbo_mse(x)
            r = x - base
            r_norm = r.norm(dim=-1, keepdim=True).half().float()
            S = self.sketch.to(x.device)
            signs = torch.sign(r @ S.T)
            signs[signs == 0] = 1
            out = base + math.sqrt(math.pi / 2) / self.d * r_norm * (signs @ S)
        return out.to(dtype)


def quantize_memory(memory: torch.Tensor, mem_pad: torch.Tensor | None,
                    q: VectorQuantizer | None) -> torch.Tensor:
    """Quantize the real memory tokens; padded slots are left as they are (masked anyway)."""
    if q is None:
        return memory
    out = q(memory)
    if mem_pad is not None:
        out = torch.where(mem_pad.unsqueeze(-1), memory, out)
    return out
