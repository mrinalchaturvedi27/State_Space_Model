"""CPU checks for quant/turboquant.py. Run: python quant/test_turboquant.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from turboquant import VectorQuantizer, lloyd_max_gaussian, quantize_memory, random_orthogonal  # noqa: E402

# Lloyd-Max for N(0,1): positive centroids and MSE distortion (Max 1960; standard tables).
KNOWN = {1: ([0.7979], 0.3634), 2: ([0.4528, 1.510], 0.1175), 3: ([0.2451, 0.7560, 1.344, 2.152], 0.03454)}


def test_codebook_matches_lloyd_max_tables():
    for bits, (pos, _) in KNOWN.items():
        cb = lloyd_max_gaussian(bits)
        assert np.allclose(cb[len(cb) // 2:], pos, atol=2e-3), (bits, cb)
        assert np.allclose(cb, -cb[::-1], atol=1e-6), "symmetric"


def test_rotation_is_orthogonal():
    R = random_orthogonal(64, seed=3)
    assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-5)


def test_turbo_distortion_matches_theory():
    torch.manual_seed(0)
    x = torch.randn(2000, 512) * 3 + 1
    for bits, (_, dist) in KNOWN.items():
        xh = VectorQuantizer("turbo", bits, 512)(x)
        rel = float(((x - xh) ** 2).sum(-1).div((x ** 2).sum(-1)).mean())
        assert abs(rel - dist) / dist < 0.08, (bits, rel, dist)


def test_turbo_beats_naive_with_outlier_channels():
    # Real activations often have a few large channels; min-max wastes its range on them,
    # the rotation spreads them out.
    torch.manual_seed(1)
    x = torch.randn(2000, 512)
    x[:, :4] *= 25
    for bits in (2, 3, 4):
        err = {}
        for m in ("turbo", "naive"):
            xh = VectorQuantizer(m, bits, 512)(x)
            err[m] = float(((x - xh) ** 2).sum(-1).div((x ** 2).sum(-1)).mean())
        assert err["turbo"] < err["naive"], (bits, err)


def test_turbo_prod_inner_products_are_unbiased():
    torch.manual_seed(2)
    x = torch.randn(1, 128)
    q = torch.randn(1, 128)
    true = float((x * q).sum())
    est_prod, est_mse = [], []
    for seed in range(300):
        est_prod.append(float((VectorQuantizer("turbo_prod", 2, 128, seed=seed)(x) * q).sum()))
        est_mse.append(float((VectorQuantizer("turbo", 1, 128, seed=seed)(x) * q).sum()))
    se = np.std(est_prod) / np.sqrt(len(est_prod))
    assert abs(np.mean(est_prod) - true) < 4 * se, (np.mean(est_prod), true, se)
    # The MSE quantizer at the same code budget shrinks inner products toward 0.
    assert abs(np.mean(est_mse)) < abs(true)


def test_bits_and_padding():
    assert VectorQuantizer("turbo", 3, 512).bits_per_vector() == 512 * 3 + 16
    assert VectorQuantizer("naive", 4, 512).bits_per_vector() == 512 * 4 + 32
    assert VectorQuantizer("turbo_prod", 3, 512).bits_per_vector() == 512 * 3 + 32
    mem = torch.randn(2, 5, 512)
    pad = torch.tensor([[False, False, True, True, True], [False] * 5])
    out = quantize_memory(mem, pad, VectorQuantizer("turbo", 2, 512))
    assert torch.equal(out[pad], mem[pad]) and not torch.equal(out[~pad], mem[~pad])
    assert torch.equal(quantize_memory(mem, pad, None), mem)


def test_bf16_input_roundtrip_dtype():
    x = torch.randn(3, 7, 512, dtype=torch.bfloat16)
    assert VectorQuantizer("turbo", 4, 512)(x).dtype == torch.bfloat16


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
