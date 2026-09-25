"""CPU checks for the phase-2 pooling rules. Does not import mamba_ssm."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.common import PoseToTextModel  # noqa: E402
from src.models.scope import (  # noqa: E402
    delta_pool,
    flip_valid,
    horizon_specs,
    softplus_inv,
    uniform_pool,
)


def test_horizon_math():
    specs = horizon_specs(16, short_heads=8, mid_heads=4)
    assert [s[0] for s in specs] == [0, 8, 12]
    assert [s[1] for s in specs] == [8, 12, 16]
    short_dt = 1.0 / (specs[0][3] * specs[0][2])
    # softplus(softplus_inv(dt)) == dt
    recovered = torch.nn.functional.softplus(softplus_inv(torch.tensor(short_dt)))
    assert torch.allclose(recovered, torch.tensor(short_dt), rtol=1e-5)


def test_flip_keeps_pads_at_the_right():
    hidden = torch.tensor([[[1.0], [2.0], [3.0], [9.0], [9.0]]])
    pad = torch.tensor([[False, False, False, True, True]])
    flipped = flip_valid(hidden, pad)
    assert torch.equal(flipped, torch.tensor([[[3.0], [2.0], [1.0], [0.0], [0.0]]]))
    restored = flip_valid(flipped, pad)
    assert torch.equal(restored, torch.tensor([[[1.0], [2.0], [3.0], [0.0], [0.0]]]))


def test_delta_pool_keeps_bucket_ends():
    memory = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    delta = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.0]])
    pad = torch.tensor([[False, False, False, False, True]])
    packed, packed_pad = delta_pool(memory, delta, pad, threshold=0.25)
    # cumulative Δ crosses 0.25 between frame 1 and frame 2; last real frame is 3.
    assert torch.equal(packed.view(-1), torch.tensor([1.0, 3.0]))
    assert torch.equal(packed_pad, torch.tensor([[False, False]]))


def test_delta_pool_packs_each_row():
    memory = torch.arange(8, dtype=torch.float32).view(2, 4, 1)
    delta = torch.tensor([
        [0.1, 0.1, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2],
    ])
    pad = torch.tensor([
        [False, False, True, True],
        [False, False, False, False],
    ])
    packed, packed_pad = delta_pool(memory, delta, pad, threshold=0.45)
    # Row 0 never crosses 0.45, so only its last real frame (value 1) is kept.
    # Row 1 crosses between frame 1 and 2 and also keeps its last frame.
    assert torch.equal(packed[0, 0], torch.tensor([1.0]))
    assert bool(packed_pad[0, 1])
    assert torch.equal(packed[1, :2].view(-1), torch.tensor([5.0, 7.0]))
    assert not bool(packed_pad[1].any())


def test_uniform_pool_stride():
    memory = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    pad = torch.tensor([[False, False, False, False, False, True]])
    packed, packed_pad = uniform_pool(memory, pad, stride=2)
    assert torch.equal(packed.view(-1), torch.tensor([0.0, 2.0, 4.0]))
    assert not bool(packed_pad.any())


def test_encode_uses_pooled_mask():
    class Half(nn.Module):
        def forward(self, x, src_key_padding_mask=None):
            return x[:, ::2], src_key_padding_mask[:, ::2]

    model = PoseToTextModel(
        Half(), d_model=32, vocab_size=20, d_in=8, n_dec_layers=1, n_heads=4,
        dim_feedforward=64, max_tgt_len=8,
    )
    src = torch.randn(2, 6, 8)
    src_pad = torch.tensor([
        [False, False, False, False, True, True],
        [False, False, False, False, False, False],
    ])
    tgt_in = torch.randint(1, 20, (2, 4))
    tgt_pad = torch.zeros(2, 4, dtype=torch.bool)
    memory, mem_pad = model.encode(src, src_pad)
    assert memory.shape[1] == 3
    assert mem_pad.shape == (2, 3)
    logits = model(src, tgt_in, src_pad, tgt_pad)
    assert logits.shape == (2, 4, 20)
    logits.sum().backward()


if __name__ == "__main__":
    test_horizon_math()
    test_flip_keeps_pads_at_the_right()
    test_delta_pool_keeps_bucket_ends()
    test_uniform_pool_stride()
    test_encode_uses_pooled_mask()
    print("ok")
