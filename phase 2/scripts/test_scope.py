"""CPU checks for the phase-2 pooling rules. Does not import mamba_ssm."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.common import PoseToTextModel  # noqa: E402
from src.models.scope import (  # noqa: E402
    delta_avg_pool,
    delta_pool,
    flip_valid,
    horizon_specs,
    matched_delta_pool,
    softplus_inv,
    uniform_avg_pool,
    uniform_keep_count,
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


def test_uniform_keep_count_matches_uniform_pool():
    for length in range(1, 70):
        for stride in (1, 4, 16):
            memory = torch.zeros(1, 80, 1)
            pad = torch.arange(80).unsqueeze(0) >= length
            _, packed_pad = uniform_pool(memory, pad, stride)
            expected = int(uniform_keep_count(torch.tensor([length]), stride))
            assert int((~packed_pad).sum()) == expected, (length, stride)


def test_matched_pool_same_count_as_uniform():
    torch.manual_seed(0)
    lengths = torch.tensor([1, 2, 15, 16, 17, 33, 200, 512])
    T, stride = 512, 16
    pad = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)
    memory = torch.arange(T, dtype=torch.float32).view(1, T, 1).repeat(len(lengths), 1, 1)
    # Spiky Δ, including frames that alone exceed a quantile, to exercise the dedup path.
    delta = torch.rand(len(lengths), T) ** 8 * 5
    packed, packed_pad = matched_delta_pool(memory, delta, pad, stride)
    _, uni_pad = uniform_pool(memory, pad, stride)
    assert torch.equal((~packed_pad).sum(1), (~uni_pad).sum(1))
    for row, length in enumerate(lengths.tolist()):
        kept = packed[row, ~packed_pad[row], 0]
        assert torch.all(kept[1:] > kept[:-1]), "frames must be distinct and in order"
        assert int(kept[-1]) == length - 1, "the last real frame is always kept"
        assert int(kept.max()) < length, "never a padded frame"


def test_matched_pool_follows_delta():
    # All of Δ's mass in the second half: every kept frame except the forced last one
    # should land there, whereas uniform spreads them evenly.
    T, stride = 64, 8
    memory = torch.arange(T, dtype=torch.float32).view(1, T, 1)
    delta = torch.cat([torch.full((1, T // 2), 1e-4), torch.ones(1, T // 2)], dim=1)
    packed, packed_pad = matched_delta_pool(memory, delta, None, stride)
    kept = packed[0, ~packed_pad[0], 0]
    assert int((kept >= T // 2).sum()) >= len(kept) - 1


def test_uniform_avg_pool_means():
    memory = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    pad = torch.arange(10).unsqueeze(0) >= 7
    pooled, pooled_pad = uniform_avg_pool(memory, pad, stride=3)
    # segments [0,1,2] [3,4,5] [6]; padded frames 7-9 are ignored
    assert torch.allclose(pooled[0, :, 0], torch.tensor([1.0, 4.0, 6.0]))
    assert not bool(pooled_pad.any())


def test_delta_avg_pool_counts_weights_and_grad():
    torch.manual_seed(0)
    lengths = torch.tensor([1, 5, 16, 17, 40])
    T, stride = 40, 4
    pad = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)
    memory = torch.randn(len(lengths), T, 3)
    delta = (torch.rand(len(lengths), T) ** 6 * 3 + 1e-3).requires_grad_()
    pooled, pooled_pad = delta_avg_pool(memory, delta, pad, stride)
    _, uni_pad = uniform_avg_pool(memory, pad, stride)
    assert torch.equal((~pooled_pad).sum(1), (~uni_pad).sum(1)), "same segment count as uniform_avg"
    pooled[~pooled_pad].sum().backward()
    assert delta.grad is not None and delta.grad[~pad].abs().sum() > 0, "Δ must receive gradient"
    assert torch.all(delta.grad[pad] == 0), "pads must not"
    # Each token is a convex combination of its clip's real frames.
    for row, length in enumerate(lengths.tolist()):
        real = memory[row, :length]
        toks = pooled[row, ~pooled_pad[row]]
        assert torch.all(toks <= real.max(0).values + 1e-5)
        assert torch.all(toks >= real.min(0).values - 1e-5)


def test_delta_avg_pool_constant_delta_is_uniform_avg():
    # Equal only when the length is a multiple of the stride: otherwise uniform_avg ends on a
    # short segment while Δ quantiles spread the remainder. 0.25 keeps the cumsum exact.
    memory = torch.randn(2, 32, 4)
    pad = torch.arange(32).unsqueeze(0) >= torch.tensor([[32], [16]])
    delta = torch.full((2, 32), 0.25)
    a, a_pad = delta_avg_pool(memory, delta, pad, 8)
    b, b_pad = uniform_avg_pool(memory, pad, 8)
    assert torch.equal(a_pad, b_pad)
    assert torch.allclose(a, b, atol=1e-5)


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
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
