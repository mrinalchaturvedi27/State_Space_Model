"""CPU checks for every phase-2 gate arm, end to end through PoseToTextModel.

mamba_ssm's kernels need a GPU, so Mamba2 is replaced by a small causal stand-in with the
same attributes the phase-2 code touches (nheads, in_proj with dt as the last nheads rows,
A_log, dt_bias). The stand-in carries a bias into its running state, like Mamba2's conv
bias, so padding that reaches a scan changes the result -- which is what the padding test
checks. Run: python scripts/test_phase2_arms.py
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeMamba2(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, headdim=64, d_conv=4):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.nheads = d_inner // headdim
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.nheads, bias=False)
        self.state_bias = nn.Parameter(torch.randn(d_inner) * 0.5)
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, u):
        z = self.in_proj(u)[..., : self.d_inner] + self.state_bias
        steps = torch.arange(1, u.size(1) + 1, device=u.device, dtype=u.dtype).view(1, -1, 1)
        return self.out_proj(torch.cumsum(z, dim=1) / steps)


sys.modules["mamba_ssm"] = types.SimpleNamespace(Mamba2=FakeMamba2)

from src.models import PHASE2_ARMS, build_model  # noqa: E402

CFG = dict(d_model=64, d_in=12, enc_layers=2, dec_layers=1, n_heads=4, dim_feedforward=128,
           dropout=0.0, label_smoothing=0.1, d_state=16, expand=2, headdim=8, d_conv=4,
           max_tgt_len=16, pool_every_frames=4, short_heads=8, mid_heads=4)


def make(arm):
    torch.manual_seed(0)
    return build_model({**CFG, "arm": arm}, vocab_size=30).eval()


def batch(lengths, T=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    T = T or max(lengths)
    src = torch.randn(len(lengths), T, CFG["d_in"], generator=g)
    pad = torch.arange(T).unsqueeze(0) >= torch.tensor(lengths).unsqueeze(1)
    return src.masked_fill(pad.unsqueeze(-1), 0), pad


def test_every_arm_trains_and_matches_mamba_params():
    n_mamba = sum(p.numel() for p in make("mamba").parameters())
    for arm in PHASE2_ARMS:
        model = make(arm).train()
        assert sum(p.numel() for p in model.parameters()) == n_mamba, arm
        src, pad = batch([23, 9, 17])
        tgt = torch.randint(3, 30, (3, 6))
        loss = model.compute_loss(src, tgt[:, :-1], tgt[:, 1:], pad, torch.zeros(3, 5, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss), arm


def test_padding_does_not_change_a_clip():
    """A clip's memory must not depend on how much padding its batch adds."""
    src, pad = batch([13])
    for arm in PHASE2_ARMS:
        model = make(arm)
        alone, alone_pad = model.encode(src, pad)
        big = torch.zeros(2, 40, CFG["d_in"])
        big[0, :13] = src[0]
        big_pad = torch.arange(40).unsqueeze(0) >= torch.tensor([[13], [40]])
        padded, padded_pad = model.encode(big, big_pad)
        n = int((~alone_pad[0]).sum())
        assert int((~padded_pad[0]).sum()) == n, arm
        assert torch.allclose(alone[0, :n], padded[0, :n], atol=1e-5), arm


def test_phase1_encoder_does_leak_padding():
    """Documents the phase-1 bug the phase-2 arms fix (full-tensor flip)."""
    model = make("mamba")
    src, pad = batch([13])
    alone, _ = model.encode(src, pad)
    big = torch.zeros(2, 40, CFG["d_in"])
    big[0, :13] = src[0]
    big_pad = torch.arange(40).unsqueeze(0) >= torch.tensor([[13], [40]])
    padded, _ = model.encode(big, big_pad)
    assert not torch.allclose(alone[0, :13], padded[0, :13], atol=1e-5)


def test_uniform_and_matched_keep_the_same_count():
    src, pad = batch([37, 5, 64, 16])
    counts = {}
    for arm in ("mamba_uniform", "mamba_pool_matched"):
        _, mem_pad = make(arm).encode(src, pad)
        counts[arm] = (~mem_pad).sum(1)
    assert torch.equal(counts["mamba_uniform"], counts["mamba_pool_matched"])


def test_avg_arms_keep_the_same_count_and_train_delta():
    src, pad = batch([37, 5, 64, 16])
    counts = {}
    for arm in ("mamba_uniform_avg", "mamba_pool_avg"):
        _, mem_pad = make(arm).encode(src, pad)
        counts[arm] = (~mem_pad).sum(1)
    assert torch.equal(counts["mamba_uniform_avg"], counts["mamba_pool_avg"])
    # Only the Δ-weighted arm sends the decoder loss into the last layer's dt parameters.
    for arm, expect in (("mamba_pool_avg", True), ("mamba_pool_matched", False)):
        model = make(arm).train()
        tgt = torch.randint(3, 30, (4, 6))
        model.compute_loss(src, tgt[:, :-1], tgt[:, 1:], pad, torch.zeros(4, 5, dtype=torch.bool)).backward()
        fwd = model.encoder.layers[-1].fwd
        dt_rows = fwd.in_proj.weight.grad[-fwd.nheads:]
        # The stand-in's scan ignores dt, so any gradient there comes from pooling alone.
        assert bool(dt_rows.abs().sum() > 0) == expect, arm


def test_pool_stats():
    src, pad = batch([37, 5, 64, 16])
    real = int((~pad).sum())
    for arm in PHASE2_ARMS:
        model = make(arm)
        model.encoder.pop_pool_stats()
        _, mem_pad = model.encode(src, pad)
        stats = model.encoder.pop_pool_stats()
        kept = int((~mem_pad).sum())
        assert abs(stats["kept_frac"] - kept / real) < 1e-9, arm
        assert abs(stats["mem_tokens"] - kept / 4) < 1e-9, arm
        assert model.encoder.pop_pool_stats() == {}, "pop must reset"
    for arm in ("mamba_padfix", "mamba_banks"):
        model = make(arm)
        model.encode(src, pad)
        assert model.encoder.pop_pool_stats()["kept_frac"] == 1.0


def test_padfix_uses_default_init_and_banks_do_not():
    padfix, banks = make("mamba_padfix"), make("mamba_banks")
    a_default = padfix.encoder.layers[0].fwd.A_log
    a_banks = banks.encoder.layers[0].fwd.A_log
    assert torch.all(a_default == 0), "stand-in default init"
    assert not torch.all(a_banks == 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
