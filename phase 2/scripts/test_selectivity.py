"""CPU checks for the selectivity controls (src/models/selectivity.py) and the local-attention
Transformer. Uses test_phase2_arms' Mamba2 stand-in. Run: python scripts/test_selectivity.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
import yaml  # noqa: E402

from test_phase2_arms import CFG, batch  # noqa: E402  (installs the Mamba2 stand-in first)
from src.models import build_model  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "model")


def make(arm, **kw):
    torch.manual_seed(0)
    return build_model({**CFG, "arm": arm, **kw}, vocab_size=30).eval()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_nonselective_bc_dt_ignore_the_input():
    m = make("mamba_nonselective")
    mb = m.encoder.layers[0].fwd
    start = 2 * mb.d_inner
    a, b = mb.in_proj(torch.randn(2, 7, CFG["d_model"])), mb.in_proj(torch.randn(2, 7, CFG["d_model"]))
    assert torch.equal(a[..., start:], b[..., start:]), "B, C, dt must not depend on the input"
    assert not torch.allclose(a[..., :start], b[..., :start]), "z and x must still depend on it"
    assert a.shape[-1] == mb.in_proj.out_features
    # the constants are trained: they reach the loss through the scan (the stand-in reads x only, so
    # check the hook's output gradient directly)
    a[..., start:].sum().backward()
    assert mb.const_bc.grad is not None and mb.const_dt.grad is not None
    extra = sum(2 * (l.fwd.const_bc.numel() + l.fwd.const_dt.numel()) for l in m.encoder.layers)
    assert n_params(m) == n_params(make("mamba_padfix")) + extra


def test_window_blocks_information_across_edges_and_ignores_padding():
    m = make("mamba_window", window_frames=8)
    src, pad = batch([30], seed=3)
    base = m.encode(src, pad)[0][0]
    src2 = src.clone()
    src2[0, 2] += 3.0 * torch.randn(CFG["d_in"])
    moved = (m.encode(src2, pad)[0][0] - base).abs().sum(-1)
    assert moved[:8].sum() > 0 and moved[8:].abs().max() == 0, "a change in window 0 must stay in window 0"
    big, bpad = batch([30, 45], seed=4)
    big[0, :30] = src[0]
    both = m.encode(big, bpad)[0][0, :30]
    assert torch.allclose(base, both, atol=1e-5), "padding / batch-mates must not change a clip"
    assert n_params(m) == n_params(make("mamba_padfix"))


def test_local_attention_receptive_field_and_padding():
    full = yaml.safe_load(open(os.path.join(CFG_DIR, "transformer_local.yaml")))
    cfg = {**full, "d_model": 32, "d_in": 12, "dim_feedforward": 64, "n_heads": 4, "dec_layers": 1, "dropout": 0.0}
    torch.manual_seed(0)
    m = build_model(cfg, vocab_size=30).eval()
    enc = m.encoder
    assert enc.receptive_field() == 121
    x = torch.randn(1, 300, 32)
    x2 = x.clone()
    x2[0, 150] += 3.0 * torch.randn(32)
    d = (enc(x2) - enc(x)).abs().sum(-1)[0]
    span = (d > 0).nonzero().flatten()
    assert int(span.max() - span.min() + 1) == 121
    src = torch.randn(1, 20, 12)
    alone = m.encode(src, torch.zeros(1, 20, dtype=torch.bool))[0]
    big = torch.zeros(2, 50, 12)
    big[0, :20], big[1] = src[0], torch.randn(50, 12)
    pad = torch.arange(50).unsqueeze(0) >= torch.tensor([[20], [50]])
    out = m.encode(big, pad)[0]
    assert torch.isfinite(out).all(), "padded queries must not produce NaN"
    assert torch.allclose(alone[0], out[0, :20], atol=1e-5)
    base = build_model({**cfg, "arm": "transformer"}, vocab_size=30)
    assert n_params(m) == n_params(base)


def test_tcn_wide_reaches_the_whole_clip():
    cfg = yaml.safe_load(open(os.path.join(CFG_DIR, "tcn_wide.yaml")))
    assert cfg["tag"] == "tcn_wide", "own results directory, not tcn's"
    m = build_model({**cfg, "d_model": 32, "d_in": 12, "dim_feedforward": 64, "n_heads": 4, "dec_layers": 1},
                    vocab_size=30)
    assert m.encoder.receptive_field() >= 512


def test_training_step_all_arms():
    for arm in ("mamba_nonselective", "mamba_window"):
        m = make(arm, window_frames=8).train()
        src, pad = batch([13, 21], seed=5)
        tgt = torch.randint(4, 30, (2, 6))
        loss = m.compute_loss(src, tgt[:, :-1], tgt[:, 1:], pad, torch.zeros(2, 5, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss), arm


def test_window_modes_cover_every_frame_once_and_limit_context():
    from src.models.selectivity import WindowedEncoder
    for mode, off in (("block", 0), ("block", 3), ("overlap", 0)):
        w = WindowedEncoder(torch.nn.Identity(), window=8, mode=mode, offset=off)
        for T in (1, 7, 8, 9, 30):
            kept = [t for _, _, ks, ke in w.spans(T) for t in range(ks, ke)]
            assert kept == list(range(T)), (mode, off, T)
            assert all(e - s <= 8 and s <= ks and ke <= e for s, e, ks, ke in w.spans(T))
    W = 8
    m = make("mamba_window", window_frames=W, window_mode="overlap")
    src, pad = batch([40], seed=6)
    base = m.encode(src, pad)[0][0]
    src2 = src.clone()
    src2[0, 20] += 3.0 * torch.randn(CFG["d_in"])
    moved = (m.encode(src2, pad)[0][0] - base).abs().sum(-1)
    span = (moved > 0).nonzero().flatten()
    assert span.min() >= 20 - W and span.max() <= 20 + W, "an output may only see its own <= W-frame window"
    big, bpad = batch([40, 52], seed=7)
    big[0, :40] = src[0]
    assert torch.allclose(base, m.encode(big, bpad)[0][0, :40], atol=1e-5), "padding must not leak"
    # offset 0 block (the trained setup) is unchanged by the generic path
    blk = make("mamba_window", window_frames=W)
    gen = make("mamba_window", window_frames=W)
    gen.encoder.offset = W  # same grid, generic loop
    assert torch.allclose(blk.encode(src, pad)[0], gen.encode(src, pad)[0], atol=1e-5)


def test_s4d_padding_invariance_params_and_training():
    full = yaml.safe_load(open(os.path.join(CFG_DIR, "s4d.yaml")))
    base = yaml.safe_load(open(os.path.join(CFG_DIR, "transformer.yaml")))
    n = lambda c: n_params(build_model(c, vocab_size=4000))
    assert abs(n(full) - n(base)) / n(base) <= 0.02
    cfg = {**full, "d_model": 32, "d_in": 12, "dim_feedforward": 64, "n_heads": 4, "dec_layers": 1, "dropout": 0.0}
    torch.manual_seed(0)
    m = build_model(cfg, vocab_size=30).eval()
    src = torch.randn(1, 20, 12)
    alone = m.encode(src, torch.zeros(1, 20, dtype=torch.bool))[0]
    big = torch.zeros(2, 50, 12)
    big[0, :20], big[1] = src[0], torch.randn(50, 12)
    pad = torch.arange(50).unsqueeze(0) >= torch.tensor([[20], [50]])
    assert torch.allclose(alone[0], m.encode(big, pad)[0][0, :20], atol=1e-4)
    x2 = src.clone()
    x2[0, 0] += 1.0
    d = (m.encode(x2, torch.zeros(1, 20, dtype=torch.bool))[0] - alone).abs().sum(-1)[0]
    assert (d[-1] > 0), "an LTI SSM layer reaches the whole clip"
    m.train()
    tgt = torch.randint(4, 30, (2, 6))
    loss = m.compute_loss(big, tgt[:, :-1], tgt[:, 1:], pad, torch.zeros(2, 5, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and m.encoder.layers[0].fwd.C.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
