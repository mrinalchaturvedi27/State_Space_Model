"""CPU checks for the step-4 control encoders (src/models/controls.py) and the lexical grounding head
(bow_weight in PoseToTextModel). Run: python scripts/test_controls.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
import yaml  # noqa: E402

from src.models import build_model  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "model")
CONTROLS = ("transformer_convstem", "transformer_relpos", "tcn")


def cfg(name):
    return yaml.safe_load(open(os.path.join(CFG_DIR, f"{name}.yaml")))


def small(name):
    c = {**cfg(name), "d_model": 64, "d_in": 12, "dim_feedforward": 128, "n_heads": 4, "dec_layers": 1, "dropout": 0.0}
    torch.manual_seed(0)
    return build_model(c, vocab_size=30).eval()


def test_params_match_transformer_within_2pct():
    base = sum(p.numel() for p in build_model(cfg("transformer"), vocab_size=4000).parameters())
    for name in CONTROLS:
        n = sum(p.numel() for p in build_model(cfg(name), vocab_size=4000).parameters())
        assert abs(n - base) / base <= 0.02, (name, n, base)


def test_padding_invariance_and_training_step():
    for name in CONTROLS:
        m = small(name)
        x = torch.randn(1, 13, 12)
        alone = m.encode(x, torch.zeros(1, 13, dtype=torch.bool))[0]
        big = torch.zeros(2, 40, 12)
        big[0, :13] = x[0]
        big[1] = torch.randn(40, 12)
        pad = torch.arange(40).unsqueeze(0) >= torch.tensor([[13], [40]])
        both = m.encode(big, pad)[0]
        assert torch.allclose(alone[0], both[0, :13], atol=1e-5), name
        m.train()
        tgt = torch.randint(4, 30, (2, 6))
        loss = m.compute_loss(big, tgt[:, :-1], tgt[:, 1:], pad, torch.zeros(2, 5, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss), name


def test_receptive_fields_are_what_we_report():
    stem = small("transformer_convstem").encoder
    x = torch.randn(1, 40, 64)
    y0 = x.clone()
    for blk in stem.stem:
        y0 = blk(y0)
    x2 = x.clone()
    x2[0, 20] += 5.0 * torch.randn(64)  # not a constant shift: LayerNorm would erase that
    y1 = x2.clone()
    for blk in stem.stem:
        y1 = blk(y1)
    changed = ((y1 - y0).abs().sum(-1)[0] > 1e-6).nonzero().flatten().tolist()
    assert changed == list(range(16, 25)) and len(changed) == stem.stem_receptive_field
    tcn = small("tcn").encoder
    T = 200
    x = torch.randn(1, T, 64)
    x2 = x.clone()
    x2[0, 100] += 5.0 * torch.randn(64)
    d = (tcn(x2) - tcn(x)).abs().sum(-1)[0]
    span = (d > 0).nonzero().flatten()  # outside the field the change is exactly zero
    width = int(span.max() - span.min() + 1)
    assert width == tcn.receptive_field(), (width, tcn.receptive_field())


def test_relpos_has_no_absolute_positions_but_is_order_aware():
    m = small("transformer_relpos").encoder
    x = torch.randn(1, 20, 64)
    pad = torch.zeros(1, 20, dtype=torch.bool)
    a = m(x, pad)
    b = m(x.flip(1), pad).flip(1)
    assert torch.allclose(a, b, atol=1e-5), "symmetric |i-j| bias: reversing the input reverses the output"
    perm = torch.randperm(20)
    assert not torch.allclose(m(x[:, perm], pad), a[:, perm], atol=1e-4), "but not permutation-invariant"


def test_bow_head_off_by_default_and_trains_when_on():
    base = {**cfg("transformer"), "d_model": 64, "d_in": 12, "dim_feedforward": 128, "n_heads": 4, "dec_layers": 1}
    off = build_model(base, vocab_size=30)
    assert not any(k.startswith("bow_head") for k in off.state_dict())
    on = build_model({**base, "bow_weight": 0.3}, vocab_size=30)
    on.load_state_dict(off.state_dict(), strict=False)
    on.set_bow_ignore([0, 1, 2, 3, 5])
    x, pad = torch.randn(2, 9, 12), torch.zeros(2, 9, dtype=torch.bool)
    tgt = torch.tensor([[1, 5, 6, 7, 2, 0], [1, 8, 9, 2, 0, 0]])
    tp = tgt[:, 1:] == 0
    on.train()
    l_train = on.compute_loss(x, tgt[:, :-1], tgt[:, 1:], pad, tp)
    l_train.backward()
    assert on.bow_head.weight.grad is not None and on.bow_head.weight.grad.abs().sum() > 0
    assert on.bow_head.weight.grad[5].abs().sum() == 0, "ignored ids get no gradient"
    on.eval()
    l_eval = on.compute_loss(x, tgt[:, :-1], tgt[:, 1:], pad, tp)
    assert l_train.item() > l_eval.item() - 1e-6, "the auxiliary term is training-only"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
