"""CPU checks for C1: ContextPoseTextDataset on a tiny real cache, and ContextPoseToTextModel.

SentencePiece and mamba_ssm are replaced by stand-ins (see test_phase2_arms.py for the Mamba2
one). Run: python scripts/test_context.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd
import torch


class FakeSP:
    def __init__(self, model_file=None):
        pass

    def encode(self, text, out_type=int):
        return [5 + (len(w) % 20) for w in text.split()]

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)

    def bos_id(self): return 1
    def eos_id(self): return 2
    def pad_id(self): return 0
    def vocab_size(self): return 30


sys.modules.setdefault("sentencepiece", types.SimpleNamespace(SentencePieceProcessor=FakeSP))
sys.path.insert(0, os.path.dirname(__file__))
import test_phase2_arms as arms  # noqa: E402  (installs the Mamba2 stand-in)

from src.data import ContextPoseTextDataset, make_collate, make_dataset  # noqa: E402
from src.models import build_model  # noqa: E402

D = 356
# uid -> (n_frames, fill value). vidA/vidB have numbered segments; odd_d has none.
CLIPS = {"vidA-3": (7, 3.0), "vidA-1": (5, 1.0), "vidB-1": (4, 11.0), "vidA-2": (6, 2.0),
         "vidB-2": (3, 12.0), "odd_d": (4, 99.0)}


def make_cache(root: str) -> None:
    ds_dir = os.path.join(root, "isign")
    os.makedirs(ds_dir)
    rows, offset, blocks = [], 0, []
    for uid, (n, v) in CLIPS.items():
        rows.append({"uid": uid, "offset": offset, "n_frames": n, "text": "a bb ccc"})
        blocks.append(np.full((n, D), v, dtype=np.float16))
        offset += n
    data = np.concatenate(blocks)
    mm = np.memmap(os.path.join(ds_dir, "val.memmap"), dtype="float16", mode="w+", shape=data.shape)
    mm[:] = data
    mm.flush()
    with open(os.path.join(ds_dir, "val.shape.json"), "w") as f:
        json.dump({"dtype": "float16", "shape": list(data.shape)}, f)
    pd.DataFrame(rows).to_parquet(os.path.join(ds_dir, "val_index.parquet"))


def ds(root, **kw):
    return ContextPoseTextDataset(root, "isign", "val", "spm.model", 16, 512, augment=False, **kw)


def pos(d, uid):
    return list(d.index["uid"]).index(uid)


def test_prev_context_order_flags_and_length():
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        d = ds(root, context_clips=2)
        item = d[pos(d, "vidA-3")]
        feat = item["feat"]
        assert feat.shape == (5 + 6 + 7, D + 2)
        assert np.all(feat[:5, 0] == 1.0) and np.all(feat[5:11, 0] == 2.0) and np.all(feat[11:, 0] == 3.0)
        assert np.all(feat[:11, D] == 1) and np.all(feat[11:, D] == 0), "is_context"
        assert list(np.nonzero(feat[:, D + 1])[0]) == [0, 5, 11], "is_start on each clip's first frame"
        assert item["n_cur"] == 7


def test_no_context_for_first_segment_and_unnumbered_uid():
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        d = ds(root, context_clips=2)
        for uid, n in (("vidA-1", 5), ("odd_d", 4)):
            item = d[pos(d, uid)]
            assert item["feat"].shape[0] == n and item["feat"][:, D].sum() == 0
            assert item["feat"][0, D + 1] == 1


def test_context_budget_cuts_oldest_from_its_start():
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        d = ds(root, context_clips=2, max_context_frames=8)
        feat = d[pos(d, "vidA-3")]["feat"]
        # nearest clip (vidA-2, 6 frames) whole, then the last 2 frames of vidA-1
        assert feat.shape[0] == 2 + 6 + 7
        assert np.all(feat[:2, 0] == 1.0) and np.all(feat[2:8, 0] == 2.0)


def test_random_context_is_another_video_and_fixed_at_eval():
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        d = ds(root, context_clips=2, context_mode="random")
        i = pos(d, "vidA-3")
        a, b = d[i]["feat"], d[i]["feat"]
        assert np.array_equal(a, b)
        ctx_values = set(np.unique(a[a[:, D] == 1, 0]).tolist())
        assert ctx_values and ctx_values <= {11.0, 12.0}, ctx_values


def test_make_dataset_switch_and_collate_reports_current_length():
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        plain = make_dataset(root, "isign", "val", "spm.model", 16, 512, model_cfg={"arm": "mamba"})
        assert not isinstance(plain, ContextPoseTextDataset)
        d = make_dataset(root, "isign", "val", "spm.model", 16, 512, model_cfg={"context_clips": 2})
        batch = make_collate(0)([d[pos(d, "vidA-3")], d[pos(d, "vidB-1")]])
        assert batch.src.shape[-1] == D + 2
        assert batch.n_frames.tolist() == [7, 4]


def test_model_decoder_sees_current_clip_and_context_reaches_it():
    cfg = {**arms.CFG, "arm": "mamba_ctx", "d_in": D}
    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=30).eval()
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        with_ctx = ds(root, context_clips=2)[pos(ds(root, context_clips=2), "vidA-3")]
        no_ctx = ds(root, context_clips=0)[pos(ds(root, context_clips=0), "vidA-3")]
        b1 = make_collate(0)([with_ctx])
        b0 = make_collate(0)([no_ctx])
        m1, p1 = model.encode(b1.src, b1.src_key_padding_mask)
        m0, p0 = model.encode(b0.src, b0.src_key_padding_mask)
        assert m1.shape[1] == m0.shape[1] == 7 and not p1.any() and not p0.any()
        assert not torch.allclose(m1, m0), "context must change the current clip's encoding"
        # padding from a longer batch-mate must not change it
        b2 = make_collate(0)([with_ctx, ds(root, context_clips=0)[pos(ds(root, context_clips=0), "odd_d")]])
        big = torch.zeros(2, 30, D + 2)
        big[:, : b2.src.shape[1]] = b2.src
        pad = torch.ones(2, 30, dtype=torch.bool)
        pad[:, : b2.src.shape[1]] = b2.src_key_padding_mask
        m2, _ = model.encode(big, pad)
        assert torch.allclose(m2[0, :7], m1[0], atol=1e-5)
        tgt = torch.randint(3, 30, (1, 5))
        loss = model.train().compute_loss(b1.src, tgt[:, :-1], tgt[:, 1:], b1.src_key_padding_mask,
                                          torch.zeros(1, 4, dtype=torch.bool))
        loss.backward()
        assert model.context_embed.grad is not None and model.boundary_embed.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
