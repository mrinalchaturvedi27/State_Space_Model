"""CPU checks for Stage 3.1 pretraining (stitching, masking, model, pretrain loop, --init-encoder
compatibility). Uses phase 2's mamba_ssm stand-in. Run: python scripts/test_pretrain.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from p3 import paths  # noqa: E402

sys.path.insert(0, os.path.join(paths.PHASE2_DIR, "scripts"))
import test_context  # noqa: E402,F401  (SentencePiece stand-in)
import test_phase2_arms as arms  # noqa: E402  (mamba_ssm stand-in)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from p3.masking import HAND_DIMS, span_masks  # noqa: E402
from p3.pretrain_model import MaskedPoseModel  # noqa: E402
from p3.stitch import EpochTokenSampler, PretrainDataset, VideoIndex, collate, parse_uid  # noqa: E402
from src.models import build_model  # noqa: E402

D = 356
# video -> list of (segment number, n_frames); index rows deliberately out of segment order
VIDEOS = {"vidA": [(3, 40), (1, 30), (2, 50), (7, 20)], "vidB": [(1, 60), (2, 45)], "x_solo": [(-1, 25)]}


def make_cache(root, split="train"):
    ds = os.path.join(root, "isign")
    os.makedirs(ds, exist_ok=True)
    rows, blocks, off = [], [], 0
    for vid, clips in VIDEOS.items():
        t0 = 0
        for seg, n in sorted(clips, key=lambda c: c[0]):  # frame values = position in the video
            uid = f"{vid}-{seg}" if seg >= 0 else "x_solo_e1"
            block = np.zeros((n, D), dtype=np.float16)
            block[:, 0] = np.arange(t0, t0 + n) % 2048
            block[:, 1] = hash(vid) % 7
            rows.append((uid, off, n))
            blocks.append(block)
            off += n
            t0 += n
    order = [3, 0, 2, 1, 5, 4, 6]  # shuffle index rows
    rows = [rows[i] for i in order if i < len(rows)]
    data = np.concatenate(blocks)
    mm = np.memmap(os.path.join(ds, f"{split}.memmap"), dtype="float16", mode="w+", shape=data.shape)
    mm[:] = data
    mm.flush()
    json.dump({"dtype": "float16", "shape": list(data.shape)}, open(os.path.join(ds, f"{split}.shape.json"), "w"))
    pd.DataFrame(rows, columns=["uid", "offset", "n_frames"]).assign(text="x").to_parquet(
        os.path.join(ds, f"{split}_index.parquet"))


def test_parse_uid_for_all_three_datasets():
    assert parse_uid("ed5590797795-8") == ("ed5590797795", 8)
    assert parse_uid("89DRLAodtpQ--26") == ("89DRLAodtpQ", 26)
    assert parse_uid("--7E2sU6zP4_10-5-rgb_front") == ("--7E2sU6zP4", 10)
    assert parse_uid("01April_2010_Thursday_heute-6694") == ("01April_2010_Thursday_heute", 6694)
    assert parse_uid("1Kus_VFkXlk_e1")[1] == -1


def test_stitching_order_coverage_and_flags():
    with tempfile.TemporaryDirectory() as root:
        make_cache(root)
        v = VideoIndex(root, "isign", "train")
        st = v.stats()
        assert st["videos"] == 3 and st["clips"] == 7 and st["unparsed_uids"] == 1
        total = sum(n for c in VIDEOS.values() for _, n in c)
        for mode, window in (("clip", 0), ("long", 64)):
            ds = PretrainDataset(v, mode, window or 4096, seed=1, augment=False)
            for epoch in (0, 1, 2):
                ds.set_epoch(epoch)
                assert ds.lengths().sum() == total, (mode, epoch)  # every frame once per epoch
                for i in range(len(ds)):
                    item = ds[i]
                    pos = item["feat"][:, 0]
                    assert np.all(np.diff(pos) == 1), (mode, item["video"])  # consecutive, in segment order
                    if mode == "long":
                        assert len(pos) <= window
            if mode == "clip":
                assert sorted(ds.lengths().tolist()) == sorted(n for c in VIDEOS.values() for _, n in c)
        # is_start marks every clip start inside a long window
        ds = PretrainDataset(v, "long", 4096, seed=0, augment=False)
        a = next(ds[i] for i in range(len(ds)) if ds[i]["video"] == "vidA")
        assert np.nonzero(a["is_start"])[0].tolist() == [0, 30, 80, 120]
        batches = list(EpochTokenSampler(ds, max_tokens=150, shuffle=False))
        assert sorted(i for b in batches for i in b) == list(range(len(ds)))
        b = collate([ds[i] for i in batches[0]])
        assert b["feat"].shape[2] == D and b["pad"].dtype == torch.bool


def test_span_masks():
    rng = np.random.default_rng(0)
    lengths = [400, 120, 37]
    fm, hm = span_masks(lengths, 400, ratio=0.3, span_min=8, span_max=32, hand_frac=0.3, rng=rng)
    for b, L in enumerate(lengths):
        cov = (fm[b, :L] | hm[b, :L]).float().mean().item()
        assert 0.25 <= cov <= 0.6, cov
        assert not (fm[b, L:] | hm[b, L:]).any(), "padding never masked"
    assert not (fm & hm).any()
    assert hm.any() and fm.any()


def test_masked_model_and_init_encoder_compatibility():
    for arm in ("mamba_padfix", "transformer"):
        cfg = {**arms.CFG, "arm": arm, "d_in": D, "max_src_len": 256}
        torch.manual_seed(0)
        m = MaskedPoseModel(cfg)
        feat = torch.randn(2, 50, D)
        pad = torch.zeros(2, 50, dtype=torch.bool)
        pad[1, 30:] = True
        fm, hm = span_masks([50, 30], 50, rng=np.random.default_rng(1))
        loss, n = m(feat, pad, fm, hm)
        loss.backward()
        assert torch.isfinite(loss) and n > 0
        # hand-only masking hides exactly the hand dims
        only_hand = torch.zeros_like(fm)
        only_hand[0, 5:15] = True
        loss_h, n_h = m(feat, pad, torch.zeros_like(fm), only_hand)
        assert n_h == 10 * (HAND_DIMS.stop - HAND_DIMS.start)
        # the saved weights load into the translation model: every key lands, values identical
        sd = m.encoder_state_dict()
        target = build_model(cfg, vocab_size=30)
        missing, unexpected = target.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(not k.startswith(("front_end.", "encoder.")) for k in missing), missing
        assert torch.equal(target.front_end.proj.weight, m.front_end.proj.weight.detach().cpu())


def test_pretrain_script_end_to_end_both_modes():
    import importlib
    with tempfile.TemporaryDirectory() as root:
        make_cache(root, "train")
        make_cache(root, "val")
        cfg = {**arms.CFG, "arm": "mamba_padfix", "d_in": D}
        m_yaml, d_yaml = os.path.join(root, "m.yaml"), os.path.join(root, "d.yaml")
        yaml.safe_dump(cfg, open(m_yaml, "w"))
        yaml.safe_dump({"dataset": "isign"}, open(d_yaml, "w"))
        sys.path.insert(0, HERE)
        pretrain = importlib.import_module("pretrain")
        for mode in ("clip", "long"):
            out = os.path.join(root, f"P_{mode}")
            sys.argv = ["pretrain", "--data", d_yaml, "--model", m_yaml, "--cache-dir", root, "--mode", mode,
                        "--window", "64", "--epochs", "2", "--max-tokens", "160", "--num-workers", "0", "--out", out]
            pretrain.main()
            ck = torch.load(os.path.join(out, "encoder_init.pt"), weights_only=False)
            assert any(k.startswith("encoder.") for k in ck["model"]) and "val_l1" in ck
            log = pd.read_csv(os.path.join(out, "log.csv"))
            assert len(log) == 2 and np.isfinite(log.val_l1).all()
        sys.argv = ["pretrain", "--data", d_yaml, "--model", m_yaml, "--cache-dir", root, "--stats-only", "--out", root]
        pretrain.main()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
