"""CPU checks for the diagnostic scripts (stand-ins for mamba_ssm and SentencePiece).
The padding check must still be run on the REAL checkpoints on the GPU (scripts/diag_padding.py).
Run: python scripts/test_diagnostics.py"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from p3 import paths  # noqa: E402

sys.path.insert(0, os.path.join(paths.PHASE2_DIR, "scripts"))
import test_context as tc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from p3.diag import density, resample, reverse, shuffle_blocks, shuffle_frames, sinusoid, teacher_forced, time_positions, word_scores  # noqa: E402
from src.data import make_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.common import SinusoidalPositionalEncoding  # noqa: E402


def test_transforms_keep_extent():
    f = np.arange(40, dtype=np.float32).reshape(20, 2)
    for k in (0.5, 1.0, 2.0):
        g = density(f, k)
        assert g.shape[0] == round(20 * k)
        assert np.allclose(g[0], f[0]) and np.allclose(g[-1], f[-1]), "whole time span kept"
    assert np.allclose(resample(f, 20), f)
    assert np.allclose(reverse(f)[0], f[-1])
    rng = np.random.default_rng(0)
    for g in (shuffle_frames(f, rng), shuffle_blocks(f, 4, rng)):
        assert sorted(g[:, 0].tolist()) == sorted(f[:, 0].tolist())
    b = shuffle_blocks(f, 4, np.random.default_rng(1))
    assert all(np.all(np.diff(b[i:i + 4, 0]) == 2) for i in range(0, 20, 4)), "blocks stay intact"


def test_sinusoid_matches_phase2_table_and_time_pe_at_rate_1_is_identity():
    pe = SinusoidalPositionalEncoding(64, 100)
    assert torch.allclose(sinusoid(torch.arange(50), 64), pe.pe[:, :50], atol=1e-5)
    cfg = {**tc.arms.CFG, "arm": "transformer", "d_in": 12, "max_src_len": 128}
    torch.manual_seed(0)
    m = build_model(cfg, vocab_size=30).eval()
    src, pad = torch.randn(2, 17, 12), torch.zeros(2, 17, dtype=torch.bool)
    a = m.encode(src, pad)[0]
    with time_positions(m, 1.0):
        b = m.encode(src, pad)[0]
    with time_positions(m, 2.0):
        c = m.encode(src, pad)[0]
    assert torch.allclose(a, b, atol=1e-5) and not torch.allclose(a, c, atol=1e-3)
    assert torch.allclose(m.encode(src, pad)[0], a), "positions restored afterwards"


def test_word_scores_sum_tokens_and_teacher_forcing_is_batch_invariant():
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        ds = tc.ContextPoseTextDataset(root, "isign", "val", "spm.model", 16, 512, augment=False, context_clips=0)
        cfg = {**tc.arms.CFG, "arm": "transformer", "d_in": tc.D, "max_src_len": 128}
        torch.manual_seed(0)
        m = build_model(cfg, vocab_size=30).eval()
        from src.data import PoseTextDataset
        plain = PoseTextDataset(root, "isign", "val", "spm.model", 16, 512, augment=False)
        col = make_collate(plain.pad_id)
        one = teacher_forced(m, col([plain[0]]), "cpu")[0]
        two = teacher_forced(m, col([plain[0], plain[3]]), "cpu")[0]
        assert one[0] == two[0] and np.allclose(one[1], two[1], atol=1e-5)
        ws = word_scores(plain.sp, *one)
        assert abs(sum(w[1] for w in ws) - sum(lp for t, lp in zip(*one) if t not in (0, 1, 2, 3))) < 1e-6


def _write_cfgs(root, arm):
    cfg = {**tc.arms.CFG, "arm": arm, "d_in": tc.D, "max_src_len": 128}
    m, d, x = (os.path.join(root, f) for f in ("m.yaml", "d.yaml", "x.yaml"))
    yaml.safe_dump(cfg, open(m, "w"))
    yaml.safe_dump({"dataset": "isign", "lang": "en"}, open(d, "w"))
    yaml.safe_dump({"dataset": "how2sign"}, open(x, "w"))
    ck = os.path.join(root, arm, "best.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.manual_seed(0)
    torch.save({"model": build_model(cfg, vocab_size=30).state_dict()}, ck)
    return m, d, x, ck


def test_padding_check_flags_the_phase1_leak_only():
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        mod = importlib.import_module("diag_padding")
        res = {}
        for arm in ("mamba", "mamba_padfix", "transformer"):
            m, d, _, ck = _write_cfgs(root, arm)
            out = os.path.join(root, f"pad_{arm}.csv")
            sys.argv = ["diag_padding", "--data", d, "--model", m, "--checkpoint", ck, "--cache-dir", root, "--n", "5", "--out", out]
            mod.main()
            res[arm] = pd.read_csv(out).max_abs_memory_diff.max()
        assert res["mamba"] > 1e-3, res
        assert res["mamba_padfix"] < 1e-4 and res["transformer"] < 1e-4, res


def test_video_and_density_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        shutil.copytree(os.path.join(root, "isign"), os.path.join(root, "how2sign"))
        for arm in ("mamba_padfix", "transformer"):
            m, d, x, ck = _write_cfgs(root, arm)
            vid = importlib.import_module("diag_video")
            out = os.path.join(root, "video", arm)
            sys.argv = ["diag_video", "--data", d, "--model", m, "--checkpoint", ck, "--cross-data", x,
                        "--cache-dir", root, "--n", "6", "--n-decode", "3", "--out", out]
            vid.main()
            s = pd.read_csv(os.path.join(out, "summary.csv"))
            assert list(s.condition) == ["correct", "wrong_length", "wrong_topic", "cross_domain", "zero", "static"]
            assert len(pd.read_csv(os.path.join(out, "decodes.csv"))) == 3 * 4
            den = importlib.import_module("diag_density")
            out = os.path.join(root, "density", arm)
            sys.argv = ["diag_density", "--data", d, "--model", m, "--checkpoint", ck, "--cache-dir", root, "--out", out]
            den.main()
            s = pd.read_csv(os.path.join(out, "summary.csv")).set_index("condition")
            assert ("d2_timePE" in s.index) == (arm == "transformer")
            assert s.loc["d2", "mean_frames"] > s.loc["d1", "mean_frames"] > s.loc["d0.5", "mean_frames"]
        # both models used the same fixed samples
        assert os.path.exists(os.path.join(root, "video", "sample_video_isign.csv"))
        assert os.path.exists(os.path.join(root, "density", "sample_density_isign_0-256.csv"))


def test_window_partners_and_retention_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        shutil.copytree(os.path.join(root, "isign"), os.path.join(root, "how2sign"))
        # windowed checkpoint (window 8): boundary diagnostic over every condition family
        cfg = {**tc.arms.CFG, "arm": "mamba_window", "window_frames": 8, "d_in": tc.D, "max_src_len": 128}
        m, d, x, ck = _write_cfgs(root, "mamba_window")
        yaml.safe_dump(cfg, open(m, "w"))
        torch.manual_seed(0)
        torch.save({"model": build_model(cfg, vocab_size=30).state_dict()}, ck)
        den = importlib.import_module("diag_density")
        sys.argv = ["diag_density", "--data", d, "--model", m, "--checkpoint", ck, "--cache-dir", root,
                    "--out", os.path.join(root, "density", "w")]
        den.main()
        smp = os.path.join(root, "density", "sample_density_isign_0-256.csv")
        win = importlib.import_module("diag_window")
        out = os.path.join(root, "window")
        sys.argv = ["diag_window", "--data", d, "--model", m, "--checkpoint", ck, "--cache-dir", root,
                    "--sample-short", smp, "--sample-long", smp, "--conditions", "full block8 overlap8 block8_off3",
                    "--out", out]
        win.main()
        s = pd.read_csv(os.path.join(out, "summary.csv")).set_index("condition")
        assert list(s.index) == ["full", "block8", "overlap8", "block8_off3"]
        assert s.chrF2_long_minus_short.abs().max() < 1e-9, "same sample for both ranges here"
        # partner check on a real E1 sample
        vid = importlib.import_module("diag_video")
        sys.argv = ["diag_video", "--data", d, "--model", m, "--checkpoint", ck, "--cross-data", x,
                    "--cache-dir", root, "--n", "6", "--n-decode", "2", "--out", os.path.join(root, "video", "w")]
        vid.main()
        chk = importlib.import_module("check_video_partners")
        sys.argv = ["check_video_partners", "--cache-dir", root, "--dataset", "isign", "--video-dir", os.path.join(root, "video")]
        chk.main()
        assert os.path.exists(os.path.join(root, "video", "w", "summary_partner_checked.csv"))
        # retention times: non-selective Mamba and S4D checkpoints
        ret = importlib.import_module("inspect_retention")
        for arm in ("mamba_nonselective", "s4d"):
            c = {**tc.arms.CFG, "arm": arm, "d_in": tc.D, "enc_layers": 2}
            p = os.path.join(root, f"{arm}.pt")
            torch.save({"model": build_model(c, vocab_size=30).state_dict()}, p)
            o = os.path.join(root, f"{arm}.csv")
            sys.argv = ["inspect_retention", "--checkpoint", p, "--out", o]
            ret.main()
            r = pd.read_csv(o)
            assert (r.tau_frames > 0).all() and set(r.direction) == {"fwd", "bwd"} and not r.selective.any(), arm


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
