"""CPU checks for scripts/lp_tune.py. Run: python scripts/test_lp_tune.py"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_context as tc  # noqa: E402  (stubs sentencepiece + mamba_ssm, tiny cache)
import lp_tune  # noqa: E402
from src.evaluate import beam_search_decode  # noqa: E402
from src.models import build_model  # noqa: E402


def test_one_search_matches_separate_searches_for_every_lp():
    for arm in ("transformer", "mamba", "mamba_pool_avg"):
        torch.manual_seed(0)
        model = build_model({**tc.arms.CFG, "arm": arm, "d_in": 12, "max_src_len": 128}, vocab_size=30).eval()
        for seed in range(6):
            g = torch.Generator().manual_seed(seed)
            src = torch.randn(1, 9 + seed * 5, 12, generator=g)
            pad = torch.zeros(1, src.size(1), dtype=torch.bool)
            many = lp_tune.beam_search_all_lp(model, src, pad, 1, 2, 0, beam_size=5, max_new_tokens=12)
            for lp, seq in many.items():
                one = beam_search_decode(model, src, pad, 1, 2, 0, beam_size=5, length_penalty=lp, max_new_tokens=12)
                assert torch.equal(seq, one), (arm, seed, lp, seq.tolist(), one.tolist())


def test_end_to_end_selects_on_val_and_writes_files():
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        ds_dir = os.path.join(root, "isign")
        for f in ("val.memmap", "val.shape.json", "val_index.parquet"):
            shutil.copy(os.path.join(ds_dir, f), os.path.join(ds_dir, f.replace("val", "test")))
        cfg = {**tc.arms.CFG, "arm": "mamba_pool_avg", "d_in": tc.D}
        m_yaml, d_yaml = os.path.join(root, "m.yaml"), os.path.join(root, "d.yaml")
        yaml.safe_dump(cfg, open(m_yaml, "w"))
        yaml.safe_dump({"dataset": "isign"}, open(d_yaml, "w"))
        ck = os.path.join(root, "run", "checkpoints")
        os.makedirs(ck)
        torch.manual_seed(0)
        torch.save({"model": build_model(cfg, vocab_size=30).state_dict(), "epoch": 3}, os.path.join(ck, "best.pt"))
        sys.argv = ["lp_tune", "--data", d_yaml, "--model", m_yaml, "--checkpoint",
                    os.path.join(ck, "best.pt"), "--cache-dir", root]
        lp_tune.main()
        out = os.path.join(root, "run", "lp_tune")
        summary = list(csv.DictReader(open(os.path.join(out, "summary.csv"))))
        assert len(summary) == 2 * len(lp_tune.LP_GRID)
        chosen = list(csv.DictReader(open(os.path.join(out, "chosen.csv"))))
        assert [c["criterion"] for c in chosen] == ["val_BLEU4", "val_chrF2"]
        for c in chosen:
            val_best = max(float(r[c["criterion"].split("_")[1]]) for r in summary if r["split"] == "val")
            assert float(c[f"val_{c['criterion'].split('_')[1]}"]) == val_best, "chosen on val, not test"
            assert os.path.exists(os.path.join(out, f"test_lp{c['lp']}_beam5.csv"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
