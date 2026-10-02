"""CPU checks for Stage 3.0 (n-best decoding, consensus/MBR, selective translation).
Uses phase 2's test stand-ins for mamba_ssm and SentencePiece. Run: python scripts/test_stage30.py"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from p3 import paths  # noqa: E402

sys.path.insert(0, os.path.join(paths.PHASE2_DIR, "scripts"))
import test_context as tc  # noqa: E402  (stubs sentencepiece + mamba_ssm, tiny cache)

import pandas as pd  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from p3.chrf import chrf, ngram_stats  # noqa: E402
from p3.consensus import mbr_pick  # noqa: E402
from p3.nbest import beam_search_nbest, rank  # noqa: E402
from src.evaluate import beam_search_decode  # noqa: E402
from src.models import build_model  # noqa: E402


def test_nbest_rerank_matches_beam_search_for_every_lp():
    for arm in ("transformer", "mamba", "mamba_pool_avg"):
        torch.manual_seed(0)
        model = build_model({**tc.arms.CFG, "arm": arm, "d_in": 12, "max_src_len": 128}, vocab_size=30).eval()
        for seed in range(5):
            g = torch.Generator().manual_seed(seed)
            src = torch.randn(1, 8 + 4 * seed, 12, generator=g)
            pad = torch.zeros(1, src.size(1), dtype=torch.bool)
            fin = beam_search_nbest(model, src, pad, 1, 2, 0, beam_size=5, max_new_tokens=12)
            for lp in (0.6, 1.0, 1.3, 2.0):
                one = beam_search_decode(model, src, pad, 1, 2, 0, beam_size=5, length_penalty=lp, max_new_tokens=12)
                assert torch.equal(rank(fin, lp)[0][0], one), (arm, seed, lp)


def test_fast_chrf_tracks_sacrebleu():
    random.seed(0)
    words = "the a sign hand video story boy girl went home school read write blue red".split()
    pairs = [(" ".join(random.choices(words, k=random.randint(2, 10))), " ".join(random.choices(words, k=random.randint(2, 10)))) for _ in range(300)]
    ours = [chrf(ngram_stats(a), ngram_stats(b)) for a, b in pairs]
    ref = [sacrebleu.sentence_chrf(a, [b]).score for a, b in pairs]
    corr = pd.Series(ours).corr(pd.Series(ref))
    assert corr > 0.99, corr
    assert abs(chrf(ngram_stats("Trace the dotted lines."), ngram_stats("Trace the dotted lines.")) - 100) < 1e-9


def test_mbr_picks_the_consensus():
    pool = ["the boy went to school", "the boy went to the school", "a boy goes to school",
            "completely unrelated output here", "the boy went to school today"]
    i, u = mbr_pick(pool, pool)
    assert pool[i] in ("the boy went to school", "the boy went to the school", "the boy went to school today"), pool[i]
    assert pool[mbr_pick(pool, pool)[0]] != "completely unrelated output here"


def test_selective_metrics_on_known_signal():
    sys.path.insert(0, HERE)
    import selective
    good = ["trace the dotted lines", "colour the parts of the plant", "the boy went home"]
    bad = ["completely unrelated words", "nothing in common here", "another wrong answer"]
    rows = [{"output": g, "ref": g, "failure": False} for g in good] + \
           [{"output": b, "ref": g, "failure": True} for b, g in zip(bad, good)]
    df = pd.DataFrame(rows)
    conf = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]          # perfect: every good one above every bad one
    assert selective.auroc([-c for c in conf], df.failure) == 1.0
    assert selective.auroc(conf, df.failure) == 0.0
    cov = selective.coverage(df, conf)
    assert cov[0]["chrF2"] < cov[-1]["chrF2"] and cov[-1]["failure_rate"] == 0.0
    assert abs(selective.spearman([1, 2, 3], [10, 20, 30]) - 1) < 1e-9


def test_end_to_end_nbest_mbr_selective():
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        ds = os.path.join(root, "isign")
        for f in ("val.memmap", "val.shape.json", "val_index.parquet"):
            shutil.copy(os.path.join(ds, f), os.path.join(ds, f.replace("val", "test")))
        cfg = {**tc.arms.CFG, "arm": "mamba", "d_in": tc.D}
        m_yaml, d_yaml = os.path.join(root, "m.yaml"), os.path.join(root, "d.yaml")
        yaml.safe_dump(cfg, open(m_yaml, "w"))
        yaml.safe_dump({"dataset": "isign"}, open(d_yaml, "w"))
        runs = []
        for s in (13, 42, 1337):
            ck = os.path.join(root, f"run_s{s}", "checkpoints")
            os.makedirs(ck)
            torch.manual_seed(s)
            torch.save({"model": build_model(cfg, vocab_size=30).state_dict()}, os.path.join(ck, "best.pt"))
            sys.argv = ["nbest_decode", "--data", d_yaml, "--model", m_yaml, "--checkpoint",
                        os.path.join(ck, "best.pt"), "--cache-dir", root]
            import importlib
            nb = importlib.import_module("nbest_decode")
            nb.main()
            runs.append(os.path.join(root, f"run_s{s}", "nbest"))
            d = pd.read_csv(os.path.join(runs[-1], "test.csv"), keep_default_na=False)
            assert d.uid.nunique() == len(tc.CLIPS) and (d.groupby("uid").cand.min() == 0).all()
        out = os.path.join(root, "stage30")
        import mbr
        sys.argv = ["mbr", "--runs", *runs, "--names", "s13", "s42", "s1337", "--split", "test", "--out", out]
        mbr.main()
        summ = pd.read_csv(os.path.join(out, "mbr_test_summary.csv"))
        assert {"single:s13", "mbr_self", "mbr_top1", "mbr_pool", "oracle"} <= set(summ.system)
        import selective
        sys.argv = ["selective", "--runs", *runs, "--out", out]
        selective.main()
        sel = pd.read_csv(os.path.join(out, "selective_mbr_summary.csv"))
        assert {"seed_agreement", "consensus", "combined", "oracle", "random"} <= set(sel.signal)
        assert os.path.exists(os.path.join(out, "selective_mbr_curves.csv"))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
