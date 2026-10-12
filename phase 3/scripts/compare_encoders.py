"""Headline encoder comparison on test: mean +- std over seeds of corpus chrF2 / BLEU-4, content-word F1
(clipped multiset counts), number-token F1, negation recall / false-negation rate,
and each model minus a reference model with a source-video-grouped bootstrap interval (mean over common
seeds), overall and on short (<=128) / long (>256 frames) clips. CPU only; reads prediction CSVs.

  python "phase 3/scripts/compare_encoders.py" --ref Mamba \\
      --model Transformer=results/isign/transformer --model Mamba="phase 2/results/isign/mamba_padfix" ...
(each path is a model directory holding lr0.0003_s<seed>/ runs)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

from p3.content import content_tokens, has_negation, number_tokens  # noqa: E402
from p3.stitch import source_video  # noqa: E402

METRICS = {"chrF2": CHRF(), "BLEU4": BLEU()}


def clipped_f1(hyp, ref, tok) -> float:
    """Corpus F1 over clipped multiset matches (a word repeated twice in both counts twice)."""
    tp = pp = rr = 0
    for h, r in zip(hyp, ref):
        a, b = Counter(tok(h)), Counter(tok(r))
        tp, pp, rr = tp + sum((a & b).values()), pp + sum(a.values()), rr + sum(b.values())
    return 200 * tp / max(1, pp + rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=model_dir")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", default="42,13,1337")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = dict(m.split("=", 1) for m in args.model)
    seeds = [int(s) for s in args.seeds.split(",")]
    path = lambda n, s: os.path.join(R[n], f"lr0.0003_s{s}", "predictions", "test_best_beam5.csv")
    load = lambda p: pd.read_csv(p, keep_default_na=False).set_index("video_id")
    base = load(path(args.ref, next(s for s in seeds if os.path.exists(path(args.ref, s)))))
    uids, ref = base.index, base.ground_truth.astype(str).tolist()
    _, g = np.unique([source_video(str(u)) for u in uids], return_inverse=True)
    G = g.max() + 1
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G)[g] for _ in range(args.n_boot)]).astype(float)
    nf = base.n_frames.to_numpy()
    masks = {"all": np.ones(len(uids)), "<=128f": (nf <= 128).astype(float), ">256f": (nf > 256).astype(float)}

    def stats(n, s):
        if not os.path.exists(path(n, s)):
            return None
        hyp = load(path(n, s)).loc[uids].prediction.astype(str).tolist()
        extra = {"contentF1": clipped_f1(hyp, ref, lambda x: content_tokens(x, args.lang)),
                 "numberF1": clipped_f1(hyp, ref, lambda x: number_tokens(x, args.lang))}
        hn = np.array([has_negation(h, args.lang) for h in hyp])
        rn = np.array([has_negation(r, args.lang) for r in ref])
        extra["neg_recall"] = 100 * (hn & rn).sum() / max(1, rn.sum())        # negated refs whose hyp negates
        extra["false_neg"] = 100 * (hn & ~rn).sum() / max(1, (~rn).sum())   # unnegated refs whose hyp negates
        return {k: np.asarray(m._extract_corpus_statistics(hyp, [ref]), float) for k, m in METRICS.items()}, extra

    sc = lambda k, st, w: METRICS[k]._compute_score_from_stats(list(w @ st[k])).score
    D = {n: {s: stats(n, s) for s in seeds} for n in R}
    rows = []
    for n in R:
        ss = [s for s in seeds if D[n][s]]
        if not ss:
            continue
        row = {"model": n, "seeds": len(ss)}
        for k in METRICS:
            v = [sc(k, D[n][s][0], masks["all"]) for s in ss]
            row[k] = f"{np.mean(v):.2f}±{np.std(v):.2f}"
        for k in ("contentF1", "numberF1", "neg_recall", "false_neg"):
            row[k] = round(np.mean([D[n][s][1][k] for s in ss]), 2)
        cs = [s for s in ss if D[args.ref][s]]
        for k in METRICS:
            for mname, mk in masks.items():
                gap = lambda w: np.mean([sc(k, D[n][s][0], w) - sc(k, D[args.ref][s][0], w) for s in cs])
                pt = gap(mk)
                if mname == "all" or k == "chrF2":
                    bs = np.array([gap(w * mk) for w in W])
                    row[f"{k} {mname} vs {args.ref}"] = f"{pt:+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]"
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 320)
    print(df.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
