"""Selective translation: which outputs can be trusted? (Stage 3.0b)

Target output per clip: the MBR consensus over all systems (default) or one system's best.
Confidence signals (all computed without the reference):
  seed_agreement   mean pairwise chrF between the systems' best hypotheses
  consensus        mean chrF of the target output against every pooled n-best hypothesis
  avg_logprob      mean over systems of their best hypothesis' log-prob per token
  margin           first system: per-token log-prob gap between its best and second-best
  nbest_agreement  first system: mean chrF between its best and its other n-best entries
  src_frames       clip length in frames
  combined         mean of the z-scored signals above, each oriented on validation
Each signal's direction (higher = more trustworthy or less) is fixed on VALIDATION, then test is
evaluated: Spearman with sentence chrF2, AUROC for catching failures (no content word shared with
the reference), and coverage-quality curves (corpus chrF2 and failure rate of the kept outputs when
only the most confident 100/80/60/40/20% are shown). Oracle (true sentence chrF) and random bounds.

  python scripts/selective.py --runs ../results/isign/mamba/lr0.0003_s{13,42,1337}/nbest \\
      --lang en --out results/stage30/isign_mamba
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3.consensus import mbr_pick, mean_chrf_against  # noqa: E402
from p3.content import content  # noqa: E402
from p3.nbestio import load_nbest  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sacrebleu  # noqa: E402

COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)
SIGNALS = ["seed_agreement", "consensus", "avg_logprob", "margin", "nbest_agreement", "src_frames"]


def table(runs, split, target, lang, top_k):
    systems = [load_nbest(r, split, top_k=top_k) for r in runs]
    rows = []
    for u in [u for u in systems[0] if all(u in s for s in systems)]:
        tops = [s[u]["hyps"][0] for s in systems]
        pool = [h[0] for s in systems for h in s[u]["hyps"]]
        if target == "mbr":
            i, _ = mbr_pick(pool, pool)
            out = pool[i]
            others = pool[:i] + pool[i + 1:]
        else:
            out = tops[0][0]
            others = pool[1:]
        h0 = systems[0][u]["hyps"]
        ref = systems[0][u]["ref"]
        pair = [mean_chrf_against(tops[a][0], [tops[b][0]]) for a in range(len(tops)) for b in range(a + 1, len(tops))]
        rows.append({
            "uid": u, "output": out, "ref": ref,
            "seed_agreement": float(np.mean(pair)) if pair else 0.0,
            "consensus": mean_chrf_against(out, others),
            "avg_logprob": float(np.mean([t[1] / max(1, t[2]) for t in tops])),
            "margin": (h0[0][1] / max(1, h0[0][2]) - h0[1][1] / max(1, h0[1][2])) if len(h0) > 1 else 0.0,
            "nbest_agreement": mean_chrf_against(h0[0][0], [h[0] for h in h0[1:]]),
            "src_frames": systems[0][u]["n_frames"],
            "sent_chrF": sacrebleu.sentence_chrf(out, [ref]).score,
            "failure": len(content(ref, lang)) > 0 and len(content(out, lang) & content(ref, lang)) == 0,
        })
    return pd.DataFrame(rows)


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def auroc(score, positive):
    """P(score of a positive > score of a negative); positive = failure, score = -confidence."""
    r = pd.Series(score).rank().to_numpy()
    pos = np.asarray(positive, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def coverage(df, conf):
    order = np.argsort(-np.asarray(conf), kind="stable")
    rows = []
    for c in COVERAGES:
        k = order[: max(1, int(round(c * len(df))))]
        d = df.iloc[k]
        rows.append({"coverage": c, "chrF2": sacrebleu.corpus_chrf(d.output.tolist(), [d.ref.tolist()]).score,
                     "failure_rate": float(d.failure.mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--target", choices=("mbr", "single"), default="mbr")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    val = table(args.runs, "val", args.target, args.lang, args.top_k)
    test = table(args.runs, "test", args.target, args.lang, args.top_k)
    print(f"val {len(val)} / test {len(test)} clips; target={args.target}; test failure rate {test.failure.mean():.1%}")

    orient = {s: (1.0 if spearman(val[s], val.sent_chrF) >= 0 else -1.0) for s in SIGNALS}
    def z(df, s):
        mu, sd = val[s].mean(), val[s].std() or 1.0
        return orient[s] * (df[s] - mu) / sd
    test["combined"] = sum(z(test, s) for s in SIGNALS) / len(SIGNALS)
    val["combined"] = sum(z(val, s) for s in SIGNALS) / len(SIGNALS)
    orient["combined"] = 1.0
    test["oracle"], orient["oracle"] = test.sent_chrF, 1.0
    rng = np.random.default_rng(0)
    test["random"], orient["random"] = rng.random(len(test)), 1.0

    summary, curves = [], []
    for s in SIGNALS + ["combined", "oracle", "random"]:
        conf = orient[s] * test[s].to_numpy()
        cov = coverage(test, conf)
        curves += [{"signal": s, **r} for r in cov]
        summary.append({"signal": s, "direction_from_val": "+" if orient[s] > 0 else "-",
                        "spearman_val": round(spearman(orient[s] * val[s], val.sent_chrF), 3) if s in val else None,
                        "spearman_test": round(spearman(conf, test.sent_chrF), 3),
                        "auroc_failure_test": round(auroc(-conf, test.failure), 3),
                        **{f"chrF2@{int(r['coverage'] * 100)}%": round(r["chrF2"], 2) for r in cov},
                        **{f"fail@{int(r['coverage'] * 100)}%": round(100 * r["failure_rate"], 1) for r in cov}})
    os.makedirs(args.out, exist_ok=True)
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(args.out, f"selective_{args.target}_summary.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(args.out, f"selective_{args.target}_curves.csv"), index=False)
    test.drop(columns=["output", "ref"]).to_csv(os.path.join(args.out, f"selective_{args.target}_test_signals.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
