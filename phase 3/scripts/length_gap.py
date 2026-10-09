"""E3 (RESEARCH_PROGRAM_2026-10-08.md): where does encoder B beat encoder A, with uncertainty that
respects clips from the same source video being correlated. CPU only; reads test prediction CSVs.

Every model is given as one or more seed run directories; seeds are paired across models by their
_s<seed> suffix. Per stratum the statistic is the mean over seeds of corpus chrF2 (and BLEU-4) of
B minus A, recomputed from summed sentence statistics inside each bootstrap resample. Resampling
draws source VIDEOS with replacement (all their clips come along), the review's recommended unit.
Strata:
  frames      <=128 / 129-256 / >256 input frames (after the 512-frame cap)
  ref_words   reference length terciles: frame count and sentence length are correlated, so
  frames x ref_words                     the cross-tab shows whether the gap follows frames at
                                         fixed sentence length
  cap         clips at the 512-frame cap vs the rest
Also the interaction: gap on >256-frame clips minus gap on <=128-frame clips.

  python "phase 3/scripts/length_gap.py" --a transformer=results/isign/transformer/lr0.0003_s{13,42,1337} \\
      --b mamba_padfix="phase 2/results/isign/mamba_padfix/lr0.0003_s{13,42,1337}" --out <dir>
(the shell expands the braces; quote paths with spaces)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

from p3.stitch import parse_uid  # noqa: E402

METRICS = {"chrF2": CHRF(), "BLEU4": BLEU()}


def parse_model(spec: str, extra: list[str]):
    name, first = spec.split("=", 1)
    runs = {}
    for d in [first] + extra:
        d = d[len(name) + 1:] if d.startswith(name + "=") else d  # brace expansion repeats the prefix
        m = re.search(r"_s(\d+)/?$", d)
        assert m, f"cannot read the seed from {d}"
        runs[int(m.group(1))] = d
    return name, runs


def load(run_dir: str) -> pd.DataFrame:
    p = os.path.join(run_dir, "predictions", "test_best_beam5.csv")
    return pd.read_csv(p, keep_default_na=False).set_index("video_id")


def stats(df: pd.DataFrame) -> dict[str, np.ndarray]:
    hyp, ref = df.prediction.astype(str).tolist(), df.ground_truth.astype(str).tolist()
    return {k: np.asarray(m._extract_corpus_statistics(hyp, [ref]), dtype=np.float64) for k, m in METRICS.items()}


def score(metric: str, s: np.ndarray, w: np.ndarray) -> float:
    return METRICS[metric]._compute_score_from_stats(list(w @ s)).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="name=run_dir_s<seed> [more seed dirs]")
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    na, ra = parse_model(args.a[0], args.a[1:])
    nb, rb = parse_model(args.b[0], args.b[1:])
    seeds = sorted(set(ra) & set(rb))
    assert seeds, "no common seeds"
    base = load(ra[seeds[0]])
    uids = base.index
    S = {}
    for s in seeds:
        A, B = load(ra[s]).loc[uids], load(rb[s]).loc[uids]
        assert (A.ground_truth.values == B.ground_truth.values).all()
        S[s] = (stats(A), stats(B))
    nf = base.n_frames.to_numpy()
    rw = base.ref_words.to_numpy()
    t1, t2 = np.quantile(rw, [1 / 3, 2 / 3])
    rw_b = np.where(rw <= t1, f"ref<={t1:.0f}w", np.where(rw <= t2, f"ref{t1 + 1:.0f}-{t2:.0f}w", f"ref>{t2:.0f}w"))
    fr_b = np.where(nf <= 128, "<=128f", np.where(nf <= 256, "129-256f", ">256f"))
    strata = {"all": np.ones(len(uids), bool)}
    for v in ("<=128f", "129-256f", ">256f"):
        strata[f"frames {v}"] = fr_b == v
    for v in sorted(set(rw_b), key=lambda x: float(re.findall(r"\d+", x)[0]) + (0.5 if ">" in x else 0)):
        strata[f"ref_words {v}"] = rw_b == v
        for f in ("<=128f", "129-256f", ">256f"):
            strata[f"{v} x {f}"] = (rw_b == v) & (fr_b == f)
    cap = nf >= nf.max()
    strata["cap-hit (512f)"], strata["not cap-hit"] = cap, ~cap

    vid = np.array([parse_uid(str(u))[0] for u in uids])
    groups, g_idx = np.unique(vid, return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, G, size=(args.n_boot, G))

    def gap(metric, w):  # mean over seeds of B - A at clip weights w
        return np.mean([score(metric, S[s][1][metric], w) - score(metric, S[s][0][metric], w) for s in seeds])

    rows, boots = [], {}
    for name, mask in strata.items():
        if mask.sum() < 30:
            continue
        w0 = mask.astype(np.float64)
        row = {"stratum": name, "n_clips": int(mask.sum()), "n_videos": int(len(np.unique(g_idx[mask])))}
        for metric in METRICS:
            row[f"{na}_{metric}"] = round(np.mean([score(metric, S[s][0][metric], w0) for s in seeds]), 2)
            row[f"{nb}_{metric}"] = round(np.mean([score(metric, S[s][1][metric], w0) for s in seeds]), 2)
            row[f"gap_{metric}"] = round(gap(metric, w0), 2)
            bs = []
            for d in draws:
                gw = np.bincount(d, minlength=G).astype(np.float64)
                bs.append(gap(metric, gw[g_idx] * w0))
            bs = np.asarray(bs)
            boots[(name, metric)] = bs
            row[f"gap_{metric}_ci95"] = f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
        rows.append(row)
        print(row)
    out = pd.DataFrame(rows)
    inter = []
    for metric in METRICS:
        long_, short = boots[("frames >256f", metric)], boots[("frames <=128f", metric)]
        point = out.set_index("stratum").loc["frames >256f", f"gap_{metric}"] - out.set_index("stratum").loc["frames <=128f", f"gap_{metric}"]
        d = long_ - short
        inter.append({"metric": metric, "long_minus_short_gap": round(point, 2),
                      "ci95": f"[{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]",
                      "share_of_draws_>0": round(float((d > 0).mean()), 3)})
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "strata.csv"), index=False)
    pd.DataFrame(inter).to_csv(os.path.join(args.out, "interaction.csv"), index=False)
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(f"B={nb} minus A={na}; seeds {seeds}; {G} source videos; {args.n_boot} video-grouped bootstrap draws\n")
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    print(pd.DataFrame(inter).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
