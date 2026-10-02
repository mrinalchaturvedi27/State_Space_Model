"""Consensus (minimum Bayes risk) decoding over several systems' n-best lists (Stage 3.0a).

For each clip, every candidate is scored by its average chrF (p3/chrf.py) against all the other
hypotheses in the pool, which act as pseudo-references; the highest-scoring candidate is chosen.
Its expected utility doubles as a confidence score for scripts/selective.py.

Systems compared (all scored with sacreBLEU corpus chrF2 / BLEU-4):
  single:<name>  each system's own best hypothesis
  mbr_self       MBR within one system's top-k (shown for the first system)
  mbr_top1       MBR among the systems' best hypotheses only
  mbr_pool       MBR over the union of all systems' top-k   <- the method
  oracle         the pool candidate with the best true sentence chrF (upper bound, not a system)
Paired bootstrap (1,000 samples): mbr_pool vs every single system.

  python scripts/mbr.py --runs ../results/isign/mamba/lr0.0003_s{13,42,1337}/nbest --split test \\
      --out results/stage30/isign_mamba
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3.consensus import mbr_pick  # noqa: E402
from p3.nbestio import load_nbest  # noqa: E402

import sacrebleu  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402
from sacrebleu.significance import PairedTest  # noqa: E402


def corpus(preds, refs):
    return sacrebleu.corpus_chrf(preds, [refs]).score, sacrebleu.corpus_bleu(preds, [refs]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="nbest directories, one per system")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = args.names or [f"sys{i}" for i in range(len(args.runs))]
    systems = [load_nbest(r, args.split, top_k=args.top_k) for r in args.runs]
    uids = [u for u in systems[0] if all(u in s for s in systems)]
    refs = [systems[0][u]["ref"] for u in uids]
    print(f"{len(uids)} clips common to {len(systems)} systems")

    out = {f"single:{n}": [] for n in names}
    out.update({"mbr_self": [], "mbr_top1": [], "mbr_pool": [], "oracle": []})
    conf = []
    for u in uids:
        tops = [s[u]["hyps"][0][0] for s in systems]
        for n, t in zip(names, tops):
            out[f"single:{n}"].append(t)
        own = [h[0] for h in systems[0][u]["hyps"]]
        out["mbr_self"].append(own[mbr_pick(own, own)[0]])
        out["mbr_top1"].append(tops[mbr_pick(tops, tops)[0]])
        pool = [h[0] for s in systems for h in s[u]["hyps"]]
        i, eu = mbr_pick(pool, pool)
        out["mbr_pool"].append(pool[i])
        conf.append(eu)
        ref = systems[0][u]["ref"]
        out["oracle"].append(max(set(pool), key=lambda c: sacrebleu.sentence_chrf(c, [ref]).score))

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for k, preds in out.items():
        c, b = corpus(preds, refs)
        rows.append({"system": k, "chrF2": round(c, 3), "BLEU4": round(b, 3)})
        print(f"  {k:28s} chrF2 {c:6.2f}  BLEU-4 {b:5.2f}")
    sys_list = [(n, out[n]) for n in out if n.startswith("single:")] + [("mbr_pool", out["mbr_pool"])]
    sys_list = [sys_list[-1]] + sys_list[:-1]  # baseline = mbr_pool, others compared to it
    _, res = PairedTest(sys_list, {"chrF": CHRF(), "BLEU": BLEU()}, [refs], test_type="bs", n_samples=1000)()
    for n, c, b in zip(res["System"][1:], res["chrF2"][1:], res["BLEU"][1:]):
        print(f"  mbr_pool vs {n}: chrF p={c.p_value:.3f}, BLEU p={b.p_value:.3f}")
        rows.append({"system": f"p_mbr_pool_vs_{n}", "chrF2": round(c.p_value, 4), "BLEU4": round(b.p_value, 4)})
    with open(os.path.join(args.out, f"mbr_{args.split}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "chrF2", "BLEU4"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, f"mbr_{args.split}_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "prediction", "ground_truth", "mbr_expected_chrF", "n_frames"])
        for u, p, r, e in zip(uids, out["mbr_pool"], refs, conf):
            w.writerow([u, p, r, round(e, 3), systems[0][u]["n_frames"]])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
