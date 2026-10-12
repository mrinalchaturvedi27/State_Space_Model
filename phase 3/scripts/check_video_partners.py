"""E1 partner check (2026-10-12 review): the 'wrong video' partners were chosen with parse_uid, which
treats iSign dictionary clips (<id>_d, <id>_e1..e4) as one-clip videos. This maps every fixed sample's
partner indices to uids with source_video(), flags pairs from the SAME source video, and
recomputes each model's E1 summary without the flagged clips (summary_partner_checked.csv next to the
original). CPU, seconds.

  python scripts/check_video_partners.py --cache-dir ../cache --dataset isign \\
      --video-dir results/diagnostics/screen/video
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from p3.stitch import source_video  # noqa: E402


def boot_ci(x, rng, n=1000):
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-dir", required=True)
    args = ap.parse_args()
    uids = pd.read_parquet(os.path.join(args.cache_dir, args.dataset, "val_index.parquet"))["uid"].astype(str).to_numpy()
    S = pd.read_csv(os.path.join(args.video_dir, f"sample_video_{args.dataset}.csv"))
    assert (uids[S.idx.to_numpy()] == S.uid.astype(str).to_numpy()).all(), "sample idx does not match the cache index"
    fam = lambda u: source_video(str(u))
    bad = set()
    for col in ("wrong_length_idx", "wrong_topic_idx"):
        same = [fam(u) == fam(uids[j]) for u, j in zip(S.uid, S[col])]
        print(f"{col}: {sum(same)} of {len(S)} partners come from the same source video")
        bad |= set(S.uid[np.array(same)])
    print(f"clips to exclude: {len(bad)}")
    for d in sorted(glob.glob(os.path.join(args.video_dir, "*", "per_clip.csv"))):
        D = pd.read_csv(d)
        D = D[~D.uid.isin(bad)]
        base = D[D.condition == "correct"].set_index("uid")
        rng = np.random.default_rng(1)
        rows = []
        for cond in D.condition.unique():
            c = D[D.condition == cond].set_index("uid").loc[base.index]
            row = {"condition": cond, "n_clips": len(c), "token_nll": round(-c.ref_ll.sum() / c.n_tokens.sum(), 4)}
            if cond != "correct":
                d_all = (base.ref_ll - c.ref_ll) / base.n_tokens
                lo, hi = boot_ci(d_all, rng)
                row.update({"delta_token_ll_vs_correct": round(d_all.mean(), 4), "ci95": f"[{lo:.4f}, {hi:.4f}]",
                            "correct_better_%": round(100 * (d_all > 0).mean(), 1)})
            rows.append(row)
        out = os.path.join(os.path.dirname(d), "summary_partner_checked.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n{os.path.basename(os.path.dirname(d))} (without {len(bad)} flagged clips):")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
