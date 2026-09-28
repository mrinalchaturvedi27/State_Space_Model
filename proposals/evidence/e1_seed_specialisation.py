"""E1 -- Do the 3 seeds act like separate "signer experts"? (tests the rejected MoE-as-signers idea, R1)

If seeds specialised, (a) the seed that wins a sentence would depend on the video (a proxy for
signer, since iSign has no signer ids), and (b) picking the best seed per sentence would sit far
above the logit-averaging ensemble. If they are plain variance, winners are spread at random over
videos and the ensemble already captures much of what the oracle shows.

Per arm and split:
  - corpus BLEU-4 / chrF2 per seed, seed mean, ensemble (test only), oracle (best seed per
    sentence by sentence chrF2, an upper bound that peeks at the reference) and random pick
  - Pearson r of sentence chrF2 between seeds (how correlated their errors are)
  - winner clustering: within-video concentration of the winning seed (Simpson index,
    sentence-weighted over videos with >= 5 decided sentences) vs a permutation null

  python proposals/evidence/e1_seed_specialisation.py
"""
from __future__ import annotations

import os
import random
from collections import Counter, defaultdict

import numpy as np

from common import SEEDS, RESULTS, corpus, load_preds, pred_csv, sent_chrf, video_of, write_report

ENSEMBLE = {"mamba": "mamba_ensemble/lp1.3", "transformer": "transformer_ensemble/lp1.2"}
N_PERM = 2000
MIN_PER_VIDEO = 5


def concentration(winners: np.ndarray, groups: list[np.ndarray]) -> float:
    """Sentence-weighted mean over videos of sum_s p_{v,s}^2 (1/3 = no preference, 1 = one seed)."""
    tot, n = 0.0, 0
    for idx in groups:
        c = np.bincount(winners[idx], minlength=len(SEEDS))
        p = c / c.sum()
        tot += len(idx) * float((p ** 2).sum())
        n += len(idx)
    return tot / n


def analyse(arm: str, split: str) -> str:
    runs = {s: load_preds(pred_csv("isign", arm, s, split)) for s in SEEDS}
    uids = sorted(set.intersection(*(set(r) for r in runs.values())))
    refs = [runs[SEEDS[0]][u][1] for u in uids]
    preds = {s: [runs[s][u][0] for u in uids] for s in SEEDS}
    lines = [f"### {arm}, {split} ({len(uids)} sentences)", "", "| system | BLEU-4 | chrF2 |", "|---|---|---|"]

    per_seed = {s: corpus(preds[s], refs) for s in SEEDS}
    for s in SEEDS:
        lines.append(f"| seed {s} | {per_seed[s]['BLEU-4']} | {per_seed[s]['chrF2']} |")
    lines.append(f"| seed mean | {np.mean([m['BLEU-4'] for m in per_seed.values()]):.2f} | "
                 f"{np.mean([m['chrF2'] for m in per_seed.values()]):.2f} |")
    ens_path = os.path.join(RESULTS, "isign", ENSEMBLE[arm], "predictions", f"{split}_best_beam5.csv")
    if os.path.exists(ens_path):
        ens = load_preds(ens_path)
        m = corpus([ens[u][0] for u in uids if u in ens], [ens[u][1] for u in uids if u in ens])
        lines.append(f"| ensemble (logit avg) | {m['BLEU-4']} | {m['chrF2']} |")

    chrf = np.array([[sent_chrf(p, r) for p, r in zip(preds[s], refs)] for s in SEEDS])  # (3, N)
    oracle_idx = chrf.argmax(0)
    m = corpus([preds[SEEDS[k]][i] for i, k in enumerate(oracle_idx)], refs)
    lines.append(f"| oracle best-of-3 (peeks at ref) | {m['BLEU-4']} | {m['chrF2']} |")
    rng = random.Random(0)
    rand = [corpus([preds[rng.choice(SEEDS)][i] for i in range(len(uids))], refs) for _ in range(5)]
    lines.append(f"| random pick per sentence | {np.mean([x['BLEU-4'] for x in rand]):.2f} | "
                 f"{np.mean([x['chrF2'] for x in rand]):.2f} |")

    r = np.corrcoef(chrf)
    lines += ["", "Pearson r of sentence chrF2 between seeds: " + ", ".join(
        f"{SEEDS[i]}-{SEEDS[j]} {r[i, j]:.2f}" for i in range(3) for j in range(i + 1, 3))]

    # winner clustering over videos
    top = np.sort(chrf, 0)
    decided = top[-1] - top[-2] > 1e-9
    winners = oracle_idx[decided]
    vids = [video_of(u) for u, d in zip(uids, decided) if d]
    by_vid = defaultdict(list)
    for i, v in enumerate(vids):
        by_vid[v].append(i)
    groups = [np.array(ix) for ix in by_vid.values() if len(ix) >= MIN_PER_VIDEO]
    keep = np.concatenate(groups)
    obs = concentration(winners, groups)
    nprng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):
        w = winners.copy()
        w[keep] = nprng.permutation(w[keep])
        null.append(concentration(w, groups))
    null = np.array(null)
    p = (1 + (null >= obs).sum()) / (1 + N_PERM)
    share = Counter(SEEDS[k] for k in winners)
    lines += [
        f"Decided sentences: {decided.sum()} of {len(uids)}; winner share "
        + ", ".join(f"s{s} {share[s] / len(winners):.1%}" for s in SEEDS),
        f"Winner concentration within video ({len(groups)} videos, {len(keep)} sentences): "
        f"observed {obs:.4f}, permutation null {null.mean():.4f} ± {null.std():.4f}, p = {p:.3g}",
        "",
    ]
    return "\n".join(lines)


def main():
    parts = ["# E1 -- seed specialisation (R1: MoE-as-signers)", ""]
    for split in ("test", "val"):
        for arm in ("mamba", "transformer"):
            parts.append(analyse(arm, split))
            print(parts[-1])
    print("wrote", write_report("e1_seed_specialisation.md", "\n".join(parts)))


if __name__ == "__main__":
    main()
