"""Fast sentence chrF (character n-grams 1..6, beta 2, whitespace removed) with cached n-gram
counts, for consensus decoding where every candidate is scored against every other one.

Follows sacrebleu's chrF: precision and recall are averaged over the n-gram orders the strings
have, then combined into F-beta. Used as the MBR *utility*; reported scores always use sacrebleu.
"""
from __future__ import annotations

from collections import Counter

ORDER, BETA = 6, 2.0


def ngram_stats(text: str) -> list[Counter]:
    s = "".join(text.split())
    return [Counter(s[i:i + n] for i in range(len(s) - n + 1)) for n in range(1, ORDER + 1)]


def chrf(hyp: list[Counter], ref: list[Counter]) -> float:
    precs, recs = [], []
    for h, r in zip(hyp, ref):
        th, tr = sum(h.values()), sum(r.values())
        if th == 0 or tr == 0:
            continue
        m = sum((h & r).values())
        precs.append(m / th)
        recs.append(m / tr)
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = BETA ** 2
    return 100 * (1 + b2) * p * r / (b2 * p + r)
