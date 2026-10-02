"""Minimum Bayes risk choice with a chrF utility (shared by scripts/mbr.py and scripts/selective.py)."""
from __future__ import annotations

from p3.chrf import chrf, ngram_stats


def mbr_pick(cands: list[str], refs: list[str], same: bool = True):
    """(index of the candidate with the highest mean chrF against refs, that mean).
    same=True: cands and refs are the same list, so a candidate is not scored against itself
    (identical strings from *other* systems still count: that is agreement)."""
    cs = [ngram_stats(c) for c in cands]
    rs = cs if same else [ngram_stats(r) for r in refs]
    best, best_u = 0, -1.0
    for i, c in enumerate(cs):
        us = [chrf(c, r) for j, r in enumerate(rs) if not (same and j == i)]
        u = sum(us) / len(us) if us else 0.0
        if u > best_u:
            best, best_u = i, u
    return best, best_u


def mean_chrf_against(text: str, others: list[str]) -> float:
    if not others:
        return 0.0
    t = ngram_stats(text)
    return sum(chrf(t, ngram_stats(o)) for o in others) / len(others)
