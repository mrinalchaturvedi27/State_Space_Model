"""Span masking for masked-frame pretraining.

Hand keypoints are 136..177 of the cache's 178 (reduce_holistic order: body 8, face 128, left hand
21, right hand 21), stored as (x, y) pairs, so feature dims 272..355. A `hand_frac` share of the
spans masks only those dims (the rest of the body stays visible), pushing the encoder to infer
handshape and movement, which carry most lexical content.
"""
from __future__ import annotations

import numpy as np
import torch

HAND_DIMS = slice(272, 356)


def span_masks(lengths, T: int, ratio: float = 0.3, span_min: int = 8, span_max: int = 32,
               hand_frac: float = 0.3, rng: np.random.Generator | None = None):
    """(frame_mask, hand_mask), bool tensors (B, T). frame_mask = whole frame hidden; hand_mask =
    only hand dims hidden (never overlapping frame_mask). About `ratio` of each clip's real frames
    are covered by spans; padding is never masked."""
    rng = rng or np.random.default_rng()
    B = len(lengths)
    frame = np.zeros((B, T), dtype=bool)
    hand = np.zeros((B, T), dtype=bool)
    for b, L in enumerate(lengths):
        L = int(L)
        target = int(round(ratio * L))
        covered = np.zeros(L, dtype=bool)
        for _ in range(1000):
            if covered.sum() >= target or L == 0:
                break
            s = int(rng.integers(span_min, span_max + 1))
            s = min(s, L)
            st = int(rng.integers(0, L - s + 1))
            (hand if rng.random() < hand_frac else frame)[b, st:st + s] = True
            covered[st:st + s] = True
    hand &= ~frame
    return torch.from_numpy(frame), torch.from_numpy(hand)
