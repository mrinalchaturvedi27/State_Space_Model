"""Shared pieces for the diagnostic program (RESEARCH_PROGRAM_2026-10-08.md steps 1-3).

Word-level scoring: teacher-forced log-probabilities of the reference's SentencePiece tokens are
summed into words (a piece starting with U+2581 opens a new word); a word counts as content when
p3.content.content() keeps it. That proxy excludes function words, digits and negation, so
content-word scores measure lexical evidence, not full semantics.
Temporal transforms keep a clip's full time span and only change how many frames observe it
(sampling density), or destroy order (reverse / shuffle: stress tests, not meaning-preserving).
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch

from p3.content import content


def resample(feat: np.ndarray, new_len: int) -> np.ndarray:
    """Linear interpolation over the clip's whole extent to new_len frames (first and last kept)."""
    T = feat.shape[0]
    new_len = max(2, int(new_len))
    if T == new_len:
        return feat.copy()
    if T < 2:
        return np.repeat(feat, new_len, axis=0)
    x = np.linspace(0, T - 1, new_len)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (x - lo)[:, None].astype(feat.dtype)
    return feat[lo] * (1 - w) + feat[hi] * w


def density(feat: np.ndarray, factor: float) -> np.ndarray:
    return resample(feat, round(feat.shape[0] * factor))


def reverse(feat: np.ndarray) -> np.ndarray:
    return feat[::-1].copy()


def shuffle_frames(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return feat[rng.permutation(feat.shape[0])]


def shuffle_blocks(feat: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [feat[i:i + block] for i in range(0, feat.shape[0], block)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def sinusoid(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """The standard sinusoidal table (phase 2 common.SinusoidalPositionalEncoding) at arbitrary,
    possibly fractional, positions. positions (T,) -> (1, T, d_model)."""
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(positions.numel(), d_model, device=positions.device)
    p = positions.float().unsqueeze(1)
    pe[:, 0::2] = torch.sin(p * div)
    pe[:, 1::2] = torch.cos(p * div)
    return pe.unsqueeze(0)


@contextmanager
def time_positions(model, rate: float):
    """Transformer only: position i of a resampled clip gets the sinusoid of original time i / rate,
    so a clip sampled at twice the density sees the same positional codes per second as at training."""
    pos = model.encoder.pos
    original = pos.forward

    def forward(x):
        p = torch.arange(x.size(1), device=x.device) / rate
        return x + sinusoid(p, x.size(-1)).to(x.dtype)

    pos.forward = forward
    try:
        yield
    finally:
        pos.forward = original


def word_scores(sp, tgt_out_row: list[int], token_logp: list[float], lang: str = "en"):
    """[(word, summed log-prob, is_content)] for one reference, from its target ids (no bos; pad and
    eos dropped) and the matching per-token log-probs."""
    words, cur_ids, cur_lp = [], [], 0.0
    specials = {sp.pad_id(), sp.eos_id(), sp.bos_id()}

    def flush():
        if cur_ids:
            w = sp.decode(cur_ids).strip()
            words.append((w, cur_lp, bool(content(w, lang))))

    for tid, lp in zip(tgt_out_row, token_logp):
        if tid in specials:
            continue
        if sp.id_to_piece(tid).startswith("▁") and cur_ids:
            flush()
            cur_ids, cur_lp = [], 0.0
        cur_ids.append(tid)
        cur_lp += lp
    flush()
    return words


@torch.no_grad()
def teacher_forced(model, batch, device):
    """Per-example token log-probs of the reference under the given source, as lists aligned with
    batch.tgt_out (pad positions dropped)."""
    b = batch.to(device)
    logits = model(b.src, b.tgt_in, b.src_key_padding_mask, b.tgt_key_padding_mask).float()
    lp = torch.log_softmax(logits, -1).gather(-1, b.tgt_out.unsqueeze(-1)).squeeze(-1)
    out = []
    for i in range(lp.size(0)):
        keep = ~b.tgt_key_padding_mask[i]
        out.append((b.tgt_out[i][keep].tolist(), lp[i][keep].tolist()))
    return out
