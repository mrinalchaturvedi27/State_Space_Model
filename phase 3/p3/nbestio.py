"""Read n-best lists written by scripts/nbest_decode.py."""
from __future__ import annotations

import json
import os

import pandas as pd


def load_nbest(run_nbest_dir: str, split: str, lp: float | None = None, top_k: int = 5):
    """{uid: {"ref", "n_frames", "hyps": [(text, raw_score, n_tokens), ...] best-first}}.
    Re-ranks by raw/len**lp when lp is given (else keeps the file's order = its own lp)."""
    df = pd.read_csv(os.path.join(run_nbest_dir, f"{split}.csv"), keep_default_na=False)
    out = {}
    for uid, g in df.groupby("uid", sort=False):
        hyps = list(zip(g.hypothesis.astype(str), g.raw_score.astype(float), g.n_tokens.astype(int)))
        if lp is not None:
            hyps.sort(key=lambda h: -h[1] / (h[2] ** lp))
        out[uid] = {"ref": str(g.reference.iloc[0]), "n_frames": int(g.n_frames.iloc[0]), "hyps": hyps[:top_k]}
    return out


def meta(run_nbest_dir: str) -> dict:
    with open(os.path.join(run_nbest_dir, "meta.json")) as f:
        return json.load(f)
