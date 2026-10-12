"""Boundary vs context (2026-10-12 review): does the windowed-Mamba long-clip penalty come from losing
distant context, or from the artificial resets at window edges? Inference only, dev data.

The encoder of a trained Mamba checkpoint (padfix, non-selective, or a window-trained one) is run under
each condition; the restriction applies to the complete encoder, not per layer:
  full           no windows
  block{W}       non-overlapping W-frame windows on the grid 0, W, 2W, ... (mamba_window's training setup)
  block{W}_off{o} the same grid shifted: the first block has o frames
  overlap{W}     W-frame windows at stride W/2, each output from the window in whose central half the
                 frame lies: at most W frames of context, but no frame (except clip ends) at an edge
Reported on the fixed short (<=256) and long (257-512 frames) dev samples of diag_density.py:
reference token NLL (teacher forcing) and greedy chrF2 / BLEU-4.
Reading: if overlap{W} removes most of block{W}'s long-clip penalty, the penalty is mainly boundaries;
if the penalty persists and shrinks steadily with W, distant context matters.

  python scripts/diag_window.py --data configs/data/isign.yaml --model configs/model/mamba_window.yaml \\
      --checkpoint "../phase 2/results/isign/mamba_window/lr0.0003_s42/checkpoints/best.pt" --cache-dir ../cache \\
      --sample-short results/diagnostics/queue/density_short/sample_density_isign_0-256.csv \\
      --sample-long results/diagnostics/queue/density_long/sample_density_isign_257-512.csv \\
      --out results/diagnostics/window/isign_mamba_window_s42
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402,F401
from p3.diag import teacher_forced  # noqa: E402
from p3.loading import load  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402

from src.data import make_collate  # noqa: E402
from src.models.selectivity import WindowedEncoder  # noqa: E402
from src.train import greedy_decode, ids_to_text  # noqa: E402

DEFAULT = "full block64 block128 block256 overlap64 overlap128 overlap256 block128_off32 block128_off64 block128_off96"


def encoder_for(cond: str, inner: torch.nn.Module) -> torch.nn.Module:
    if cond == "full":
        return inner
    m = re.fullmatch(r"(block|overlap)(\d+)(?:_off(\d+))?", cond)
    assert m, f"unknown condition {cond}"
    return WindowedEncoder(inner, window=int(m.group(2)), mode=m.group(1), offset=int(m.group(3) or 0))


@torch.no_grad()
def evaluate(model, ds, S, device, max_tgt, batch):
    collate = make_collate(ds.pad_id)
    ll = ntok = 0
    hyps, refs = [], []
    order = S.sort_values("frames").reset_index(drop=True)
    for s in range(0, len(order), batch):
        items = [ds[int(r.idx)] for r in order.iloc[s:s + batch].itertuples()]
        b = collate([{"uid": i["uid"], "feat": i["feat"], "ids": i["ids"]} for i in items])
        for _, lps in teacher_forced(model, b, device):
            ll, ntok = ll + float(np.sum(lps)), ntok + len(lps)
        bb = b.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = greedy_decode(model, bb.src, bb.src_key_padding_mask, ds.bos_id, ds.eos_id, ds.pad_id, max_tgt)
        hyps += ids_to_text(ds.sp, out, ds.bos_id, ds.eos_id, ds.pad_id)
        refs += ids_to_text(ds.sp, bb.tgt_out, ds.bos_id, ds.eos_id, ds.pad_id)
    return {"token_nll": round(-ll / ntok, 4), "greedy_chrF2": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
            "greedy_BLEU4": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--sample-short", required=True)
    ap.add_argument("--sample-long", required=True)
    ap.add_argument("--conditions", default=DEFAULT)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ds, _, model_cfg = load(args.data, args.model, args.checkpoint, args.cache_dir, device)
    trained = model.encoder
    inner = trained.inner if isinstance(trained, WindowedEncoder) else trained
    train_cond = (f"{trained.mode}{trained.window}" + (f"_off{trained.offset}" if trained.offset else "")
                  if isinstance(trained, WindowedEncoder) else "full")
    rows = []
    for cond in args.conditions.split():
        model.encoder = encoder_for(cond, inner)
        for rng, f in (("short", args.sample_short), ("long", args.sample_long)):
            S = pd.read_csv(f)
            r = {"condition": cond, "trained_as": train_cond, "range": rng, "n": len(S),
                 **evaluate(model, ds, S, device, model_cfg.get("max_tgt_len", 64), args.batch)}
            rows.append(r)
            print(r, flush=True)
    model.encoder = trained
    df = pd.DataFrame(rows)
    wide = df.pivot(index="condition", columns="range", values=["greedy_chrF2", "token_nll"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["chrF2_long_minus_short"] = (wide.greedy_chrF2_long - wide.greedy_chrF2_short).round(2)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "summary_long_format.csv"), index=False)
    wide.reindex([c for c in args.conditions.split() if c in wide.index]).to_csv(os.path.join(args.out, "summary.csv"))
    pd.set_option("display.width", 200)
    print(wide.to_string())
    print(f"wrote {args.out} (trained as {train_cond})")


if __name__ == "__main__":
    main()
