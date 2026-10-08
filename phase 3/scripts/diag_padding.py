"""Diagnostic 1b: padding invariance on REAL trained checkpoints (GPU; mamba_ssm kernels).

For a fixed set of validation clips, encodes each clip alone and again batched next to the longest
validation clip (so it is padded), in fp32, and reports the largest difference over the clip's real
positions in (a) the encoder memory and (b) the teacher-forced reference log-probabilities.
Expected: ~0 for the Transformer and mamba_padfix; non-zero for the phase-1 Mamba (backward-scan leak).

  python scripts/diag_padding.py --data configs/data/isign.yaml --model configs/model/mamba_padfix.yaml \\
      --checkpoint "../phase 2/results/isign/mamba_padfix/lr0.0003_s42/checkpoints/best.pt" --cache-dir ../cache \\
      --out results/diagnostics/padding/isign_mamba_padfix.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402,F401
from p3.diag import teacher_forced  # noqa: E402
from p3.loading import load  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data import make_collate  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ds, _, _ = load(args.data, args.model, args.checkpoint, args.cache_dir, device)
    nf = ds.n_frames_array()
    longest = int(np.argmax(nf))
    picks = [int(i) for i in np.argsort(nf)[np.linspace(0, len(nf) - 2, args.n).astype(int)] if i != longest]
    collate = make_collate(ds.pad_id)
    rows = []
    for i in picks:
        alone = collate([ds[i]])
        padded = collate([ds[i], ds[longest]])
        L = int((~alone.src_key_padding_mask[0]).sum())
        m1, p1 = model.encode(alone.src.to(device), alone.src_key_padding_mask.to(device))
        m2, p2 = model.encode(padded.src.to(device), padded.src_key_padding_mask.to(device))
        n1 = int((~p1[0]).sum()) if p1 is not None else m1.size(1)
        mem_diff = float((m1[0, :n1] - m2[0, :n1]).abs().max())
        t1 = teacher_forced(model, alone, device)[0][1]
        t2 = teacher_forced(model, padded, device)[0][1]
        lp_diff = float(np.max(np.abs(np.array(t1) - np.array(t2[:len(t1)]))))
        rows.append({"uid": ds[i]["uid"], "frames": L, "pad_frames": int(nf[longest]) - L,
                     "max_abs_memory_diff": mem_diff, "max_abs_token_logprob_diff": lp_diff})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nmax memory diff {df.max_abs_memory_diff.max():.2e}, max token log-prob diff "
          f"{df.max_abs_token_logprob_diff.max():.2e} over {len(df)} clips -> {args.out}")


if __name__ == "__main__":
    main()
