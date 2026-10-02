"""Beam-5 n-best lists for one trained model (Stage 3.0, no training).

Writes <run dir>/nbest/<split>.csv with one row per finished beam hypothesis:
uid, n_frames, reference, cand (0 = best under the length penalty), hypothesis, raw_score, n_tokens
plus <run dir>/nbest/meta.json (length penalty, checkpoint, beam). Raw scores and token counts are
kept, so scripts/mbr.py and scripts/selective.py can re-rank offline at any length penalty.

The length penalty defaults to the one chosen on validation by phase 2's lp_tune
(<run dir>/lp_tune/chosen.csv, criterion val_BLEU4), else 1.0.

  python scripts/nbest_decode.py --data configs/data/isign.yaml --model configs/model/mamba.yaml \\
      --checkpoint ../results/isign/mamba/lr0.0003_s42/checkpoints/best.pt --cache-dir ../cache
(run from "phase 3"; --data/--model paths are resolved against phase 2's configs when not found)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402,F401  (puts phase 2 on sys.path)
from p3.nbest import beam_search_nbest, rank  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import slt_reporting as R  # noqa: E402
from src.data import make_collate, make_dataset  # noqa: E402
from src.evaluate import strip_specials  # noqa: E402
from src.models import build_model  # noqa: E402


def resolve(path: str) -> str:
    return path if os.path.exists(path) else os.path.join(paths.PHASE2_DIR, path)


def chosen_lp(run_dir: str, criterion: str = "val_BLEU4") -> float:
    f = os.path.join(run_dir, "lp_tune", "chosen.csv")
    if os.path.exists(f):
        for row in csv.DictReader(open(f)):
            if row["criterion"] == criterion:
                return float(row["lp"])
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--lp", type=float, default=None, help="default: lp_tune's val-chosen value, else 1.0")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(resolve(args.data)) as f:
        data_cfg = yaml.safe_load(f)
    with open(resolve(args.model)) as f:
        model_cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    lp = args.lp if args.lp is not None else chosen_lp(run_dir)
    out_dir = args.out or os.path.join(run_dir, "nbest")
    os.makedirs(out_dir, exist_ok=True)
    name, max_new = data_cfg["dataset"], model_cfg.get("max_tgt_len", 64)
    spm = os.path.join(args.cache_dir, name, "spm.model")

    model = None
    for split in args.splits:
        ds = make_dataset(args.cache_dir, name, split, spm, max_new, 512, augment=False, model_cfg=model_cfg)
        if model is None:
            model = build_model(model_cfg, vocab_size=ds.sp.vocab_size(), pad_id=ds.pad_id).to(device).eval()
            ckpt = R.load_checkpoint(args.checkpoint, model=model, map_location=device)
            print(f"{args.checkpoint} (epoch {ckpt.get('epoch')}), lp={lp}")
        idx = list(range(len(ds)))
        if args.max_clips and args.max_clips < len(idx):
            idx = idx[:: max(1, len(idx) // args.max_clips)][: args.max_clips]
        loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, collate_fn=make_collate(ds.pad_id))
        t0, rows = time.time(), []
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                finished = beam_search_nbest(model, batch.src, batch.src_key_padding_mask, ds.bos_id,
                                             ds.eos_id, ds.pad_id, args.beam_size, max_new)
            ref = ds.sp.decode(strip_specials(batch.tgt_out[0].tolist(), ds.bos_id, ds.eos_id, ds.pad_id))
            for c, (seq, raw) in enumerate(rank(finished, lp)):
                rows.append({"uid": batch.uids[0], "n_frames": int(batch.n_frames[0]), "reference": ref,
                             "cand": c, "hypothesis": ds.sp.decode(strip_specials(seq.tolist(), ds.bos_id, ds.eos_id, ds.pad_id)),
                             "raw_score": round(raw, 5), "n_tokens": int(seq.size(0))})
        with open(os.path.join(out_dir, f"{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  {split}: {len(idx)} clips, {len(rows)} hypotheses, {time.time() - t0:.0f}s")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"lp": lp, "beam_size": args.beam_size, "checkpoint": args.checkpoint,
                   "data": args.data, "model": args.model, "max_clips": args.max_clips}, f, indent=1)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
