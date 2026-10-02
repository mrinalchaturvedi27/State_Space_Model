"""Masked-frame self-supervised pretraining of the translation encoder (Stage 3.1).

  --mode long   windows of up to --window frames cut from stitched story-level videos (P2)
  --mode clip   the same frames, one clip per item (P1, the control for long context)
Same epochs and frame budget per step in both modes, so the only difference is context length.
Train split only (no val/test poses are seen); val masked-reconstruction loss (fixed masks) picks
the checkpoint. Writes <out>/encoder_init.pt ({'model': front_end+encoder state dict, 'config': ...}),
<out>/log.csv, and with --stats-only just prints the stitching statistics.

  python scripts/pretrain.py --data configs/data/isign.yaml --model configs/model/mamba_padfix.yaml \\
      --cache-dir ../cache --mode long --out results/pretrain/P2_isign_mamba
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from p3.masking import span_masks  # noqa: E402
from p3.pretrain_model import MaskedPoseModel  # noqa: E402
from p3.stitch import EpochTokenSampler, PretrainDataset, VideoIndex, collate  # noqa: E402
from src.train import cosine_with_warmup  # noqa: E402

NO_DECAY = ("bias", "A_log", "D", "dt_bias", "mask_token")


def resolve(path: str) -> str:
    return path if os.path.exists(path) else os.path.join(paths.PHASE2_DIR, path)


def param_groups(model, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        (no if n.rsplit(".", 1)[-1] in NO_DECAY or "norm" in n.lower() else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def run_epoch(model, loader, device, args, rng, train, opt=None, sched=None, step=0):
    model.train(train)
    tot, n_tok, n_b = 0.0, 0, 0
    for batch in loader:
        lengths = (~batch["pad"]).sum(1).tolist()
        fm, hm = span_masks(lengths, batch["feat"].size(1), args.mask_ratio, args.span_min, args.span_max,
                            args.hand_frac, rng)
        feat, pad, fm, hm = (t.to(device, non_blocking=True) for t in (batch["feat"], batch["pad"], fm, hm))
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                           enabled=device == "cuda"):
            loss, n = model(feat, pad, fm, hm)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
        tot += loss.item() * n
        n_tok += n
        n_b += 1
        if train and args.max_steps and step >= args.max_steps:
            break
    return tot / max(1, n_tok), step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--mode", choices=("clip", "long"), default="long")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--span-min", type=int, default=8)
    ap.add_argument("--span-max", type=int, default=32)
    ap.add_argument("--hand-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0, help="cap, for smoke tests")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(resolve(args.data)) as f:
        data_cfg = yaml.safe_load(f)
    with open(resolve(args.model)) as f:
        model_cfg = yaml.safe_load(f)
    if model_cfg["arm"] == "transformer" and args.mode == "long":
        model_cfg["max_src_len"] = max(model_cfg.get("max_src_len", 512), args.window)  # positional table only
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_index = VideoIndex(args.cache_dir, data_cfg["dataset"], "train")
    va_index = VideoIndex(args.cache_dir, data_cfg["dataset"], "val")
    print(f"[{data_cfg['dataset']}] train {json.dumps(tr_index.stats())}")
    print(f"[{data_cfg['dataset']}] val   {json.dumps(va_index.stats())}")
    if args.stats_only:
        return

    tr = PretrainDataset(tr_index, args.mode, args.window, seed=args.seed, augment=True)
    va = PretrainDataset(va_index, args.mode, args.window, seed=0, augment=False)
    tr_s = EpochTokenSampler(tr, args.max_tokens, seed=args.seed, shuffle=True)
    va_s = EpochTokenSampler(va, args.max_tokens, seed=0, shuffle=False)
    mk = lambda ds, s, w: DataLoader(ds, batch_sampler=s, collate_fn=collate, num_workers=w, pin_memory=True)
    tr_l, va_l = mk(tr, tr_s, args.num_workers), mk(va, va_s, min(2, args.num_workers))

    model = MaskedPoseModel(model_cfg).to(device)
    n_enc = sum(p.numel() for p in list(model.front_end.parameters()) + list(model.encoder.parameters()))
    opt = AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    total = args.max_steps or len(tr_s) * args.epochs
    sched = LambdaLR(opt, functools.partial(cosine_with_warmup, warmup_steps=max(1, int(args.warmup_ratio * total)),
                                            total_steps=total, min_lr_ratio=0.1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump({**vars(args), "model_cfg": model_cfg, "encoder_params": n_enc}, f, indent=1)
    print(f"mode={args.mode} encoder params {n_enc:,}; ~{len(tr_s)} steps/epoch, {total} total; out={args.out}")

    best, step, log = math.inf, 0, []
    for epoch in range(args.epochs):
        tr.set_epoch(epoch)
        tr_s.set_epoch(epoch)
        t0 = time.time()
        tr_loss, step = run_epoch(model, tr_l, device, args, np.random.default_rng([args.seed, epoch]), True, opt, sched, step)
        with torch.no_grad():
            va_loss, _ = run_epoch(model, va_l, device, args, np.random.default_rng(0), False)
        log.append({"epoch": epoch, "step": step, "train_l1": round(tr_loss, 5), "val_l1": round(va_loss, 5),
                    "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: train L1 {tr_loss:.4f} val L1 {va_loss:.4f} ({time.time() - t0:.0f}s, step {step})")
        payload = {"model": model.encoder_state_dict(), "config": model_cfg, "epoch": epoch, "val_l1": va_loss,
                   "pretrain_args": vars(args)}
        torch.save(payload, os.path.join(args.out, "encoder_last.pt"))
        if va_loss < best:
            best = va_loss
            torch.save(payload, os.path.join(args.out, "encoder_init.pt"))
        with open(os.path.join(args.out, "log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done; best val L1 {best:.4f} -> {os.path.join(args.out, 'encoder_init.pt')}")


if __name__ == "__main__":
    main()
