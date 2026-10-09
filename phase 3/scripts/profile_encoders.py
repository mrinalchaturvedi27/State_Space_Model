"""Resource profile for the main results table (review: parameters, training cost, batch-1 latency,
memory). No checkpoints needed: weights are random, only shapes and kernels matter.

Per model config:
  params_total / params_encoder
  train_step_ms, train_peak_GB   one optimizer step on a token-budget batch (default 24,576 frames
                                 = 96 clips x 256 frames, 30 target tokens), bf16 autocast on GPU
  enc_ms_T{128,256,512}          batch-1 encoder latency (median of --reps)
The decoder is identical across encoders, so decode cost is not repeated here.

  python scripts/profile_encoders.py --models transformer mamba_padfix tcn transformer_convstem \\
      --out results/profile/profile.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.models import build_model  # noqa: E402


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, reps):
    for _ in range(3):
        fn()
    sync(device)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(device)
        ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="config names in phase 2/configs/model")
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--clips", type=int, default=96)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--tgt-len", type=int, default=30)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--small", action="store_true", help="tiny dims, CPU smoke test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in args.models:
        cfg = yaml.safe_load(open(os.path.join(paths.PHASE2_DIR, "configs", "model", f"{name}.yaml")))
        if args.small:
            cfg.update(d_model=32, dim_feedforward=64, n_heads=4)
        torch.manual_seed(0)
        model = build_model(cfg, vocab_size=args.vocab).to(device)
        row = {"model": name, "params_total": sum(p.numel() for p in model.parameters()),
               "params_encoder": sum(p.numel() for p in model.encoder.parameters())}
        d_in = cfg["d_in"]
        src = torch.randn(args.clips, args.frames, d_in, device=device)
        pad = torch.zeros(args.clips, args.frames, dtype=torch.bool, device=device)
        tgt = torch.randint(4, args.vocab, (args.clips, args.tgt_len + 1), device=device)
        tpad = torch.zeros(args.clips, args.tgt_len, dtype=torch.bool, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model.compute_loss(src, tgt[:, :-1], tgt[:, 1:], pad, tpad)
            loss.backward()
            opt.step()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        row["train_step_ms"] = round(timed(step, device, max(3, args.reps // 4)), 1)
        if device == "cuda":
            row["train_peak_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        model.eval()
        for T in (128, 256, 512):
            x = torch.randn(1, T, d_in, device=device)
            p = torch.zeros(1, T, dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                row[f"enc_ms_T{T}"] = round(timed(lambda: model.encode(x, p), device, args.reps), 2)
        rows.append(row)
        print(row)
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {args.out} (device {device}{', ' + torch.cuda.get_device_name() if device == 'cuda' else ''})")


if __name__ == "__main__":
    main()
