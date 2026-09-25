"""Verify the matched-params claim in BENCHMARK_PLAN.md §3: A1 (Transformer) vs A2 (Mamba)
must differ by <= 2%. Run at the start of every training job, not just once -- a config edit
to either arm should be caught here before burning GPU-hours on a confounded comparison.

Usage: python scripts/count_params.py --vocab-size 4000
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import build_model  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "model")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    rows = []
    for name in ("transformer", "mamba", "mamba_depth", "mamba_pool", "mamba_uniform"):
        cfg = load(name)
        model = build_model(cfg, vocab_size=args.vocab_size)
        n = count_params(model)
        rows.append((name, cfg["arm"], n))
        print(f"{name:16s} arm={cfg['arm']:14s} params={n:,} ({n / 1e6:.2f} M)")

    by_name = {name: n for name, _, n in rows}
    a1_n, a2_n = by_name["transformer"], by_name["mamba"]
    diff = abs(a1_n - a2_n) / a1_n
    print(f"\n|transformer - mamba| / transformer = {diff:.4f} (tolerance {args.tolerance})")
    if diff > args.tolerance:
        print(f"ABORT: transformer vs mamba differ by more than {args.tolerance:.0%} "
              f"-- adjust d_state/enc_layers in configs/model/*.yaml before training.")
        sys.exit(1)
    for name in ("mamba_pool", "mamba_uniform"):
        if by_name[name] != a2_n:
            print(f"ABORT: {name} has {by_name[name]:,} params, mamba has {a2_n:,}. "
                  "The phase-2 gate must not add parameters.")
            sys.exit(1)
    print("OK: primary pair is within tolerance, and both phase-2 arms match mamba.")


if __name__ == "__main__":
    main()
