"""Learned retention times of the SSM encoders, from checkpoints alone (CPU; no mamba_ssm needed).

2026-10-12 review: Mamba-2 has one scalar A per head, so with input-independent B, C and Δ (the
mamba_nonselective arm, ngroups = 1) each head's long-range filter is a single decaying exponential,
    k_h(l) = g * dt_h * exp(-exp(A_log_h) * dt_h * l),   g = C_eff . B_eff (shared by all heads),
where dt_h = softplus(const_dt_h + dt_bias_h) and B_eff / C_eff are the constants after Mamba-2's
depthwise conv and SiLU (steady state). d_state does not give extra decay rates here. Reported per
layer / direction / head: dt, time constant tau = 1 / (exp(A_log) * dt) in frames and seconds (25 fps).
For selective Mamba checkpoints Δ depends on the input; tau is reported at Δ = softplus(dt_bias), the
input-neutral reference, and marked as such. S4D checkpoints: tau per complex mode = 1 / (dt * |Re A|).

  python scripts/inspect_retention.py --checkpoint "../phase 2/results/isign/mamba_nonselective/lr0.0003_s42/checkpoints/best.pt" \\
      --out results/retention/isign_mamba_nonselective_s42.csv
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

FPS = 25.0


def mamba_rows(sd: dict) -> list[dict]:
    rows = []
    for key in sorted(k for k in sd if re.search(r"layers\.\d+\.(fwd|bwd)\.A_log$", k)):
        pre = key[: -len("A_log")]
        layer, direction = re.search(r"layers\.(\d+)\.(fwd|bwd)\.", key).groups()
        A = sd[pre + "A_log"].float().exp()
        selective = pre + "const_dt" not in sd
        dt = F.softplus(sd[pre + "dt_bias"].float() + (0 if selective else sd[pre + "const_dt"].float()))
        tau = 1.0 / (A * dt)
        g = float("nan")
        if not selective:
            bc = sd[pre + "const_bc"].float()
            n = bc.numel() // 2
            if pre + "conv1d.weight" in sd:
                w, b = sd[pre + "conv1d.weight"].float()[-2 * n:, 0, :].sum(-1), sd[pre + "conv1d.bias"].float()[-2 * n:]
                bc = F.silu(bc * w + b)
            g = float(bc[:n] @ bc[n:])
        for h in range(A.numel()):
            rows.append({"layer": int(layer), "direction": direction, "head": h, "selective": selective,
                         "dt": float(dt[h]), "tau_frames": float(tau[h]), "tau_seconds": float(tau[h]) / FPS,
                         "gain_CB": g, "D": float(sd[pre + "D"].float()[h]) if pre + "D" in sd else float("nan")})
    return rows


def s4d_rows(sd: dict) -> list[dict]:
    rows = []
    for key in sorted(k for k in sd if re.search(r"layers\.\d+\.(fwd|bwd)\.log_dt$", k)):
        pre = key[: -len("log_dt")]
        layer, direction = re.search(r"layers\.(\d+)\.(fwd|bwd)\.", key).groups()
        tau = 1.0 / (sd[pre + "log_dt"].float().exp().unsqueeze(-1) * sd[pre + "log_A_real"].float().exp())
        for t in tau.flatten().tolist():
            rows.append({"layer": int(layer), "direction": direction, "selective": False, "tau_frames": t,
                         "tau_seconds": t / FPS})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    df = pd.DataFrame(mamba_rows(sd) or s4d_rows(sd))
    assert len(df), "no Mamba-2 (A_log) or S4D (log_dt) parameters found"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    q = lambda s: pd.Series({"min": s.min(), "median": s.median(), "max": s.max(), "share_>125f(5s)": (s > 125).mean(),
                             "share_>512f": (s > 512).mean()})
    summ = df.groupby(["layer", "direction"]).tau_frames.apply(q).unstack().round(3)
    pd.set_option("display.width", 200)
    if df.selective.any():
        print("SELECTIVE checkpoint: tau at the input-neutral Δ = softplus(dt_bias), for reference only")
    print(f"{args.checkpoint}\ntime constants in frames (25 fps) per layer / direction:")
    print(summ.to_string())
    if "gain_CB" in df and df.gain_CB.notna().any():
        print("C.B gain per layer/direction:", df.groupby(["layer", "direction"]).gain_CB.first().round(4).to_dict())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
