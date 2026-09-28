"""What does the short bank's Δ track, and where does Δ pooling put the decoder memory?

For a trained horizon-banked checkpoint (any phase-2 arm except mamba_padfix), over one split:

1. Spearman ρ, per clip, between each layer/bank's forward Δ_t and body / face / hand speed
   (mean keypoint displacement per frame), with a time-shuffled Δ as the chance baseline.
2. Lagged ρ(Δ_t, hand speed_{t+lag}) for the pooling layer's short bank: does Δ lead or lag motion?
3. Hand speed (z-scored per clip) at the frames Δ keeps vs the frames a uniform stride keeps,
   at the same count (matched_delta_pool vs uniform_pool), plus how irregular Δ's spacing is.

Writes <out>/delta_summary.json and <out>/delta_clips.npz (first --n-dump clips: Δ per bank,
speeds, kept indices) for plotting. Keypoint layout follows the cache: reduce_holistic's
178 points as (x, y) pairs, POSE 8 | FACE 128 | LEFT_HAND 21 | RIGHT_HAND 21.

  python scripts/analyze_delta.py --data configs/data/isign.yaml \\
      --model configs/model/mamba_pool_matched.yaml \\
      --checkpoint results/isign/mamba_pool_matched/lr0.0003_s42/checkpoints/best.pt \\
      --cache-dir ../cache --out results/isign/mamba_pool_matched/lr0.0003_s42/delta_analysis
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import slt_reporting as R  # noqa: E402
from src.data import PoseTextDataset, make_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.scope import horizon_specs, matched_delta_pool, uniform_pool  # noqa: E402

PARTS = {"body": (0, 8), "face": (8, 136), "hands": (136, 178)}
LAGS = list(range(-8, 9))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def part_speed(feat: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean per-keypoint displacement between consecutive frames, length T (first frame copied)."""
    kp = feat.reshape(feat.shape[0], -1, 2)[:, lo:hi]
    step = np.linalg.norm(np.diff(kp, axis=0), axis=-1).mean(axis=1)
    return np.concatenate([step[:1], step]) if len(step) else np.zeros(feat.shape[0])


@torch.no_grad()
def layer_deltas(model, src: torch.Tensor) -> list[torch.Tensor]:
    """Forward Δ (T, nheads) of every encoder layer, as the fused scan computes it."""
    enc = model.encoder
    x = model.front_end(src)
    out = []
    for layer in enc.layers:
        x, hidden = layer(x, None, return_short_delta=True)
        fwd = layer.fwd
        nh = fwd.nheads
        dt_raw = F.linear(hidden.float(), fwd.in_proj.weight[-nh:].float())
        out.append(F.softplus(dt_raw + fwd.dt_bias.float())[0])
    return out


def summarize(values: list[float]) -> dict:
    v = np.array([x for x in values if np.isfinite(x)])
    if not len(v):
        return {"n": 0}
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4),
            "median": round(float(np.median(v)), 4), "frac_pos": round(float((v > 0).mean()), 4),
            "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = whole split")
    ap.add_argument("--n-dump", type=int, default=40)
    ap.add_argument("--min-frames", type=int, default=32, help="skip clips too short to correlate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)
    if model_cfg["arm"] == "mamba_padfix":
        sys.exit("mamba_padfix has no horizon banks; pick a banked arm")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_name = data_cfg["dataset"]
    ds = PoseTextDataset(args.cache_dir, ds_name, args.split,
                         os.path.join(args.cache_dir, ds_name, "spm.model"),
                         model_cfg.get("max_tgt_len", 64), 512, augment=False)
    model = build_model(model_cfg, vocab_size=ds.sp.vocab_size(), pad_id=ds.pad_id).to(device).eval()
    R.load_checkpoint(args.checkpoint, model=model, map_location=device)

    enc = model.encoder
    nh = enc.layers[0].fwd.nheads
    banks = {name: (s, e) for name, (s, e, _, _) in zip(
        ("short", "mid", "long"),
        horizon_specs(nh, model_cfg.get("short_heads", 8), model_cfg.get("mid_heads", 4)))}
    stride = model_cfg.get("pool_every_frames", 16)
    last = len(enc.layers) - 1
    rng = np.random.default_rng(0)

    rho = {}          # (layer, bank, part) -> per-clip ρ
    rho_shuffled = []  # pooling layer short bank vs hands, Δ shuffled in time
    lag_rho = {lag: [] for lag in LAGS}
    kept_hand_z = {"delta": [], "uniform": []}
    spacing_cv = {"delta": [], "uniform": []}
    dump = {}

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=make_collate(ds.pad_id))
    n_seen = 0
    for batch in loader:
        feat = batch.src[0].numpy()
        T = feat.shape[0]
        if T < args.min_frames:
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            deltas = layer_deltas(model, batch.src.to(device))
        speeds = {p: part_speed(feat, lo, hi) for p, (lo, hi) in PARTS.items()}

        for li, d in enumerate(deltas):
            for bname, (s, e) in banks.items():
                series = d[:, s:e].mean(-1).cpu().numpy()
                for p, sp in speeds.items():
                    rho.setdefault((li, bname, p), []).append(spearman(series, sp))

        short = deltas[last][:, banks["short"][0]:banks["short"][1]].mean(-1).float().cpu()
        short_np, hand = short.numpy(), speeds["hands"]
        rho_shuffled.append(spearman(rng.permutation(short_np), hand))
        for lag in LAGS:
            if lag >= 0:
                lag_rho[lag].append(spearman(short_np[: T - lag], hand[lag:]))
            else:
                lag_rho[lag].append(spearman(short_np[-lag:], hand[: T + lag]))

        idx_mem = torch.arange(T, dtype=torch.float32).view(1, T, 1)
        kept_d, pad_d = matched_delta_pool(idx_mem, short.view(1, T), None, stride)
        kept_u, pad_u = uniform_pool(idx_mem, None, stride)
        kd = kept_d[0, ~pad_d[0], 0].long().numpy()
        ku = kept_u[0, ~pad_u[0], 0].long().numpy()
        hz = (hand - hand.mean()) / (hand.std() + 1e-8)
        kept_hand_z["delta"].append(float(hz[kd].mean()))
        kept_hand_z["uniform"].append(float(hz[ku].mean()))
        for name, k in (("delta", kd), ("uniform", ku)):
            gaps = np.diff(k)
            if len(gaps) > 1:
                spacing_cv[name].append(float(gaps.std() / gaps.mean()))

        if len(dump) < args.n_dump:
            dump[batch.uids[0]] = {
                **{f"delta_{b}": deltas[last][:, s:e].mean(-1).float().cpu().numpy()
                   for b, (s, e) in banks.items()},
                **{f"speed_{p}": sp for p, sp in speeds.items()},
                "kept_delta": kd, "kept_uniform": ku,
            }
        n_seen += 1
        if args.max_clips and n_seen >= args.max_clips:
            break

    summary = {
        "checkpoint": args.checkpoint, "split": args.split, "clips": n_seen, "stride": stride,
        "pooling_layer": last,
        "rho_delta_vs_speed": {f"L{li}/{b}/{p}": summarize(v) for (li, b, p), v in sorted(rho.items())},
        "rho_shuffled_baseline": summarize(rho_shuffled),
        "lag_rho_short_vs_hands": {str(l): summarize(v).get("mean") for l, v in lag_rho.items()},
        "hand_speed_z_at_kept_frames": {k: summarize(v) for k, v in kept_hand_z.items()},
        "kept_spacing_cv": {k: summarize(v) for k, v in spacing_cv.items()},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "delta_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "delta_clips.npz"),
                        **{f"{uid}|{k}": v for uid, d in dump.items() for k, v in d.items()})

    top = summary["rho_delta_vs_speed"]
    print(f"{n_seen} clips. pooling layer L{last}, short bank:")
    for p in PARTS:
        print(f"  ρ(Δ, {p} speed) = {top[f'L{last}/short/{p}']}")
    print(f"  shuffled baseline = {summary['rho_shuffled_baseline']}")
    print(f"  hand-speed z at kept frames: {summary['hand_speed_z_at_kept_frames']}")
    print(f"  spacing CV (0 = even): {summary['kept_spacing_cv']}")
    print(f"wrote {args.out}/delta_summary.json and delta_clips.npz")


if __name__ == "__main__":
    main()
