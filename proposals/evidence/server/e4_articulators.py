"""E4 -- Is there anything for articulator-factored scans to exploit? (Proposal B gate, runs on the server)

Part 1, data only (no checkpoint): are hand and face motion asynchronous? Per clip, lagged Spearman
ρ between hand speed and *non-rigid* face speed (face keypoints minus their per-frame centroid, so
head translation is removed), lags -16..16 frames. Peak lags far from 0, or a weak lag-0 ρ, mean
the two channels carry separate timing that a shared Δ would have to average over.

Part 2, with --checkpoint (any horizon-banked phase-2 arm; mamba_banks is the cleanest): does any
head already specialise? Per layer and head, partial Spearman ρ(Δ, hands | face) and
ρ(Δ, face | hands), each against a time-shuffled Δ baseline. If the heads split into hand-tracking
and face-tracking groups, factored scans would only formalise what the shared scan learns; if every
head tracks hands and none tracks the face, the face channel is under-used and B has a case.

Keypoint layout as in "phase 2/scripts/analyze_delta.py": POSE 8 | FACE 128 | LEFT_HAND 21 | RIGHT_HAND 21.

On t3ihpc07, from "phase 2/":
  python ../proposals/evidence/server/e4_articulators.py --data configs/data/isign.yaml \\
      --model configs/model/mamba_banks.yaml \\
      --checkpoint results/isign/mamba_banks/lr0.0003_s42/checkpoints/best.pt \\
      --cache-dir ../cache --max-clips 2000 --out ../proposals/results/e4_isign
Drop --checkpoint (keep --model for the tokenizer settings) to run part 1 only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "phase 2"))
LAGS = list(range(-16, 17))
FACE = (8, 136)
HANDS = (136, 178)


def ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """ρ(a, b | c) on ranks: correlate the residuals of a and b after regressing each on c."""
    if len(a) < 6 or min(np.std(a), np.std(b), np.std(c)) == 0:
        return float("nan")
    ra, rb, rc = ranks(a), ranks(b), ranks(c)
    X = np.stack([rc, np.ones_like(rc)], 1)
    res_a = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    res_b = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    if np.std(res_a) == 0 or np.std(res_b) == 0:
        return float("nan")
    return float(np.corrcoef(res_a, res_b)[0, 1])


def speeds(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hand speed, non-rigid face speed), length T, first frame copied."""
    kp = feat.reshape(feat.shape[0], -1, 2)
    hand = kp[:, HANDS[0]:HANDS[1]]
    face = kp[:, FACE[0]:FACE[1]]
    face = face - face.mean(axis=1, keepdims=True)
    out = []
    for part in (hand, face):
        step = np.linalg.norm(np.diff(part, axis=0), axis=-1).mean(axis=1)
        out.append(np.concatenate([step[:1], step]) if len(step) else np.zeros(feat.shape[0]))
    return out[0], out[1]


def lagged(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """ρ(a_t, b_{t+lag})."""
    T = len(a)
    return spearman(a[: T - lag], b[lag:]) if lag >= 0 else spearman(a[-lag:], b[: T + lag])


def stats(v) -> dict:
    v = np.array([x for x in v if np.isfinite(x)])
    if not len(v):
        return {"n": 0}
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = whole split")
    ap.add_argument("--min-frames", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import yaml
    from torch.utils.data import DataLoader

    sys.path[:0] = [PHASE2, os.path.join(PHASE2, "scripts")]
    import slt_reporting as R
    from analyze_delta import layer_deltas
    from src.data import PoseTextDataset, make_collate
    from src.models import build_model

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)
    ds_name = data_cfg["dataset"]
    ds = PoseTextDataset(args.cache_dir, ds_name, args.split, os.path.join(args.cache_dir, ds_name, "spm.model"),
                         model_cfg.get("max_tgt_len", 64), 512, augment=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if args.checkpoint:
        if model_cfg["arm"] == "mamba_padfix":
            sys.exit("mamba_padfix has no horizon banks; pick a banked arm")
        model = build_model(model_cfg, vocab_size=ds.sp.vocab_size(), pad_id=ds.pad_id).to(device).eval()
        R.load_checkpoint(args.checkpoint, model=model, map_location=device)

    lag_rho = {lag: [] for lag in LAGS}
    peak_lag = []
    head_hand, head_face, head_hand_sh, head_face_sh = {}, {}, {}, {}
    rng = np.random.default_rng(0)
    n_seen = 0
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=make_collate(ds.pad_id))
    for batch in loader:
        feat = batch.src[0].numpy()
        T = feat.shape[0]
        if T < args.min_frames:
            continue
        hand, face = speeds(feat)
        per_lag = [lagged(hand, face, lag) for lag in LAGS]
        for lag, r in zip(LAGS, per_lag):
            lag_rho[lag].append(r)
        if np.isfinite(per_lag).any():
            peak_lag.append(LAGS[int(np.nanargmax(per_lag))])

        if model is not None:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                deltas = layer_deltas(model, batch.src.to(device))
            for li, d in enumerate(deltas):
                d = d.float().cpu().numpy()  # (T, nheads)
                for h in range(d.shape[1]):
                    s = d[:, h]
                    sh = rng.permutation(s)
                    head_hand.setdefault((li, h), []).append(partial_spearman(s, hand, face))
                    head_face.setdefault((li, h), []).append(partial_spearman(s, face, hand))
                    head_hand_sh.setdefault((li, h), []).append(partial_spearman(sh, hand, face))
                    head_face_sh.setdefault((li, h), []).append(partial_spearman(sh, face, hand))
        n_seen += 1
        if args.max_clips and n_seen >= args.max_clips:
            break

    pk = np.array(peak_lag)
    summary = {
        "split": args.split, "clips": n_seen,
        "part1_hand_vs_face": {
            "rho_by_lag": {str(l): stats(v).get("mean") for l, v in lag_rho.items()},
            "rho_lag0": stats(lag_rho[0]),
            "peak_lag_median": float(np.median(pk)) if len(pk) else None,
            "frac_peak_within_2_frames": round(float((np.abs(pk) <= 2).mean()), 4) if len(pk) else None,
            "frac_peak_beyond_8_frames": round(float((np.abs(pk) > 8).mean()), 4) if len(pk) else None,
        },
    }
    lines = [f"{n_seen} clips ({args.split})",
             f"hand vs non-rigid face speed: ρ at lag 0 = {summary['part1_hand_vs_face']['rho_lag0']}",
             f"peak lag median {summary['part1_hand_vs_face']['peak_lag_median']}, "
             f"|peak| <= 2: {summary['part1_hand_vs_face']['frac_peak_within_2_frames']}, "
             f"|peak| > 8: {summary['part1_hand_vs_face']['frac_peak_beyond_8_frames']}"]

    if model is not None:
        heads = {}
        for key in sorted(head_hand):
            hh, hf = stats(head_hand[key]), stats(head_face[key])
            heads[f"L{key[0]}/h{key[1]}"] = {
                "hands_given_face": hh, "face_given_hands": hf,
                "shuffled_hands": stats(head_hand_sh[key]).get("mean"),
                "shuffled_face": stats(head_face_sh[key]).get("mean")}
        hm = np.array([v["hands_given_face"].get("mean", np.nan) for v in heads.values()])
        fm = np.array([v["face_given_hands"].get("mean", np.nan) for v in heads.values()])
        sh_sd = float(np.nanstd([v["shuffled_face"] for v in heads.values()] + [v["shuffled_hands"] for v in heads.values()]))
        face_heads = [k for k, v in heads.items() if v["face_given_hands"].get("mean", 0) > max(0.05, v["hands_given_face"].get("mean", 0))]
        summary["part2_heads"] = heads
        summary["part2_summary"] = {
            "corr_across_heads_hands_vs_face": round(float(np.corrcoef(hm, fm)[0, 1]), 4),
            "max_hands_given_face": round(float(np.nanmax(hm)), 4),
            "max_face_given_hands": round(float(np.nanmax(fm)), 4),
            "shuffled_sd_of_head_means": round(sh_sd, 4),
            "face_preferring_heads": face_heads,
        }
        lines += [f"heads: max ρ(Δ,hands|face) {np.nanmax(hm):.3f}, max ρ(Δ,face|hands) {np.nanmax(fm):.3f}, "
                  f"shuffled sd {sh_sd:.3f}",
                  f"across-head corr(hands, face) {summary['part2_summary']['corr_across_heads_hands_vs_face']}, "
                  f"face-preferring heads: {face_heads or 'none'}"]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e4_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines))
    print(f"wrote {args.out}/e4_summary.json")


if __name__ == "__main__":
    main()
