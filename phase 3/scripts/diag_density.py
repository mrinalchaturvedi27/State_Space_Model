"""Diagnostic 3: what does input length measure? Sampling density and order (dev data only).

For a fixed sample of validation clips with <= --max-frames frames (default 256, so doubling never
reaches the 512-frame cap), the clip's full time span is kept and only the number of observations
changes; order-destroying transforms are separate stress tests:
  d0.5 / d1 / d2      half / original / double sampling density (linear interpolation over the clip)
  reverse             frames in reverse order
  shuffle_blocks16    16-frame blocks permuted (local motion kept, global order destroyed)
  shuffle_frames      every frame permuted
Transformer only, also d0.5_timePE / d2_timePE: positional codes follow original time (i / density),
separating "more frames" from "positions the model never saw".
Reported per condition: reference token NLL and content-word log-likelihood (teacher forcing), and
greedy-decode corpus chrF2 / BLEU-4 on the same clips. Compare how the two encoders degrade.

  python scripts/diag_density.py --data configs/data/isign.yaml --model configs/model/transformer.yaml \\
      --checkpoint ../results/isign/transformer/lr0.0003_s42/checkpoints/best.pt --cache-dir ../cache \\
      --out results/diagnostics/density/isign_transformer
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402,F401
from p3.diag import density, reverse, shuffle_blocks, shuffle_frames, teacher_forced, time_positions, word_scores  # noqa: E402
from p3.loading import load  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402

from src.data import make_collate  # noqa: E402
from src.train import greedy_decode, ids_to_text  # noqa: E402

BASE = ["d0.5", "d1", "d2", "reverse", "shuffle_blocks16", "shuffle_frames"]


def transform(cond, feat, rng):
    if cond.startswith("d"):
        return density(feat, float(cond[1:].split("_")[0]))
    if cond == "reverse":
        return reverse(feat)
    if cond == "shuffle_blocks16":
        return shuffle_blocks(feat, 16, rng)
    if cond == "shuffle_frames":
        return shuffle_frames(feat, rng)
    raise ValueError(cond)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ds, data_cfg, model_cfg = load(args.data, args.model, args.checkpoint, args.cache_dir, device)
    assert model_cfg.get("context_clips") is None, "context models are not supported here"
    lang = data_cfg.get("lang", "en")
    sample_file = args.sample_file or os.path.join(os.path.dirname(args.out.rstrip("/")), f"sample_density_{data_cfg['dataset']}.csv")
    if os.path.exists(sample_file):
        S = pd.read_csv(sample_file)
        print(f"reusing fixed sample {sample_file} ({len(S)} clips)")
    else:
        nf = ds.n_frames_array()
        pool = np.nonzero(nf <= args.max_frames)[0]
        pick = np.sort(np.random.default_rng(args.seed).choice(pool, size=min(args.n, len(pool)), replace=False))
        S = pd.DataFrame({"idx": pick, "uid": ds.index["uid"].astype(str).to_numpy()[pick], "frames": nf[pick]})
        os.makedirs(os.path.dirname(sample_file) or ".", exist_ok=True)
        S.to_csv(sample_file, index=False)
        print(f"fixed sample written: {sample_file} ({len(S)} clips, <= {args.max_frames} frames)")

    conds = list(BASE)
    if model_cfg["arm"] in ("transformer", "transformer_bow"):
        conds += ["d0.5_timePE", "d2_timePE"]
    collate = make_collate(ds.pad_id)
    summ = []
    for cond in conds:
        rng = np.random.default_rng([args.seed, len(cond)])
        ctx = time_positions(model, float(cond[1:].split("_")[0])) if cond.endswith("_timePE") else contextlib.nullcontext()
        ll = ntok = cll = ncon = frames = 0
        hyps, refs = [], []
        with ctx:
            order = S.sort_values("frames").reset_index(drop=True)
            for s in range(0, len(order), args.batch):
                chunk = order.iloc[s:s + args.batch]
                items = []
                for r in chunk.itertuples():
                    it = ds[int(r.idx)]
                    items.append({"uid": it["uid"], "feat": transform(cond, it["feat"], rng), "ids": it["ids"]})
                frames += sum(i["feat"].shape[0] for i in items)
                b = collate(items)
                for ids, lps in teacher_forced(model, b, device):
                    ws = word_scores(ds.sp, ids, lps, lang)
                    ll, ntok = ll + float(np.sum(lps)), ntok + len(lps)
                    cw = [w for w in ws if w[2]]
                    cll, ncon = cll + sum(w[1] for w in cw), ncon + len(cw)
                bb = b.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = greedy_decode(model, bb.src, bb.src_key_padding_mask, ds.bos_id, ds.eos_id, ds.pad_id,
                                        model_cfg.get("max_tgt_len", 64))
                hyps += ids_to_text(ds.sp, out, ds.bos_id, ds.eos_id, ds.pad_id)
                refs += ids_to_text(ds.sp, bb.tgt_out, ds.bos_id, ds.eos_id, ds.pad_id)
        summ.append({"condition": cond, "n": len(S), "mean_frames": round(frames / len(S), 1),
                     "token_nll": round(-ll / ntok, 4), "content_word_ll": round(cll / max(1, ncon), 4),
                     "greedy_chrF2": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
                     "greedy_BLEU4": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)})
        print(summ[-1])
    os.makedirs(args.out, exist_ok=True)
    Sm = pd.DataFrame(summ)
    d1 = Sm.set_index("condition").loc["d1"]
    Sm["delta_chrF2_vs_d1"] = (Sm.greedy_chrF2 - d1.greedy_chrF2).round(2)
    Sm["delta_token_nll_vs_d1"] = (Sm.token_nll - d1.token_nll).round(4)
    Sm.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(Sm.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
