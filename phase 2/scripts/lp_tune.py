"""Length penalty chosen on validation, applied once to test -- no training.

Beam search here ranks hypotheses by their raw log-probability while searching; the length
penalty only picks among the finished hypotheses (score / len**lp, src/evaluate.py). So ONE
beam decode per split yields the output for every length penalty: this script keeps each
example's finished hypotheses and rescores them for the whole grid.

Selection is on VALIDATION only: the lp with the best val corpus BLEU-4 (and, separately, the
best val chrF2) is applied to test. Test scores at the other lps are written too, for
transparency, but are never used to choose.

Writes <run dir>/lp_tune/: summary.csv (split, lp, chrF2, BLEU-4), chosen.csv (the val-chosen lp
per criterion and its test scores), and test predictions at each chosen lp.

  python scripts/lp_tune.py --data configs/data/isign.yaml --model configs/model/mamba.yaml \\
      --checkpoint ../results/isign/mamba/lr0.0003_s13/checkpoints/best.pt --cache-dir ../cache
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slt_reporting as R  # noqa: E402
from src.data import make_collate, make_dataset  # noqa: E402
from src.evaluate import strip_specials  # noqa: E402
from src.models import build_model  # noqa: E402

LP_GRID = [round(0.6 + 0.1 * i, 1) for i in range(15)]  # 0.6 ... 2.0


@torch.no_grad()
def beam_search_all_lp(model, src, src_key_padding_mask, bos_id, eos_id, pad_id,
                       beam_size=5, max_new_tokens=64, lps=LP_GRID):
    """src/evaluate.py:beam_search_decode for a single model, returning {lp: sequence} for
    every lp from one search. The search below is that function's, line for line; only the
    final choice among finished hypotheses is repeated per lp."""
    device = src.device
    raw_memory, mem_pad = model.encode(src, src_key_padding_mask)
    memory = raw_memory.expand(beam_size, -1, -1)
    mem_pad = mem_pad.expand(beam_size, -1) if mem_pad is not None else None

    beams = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    beam_scores = torch.full((beam_size,), float("-inf"), device=device)
    beam_scores[0] = 0.0
    finished = []  # (sequence, raw score)

    for _ in range(max_new_tokens):
        logits = model.decode(beams, memory, memory_key_padding_mask=mem_pad)
        log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
        vocab_size = log_probs.size(-1)
        cand = (beam_scores.unsqueeze(1) + log_probs).view(-1)
        topk_scores, topk_idx = cand.topk(beams.size(0))
        beam_idx = torch.div(topk_idx, vocab_size, rounding_mode="floor")
        tok_idx = topk_idx % vocab_size
        beams = torch.cat([beams[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        beam_scores = topk_scores
        is_eos = tok_idx == eos_id
        if is_eos.any():
            for i in torch.nonzero(is_eos).flatten().tolist():
                finished.append((beams[i].clone(), beam_scores[i].item()))
            keep = ~is_eos
            if keep.sum() == 0:
                beams = beams[:0]
                break
            beams, beam_scores = beams[keep], beam_scores[keep]
            memory = memory[: keep.sum()]
            mem_pad = mem_pad[: keep.sum()] if mem_pad is not None else None
        if beams.size(0) == 0:
            break
    for i in range(beams.size(0)):
        finished.append((beams[i], beam_scores[i].item()))

    out = {}
    for lp in lps:
        if not finished:
            out[lp] = torch.tensor([bos_id, eos_id], device=device)
            continue
        best = max(range(len(finished)), key=lambda i: finished[i][1] / (finished[i][0].size(0) ** lp))
        out[lp] = finished[best][0]
    return out


def decode_split(model, ds, device, beam_size, max_new, lps, max_clips=0):
    idx = list(range(len(ds)))
    if max_clips and max_clips < len(idx):
        idx = idx[:: max(1, len(idx) // max_clips)][:max_clips]
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, collate_fn=make_collate(ds.pad_id))
    uids, refs, preds = [], [], {lp: [] for lp in lps}
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            seqs = beam_search_all_lp(model, batch.src, batch.src_key_padding_mask, ds.bos_id, ds.eos_id,
                                      ds.pad_id, beam_size=beam_size, max_new_tokens=max_new, lps=lps)
        uids.append(batch.uids[0])
        refs.append(ds.sp.decode(strip_specials(batch.tgt_out[0].tolist(), ds.bos_id, ds.eos_id, ds.pad_id)))
        for lp in lps:
            preds[lp].append(ds.sp.decode(strip_specials(seqs[lp].tolist(), ds.bos_id, ds.eos_id, ds.pad_id)))
    return uids, refs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-clips", type=int, default=0, help="subset per split, for a quick check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = data_cfg["dataset"]
    max_new = model_cfg.get("max_tgt_len", 64)
    spm = os.path.join(args.cache_dir, name, "spm.model")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))), "lp_tune")
    os.makedirs(out_dir, exist_ok=True)

    val = make_dataset(args.cache_dir, name, "val", spm, max_new, 512, augment=False, model_cfg=model_cfg)
    test = make_dataset(args.cache_dir, name, "test", spm, max_new, 512, augment=False, model_cfg=model_cfg)
    model = build_model(model_cfg, vocab_size=val.sp.vocab_size(), pad_id=val.pad_id).to(device).eval()
    ckpt = R.load_checkpoint(args.checkpoint, model=model, map_location=device)
    print(f"{args.checkpoint} (epoch {ckpt.get('epoch')}), lp grid {LP_GRID[0]}-{LP_GRID[-1]}")

    rows, scores = [], {}
    for split, ds in (("val", val), ("test", test)):
        t0 = time.time()
        uids, refs, preds = decode_split(model, ds, device, args.beam_size, max_new, LP_GRID, args.max_clips)
        for lp in LP_GRID:
            m = R.compute_metrics(preds[lp], refs)
            scores[(split, lp)] = (m["corpus_chrF2"], m["corpus_BLEU-4"])
            rows.append({"split": split, "lp": lp, "chrF2": round(m["corpus_chrF2"], 3),
                         "BLEU4": round(m["corpus_BLEU-4"], 3), "n": len(refs)})
        if split == "test":
            test_out = (uids, refs, preds)
        print(f"  {split}: {len(refs)} clips, {time.time() - t0:.0f}s")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    chosen = []
    for criterion, k in (("BLEU4", 1), ("chrF2", 0)):
        lp = max(LP_GRID, key=lambda x: (scores[("val", x)][k], -abs(x - 1.0)))  # ties -> closest to 1.0
        c, b = scores[("test", lp)]
        chosen.append({"criterion": f"val_{criterion}", "lp": lp,
                       "val_chrF2": round(scores[("val", lp)][0], 3), "val_BLEU4": round(scores[("val", lp)][1], 3),
                       "test_chrF2": round(c, 3), "test_BLEU4": round(b, 3),
                       "test_chrF2_lp1.0": round(scores[("test", 1.0)][0], 3),
                       "test_BLEU4_lp1.0": round(scores[("test", 1.0)][1], 3)})
        uids, refs, preds = test_out
        R.save_predictions(uids, preds[lp], refs, os.path.join(out_dir, f"test_lp{lp}_beam{args.beam_size}.csv"))
        print(f"  chosen on val {criterion}: lp={lp} -> test chrF2 {c:.2f} BLEU-4 {b:.2f} "
              f"(lp 1.0: {scores[('test', 1.0)][0]:.2f} / {scores[('test', 1.0)][1]:.2f})")
    with open(os.path.join(out_dir, "chosen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0]))
        w.writeheader()
        w.writerows(chosen)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
