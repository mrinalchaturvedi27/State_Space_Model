"""Post-hoc quantization of the decoder memory: quality vs bytes, no training.

Loads one trained checkpoint, then for each setting quantizes the encoder memory the decoder
cross-attends to (after pooling, for pooled arms) and decodes a split. Storing this memory
(d_model = 512 values per token) is already 6x cheaper than caching cross-attention K/V for
3 decoder layers (3 x 2 x 512), so bytes here = what a streaming/on-device system would keep.

Settings: "fp16" (reference: memory rounded to fp16, 16 bits/value), "<method>:<bits>" with
method in turbo | turbo_prod | naive (see quant/turboquant.py).

Writes <out>/summary.csv (one row per setting, appended) and <out>/<split>_<setting>_<decode>.csv
predictions. Default <out> = <run dir of the checkpoint>/quant/.

  python quant/quant_eval.py --data configs/data/isign.yaml \\
      --model configs/model/mamba_pool_avg.yaml \\
      --checkpoint results/isign/mamba_pool_avg/lr0.0003_s42/checkpoints/best.pt \\
      --cache-dir ../cache --settings fp16,turbo:4,turbo:3,turbo:2,naive:4,naive:3,naive:2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import slt_reporting as R  # noqa: E402
from src.data import TokenBudgetBatchSampler, make_collate, make_dataset  # noqa: E402
from src.evaluate import beam_search_decode, strip_specials  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import greedy_decode, ids_to_text  # noqa: E402
from turboquant import VectorQuantizer, quantize_memory  # noqa: E402


class QuantizedMemoryModel(nn.Module):
    """Wraps a trained model: encode() returns quantized memory; decode() is unchanged."""

    def __init__(self, model: nn.Module, quantizer: VectorQuantizer | None, fp16: bool):
        super().__init__()
        self.model, self.quantizer, self.fp16 = model, quantizer, fp16
        self.reset()

    def reset(self):
        self.tokens = self.clips = 0
        self.err = self.energy = 0.0
        self.cos_sum = 0.0

    def encode(self, src, src_key_padding_mask=None):
        memory, mem_pad = self.model.encode(src, src_key_padding_mask)
        if self.fp16:
            out = memory.half().to(memory.dtype)
        else:
            out = quantize_memory(memory, mem_pad, self.quantizer)
        valid = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device) \
            if mem_pad is None else ~mem_pad
        a, b = memory[valid].float(), out[valid].float()
        self.err += float(((a - b) ** 2).sum())
        self.energy += float((a ** 2).sum())
        self.cos_sum += float(nn.functional.cosine_similarity(a, b, dim=-1).sum())
        self.tokens += int(valid.sum())
        self.clips += memory.size(0)
        return out, mem_pad

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)


def parse_setting(s: str, d: int):
    if s == "fp16":
        return None, True, 16 * d
    method, bits = s.split(":")
    q = VectorQuantizer(method, int(bits), d)
    return q, False, q.bits_per_vector()


@torch.no_grad()
def decode_greedy(model, ds, indices, device, max_tokens, max_new, num_workers=2):
    sub = Subset(ds, indices)
    nf = ds.n_frames_array()[indices]
    sampler = TokenBudgetBatchSampler(nf, max_tokens, seed=0, shuffle=False)
    loader = DataLoader(sub, batch_sampler=sampler, collate_fn=make_collate(ds.pad_id),
                        num_workers=num_workers)
    uids, preds, refs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            ids = greedy_decode(model, batch.src, batch.src_key_padding_mask,
                                ds.bos_id, ds.eos_id, ds.pad_id, max_new)
        uids += batch.uids
        preds += ids_to_text(ds.sp, ids, ds.bos_id, ds.eos_id, ds.pad_id)
        refs += ids_to_text(ds.sp, batch.tgt_out, ds.bos_id, ds.eos_id, ds.pad_id)
    return uids, preds, refs


@torch.no_grad()
def decode_beam(model, ds, indices, device, beam, lp, max_new):
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, collate_fn=make_collate(ds.pad_id))
    uids, preds, refs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            seq = beam_search_decode(model, batch.src, batch.src_key_padding_mask, ds.bos_id, ds.eos_id,
                                     ds.pad_id, beam_size=beam, length_penalty=lp, max_new_tokens=max_new)
        uids.append(batch.uids[0])
        preds.append(ds.sp.decode(strip_specials(seq.tolist(), ds.bos_id, ds.eos_id, ds.pad_id)))
        refs.append(ds.sp.decode(strip_specials(batch.tgt_out[0].tolist(), ds.bos_id, ds.eos_id, ds.pad_id)))
    return uids, preds, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--settings", default="fp16,turbo:4,turbo:3,turbo:2,naive:4,naive:3,naive:2,turbo_prod:3")
    ap.add_argument("--decode", choices=("greedy", "beam"), default="greedy")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=24576, help="greedy batching budget (frames)")
    ap.add_argument("--max-clips", type=int, default=0, help="evenly spaced subset; 0 = whole split")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = data_cfg["dataset"]
    max_new = model_cfg.get("max_tgt_len", 64)
    ds = make_dataset(args.cache_dir, name, args.split, os.path.join(args.cache_dir, name, "spm.model"),
                      max_new, 512, augment=False, model_cfg=model_cfg)
    model = build_model(model_cfg, vocab_size=ds.sp.vocab_size(), pad_id=ds.pad_id).to(device).eval()
    ckpt = R.load_checkpoint(args.checkpoint, model=model, map_location=device)

    n = len(ds)
    indices = np.arange(n) if not args.max_clips or args.max_clips >= n else \
        np.unique(np.linspace(0, n - 1, args.max_clips).round().astype(int))
    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))), "quant")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    d = model_cfg["d_model"]
    tag = model_cfg.get("tag", model_cfg["arm"])
    print(f"{tag}: {args.checkpoint} (epoch {ckpt.get('epoch')}), {args.split} n={len(indices)}, "
          f"decode={args.decode}, device={device}")

    for setting in args.settings.split(","):
        setting = setting.strip()
        q, fp16, bits_vec = parse_setting(setting, d)
        wrapped = QuantizedMemoryModel(model, q, fp16).eval()
        t0 = time.time()
        if args.decode == "greedy":
            uids, preds, refs = decode_greedy(wrapped, ds, indices, device, args.max_tokens, max_new,
                                              args.num_workers)
        else:
            uids, preds, refs = decode_beam(wrapped, ds, indices, device, args.beam_size,
                                            args.length_penalty, max_new)
        m = R.compute_metrics(preds, refs)
        mem_tokens = wrapped.tokens / max(1, wrapped.clips)
        row = {
            "tag": tag, "checkpoint": args.checkpoint, "split": args.split, "n": len(uids),
            "decode": args.decode if args.decode == "greedy" else f"beam{args.beam_size}",
            "setting": setting, "bits_per_value": round(bits_vec / d, 4),
            "mem_tokens_per_clip": round(mem_tokens, 2),
            "bytes_per_clip": round(mem_tokens * bits_vec / 8, 1),
            "rel_mse": round(wrapped.err / max(wrapped.energy, 1e-12), 6),
            "mean_cos": round(wrapped.cos_sum / max(1, wrapped.tokens), 6),
            "chrF2": round(m["corpus_chrF2"], 3), "BLEU4": round(m["corpus_BLEU-4"], 3),
            "seconds": round(time.time() - t0, 1),
        }
        R.save_predictions(uids, preds, refs, os.path.join(out_dir, f"{args.split}_{setting.replace(':', 'b')}_{row['decode']}.csv"))
        new = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  {setting:13s} {row['bits_per_value']:6.2f} b/val  {row['bytes_per_clip']:9.1f} B/clip  "
              f"relMSE {row['rel_mse']:.4f}  cos {row['mean_cos']:.4f}  chrF2 {row['chrF2']:.2f}  "
              f"BLEU4 {row['BLEU4']:.2f}  ({row['seconds']:.0f}s)")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
