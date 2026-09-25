"""Beam-search evaluation for a trained checkpoint (BENCHMARK_PLAN.md §6): the paper-grade
numbers, as opposed to train.py's fast greedy per-epoch eval.

Usage:
  python src/evaluate.py --data configs/data/isign.yaml --model configs/model/transformer.yaml \
      --checkpoint results/isign/transformer/lr0.0003_s42/checkpoints/best.pt --splits val test
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slt_reporting as R  # noqa: E402
from src.data import PoseTextDataset, make_collate  # noqa: E402
from src.models import build_model  # noqa: E402


class EnsembleModel(torch.nn.Module):
    """Wraps N models and averages their decoder logits for ensemble decoding.
    Each model encodes independently; decode averages log-probs across all models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def encode(self, src, src_key_padding_mask):
        return [m.encode(src, src_key_padding_mask) for m in self.models]

    def decode(self, tgt, memories, **kwargs):
        logits = torch.stack([
            m.decode(tgt, mem, **kwargs)
            for m, mem in zip(self.models, memories)
        ])
        return logits.mean(dim=0)


@torch.no_grad()
def beam_search_decode(model, src, src_key_padding_mask, bos_id, eos_id, pad_id,
                       beam_size=5, length_penalty=1.0, max_new_tokens=64):
    """One example at a time (src: (1, T, D)). Standard beam search with GNMT-style length
    normalization (score / len**length_penalty); no n-gram blocking -- §6's protocol, deliberately.
    Not batched across examples -- simplicity/correctness first, see note in run_split().
    Supports both single models and EnsembleModel (which returns a list of memories)."""
    device = src.device
    is_ensemble = isinstance(model, EnsembleModel)

    raw_memory = model.encode(src, src_key_padding_mask)
    if is_ensemble:
        memory = [m.expand(beam_size, -1, -1) for m in raw_memory]
    else:
        memory = raw_memory.expand(beam_size, -1, -1)
    mem_pad = (src_key_padding_mask.expand(beam_size, -1) if src_key_padding_mask is not None else None)

    beams = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    beam_scores = torch.full((beam_size,), float("-inf"), device=device)
    beam_scores[0] = 0.0
    finished_seqs, finished_scores = [], []

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
                finished_seqs.append(beams[i].clone())
                finished_scores.append(beam_scores[i].item() / (beams[i].size(0) ** length_penalty))
            keep = ~is_eos
            if keep.sum() == 0:
                beams = beams[:0]
                break
            beams, beam_scores = beams[keep], beam_scores[keep]
            if is_ensemble:
                memory = [m[: keep.sum()] for m in memory]
            else:
                memory = memory[: keep.sum()]
            mem_pad = mem_pad[: keep.sum()] if mem_pad is not None else None
        if beams.size(0) == 0:
            break

    for i in range(beams.size(0)):
        finished_seqs.append(beams[i])
        finished_scores.append(beam_scores[i].item() / (beams[i].size(0) ** length_penalty))

    if not finished_seqs:
        return torch.tensor([bos_id, eos_id], device=device)
    best = max(range(len(finished_seqs)), key=lambda i: finished_scores[i])
    return finished_seqs[best]


def strip_specials(ids: list[int], bos_id: int, eos_id: int, pad_id: int) -> list[int]:
    out = []
    for t in ids:
        if t == bos_id or t == pad_id:
            continue
        if t == eos_id:
            break
        out.append(t)
    return out


def load_reporter_state(path: str, select_on: str = "val_corpus_chrF2", select_mode: str = "max") -> R.ExcelReporter:
    """Reconstruct an ExcelReporter's in-memory state from an existing workbook, so evaluate.py's
    flush() extends the training run's metrics.xlsx instead of overwriting the epoch history with
    an empty one (ExcelReporter itself has no load path -- ExcelReporter is write-oriented, built
    incrementally during training; this is the read-back counterpart needed for a second process)."""
    xl = R.ExcelReporter(path=path, select_on=select_on, select_mode=select_mode)
    try:
        cfg_df = pd.read_excel(path, sheet_name="run_config")
        xl.config = dict(zip(cfg_df["key"], cfg_df["value"]))
    except Exception:
        pass
    try:
        xl.history = pd.read_excel(path, sheet_name="epoch_metrics").to_dict("records")
    except Exception:
        pass
    try:
        xl.final = pd.read_excel(path, sheet_name="test_final").to_dict("records")
    except Exception:
        pass
    for split in ("val", "test"):
        try:
            df = pd.read_excel(path, sheet_name=f"pred_{split}")
            xl.predictions[split] = df
            if "uid" in df:
                xl._sample_uids[split] = list(df["uid"].head(xl.n_samples))
        except Exception:
            pass
    try:
        xl.samples = pd.read_excel(path, sheet_name="samples").to_dict("records")
    except Exception:
        pass
    return xl


def length_bucket_report(preds, refs, values, buckets, label):
    print(f"--- length-bucketed quality by {label} ---")
    values = np.asarray(values)
    for lo, hi in buckets:
        mask = (values >= lo) & (values <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bp = [p for p, m in zip(preds, mask) if m]
        br = [r for r, m in zip(refs, mask) if m]
        m = R.compute_metrics(bp, br)
        print(f"  {label}[{lo}-{hi}] n={n}: chrF2={m['corpus_chrF2']:.2f} BLEU-4={m['corpus_BLEU-4']:.2f}")


def run_split(model, ds, device, beam_size, length_penalty, max_new_tokens):
    """Per-example beam search (batch_size=1): simple and unambiguously correct for a first
    implementation. If eval throughput on the full test set (6,069 iSign sentences) turns out
    too slow, the optimization is to batch multiple examples' beams together with shared padding
    -- not attempted here to avoid a subtle batched-beam bug going unnoticed."""
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=make_collate(ds.pad_id))
    uids, preds, refs, n_frames = [], [], [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            seq = beam_search_decode(model, batch.src, batch.src_key_padding_mask,
                                     ds.bos_id, ds.eos_id, ds.pad_id,
                                     beam_size=beam_size, length_penalty=length_penalty,
                                     max_new_tokens=max_new_tokens)
        pred_ids = strip_specials(seq.tolist(), ds.bos_id, ds.eos_id, ds.pad_id)
        ref_ids = strip_specials(batch.tgt_out[0].tolist(), ds.bos_id, ds.eos_id, ds.pad_id)
        uids.append(batch.uids[0])
        preds.append(ds.sp.decode(pred_ids))
        refs.append(ds.sp.decode(ref_ids))
        n_frames.append(int(batch.n_frames[0]))
    return uids, preds, refs, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint, or comma-separated paths for ensemble decoding")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--results-dir", default=None,
                    help="Override output directory (default: inferred from checkpoint path)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--t-max", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=25)
    args = ap.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)

    device = "cuda"
    dataset_name = data_cfg["dataset"]
    max_tgt_len = model_cfg.get("max_tgt_len", 64)
    spm_model = os.path.join(args.cache_dir, dataset_name, "spm.model")

    probe_ds = PoseTextDataset(args.cache_dir, dataset_name, "val", spm_model, max_tgt_len, args.t_max, augment=False)

    ckpt_paths = [p.strip() for p in args.checkpoint.split(",")]
    if len(ckpt_paths) > 1:
        # Ensemble mode
        models = []
        for cp in ckpt_paths:
            m = build_model(model_cfg, vocab_size=probe_ds.sp.vocab_size(), pad_id=probe_ds.pad_id).to(device)
            R.load_checkpoint(cp, model=m, map_location=device)
            m.eval()
            models.append(m)
        model = EnsembleModel(models)
        model.eval()
        ckpt = {"epoch": "ensemble", "global_step": "ensemble"}
        print(f"loaded ENSEMBLE of {len(ckpt_paths)} checkpoints: {ckpt_paths}")
    else:
        model = build_model(model_cfg, vocab_size=probe_ds.sp.vocab_size(), pad_id=probe_ds.pad_id).to(device)
        ckpt = R.load_checkpoint(ckpt_paths[0], model=model, map_location=device)
        model.eval()
        print(f"loaded {ckpt_paths[0]} (epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')})")

    if args.results_dir:
        run_dir = args.results_dir
        os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    else:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_paths[0])))
    xl_path = os.path.join(run_dir, "results", "metrics.xlsx")
    xl = load_reporter_state(xl_path) if os.path.exists(xl_path) else R.ExcelReporter(path=xl_path, select_on="val_corpus_chrF2")
    xl.n_samples = args.n_samples

    for split in args.splits:
        ds = probe_ds if split == "val" else PoseTextDataset(
            args.cache_dir, dataset_name, split, spm_model, max_tgt_len, args.t_max, augment=False)
        uids, preds, refs, n_frames = run_split(model, ds, device, args.beam_size, args.length_penalty, max_tgt_len)
        metrics = R.compute_metrics(preds, refs)

        pred_dir = os.path.join(run_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        R.save_predictions(uids, preds, refs, os.path.join(pred_dir, f"{split}_best_beam{args.beam_size}.csv"),
                           n_frames=n_frames)
        xl.log_predictions(split, uids, preds, refs, epoch=ckpt.get("epoch"), extra={"n_frames": n_frames})


        if split == "test":
            xl.log_final("test", metrics, decoding=f"beam{args.beam_size}_lp{args.length_penalty}",
                        checkpoint_epoch=ckpt.get("epoch"))
            samples_df = pd.DataFrame({"uid": uids, "ground_truth": refs, "model_translation": preds}).head(args.n_samples)
            samples_path = os.path.join(run_dir, "results", "samples.tsv")
            samples_df.to_csv(samples_path, sep="\t", index=False)
            print(f"wrote {samples_path} ({len(samples_df)} fixed examples)")

        print(f"\n=== {split} corpus metrics (beam={args.beam_size}, n={len(preds)}) ===")
        for k in ("corpus_BLEU-1", "corpus_BLEU-4", "corpus_chrF2", "corpus_ROUGE-L", "corpus_METEOR", "corpus_WER"):
            print(f"  {k}: {metrics[k]:.2f}")

        length_bucket_report(preds, refs, n_frames, [(0, 128), (129, 256), (257, 512), (513, 10**9)], "src_frames")
        ref_words = [len(r.split()) for r in refs]
        length_bucket_report(preds, refs, ref_words, [(0, 8), (9, 16), (17, 10**9)], "tgt_words")

    xl.flush()
    print(f"\nupdated {xl_path}")


if __name__ == "__main__":
    main()
