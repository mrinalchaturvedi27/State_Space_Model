"""Training loop for the primary A1 (Transformer) vs A2 (Mamba) comparison, BENCHMARK_PLAN.md §4.

Usage:
  python src/train.py --data configs/data/isign.yaml --model configs/model/transformer.yaml \
      --lr 3e-4 --seed 42
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slt_reporting as R  # noqa: E402
from src.data import PoseTextDataset, TokenBudgetBatchSampler, make_collate  # noqa: E402
from src.models import build_model  # noqa: E402


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


@torch.no_grad()
def greedy_decode(model, src, src_key_padding_mask, bos_id, eos_id, pad_id, max_new_tokens=64):
    memory = model.encode(src, src_key_padding_mask)
    B, device = src.size(0), src.device
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits = model.decode(ys, memory, memory_key_padding_mask=src_key_padding_mask)
        next_tok = logits[:, -1].argmax(-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_id)
        if bool(finished.all()):
            break
    return ys


def ids_to_text(sp, ids: torch.Tensor, bos_id: int, eos_id: int, pad_id: int) -> list[str]:
    out = []
    for row in ids.tolist():
        toks = []
        for t in row:
            if t == bos_id:
                continue
            if t == eos_id:
                break
            if t == pad_id:
                continue
            toks.append(t)
        out.append(sp.decode(toks))
    return out


@torch.no_grad()
def run_eval(model, loader, sp, device, bos_id, eos_id, pad_id, max_new_tokens=64):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_uids, all_preds, all_refs, all_nframes = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.compute_loss(batch.src, batch.tgt_in, batch.tgt_out,
                                      batch.src_key_padding_mask, batch.tgt_key_padding_mask)
            pred_ids = greedy_decode(model, batch.src, batch.src_key_padding_mask,
                                     bos_id, eos_id, pad_id, max_new_tokens)
        total_loss += loss.item()
        n_batches += 1
        all_uids += batch.uids
        all_preds += ids_to_text(sp, pred_ids, bos_id, eos_id, pad_id)
        all_refs += ids_to_text(sp, batch.tgt_out, bos_id, eos_id, pad_id)
        all_nframes += batch.n_frames.tolist()
    metrics = R.compute_metrics(all_preds, all_refs)
    return total_loss / max(1, n_batches), metrics, all_uids, all_preds, all_refs, all_nframes


def build_loaders(args, data_cfg, model_cfg, train_cfg, seed):
    dataset_name = data_cfg["dataset"]
    spm_model = os.path.join(args.cache_dir, dataset_name, "spm.model")
    max_tgt_len = model_cfg.get("max_tgt_len", 64)

    train_ds = PoseTextDataset(args.cache_dir, dataset_name, "train", spm_model, max_tgt_len,
                              train_cfg["t_max"], augment=True, seed=seed)
    val_ds = PoseTextDataset(args.cache_dir, dataset_name, "val", spm_model, max_tgt_len,
                            train_cfg["t_max"], augment=False)

    collate = make_collate(train_ds.pad_id)
    train_sampler = TokenBudgetBatchSampler(train_ds.n_frames_array(), train_cfg["max_tokens"],
                                            seed=seed, shuffle=True)
    val_sampler = TokenBudgetBatchSampler(val_ds.n_frames_array(), train_cfg["max_tokens"],
                                          seed=0, shuffle=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                              num_workers=train_cfg.get("num_workers", 8), pin_memory=True,
                              persistent_workers=train_cfg.get("num_workers", 8) > 0)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate,
                            num_workers=2, pin_memory=True)
    return train_ds, val_ds, train_loader, val_loader, train_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default=os.path.join(os.path.dirname(__file__), "..", "configs", "train", "base.yaml"))
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="hard cap on optimizer steps, for smoke tests")
    ap.add_argument("--wandb-project", default="ARR-SSM-vs-TF-SLT")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train) as f:
        train_cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    dataset_name, arm = data_cfg["dataset"], model_cfg["arm"]
    train_ds, val_ds, train_loader, val_loader, train_sampler = build_loaders(
        args, data_cfg, model_cfg, train_cfg, args.seed)

    vocab_size = train_ds.sp.vocab_size()
    model = build_model(model_cfg, vocab_size=vocab_size, pad_id=train_ds.pad_id).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    decay, no_decay = model.no_decay_params()
    optimizer = AdamW(
        [{"params": decay, "weight_decay": train_cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=tuple(train_cfg["betas"]), eps=float(train_cfg["eps"]))

    steps_per_epoch = max(1, len(train_sampler) // train_cfg.get("grad_accum", 1))
    max_epochs = args.max_epochs or train_cfg["max_epochs"]
    total_steps = args.max_steps or steps_per_epoch * max_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=functools.partial(
        cosine_with_warmup, warmup_steps=train_cfg["warmup_steps"], total_steps=total_steps,
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.1)))

    run_name = f"{dataset_name}-{arm}-lr{args.lr:g}-s{args.seed}"
    run_dir = R.run_dir(args.results_dir, dataset_name, arm, args.lr, args.seed)
    config = {**{f"data.{k}": v for k, v in data_cfg.items()},
             **{f"model.{k}": v for k, v in model_cfg.items()},
             **{f"train.{k}": v for k, v in train_cfg.items()},
             "lr": args.lr, "seed": args.seed, "run_name": run_name, "params": n_params}
    xl = R.ExcelReporter(path=os.path.join(run_dir, "results", "metrics.xlsx"), config=config,
                         select_on="val_corpus_chrF2", select_mode="max")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=config)

    grad_accum = train_cfg.get("grad_accum", 1)
    best_metric, best_epoch, patience_ctr = -1.0, -1, 0
    global_step, stop = 0, False

    print(f"[{run_name}] params={n_params:,} steps_per_epoch={steps_per_epoch} "
         f"total_steps={total_steps} run_dir={run_dir}")

    for epoch in range(max_epochs):
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss, n_steps, t0 = 0.0, 0, time.time()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.compute_loss(batch.src, batch.tgt_in, batch.tgt_out,
                                          batch.src_key_padding_mask, batch.tgt_key_padding_mask)
            (loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
            epoch_loss += loss.item()
            n_steps += 1
            if stop:
                break
        train_loss = epoch_loss / max(1, n_steps)

        val_loss, val_metrics, val_uids, val_preds, val_refs, val_nf = run_eval(
            model, val_loader, train_ds.sp, device, train_ds.bos_id, train_ds.eos_id, train_ds.pad_id,
            max_new_tokens=model_cfg.get("max_tgt_len", 64))

        xl.log_epoch(epoch, train_loss, val_loss, val_metrics)
        xl.log_predictions("val", val_uids, val_preds, val_refs, epoch=epoch, extra={"n_frames": val_nf})
        print(f"[{run_name}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
             f"val_chrF2={val_metrics['corpus_chrF2']:.2f} val_BLEU4={val_metrics['corpus_BLEU-4']:.2f} "
             f"({time.time() - t0:.0f}s)")

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}},
                     step=global_step)

        is_best = val_metrics["corpus_chrF2"] > best_metric
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        R.save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, scheduler,
                          epoch=epoch, global_step=global_step, metrics=val_metrics, config=config,
                          is_best=is_best, best_path=os.path.join(ckpt_dir, "best.pt") if is_best else None)
        if epoch % 10 == 0:
            R.save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch}.pt"), model, optimizer, scheduler,
                              epoch=epoch, global_step=global_step, metrics=val_metrics, config=config)

        if is_best:
            best_metric, best_epoch, patience_ctr = val_metrics["corpus_chrF2"], epoch, 0
        else:
            patience_ctr += 1
        if patience_ctr >= train_cfg.get("patience", 8) or stop:
            print(f"[{run_name}] stopping at epoch {epoch} (best epoch {best_epoch}, chrF2={best_metric:.2f})")
            break

    print(f"[{run_name}] training complete. best epoch {best_epoch}, val_corpus_chrF2={best_metric:.2f}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
