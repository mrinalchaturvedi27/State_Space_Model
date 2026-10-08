"""Diagnostic step 1: provenance audit of every trained run (CPU; reads only pushed result files).

For each metrics.xlsx under results/, results_f3/, phase 2/results/ and phase 3/results/finetune/:
dataset, model (tag/arm), seed, schedule (default vs short-warmup), pretrained-encoder init, epochs run and
best epoch, every logged test decode (decoding setting, BLEU-4, chrF2), and whether the beam-5 prediction
CSV re-scores to the workbook's length-penalty-1.0 numbers. Flags:
  CSV_MISMATCH   test_best_beam5.csv does not reproduce the lp1.0 test row (|delta chrF2| > 0.05)
  MULTI_DECODE   several different test decodes logged (e.g. a length-penalty sweep on test)
  NO_TEST        no test decode logged
  STOPPED_EARLY  best epoch < 10 or fewer than 15 epochs run
Writes AUDIT_runs.csv (repo root by default) and prints the flagged runs.

  python "phase 3/scripts/diag_audit.py" --root . --out AUDIT_runs.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd
import sacrebleu

ROOTS = ("results", "results_f3", "phase 2/results", "phase 3/results/finetune")


def run_info(path: str, root: str) -> dict:
    run_dir = os.path.dirname(os.path.dirname(path))
    rel = os.path.relpath(run_dir, root)
    m = re.search(r"(.*)/([^/]+)/([^/]+)/lr([\d.e-]+)_s(\d+)$", rel)
    base, dataset, model, lr, seed = m.groups() if m else (rel, "?", "?", "?", "?")
    x = pd.ExcelFile(path)
    cfg = dict(zip(*[x.parse("run_config")[c].astype(str) for c in ("key", "value")])) if "run_config" in x.sheet_names else {}
    e = x.parse("epoch_metrics") if "epoch_metrics" in x.sheet_names else pd.DataFrame()
    t = x.parse("test_final") if "test_final" in x.sheet_names else pd.DataFrame()
    row = {"run": rel, "group": base, "dataset": dataset, "model": model, "lr": lr, "seed": seed,
           "schedule": "short-warmup" if cfg.get("train.warmup_ratio", "nan") not in ("nan", "", "None") else "default",
           "init_encoder": os.path.basename(os.path.dirname(cfg.get("init_encoder", ""))) if cfg.get("init_encoder", "") not in ("", "nan") else "",
           "epochs_run": len(e), "best_epoch": int(e.loc[e.val_corpus_chrF2.idxmax(), "epoch"]) if len(e) else None,
           "best_val_chrF2": round(float(e.val_corpus_chrF2.max()), 3) if len(e) else None}
    decs = t["decoding"].astype(str).tolist() if "decoding" in t else ["?"] * len(t)
    row["test_decodes"] = ";".join(sorted(set(decs)))
    lp1 = t[t["decoding"].astype(str).str.contains("lp1.0")] if "decoding" in t else t
    if len(lp1):
        row["test_chrF2_lp1"] = round(float(lp1.iloc[-1]["corpus_chrF2"]), 3)
        row["test_BLEU4_lp1"] = round(float(lp1.iloc[-1]["corpus_BLEU-4"]), 3)
    csvp = os.path.join(run_dir, "predictions", "test_best_beam5.csv")
    if os.path.exists(csvp):
        d = pd.read_csv(csvp, keep_default_na=False)
        p, r = d.prediction.astype(str).tolist(), [d.ground_truth.astype(str).tolist()]
        row["csv_chrF2"] = round(sacrebleu.corpus_chrf(p, r).score, 3)
        row["csv_BLEU4"] = round(sacrebleu.corpus_bleu(p, r).score, 3)
        row["csv_n"] = len(d)
    flags = []
    if not len(t):
        flags.append("NO_TEST")
    if len(set(decs)) > 1:
        flags.append("MULTI_DECODE")
    if "csv_chrF2" in row and "test_chrF2_lp1" in row and abs(row["csv_chrF2"] - row["test_chrF2_lp1"]) > 0.05:
        flags.append("CSV_MISMATCH")
    if len(e) and (row["best_epoch"] < 10 or len(e) < 15):
        flags.append("STOPPED_EARLY")
    row["flags"] = " ".join(flags)
    row["nbest"] = os.path.exists(os.path.join(run_dir, "nbest", "test.csv"))
    row["lp_tune"] = os.path.exists(os.path.join(run_dir, "lp_tune", "chosen.csv"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="AUDIT_runs.csv")
    args = ap.parse_args()
    paths = []
    for r in ROOTS:
        paths += [p for p in glob.glob(os.path.join(args.root, r, "**", "results", "metrics.xlsx"), recursive=True)
                  if "smoke" not in p and "/lr" in p]
    rows = [run_info(p, args.root) for p in sorted(paths)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(f"{len(df)} runs audited -> {args.out}")
    print(df.groupby(["dataset", "group"]).size().to_string())
    flagged = df[df["flags"] != ""]
    print(f"\n{len(flagged)} flagged runs:")
    print(flagged[["run", "schedule", "epochs_run", "best_epoch", "test_decodes", "test_chrF2_lp1", "csv_chrF2", "flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
