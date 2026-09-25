"""
Reporting utilities for the SSM-vs-Transformer sign language translation benchmark.

Provides the four artifacts the lab expects from every run:
  1. checkpoints            -> save_checkpoint() / load_checkpoint()
  2. val + test predictions -> save_predictions()
  3. an Excel of all metrics-> ExcelReporter  (schema-compatible with metrics_summary.xlsx)
  4. BLEU-1..4, SacreBLEU, ROUGE-L (+ chrF2, METEOR, WER) -> compute_metrics()

Metric policy
-------------
`compute_metrics` returns BOTH:
  * legacy_*  : sentence-level scores averaged over the corpus, replicating the existing
                qwen/gemma/phi scripts exactly, so new numbers stay comparable with the
                metrics_summary.xlsx files already sitting in the project tree.
  * corpus_*  : standard corpus-level SacreBLEU / chrF2 / WER, which is what goes in the paper.
The Excel keeps the legacy columns under their original names and appends the corpus columns,
so old sheets and new sheets can be concatenated without renaming anything.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import sacrebleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

# BLEU column names, kept in the exact order used by the existing metrics_summary.xlsx files.
_LEGACY_KEYS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "SacreBLEU"]
_CORPUS_KEYS = [
    "corpus_BLEU-1", "corpus_BLEU-2", "corpus_BLEU-3", "corpus_BLEU-4",
    "corpus_chrF2", "corpus_ROUGE-L", "corpus_METEOR", "corpus_WER",
]
ALL_METRIC_KEYS = _LEGACY_KEYS + _CORPUS_KEYS


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _word_error_rate(pred: str, ref: str) -> float:
    p, r = pred.split(), ref.split()
    if not r:
        return 1.0 if p else 0.0
    d = np.zeros((len(r) + 1, len(p) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(p) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(p) + 1):
            cost = 0 if r[i - 1] == p[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(p)]) / len(r)


def _meteor(preds: Sequence[str], refs: Sequence[str]) -> float:
    """METEOR needs the nltk wordnet corpus; returns nan if it is unavailable."""
    try:
        from nltk.translate.meteor_score import meteor_score
        return float(np.mean([meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]))
    except Exception:
        return float("nan")


def compute_metrics(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """All metrics for one (predictions, references) pair. Empty pairs are skipped, as in the
    existing scripts, and the number kept is reported as `n_scored`."""
    pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not pairs:
        return {k: 0.0 for k in ALL_METRIC_KEYS} | {"n_scored": 0, "n_empty_pred": len(predictions)}
    preds, refs = [p for p, _ in pairs], [r for _, r in pairs]

    # --- legacy: sentence-level, averaged (matches qwen-3-8bVT.py exactly) ---
    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    acc = {k: 0.0 for k in _LEGACY_KEYS}
    for p, r in zip(preds, refs):
        rt, pt = [r.split()], p.split()
        acc["BLEU-1"] += sentence_bleu(rt, pt, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        acc["BLEU-2"] += sentence_bleu(rt, pt, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        acc["BLEU-3"] += sentence_bleu(rt, pt, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        acc["BLEU-4"] += sentence_bleu(rt, pt, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        acc["ROUGE-L"] += rouge.score(r, p)["rougeL"].fmeasure
        acc["SacreBLEU"] += sacrebleu.sentence_bleu(p, [r]).score
    out = {k: v / len(preds) for k, v in acc.items()}

    # --- corpus level: the paper-grade numbers (all on the 0-100 scale) ---
    for n in (1, 2, 3, 4):
        out[f"corpus_BLEU-{n}"] = sacrebleu.BLEU(max_ngram_order=n).corpus_score(preds, [refs]).score
    out["corpus_chrF2"] = sacrebleu.CHRF(word_order=0, beta=2).corpus_score(preds, [refs]).score
    out["corpus_ROUGE-L"] = 100.0 * np.mean([rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)])
    out["corpus_METEOR"] = 100.0 * _meteor(preds, refs)
    out["corpus_WER"] = 100.0 * float(np.mean([_word_error_rate(p, r) for p, r in zip(preds, refs)]))

    out["n_scored"] = len(preds)
    out["n_empty_pred"] = len(predictions) - len(preds)
    return out


def sacrebleu_signature(predictions: Optional[Sequence[str]] = None,
                        references: Optional[Sequence[str]] = None) -> str:
    """Printed once per run so the BLEU numbers are reproducible by anyone.
    sacrebleu only exposes a signature after a score has been computed, so a trivial pair is
    scored when none is supplied."""
    metric = sacrebleu.BLEU()
    preds = list(predictions) if predictions else ["a"]
    refs = list(references) if references else ["a"]
    metric.corpus_score(preds, [refs])
    return str(metric.get_signature())


# --------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------
def save_predictions(uids: Sequence[str], predictions: Sequence[str], references: Sequence[str],
                     path: str, n_frames: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Writes `video_id,prediction,ground_truth` (the lab's existing schema) plus per-sentence
    diagnostic columns used for the length-bucketed analysis."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    df = pd.DataFrame({"video_id": list(uids), "prediction": list(predictions),
                       "ground_truth": list(references)})
    df["sent_chrF2"] = [sacrebleu.sentence_chrf(p, [r]).score for p, r in zip(df.prediction, df.ground_truth)]
    df["sent_ROUGE-L"] = [rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(df.prediction, df.ground_truth)]
    df["ref_words"] = df.ground_truth.str.split().str.len()
    df["pred_words"] = df.prediction.str.split().str.len()
    if n_frames is not None:
        df["n_frames"] = list(n_frames)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, *,
                    epoch: int = 0, global_step: int = 0, metrics: Optional[Dict] = None,
                    config: Optional[Dict] = None, is_best: bool = False,
                    best_path: Optional[str] = None,
                    max_retries: int = 5, retry_base_delay: float = 2.0) -> str:
    """Atomic checkpoint write with retry for Lustre filesystem flakes.

    Includes RNG state so a resumed run is bit-comparable. Retries up to
    *max_retries* times on write failures (PytorchStreamWriter / OSError),
    with exponential backoff starting at *retry_base_delay* seconds.
    """
    import time as _time
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "global_step": global_step,
        "metrics": metrics or {}, "config": config or {},
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state()},
    }
    tmp = path + ".tmp"
    for attempt in range(1, max_retries + 1):
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            break
        except (RuntimeError, OSError) as e:
            # Clean up the partial file so the next attempt starts fresh
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt == max_retries:
                print(f"[save_checkpoint] FAILED after {max_retries} attempts: {path} — {e}")
                raise
            delay = retry_base_delay * (2 ** (attempt - 1))
            print(f"[save_checkpoint] attempt {attempt}/{max_retries} failed ({e}), "
                  f"retrying in {delay:.0f}s...")
            _time.sleep(delay)
    if is_best and best_path:
        for attempt in range(1, max_retries + 1):
            try:
                shutil.copyfile(path, best_path)
                break
            except (RuntimeError, OSError) as e:
                if attempt == max_retries:
                    print(f"[save_checkpoint] FAILED copying best after {max_retries} attempts: {e}")
                    raise
                delay = retry_base_delay * (2 ** (attempt - 1))
                print(f"[save_checkpoint] best-copy attempt {attempt}/{max_retries} failed ({e}), "
                      f"retrying in {delay:.0f}s...")
                _time.sleep(delay)
    return path


def load_checkpoint(path: str, model=None, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu") -> Dict[str, Any]:
    import torch
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ck["model"])
    for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
        if obj is not None and ck.get(key) is not None:
            obj.load_state_dict(ck[key])
    return ck


# --------------------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------------------
@dataclass
class ExcelReporter:
    """Maintains one workbook per run, rewritten atomically after every evaluation.

    Sheets
      run_config    : one row per hyperparameter (key/value)
      epoch_metrics : one row per evaluated epoch -- legacy columns first (identical names and
                      order to the existing metrics_summary.xlsx), corpus columns appended
      best          : the selected epoch, val + test, one row
      test_final    : beam-search test metrics from the best checkpoint, one row
      pred_val      : every validation sentence -- ground truth beside the model's translation
      pred_test     : every test sentence       -- ground truth beside the model's translation
      samples       : a fixed handful of uids, one block per epoch, to watch translations evolve
    """
    path: str
    config: Dict[str, Any] = field(default_factory=dict)
    select_on: str = "val_corpus_chrF2"
    select_mode: str = "max"
    history: List[Dict[str, Any]] = field(default_factory=list)
    final: List[Dict[str, Any]] = field(default_factory=list)
    predictions: Dict[str, pd.DataFrame] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    n_samples: int = 25
    max_pred_rows: int = 200_000
    _sample_uids: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_metrics: Dict[str, float], test_metrics: Optional[Dict[str, float]] = None,
                  **extra) -> Dict[str, Any]:
        row = {"epoch": epoch, "train_loss": round(float(train_loss), 4),
               "val_loss": round(float(val_loss), 4)}
        row |= {f"val_{k}": _r(v) for k, v in val_metrics.items()}
        if test_metrics:
            row |= {f"test_{k}": _r(v) for k, v in test_metrics.items()}
        row |= extra
        self.history.append(row)
        self.flush()
        return row

    def log_final(self, split: str, metrics: Dict[str, float], **extra) -> None:
        self.final.append({"split": split, **{k: _r(v) for k, v in metrics.items()}, **extra})
        self.flush()

    def log_predictions(self, split: str, uids: Sequence[str], predictions: Sequence[str],
                        references: Sequence[str], epoch: Optional[int] = None,
                        extra: Optional[Dict[str, Sequence[Any]]] = None,
                        write: bool = True) -> pd.DataFrame:
        """Store the ground truth beside the model's translation, sentence by sentence, in the
        workbook (sheet `pred_<split>`). Call it whenever a new best checkpoint is found and after
        the final beam-search pass; the sheet always holds the most recently logged set, so at the
        end of training it holds the reported system's output.

        Also appends `n_samples` fixed uids to the `samples` sheet, so one sheet shows how the
        translation of the same sentences changes across epochs.
        """
        df = pd.DataFrame({"uid": list(uids), "ground_truth": list(references),
                           "model_translation": list(predictions)})
        if epoch is not None:
            df.insert(0, "epoch", epoch)
        for col, vals in (extra or {}).items():
            df[col] = list(vals)
        if len(df) > self.max_pred_rows:
            df = df.head(self.max_pred_rows)
        self.predictions[split] = df

        if split not in self._sample_uids:
            self._sample_uids[split] = list(df["uid"].head(self.n_samples))
        keep = set(self._sample_uids[split])
        for r in df[df["uid"].isin(keep)].to_dict("records"):
            self.samples.append({"split": split, "epoch": epoch, "uid": r["uid"],
                                 "ground_truth": r["ground_truth"],
                                 "model_translation": r["model_translation"]})
        if write:
            self.flush()
        return df

    def best_row(self) -> Optional[Dict[str, Any]]:
        rows = [r for r in self.history if self.select_on in r and pd.notna(r[self.select_on])]
        if not rows:
            return None
        pick = max if self.select_mode == "max" else min
        return pick(rows, key=lambda r: r[self.select_on])

    def flush(self) -> None:
        # pandas validates the writer extension, so the temp file must stay .xlsx
        tmp = self.path[:-5] + ".tmp.xlsx" if self.path.endswith(".xlsx") else self.path + ".tmp.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as xl:
            _excel_safe(pd.DataFrame([{"key": k, "value": str(v)} for k, v in self.config.items()])) \
                .to_excel(xl, sheet_name="run_config", index=False)
            hist = pd.DataFrame(self.history)
            if not hist.empty:
                hist = hist[_ordered_columns(hist.columns)]
            _excel_safe(hist).to_excel(xl, sheet_name="epoch_metrics", index=False)
            best = self.best_row()
            _excel_safe(pd.DataFrame([best] if best else [])).to_excel(xl, sheet_name="best", index=False)
            _excel_safe(pd.DataFrame(self.final)).to_excel(xl, sheet_name="test_final", index=False)
            for split, df in self.predictions.items():
                _excel_safe(df).to_excel(xl, sheet_name=f"pred_{split}"[:31], index=False)
            if self.samples:
                _excel_safe(pd.DataFrame(self.samples)).to_excel(xl, sheet_name="samples", index=False)
        os.replace(tmp, self.path)


_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Excel rejects control characters and treats a leading '=' as a formula. Model output can
    contain both, and an exception inside flush() would take down a multi-hour run, so text
    columns are sanitised on the way into the workbook (the CSVs keep the raw text)."""
    if df.empty:
        return df
    df = df.copy()
    for c in df.columns:
        # pandas >= 3.0 gives text columns the new `str` dtype, not `object`, so test by exclusion
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].astype(str)
        s = s.str.replace(_ILLEGAL_XLSX, "", regex=True)
        s = s.mask(s.str.startswith(("=", "+", "@")), "'" + s)
        s = s.str.slice(0, 32000)              # Excel's per-cell limit is 32,767 characters
        df[c] = s.where(df[c].notna(), None)
    return df


def _r(v):
    return round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v


def _ordered_columns(cols: Iterable[str]) -> List[str]:
    """epoch, losses, then legacy val/test columns in their historical order, then the rest."""
    cols = list(cols)
    head = [c for c in ("epoch", "train_loss", "val_loss") if c in cols]
    legacy = [f"{s}_{k}" for s in ("val", "test") for k in _LEGACY_KEYS if f"{s}_{k}" in cols]
    rest = [c for c in cols if c not in head + legacy]
    return head + legacy + rest


# --------------------------------------------------------------------------------------
# Cross-run aggregation -> the paper tables
# --------------------------------------------------------------------------------------
def _run_test_predictions(run_results_dir: str) -> Optional[pd.DataFrame]:
    """Final test predictions of one run: the beam-search CSV if present, else the workbook sheet."""
    csv = os.path.join(os.path.dirname(run_results_dir), "predictions", "test_best_beam5.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        return df.rename(columns={"video_id": "uid", "prediction": "model_translation"})
    try:
        return pd.read_excel(os.path.join(run_results_dir, "metrics.xlsx"), sheet_name="pred_test")
    except Exception:
        return None


def aggregate_runs(results_root: str, out_xlsx: str,
                   group_keys: Sequence[str] = ("dataset", "arm"),
                   side_by_side: bool = True) -> pd.DataFrame:
    """Walks results_root for */metrics.xlsx, collects each run's best epoch and final test
    metrics, and writes mean +/- std over seeds per (dataset, arm).

    With side_by_side=True it also writes one `pred_<dataset>` sheet per dataset holding the
    ground truth next to every arm's translation of the same uid, side by side, for error analysis.
    """
    rows: List[Dict[str, Any]] = []
    for dirpath, _, files in os.walk(results_root):
        if "metrics.xlsx" not in files:
            continue
        xls = os.path.join(dirpath, "metrics.xlsx")
        cfg = pd.read_excel(xls, sheet_name="run_config")
        cfg = dict(zip(cfg["key"], cfg["value"])) if not cfg.empty else {}
        try:
            final = pd.read_excel(xls, sheet_name="test_final")
        except Exception:
            final = pd.DataFrame()
        best = pd.read_excel(xls, sheet_name="best")
        row = {"run_dir": dirpath, **{k: cfg.get(k) for k in ("dataset", "arm", "seed", "lr", "params")}}
        if not best.empty:
            row |= {f"best_{c}": best.iloc[0][c] for c in best.columns}
        if not final.empty:
            t = final[final["split"] == "test"]
            if not t.empty:
                row |= {f"final_{c}": t.iloc[-1][c] for c in t.columns if c != "split"}
        if side_by_side:
            row["_preds"] = _run_test_predictions(dirpath)
        rows.append(row)

    runs = pd.DataFrame(rows)
    sbs: Dict[str, pd.DataFrame] = {}
    if side_by_side and "_preds" in runs.columns:
        for ds, grp in runs.groupby(runs["dataset"].fillna("unknown")):
            merged = None
            for _, r in grp.iterrows():
                pr = r["_preds"]
                if pr is None or "uid" not in pr:
                    continue
                col = f"{r.get('arm', 'arm')}_s{r.get('seed', '?')}"
                part = pr[["uid", "ground_truth", "model_translation"]].rename(
                    columns={"model_translation": col})
                merged = part if merged is None else merged.merge(
                    part.drop(columns=["ground_truth"]), on="uid", how="outer")
            if merged is not None:
                sbs[str(ds)] = merged
        runs = runs.drop(columns=["_preds"])
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        runs.to_excel(xl, sheet_name="all_runs", index=False)
        keys = [k for k in group_keys if k in runs.columns]
        num = runs.select_dtypes("number").columns
        if keys and len(runs) and len(num):
            g = runs.groupby(keys)[list(num)]
            summary = g.mean().round(3).add_suffix("_mean").join(g.std().round(3).add_suffix("_std"))
            summary.reset_index().to_excel(xl, sheet_name="summary_mean_std", index=False)
        for ds, df in sbs.items():
            _excel_safe(df).to_excel(xl, sheet_name=f"pred_{ds}"[:31], index=False)
    return runs


def run_dir(base: str, dataset: str, arm: str, lr: float, seed: int) -> str:
    """<base>/<dataset>/<arm>/lr<lr>_s<seed>/ -- mirrors the existing project/dataset/output layout."""
    d = os.path.join(base, dataset, arm, f"lr{lr:g}_s{seed}")
    for sub in ("checkpoints", "predictions", "results"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d
