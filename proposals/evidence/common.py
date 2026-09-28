"""Shared loaders for the local evidence tests. Needs numpy + sacrebleu (+ openpyxl for gate curves).

Metrics match slt_reporting.compute_metrics: corpus BLEU-4 is sacrebleu's default BLEU, chrF2 is
CHRF(word_order=0, beta=2), empty predictions/references are skipped.
"""
from __future__ import annotations

import csv
import os

import sacrebleu

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(REPO, "results")
OUT_DIR = os.path.join(REPO, "proposals", "results")
SEEDS = (13, 42, 1337)

_BLEU = sacrebleu.BLEU()
_CHRF = sacrebleu.CHRF(word_order=0, beta=2)


def pred_csv(dataset: str, arm: str, seed: int, split: str = "test", lr: str = "0.0003") -> str:
    return os.path.join(RESULTS, dataset, arm, f"lr{lr}_s{seed}", "predictions", f"{split}_best_beam5.csv")


def load_preds(path: str) -> dict[str, tuple[str, str]]:
    """uid -> (prediction, reference)."""
    with open(path, newline="") as f:
        return {r["video_id"]: (r["prediction"] or "", r["ground_truth"] or "") for r in csv.DictReader(f)}


def video_of(uid: str) -> str:
    """iSign uids are <video>-<segment>."""
    return uid.rsplit("-", 1)[0]


def corpus(preds: list[str], refs: list[str]) -> dict[str, float]:
    pairs = [(p.strip(), r.strip()) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    p, r = [a for a, _ in pairs], [b for _, b in pairs]
    return {"BLEU-4": round(_BLEU.corpus_score(p, [r]).score, 2),
            "chrF2": round(_CHRF.corpus_score(p, [r]).score, 2), "n": len(p)}


def sent_chrf(pred: str, ref: str) -> float:
    return _CHRF.sentence_score(pred, [ref]).score


def write_report(name: str, text: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    return path
