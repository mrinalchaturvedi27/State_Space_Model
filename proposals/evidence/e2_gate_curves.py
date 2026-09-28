"""E2 -- Is "Δ places the decoder memory better than a uniform stride" above noise? (Proposal A's premise)

Reads every phase-2 iSign run under "phase 2/results/isign/mamba_*/lr0.0003_s*/results/metrics.xlsx"
(so round-2 runs are picked up once they are pushed) and reports per run: best val chrF2 and its
epoch, epochs trained, last-5-epoch slope (still improving?), kept_frac. Then, for every epoch both
have reached, the chrF2 gap mamba_pool_matched - mamba_uniform per seed.

Noise reference: sd over seeds of best val chrF2 for the phase-1 iSign runs (lr 3e-4, 3 seeds).

  python proposals/evidence/e2_gate_curves.py
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import numpy as np
import openpyxl

from common import REPO, RESULTS, SEEDS, write_report

PHASE2 = os.path.join(REPO, "phase 2", "results", "isign")
KEY = "val_corpus_chrF2"
PAIRS = [("mamba_pool_matched", "mamba_uniform"), ("mamba_pool_avg", "mamba_uniform_avg"),
         ("mamba_pool_matched_s8", "mamba_uniform_s8"), ("mamba_pool_matched_s32", "mamba_uniform_s32")]


def epoch_rows(path: str) -> list[dict]:
    ws = openpyxl.load_workbook(path, read_only=True)["epoch_metrics"]
    rows = list(ws.iter_rows(values_only=True))
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def main():
    out = ["# E2 -- phase-2 gate curves (Proposal A premise)", ""]

    # seed noise from phase 1
    out += ["## Seed noise (phase 1, iSign, lr 3e-4, best val chrF2)", "", "| arm | seeds | best chrF2 | sd |", "|---|---|---|---|"]
    for arm in ("mamba", "transformer"):
        best = [max(r[KEY] for r in epoch_rows(os.path.join(RESULTS, "isign", arm, f"lr0.0003_s{s}", "results", "metrics.xlsx")))
                for s in SEEDS]
        out.append(f"| {arm} | {', '.join(map(str, SEEDS))} | {', '.join(f'{b:.2f}' for b in best)} | {np.std(best, ddof=1):.2f} |")

    runs = {}
    for path in sorted(glob.glob(os.path.join(PHASE2, "mamba_*", "lr0.0003_s*", "results", "metrics.xlsx"))):
        arm = path.split(os.sep)[-4]
        seed = int(re.search(r"_s(\d+)", path.split(os.sep)[-3]).group(1))
        runs[(arm, seed)] = epoch_rows(path)

    out += ["", "## Phase-2 runs", "", "| arm | seed | epochs | best chrF2 (epoch) | last chrF2 | slope last 5 ep | val kept_frac |",
            "|---|---|---|---|---|---|---|"]
    for (arm, seed), rows in sorted(runs.items()):
        ch = [r[KEY] for r in rows]
        b = int(np.argmax(ch))
        slope = np.polyfit(np.arange(5), ch[-5:], 1)[0] if len(ch) >= 5 else float("nan")
        kept = rows[-1].get("val_kept_frac")
        out.append(f"| {arm} | {seed} | {len(ch)} | {ch[b]:.2f} ({rows[b]['epoch']}) | {ch[-1]:.2f} | {slope:+.3f}/ep | "
                   f"{kept if kept is None else f'{kept:.3f}'} |")

    out += ["", "## Δ vs uniform at the same frame count", ""]
    for a, b in PAIRS:
        seeds = sorted(s for (arm, s) in runs if arm == a and (b, s) in runs)
        if not seeds:
            continue
        out.append(f"### {a} - {b}")
        out += ["", "| seed | best-vs-best gap | epochs compared | per-epoch gap mean (ep >= 10) | epochs with gap > 0 (ep >= 10) |", "|---|---|---|---|---|"]
        best_gaps = []
        for s in seeds:
            ca, cb = [r[KEY] for r in runs[(a, s)]], [r[KEY] for r in runs[(b, s)]]
            n = min(len(ca), len(cb))
            g = np.array(ca[:n]) - np.array(cb[:n])
            late = g[10:]
            best_gaps.append(max(ca) - max(cb))
            out.append(f"| {s} | {best_gaps[-1]:+.2f} | {n} | {late.mean():+.2f} | {(late > 0).sum()}/{len(late)} |")
        if len(best_gaps) > 1:
            out.append(f"| mean ± sd | {np.mean(best_gaps):+.2f} ± {np.std(best_gaps, ddof=1):.2f} | | | |")
        out.append("")

    text = "\n".join(out)
    print(text)
    print("wrote", write_report("e2_gate_curves.md", text))


if __name__ == "__main__":
    main()
