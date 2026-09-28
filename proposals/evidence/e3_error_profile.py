"""E3 -- Where do the errors sit: the language side or recognising the signs? (Proposal A premise)

Proposal A swaps the 3-layer decoder for a pretrained LLM. That pays off most if outputs are
disfluent or generic (language side), and least if they are fluent frames around the wrong content
words (recognition side, which an LLM cannot fix from the text alone).

Per system, test split, clipped unigram precision / recall split into function words (a fixed
English stopword list) and content words, plus output diversity: distinct outputs, share of
outputs that are one of the 20 most frequent outputs, and word types used vs types in the refs.

  python proposals/evidence/e3_error_profile.py
"""
from __future__ import annotations

import os
import re
from collections import Counter

from common import RESULTS, load_preds, pred_csv, write_report

STOP = set("""a an the and or but if then so of to in on at by for with from into onto up down out over under
about as than too very is am are was were be been being do does did done have has had having i me my
mine we us our you your he him his she her hers it its they them their this that these those there here
what which who whom whose when where why how not no nor all any both each few more most other some such
only own same can will just should would could may might must shall s t don won didn isn wasn aren
weren haven hasn hadn doesn let also again once""".split())
TOK = re.compile(r"[a-z0-9']+")


def words(s: str) -> list[str]:
    return TOK.findall(s.lower())


def pr(preds: list[str], refs: list[str], keep) -> tuple[float, float]:
    match = n_pred = n_ref = 0
    for p, r in zip(preds, refs):
        cp = Counter(w for w in words(p) if keep(w))
        cr = Counter(w for w in words(r) if keep(w))
        match += sum((cp & cr).values())
        n_pred += sum(cp.values())
        n_ref += sum(cr.values())
    return 100 * match / max(n_pred, 1), 100 * match / max(n_ref, 1)


def profile(name: str, rows: dict[str, tuple[str, str]]) -> str:
    preds = [p for p, _ in rows.values()]
    refs = [r for _, r in rows.values()]
    fp, fr = pr(preds, refs, lambda w: w in STOP)
    cp, cr = pr(preds, refs, lambda w: w not in STOP)
    c = Counter(p.strip().lower() for p in preds)
    top20 = sum(n for _, n in c.most_common(20)) / len(preds)
    ptypes = len({w for p in preds for w in words(p)})
    rtypes = len({w for r in refs for w in words(r)})
    return (f"| {name} | {fp:.1f} / {fr:.1f} | {cp:.1f} / {cr:.1f} | {len(c) / len(preds):.1%} | {top20:.1%} | "
            f"{ptypes} / {rtypes} |")


def main():
    systems = {
        "mamba s42": pred_csv("isign", "mamba", 42),
        "mamba ensemble": os.path.join(RESULTS, "isign", "mamba_ensemble", "lp1.3", "predictions", "test_best_beam5.csv"),
        "transformer s42": pred_csv("isign", "transformer", 42),
        "transformer ensemble": os.path.join(RESULTS, "isign", "transformer_ensemble", "lp1.2", "predictions", "test_best_beam5.csv"),
    }
    out = ["# E3 -- error profile, iSign test (Proposal A premise)", "",
           "| system | function words P / R | content words P / R | distinct outputs | outputs in top-20 | word types pred / ref |",
           "|---|---|---|---|---|---|"]
    loaded = {k: load_preds(v) for k, v in systems.items()}
    for k, rows in loaded.items():
        out.append(profile(k, rows))

    rows = loaded["mamba ensemble"]
    c = Counter(p.strip() for p, _ in rows.values())
    out += ["", "Most frequent mamba-ensemble outputs:", ""] + [f"- {n}x  {p}" for p, n in c.most_common(10)]
    text = "\n".join(out)
    print(text)
    print("wrote", write_report("e3_error_profile.md", text))


if __name__ == "__main__":
    main()
