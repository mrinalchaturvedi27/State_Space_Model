"""Are a dataset's target texts formatted the same way in train, val and test?

ABLATION_AND_ERROR_ANALYSIS.md §2.6: How2Sign models only ever output lowercase text with literal
<UNKNOWN> tokens, while every test reference is cased. This prints, for every string column of each
split's CSV and for the texts actually stored in the built cache, how often entries contain
uppercase letters, contain <UNKNOWN>/<unk>, and end with punctuation, plus a few examples for the
same row of each column. Read-only: it changes nothing.

  python scripts/check_text_columns.py --data configs/data/how2sign.yaml --cache-dir cache
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd
import yaml

UNK = re.compile(r"<\s*unk(nown)?\s*>", re.IGNORECASE)


def profile(texts: pd.Series) -> dict:
    t = texts.dropna().astype(str)
    if t.empty:
        return {}
    return {"n": len(t),
            "uppercase %": round(100 * t.map(lambda s: any(c.isupper() for c in UNK.sub("", s))).mean(), 1),
            "<UNKNOWN> %": round(100 * t.map(lambda s: bool(UNK.search(s))).mean(), 1),
            "ends with .!? %": round(100 * t.map(lambda s: s.rstrip()[-1:] in ".!?").mean(), 1),
            "mean words": round(t.map(lambda s: len(s.split())).mean(), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    with open(args.data) as f:
        dc = yaml.safe_load(f)
    print(f"dataset {dc['dataset']}: configured text_col = {dc['text_col']!r}\n")

    for split in ("train", "val", "test"):
        df = pd.read_csv(dc[f"{split}_csv"])
        cols = [c for c in df.columns
                if (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
                and c not in (dc["uid_col"], dc.get("pose_id_col"))]
        print(f"=== {split}: {dc[f'{split}_csv']} ({len(df)} rows)")
        rows = []
        for c in cols:
            p = profile(df[c])
            if p and p["mean words"] >= 2:  # skip ids and short codes
                rows.append({"column": c + (" <- used" if c == dc["text_col"] else ""), **p})
        print(pd.DataFrame(rows).to_string(index=False))
        text_cols = [r["column"].split(" ")[0] for r in rows]
        for i in range(min(args.examples, len(df))):
            print(f"  row {i}: " + " || ".join(f"{c}: {str(df.iloc[i][c])[:90]!r}" for c in text_cols))
        print()

    print("=== texts stored in the built cache (what the model was actually trained / scored on)")
    rows = []
    for split in ("train", "val", "test"):
        path = os.path.join(args.cache_dir, dc["dataset"], f"{split}_index.parquet")
        if os.path.exists(path):
            idx = pd.read_parquet(path)
            rows.append({"split": split, **profile(idx["text"])})
            print(f"  {split} example: {str(idx['text'].iloc[0])[:100]!r}")
        else:
            print(f"  {split}: no {path}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nA mismatch = uppercase % or <UNKNOWN> % differing sharply between train and val/test for the "
          "used column. Pick a column whose profile is the same in all three splits.")


if __name__ == "__main__":
    main()
