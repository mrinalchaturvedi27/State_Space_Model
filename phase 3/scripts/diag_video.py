"""Diagnostic 2: does the model's lexical prediction depend on the correct signing? (dev data only)

For a fixed sample of validation clips, the REFERENCE is scored (teacher forcing) with the source pose
replaced by:
  correct        the clip itself
  wrong_length   a clip from a different video with length within 15% (resampled to the exact length)
  wrong_topic    among those, the one whose reference text is most similar (TF-IDF over content words),
                 identical sentences excluded: a length- and topic-matched wrong video
  cross_domain   a clip from another dataset's validation split (--cross-data), resampled to length
  zero           all-zero pose (diagnostic)
  static         the clip's mean frame repeated (body and hand shape kept, motion removed)
Reported: per-token reference log-likelihood, content-word log-likelihood, paired differences vs correct
with 95% bootstrap intervals, and the share of clips where the correct pose scores the reference higher.
Then the first --n-decode clips are free-decoded (beam 5, lp 1.0) under correct / wrong_length /
wrong_topic / zero, since teacher forcing can hide visual weakness behind the true text prefix.
The sample (uids and partners) is fixed in --sample-file and reused by every model.

  python scripts/diag_video.py --data configs/data/isign.yaml --model configs/model/mamba_padfix.yaml \\
      --checkpoint "../phase 2/results/isign/mamba_padfix/lr0.0003_s42/checkpoints/best.pt" \\
      --cross-data configs/data/how2sign.yaml --cache-dir ../cache --out results/diagnostics/video/isign_mamba_padfix
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p3 import paths  # noqa: E402,F401
from p3.content import content  # noqa: E402
from p3.diag import resample, teacher_forced, word_scores  # noqa: E402
from p3.loading import load, load_cfg  # noqa: E402
from p3.stitch import parse_uid  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402

from src.data import make_collate, make_dataset  # noqa: E402
from src.evaluate import beam_search_decode, strip_specials  # noqa: E402

CONDITIONS = ["correct", "wrong_length", "wrong_topic", "cross_domain", "zero", "static"]
DECODE_CONDITIONS = ["correct", "wrong_length", "wrong_topic", "zero"]


def tfidf_vectors(texts, lang):
    docs = [Counter(content(t, lang)) for t in texts]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    vecs = []
    for d in docs:
        v = {w: c * math.log((1 + n) / (1 + df[w])) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def build_sample(ds, cross, n, seed, lang, path):
    if os.path.exists(path):
        s = pd.read_csv(path)
        print(f"reusing fixed sample {path} ({len(s)} clips)")
        return s
    rng = np.random.default_rng(seed)
    uids = ds.index["uid"].astype(str).tolist()
    texts = ds.index["text"].astype(str).tolist()
    nf = ds.n_frames_array()
    vids = [parse_uid(u)[0] for u in uids]
    vecs = tfidf_vectors(texts, lang)
    norm = [" ".join(t.lower().split()) for t in texts]
    cnf = cross.n_frames_array()
    rows = []
    for i in rng.choice(len(uids), size=min(n, len(uids)), replace=False):
        i = int(i)
        L = nf[i]
        # different video and different sentence; fall back to any other video, then any other clip
        diff = ([j for j in range(len(uids)) if vids[j] != vids[i] and norm[j] != norm[i]]
                or [j for j in range(len(uids)) if vids[j] != vids[i]] or [j for j in range(len(uids)) if j != i])
        near = [j for j in diff if abs(nf[j] - L) <= 0.15 * L] or sorted(diff, key=lambda j: abs(nf[j] - L))[:50]
        wl = int(rng.choice(near))
        wt = max(near, key=lambda j: cos(vecs[i], vecs[j]))
        cnear = np.nonzero(np.abs(cnf - L) <= 0.15 * L)[0]
        cd = int(rng.choice(cnear)) if len(cnear) else int(np.argmin(np.abs(cnf - L)))
        rows.append({"idx": i, "uid": uids[i], "frames": int(L), "wrong_length_idx": wl, "wrong_topic_idx": wt,
                     "topic_sim": round(cos(vecs[i], vecs[wt]), 4), "cross_idx": cd})
    s = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    s.to_csv(path, index=False)
    print(f"fixed sample written: {path} ({len(s)} clips)")
    return s


def cond_feat(cond, r, ds, cross):
    own = ds[int(r.idx)]["feat"]
    L = own.shape[0]
    if cond == "correct":
        return own
    if cond == "wrong_length":
        return resample(ds[int(r.wrong_length_idx)]["feat"], L)
    if cond == "wrong_topic":
        return resample(ds[int(r.wrong_topic_idx)]["feat"], L)
    if cond == "cross_domain":
        return resample(cross[int(r.cross_idx)]["feat"], L)
    if cond == "zero":
        return np.zeros_like(own)
    if cond == "static":
        return np.repeat(own.mean(0, keepdims=True), L, axis=0)
    raise ValueError(cond)


def boot_ci(x, rng, n=1000):
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cross-data", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-decode", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ds, data_cfg, model_cfg = load(args.data, args.model, args.checkpoint, args.cache_dir, device)
    assert model_cfg.get("context_clips") is None, "context models are not supported here"
    lang = data_cfg.get("lang", "en")
    xc = load_cfg(args.cross_data)
    cross = make_dataset(args.cache_dir, xc["dataset"], "val", os.path.join(args.cache_dir, xc["dataset"], "spm.model"),
                         64, 512, augment=False)
    sample_file = args.sample_file or os.path.join(os.path.dirname(args.out.rstrip("/")), f"sample_video_{data_cfg['dataset']}.csv")
    S = build_sample(ds, cross, args.n, args.seed, lang, sample_file)
    collate = make_collate(ds.pad_id)

    rows = []
    for cond in CONDITIONS:
        for s in range(0, len(S), args.batch):
            chunk = S.iloc[s:s + args.batch]
            items = [{"uid": r.uid, "feat": cond_feat(cond, r, ds, cross), "ids": ds[int(r.idx)]["ids"]}
                     for r in chunk.itertuples()]
            for r, (ids, lps) in zip(chunk.itertuples(), teacher_forced(model, collate(items), device)):
                ws = word_scores(ds.sp, ids, lps, lang)
                cw = [w for w in ws if w[2]]
                rows.append({"uid": r.uid, "condition": cond, "n_tokens": len(lps), "ref_ll": float(np.sum(lps)),
                             "n_content": len(cw), "content_ll": float(sum(w[1] for w in cw))})
    D = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    D.to_csv(os.path.join(args.out, "per_clip.csv"), index=False)

    rng = np.random.default_rng(1)
    base = D[D.condition == "correct"].set_index("uid")
    summ = []
    for cond in CONDITIONS:
        c = D[D.condition == cond].set_index("uid").loc[base.index]
        row = {"condition": cond, "token_nll": round(-c.ref_ll.sum() / c.n_tokens.sum(), 4),
               "content_word_ll": round(c.content_ll.sum() / max(1, c.n_content.sum()), 4)}
        if cond != "correct":
            d_all = (base.ref_ll - c.ref_ll) / base.n_tokens            # per-token, >0 = correct pose better
            has = base.n_content > 0
            d_con = (base.content_ll[has] - c.content_ll[has]) / base.n_content[has]
            lo, hi = boot_ci(d_all, rng)
            clo, chi = boot_ci(d_con, rng)
            row.update({"delta_token_ll_vs_correct": round(d_all.mean(), 4), "ci95": f"[{lo:.4f}, {hi:.4f}]",
                        "delta_content_ll_vs_correct": round(d_con.mean(), 4), "content_ci95": f"[{clo:.4f}, {chi:.4f}]",
                        "correct_better_%": round(100 * (d_all > 0).mean(), 1)})
        summ.append(row)
    Sm = pd.DataFrame(summ)
    Sm.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(Sm.to_string(index=False))

    dec = []
    sub = S.iloc[: args.n_decode]
    for cond in DECODE_CONDITIONS:
        for r in sub.itertuples():
            b = collate([{"uid": r.uid, "feat": cond_feat(cond, r, ds, cross), "ids": ds[int(r.idx)]["ids"]}]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                seq = beam_search_decode(model, b.src, b.src_key_padding_mask, ds.bos_id, ds.eos_id, ds.pad_id,
                                         beam_size=5, max_new_tokens=model_cfg.get("max_tgt_len", 64))
            hyp = ds.sp.decode(strip_specials(seq.tolist(), ds.bos_id, ds.eos_id, ds.pad_id))
            ref = ds.sp.decode(strip_specials(b.tgt_out[0].tolist(), ds.bos_id, ds.eos_id, ds.pad_id))
            dec.append({"uid": r.uid, "condition": cond, "hypothesis": hyp, "reference": ref,
                        "sent_chrF": sacrebleu.sentence_chrf(hyp, [ref]).score,
                        "content_overlap": len(content(hyp, lang) & content(ref, lang)) > 0})
    E = pd.DataFrame(dec)
    E.to_csv(os.path.join(args.out, "decodes.csv"), index=False)
    corr = E[E.condition == "correct"].set_index("uid").hypothesis
    drows = []
    for cond in DECODE_CONDITIONS:
        e = E[E.condition == cond]
        drows.append({"condition": cond, "n": len(e),
                      "corpus_chrF2": round(sacrebleu.corpus_chrf(e.hypothesis.tolist(), [e.reference.tolist()]).score, 2),
                      "content_overlap_%": round(100 * e.content_overlap.mean(), 1),
                      "same_output_as_correct_%": round(100 * (e.set_index("uid").hypothesis == corr.loc[e.uid].values).mean(), 1)})
    Dd = pd.DataFrame(drows)
    Dd.to_csv(os.path.join(args.out, "decode_summary.csv"), index=False)
    print(Dd.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
