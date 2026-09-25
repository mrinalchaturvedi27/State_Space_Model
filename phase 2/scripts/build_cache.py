"""
Build the frozen preprocessing cache described in BENCHMARK_PLAN.md §2.

Pipeline per clip (identical for every dataset and both model arms):
  Pose.read -> reduce_holistic (178 kp) -> normalize (per-clip shoulder scale/origin)
  -> keep x,y, drop z/confidence -> interpolate missing frames -> resample to target_fps
  -> uniformly subsample to t_max (not truncated) -> float16

Two-stage and resumable: stage 1 writes one .npy shard per clip (skipped if it already
exists, so a killed run resumes for free); stage 2 concatenates shards for a split into a
single memmap + Parquet index. Re-running with a changed preprocess.yaml gets a new
config_hash and therefore a fresh shard directory -- it never silently mixes cache versions.

Usage:
  python scripts/build_cache.py --data configs/data/isign.yaml --out-dir cache \
      --splits train val test --workers 32
  python scripts/build_cache.py --data configs/data/isign.yaml --out-dir cache --limit 200   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    n_keypoints: int = 178
    dims: int = 2
    target_fps: int = 25
    t_max: int = 512
    cache_dtype: str = "float16"


@dataclass
class DataConfig:
    dataset: str
    train_csv: str
    val_csv: str
    test_csv: str
    pose_dir: str
    pose_ext: str
    uid_col: str
    text_col: str
    pose_id_col: str
    source_fps: int
    vocab_size: int
    lang: str = "en"

    def csv_for(self, split: str) -> str:
        return {"train": self.train_csv, "val": self.val_csv, "test": self.test_csv}[split]


def load_configs(data_yaml: str, preprocess_yaml: str) -> tuple[DataConfig, PreprocessConfig]:
    with open(data_yaml) as f:
        dc = DataConfig(**yaml.safe_load(f))
    with open(preprocess_yaml) as f:
        pc = PreprocessConfig(**yaml.safe_load(f))
    return dc, pc


def config_hash(dc: DataConfig, pc: PreprocessConfig) -> str:
    """Stamped into the cache dir name; a changed preprocessing decision or column mapping
    produces a new hash rather than silently mixing with old cached tensors."""
    payload = {
        "preprocess": asdict(pc),
        "uid_col": dc.uid_col, "text_col": dc.text_col, "pose_id_col": dc.pose_id_col,
        "pose_ext": dc.pose_ext,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Per-clip feature extraction
# --------------------------------------------------------------------------------------
def _interpolate_missing(xy: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """xy: (T, D) raw values. mask: (T, D) bool, True = missing.
    Linear interpolation across time per column; leading/trailing gaps held at the nearest
    valid frame (np.interp's default flat-extrapolation gives this for free). A column with
    no valid frame at all is zeroed and counted toward the returned missing fraction."""
    T, D = xy.shape
    out = xy.copy()
    frame_idx = np.arange(T)
    n_missing_cells = 0
    for d in range(D):
        col_mask = mask[:, d]
        if not col_mask.any():
            continue
        valid = ~col_mask
        if not valid.any():
            out[:, d] = 0.0
            n_missing_cells += T
            continue
        out[col_mask, d] = np.interp(frame_idx[col_mask], frame_idx[valid], xy[valid, d])
        n_missing_cells += int(col_mask.sum())
    return out, n_missing_cells / (T * D)


def _resample_linear(x: np.ndarray, new_len: int) -> np.ndarray:
    """(T, D) -> (new_len, D) by linear interpolation across the time axis (fps conversion)."""
    T = x.shape[0]
    if T == new_len:
        return x
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_len)
    idx_floor = np.floor(new_idx).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, T - 1)
    frac = (new_idx - idx_floor)[:, None]
    return x[idx_floor] * (1 - frac) + x[idx_ceil] * frac


def _uniform_subsample_idx(T: int, t_max: int) -> np.ndarray:
    """Index-based (not interpolated) subsampling to t_max, spanning the full clip so the
    end of the sentence is never cut off -- see §2's "not truncated" requirement."""
    if T <= t_max:
        return np.arange(T)
    return np.round(np.linspace(0, T - 1, t_max)).astype(int)


def extract_features(pose_path: str, pc: PreprocessConfig) -> Optional[dict]:
    """Returns None if the pose file is missing, empty, or fails to parse -- counted (with the
    reason) by the caller, not hidden. A corrupt/truncated .pose file must not crash the whole
    multiprocessing pool; two such files exist in iSign's 254,474 (confirmed 2026-08-24)."""
    from pose_format import Pose
    from pose_format.utils.generic import reduce_holistic

    if not os.path.exists(pose_path):
        return None
    if os.path.getsize(pose_path) == 0:
        print(f"[build_cache] WARN empty file, treating as missing: {pose_path}", flush=True)
        return None
    try:
        with open(pose_path, "rb") as f:
            raw = f.read()
        pose = Pose.read(raw)
        pose = reduce_holistic(pose)
    except Exception as e:
        print(f"[build_cache] WARN failed to parse, treating as missing: {pose_path} ({type(e).__name__}: {e})",
              flush=True)
        return None
    pose = pose.normalize()  # per-clip shoulder-distance scale + shoulder-midpoint origin (§2.2)

    data = pose.body.data  # masked array (T, people, n_keypoints, 3)
    if data.shape[1] > 1:
        data = data[:, :1]  # single-signer clips only; keep the first person if more are present
    orig_fps = float(pose.body.fps)
    T = data.shape[0]

    xy = np.ma.getdata(data)[:, 0, :, :pc.dims].astype(np.float32).reshape(T, -1)  # (T, n_kp*dims)
    mask = np.ma.getmaskarray(data)[:, 0, :, :pc.dims].reshape(T, -1)

    if mask.all():
        xy_filled, frac_missing = np.zeros_like(xy), 1.0
    else:
        xy_filled, frac_missing = _interpolate_missing(xy, mask)

    if abs(orig_fps - pc.target_fps) > 1e-6:
        new_T = max(1, round(T * pc.target_fps / orig_fps))
        xy_filled = _resample_linear(xy_filled, new_T)

    keep_idx = _uniform_subsample_idx(xy_filled.shape[0], pc.t_max)
    feat = xy_filled[keep_idx].astype(np.float16)

    return {
        "feat": feat, "n_frames": feat.shape[0], "orig_n_frames": T,
        "orig_fps": orig_fps, "frac_missing": round(float(frac_missing), 4),
        "fully_empty": bool(mask.all()),
    }


# --------------------------------------------------------------------------------------
# Stage 1: per-clip shards (resumable)
# --------------------------------------------------------------------------------------
def _process_row(args) -> dict:
    uid, pose_id, text, pose_dir, pose_ext, shard_dir, pc_dict = args
    pc = PreprocessConfig(**pc_dict)
    shard_path = os.path.join(shard_dir, f"{uid}.npz")
    if os.path.exists(shard_path):
        with np.load(shard_path) as z:
            return {"uid": uid, "text": text, "n_frames": int(z["n_frames"]),
                    "orig_n_frames": int(z["orig_n_frames"]), "frac_missing": float(z["frac_missing"]),
                    "fully_empty": bool(z["fully_empty"]), "missing_pose": False, "shard_path": shard_path}
    pose_path = os.path.join(pose_dir, pose_id + pose_ext)
    out = extract_features(pose_path, pc)
    if out is None:
        return {"uid": uid, "text": text, "missing_pose": True, "shard_path": None}
    tmp = shard_path + ".tmp.npz"
    np.savez(tmp, feat=out["feat"], n_frames=out["n_frames"], orig_n_frames=out["orig_n_frames"],
              orig_fps=out["orig_fps"], frac_missing=out["frac_missing"], fully_empty=out["fully_empty"])
    os.replace(tmp, shard_path)
    return {"uid": uid, "text": text, "n_frames": out["n_frames"], "orig_n_frames": out["orig_n_frames"],
            "frac_missing": out["frac_missing"], "fully_empty": out["fully_empty"],
            "missing_pose": False, "shard_path": shard_path}


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text))


def build_split(dc: DataConfig, pc: PreprocessConfig, split: str, out_dir: str,
                workers: int, limit: Optional[int] = None) -> pd.DataFrame:
    h = config_hash(dc, pc)
    ds_dir = os.path.join(out_dir, dc.dataset)
    shard_dir = os.path.join(ds_dir, f"shards_{h}", split)
    os.makedirs(shard_dir, exist_ok=True)

    df = pd.read_csv(dc.csv_for(split))
    if limit:
        df = df.head(limit)
    df[dc.text_col] = df[dc.text_col].map(nfkc)

    jobs = [(row[dc.uid_col], row[dc.pose_id_col], row[dc.text_col], dc.pose_dir, dc.pose_ext,
             shard_dir, asdict(pc)) for _, row in df.iterrows()]

    print(f"[{dc.dataset}/{split}] {len(jobs)} rows, {workers} workers, shard_dir={shard_dir}")
    with Pool(workers) as pool:
        results = list(pool.imap(_process_row, jobs, chunksize=32))

    results_df = pd.DataFrame(results)
    n_missing = int(results_df["missing_pose"].sum())
    n_empty = int(results_df.get("fully_empty", pd.Series(dtype=bool)).sum())
    print(f"[{dc.dataset}/{split}] done: {n_missing} missing pose files, "
          f"{n_empty} fully-empty clips (of {len(results_df)})")
    if n_missing:
        missing_uids = results_df.loc[results_df["missing_pose"], "uid"].tolist()[:20]
        print(f"[{dc.dataset}/{split}] first missing uids: {missing_uids}")
    return results_df


# --------------------------------------------------------------------------------------
# Stage 2: consolidate shards -> one memmap + Parquet index
# --------------------------------------------------------------------------------------
def consolidate(dc: DataConfig, pc: PreprocessConfig, split: str, results_df: pd.DataFrame,
                out_dir: str) -> None:
    h = config_hash(dc, pc)
    ds_dir = os.path.join(out_dir, dc.dataset)
    ok = results_df[~results_df["missing_pose"]].reset_index(drop=True)
    if ok.empty:
        raise RuntimeError(f"[{dc.dataset}/{split}] every row failed to load a pose file -- nothing to write")
    # n_frames/orig_n_frames come back as float64 after filtering (the unfiltered column had NaN
    # in the missing-pose rows, which upcasts the whole column) -- must be int before use as a
    # memmap slice bound, or Python raises "slice indices must be integers".
    ok["n_frames"] = ok["n_frames"].astype(int)
    ok["orig_n_frames"] = ok["orig_n_frames"].astype(int)

    total_frames = int(ok["n_frames"].sum())
    feat_dim = pc.n_keypoints * pc.dims
    memmap_path = os.path.join(ds_dir, f"{split}.memmap")
    mm = np.memmap(memmap_path, dtype=pc.cache_dtype, mode="w+", shape=(total_frames, feat_dim))

    offset = 0
    offsets, n_frames_col = [], []
    for shard_path, n_frames in zip(ok["shard_path"], ok["n_frames"]):
        n_frames = int(n_frames)
        with np.load(shard_path) as z:
            mm[offset:offset + n_frames] = z["feat"]
        offsets.append(offset)
        n_frames_col.append(n_frames)
        offset += n_frames
    mm.flush()

    index = pd.DataFrame({
        "uid": ok["uid"], "text": ok["text"], "offset": offsets, "n_frames": n_frames_col,
        "orig_n_frames": ok["orig_n_frames"], "frac_missing": ok["frac_missing"],
        "fully_empty": ok["fully_empty"],
    })
    index_path = os.path.join(ds_dir, f"{split}_index.parquet")
    index.to_parquet(index_path, index=False)

    with open(os.path.join(ds_dir, f"{split}.shape.json"), "w") as f:
        json.dump({"shape": [total_frames, feat_dim], "dtype": pc.cache_dtype}, f)

    print(f"[{dc.dataset}/{split}] wrote {memmap_path} "
          f"({total_frames:,} frames x {feat_dim}, {total_frames * feat_dim * 2 / 1e9:.2f} GB) "
          f"and {index_path} ({len(index)} rows)")


# --------------------------------------------------------------------------------------
# SentencePiece (train split only, per §2)
# --------------------------------------------------------------------------------------
def train_tokenizer(dc: DataConfig, out_dir: str, train_texts: pd.Series) -> None:
    import sentencepiece as spm

    ds_dir = os.path.join(out_dir, dc.dataset)
    corpus_path = os.path.join(ds_dir, "spm_train_corpus.txt")
    with open(corpus_path, "w") as f:
        for t in train_texts:
            f.write(nfkc(t).replace("\n", " ") + "\n")

    model_prefix = os.path.join(ds_dir, "spm")
    spm.SentencePieceTrainer.train(
        input=corpus_path, model_prefix=model_prefix, vocab_size=dc.vocab_size,
        model_type="unigram", character_coverage=1.0, byte_fallback=True,
        normalization_rule_name="identity",  # text is already NFKC-normalized above
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,  # explicit pad id -- sentencepiece's default is
                                                   # pad_id=-1 (no pad token), which would break batching
    )
    print(f"[{dc.dataset}] trained SentencePiece unigram vocab={dc.vocab_size} -> {model_prefix}.model "
          f"(unk=0, bos=1, eos=2, pad=3)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to configs/data/<dataset>.yaml")
    ap.add_argument("--preprocess", default=os.path.join(os.path.dirname(__file__), "..", "configs", "preprocess.yaml"))
    ap.add_argument("--out-dir", default="cache")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per split, for a quick smoke test")
    args = ap.parse_args()

    dc, pc = load_configs(args.data, args.preprocess)
    os.makedirs(os.path.join(args.out_dir, dc.dataset), exist_ok=True)
    h = config_hash(dc, pc)
    print(f"[{dc.dataset}] config_hash={h} preprocess={asdict(pc)}")

    train_texts = None
    for split in args.splits:
        results_df = build_split(dc, pc, split, args.out_dir, args.workers, args.limit)
        consolidate(dc, pc, split, results_df, args.out_dir)
        if split == "train":
            train_texts = results_df.loc[~results_df["missing_pose"], "text"]

    if train_texts is not None:
        train_tokenizer(dc, args.out_dir, train_texts)

    manifest = {"dataset": dc.dataset, "config_hash": h, "preprocess": asdict(pc),
                "data_config": asdict(dc), "splits_built": args.splits}
    with open(os.path.join(args.out_dir, dc.dataset, "cache_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{dc.dataset}] cache_manifest.json written. Cache build complete.")


if __name__ == "__main__":
    main()
