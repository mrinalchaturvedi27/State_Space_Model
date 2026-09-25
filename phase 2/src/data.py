"""Dataset, train-time augmentation, and token-budget batching (BENCHMARK_PLAN.md §2, §4).

Reads the memmap + Parquet index written by scripts/build_cache.py. Augmentation uses a
per-(seed, epoch, idx) RNG so the exact same augmented tensors can be reproduced for both
arms when trained with the same seed -- "identical, same seed stream, applied to both arms" (§2).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, Sampler


def _resample_linear(x: np.ndarray, new_len: int) -> np.ndarray:
    T = x.shape[0]
    if T == new_len or T < 2:
        return x
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_len)
    idx_floor = np.floor(new_idx).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, T - 1)
    frac = (new_idx - idx_floor)[:, None]
    return x[idx_floor] * (1 - frac) + x[idx_ceil] * frac


def augment_clip(feat: np.ndarray, rng: np.random.Generator, t_max: int) -> np.ndarray:
    """§2's train-time augmentation recipe, applied in the order listed there. `feat` is
    (T, n_kp*2) float32, already shoulder-scale/origin normalized at cache time.
    Excluded: horizontal flip -- swaps dominant/non-dominant hand, linguistically meaningful in ISL."""
    T, D = feat.shape

    factor = rng.uniform(0.8, 1.2)  # 1. temporal resample x U(0.8, 1.2)
    feat = _resample_linear(feat, max(2, min(t_max, round(T * factor))))
    T = feat.shape[0]

    if T > 4:  # 2. random frame drop p=0.05 (frames removed, sequence shortens)
        keep = rng.random(T) >= 0.05
        if keep.sum() >= 2:
            feat = feat[keep]
            T = feat.shape[0]

    feat = feat + rng.normal(0.0, 0.01, size=feat.shape).astype(np.float32)  # 3. Gaussian noise

    xy = feat.reshape(T, D // 2, 2)  # 4. random 2D affine, one transform per clip
    theta = np.radians(rng.uniform(-5, 5))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    scale = rng.uniform(0.9, 1.1)
    translate = rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    xy = (xy @ rot.T) * scale + translate
    return xy.reshape(T, D).astype(np.float32)


@dataclass
class Batch:
    src: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    tgt_key_padding_mask: torch.Tensor
    uids: List[str]
    n_frames: torch.Tensor

    def to(self, device, dtype=None) -> "Batch":
        src = self.src.to(device, dtype=dtype) if dtype else self.src.to(device)
        return Batch(src, self.src_key_padding_mask.to(device), self.tgt_in.to(device),
                    self.tgt_out.to(device), self.tgt_key_padding_mask.to(device),
                    self.uids, self.n_frames)


class PoseTextDataset(Dataset):
    def __init__(self, cache_dir: str, dataset: str, split: str, spm_model: str,
                max_tgt_len: int = 64, t_max: int = 512, augment: bool = False, seed: int = 0):
        ds_dir = os.path.join(cache_dir, dataset)
        self.index = pd.read_parquet(os.path.join(ds_dir, f"{split}_index.parquet"))
        with open(os.path.join(ds_dir, f"{split}.shape.json")) as f:
            shape_info = json.load(f)
        self.memmap = np.memmap(os.path.join(ds_dir, f"{split}.memmap"), dtype=shape_info["dtype"],
                                mode="r", shape=tuple(shape_info["shape"]))
        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.max_tgt_len = max_tgt_len
        self.t_max = t_max
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.bos_id, self.eos_id, self.pad_id = self.sp.bos_id(), self.sp.eos_id(), self.sp.pad_id()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def n_frames_array(self) -> np.ndarray:
        return self.index["n_frames"].to_numpy()

    def __getitem__(self, idx: int) -> dict:
        row = self.index.iloc[idx]
        feat = np.asarray(self.memmap[row.offset: row.offset + row.n_frames], dtype=np.float32)

        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, int(idx)])
            feat = augment_clip(feat, rng, self.t_max)
        elif feat.shape[0] > self.t_max:
            step = max(1, feat.shape[0] // self.t_max)
            feat = feat[::step][: self.t_max]

        ids = self.sp.encode(str(row.text), out_type=int)[: self.max_tgt_len - 2]
        ids = [self.bos_id] + ids + [self.eos_id]
        return {"uid": row.uid, "feat": feat, "ids": ids}


def make_collate(pad_id: int):
    def collate(batch: List[dict]) -> Batch:
        B = len(batch)
        Tm = max(b["feat"].shape[0] for b in batch)
        D = batch[0]["feat"].shape[1]
        Lm = max(len(b["ids"]) for b in batch)

        src = torch.zeros(B, Tm, D, dtype=torch.float32)
        src_pad = torch.ones(B, Tm, dtype=torch.bool)
        tgt_in = torch.full((B, Lm - 1), pad_id, dtype=torch.long)
        tgt_out = torch.full((B, Lm - 1), pad_id, dtype=torch.long)
        tgt_pad = torch.ones(B, Lm - 1, dtype=torch.bool)
        n_frames = torch.zeros(B, dtype=torch.long)
        uids = []

        for i, b in enumerate(batch):
            T = b["feat"].shape[0]
            src[i, :T] = torch.from_numpy(b["feat"])
            src_pad[i, :T] = False
            n_frames[i] = T
            uids.append(b["uid"])

            ids = b["ids"]
            L = len(ids) - 1
            tgt_in[i, :L] = torch.tensor(ids[:-1], dtype=torch.long)
            tgt_out[i, :L] = torch.tensor(ids[1:], dtype=torch.long)
            tgt_pad[i, :L] = False

        return Batch(src, src_pad, tgt_in, tgt_out, tgt_pad, uids, n_frames)
    return collate


class TokenBudgetBatchSampler(Sampler[List[int]]):
    """Batches by total source frames (~max_tokens/batch), not clip count -- §4: "equal frames
    per step, not equal clips, so both arms see identical data per step". Locally sorts within
    shuffled pools to reduce padding waste; batch order and composition are deterministic given
    (seed, epoch), so both arms see the same batches when run with the same seed."""

    def __init__(self, n_frames: np.ndarray, max_tokens: int = 24576, seed: int = 0,
                shuffle: bool = True, pool_mult: int = 64):
        self.n_frames = n_frames
        self.max_tokens = max_tokens
        self.seed = seed
        self.shuffle = shuffle
        self.pool_mult = pool_mult
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> List[List[int]]:
        idx = np.arange(len(self.n_frames))
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.shuffle:
            rng.shuffle(idx)

        mean_len = max(1, int(self.n_frames.mean()))
        pool_size = max(1, self.max_tokens // mean_len) * self.pool_mult
        batches: List[List[int]] = []
        for start in range(0, len(idx), pool_size):
            pool = idx[start:start + pool_size]
            pool = pool[np.argsort(self.n_frames[pool])]
            cur, cur_tokens = [], 0
            for i in pool:
                nf = int(self.n_frames[i])
                if cur and cur_tokens + nf > self.max_tokens:
                    batches.append(cur)
                    cur, cur_tokens = [], 0
                cur.append(int(i))
                cur_tokens += nf
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        total_frames = self.n_frames.sum()
        return max(1, int(np.ceil(total_frames / self.max_tokens)))
