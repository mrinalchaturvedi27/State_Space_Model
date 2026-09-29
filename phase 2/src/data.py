"""Dataset, train-time augmentation, and token-budget batching (BENCHMARK_PLAN.md §2, §4).

Reads the memmap + Parquet index written by scripts/build_cache.py. Augmentation uses a
per-(seed, epoch, idx) RNG so the exact same augmented tensors can be reproduced for both
arms when trained with the same seed -- "identical, same seed stream, applied to both arms" (§2).
"""
from __future__ import annotations

import json
import os
import re
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
            # Uniform subsample over the whole clip (§2). The phase-1 `feat[::T // t_max][:t_max]`
            # has step 1 for 513-1023 frames and so cut off the end of the sentence.
            keep = np.linspace(0, feat.shape[0] - 1, self.t_max).round().astype(int)
            feat = feat[keep]

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
            n_frames[i] = b.get("n_cur", T)  # context datasets: the current clip's length only
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


_SEGMENT_UID = re.compile(r"^(.*?)-+(\d+)$")


class ContextPoseTextDataset(PoseTextDataset):
    """C1 (PHASE2_PLAN.md): each item is [previous clips of the same video ..., current clip].

    Only pose frames of earlier clips are added -- never their text, which would leak labels;
    val/test context comes from the same split, which is video-disjoint from train. Two flag
    channels are appended to every frame so the model can find the parts without any change
    to the train/eval loops: is_context (1 on context frames) and is_start (1 on each clip's
    first frame). Feature width becomes D + 2.

    mode="prev": the k clips just before this one in its video (nearest first until
    max_context_frames is used up; the oldest is cut from its start if needed).
    mode="random": k consecutive clips from a different, random video -- the control that
    separates discourse context from "more frames". Fixed per item at eval; redrawn per epoch
    in training. k=0 gives the plain clip with the flag channels (the C1 code-path baseline).
    Clips whose uid has no numeric segment suffix get no context.
    """

    def __init__(self, *args, context_clips: int = 2, context_mode: str = "prev",
                 max_context_frames: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        if context_mode not in ("prev", "random"):
            raise ValueError(f"unknown context_mode: {context_mode!r}")
        self.k = context_clips
        self.mode = context_mode
        self.max_context_frames = max_context_frames
        videos: dict[str, list[tuple[int, int]]] = {}
        self.video_of = [None] * len(self.index)
        for i, uid in enumerate(self.index["uid"].astype(str)):
            m = _SEGMENT_UID.match(uid)
            if m:
                videos.setdefault(m.group(1), []).append((int(m.group(2)), i))
                self.video_of[i] = m.group(1)
        self.prev: list[list[int]] = [[] for _ in range(len(self.index))]
        self.video_clips: dict[str, list[int]] = {}
        for vid, segs in videos.items():
            order = [i for _, i in sorted(segs)]
            self.video_clips[vid] = order
            for pos, i in enumerate(order):
                self.prev[i] = order[max(0, pos - self.k):pos][::-1]  # nearest first
        self.video_ids = sorted(self.video_clips)

    def _context_indices(self, idx: int) -> list[int]:
        if self.k == 0 or self.video_of[idx] is None:
            return []
        if self.mode == "prev":
            return self.prev[idx]
        rng = np.random.default_rng([self.seed, self.epoch if self.augment else 0, int(idx), 7])
        own = self.video_of[idx]
        for _ in range(10):
            vid = self.video_ids[int(rng.integers(len(self.video_ids)))]
            if vid != own:
                break
        clips = self.video_clips[vid]
        end = int(rng.integers(len(clips))) + 1
        return clips[max(0, end - self.k):end][::-1]

    def _clip(self, idx: int, rng: np.random.Generator | None) -> np.ndarray:
        row = self.index.iloc[idx]
        feat = np.asarray(self.memmap[row.offset: row.offset + row.n_frames], dtype=np.float32)
        if rng is not None:
            feat = augment_clip(feat, rng, self.t_max)
        elif feat.shape[0] > self.t_max:
            keep = np.linspace(0, feat.shape[0] - 1, self.t_max).round().astype(int)
            feat = feat[keep]
        return feat

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        cur = item["feat"]
        parts, budget = [], self.max_context_frames
        for j, c in enumerate(self._context_indices(idx)):
            if budget <= 0:
                break
            rng = np.random.default_rng([self.seed, self.epoch, int(idx), j + 1]) if self.augment else None
            clip = self._clip(c, rng)[-budget:]
            parts.append(clip)
            budget -= clip.shape[0]
        parts = parts[::-1] + [cur]  # oldest first, current clip last
        feat = np.concatenate(parts, axis=0)
        flags = np.zeros((feat.shape[0], 2), dtype=np.float32)
        start = 0
        for p in parts:
            flags[start, 1] = 1.0
            start += p.shape[0]
        flags[: feat.shape[0] - cur.shape[0], 0] = 1.0
        item["feat"] = np.concatenate([feat, flags], axis=1)
        item["n_cur"] = cur.shape[0]
        return item


def make_dataset(cache_dir: str, dataset: str, split: str, spm_model: str, max_tgt_len: int,
                 t_max: int, augment: bool = False, seed: int = 0,
                 model_cfg: dict | None = None) -> PoseTextDataset:
    """ContextPoseTextDataset when the model config sets context_clips, else PoseTextDataset."""
    if model_cfg is not None and model_cfg.get("context_clips") is not None:
        return ContextPoseTextDataset(
            cache_dir, dataset, split, spm_model, max_tgt_len, t_max, augment=augment, seed=seed,
            context_clips=model_cfg["context_clips"],
            context_mode=model_cfg.get("context_mode", "prev"),
            max_context_frames=model_cfg.get("max_context_frames", 1024))
    return PoseTextDataset(cache_dir, dataset, split, spm_model, max_tgt_len, t_max,
                           augment=augment, seed=seed)
