"""Stitched story-level pose data for self-supervised pretraining (Stage 3.1).

Clips are grouped into their source video by uid and ordered by segment number, then read from the
same memmap cache the translation models use (no new preprocessing). Two modes over the SAME frames:
  clip  every clip is one item (the control: pretraining without long context)
  long  each video is cut into windows of up to `window` frames, with a random phase per epoch,
        so every frame is seen once per epoch in both modes
Each frame also carries an is_start flag (1 on the first frame of every clip), so clip boundaries
and gaps left by filtered segments are visible to the model if needed.
"""
from __future__ import annotations

import json
import os
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

PATTERNS = [
    re.compile(r"^(.+?)_(\d+)-\d+-rgb_front$"),  # How2Sign uid: <video>_<sentence>-<take>-rgb_front
    re.compile(r"^(.*?)-+(\d+)$"),               # iSign <video>-<segment>; PHOENIX <broadcast>-<n>
    re.compile(r"^(.+)_(\d+)$"),
]


def parse_uid(uid: str):
    """(video id, segment number), or (uid, -1) when no pattern matches (a one-clip 'video')."""
    for p in PATTERNS:
        m = p.match(uid)
        if m:
            return m.group(1), int(m.group(2))
    return uid, -1


class VideoIndex:
    """Clips of one split grouped into videos, in segment order."""

    def __init__(self, cache_dir: str, dataset: str, split: str):
        ds_dir = os.path.join(cache_dir, dataset)
        idx = pd.read_parquet(os.path.join(ds_dir, f"{split}_index.parquet"))
        with open(os.path.join(ds_dir, f"{split}.shape.json")) as f:
            shape = json.load(f)
        self.memmap = np.memmap(os.path.join(ds_dir, f"{split}.memmap"), dtype=shape["dtype"], mode="r",
                                shape=tuple(shape["shape"]))
        parsed = idx["uid"].astype(str).map(parse_uid)
        idx = idx.assign(video=parsed.map(lambda p: p[0]), seg=parsed.map(lambda p: p[1]))
        self.unparsed = int((idx.seg < 0).sum())
        self.videos = []  # list of (video id, offsets array, n_frames array)
        for vid, g in idx.sort_values(["video", "seg"]).groupby("video", sort=True):
            self.videos.append((vid, g["offset"].to_numpy(np.int64), g["n_frames"].to_numpy(np.int64)))
        self.dim = shape["shape"][1]

    def stats(self) -> dict:
        clips = np.array([len(o) for _, o, _ in self.videos])
        frames = np.array([n.sum() for _, _, n in self.videos])
        return {"videos": len(self.videos), "clips": int(clips.sum()), "unparsed_uids": self.unparsed,
                "clips_per_video_median": float(np.median(clips)), "clips_per_video_max": int(clips.max()),
                "frames_per_video_median": float(np.median(frames)), "frames_per_video_max": int(frames.max()),
                "frames_total": int(frames.sum()), "videos_over_4096_frames": int((frames > 4096).sum())}


def light_augment(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase 1/2's spatial augmentation (noise sigma 0.01 + one random 2D affine per item), without
    the temporal resampling / frame drop, so masked targets stay aligned with their frames."""
    T, D = feat.shape
    feat = feat + rng.normal(0.0, 0.01, size=feat.shape).astype(np.float32)
    xy = feat.reshape(T, D // 2, 2)
    th = np.radians(rng.uniform(-5, 5))
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    xy = (xy @ rot.T) * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    return xy.reshape(T, D).astype(np.float32)


class PretrainDataset(Dataset):
    def __init__(self, vindex: VideoIndex, mode: str = "long", window: int = 4096, seed: int = 0,
                 augment: bool = True):
        if mode not in ("clip", "long"):
            raise ValueError(mode)
        self.v, self.mode, self.window, self.seed, self.augment = vindex, mode, window, seed, augment
        self.starts = [np.concatenate([[0], np.cumsum(n)[:-1]]) for _, _, n in vindex.videos]
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        items = []
        if self.mode == "clip":
            for vi, (_, _, n) in enumerate(self.v.videos):
                for ci, L in enumerate(n):
                    items.append((vi, int(self.starts[vi][ci]), int(self.starts[vi][ci] + L)))
        else:
            rng = np.random.default_rng([self.seed, epoch, 17])
            for vi, (_, _, n) in enumerate(self.v.videos):
                total = int(n.sum())
                phase = int(rng.integers(self.window)) if total > self.window else 0
                edges = sorted({0, total, *range(phase, total, self.window)})
                # Every piece is kept, even a short one at the phase cut, so both modes see
                # exactly the same frames each epoch (P1 vs P2 differ only in context length).
                items += [(vi, a, b) for a, b in zip(edges[:-1], edges[1:])]
        self.items = items

    def lengths(self) -> np.ndarray:
        return np.array([b - a for _, a, b in self.items], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        vi, a, b = self.items[i]
        _, offsets, n = self.v.videos[vi]
        st = self.starts[vi]
        parts, flags = [], []
        for ci in np.nonzero((st < b) & (st + n > a))[0]:
            lo, hi = max(a, st[ci]), min(b, st[ci] + n[ci])
            seg = np.asarray(self.v.memmap[offsets[ci] + (lo - st[ci]): offsets[ci] + (hi - st[ci])], dtype=np.float32)
            parts.append(seg)
            f = np.zeros(hi - lo, dtype=np.float32)
            if lo == st[ci]:
                f[0] = 1.0
            flags.append(f)
        feat = np.concatenate(parts)
        if self.augment:
            feat = light_augment(feat, np.random.default_rng([self.seed, self.epoch, i]))
        return {"feat": feat, "is_start": np.concatenate(flags), "video": self.v.videos[vi][0]}


class EpochTokenSampler(Sampler[List[int]]):
    """Token-budget batches over the dataset's current item lengths (recomputed every epoch, since
    long-mode windows move). Same budget in both modes = same frames per optimizer step."""

    def __init__(self, ds: PretrainDataset, max_tokens: int, seed: int = 0, shuffle: bool = True):
        self.ds, self.max_tokens, self.seed, self.shuffle, self.epoch = ds, max_tokens, seed, shuffle, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        lens = self.ds.lengths()
        idx = np.arange(len(lens))
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.shuffle:
            rng.shuffle(idx)
        pool = max(1, self.max_tokens // max(1, int(lens.mean()))) * 64
        batches = []
        for s in range(0, len(idx), pool):
            p = idx[s:s + pool]
            p = p[np.argsort(lens[p], kind="stable")]
            cur, tok = [], 0
            for i in p:
                if cur and tok + lens[i] > self.max_tokens:
                    batches.append(cur)
                    cur, tok = [], 0
                cur.append(int(i))
                tok += int(lens[i])
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return max(1, int(np.ceil(self.ds.lengths().sum() / self.max_tokens)))


def collate(batch: list[dict]) -> dict:
    B, T, D = len(batch), max(b["feat"].shape[0] for b in batch), batch[0]["feat"].shape[1]
    feat = torch.zeros(B, T, D)
    pad = torch.ones(B, T, dtype=torch.bool)
    starts = torch.zeros(B, T)
    for i, b in enumerate(batch):
        L = b["feat"].shape[0]
        feat[i, :L] = torch.from_numpy(b["feat"])
        pad[i, :L] = False
        starts[i, :L] = torch.from_numpy(b["is_start"])
    return {"feat": feat, "pad": pad, "is_start": starts}


class MultiPretrainDataset(Dataset):
    """Several PretrainDatasets (e.g. iSign + How2Sign + PHOENIX train poses) as one. Items keep
    their own video and frames; lengths(), set_epoch() and indexing are delegated, so the token
    sampler and both modes work unchanged. Pretraining needs no text, so a dataset's target-text
    issues (How2Sign) do not matter here."""

    def __init__(self, parts: list[PretrainDataset]):
        self.parts = parts
        self._offsets()

    def _offsets(self) -> None:
        self.bounds = np.cumsum([0] + [len(p) for p in self.parts])

    def set_epoch(self, epoch: int) -> None:
        for p in self.parts:
            p.set_epoch(epoch)
        self._offsets()  # long-mode window counts change with the epoch's phase

    def lengths(self) -> np.ndarray:
        return np.concatenate([p.lengths() for p in self.parts])

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def __getitem__(self, i: int) -> dict:
        k = int(np.searchsorted(self.bounds, i, side="right") - 1)
        return self.parts[k][i - int(self.bounds[k])]
