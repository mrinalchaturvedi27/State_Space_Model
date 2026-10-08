"""Load a trained translation model and a split of its dataset (phase 2 code underneath)."""
from __future__ import annotations

import os

import yaml

from p3 import paths
import slt_reporting as R
from src.data import make_dataset
from src.models import build_model


def resolve(path: str) -> str:
    return path if os.path.exists(path) else os.path.join(paths.PHASE2_DIR, path)


def load_cfg(path: str) -> dict:
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def load(data_yaml: str, model_yaml: str, checkpoint: str, cache_dir: str, device: str, split: str = "val"):
    data_cfg, model_cfg = load_cfg(data_yaml), load_cfg(model_yaml)
    name = data_cfg["dataset"]
    ds = make_dataset(cache_dir, name, split, os.path.join(cache_dir, name, "spm.model"),
                      model_cfg.get("max_tgt_len", 64), 512, augment=False, model_cfg=model_cfg)
    model = build_model(model_cfg, vocab_size=ds.sp.vocab_size(), pad_id=ds.pad_id)
    R.load_checkpoint(checkpoint, model=model, map_location="cpu")
    return model.to(device).eval(), ds, data_cfg, model_cfg
