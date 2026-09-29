"""End-to-end CPU check of quant/quant_eval.py on a tiny cache with stand-in Mamba2 and
SentencePiece (from scripts/test_context.py). Run: python quant/test_quant_eval.py"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)
import test_context as tc  # noqa: E402  (stubs sentencepiece + mamba_ssm, builds a tiny cache)
import quant_eval  # noqa: E402
from src.models import build_model  # noqa: E402


def run(arm: str, settings: str, decode: str = "greedy") -> list[dict]:
    with tempfile.TemporaryDirectory() as root:
        tc.make_cache(root)
        cfg = {**tc.arms.CFG, "arm": arm, "d_in": tc.D}
        model_yaml, data_yaml = os.path.join(root, "m.yaml"), os.path.join(root, "d.yaml")
        yaml.safe_dump(cfg, open(model_yaml, "w"))
        yaml.safe_dump({"dataset": "isign"}, open(data_yaml, "w"))
        ckpt_dir = os.path.join(root, "run", "checkpoints")
        os.makedirs(ckpt_dir)
        torch.manual_seed(0)
        torch.save({"model": build_model(cfg, vocab_size=30).state_dict(), "epoch": 7},
                   os.path.join(ckpt_dir, "best.pt"))
        sys.argv = ["quant_eval", "--data", data_yaml, "--model", model_yaml, "--checkpoint",
                    os.path.join(ckpt_dir, "best.pt"), "--cache-dir", root, "--split", "val",
                    "--settings", settings, "--decode", decode, "--num-workers", "0"]
        quant_eval.main()
        out = os.path.join(root, "run", "quant")
        rows = list(csv.DictReader(open(os.path.join(out, "summary.csv"))))
        assert all(os.path.exists(os.path.join(out, f)) for f in os.listdir(out))
        return rows


def test_pooled_arm_all_methods_greedy():
    rows = run("mamba_pool_avg", "fp16,turbo:4,turbo:2,naive:2,turbo_prod:3")
    by = {r["setting"]: r for r in rows}
    assert set(by) == {"fp16", "turbo:4", "turbo:2", "naive:2", "turbo_prod:3"}
    assert all(int(r["n"]) == len(tc.CLIPS) for r in rows)
    assert float(by["fp16"]["rel_mse"]) < 1e-5
    assert float(by["turbo:4"]["rel_mse"]) < float(by["turbo:2"]["rel_mse"])
    assert float(by["turbo:2"]["bytes_per_clip"]) < float(by["turbo:4"]["bytes_per_clip"]) < float(by["fp16"]["bytes_per_clip"])
    # pooled arm: fewer memory tokens than frames (stride 4 in the test config)
    assert float(by["fp16"]["mem_tokens_per_clip"]) < 5.5


def test_full_memory_arm_beam():
    rows = run("mamba_padfix", "fp16,turbo:3", decode="beam")
    assert [r["decode"] for r in rows] == ["beam5", "beam5"]
    # full memory keeps every frame: mean clip length of the tiny cache
    assert abs(float(rows[0]["mem_tokens_per_clip"]) - sum(n for n, _ in tc.CLIPS.values()) / len(tc.CLIPS)) < 0.01  # CSV rounds to 2 decimals


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ok")
