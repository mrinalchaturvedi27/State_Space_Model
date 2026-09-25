from __future__ import annotations

from .common import PoseToTextModel
from .mamba import BiMambaEncoder
from .transformer import TransformerEncoder


def build_model(model_cfg: dict, vocab_size: int, pad_id: int = 0) -> PoseToTextModel:
    """Dispatches on model_cfg['arm'] ('transformer' or 'mamba'). See configs/model/*.yaml."""
    arm = model_cfg["arm"]
    d_model = model_cfg["d_model"]
    dropout = model_cfg.get("dropout", 0.1)

    if arm == "transformer":
        encoder = TransformerEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], n_heads=model_cfg["n_heads"],
            dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout,
            max_len=model_cfg.get("max_src_len", 1024),
        )
    elif arm == "mamba":
        encoder = BiMambaEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], d_state=model_cfg["d_state"],
            expand=model_cfg["expand"], headdim=model_cfg["headdim"], d_conv=model_cfg["d_conv"],
            dropout=dropout,
        )
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    return PoseToTextModel(
        encoder=encoder, d_model=d_model, vocab_size=vocab_size, d_in=model_cfg.get("d_in", 356),
        n_dec_layers=model_cfg["dec_layers"], n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout, pad_id=pad_id,
        max_tgt_len=model_cfg.get("max_tgt_len", 64), label_smoothing=model_cfg.get("label_smoothing", 0.1),
    )
