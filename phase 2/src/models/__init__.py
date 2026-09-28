from __future__ import annotations

from .common import PoseToTextModel

# Phase-2 arm -> (pool_mode, horizon_banks). All share ScopePoolEncoder and mamba's params.
PHASE2_ARMS = {
    "mamba_padfix": ("none", False),
    "mamba_banks": ("none", True),
    "mamba_uniform": ("uniform", True),
    "mamba_pool_matched": ("delta_matched", True),
    "mamba_pool": ("delta", True),
    "mamba_uniform_avg": ("uniform_avg", True),
    "mamba_pool_avg": ("delta_avg", True),
    "mamba_pool_random": ("random_matched", True),
}


def build_model(model_cfg: dict, vocab_size: int, pad_id: int = 0) -> PoseToTextModel:
    """Dispatches on model_cfg['arm']. See configs/model/*.yaml.

    transformer / mamba are the phase-1 arms. PHASE2_ARMS are the phase-2 gate; see
    src/models/mamba_pool.py for what each one isolates.
    """
    arm = model_cfg["arm"]
    d_model = model_cfg["d_model"]
    dropout = model_cfg.get("dropout", 0.1)

    if arm == "transformer":
        from .transformer import TransformerEncoder
        encoder = TransformerEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], n_heads=model_cfg["n_heads"],
            dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout,
            max_len=model_cfg.get("max_src_len", 1024),
        )
    elif arm == "mamba":
        from .mamba import BiMambaEncoder
        encoder = BiMambaEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], d_state=model_cfg["d_state"],
            expand=model_cfg["expand"], headdim=model_cfg["headdim"], d_conv=model_cfg["d_conv"],
            dropout=dropout,
        )
    elif arm in PHASE2_ARMS:
        from .mamba_pool import ScopePoolEncoder
        pool_mode, horizon_banks = PHASE2_ARMS[arm]
        encoder = ScopePoolEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], d_state=model_cfg["d_state"],
            expand=model_cfg["expand"], headdim=model_cfg["headdim"], d_conv=model_cfg["d_conv"],
            dropout=dropout,
            pool_mode=pool_mode, horizon_banks=horizon_banks,
            pool_every_frames=model_cfg.get("pool_every_frames", 16),
            short_heads=model_cfg.get("short_heads", 8),
            mid_heads=model_cfg.get("mid_heads", 4),
            dt_weight_scale=model_cfg.get("dt_weight_scale", 0.05),
            random_sigma=model_cfg.get("random_sigma", 1.0),
        )
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    return PoseToTextModel(
        encoder=encoder, d_model=d_model, vocab_size=vocab_size, d_in=model_cfg.get("d_in", 356),
        n_dec_layers=model_cfg["dec_layers"], n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout, pad_id=pad_id,
        max_tgt_len=model_cfg.get("max_tgt_len", 64), label_smoothing=model_cfg.get("label_smoothing", 0.1),
    )
