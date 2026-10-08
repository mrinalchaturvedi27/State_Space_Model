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
    "mamba_pool_random_avg": ("random_avg", True),
    "mamba_uniform_jitter": ("uniform_jitter", True),
    "mamba_pool_avg_jitter": ("delta_avg_jitter", True),
}


def build_model(model_cfg: dict, vocab_size: int, pad_id: int = 0) -> PoseToTextModel:
    """Dispatches on model_cfg['arm']. See configs/model/*.yaml.

    transformer / mamba are the phase-1 arms. PHASE2_ARMS are the phase-2 gate; see
    src/models/mamba_pool.py for what each one isolates.
    """
    arm = model_cfg["arm"]
    model_cls = PoseToTextModel
    d_model = model_cfg["d_model"]
    dropout = model_cfg.get("dropout", 0.1)

    if arm == "transformer":
        from .transformer import TransformerEncoder
        encoder = TransformerEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], n_heads=model_cfg["n_heads"],
            dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout,
            max_len=model_cfg.get("max_src_len", 1024),
        )
    elif arm in ("transformer_convstem", "transformer_relpos", "tcn"):
        # Step-4 controls (src/models/controls.py): what explains Mamba's advantage?
        from .controls import ConvStemTransformerEncoder, RelPosTransformerEncoder, TCNEncoder
        common = dict(d_model=d_model, dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout)
        if arm == "transformer_convstem":
            encoder = ConvStemTransformerEncoder(n_layers=model_cfg["enc_layers"], n_heads=model_cfg["n_heads"],
                                                 max_len=model_cfg.get("max_src_len", 1024),
                                                 stem_blocks=model_cfg.get("stem_blocks", 2),
                                                 stem_kernel=model_cfg.get("stem_kernel", 5), **common)
        elif arm == "transformer_relpos":
            encoder = RelPosTransformerEncoder(n_layers=model_cfg["enc_layers"], n_heads=model_cfg["n_heads"], **common)
        else:
            encoder = TCNEncoder(n_layers=model_cfg["enc_layers"], kernel=model_cfg.get("tcn_kernel", 5),
                                 dilations=tuple(model_cfg.get("tcn_dilations", [1, 2, 4, 8])), **common)
    elif arm == "transformer_uniform_avg":
        # Control: phase-1 Transformer encoder + the same plain 16-frame averaging as mamba_uniform_avg.
        from .pooled import UniformAvgPooledEncoder
        from .transformer import TransformerEncoder
        encoder = UniformAvgPooledEncoder(TransformerEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], n_heads=model_cfg["n_heads"],
            dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout,
            max_len=model_cfg.get("max_src_len", 1024),
        ), stride=model_cfg.get("pool_every_frames", 16))
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
    elif arm == "mamba_ctx":
        # C1: padfix encoder (phase-1 init, per-clip reversal, no pooling) over [context, current].
        from .context import ContextPoseToTextModel
        from .mamba_pool import ScopePoolEncoder
        encoder = ScopePoolEncoder(
            d_model=d_model, n_layers=model_cfg["enc_layers"], d_state=model_cfg["d_state"],
            expand=model_cfg["expand"], headdim=model_cfg["headdim"], d_conv=model_cfg["d_conv"],
            dropout=dropout, pool_mode="none", horizon_banks=False,
        )
        model_cls = ContextPoseToTextModel
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    return model_cls(
        encoder=encoder, d_model=d_model, vocab_size=vocab_size, d_in=model_cfg.get("d_in", 356),
        n_dec_layers=model_cfg["dec_layers"], n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=dropout, pad_id=pad_id,
        max_tgt_len=model_cfg.get("max_tgt_len", 64), label_smoothing=model_cfg.get("label_smoothing", 0.1),
        bow_weight=model_cfg.get("bow_weight", 0.0),
    )
