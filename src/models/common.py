"""Shared building blocks -- identical across arms, per BENCHMARK_PLAN.md §3."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FrontEnd(nn.Module):
    """Linear(d_in -> d_model) -> LayerNorm -> Dropout. Identical for both arms (§2):
    no conv downsampling stem, since that would shorten exactly the sequences whose length
    is the object of study."""

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj(x)))


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal PE. Used for the Transformer encoder and for the decoder
    (both arms) -- the Mamba encoder is the only place PE is deliberately omitted (§3)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.dtype)


def build_decoder(d_model: int, n_layers: int = 3, n_heads: int = 8,
                  dim_feedforward: int = 2048, dropout: float = 0.1) -> nn.TransformerDecoder:
    """The 3-layer Transformer decoder with cross-attention, held byte-identical across
    both arms (§3's "decoder held byte-identical" comparison).

    `norm=` (final LayerNorm) is required with norm_first=True: pre-LN's residual stream is
    not renormalized between layers by construction, so without a final norm its magnitude
    grows with depth and blows up the tied-embedding output projection (embeddings default to
    std=1, so an unnormalized residual stream produced logit std in the hundreds at random init
    -- caught via a smoke test where untrained loss was ~11,000 instead of ~ln(vocab_size))."""
    layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


class PoseToTextModel(nn.Module):
    """Encoder-agnostic seq2seq wrapper: FrontEnd -> {Transformer,Mamba} encoder -> shared
    Transformer decoder -> tied output projection. `encoder` must expose
    forward(x, src_key_padding_mask=...) -> (B, T, d_model)."""

    def __init__(self, encoder: nn.Module, d_model: int, vocab_size: int, d_in: int = 356,
                n_dec_layers: int = 3, n_heads: int = 8, dim_feedforward: int = 2048,
                dropout: float = 0.1, pad_id: int = 0, max_tgt_len: int = 64,
                label_smoothing: float = 0.1):
        super().__init__()
        self.front_end = FrontEnd(d_in, d_model, dropout)
        self.encoder = encoder
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # nn.Embedding's default init is std=1, which is far too large once tied to out_proj:
        # decode() scales the embedding output by sqrt(d_model) on the way in (standard practice),
        # which only gives unit-variance input if the embedding itself has std ~= d_model**-0.5.
        # Left at the default std=1, tied logits came out with std ~400+ at random init (loss
        # ~11,000 instead of ~ln(vocab_size)) -- caught via a smoke test, fixed here explicitly.
        nn.init.normal_(self.tgt_embed.weight, mean=0.0, std=d_model ** -0.5)
        self.tgt_pos = SinusoidalPositionalEncoding(d_model, max_tgt_len + 8)
        self.tgt_dropout = nn.Dropout(dropout)
        self.decoder = build_decoder(d_model, n_dec_layers, n_heads, dim_feedforward, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.tgt_embed.weight  # tied embeddings (§3)
        self.pad_id = pad_id
        self.d_model = d_model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.front_end(src)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
              tgt_key_padding_mask: torch.Tensor | None = None,
              memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.tgt_embed(tgt_ids) * math.sqrt(self.d_model)
        x = self.tgt_dropout(self.tgt_pos(x))
        tgt_mask = causal_mask(tgt_ids.size(1), tgt_ids.device)
        h = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        return self.out_proj(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor,
               src_key_padding_mask: torch.Tensor | None = None,
               tgt_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask)
        return self.decode(tgt_in, memory, tgt_key_padding_mask, src_key_padding_mask)

    def compute_loss(self, src, tgt_in, tgt_out, src_key_padding_mask=None, tgt_key_padding_mask=None):
        logits = self.forward(src, tgt_in, src_key_padding_mask, tgt_key_padding_mask)
        return self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

    def no_decay_params(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Split params into (decay, no_decay) for AdamW per §4: no weight decay on biases,
        all norms, and Mamba's A_log/D/dt_bias."""
        decay, no_decay = [], []
        no_decay_names = {"bias", "A_log", "D", "dt_bias"}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf in no_decay_names or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay
