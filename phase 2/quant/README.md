# Decoder-memory quantization (post-hoc, no training)

Does TurboQuant-style vector quantization stack with Δ-weighted memory compression?
Δ pooling cuts the number of memory tokens (~15x); quantization cuts the bits per token.

| file | what |
|---|---|
| `turboquant.py` | `turbo` (random rotation + Lloyd-Max Gaussian codebook + fp16 norm), `turbo_prod` (turbo at b-1 bits + 1-bit QJL residual: unbiased inner products), `naive` (per-token min-max) |
| `quant_eval.py` | loads a checkpoint, quantizes the memory the decoder cross-attends to, decodes, writes quality / bytes / reconstruction error per setting |
| `run_quant.sh` | the server sweep (3 models, 12 settings, greedy; beam-5 confirmation) |
| `test_turboquant.py`, `test_quant_eval.py` | CPU tests (codebook vs Lloyd-Max tables, distortion vs theory, unbiasedness, end-to-end eval) |

Bytes are for storing the encoder memory (512 values per token), which is already 6x smaller
than caching cross-attention K/V for the 3 decoder layers. Verify the TurboQuant reference
(Zandieh et al., Google Research, 2025) before citing; this is a reimplementation, not their code.
