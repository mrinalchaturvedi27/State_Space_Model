# E2 -- phase-2 gate curves (Proposal A premise)

## Seed noise (phase 1, iSign, lr 3e-4, best val chrF2)

| arm | seeds | best chrF2 | sd |
|---|---|---|---|
| mamba | 13, 42, 1337 | 20.98, 20.74, 20.73 | 0.14 |
| transformer | 13, 42, 1337 | 19.29, 18.92, 19.16 | 0.19 |

## Phase-2 runs

| arm | seed | epochs | best chrF2 (epoch) | last chrF2 | slope last 5 ep | val kept_frac |
|---|---|---|---|---|---|---|
| mamba_banks | 42 | 32 | 21.06 (23) | 21.00 | +0.061/ep | 1.000 |
| mamba_padfix | 42 | 29 | 21.06 (20) | 20.94 | +0.011/ep | 1.000 |
| mamba_pool_matched | 42 | 40 | 20.36 (34) | 20.15 | -0.008/ep | 0.070 |
| mamba_uniform | 42 | 28 | 19.63 (19) | 19.50 | -0.013/ep | 0.070 |

## Δ vs uniform at the same frame count

### mamba_pool_matched - mamba_uniform

| seed | best-vs-best gap | epochs compared | per-epoch gap mean (ep >= 10) | epochs with gap > 0 (ep >= 10) |
|---|---|---|---|---|
| 42 | +0.73 | 28 | +0.36 | 17/18 |
