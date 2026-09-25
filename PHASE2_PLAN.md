# Phase 2 — from "Mamba wins" to "why, and what only an SSM can do"

Phase 1 (`BENCHMARK_PLAN.md`) answered *whether* a Bi-Mamba-2 encoder beats a matched Transformer
encoder on pose-to-text SLT. Phase 2 has to answer *why*, and turn that into a method contribution.
A benchmark win alone reads as "we swapped the encoder"; reviewers will ask what is new.

Every idea below is tied to a Phase 1 measurement and has a cheap go/no-go test before any
expensive training. Numbers marked **measured** come from the result files in this repo (2026-09-25).

---

## 0. Where Phase 1 actually stands

| dataset | split / decoding | Mamba chrF2 / BLEU-4 | Transformer chrF2 / BLEU-4 | status |
|---|---|---|---|---|
| iSign | **test**, beam 5, 3 seeds | 20.86 / 3.73 | 19.05 / 2.90 | solid; ensembles 4.91 vs 3.98 BLEU, paired bootstrap p < 0.001 |
| PHOENIX-2014T | val, greedy, 3 seeds | 31.86 / 10.03 | 28.99 / 8.30 | every Mamba seed > every TF seed; **no test pass yet**; all 6 runs peak at epoch 38–39/39 (under-trained) |
| How2Sign | val, greedy, 3 seeds | 20.49 / 2.13 | 17.46 / 1.15 | **not a fair comparison yet**: all 3 TF runs and Mamba s1337 early-stopped at epochs 10–16 |

**Measured, iSign test — the Mamba gap grows with source length** (3-seed corpus BLEU-4):

| source frames | n | TF | Mamba | relative gain |
|---|---|---|---|---|
| ≤128 | 1814 | 3.56 | 4.33 | +22% |
| 129–256 | 2822 | 2.90 | 3.72 | +28% |
| 257–512 | 1432 | 2.25 | 3.29 | **+46%** |

That trend is the thread Phase 2 pulls on: **the SSM's advantage is about integrating long temporal
context.** The two contributions below push that past the single-clip boundary (C1) and explain the
mechanism inside the clip (C2).

---

## Stage 2.0 — Make Phase 1 defensible (prerequisite, do first)

The new methods are built on the Mamba baseline, so its known issues have to be fixed first,
otherwise every Phase 2 delta is measured against a moving target.

| # | issue | where | fix | reruns needed |
|---|---|---|---|---|
| F1 | **Pad frames leak into the backward scan.** `zero_pad` zeroes pads, but `LayerNorm(0) = β ≠ 0`, and `flip` then puts every short clip's pads *before* its real frames in the backward direction. A clip's encoding depends on the other clips in its batch. | `src/models/mamba.py:28` | per-sequence reversal (flip within each clip's own length), or varlen `cu_seqlens` packing | Mamba arm, all datasets |
| F2 | **Eval truncates instead of subsampling** for 513–1023-frame clips (`step = T // 512 = 1`, then `[:512]` cuts the sentence end). Contradicts §2 of the plan. | `src/data.py` non-augment branch | `_resample_linear(feat, t_max)` or `np.linspace` index selection | re-eval only (both arms) |
| F3 | **Warmup is fixed at 4,000 steps but steps/epoch scales with dataset frames** (token-budget batching). How2Sign/PHOENIX get far fewer steps per epoch than iSign, so patience-8 fires during warmup (How2Sign) or the LR barely reaches peak (PHOENIX). | `configs/train/base.yaml`, `src/train.py:156–240` | `warmup = min(4000, 0.10 × total_steps)` — keeps iSign ≈ unchanged (~3.5–4k) — and count patience only after warmup ends | both arms, How2Sign + PHOENIX |
| F4 | No test pass for How2Sign / PHOENIX | — | beam-5 test decode from `best.pt` | — |
| F5 | iSign s13 `test_best_beam5.csv` disagrees with its own `metrics.xlsx` (chrF 22.27 vs 21.21; mean pred length 12.1 vs 10.4 words for other seeds) — CSV likely overwritten by a different-length-penalty decode | `results/isign/*/lr0.0003_s13/` | regenerate from `best.pt` with the frozen decoding settings | re-decode only |
| F6 | Efficiency table (§6 of plan) not measured on H200 yet — it underwrites C1's cost claim | `scripts/` (new `bench_efficiency.py`) | memory / latency / throughput at T ∈ {128, 256, 512, 1024, 2048, 4096} | 1 idle GPU, ~2 h |

Confirm F3 on the server first: `grep steps_per_epoch results/*/*/*/results/train.log`.

**Runs:** iSign Mamba × 3 seeds (F1), How2Sign 2 arms × 3 seeds, PHOENIX 2 arms × 3 seeds
→ 15 training runs. Transformer iSign runs are unaffected by F1/F3 and only need re-decoding (F2, F5).

**Exit criterion:** a Phase 1 table where every cell is *test, beam 5, 3 seeds, mean ± std*, plus
the efficiency table. If the Mamba gap survives F1–F3 on all three datasets, Phase 1 is paper-ready.

---

## Contribution C1 (primary) — the SSM state as discourse memory

### Rationale
Current SLT systems (both our arms) translate each clip in isolation. But iSign and How2Sign clips are
consecutive sentences from the same video — a story, a lecture — and **the previous sentences carry a
large share of the current sentence's content.** Measured on the iSign test references
(197 videos, 6,005 segments, median 23 segments/video; 63 uids with non-numeric suffixes skipped):

| previous clips k | current-sentence content words already seen in the previous k references, same video | same, random other video (chance) |
|---|---|---|
| 1 | 9.0% | 0.5% |
| 2 | 14.4% | 0.9% |
| 4 | 20.4% | 1.6% |
| 8 | **26.0%** | 2.4% |

37% of capitalised name tokens in test references (e.g. *Bobby*, *Sundari*) already appeared earlier in
the same video. Isolated-clip models recover only a fraction of them (Mamba-ens 28%, TF-ens 20% —
a crude string-match estimate), because a name that was fingerspelled once is typically a short
name-sign afterwards, which is ambiguous without context.

This is ~10× above chance, so context is information-rich *on the text side*. Whether the model
can extract it *from the previous clips' poses* is exactly the experiment.

**Why this is an SSM contribution, not just "add context":** a Transformer encoder over k previous clips
costs O((k+1)²·T²) attention and its memory grows with k. The Mamba encoder's cost grows linearly, and
its forward-direction state is a **fixed-size summary** that can be carried across clips at O(1) memory
— effectively streaming, story-level translation. Phase 1 already shows the SSM's gain grows with
sequence length; C1 extends "length" past the clip boundary, where the SSM advantage should be largest.

**Prior art to position against (verify before writing):** context-aware SLT exists — e.g. Sincan,
Camgöz & Bowden, *"Is context all you need? Scaling neural sign language translation to large domains
of discourse"* (ICCV 2023 workshops, BOBSL, uses previous-sentence context). Novelty is therefore **not**
"context helps"; it is *(a)* recurrent-state context with linear/constant cost, *(b)* the controlled
SSM-vs-attention comparison as k grows, *(c)* pose-only, on iSign/How2Sign. Also search arXiv for
existing Mamba-for-SLT/SLR work so the paper does not claim "first Mamba SLT".

### Method
Input for clip i = `[clip_{i-k}, …, clip_{i-1}, clip_i]` from the same video (pose frames only).
- **Never** the reference text of previous clips (label leakage). Previous *poses* are legitimate
  input; test context comes from the test split itself, which is video-disjoint from train.
- A learned **boundary embedding** added at each clip's first frame + a context/current type embedding.
- The decoder cross-attends to the **current clip's encoder states only**. Context can reach the
  translation only through the encoder (Mamba recurrence / Transformer self-attention). Same decoder,
  same one-variable-changed discipline as Phase 1.
- Cap total context frames (default 2,048; sweep in ablation).
- Streaming inference variant (Mamba only): carry the forward scan's final SSM state + last `d_conv−1`
  frames across clips instead of re-encoding the context. Verify numerically that it matches the
  concatenated forward pass, then report its latency/memory. (`mamba_chunk_scan_combined` accepts
  `initial_states` / `return_final_states`; needs a thin Mamba2 wrapper, since the fused
  `Mamba2.forward` path does not expose them.)

Batching: keep the token budget on **current-clip frames** so the number of target sentences per step
is identical to k = 0; context frames are extra compute, reported explicitly.

### Experiments
| id | arm | k | purpose |
|---|---|---|---|
| C1.0 | both | 0 | = Stage 2.0 baselines |
| C1.1 | both | 1, 2, 4, 8 | quality vs context length (the headline plot) |
| C1.2 | Mamba streaming | 1–32 | state-carry: quality & constant memory beyond what fits for TF |
| C1.3 | TF, compressive | 1, 2, 4, 8 | strongest fair attention baseline: previous clips summarised into M memory tokens |
| C1.4 | both | best k | shuffled-context control (context from a *random* video) — proves gains come from discourse, not from extra frames |

Metrics: chrF2 / BLEU-4 (test, beam 5, 3 seeds) + **recurring-name recall** (the context-specific metric
from the table above) + peak memory and latency vs k.

**Go/no-go gate (cheap, first):** iSign, Mamba, k ∈ {0, 2}, seed 42 only. If k = 2 does not beat
k = 0 by more than seed noise (~0.5 chrF), and the shuffled-context control does not fall back to
the k = 0 level, stop C1 and promote C2.

---

## Contribution C2 (secondary) — selectivity that tracks signing kinematics

### Rationale
Mamba's step size Δ_t decides, per frame and per head, how much to write into vs retain the state.
Signing alternates between **holds** (lexical content) and **transitions / movement epenthesis**
(between signs). If Δ_t correlates with hand kinematics, the SSM has learned unsupervised sign
segmentation — an interpretable *why* for the Phase 1 win that attention has no analogue for.

### Step 1 — analysis on existing checkpoints (no training)
Hook each Mamba2 `in_proj`: the last `nheads` (= 16) output channels are the raw dt, and
`Δ = softplus(dt_raw + dt_bias)`. Over the iSign test set, correlate per-layer, per-head Δ_t (forward and
backward) with wrist speed / acceleration and handshape change rate. For boundary ground truth, use an
off-the-shelf pose-based sign segmenter (e.g. Moryossef et al., *Linguistically Motivated Sign Language
Segmentation*, Findings of EMNLP 2023, built on the same `pose-format` stack — verify it runs on
our `.pose` files). Deliverable: a figure of Δ over a clip with predicted sign boundaries overlaid.

**Gate:** a clear, head-specific correlation (|ρ| ≫ random-init model) → proceed to step 2. Either way
step 1 is a paper figure.

### Step 2 — motion-gated Δ
`dt_raw ← dt_raw + W_k · kin_t`, where `kin_t` = wrist/fingertip velocity and acceleration (computed
from the cached coordinates, no new preprocessing). Implemented by wrapping `in_proj` to add the
kinematic term to its dt slice, which keeps the fused kernels. Mandatory control: feed the same
kinematic features as extra **input** to both arms (front-end concat), which separates "better features"
from "better gating".

---

## Backup — C3: articulator-factored scanning
Separate scans for hands / face / body with light cross-stream fusion (manual and non-manual channels are
asynchronous in sign languages). Plausible, but multi-stream designs are common in sign recognition, so
the novelty margin is thinner. Only if C1 fails its gate.

---

## Order of work and compute

| stage | what | runs (approx.) | gate |
|---|---|---|---|
| 2.0 | fixes F1–F6, reruns, test decodes, efficiency table | 15 train + re-decodes | Mamba gap survives on all 3 datasets |
| 2.1 | C1 gate (iSign, Mamba, k ∈ {0,2}, s42) ‖ C2 step 1 (analysis only) | 2 | k=2 > k=0 beyond noise; Δ–kinematics correlation |
| 2.2 | C1 full on iSign: both arms, k grid, controls, 3 seeds at best k | ~20 | — |
| 2.3 | C1 on How2Sign (best k, both arms, 3 seeds); C2 step 2 on iSign | ~10 | — |
| 2.4 | write-up | — | — |

2.0 and 2.1 can overlap: C2 step 1 needs only the existing checkpoints, and the C1 data loader can be built
while the 2.0 reruns train. PHOENIX gets C1 only if its segments turn out to be consecutive within a
broadcast (check ids first); otherwise it stays a single-clip dataset in the paper.

## Paper shape this leads to
1. Controlled SSM-vs-Transformer SLT benchmark, 3 sign languages, matched params, gap grows with length.
2. **Recurrent state as discourse memory** — quality vs context k, and constant memory vs attention.
3. **What the selection mechanism learns** — Δ aligns with sign kinematics (+ motion-gated variant if it helps).
