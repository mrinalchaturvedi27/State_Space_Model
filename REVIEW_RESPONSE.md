# Response to the review: results of the suggested experiments

**Follows:** `EXPERT_BRIEF.md` (what you reviewed) and your research program of 8 October 2026 (`RESEARCH_PROGRAM_2026-10-08.md`).
**Revised:** 12 October 2026, after your feedback (`REVIEW_FEEDBACK_2026-10-12.md`). Section 0 lists what changed. All numbers below are recomputed.
**Updated:** 13 October 2026, with the corrected How2Sign results (§6, How2Sign).

We followed your order: audit → correct-video dependence → same-clip temporal controls → matched training controls → confirmation. Experiment IDs (E0–E13, C1–C12) are yours.

## 0. Changes after your 12 October feedback

| Your point | What we did | Effect |
|---|---|---|
| Source-video grouping incomplete | A new `source_video()` groups iSign dictionary clips (`<id>_d`, `<id>_e1`–`_e4`; 19 test and 34 val families) by source video for all statistics and wrong-video partners. The stitching parser keeps them apart, since a definition plus unrelated examples isn't a story. iSign test has **216** source videos, not 260. Every grouped interval was recomputed | Point estimates unchanged; intervals moved by ≤0.05. One conclusion changed: the TCN's BLEU-4 lead on PHOENIX is no longer significant |
| E1 partners chosen with the same parser | `check_video_partners.py` maps every partner to its source video and recomputes E1 without same-video pairs. It runs on the server (it needs the cache index) | Pending |
| Content F1 changed definition (set, not clipped multiset) | All content F1 is now clipped-multiset. Your numbers are reproduced exactly (Transformer 10.15, Mamba 14.05, TCN 14.29, input-independent 14.94). Number-token F1 and negation recall / false-negation rate are reported separately | Ranking unchanged |
| Central claim too strong | Replaced with your scoped wording (§1) | — |
| Labels | "Mamba with input-independent scan coefficients" (was "non-selective Mamba"); "wide-receptive-field TCN" (was "whole-clip TCN"). We now note that the decoder attends globally to all encoder outputs | — |
| Parameter bookkeeping | The input-independent arm leaves 737,280 `in_proj` weights unused: **33.27M active** vs 34.01M for selective Mamba. Reported, with regularisation and parameterisation as open explanations | — |
| Finish corrected How2Sign | Done: 4 encoders × 3 seeds on consistently cased targets, plus order diagnostics (§6) | Reproduces the iSign ordering; adds a boundary-free long-context result |
| Boundaries vs context | New window diagnostic and training controls (§8) | Running |
| One-exponential-per-head kernel | Confirmed against the implementation: A is a scalar per head and ngroups = 1, so with constant B, C and Δ each head's long-range filter is a single decaying exponential, `k_h(ℓ) = (C·B) Δ_h exp(−e^{A_log,h} Δ_h ℓ)`, with C·B shared across heads. `inspect_retention.py` reads the learned time constants from the checkpoints | Running (CPU) |
| Independent LTI baseline | S4D encoder added (diagonal complex SSM, S4D-Lin init, no gating inside the SSM; +1.20% parameters vs the Transformer) | Queued, 3 seeds |
| Softer readings | "Weak order sensitivity relative to Mamba" (shuffling every frame lowers the Transformer's greedy BLEU-4 from 2.63 to 1.89). The topic-matched overlap result is no longer read as a share of recognition | — |
| Efficiency | Kept to a table; absolute differences only; GPU-hours to the selected checkpoint added to the plan | — |

## 1. Summary

1. **Both encoders depend on the correct video** (E1). The correct pose scores the reference higher than a length- and topic-matched wrong video for 94–98% of clips, and free decoding loses almost all content-word overlap without it. (Partner check pending; §0.)
2. **The Transformer is weakly order-sensitive relative to Mamba** (E2).
   - Shuffling every frame costs the Transformer −0.35 greedy chrF2 and −0.74 BLEU-4 (2.63 → 1.89). It costs Mamba −4.21 chrF2.
   - The Transformer's sensitivity to sampling density disappears when positions follow original time (−1.97 → −0.12).
3. **A TCN recovers most of the Transformer → Mamba improvement** (E5, parameter-matched, 3 seeds). On chrF2 the TCN is 0.41 below Mamba on iSign, against the Transformer's 2.30; 0.81 below on PHOENIX, against 2.19; and 0.40 below on How2Sign, against 1.69. Its BLEU-4 difference from Mamba is not significant on any of the three.
   - This is a performance comparison, not a causal estimate of how much of the original gain locality caused.
4. **Wider encoder context appears to help on longer clips** (E3 and the window/receptive-field controls).
   - The Mamba − Transformer gap grows with clip length, including at fixed sentence length for medium and long sentences. The TCN − Transformer gap is flat.
   - Windowed Mamba (128 frames) loses specifically on long clips.
   - Whether that comes from lost context or from window boundaries is being tested (§8).
   - **How2Sign gives evidence without window boundaries:** the TCN (121-frame reach, no resets) beats Mamba on clips ≤128 frames (+0.71 chrF2) but loses on clips >256 frames (−1.45). Long-minus-short, Mamba − TCN: +2.16 [+1.56, +2.74].
5. **Replacing Mamba's input-dependent scan coefficients with learned input-independent ones preserves or improves translation.**
   - iSign: +0.43 [+0.28, +0.57] chrF2 and +0.31 [+0.17, +0.44] BLEU-4, the same at every length.
   - PHOENIX: −0.20 (n.s.) chrF2 and +0.99 [+0.39, +1.52] BLEU-4.
   - How2Sign: +0.78 [+0.52, +1.05] chrF2 and +0.49 [+0.30, +0.69] BLEU-4. Here the gain is concentrated on shorter clips (+1.36 at ≤128 frames, +0.25 n.s. at >256).

**Central claim (your wording):** in scratch-trained pose-based SLT, temporal convolution recovers most of the advantage over the tested Transformer, while replacing input-dependent SSM coefficients with learned input-independent coefficients preserves or improves translation performance. Wider encoder context appears beneficial on longer clips.

**Scope:**
- These architectures, pose inputs and training conditions only. Not "selectivity is harmful for SLT".
- The input-independent arm is still a nonlinear, input-dependent network: its x/z paths, gating and normalisation stay active. Only its scan coefficients are fixed.

All intervals are 95% source-video-grouped bootstrap intervals (resampling the 216 iSign / 331 PHOENIX / 158 How2Sign test source videos with all their clips; 1,000–2,000 draws), averaged over paired seeds. "n.s." means the interval includes 0.

## 2. Validity work (E0)

| Check | Result | Consequence |
|---|---|---|
| Provenance audit of all 139 trained runs (`AUDIT_runs.csv`) | Every reported headline reproduces from its stored predictions. Flags: 21 runs without a test decode (validation-only or stale copies), 9 stopped early, 4 with several logged decodes (length-penalty sweeps), 2 CSV mismatches in superseded runs | No headline number changed |
| Padding invariance on real checkpoints: clip alone vs batched with the longest clip, fp32 | Transformer: max change in encoder output 1.7e-5, in token log-prob 1.1e-5. **Phase 1 Mamba: 5.6 and 4.6.** Mamba with the padding fix: 0.003 and 0.001 | Phase 1 test numbers are unaffected (decoded one clip at a time). Batched analyses use only the fixed Mamba |
| Source-video identity | 63 iSign test clips (19 families) and 120 val clips (34 families) were treated as single-clip videos | Fixed (§0) |
| How2Sign targets | The server check confirmed it. Train `text` is lowercase with 22.8% `<UNKNOWN>`; val/test `text` is cased with 0%; train `SENTENCE` matches val/test | All four encoders retrained on train `SENTENCE` / val-test `text` (§6). The runner refused to train unless every split of the built cache was ≥90% cased with ≤1% `<UNKNOWN>` |
| Convergence | On PHOENIX the Transformer and TCN were still improving at the 40-epoch cap while Mamba converged by epoch ~30 | PHOENIX rerun with an 80-epoch budget and the same early stopping; every run stopped on its own |

## 3. Does the model use the correct video? (E1)

**Setup:** 1,000 fixed iSign dev clips, chosen before scoring. The reference is teacher-forced under each source; the first 200 clips are also free-decoded with beam 5. The wrong-video partners are being re-checked against the corrected source-video mapping (§0).

| Source pose | Fixed Mamba: Δ token log-lik vs correct | Correct better (% of clips) | Transformer: Δ token log-lik | Correct better (%) |
|---|---|---|---|---|
| Wrong video, length-matched | 1.72 | 97.6 | 1.33 | 96.5 |
| Wrong video, length- and topic-matched (TF-IDF) | 1.26 | 94.2 | 0.96 | 90.5 |
| Other dataset, length-matched | 1.45 | 95.9 | 1.17 | 94.0 |
| Zero pose | 1.82 | 97.7 | 0.67 | 87.6 |
| Static pose (clip mean repeated) | 2.13 | 96.9 | 0.99 | 90.1 |

| Free decoding, 200 clips | Fixed Mamba chrF2 | Shares a content word with the reference | Transformer chrF2 | Shares a content word |
|---|---|---|---|---|
| Correct | 22.38 | 49.0% | 19.42 | 44.5% |
| Wrong, length-matched | 12.78 | 2.5% | 13.03 | 2.5% |
| Wrong, length- and topic-matched | 14.01 | 19.0% | 14.92 | 21.5% |
| Zero | 10.98 | 3.0% | 13.90 | 3.0% |

**Reading:**
- **Visual dependence is strong for both encoders,** so (per your decision tree) E4/E6 weren't the next branch.
- **Topic-matched wrong videos still share a content word in about 20% of decodes.** Topic priors therefore inflate content overlap, but this does not measure how much of recognition comes from topic.
- **Removing motion costs Mamba about twice as much as the Transformer** (static: 2.13 vs 0.99).

## 4. Same-clip temporal interventions (E2)

**Setup:** greedy-decode changes versus the original clip on 2,000 fixed dev clips. "Long" is 257–512 frames (iSign). The time-PE row gives the Transformer positions that follow original time (an inference diagnostic, as you specified). Order changes are applied to the raw pose, not to encoded memory.

| iSign, chrF2 change | Half density | Half density, time-PE | Reverse | Shuffle 16-frame blocks | Shuffle every frame | Shuffle every frame (long clips) |
|---|---|---|---|---|---|---|
| Transformer | −1.97 | −0.12 | −1.49 | −1.37 | **−0.35** | **−0.66** |
| Fixed Mamba | −2.57 | — | −5.16 | −1.96 | **−4.21** | **−4.88** |

**Other observations:**
- **Greedy BLEU-4, Transformer:** shuffling every frame takes it from 2.63 to 1.89, so its order sensitivity is weak relative to Mamba, not absent.
- **PHOENIX:** shuffling every frame costs the Transformer −2.59 chrF2 and Mamba −7.01. Under time-PE, the Transformer's half-density change goes from −7.47 to −0.21.

## 5. Where the gap lives (E3)

**Setup:** fixed Mamba minus Transformer, iSign test, 3 seeds, 216 source videos, 2,000 video-grouped draws (`phase 3/scripts/length_gap.py`).

| Stratum | chrF2 gap |
|---|---|
| All clips | +2.30 [+2.06, +2.54] |
| ≤128 / 129–256 / >256 frames | +1.77 / +2.34 / +2.60 |
| Long-minus-short difference | **+0.83 [+0.35, +1.25]** |
| References of 10–13 words: ≤128 / 129–256 / >256 frames | +1.88 / +2.45 / +2.52 |
| References of >13 words: ≤128 / 129–256 / >256 frames | +1.40 / +1.99 / +2.67 |
| References of ≤9 words: ≤128 / 129–256 / >256 frames | +1.78 / +2.95 / +2.19 (not monotonic) |
| Clips at the 512-frame cap (n = 77) | +2.13 [+1.22, +2.98] |

**Reading:**
- At fixed sentence length, the gap still grows with frame count for medium and long sentences.
- Frame count and duration are collinear at fixed fps; E2 and the controls below address them separately.

## 6. Matched temporal controls (E5) and scan selectivity (§9, C1 + C7)

All arms share the front end, the decoder (which attends globally to all encoder outputs) and the training recipe. Encoders are parameter-matched to the Transformer within 2%; receptive fields were measured by perturbing a single frame in unit tests. Receptive field below means the **encoder's** reach, not the whole translator's.

| Arm | What changes | Encoder reach | Parameters vs Transformer |
|---|---|---|---|
| Transformer + conv stem | 2 residual depthwise-separable conv blocks (kernel 5) before attention | stem 9 frames | +1.58% |
| Transformer + relative positions | ALiBi-style symmetric biases instead of sinusoids | whole clip | 0% |
| Local-attention Transformer | Attention limited to ±10 frames per layer, sinusoids kept, no convolution | 121 frames | 0% |
| TCN | Dilated depthwise conv (kernel 5, dilations 1–8 repeated) + FFN, 8 layers, no attention | 121 frames (radius 60) | +0.08% |
| Wide-receptive-field TCN | Dilations 1–64 then 1 | 513 frames (radius 256; not the whole clip from an endpoint) | +0.08% |
| Mamba in windows | Fixed Mamba on separate 128-frame windows; nothing crosses a window edge | 128 frames | +0.74% (same as Mamba) |
| Mamba with input-independent scan coefficients | B, C and Δ channels of `in_proj` replaced by learned constants; x/z paths, gating and normalisation unchanged | whole clip | +0.74% nominal; **33.27M active** (737,280 `in_proj` weights unused) |

### iSign (test, beam 5)
Content F1 is clipped-multiset; "Neg. recall" is the share of negated references whose hypothesis also negates.

| Encoder | Seeds | chrF2 | BLEU-4 | Content F1 | Number F1 | Neg. recall | vs fixed Mamba, chrF2 | vs fixed Mamba, BLEU-4 | vs fixed Mamba, >256 frames |
|---|---|---|---|---|---|---|---|---|---|
| Transformer | 3 | 19.06 ± 0.15 | 2.90 ± 0.11 | 10.15 | 22.3 | 34.6 | −2.30 [−2.54, −2.07] | −1.10 [−1.27, −0.94] | −2.60 |
| + conv stem | 3 | 19.70 ± 0.06 | 3.15 ± 0.18 | 11.17 | 26.7 | 36.4 | −1.66 [−1.87, −1.45] | −0.85 | −1.93 |
| + relative positions | 3 | 19.78 ± 0.03 | 3.37 ± 0.10 | 11.73 | 29.7 | 34.3 | −1.58 [−1.79, −1.38] | −0.64 | −1.82 |
| Local-attention Transformer | 1 | 20.00 | 3.54 | 12.42 | 28.5 | 41.0 | −1.26 [−1.53, −1.00] | −0.56 | −1.16 |
| TCN | 3 | 20.95 ± 0.11 | 3.92 ± 0.03 | 14.29 | 38.8 | 35.8 | −0.41 [−0.54, −0.28] | −0.08 [−0.21, +0.05] | −0.68 [−0.95, −0.42] |
| Wide-RF TCN | 1 | 20.93 | 4.13 | 14.50 | 36.5 | 44.1 | −0.33 [−0.55, −0.10] | +0.04 | −0.08 [−0.47, +0.33] |
| Mamba in windows | 3 | 21.19 ± 0.14 | 3.79 ± 0.12 | 13.66 | 35.3 | 41.8 | −0.17 [−0.32, −0.01] | −0.22 [−0.35, −0.09] | −0.58 [−0.82, −0.34] |
| **Mamba, input-independent coefficients** | 3 | **21.79 ± 0.11** | **4.31 ± 0.16** | **14.94** | **40.4** | **43.0** | **+0.43 [+0.28, +0.57]** | **+0.31 [+0.17, +0.44]** | +0.46 [+0.22, +0.70] |
| Fixed Mamba | 3 | 21.36 ± 0.16 | 4.00 ± 0.16 | 14.05 | 38.5 | 39.1 | — | — | — |

**Length interactions** (long-minus-short, chrF2, 3 seeds, video-grouped):
- **Mamba − TCN:** +0.83 [+0.44, +1.24].
- **TCN − Transformer:** +0.00 [−0.56, +0.50].
- **Windowed Mamba − Mamba:** −0.92 [−1.33, −0.54].
- **Input-independent − Mamba:** −0.11 [−0.56, +0.31].

### PHOENIX (80-epoch budget, all runs stopped early on their own; 3 seeds)
| Encoder | chrF2 | BLEU-4 | Content F1 | vs fixed Mamba, chrF2 | vs fixed Mamba, BLEU-4 | Best epochs |
|---|---|---|---|---|---|---|
| Transformer | 32.25 ± 0.20 | 10.76 ± 0.14 | 22.24 | −2.19 [−2.80, −1.57] | −0.35 (n.s.) | 30–42 |
| TCN | 33.63 ± 0.18 | 11.70 ± 0.21 | 24.59 | −0.81 [−1.49, −0.14] | +0.59 [−0.05, +1.10] (n.s.) | 39–49 |
| Mamba, input-independent coefficients | 34.23 ± 0.83 | 12.10 ± 0.64 | 25.77 | −0.20 (n.s.) | +0.99 [+0.39, +1.52] | 24–47 |
| Fixed Mamba | 34.44 ± 0.30 | 11.11 ± 0.24 | 24.52 | — | — | 22–26 |

**Notes:**
- PHOENIX clips are all ≤256 frames, so the long-clip question can't be asked there.
- PHOENIX references contain few negations, so negation recall isn't reported for it.

### How2Sign (corrected targets, 80-epoch budget, all runs stopped early on their own; 3 seeds)
2,098 test clips from 158 source videos; median 137 frames, 18% over 256 frames.

| Encoder | chrF2 | BLEU-4 | Content F1 | Number F1 | Neg. recall | vs fixed Mamba, chrF2 | ≤128 frames | >256 frames | vs fixed Mamba, BLEU-4 | Best epochs |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | 20.14 ± 0.28 | 2.34 ± 0.21 | 8.10 | 13.2 | 32.7 | −1.69 [−2.06, −1.30] | −1.04 | −2.30 | −0.90 [−1.18, −0.62] | 20–33 |
| TCN | 21.43 ± 0.15 | 3.33 ± 0.24 | 10.56 | 17.5 | 47.2 | −0.40 [−0.68, −0.12] | **+0.71 [+0.25, +1.16]** | **−1.45 [−1.80, −1.07]** | +0.08 (n.s.) | 24–28 |
| **Mamba, input-independent coefficients** | **22.61 ± 0.21** | **3.73 ± 0.06** | **12.17** | **21.9** | **59.2** | **+0.78 [+0.52, +1.05]** | +1.36 [+0.95, +1.76] | +0.25 (n.s.) | **+0.49 [+0.30, +0.69]** | 19–24 |
| Fixed Mamba | 21.83 ± 0.36 | 3.24 ± 0.27 | 10.72 | 16.7 | 49.5 | — | — | — | — | 20–27 |

**Length interactions** (long-minus-short, chrF2, video-grouped):
- **Mamba − Transformer:** +1.26 [+0.59, +1.88] (iSign: +0.83).
- **Mamba − TCN:** +2.16 [+1.56, +2.74] (iSign: +0.83).
- **TCN − Transformer:** −0.90 [−1.51, −0.26] (iSign: +0.00).
- **Input-independent − Mamba:** −1.11 [−1.64, −0.58] (iSign: −0.11, n.s.).

**Reading:**
- **The iSign ordering reproduces on a third dataset and language:** Transformer < TCN ≈ fixed Mamba < input-independent Mamba. The TCN recovers 76% of the chrF2 gap and all of the BLEU-4 gap.
- **This is the cleanest long-context evidence so far.** The TCN has no window boundaries or resets, only a 121-frame reach, yet it goes from better than Mamba on short clips to clearly worse on long ones. Its advantage over the Transformer also shrinks with length. This can't be a boundary artefact. The decoder still sees every encoder output, so the limit is on how far the *encoder* mixes information. A wide-RF TCN on How2Sign would test whether reach alone closes it (not yet run).
- **Input-independent coefficients help most on short clips here.** On long clips they tie with selective Mamba rather than beat it. "Preserves or improves" still holds, but the iSign pattern of an equal gain at every length does not.
- Fixed Mamba's token NLL on dev is higher than the other encoders' (4.6 vs 3.7–4.0) despite comparable chrF2. This is noted but not interpreted.

**Order sensitivity, How2Sign** (greedy chrF2 change on dev, short ≤256 / long 257–512 frames):

| Encoder | Shuffle every frame | Reverse | Half density |
|---|---|---|---|
| Transformer | −0.33 / −0.23 | −1.64 / −0.86 | −2.63 / −2.60 (time-PE: +0.03 / −0.52) |
| TCN | −0.97 / −1.20 | −2.26 / −1.37 | −3.91 / −2.52 |
| Input-independent Mamba | −6.14 / −12.35 | −3.06 / −2.87 | −3.49 / −3.19 |
| Fixed Mamba | −6.47 / −15.88 | −2.43 / −2.19 | −3.92 / −3.06 |

The TCN is far less sensitive to frame shuffling on How2Sign than on iSign (−0.97 here vs −3.87 there). Its How2Sign gain over the Transformer therefore does not need the strong frame-order dependence Mamba shows.

**Order sensitivity of the iSign controls** (shuffle every frame, greedy chrF2, short / long clips):

| Encoder | Short | Long |
|---|---|---|
| Transformer | −0.35 | −0.66 |
| Conv stem | −2.79 | −3.51 |
| Relative positions | −3.91 | −5.04 |
| Local attention | −1.59 | −2.60 |
| TCN | −3.87 | −3.80 |
| Fixed Mamba | −4.21 | −4.88 |
| Input-independent Mamba | −5.79 | −6.95 |

**Against your gates:**
- **E5 gate ("closes ≥50% of the content gap"):** met by the TCN on all three datasets.
- **Our pre-registered iSign rule ("TCN within ~0.3 chrF2 of Mamba"):** missed (−0.41), hence "most of the advantage", not all.
- **C1 + C7:** within these conditions, input-dependent scan coefficients were not needed.
  - **Caveat 1:** the input-independent arm has fewer active parameters (33.27M vs 34.01M), so regularisation or parameterisation could contribute to its gain.
  - **Caveat 2:** its long-range filter is one exponential per head, so it isn't comparable to a fully parameterised S4D. S4D is queued (§8).

## 7. Other suggestions we ran

| Item | Result | Decision |
|---|---|---|
| E6: encoder × lexical supervision (PHOENIX, seed 42, bag-of-content-words auxiliary loss, weight 0.3) | Hurts both: Transformer −4.15 chrF2, Mamba −1.21. The interaction (+2.94) is confounded, because the Transformer + loss run was still improving at the 40-epoch cap | Dropped at this weight |
| Resource profile, H200, random weights | See below | Table and short discussion only |

| Encoder | Training step | Peak memory | Batch-1 encoder time (512 frames) | Best epoch, iSign / PHOENIX / How2Sign |
|---|---|---|---|---|
| Transformer | 66 ms | 4.38 GB | 5.5 ms | 22–37 / 30–42 / 20–33 |
| TCN | 91 ms | 4.83 GB | 3.5 ms | 23–28 / 39–49 / 24–28 |
| Fixed Mamba | 140 ms | 3.86 GB | 15.7 ms | 19–34 / 22–26 / 20–27 |
| Mamba, input-independent | 151 ms | 3.86 GB | 15.0 ms | 26 (s42) / 24–47 / 19–24 |

**How to read this table:**
- The TCN is faster per step than Mamba but slower than the Transformer, and uses more peak memory than both.
- Mamba converges in fewer epochs.
- Batch-1 timings are dominated by kernel-launch overhead.
- We will measure GPU-hours to the selected checkpoint before any training-efficiency claim, and end-to-end decoding latency before any deployment claim.

## 8. In progress

| Item | Design | Status |
|---|---|---|
| **Boundary vs context** (your main request) | **Inference** (`diag_window.py`, dev short and long samples, full-encoder restriction): no windows; non-overlapping 64 / 128 / 256; overlapping 64 / 128 / 256 (stride W/2, each output from the window in whose central half it lies, so ≤W frames of context but no frame at an edge); 128 shifted by 32 / 64 / 96. Run on the window-trained checkpoints (3 seeds), fixed Mamba and input-independent Mamba. **Training** (seed 42): windows of 64 and 256, and 128 with overlap | Queued |
| Independent LTI baseline | S4D, 3 seeds | Queued |
| Wide-RF TCN | Seeds 13 and 1337 | Queued |
| Learned retention times | Per head (input-independent Mamba) and per mode (S4D), from checkpoints | CPU, running |
| E1 partner re-check | Without same-video pairs | CPU, running |

**Decision rule** (your formulation): if overlap removes most of the block-window long-clip penalty without adding context, the windowed result is mainly about boundaries. If the penalty persists and shrinks steadily with W, the long-context reading is strengthened.

**Not run, following your prioritisation:**
- E4 (its trigger didn't occur);
- E7 (separate paper; needs annotators);
- E8–E13;
- independent signer hold-out and human adequacy ratings (not yet available).

## 9. Caveats we are aware of

- **The test sets have guided many hypotheses,** so results are exploratory in your sense. Diagnostics use dev data; test is used only for headline tables.
- **Input-independent Mamba:**
  - B and C pass through Mamba-2's short depthwise convolution, so their first few (zero-padded) frames differ from the steady state.
  - 737,280 `in_proj` weights are unused.
  - With a scalar A per head, d_state = 64 does not give 64 distinct decay rates per head.
- **Windowed Mamba** uses non-overlapping windows, so the boundary/context confound remains until §8 lands.
- **How2Sign has no wide-RF TCN, windowed or S4D arm yet,** so its long-context reading rests on the TCN's fixed reach.
- **PHOENIX fixed Mamba under the 80-epoch budget** has higher chrF2 but lower BLEU-4 than under 40 epochs (34.44 / 11.11 vs 33.94 / 11.68). The longer budget also lengthens warmup. All four PHOENIX encoders share the budget, but the PHOENIX BLEU-4 differences are less stable than chrF2.
- **chrF2 and BLEU-4 aren't adequacy.** Content F1 excludes negation and digits, which are now reported separately. No human evaluation yet.

## Reproducing the numbers

| What | Where |
|---|---|
| Headline tables (clipped content F1, number F1, negation, grouped intervals) | `phase 3/scripts/compare_encoders.py` → `phase 3/results/compare/` |
| Length strata and interactions | `phase 3/scripts/length_gap.py` → `phase 3/results/length_gap/` |
| Source-video grouping | `phase 3/p3/stitch.py` (`source_video`) |
| Diagnostics | `phase 3/scripts/diag_{audit,padding,video,density,window}.py`, `check_video_partners.py`, `inspect_retention.py` → `phase 3/results/` |
| Encoders | `phase 2/src/models/controls.py`, `selectivity.py`, `s4d.py` (tests: `phase 2/scripts/test_controls.py`, `test_selectivity.py`; `phase 3/scripts/test_diagnostics.py`) |
| Experiment queue | `phase 3/scripts/run_queue.sh`, checklist in `EXPERIMENT_QUEUE.md` |
