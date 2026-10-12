# Response to the review: results of the suggested experiments

**Follows:** `EXPERT_BRIEF.md` (what you reviewed) and your research program of 8 October 2026 (`RESEARCH_PROGRAM_2026-10-08.md`).
**Covers:** the work from 8 to 12 October 2026.

We followed your order: audit → correct-video dependence → same-clip temporal controls → matched training controls → confirmation. Experiment IDs (E0–E13, C1–C12) are yours.

## 1. Summary

1. **Both encoders depend on the correct video** (E1).
   - The correct pose scores the reference higher than a length- and topic-matched wrong video for 94–98% of clips.
   - Free decoding loses almost all content-word overlap without the right video.
2. **The Transformer nearly ignores frame order; Mamba depends on it** (E2).
   - Shuffling every frame costs the Transformer −0.35 chrF2 and Mamba −4.21 (iSign, clips ≤256 frames). Long clips show the same.
   - The Transformer's sensitivity to sampling density disappears when positions follow original time (−1.97 → −0.12).
3. **Most of the gap is local temporal modelling** (E5, parameter-matched, 3 seeds).
   - A dilated TCN with a 121-frame (4.8 s) receptive field closes 82% (iSign) and 63% (PHOENIX) of the chrF2 gap.
   - On BLEU-4 the remaining difference isn't significant on iSign (−0.08 [−0.23, +0.04]) and favours the TCN on PHOENIX.
   - Its content F1 matches or exceeds Mamba's on both datasets.
   - A convolutional stem and relative positions each close only about 30%.
4. **Long clips also need context beyond about 5 s** (E3 and the local/global receptive-field controls).
   - The Mamba − Transformer gap grows with clip length, and this holds at fixed sentence length for medium and long sentences.
   - The TCN − Transformer gap is flat across lengths.
   - Mamba restricted to 5-second windows loses specifically on long clips: long-minus-short −0.92 [−1.32, −0.54].
   - A TCN reaching the whole clip recovers the TCN's long-clip deficit (1 seed).
5. **Input-dependent selectivity isn't needed; removing it helps** (your §9, C1 + C7).
   - A Mamba-2 with B, C and Δ made input-independent (a time-invariant SSM) is the best encoder on iSign: 21.79 / 4.31 against 21.36 / 4.00. The difference is +0.43 [+0.28, +0.57] chrF2 and +0.31 [+0.17, +0.45] BLEU-4, the same at every clip length.
   - On converged PHOENIX it ties on chrF2 (−0.20, n.s.) and is better on BLEU-4 (+0.99 [+0.37, +1.54]).

**Proposed central claim:** in pose-based SLT trained from scratch, the encoder advantage comes from dense temporal filtering over the whole clip, which a time-invariant SSM (equivalently, a long learned convolution) provides. It does not come from content-dependent selection.

This is the Path A outcome you set out, with the selectivity question settled negatively. How2Sign (corrected targets) is running as the third dataset.

All intervals are 95% source-video-grouped bootstrap intervals (resampling videos with all their clips), averaged over paired seeds, unless stated otherwise. "n.s." means the interval includes 0.

## 2. Validity work (E0)

| Check | Result | Consequence |
|---|---|---|
| Provenance audit of all 139 trained runs (`AUDIT_runs.csv`) | Every reported headline reproduces from its stored predictions. Flags: 21 runs without a test decode (expected: validation-only or stale copies), 9 stopped early, 4 with several logged decodes (length-penalty sweeps), 2 CSV mismatches in superseded runs | No headline number changed |
| Padding invariance on real checkpoints: clip alone vs batched with the longest clip, fp32 | Transformer: max change in encoder output 1.7e-5, in token log-prob 1.1e-5. **Phase 1 Mamba: 5.6 and 4.6.** Mamba with the padding fix: 0.003 and 0.001 | Phase 1 test numbers are unaffected (decoded one clip at a time). Batched analyses use only the fixed Mamba |
| How2Sign targets | Trained on lowercased text with `<UNKNOWN>`, scored against cased references | Corrected run in progress (§8); no How2Sign claims until then |
| Convergence | On PHOENIX the Transformer and TCN were still improving at the 40-epoch cap while Mamba converged by epoch ~30 | PHOENIX rerun with an 80-epoch budget and the same early stopping; every run then stopped on its own |

## 3. Does the model use the correct video? (E1)

**Setup:** 1,000 fixed iSign dev clips, chosen before scoring. The reference is teacher-forced under each source; the first 200 clips are also free-decoded with beam 5.

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
- **Visual dependence is strong for both encoders,** so (per your decision tree) we did not run E4/E6 as the next branch.
- **About 40% of content-word overlap can come from topic alone** (19% vs 49%), so the overlap metric overstates word-level recognition.
- **Removing motion costs Mamba about twice as much as the Transformer** (static: 2.13 vs 0.99). This was the first sign of the temporal finding.

## 4. Same-clip temporal interventions (E2)

**Setup:** greedy chrF2 change versus the original clip on 2,000 fixed dev clips. "Long" is 257–512 frames (iSign) and was added because the length gap is largest there. The time-PE row gives the Transformer positions that follow original time (an inference diagnostic, as you specified). Order changes are applied to the raw pose, not to encoded memory.

| iSign | Half density | Half density, time-PE | Reverse | Shuffle 16-frame blocks | Shuffle every frame | Shuffle every frame (long clips) |
|---|---|---|---|---|---|---|
| Transformer | −1.97 | −0.12 | −1.49 | −1.37 | **−0.35** | **−0.66** |
| Fixed Mamba | −2.57 | — | −5.16 | −1.96 | **−4.21** | **−4.88** |

PHOENIX shows the same pattern: shuffling every frame costs the Transformer −2.59 and Mamba −7.01. Half density goes from −7.47 to −0.21 for the Transformer under time-PE.

**Reading:**
- The Transformer behaves close to a bag of frames.
- Its density sensitivity is a positional-encoding artefact, not dependence on the number of observations.

## 5. Where the gap lives (E3)

**Setup:** fixed Mamba minus Transformer, iSign test, 3 seeds, 260 source videos, 2,000 video-grouped draws (`phase 3/scripts/length_gap.py`).

| Stratum | chrF2 gap |
|---|---|
| All clips | +2.30 [+2.06, +2.55] |
| ≤128 / 129–256 / >256 frames | +1.77 / +2.34 / +2.60 |
| Long-minus-short difference | **+0.83 [+0.36, +1.23]** (BLEU-4: +0.41 [−0.04, +0.82]) |
| References of 10–13 words: ≤128 / 129–256 / >256 frames | +1.88 / +2.45 / +2.52 |
| References of >13 words: ≤128 / 129–256 / >256 frames | +1.40 / +1.99 / +2.67 |
| References of ≤9 words: ≤128 / 129–256 / >256 frames | +1.78 / +2.95 / +2.19 (not monotonic) |
| Clips at the 512-frame cap (n = 77) | +2.13 [+1.26, +2.90] |

**Reading:**
- At fixed sentence length, the gap still grows with frame count for medium and long sentences, so it isn't only sentence length.
- As you noted, frame count and duration are collinear at fixed fps. E2 and the receptive-field controls below separate them.

## 6. Matched temporal controls (E5) and the selectivity test (§9, C1 + C7)

All arms share the front end, the decoder and the training recipe. The encoder is parameter-matched to the Transformer within 2%, and receptive fields were measured by perturbing a single frame in unit tests.

| Arm | What changes | Encoder reach | Parameters vs Transformer |
|---|---|---|---|
| Transformer + conv stem | 2 residual depthwise-separable conv blocks (kernel 5) before attention | stem 9 frames | +1.58% |
| Transformer + relative positions | ALiBi-style symmetric biases instead of sinusoids | whole clip | 0% |
| Local-attention Transformer | Attention limited to ±10 frames per layer, sinusoids kept, no convolution | 121 frames | 0% |
| TCN | Dilated depthwise conv (kernel 5, dilations 1–8 repeated) + FFN, 8 layers, no attention | 121 frames (4.8 s) | +0.08% |
| TCN, whole clip | Dilations 1–64 then 1 | 513 frames | +0.08% |
| Mamba in windows | Fixed Mamba run on separate 128-frame windows; nothing crosses a window edge | 128 frames | +0.74% (same as Mamba) |
| Mamba, non-selective | The B, C and Δ channels of `in_proj` replaced by learned constants, so every scan is time-invariant. x and z stay input-dependent | whole clip | +0.74% (+1,440) |

### iSign (test, beam 5)
| Encoder | Seeds | chrF2 | BLEU-4 | Content F1 | vs fixed Mamba, chrF2 | vs fixed Mamba, BLEU-4 | vs fixed Mamba, clips >256 frames |
|---|---|---|---|---|---|---|---|
| Transformer | 3 | 19.06 ± 0.15 | 2.90 ± 0.11 | 10.38 | −2.30 [−2.55, −2.05] | −1.10 [−1.26, −0.93] | −2.60 |
| + conv stem | 3 | 19.70 ± 0.06 | 3.15 ± 0.18 | 11.45 | −1.66 [−1.89, −1.44] | −0.85 | −1.93 |
| + relative positions | 3 | 19.78 ± 0.03 | 3.37 ± 0.10 | 12.12 | −1.58 [−1.78, −1.36] | −0.64 | −1.82 |
| Local-attention Transformer | 1 | 20.00 | 3.54 | 12.69 | −1.26 [−1.51, −0.99] | −0.56 | −1.16 |
| TCN | 3 | 20.95 ± 0.11 | 3.92 ± 0.03 | 14.68 | −0.41 [−0.57, −0.27] | −0.08 [−0.23, +0.04] | −0.68 [−0.94, −0.49] |
| TCN, whole clip | 1 | 20.93 | 4.13 | 14.90 | −0.33 [−0.60, −0.06] | +0.04 | −0.08 [−0.48, +0.33] |
| Mamba in windows | 3 | 21.19 ± 0.14 | 3.79 ± 0.12 | 14.14 | −0.17 [−0.34, −0.02] | −0.22 [−0.36, −0.10] | −0.58 [−0.83, −0.36] |
| **Mamba, non-selective** | 3 | **21.79 ± 0.11** | **4.31 ± 0.16** | **15.38** | **+0.43 [+0.28, +0.57]** | **+0.31 [+0.17, +0.45]** | +0.46 [+0.21, +0.71] |
| Fixed Mamba | 3 | 21.36 ± 0.16 | 4.00 ± 0.16 | 14.44 | — | — | — |

**Length interactions** (long-minus-short, chrF2, 3 seeds):
- **Mamba − TCN:** +0.83 [+0.44, +1.26]. The TCN's deficit is on long clips.
- **Windowed Mamba − Mamba:** −0.92 [−1.32, −0.54]. Windowing hurts long clips.
- **Non-selective − Mamba:** −0.11 [−0.57, +0.32]. The gain is uniform across lengths.

### PHOENIX (80-epoch budget, all runs stopped early on their own; 3 seeds)
| Encoder | chrF2 | BLEU-4 | Content F1 | vs fixed Mamba, chrF2 | vs fixed Mamba, BLEU-4 | Best epochs |
|---|---|---|---|---|---|---|
| Transformer | 32.25 ± 0.20 | 10.76 ± 0.14 | 22.29 | −2.19 [−2.75, −1.56] | −0.35 (n.s.) | 30–42 |
| TCN | 33.63 ± 0.18 | 11.70 ± 0.21 | 24.59 | −0.81 [−1.51, −0.15] | +0.59 [+0.01, +1.05] | 39–49 |
| Mamba, non-selective | 34.23 ± 0.83 | 12.10 ± 0.64 | 25.91 | −0.20 (n.s.) | +0.99 [+0.37, +1.54] | 24–47 |
| Fixed Mamba | 34.44 ± 0.30 | 11.11 ± 0.24 | 24.63 | — | — | 22–26 |

PHOENIX clips are all ≤256 frames, so the long-clip question can't be asked there.

**Order sensitivity of the controls** (shuffle every frame, iSign, short / long clips):

| Encoder | Short | Long |
|---|---|---|
| Transformer | −0.35 | −0.66 |
| Conv stem | −2.79 | −3.51 |
| Relative positions | −3.91 | −5.04 |
| Local attention | −1.59 | −2.60 |
| TCN | −3.87 | −3.80 |
| Fixed Mamba | −4.21 | −4.88 |
| Non-selective Mamba | −5.79 | −6.95 |

Every control that gains becomes order-sensitive.

**Against your gates:**
- **E5 gate ("closes ≥50% of the content gap"):** met by the TCN on both datasets. Its content F1 is at or above Mamba's.
- **Our pre-registered iSign rule ("TCN within ~0.3 chrF2 of Mamba"):** missed (−0.41). Hence the claim is "most of the gap", not "all".
- **Your C1 + C7 test, whether selection matters beyond local convolution and fixed recurrence:** answered **no**. A fixed recurrence is at least as good on both datasets.

## 7. Other suggestions we ran

| Item | Result | Decision |
|---|---|---|
| E6: encoder × lexical supervision (PHOENIX, seed 42, bag-of-content-words auxiliary loss, weight 0.3) | Hurts both: Transformer −4.15 chrF2, Mamba −1.21. The interaction (+2.94 [+1.55, +4.18]) is confounded, because the Transformer + loss run was still improving at the 40-epoch cap | Dropped at this weight; not pursued |
| Resource profile, H200, random weights | See below | — |
| Video-grouped bootstrap for all new comparisons | Adopted throughout (`length_gap.py`, `compare_encoders.py`) | Older REPORT.md intervals to be recomputed this way |

**Resource profile** (training step at the 24,576-frame token budget; batch-1 encoder time at 512 frames):

| Encoder | Training step | Peak memory | Batch-1 encoder time |
|---|---|---|---|
| Transformer | 66 ms | 4.38 GB | 5.5 ms |
| TCN | 91 ms | 4.83 GB | 3.5 ms |
| Fixed Mamba | 140 ms | 3.86 GB | 15.7 ms |
| Non-selective Mamba | 151 ms | 3.86 GB | 15.0 ms |

Batch-1 timings are dominated by kernel-launch overhead; treat them as approximate.

## 8. Not yet done, and why

| Item | Status |
|---|---|
| How2Sign with corrected targets (Transformer, TCN, fixed and non-selective Mamba, 3 seeds, order diagnostics) | Queued; gated on the target-column check. It's the longest-clip dataset, so the strongest test of finding 4 |
| 3 seeds for the local-attention Transformer and the whole-clip TCN | Not run; both are 1 seed |
| E4 frozen lexical discrimination and reranking | Not run; its trigger (weak visual dependence) did not occur |
| E7 confidence vs published baselines, human ratings | Not run; separate paper, needs annotators |
| E8–E13 | Not run, following your prioritisation and stop list |
| Independent signer/source hold-out, human adequacy ratings | Not available yet |

## 9. Caveats we are aware of

- **The test sets have guided many hypotheses,** so all results here are exploratory in your sense. Diagnostics (E1, E2) used dev data only. Test is used only for the headline tables.
- **Non-selective Mamba implementation:**
  - The replaced `in_proj` rows stay in the module unused.
  - B and C still pass through Mamba-2's short depthwise convolution, which is constant on a constant input except for the first few (zero-padded) frames.
  - It's an LTI recurrence with the same state size, not a separately tuned S4/S5.
- **Windowed Mamba uses non-overlapping windows,** so frames near window edges see less context than a sliding window would.
- **PHOENIX fixed Mamba under the 80-epoch budget** has higher chrF2 but lower BLEU-4 than under 40 epochs (34.44 / 11.11 vs 33.94 / 11.68). The longer budget also lengthens warmup (warmup is 10% of total steps), which changes the schedule. All four PHOENIX encoders share the same budget, so the comparison is consistent, but the PHOENIX BLEU-4 differences are less stable than chrF2.
- **chrF2 and BLEU-4 aren't adequacy.** The content-word filter excludes negation and digits. No human evaluation yet.

## 10. Questions for you

1. Is "time-invariant SSM ≥ selective SSM, local filtering explains most of the gap" strong enough as the central claim? Or would you require the How2Sign confirmation and a published pose baseline in a separate pretrained track first?
2. For the non-selective arm, would you also want a separately parameterised LTI baseline (e.g. S4D or a long-convolution encoder such as Hyena) to show the result isn't specific to our Mamba-2 ablation?
3. Given that the TCN gets about 98% of Mamba's chrF2 at about 65% of its training-step time, should the efficiency angle be a section, or kept to a table?
4. Which venue category fits this evidence, by your reading of section 10 of your program?

## Reproducing the numbers

| What | Where |
|---|---|
| Headline tables | `phase 3/scripts/compare_encoders.py` → `phase 3/results/compare/` |
| Length strata and interactions | `phase 3/scripts/length_gap.py` → `phase 3/results/length_gap/` |
| Diagnostics | `phase 3/scripts/diag_{audit,padding,video,density}.py` → `phase 3/results/diagnostics/` |
| Control encoders | `phase 2/src/models/controls.py`, `phase 2/src/models/selectivity.py` (tests: `phase 2/scripts/test_controls.py`, `test_selectivity.py`) |
| Experiment queue | `phase 3/scripts/run_queue.sh`, checklist in `EXPERIMENT_QUEUE.md` |
