# Selective State Spaces (Mamba) vs Transformers for Sign Language Translation: Research Report

*As of 2 October 2026. Every number below comes from result files in this repository. Unless stated otherwise: **test set, beam 5, corpus chrF2, mean ± std over 3 seeds (13, 42, 1337)**. Significance uses paired bootstrap resampling (sacreBLEU, 1,000 samples) run separately for each seed.*

---

## 1. Executive summary

**The question.** On pose-based, gloss-free sign language translation (SLT), does a selective state-space encoder (bidirectional Mamba-2) beat a Transformer encoder when both have matched parameters, the same data and the same training budget? And can Mamba's input-dependent step size Δ be turned into something new?

**Headline results**

| # | Finding | Status |
|---|---|---|
| 1 | **Mamba beats the Transformer on all 3 sign languages**, on every seed, with p ≤ 0.012: iSign +1.81 chrF2, How2Sign +2.86, PHOENIX-2014T +2.59. On iSign the gap grows with clip length (+20% BLEU on short clips, +45% on long ones). | Solid; this is the main contribution |
| 2 | **Δ-weighted averaging** compresses the decoder memory about 15× with no extra parameters. It beats random segmentation on all 3 datasets. It beats plain averaging on iSign (p ≤ 0.001 on every seed), is borderline on How2Sign, and does not on PHOENIX. | Solid, but dataset-dependent |
| 3 | **Choosing single frames by Δ** (the DeciMamba-style approach) is no better than random. Its apparent gain over a fixed stride comes from training-time variety. | Solid negative result |
| 4 | **Tokens × bits:** Δ averaging + 3-bit rotation quantization (TurboQuant-style) gives about **80× less decoder memory** on all 3 datasets for −0.1 to −0.7 chrF2. | Solid |
| 5 | **Context from previous clips** helps slightly (+0.22 over no context; beats wrong-video context on every seed). The gain does **not** grow with 4 or 8 clips. | Modest |

**What didn't work:** horizon-bank initialisation (no effect); Δ tracking hand motion (ρ ≈ 0.02); single-frame Δ selection beyond random; the training-time shift added to Δ averaging (hurts); context scaling.

**Proposed framing:** an empirical analysis paper, *"Selective state spaces for sign language translation: what helps and what doesn't"*. It's built on the benchmark (new, as far as the literature check found), with Δ-based compression and its controls as the second contribution.

---

## 2. Setup

### 2.1 Task and data
Pose keypoints → English or German text, gloss-free, trained from scratch.

| Dataset | Language | Train / val / test clips | Mean test clip length | Notes |
|---|---|---|---|---|
| iSign | Indian SL → English | 99,923 / 5,653 / 6,069 | 194 frames | video-disjoint splits; consecutive story segments |
| How2Sign | American SL → English | 31,092 / 1,561 / 2,098 | 163 frames | pose files keyed by SENTENCE_ID |
| PHOENIX-2014T | German SL → German | 7,096 / 519 / 642 | 101 frames | weather forecasts; glosses ignored |

**Preprocessing (identical for every model):**
- MediaPipe Holistic `.pose` files are reduced to **178 keypoints** (body 8, face 128, hands 21+21), with x and y only, giving **356 features per frame**.
- Each clip is normalised by its own shoulder distance and centre.
- Missing frames are interpolated; every dataset is resampled to **25 fps**.
- Clips are capped at 512 frames by uniform subsampling.
- The features are stored as an fp16 memory-mapped cache.
- Text uses a SentencePiece unigram tokenizer: vocabulary 4,000 (1,000 for PHOENIX).

### 2.2 Models (matched parameters, only the encoder changes)
| | Transformer (A1) | Mamba (A2) |
|---|---|---|
| Encoder | 6 Transformer layers, d = 512, 8 heads, FFN 2,048, sinusoidal positional encoding | 5 bidirectional Mamba-2 layers (separate forward and backward scans), d_state = 64, expand = 2, head dim 64; no positional encoding |
| Front end | Linear(356→512) + LayerNorm + Dropout | same |
| Decoder | 3-layer Transformer decoder, tied embeddings | same, byte-identical |
| Parameters | 33.76 M | 34.01 M (+0.74%; a 2% gap is the abort threshold) |

### 2.3 Training and evaluation protocol (identical for both models)
- **Optimiser:** AdamW (0.9, 0.98), weight decay 0.01 (none on biases, norms and Mamba's A_log, D and dt_bias), cosine schedule, bf16, gradient clipping at 1.0, label smoothing 0.1.
- **Batching:** token-budget batching at about 24,576 frames per step, so both models see the same data per step.
- **Budget and selection:** 40 epochs, early stopping on val chrF2 with patience 8, learning rate 3e-4. The rate was chosen from {1e-4, 3e-4, 1e-3} on iSign seed 42 and was the best for both models.
- **Decoding:** beam 5, length penalty 1.0.
- **Metrics:** corpus chrF2 (primary), sacreBLEU BLEU-4, ROUGE-L, METEOR and WER, with the legacy sentence-level metrics also logged.
- **Hardware:** a 4× NVIDIA H200 NVL node (`t3ihpc07`).

---

## 3. Phase 1: Mamba vs Transformer benchmark

### 3.1 Main result (test, beam 5, 3 seeds)
| Dataset | Transformer chrF2 | Mamba chrF2 | Gain | Transformer BLEU-4 | Mamba BLEU-4 | Bootstrap p (per seed) |
|---|---|---|---|---|---|---|
| iSign | 19.06 ± 0.19 | **20.87 ± 0.30** | +1.81 | 2.90 | **3.74** | 0.001 / 0.001 / 0.001 |
| How2Sign (F3 schedule, §4) | 16.75 ± 1.88 | **19.61 ± 0.53** | +2.86 | 1.38 | **2.33** | 0.001 / 0.001 / 0.001 |
| PHOENIX-2014T | 29.13 ± 0.70 | **31.72 ± 0.92** | +2.59 | 9.33 | **11.39** | 0.001 / 0.012 / 0.001 |

- Mamba wins on **every seed of every dataset**.
- On How2Sign, one Transformer run (seed 42) stalled at epoch 13 even after the schedule fix. It's reported as is: rerunning with another seed would be cherry-picking.
- With the padding fix (§4), Mamba on iSign reaches **21.36 ± 0.20**, which widens the gap to +2.30.
- Ensembles of the 3 seeds (iSign test): Mamba 4.91 BLEU vs Transformer 3.98 BLEU (p < 0.001).

### 3.2 The gap grows with sequence length (iSign test, 3-seed mean)
| Source length (frames) | Clips | Transformer BLEU-4 / chrF2 | Mamba BLEU-4 / chrF2 | Relative BLEU gain | chrF2 gain |
|---|---|---|---|---|---|
| ≤ 128 | 1,814 | 3.66 / 17.84 | 4.39 / 19.11 | +20% | +1.27 |
| 129–256 | 2,822 | 2.93 / 19.33 | 3.71 / 21.12 | +26% | +1.79 |
| 257–512 | 1,432 | 2.26 / 19.44 | 3.27 / 21.65 | **+45%** | **+2.21** |

This trend is the thread the rest of the project followed: the SSM's advantage is about integrating long temporal context.

---

## 4. Bugs and protocol fixes found during the project

| Issue | What was wrong | Fix | Effect |
|---|---|---|---|
| **Padding leak in Mamba's backward scan** | Padding frames were zeroed, but LayerNorm of a zero vector is its bias (β ≠ 0). A full-tensor flip then put every short clip's padding *before* its real frames in the backward direction, so a clip's encoding depended on its batch-mates. | Reverse each clip within its own length (`flip_valid`) and re-zero the padding after LayerNorm. A CPU test shows the Phase 1 leak and the fix. | iSign test 20.87 → **21.36** (3 seeds) |
| **Warmup and early stopping on small datasets (F3)** | A fixed 4,000-step warmup combined with token-budget batching meant How2Sign and PHOENIX had few steps per epoch. Patience 8 fired *during warmup*: 3 How2Sign Transformer runs and 1 Mamba run stopped at epochs 10–16. | Warmup capped at 10% of the run, and patience counted only after warmup (`configs/train/base_f3.yaml`). The fix is opt-in, so iSign runs are unchanged. | How2Sign Mamba runs now stable at 31–38 epochs |
| **Validation numbers reported as test** | The first summary tables were validation, greedy, best-epoch numbers. | Beam-5 test decodes from `best.pt` for every run. | All numbers in this report are test |
| **Prediction CSV for seed 13 out of sync** | iSign seed 13's predictions came from a different length penalty than its metrics. | Regenerated with the frozen decoding settings. | Consistent files |
| **Long-clip "cutoff" at eval (false alarm)** | Suspected that clips over 512 frames were cut, not subsampled. | The cache is already subsampled at build time, so it never triggered. A defensive fix was added anyway. | None |
| **Repository hygiene** | Files from an unrelated project were committed to this (public) repo, and a node name was overwritten. | Untracked and ignored; node name restored. | — |

---

## 5. Phase 2: can Mamba's Δ choose what the decoder sees?

**Idea.** Mamba's input-dependent step size Δ controls how much each frame is written into the state. If Δ marks meaningful moments, it can decide which frames, or which segments, the decoder cross-attends to. That compresses the memory with no extra parameters.

All Phase 2 models use the padding fix. **"Multi-timescale" ("horizon banks")** means the 16 heads per layer are initialised as 8 short (about 14 frames), 4 mid (about 75) and 4 long (about 500).

### 5.1 The gate, and the multi-timescale init (iSign, seed 42)
| Model | Decoder memory | Val chrF2 | Test chrF2 |
|---|---|---|---|
| Full memory (padding fix) | 195 frames | 21.06 | 21.26 |
| + multi-timescale init | 195 frames | 21.06 | 21.29 |
| + keep every 16th frame | 13.6 (7%) | 19.63 | 20.06 |
| + keep 13.6 frames chosen by Δ (matched count) | 13.6 (7%) | 20.36 | 20.39 (vs uniform p = 0.002) |

- The multi-timescale init has **no effect**.
- Cutting memory to 7% costs about 1.4 val chrF2.
- Choosing frames by Δ recovers about half of that loss.

### 5.2 Compression sweep (val chrF2, Δ-chosen minus uniform at the same count)
| Kept fraction | 1/8 | 1/16 (3 seeds) | 1/32 |
|---|---|---|---|
| Δ − uniform | +0.14 | +0.71 (positive on every seed) | +1.03 |

The gap grows with compression, which is what you'd expect if Δ were informative. The 1/8 and 1/32 points are seed 42 only.

### 5.3 What does Δ track? (analysis of the trained model, 5,652 iSign val clips)
| Measure | Δ | Comparison |
|---|---|---|
| Per-clip Spearman ρ(Δ, hand speed) | +0.024 (53% of clips positive) | time-shuffled Δ: −0.002 |
| ρ(Δ, body speed) / ρ(Δ, face speed) | +0.018 / +0.002 | — |
| Hand speed (z-score) at kept frames | −0.006 | uniform stride: −0.024 |
| Unevenness of spacing (coefficient of variation) | 0.31 | uniform stride: 0.14 |

**Δ does not follow motion.** Its only clear property is spacing the kept frames about twice as unevenly as a stride.

### 5.4 The deciding controls: random spacing and a shifted stride (test, 3 seeds, same frame count in each row)
| Single-frame pooling | Uniform stride | Random spacing (calibrated to Δ's unevenness) | Δ-chosen | Stride with random shift in training |
|---|---|---|---|---|
| 1/16 | 19.87 | 20.46 | 20.55 | **20.67** |
| 1/32 | 18.81 | 19.68 | 19.65 | — |

- **Random spacing does as well as Δ:** Δ − random is +0.09 at 1/16 and −0.03 at 1/32. Only 1 of 6 seed comparisons is significant.
- **Random beats a fixed stride on every seed** (p = 0.001).
- A stride shifted randomly during training does even better (20.67). So **the gain over a fixed stride is training-time variety (a regularisation effect), not information carried by Δ.**
- **Conclusion: selecting single frames by Δ carries no information beyond random.** This is the DeciMamba-style use of Δ (§8).

---

## 6. Δ-weighted averaging: the Δ result that survives

**Method:**
1. Split each clip into n = ⌈length / 16⌉ segments, ending at equal quantiles of the short heads' accumulated Δ.
2. Replace each segment with the **Δ-weighted mean** of its encoder states.

Because Δ is the averaging weight, the translation loss trains Δ directly, unlike in single-frame selection. It adds no parameters, and the output length is fixed.

### 6.1 Three datasets, with controls (test, beam 5, 3 seeds; memory about 15× smaller)
| Dataset | Full memory | Plain 16-frame average | Random segments + weights | **Δ-weighted average** |
|---|---|---|---|---|
| iSign | 21.36 ± 0.20 | 20.48 ± 0.27 | 19.97 ± 0.08 | **20.93 ± 0.09** |
| How2Sign | 19.92 ± 0.66 | 19.65 ± 0.20 | 19.25 ± 0.15 | **19.86 ± 0.31** |
| PHOENIX-2014T | 33.94 ± 0.20 | **33.61 ± 0.20** | 32.32 ± 0.28 | 33.35 ± 0.27 |

**Cost of compression** (change vs full memory):

| Dataset | Plain | Random | Δ-weighted |
|---|---|---|---|
| iSign | −0.88 | −1.39 | **−0.43** |
| How2Sign | −0.27 | −0.68 | **−0.07** |
| PHOENIX | **−0.33** | −1.62 | −0.59 |

**Significance (per seed):**
- **Δ vs random:** iSign p = 0.001 on every seed; How2Sign 0.001 / 0.001 / 0.155; PHOENIX 0.038 / 0.010 / 0.015. Δ wins on all datasets.
- **Δ vs plain:**
  - iSign: p = 0.001 on every seed (Δ wins).
  - How2Sign: 0.055 / 0.048 / 0.214 (borderline).
  - PHOENIX: not significant, with Δ lower.

**Reading:** Δ always carries information about *where segments should end*, since random segmentation is the worst on every dataset. Whether that beats simple averaging depends on the dataset. PHOENIX clips are short (about 6 memory tokens per clip), leaving Δ little room. A breakdown by clip length does not give a clean pattern:
- iSign gains evenly at every length;
- How2Sign gains only on long clips;
- PHOENIX doesn't gain at any length.

### 6.2 Is compression specific to Mamba? (iSign, 3 seeds)
| Model | Full memory | + plain 16-frame average | Cost |
|---|---|---|---|
| Transformer | 19.06 | 18.66 | **−0.39** |
| Mamba | 21.36 | 20.48 | **−0.88** |
| Mamba, Δ-weighted | 21.36 | 20.93 | **−0.43** |

- **The Transformer tolerates plain averaging *better* than Mamba does.** A plausible reason: a recurrent encoder's per-frame states are more local than attention's, so naive averaging blurs them more.
- **Δ weighting brings Mamba's compression cost down to the Transformer's level.** So the honest claim is: *Δ fixes a compression weakness specific to SSM encoders.*
- **Mamba stays ahead even compressed.** Mamba with plain averaging (20.48) still beats the uncompressed Transformer (19.06) on every seed (p = 0.001).

### 6.3 Variants that did not help
- **Training-time random shift on top of Δ averaging:** 20.69 vs 20.93, significantly worse on 2 of 3 seeds.
- **Fixed-threshold Δ pooling (free rate):** the number of kept frames drifts as Δ trains, so it can't be compared fairly. It was replaced by the matched version.

---

## 7. Context from previous clips (C1)

**Motivation (text analysis of the iSign test references; 197 videos, 6,005 segments).** Earlier sentences of the same story contain much of the current sentence's content:

| Previous clips k | Content words of the current sentence seen in the previous k sentences | Same, random other video |
|---|---|---|
| 1 | 9.0% | 0.5% |
| 2 | 14.4% | 0.9% |
| 4 | 20.4% | 1.6% |
| 8 | **26.0%** | 2.4% |

37% of capitalised names (for example *Bobby*, *Sundari*) already appear earlier in the same video. Isolated-clip models recover only 20–28% of them.

**Design:**
- The encoder reads the poses of the previous k clips of the same video, then the current clip, with a learned context-type embedding and a clip-boundary embedding.
- The decoder cross-attends **only to the current clip**, so context can reach the translation only through the encoder.
- No text from previous clips is ever used (that would leak the labels).
- **Control:** context taken from a different video. It tells discourse context apart from "more frames".

### Results (iSign test, beam 5)
| Context | Seeds | chrF2 | BLEU-4 |
|---|---|---|---|
| None (same code path) | 3 | 21.23 ± 0.18 | 3.77 |
| 2 clips from another video | 3 | 21.19 ± 0.12 | 3.78 |
| **2 previous clips** | 3 | **21.45 ± 0.08** | **4.04** |
| 4 previous clips | 1 (s42) | 21.45 | 3.88 |
| 8 previous clips | 1 (s42) | 21.53 | 4.15 |

**Per-seed comparisons:**
- **2 previous clips vs wrong-video context:** p = 0.003 / 0.008 / 0.070 (wins on every seed).
- **2 previous clips vs no context:** p = 0.318 / 0.003 / 0.006.
- For reference, 2 previous clips (seed 42) scores 21.54.

**Verdict:**
- **Real but small:** about +0.2–0.3, and the story matters, since wrong-video context doesn't help.
- **It does not grow with 4 or 8 clips**, and recall of recurring names did not improve. The model seems to pick up the topic, not long-range content.
- **It's barely above the full-memory baseline** that uses the standard code path (21.36).

**Operational note:** the 2- and 8-clip jobs ran out of memory on GPUs shared with another project (about 60 GB needed). They were rerun unchanged on a free GPU, so results stay comparable.

---

## 8. Memory quantization (TurboQuant-style), stacked on Δ compression

**Method** (post-hoc, no retraining; code in `phase 2/quant/`). Each decoder memory token (512 values) is quantized in one of three ways:
- **turbo:** a fixed random rotation, then the optimal Lloyd–Max Gaussian codebook per coordinate, plus an fp16 norm.
- **turbo_prod:** turbo at b−1 bits plus a 1-bit QJL correction, which keeps inner products unbiased.
- **naive:** per-token min–max quantization (the baseline).

The quantizers were validated against published Lloyd–Max tables, theoretical distortion and an unbiasedness test. Bytes count the stored encoder memory. That's already 6× less than caching cross-attention keys and values for 3 decoder layers.

### 8.1 Quality vs memory
| Dataset | Full memory, fp16 | Δ average, fp16 | **Δ average, turbo 3-bit** | Approximate cost vs full memory |
|---|---|---|---|---|
| iSign | 198,496 B | 12,876 B (15×) | **2,440 B (81×)** | −0.49 (beam, 3 seeds) |
| How2Sign | 166,439 B | 10,876 B (15×) | **2,061 B (81×)** | ≈ −0.13 |
| PHOENIX | 103,081 B | 6,929 B (15×) | **1,313 B (78×)** | ≈ −0.72 |

The How2Sign and PHOENIX costs combine the beam compression cost (§6) with the greedy quantization change, so they're approximate.

### 8.2 Change in chrF2 from quantization alone (vs fp16 of the same model, mean over seeds)
| | iSign (beam, 3 seeds) | How2Sign (greedy) | PHOENIX (greedy) |
|---|---|---|---|
| 4-bit or 3-bit, turbo or naive | −0.05 to −0.11 | ≈ 0 to −0.14 | ≈ 0 to −0.14 |
| turbo 2-bit | −0.18 (full) / −0.20 (Δ average) | ≈ 0 | −0.57 to −0.71 |
| naive 2-bit | **−3.69** (full) / **−1.82** (Δ average) | −0.37 to −0.60 | −0.19 to −0.35 |
| turbo_prod 3-bit (unbiased) | −0.20 | −0.18 to −0.28 | −0.15 to −0.23 |

**Findings:**
1. **3 bits is effectively free on all datasets.** That's how Δ averaging gets to about 80× smaller memory.
2. **The unbiased variant is consistently worse** than the plain one for cross-attention memory. Lower reconstruction error matters more than unbiased attention scores.
3. **At 2 bits, the results depend on the dataset.** Rotation is essential on iSign (naive breaks badly) and helps on How2Sign, but naive is better on PHOENIX.

**Correction to an earlier claim:** an early single-seed greedy run suggested compressed memory is *more fragile* than full memory. With beam 5 on 3 seeds, naive 2-bit actually breaks full memory even harder (−3.69 vs −1.82), so that claim was withdrawn.

**Limits:** memory is not a bottleneck at this model size (full memory is about 200 KB per clip). The motivation is streaming and on-device use, which is argued but not demonstrated, and speed and energy were not measured.

---

## 9. Literature check: what is already published vs what is new

Searches covered arXiv and venues from 2024 to 2026, citing only papers actually opened (marked otherwise where not). One Google Scholar pass is still recommended before submission.

| Our contribution | Closest prior work | Verdict |
|---|---|---|
| Mamba vs Transformer for pose-based SLT, matched parameters, analysed by length, on iSign/How2Sign/PHOENIX | PhonSSM (bidirectional Mamba on MediaPipe, **recognition only**, no Transformer baseline); SignMamba (recognition, RGB); EvSLT (event camera, Mamba as one backbone part); WMT 2024 "How Effective are SSMs for MT?" (same method, **text**) | **New.** First SSM encoder for pose-based SLT to text, first SSM results on iSign, first matched comparison by length in sign language |
| Choosing frames by Δ | **DeciMamba** (ICLR 2025), **DTP** (ICLR 2026) and **MTR** use Δ to keep or prune tokens | **Already done.** Our contribution is the *negative* result (no better than random here) |
| Δ-weighted segment averaging | CIF (integrate-and-fire with a learned weight predictor); SimulSLT (CIF for sign language); Mamba + UMA for ASR (CIF-like, separate weight layer); H-Net (learned dynamic chunking); Nawrot et al. 2023 (dynamic token pooling) | **The narrow version is new:** the SSM's *own* Δ as the integration weight, with no added parameters, fixed-length quantile segments, trained end to end |
| Context from previous clips | Sincan et al. ICCVW 2023 and Jang et al. CVPR 2025 use **text** context only; Zhang et al. ACL 2021 uses previous *audio* in speech translation | **New:** previous clips' *poses* as encoder context, plus a wrong-video control |
| Quantizing encoder memory, tokens × bits | TurboQuant (ICLR 2026; disputed by the RaBitQ authors); RDKV and HqeKV (token × bit trade-off for **LLM KV caches**) | **New only as an application:** quantizing encoder/cross-attention memory with Δ pooling. The quantizer itself is borrowed |

---

## 10. Conclusions, framing and next steps

### 10.1 What the paper can claim (3 seeds, 3 datasets, with controls and significance)
1. **Benchmark:** Mamba beats a matched Transformer on 3 sign languages, and the gap grows with clip length.
2. **Δ-based memory compression:**
   - Δ-weighted averaging always beats random segmentation and brings Mamba's compression cost down to the Transformer's level on iSign.
   - It beats plain averaging on iSign only.
   - Single-frame Δ selection is no better than random.
3. **Tokens × bits:** about 80× less decoder memory at 3 bits, for −0.1 to −0.7 chrF2. Use the MSE quantizer, not the unbiased one.
4. **Context:** previous clips' poses help a little (+0.2–0.3, beats wrong-video context) but don't scale with more clips.

**Proposed title:** *"Selective State Spaces for Sign Language Translation: What Helps, What Doesn't."* The target is an analysis paper (ACL/EMNLP Findings or a sign-language workshop such as SLTAT or SignLang@LREC).

### 10.2 Limitations reviewers will raise
- **Low absolute scores,** a field-wide problem on iSign (published SignPose-to-Text: BLEU-4 0.09–1.47). Our 3.7–4.0 BLEU-4 is above those numbers, but a direct comparison with published baselines on How2Sign and PHOENIX is still missing.
- **One model size** (about 34 M parameters); scaling behaviour is unknown.
- **Speed and memory profiling** (throughput, peak memory, latency at T = 128…1024) was planned in the benchmark plan but **not yet measured**.
- **The Transformer seed-42 stall on How2Sign** is reported as is.
- **Δ averaging is dataset-dependent,** with no clear mechanism for when it helps.

### 10.3 Recommended next steps
1. **Speed and memory profiling** of both encoders on one idle H200 (from the benchmark plan). It's cheap and strengthens the benchmark claim.
2. **Comparison with published baselines** (iSign paper, How2Sign and PHOENIX pose-based results).
3. **A Google Scholar novelty pass** ("Mamba dt pooling", "selective SSM integrate-and-fire", "Mamba sign language translation").
4. **Write the paper:** tables from this report, the quality-vs-memory figure, and related work from §9.
5. **Housekeeping before code release:** revoke the W&B API key committed in the public repo's scripts, and consider removing the unrelated files from the repo history.

---

## Appendix A: Experiment log
| Round | Date | What ran | Key outcome |
|---|---|---|---|
| Phase 1 | Aug–Sep 2026 | LR grid; 3 seeds × 2 models × 3 datasets; ensembles | Mamba > Transformer everywhere |
| Gate | 25–28 Sep | padfix, banks, uniform, Δ-matched (s42) | Δ > uniform (+0.3 test); banks no effect |
| Round 2 | 28 Sep | Δ/uniform seeds; 1/8, 1/32; averaging pair | Δ gap grows with compression |
| Δ analysis | 28 Sep | Δ vs keypoint speed on val | Δ doesn't track motion |
| Round 3 | 29 Sep | random-spacing control; How2Sign F3 rerun; test decodes | Δ selection = random |
| Round 4 | 30 Sep | C1 go/no-go; random averaging; jittered stride | Δ averaging > random; jitter explains random > uniform |
| Round 5 | 30 Sep–1 Oct | C1 seeds + k = 4/8; averaging on How2Sign/PHOENIX; full-memory baselines | Context small and not scaling; Δ averaging dataset-dependent |
| Quantization | 30 Sep–2 Oct | 12 settings × 3 models × 3 datasets; beam on iSign | 3-bit free; about 80× memory |
| Round 6 | 1–2 Oct | Transformer + plain averaging; leftovers | Compression not Mamba-specific; Δ closes Mamba's gap |

## Appendix B: Where things are in the repository
- `BENCHMARK_PLAN.md`: the Phase 1 protocol. `phase 2/PHASE2_PLAN.md`: the round-by-round Phase 2 record. `phase 2/NEXT_IDEAS.md`: backup ideas (surprise-based boundaries, confidence-gated Δ, adaptive thresholds).
- `src/`: Phase 1 code. `phase 2/src/`: Phase 2 code (`models/scope.py` pooling, `models/mamba_pool.py`, `models/context.py`, `models/pooled.py`).
- `phase 2/quant/`: the quantization code and tests. `phase 2/scripts/`: queue runners, analysis and CPU tests.
- `results/`, `results_f3/` and `phase 2/results/`: `metrics.xlsx`, prediction CSVs and quantization summaries per run.

## Sources (literature check)
- PhonSSM: https://arxiv.org/abs/2604.08761
- SignMamba (CSLR, journal page not fully opened): https://pubmed.ncbi.nlm.nih.gov/42660004/
- EvSLT: https://arxiv.org/abs/2408.10488
- How Effective are SSMs for MT? (WMT 2024): https://aclanthology.org/2024.wmt-1.111.pdf
- DeciMamba: https://arxiv.org/html/2406.14528
- DTP (ICLR 2026, secondary notes page): https://en.papernotes.org/ICLR2026/model_compression/dtp_delta-guided_two_stage_pruning_for_mamba-based_multimodal_large_language_mod/
- MTR, token reduction for Vision Mamba: https://arxiv.org/html/2507.14042
- H-Net, dynamic chunking: https://arxiv.org/html/2507.07955
- Mamba + unimodal aggregation for streaming ASR: https://arxiv.org/html/2410.00070v1
- SimulSLT: https://arxiv.org/abs/2112.04228
- Dynamic Token Pooling (Nawrot et al., ACL 2023): https://aclanthology.org/2023.acl-long.353/
- Is context all you need? (Sincan et al., ICCVW 2023): https://arxiv.org/abs/2308.09622
- Lost in Translation, Found in Context (Jang et al., CVPR 2025): https://arxiv.org/abs/2501.09754
- Reconsidering Sentence-Level SLT (Tanzer et al., 2024): https://arxiv.org/abs/2406.11049
- Beyond Sentence-Level End-to-End Speech Translation (Zhang et al., ACL 2021): https://aclanthology.org/2021.acl-long.200/
- TurboQuant: https://arxiv.org/abs/2504.19874 (RaBitQ dispute: https://arxiv.org/abs/2604.19528)
- RDKV: https://arxiv.org/abs/2605.08317 · HqeKV: https://aclanthology.org/2026.findings-acl.201.pdf
