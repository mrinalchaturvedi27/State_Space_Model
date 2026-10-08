# State-Space Models for Sign Language Translation: Approach, Results and Open Questions

*Research brief for expert review, October 2026. Unless stated otherwise, numbers are **test set, beam 5, length penalty 1.0, corpus chrF2 / BLEU-4 (sacreBLEU), mean ± std over 3 seeds**. Significance is paired bootstrap (1,000 samples), run separately for each seed. Detailed tables are in `REPORT.md` and `ABLATION_AND_ERROR_ANALYSIS.md`.*

---

## 1. Summary

1. **Benchmark.** On pose-based, gloss-free sign language translation, a **bidirectional Mamba-2 encoder beats a parameter-matched Transformer encoder** on three sign languages, on every seed (p ≤ 0.031). On iSign the gap **grows with input length**.
2. **The bottleneck is lexical recognition.** Both models produce fluent, on-domain sentences with the wrong content: on iSign and How2Sign, 52–67% of outputs share no content word with the reference (35–45% on PHOENIX). Mamba's advantage is mostly better *content*-word recognition, not better language.
3. **Most of our attempts at a method contribution gave small or negative results.** They are reported because the controls make them informative:
   - Mamba's selection signal Δ is no better than random for choosing frames.
   - Large-scale masked-pose pretraining (including story-length context) does not improve content recognition.
   - Context from previous clips helps only slightly.
4. **What does work beyond the benchmark:**
   - Δ-weighted memory compression (about 15×, about 80× with 3-bit quantization);
   - **selective translation**, where agreement across seeds identifies unreliable outputs and halves the failure rate at 40% coverage.

We would value advice on which of these to build the paper around, and on how to get past the recognition bottleneck.

---

## 2. Task, data, models

| | |
|---|---|
| **Task** | MediaPipe pose keypoints → text. Gloss-free, trained from scratch. |
| **Input** | 178 keypoints (body 8, face 128, hands 2×21), x and y only (356 features), per-clip shoulder normalisation, 25 fps, ≤ 512 frames per clip |
| **Datasets** | **iSign** (Indian SL→English, 100k train clips, mean 194 frames); **How2Sign** (ASL→English, 31k); **PHOENIX-2014T** (DGS→German, 7k) |
| **Encoders (only difference)** | Transformer: 6 layers, d = 512, 33.76 M total. Mamba: 5 bidirectional Mamba-2 layers (separate forward and backward scans), d_state = 64, 34.01 M total (+0.7%). Identical front end and 3-layer Transformer decoder. |
| **Training** | AdamW, cosine schedule, bf16, token-budget batching (identical frames per step), early stopping on val chrF2. Learning rate 3e-4, chosen from a 3-point grid; best for both models. Short warmup (≤ 10% of the run) on the two smaller datasets. |
| **Text** | SentencePiece unigram, vocabulary 4k (PHOENIX 1k) |

---

## 3. Main result: Mamba vs Transformer

| Dataset | Transformer chrF2 / BLEU-4 | Mamba chrF2 / BLEU-4 | Gap (chrF2) | p (per seed) |
|---|---|---|---|---|
| iSign | 19.06 ± 0.19 / 2.90 | **20.87 ± 0.30 / 3.74** | +1.81 | 0.001 / 0.001 / 0.001 |
| How2Sign | 16.75 ± 1.88 / 1.38 | **19.61 ± 0.53 / 2.33** | +2.86 | 0.001 / 0.001 / 0.001 |
| PHOENIX-2014T | 31.87 ± 0.25 / 10.64 | **33.58 ± 1.00 / 11.77** | +1.71 | 0.004 / 0.001 / 0.031 |

- **Gap by input length** (iSign, chrF2): +1.27 for ≤ 128 frames, +1.79 for 129–256, **+2.21** for > 256. BLEU-4: +20% → +26% → **+45%** relative.
- **Same learning-rate optimum.** Both encoders peak at the same learning rate, and Mamba leads at every rate tried.
- **A correctness fix to Mamba's backward scan** (padding was being read before real frames) raises iSign Mamba to **21.36 ± 0.20 / 4.00**.

### Error analysis (iSign, Phase 1 models)
| | Transformer | Mamba |
|---|---|---|
| Outputs sharing no content word with the reference | 64.2% | 54.7% |
| Content-word F1 | 10.1 | 13.9 |
| Function-word recall | 25.1 | 26.5 |
| Recall of names (capitalised words) | 13.6 | 19.4 |

- **Mamba gains most on sentences that are partly recognisable.** Difficulty is measured on other seeds to avoid regression to the mean.
- **It wins on 91% of the 172 test videos;** the two models' per-video scores correlate at 0.93.
- **How2Sign:** the Transformer collapses to stock sentences and repetition loops (60% unique outputs, 58% with repetition); Mamba much less so (84% unique, 28% with repetition).
- **Seeds almost never agree:** all three seeds give the identical output for only 0.3% of iSign clips.

**Known data issue:** How2Sign training targets come from a lowercased column with rare words replaced by `<UNKNOWN>`, while the val/test references are cased. That costs both models about 0.9 chrF2. A rebuild with consistent text is prepared, pending a check of the CSVs.

---

## 4. Method attempts and controls

### 4.1 Using Δ (Mamba's input-dependent step size) to compress the decoder memory
The idea: Δ decides how strongly each frame is written into the state, so it might mark the informative frames.

| Compression 1/16 (iSign, test, 3 seeds) | chrF2 |
|---|---|
| Keep every 16th frame | 19.87 |
| Keep frames at random, as unevenly spaced as Δ's | 20.46 |
| Keep the frames Δ chooses | 20.55 (vs random: significant on 1 of 6 comparisons) |
| Stride with a random shift during training | **20.67** |

- **Δ is no better than random for choosing frames.** The gain over the fixed stride is training-time variety, a regularisation effect.
- **Δ does not track motion:** per-clip Spearman ρ(Δ, hand speed) = 0.02.

**Δ as an averaging weight** (each group of frames averaged with Δ weights, group boundaries at equal shares of cumulative Δ, trained end to end):

| Test chrF2, about 15× compression | iSign | How2Sign | PHOENIX |
|---|---|---|---|
| Full memory | 21.36 | 19.92 | 33.94 |
| Plain average | 20.48 | 19.65 | **33.61** |
| Random groups and weights | 19.97 | 19.25 | 32.32 |
| **Δ-weighted average** | **20.93** | **19.86** | 33.35 |

- **Δ beats random grouping on every dataset.**
- **Δ beats plain averaging on iSign** (p ≤ 0.001 on every seed), is borderline on How2Sign, and does not on PHOENIX.
- **The same averaging costs a Transformer only −0.39 chrF2, against −0.88 for Mamba.** Δ weighting brings Mamba's cost down to −0.43. Our reading: recurrent states are more local than attention's, so plain averaging hurts them more.

**Quantization on top** (TurboQuant-style rotation + Lloyd–Max codebook, post hoc):
- **3 bits per value costs almost nothing on all datasets.** Combined with Δ averaging, the decoder memory is **about 80× smaller** (iSign: 198 KB → 2.4 KB per clip, −0.5 chrF2 on beam).
- **At 2 bits, results depend on the dataset:** the rotation is essential on iSign, while plain rounding is better on PHOENIX.
- **The unbiased (QJL) variant is always worse** than plain MSE quantization for this memory.

### 4.2 Context from previous clips (iSign, consecutive sentences of one story)
Motivation: 26% of a sentence's content words already appear in the previous 8 sentences, against 2.4% for a random other story. The encoder reads the previous clips' *poses*; the decoder sees only the current clip.

| Context | chrF2 | BLEU-4 |
|---|---|---|
| None | 21.23 | 3.77 |
| 2 clips from another story (control) | 21.19 | 3.78 |
| **2 previous clips** | **21.45** | **4.04** |
| 4 / 8 previous clips (1 seed) | 21.45 / 21.53 | 3.88 / 4.15 |

It beats the wrong-story control on every seed, but the gain is small and doesn't grow with more context. Recall of recurring names does not improve.

### 4.3 Self-supervised pretraining (masked-pose reconstruction)
Pretraining is on whole stitched story videos (4,096-frame windows) vs the same frames as single clips, then fine-tuning on iSign (seed 42).

| Pretraining | chrF2 | BLEU-4 | Content-word F1 |
|---|---|---|---|
| None | 21.26 | 4.10 | 14.97 |
| Small (5 epochs, iSign only): clips / story windows | 21.74 / 21.69 | 4.19 / **4.32** | 15.18 / **15.50** |
| Large (50 epochs, 3 datasets pooled): clips / story windows | 21.53 / 21.57 | 3.99 / 4.09 | 14.76 / 14.79 |

- **Pretraining gives at most about +0.3–0.5 chrF2.**
- **Scaling it up did not help** (slightly worse than small).
- **Story-length context gives no benefit** (story vs clips: p = 0.26).
- **Content recognition is unchanged.** Our reading: reconstructing keypoint coordinates teaches motion, not lexical meaning.

### 4.4 Decoding and selective translation (no retraining)
**Length penalty chosen on validation** (66 models):
- **Choosing by BLEU-4** changes BLEU-4 by +0.01 on average.
- **Choosing by chrF2** always picks the longest outputs allowed (+1.1 chrF2, −0.6 BLEU-4). chrF2 rewards length here, so we treat chrF2-only gains with caution.

**Consensus (MBR) decoding across the 3 seeds' beam candidates:**
- **Over all candidates:** +2 to +3.3 chrF2, but BLEU-4 is flat or lower (partly metric-driven).
- **Over the seeds' best outputs only:** +1.5–2.3 chrF2 and +0.03–0.3 BLEU-4.
- **Oracle candidate (upper bound):** iSign 28.8 chrF2 / 6.9 BLEU-4 vs 21.4 / 4.0 single. **The right answer is often among the candidates.**

**Selective translation, using agreement across seeds as confidence:**

| Model (test) | Failure-detection AUROC | chrF2, all → most confident 40% | Failure rate, all → 40% |
|---|---|---|---|
| iSign Mamba | 0.778 | 23.7 → **29.0** | 49.5% → **24.9%** |
| PHOENIX Mamba | 0.766 | 37.2 → **48.4** | 27.7% → **10.5%** |
| How2Sign Mamba | 0.711 | 22.3 → 25.0 | 47.5% → 29.9% |

- **The model's own log-probability is a weaker signal,** and on How2Sign it's useless (AUROC 0.46).
- **Mamba's confidence is more reliable than the Transformer's** (iSign 0.778 vs 0.758; How2Sign 0.711 vs 0.596).

---

## 5. Our interpretation

1. **The SSM's advantage is real and comes from better integrating long, continuous motion.** That fits the gap growing with length and the content-word gains.
2. **The ceiling is the mapping from pose to words, not the language model.** Neither Mamba-specific mechanisms nor self-supervised motion objectives fix it. Supervision with *meaning* seems required.
3. **Δ is informative only when it's trained for the use it's put to.** Its raw values are not an interpretable event detector.
4. **Individual models are high-variance and poorly calibrated, but their disagreement is a strong reliability signal.** That's directly useful for assistive deployment.
5. **At these score levels, chrF2 rewards length.** BLEU-4 and content-word F1 are needed alongside it.

---

## 6. Novelty status (literature check, 2024–2026; not exhaustive)
| Contribution | Status |
|---|---|
| Matched SSM vs Transformer for pose-based SLT, with the length analysis | No prior work found (closest: SSMs for sign *recognition*; SSM vs Transformer for *text* MT at WMT 2024) |
| Selective translation with seed agreement and coverage curves | New as framed. Sentence reliability scoring exists (ICLR 2026) |
| Δ-weighted pooling (the SSM's own Δ as the integration weight, no extra parameters) | Narrow novelty. Related: CIF, SimulSLT, H-Net dynamic chunking; DeciMamba (Δ for token *selection*) |
| Previous-clip *pose* context with a wrong-story control | New in form. Context-aware SLT so far uses text context |
| Long-context self-supervised pose pretraining | Not previously done, but it **did not work** here |
| Cross-modal reranking; content-word auxiliary loss | Already done (Zhao et al. 2021; Sign2GPT 2024) |

---

## 7. Questions for you

1. **Framing:** is a controlled benchmark plus analysis paper (benchmark, compression, selective translation, informative negative results) strong enough for an ACL/EMNLP Findings or sign-language workshop venue? Which result would you lead with?
2. **The recognition bottleneck:** with about 7k–100k labelled sentences, what would you try to teach *lexical* content from poses? For example: contrastive pose–text pretraining on external data (YouTube-ASL / YouTube-SL-25), pseudo-gloss supervision, or richer hand features (crops or 3D hand models instead of 21 keypoints).
3. **Input representation:** is 2D MediaPipe pose (178 points) likely the binding ceiling? Would you expect RGB or hand-crop features to change the Mamba-vs-Transformer comparison?
4. **SSM-specific directions:** is streaming translation with Mamba's carried state worth pursuing, given that context barely helped? Is Δ-compressed memory as an LLM front end (about 13 tokens per clip) a promising direction?
5. **Evaluation:** at BLEU-4 ≈ 2–12, which metrics would you trust (BLEURT, COMET, LLM judges, human evaluation)? Is content-word F1 a reasonable complement?
6. **Selective translation:** what would make it convincing for an assistive-technology audience (user studies, cost-aware abstention, comparison with the ICLR 2026 reliability score)?
7. **Rigor:** are 3 seeds with per-seed paired bootstrap adequate, or should we pool seeds differently or report confidence intervals?
8. **Anything we should drop, or a control we've missed?**
