# Ablations and Error Analysis: Mamba vs Transformer for Sign Language Translation

*Companion to `REPORT.md`. As of 2 October 2026.*

*Unless stated otherwise: **test set, beam 5, length penalty 1.0, corpus chrF2, mean over 3 seeds (13, 42, 1337)**.*

*Length-penalty tuning on validation is still running, so the BLEU-4 values here will move once it finishes (§2.4).*

---

## Summary

1. **Most of the gain comes from the encoder itself.** Swapping the Transformer for Mamba is worth +1.8 to +2.9 chrF2. Handling padding per clip in the backward scan adds +0.49 on iSign. The multi-timescale init adds nothing.
2. **Both models mostly fail in the same way:** fluent, on-domain sentences that don't match the meaning. **54–66% of iSign/How2Sign test sentences share no content word with the reference.**
3. **Mamba is better because it gets more content right, not because it writes better English.**
   - Content-word F1 is +3.7 to +5.0 points higher; recall of names is +5.9.
   - It falls back on generic stock sentences far less often (How2Sign: 84% vs 60% unique outputs) and loops less (How2Sign: 28% vs 58% of outputs repeat themselves).
   - It wins on 91% of iSign videos.
4. **Three problems in the evaluation setup** surfaced:
   - **How2Sign's training text doesn't match its test text:** about half the predictions contain a literal `<UNKNOWN>` token.
   - **The Phase 1 PHOENIX runs were under-trained** (both models).
   - **iSign outputs are about 9% too short,** which costs BLEU-4 directly.
5. **Δ-weighted averaging keeps what plain averaging loses.** After 15× compression it keeps 96% of the full model's name recall (plain averaging keeps 89%) and recovers 0.7 of the 1.6-point content-word F1 loss.

---

## 1. Ablations

### 1.1 Encoder and training choices (iSign)
| Change | Test chrF2 | BLEU-4 | Effect |
|---|---|---|---|
| Transformer (6 layers, 33.76 M) | 19.06 ± 0.19 | 2.90 | — |
| → Mamba (5 bidirectional Mamba-2 layers, 34.01 M) | 20.87 ± 0.30 | 3.74 | **+1.81** |
| → + padding handled per clip in the backward scan | 21.36 ± 0.20 | 4.00 | **+0.49** |
| → + multi-timescale init (16 heads at 3 memory spans, s42) | 21.29 | 3.90 | +0.03 vs the same seed (none) |

**Learning rate** (seed 42, best val chrF2):

| | 1e-4 | **3e-4** | 1e-3 |
|---|---|---|---|
| Transformer | 17.88 | **18.92** | 18.56 |
| Mamba | 19.43 | **20.74** | 20.29 |

Both models peak at the same learning rate, and Mamba leads at every rate (+1.5 to +1.8). So the result is not an artefact of tuning.

### 1.2 Training schedule on the smaller datasets
| Dataset | Model | Default schedule (4,000-step warmup) | Short warmup (≤ 10%), early stopping after warmup | Effect |
|---|---|---|---|---|
| How2Sign | Transformer | 16.21 ± 1.02 | 16.75 ± 1.88 | +0.54 |
| How2Sign | Mamba | 18.93 ± 2.04 | 19.61 ± 0.53 | +0.68, and variance cut 4× |
| PHOENIX | Mamba | 31.72 ± 0.92 | 33.94 ± 0.20 * | **+2.22** |
| PHOENIX | Transformer | 29.13 ± 0.70 | **not run** | — |

\* That PHOENIX run also has the per-clip padding fix, so the two effects are mixed.

**Implication:** the Phase 1 PHOENIX comparison (Transformer 29.13 vs Mamba 31.72) compares two *under-trained* models. Its 3-seed curves peaked at the last epochs, consistent with that. For a fair final PHOENIX number, the Transformer needs the short-warmup schedule too: 3 runs, under an hour each on PHOENIX.

### 1.3 Decoding
| Dataset (Mamba, full memory) | Greedy | Beam 5 | Beam − greedy |
|---|---|---|---|
| iSign (s42) | 20.53 | 21.26 | **+0.73** |
| PHOENIX (3 seeds) | 33.36 | 33.94 | **+0.58** |
| How2Sign (3 seeds) | 20.74 | 19.92 | **−0.82** |

Beam search *hurts* on How2Sign. §2.6 shows why: beam produces the training-only `<UNKNOWN>` token far more often.

**Length penalty, iSign seed 13** (test sweep, *descriptive only*; the validation-chosen value is pending):

| Length penalty | 1.0 | 1.1 | 1.2 | 1.3 | 1.5 |
|---|---|---|---|---|---|
| Mamba BLEU-4 | 4.07 | 4.13 | 4.16 | **4.19** | 4.09 |
| Mamba chrF2 | 21.21 | 21.49 | 21.74 | 21.95 | **22.27** |
| Transformer BLEU-4 | 2.99 | 2.99 | 2.99 | 2.96 | 2.89 |

Longer outputs help Mamba on both metrics and leave the Transformer flat. That fits the brevity finding in §2.4.

### 1.4 Decoder-memory compression (iSign test; the same number of memory tokens within each column)
**Keeping single frames:**

| How frames are chosen | 1/8 of frames (s42) | 1/16 (3 seeds) | 1/32 (3 seeds) |
|---|---|---|---|
| Fixed stride | 20.82 | 19.87 ± 0.20 | 18.81 ± 0.03 |
| Random (as uneven as Δ) | — | 20.46 ± 0.06 | 19.68 ± 0.05 |
| Chosen by Δ | 20.89 | 20.55 ± 0.21 | 19.65 ± 0.11 |
| Stride with random shift during training | — | **20.67 ± 0.12** | — |

**Averaging each group of 16 frames (3 seeds; full memory = 21.36):**

| How groups are averaged | Test chrF2 | Cost vs full memory |
|---|---|---|
| Plain average | 20.48 ± 0.27 | −0.88 |
| Random groups and weights | 19.97 ± 0.08 | −1.39 |
| **Δ-weighted** | **20.93 ± 0.09** | **−0.43** |
| Δ-weighted + random shift during training | 20.69 ± 0.15 | −0.67 |
| *Transformer* + plain average (vs Transformer full 19.06) | 18.66 ± 0.19 | −0.39 |

**What this shows:**
- **Keeping single frames:** Δ is no better than random, and randomness during training is what beats the fixed stride.
- **Averaging:** Δ weighting is the best option, and it brings Mamba's compression cost down to the Transformer's level.
- **Other datasets** (from `REPORT.md` §5): Δ beats random everywhere. It beats plain averaging on iSign, is borderline on How2Sign, and doesn't on PHOENIX.

### 1.5 Context from previous clips (iSign)
| Context | Seeds | Test chrF2 | BLEU-4 |
|---|---|---|---|
| None | 3 | 21.23 ± 0.18 | 3.77 |
| 2 clips from a different video | 3 | 21.19 ± 0.12 | 3.78 |
| **2 previous clips** | 3 | **21.45 ± 0.08** | **4.04** |
| 4 previous clips | 1 | 21.45 | 3.88 |
| 8 previous clips | 1 | 21.53 | 4.15 |

### 1.6 Memory quantization (summary; full tables in `REPORT.md` §7)
- **4 and 3 bits cost almost nothing** on all datasets (≤ 0.14 chrF2).
- **2 bits depends on the dataset:** rotation is essential on iSign, but plain rounding is better on PHOENIX.
- **The unbiased (QJL) variant is always worse** at 3 bits (−0.15 to −0.28).

### 1.7 Ablations from the original plan that were not run
These were in the original plan but not run:
- unidirectional vs bidirectional Mamba;
- positional encoding (Transformer without it, Mamba with it);
- depth at fixed parameters;
- the source-length cap T ∈ {128, 256, 512, 1024};
- a temporal downsampling stem;
- the keypoint subset (hands+body only vs the full 543 points);
- a hybrid Mamba+attention encoder.

The cheapest and most informative of these are **unidirectional vs bidirectional** (also needed for streaming) and **the length cap** (it directly tests the length claim).

---

## 2. Error analysis (Phase 1, Transformer vs Mamba)

### 2.1 What the outputs look like
| Measure (3-seed mean) | iSign TF | iSign Mamba | How2Sign TF | How2Sign Mamba | PHOENIX TF | PHOENIX Mamba |
|---|---|---|---|---|---|---|
| Sentences with sentence-chrF2 < 10 ("total miss") | 9.5% | 8.5% | 18.1% | **12.9%** | 0.6% | 0.9% |
| Sentence-chrF2 10–25 | 73.5% | 68.0% | 73.2% | 68.7% | 56.9% | 46.2% |
| Sentence-chrF2 25–50 | 15.4% | **21.1%** | 8.4% | **17.8%** | 30.1% | **39.1%** |
| Sentence-chrF2 ≥ 50 | 1.5% | 2.4% | 0.2% | 0.6% | 12.4% | 13.8% |
| Exact match | 0.19% | 0.23% | 0.03% | 0.06% | 0.93% | 1.40% |
| **No content word shared with the reference** | 64.2% | **54.7%** | 66.5% | **55.1%** | 45.3% | **35.4%** |

The main failure is **fluent but unrelated output**. Most sentences land at sentence-chrF2 10–25, which is roughly what an on-domain sentence scores against the wrong reference. On iSign and How2Sign, more than half of outputs share no content word with the reference. Mamba's improvement is mostly sentences moving from that band into the 25–50 band (+5.7 points of the test set on iSign, +9.4 on How2Sign, +9.0 on PHOENIX).

### 2.2 Where Mamba gains: content, not fluency
| Measure (3-seed mean) | iSign TF | iSign Mamba | How2Sign TF | How2Sign Mamba | PHOENIX TF | PHOENIX Mamba |
|---|---|---|---|---|---|---|
| Content-word precision | 11.3 | **15.6** | 6.4 | **11.0** | 19.7 | **26.1** |
| Content-word recall | 9.2 | **12.5** | 5.7 | **9.0** | 18.5 | **22.4** |
| Content-word F1 | 10.1 | **13.9** | 6.0 | **9.9** | 19.1 | **24.1** |
| Function-word recall | 25.1 | 26.5 | 29.0 | 33.9 | 34.6 | 37.9 |
| Name precision / recall * | 16.2 / 13.6 | **22.5 / 19.4** | — | — | — | — |
| BLEU unigram / 4-gram precision | 21.7 / 0.72 | **25.0 / 0.98** | 12.4 / 0.20 | **17.5 / 0.39** | 25.3 / 4.6 | **31.0 / 6.1** |

\* Names = capitalised words not at the start of a sentence. Not applicable to How2Sign (lowercased outputs, §2.6) or PHOENIX (all lowercase).

**What this shows:**
- **The content-word gains are large:** +37% relative on iSign, +65% on How2Sign, +26% on PHOENIX.
- **Function-word recall barely moves.** Both models already produce grammatical English and German; Mamba's advantage is in recognising what is signed (nouns, verbs, names).
- **Mamba gets more names right** (19.4% vs 13.6% recall), even though recurring names stay hard.

**Sentence by sentence (same seed, sentence chrF2 Mamba − Transformer):**

| Dataset | Mamba better by > 5 | Within ± 5 | Transformer better by > 5 |
|---|---|---|---|
| iSign | 28.0% | 55.6% | 16.4% |
| How2Sign | 32.4% | 53.1% | 14.5% |
| PHOENIX | 32.0% | 50.6% | 17.4% |

**Which sentences gain most?** Difficulty here is measured by the Transformer's score from the *other two* seeds. Binning on the same run's score would create a regression-to-the-mean artefact; that naive version suggested the opposite.

| Dataset | Hardest 25% | 25–50% | 50–75% | Easiest 25% |
|---|---|---|---|---|
| iSign | +0.95 | +1.43 | +1.76 | **+2.86** |
| How2Sign | +1.58 | +2.11 | +3.00 | **+3.32** |
| PHOENIX | **+3.56** | +1.84 | +2.82 | +3.23 |

On iSign and How2Sign, **Mamba gains most where the signing is partly recognisable**, and helps least on sentences neither model can read. By reference length the gain is flat on iSign (+1.65 / +1.78 / +1.83 for ≤ 8 / 9–16 / > 16 words). The source-length trend still holds: +1.27 → +1.79 → +2.21 chrF2 for clips ≤ 128 / 129–256 / > 256 frames.

### 2.3 Collapse into stock sentences and repetition loops
| Measure | iSign TF | iSign Mamba | How2Sign TF | How2Sign Mamba | PHOENIX TF | PHOENIX Mamba |
|---|---|---|---|---|---|---|
| Unique outputs / test sentences | 98.2% | 98.7% | **59.7%** | 83.6% | **65.3%** | 82.4% |
| Share of the 10 most frequent outputs | 1.0% | 0.9% | **14.2%** | 1.8% | **21.0%** | 14.4% |
| Distinct bigrams, outputs (references) | 39.5 (57.8) | 40.0 | 9.5 (45.1) | 22.3 | 9.4 (54.1) | 14.0 |
| Words used (references use) | 6,023 (8,252) | 5,789 | 893 (3,441) | 1,711 | 244 (1,001) | 307 |
| Outputs with a repeated 3-word phrase or word | 6.3% | 6.4% | **57.8%** | 28.1% | 8.4% | 7.2% |

**Most frequent outputs (seed 42):**

| Dataset | Transformer | Mamba |
|---|---|---|
| How2Sign | "i am going to show you how to make a `<UNKNOWN>` `<UNKNOWN>`" (6.7% of all outputs); "…how to make a `<UNKNOWN>` `<UNKNOWN>` `<UNKNOWN>` `<UNKNOWN>`" (6.2%); "the next thing we are going to talk about is…" (4.8%) | top output is "good" (0.3%) |
| PHOENIX | "ihnen einen schönen abend und machen sie es gut" (5.6%); "der wind weht schwach bis mäßig aus unterschiedlichen richtungen" (5.0%) | the same two sentences at 4.0% and 3.0% |
| iSign | "Let me tell you about it." 18 times (0.3%) | the same sentence, 14 times |

**What this shows:**
- **On How2Sign the Transformer largely collapses** to a few "instructional video" templates and repetition loops. Mamba collapses far less.
- **On PHOENIX both models fall back on the weather-report stock phrases**, which is the classic mode collapse of a low-resource domain.
- **On iSign collapse is negligible.** Errors there are diverse but unrelated sentences.

### 2.4 Length: iSign outputs are too short
| Dataset | Output / reference length (TF / Mamba) | BLEU brevity penalty (TF / Mamba) | Outputs > 1.5× the reference length |
|---|---|---|---|
| iSign | 0.93 / 0.91 | 0.92 / 0.90 | 9.2% / 7.9% |
| How2Sign | **1.21** / 1.04 | 1.00 / 1.00 | **30.7%** / 18.5% |
| PHOENIX | 0.98 / 0.92 | 0.97 / 0.92 | 10.0% / 6.8% |

- **On iSign and PHOENIX, Mamba's outputs are 8–9% too short.** BLEU's brevity penalty then removes about 10% of its BLEU-4. This is why a validation-chosen length penalty (running now) should lift Mamba's BLEU-4 more than the Transformer's.
- **On How2Sign the Transformer is too long,** because of repetition loops.

### 2.5 Variance across seeds and videos
- **Different seeds almost never agree.** All 3 seeds give the identical output for only 0.3% of iSign clips, 0.0% of How2Sign and 4–5% of PHOENIX. Individual translations are therefore very seed-dependent, which is why averaging seeds and ensembling help. A 3-seed Mamba ensemble scores 4.91 BLEU-4 vs 3.74 for single models; ensemble length penalties were not chosen on validation.
- **iSign per video** (172 test videos with ≥ 15 clips): video-level chrF2 ranges from 12.4 to 38.6 (sd 3.5).
  - **Mamba beats the Transformer on 91.3% of videos.**
  - The two models' per-video scores correlate at 0.93. Difficulty is mostly a property of the video (signer, topic, recording), and Mamba's advantage is consistent across it.
- **Position in the story** (iSign, Mamba without context): the first clip of a video scores 28.3, the second 24.1, later ones 20.9. Opening clips are titles and formulaic introductions.

### 2.6 How2Sign: training text doesn't match test text
| Seed 42 | Outputs containing the literal token `<UNKNOWN>` | Outputs with any uppercase other than `<UNKNOWN>` | References with uppercase | References containing `<UNKNOWN>` |
|---|---|---|---|---|
| Transformer (test) | 54.2% | ≈ 0% | 99.9% | 0% |
| Mamba (test) | 45.4% | ≈ 0% | 99.9% | 0% |

The models only ever produce **lowercase text with `<UNKNOWN>` tokens**, while every reference is normally cased and has no such token. The training targets must therefore have been preprocessed differently: lowercased, with rare words replaced by `<UNKNOWN>`. The val/test references used the raw text. That's strong evidence but should be **verified against the How2Sign CSVs on the server**.

**Impact (3-seed mean, test):**

| | As scored | Lowercased | Lowercased + `<UNKNOWN>` removed |
|---|---|---|---|
| Transformer chrF2 / BLEU-4 | 16.75 / 1.38 | 17.69 / 1.55 | 17.60 / 1.81 |
| Mamba chrF2 / BLEU-4 | 19.61 / 2.33 | 20.53 / 2.63 | 20.43 / 2.75 |

- **Both models lose about 0.9 chrF2 and 0.4 BLEU-4 to this.** The Mamba vs Transformer comparison is unaffected, since both are trained and scored the same way.
- **The proper fix** is to rebuild the How2Sign targets from one consistent text column for all splits and retrain. Post-hoc lowercasing doesn't recover the content of words replaced by `<UNKNOWN>`.
- **This also explains the beam-search penalty (§1.3).** On Mamba full memory, beam-5 outputs contain `<UNKNOWN>` 44–55% of the time vs 25–32% for greedy, because beam favours that high-frequency token.

### 2.7 Examples (iSign, seed 42, sentence chrF2 in brackets)
**Mamba gets the content, the Transformer produces a template:**

| Reference | Transformer | Mamba |
|---|---|---|
| Color the parts of the plant. | Complete this table according to your observations. (15) | Color the parts of the plant. (100) |
| Madhya Pradesh Home Minister Narottam Mishra said that | NPAD's Union Home Minister Rajnath Singh said, (30) | Madhya Pradesh Home Minister Narotttam Mishra said, (90) |
| Yesterday at the India vs England match, | India and Australia are playing a match against each other. (28) | Yesterday the India vs England match was held. (80) |

**The Transformer is better (Mamba loops or drifts):**

| Reference | Transformer | Mamba |
|---|---|---|
| Here is a sentence given below. | Here is a sentence given below. (100) | Fill in the blanks with the words given below. (39) |
| must work very hard for PM Modi to win the elections. | PM Modi wanted to work hard to win the election. (52) | They all agreed to this and they all agreed to this. (12) |

**Both fail (typical: an on-domain classroom template, wrong content):**

| Reference | Transformer | Mamba |
|---|---|---|
| Next, divide it by 2. | Following the blank in brackets. Now trace the dotted lines. (8) | Complete the right side. (11) |
| Which one is right? 2 | Answer the following questions. 1. (9) | A. Fill in the blanks. 2 (8) |

iSign mixes news broadcasts with educational/classroom content. Both models have learned the domain's stock phrases ("Fill in the blanks", "Trace the dotted lines", "Complete this table"), and these dominate the failures.

---

## 3. Error analysis of Phase 2

### 3.1 What compressing the memory loses (iSign test, 3 seeds, about 15× compression)
| Model | chrF2 | Content-word F1 | Name recall | No content overlap | chrF2 by clip length (≤ 128 / 129–256 / > 256) |
|---|---|---|---|---|---|
| Transformer, full | 19.06 | 10.1 | 13.5 | 64.2% | 17.84 / 19.33 / 19.44 |
| Transformer + plain average | 18.66 | 9.4 | 12.9 | 65.8% | 17.42 / 18.83 / 19.22 |
| Mamba, full | 21.36 | 14.0 | 19.8 | 53.7% | 19.61 / 21.66 / 22.04 |
| Mamba + plain average | 20.48 | 12.4 | 17.7 | 57.7% | 18.98 / 20.66 / 21.20 |
| Mamba + random average | 19.97 | 11.5 | 16.3 | 60.5% | 18.30 / 20.23 / 20.66 |
| **Mamba + Δ-weighted average** | **20.93** | **13.1** | **19.0** | 56.7% | 19.37 / 21.16 / 21.59 |

- **Compression costs content, not fluency.** Output length and uniqueness are unchanged (length ratio 0.94–0.95, 98–99% unique).
- **Δ weighting keeps much more content than plain averaging:**

  | | Plain average | Δ-weighted |
  |---|---|---|
  | Name recall kept | 17.7 of 19.8 (89%) | **19.0 of 19.8 (96%)** |
  | Content-word F1 loss recovered | — | **0.7 of 1.6** |

- **On iSign the loss is spread evenly across clip lengths.**
- **On How2Sign, Δ averaging matches or beats full memory on long clips** (21.15 vs 21.07, above the median length) and loses on short ones (16.97 vs 17.39).
- **On PHOENIX it's the opposite:** plain averaging is best on long clips (33.73 vs Δ 33.09), Δ on short ones.

There is no single length rule for when Δ helps.

### 3.2 What context changes (iSign test, 3 seeds)
| Context | chrF2 | Content-word F1 | Name precision / recall | No content overlap | chrF2 by position: 1st clip / 2nd / 3rd+ |
|---|---|---|---|---|---|
| None | 21.23 | 13.59 | 22.4 / 19.5 | 54.6% | 28.33 / 24.09 / 20.92 |
| 2 clips, different video | 21.19 | 13.80 | 22.2 / 20.0 | 54.3% | 27.97 / 24.24 / 20.87 |
| 2 previous clips | **21.45** | **14.19** | 22.1 / 20.1 | **53.5%** | 29.36 / 24.86 / 21.08 |

- **Context raises content-word F1 a little (+0.6) but not name recall.** The motivating example, names that recur across a story, isn't what improves.
- **The model trained with context also scores higher on each video's *first* clip** (+1.0), where it sees no context at all. Part of the gain is therefore a training-run effect rather than context.
- **Clips 2 onwards gain only +0.16–0.77.**

---

## 4. What to fix or run next (ordered by value for the paper)
1. **How2Sign targets:** confirm the train/test text mismatch on the server, rebuild the targets from one consistent column, and retrain the How2Sign models (both encoders, 3 seeds). Expect about +1 chrF2 for both, plus more fluent and cased outputs.
2. **PHOENIX Transformer with the short-warmup schedule** (3 seeds, about 1 GPU-hour). Without it, the PHOENIX comparison uses an under-trained Transformer.
3. **Length penalty chosen on validation** (running). It targets iSign's −9% length and brevity penalty.
4. **Repetition control in decoding** (no repeated 3-word phrases), tested on validation. It's cheap, and How2Sign has 28–58% of outputs looping.
5. **The two cheapest missing ablations:** unidirectional vs bidirectional Mamba, and the source-length cap (128 / 256 / 512 / 1,024).

*How these numbers were produced: sacreBLEU 2.x (chrF2 and BLEU-4 with 13a tokenisation, case-sensitive unless stated). Content words = lowercase tokens not in an English/German function-word list and not digits, counted as clipped multisets. Per-video statistics cover iSign test videos with ≥ 15 clips. Per-video scores are saved in the analysis scratch file and can be added to the repo on request.*
