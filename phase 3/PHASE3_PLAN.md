# Phase 3: from a benchmark to a method

*Phase 1 (`BENCHMARK_PLAN.md`) showed Mamba beats a matched Transformer on three sign languages. Phase 2 (`phase 2/PHASE2_PLAN.md`) showed what Mamba's Δ can and cannot do. Phase 3 attacks the bottleneck the error analysis found (`ABLATION_AND_ERROR_ANALYSIS.md`).*

---

## 0. Why Phase 3: the bottleneck is recognising content, not producing language

Measured on test, 3 seeds (`ABLATION_AND_ERROR_ANALYSIS.md` §2):

| Finding | Number | What it means |
|---|---|---|
| Outputs sharing **no content word** with the reference | iSign 55–64%, How2Sign 55–67%, PHOENIX 35–45% | most errors are fluent, on-domain sentences about the wrong thing |
| Function-word recall vs content-word recall (iSign, Mamba) | 26.5 vs 12.5 | grammar is already learned; *which signs mean which words* is not |
| Mamba's gain over the Transformer | content-word F1 +37 to +65% relative; function words barely change | a better encoder helps exactly where the bottleneck is |
| Mamba's gain grows with clip length | +1.27 → +1.79 → +2.21 chrF2 | the SSM advantage is about integrating long temporal context |
| Identical outputs across the 3 seeds | 0.3% (iSign), 0.0% (How2Sign) | individual models are very high-variance; a seed ensemble gave +1.2 BLEU-4 |
| Context from 2 previous clips | +0.22 chrF2, not growing with 4 or 8 clips | supervised training on 100k labelled sentences can't learn to use long context |

**Diagnosis:** with about 7k–100k labelled sentences, the encoder doesn't learn a rich enough mapping from movement to meaning. The decoder fills the gap with the domain's stock sentences. The fix has to give the *encoder* more signal than the labels provide. The strongest SSM-specific way to do that is **self-supervised pretraining on long, untrimmed sign video**, which a Transformer can't afford.

---

## 1. Hypotheses

| # | Hypothesis | Tested by |
|---|---|---|
| H1 | Self-supervised pose pretraining improves *content* recognition (content-word F1, no-overlap rate), not just chrF2 | Stage 3.1, P0 vs P1/P2 |
| H2 | Pretraining on **whole untrimmed videos** beats pretraining on the **same data cut into clips** | Stage 3.1, P1 vs P2 (same frames, same steps) |
| H3 | The SSM's advantage over the Transformer **grows** under long-context pretraining, because long context is cheap for the SSM and quadratic for attention | Stage 3.1, P2 vs T1/T2 + cost curves |
| H4 | Pretraining on three sign languages together helps each, most of all the small ones (PHOENIX 7k, How2Sign 31k). Secondary: partly prior work | Stage 3.2 |
| H5 | Seed disagreement is exploitable: consensus decoding beats any single model, and disagreement predicts when a translation is wrong | Stage 3.0 |

---

## 2. Stage 3.0: no training, starts immediately (existing checkpoints)

### 3.0a Consensus (MBR) decoding across seeds
- **Candidates:** for each test clip, the beam-5 n-best lists (5 hypotheses each) of the 3 seed models, giving 15 candidates. Optionally also the full-memory and Δ-averaged models: different systems agree less, which can help MBR.
- **Choice:** pick the candidate with the highest *average* sentence-chrF2 against all the others (minimum Bayes risk with a chrF utility). Also try a BLEU utility.
- **Controls:** the best single seed; the 3-seed logit ensemble (already 4.91 BLEU-4 on iSign, with length penalties to be re-chosen on validation); and an *oracle* pick (best candidate by its true score), as an upper bound.
- **Datasets:** iSign, How2Sign, PHOENIX, Transformer and Mamba each.
- **Success:** MBR ≥ the logit ensemble, with no extra training.

### 3.0b Selective translation (knowing when it doesn't know)
- **Confidence signals per sentence:**
  - agreement between seeds (mean pairwise chrF2 of the 3 outputs);
  - the beam's length-normalised log-probability and the score gap between its top 2 hypotheses;
  - how varied Δ is across the clip;
  - clip length.
- **Evaluation:**
  - Spearman correlation of each signal with sentence chrF2;
  - **coverage vs quality curves**: chrF2 of the kept sentences at 100 / 80 / 60 / 40% kept, and how often a "no content overlap" output is caught.
- **Why it matters:** with more than half of outputs unrelated to the signing, an assistive system must abstain rather than mislead.
- **Positioning:** sentence-level reliability scoring exists (Grounding or Guessing?, ICLR 2026; §11) and is compared as a baseline signal. The new parts are abstention with coverage–quality curves, and seed disagreement / MBR consensus as the confidence signal.

### 3.0c Phase 2 fixes (already queued)
- Length penalty chosen on validation (`phase 2/scripts/run_lp_tune.sh`).
- PHOENIX pair rerun with the short-warmup schedule (`scripts/run_fixes.sh PART=phoenix`).
- How2Sign retrained on consistently cased text (`PART=how2sign`, after `scripts/check_text_columns.py`).

These give the corrected baselines Phase 3 is measured against.

---

## 3. Stage 3.1: long-context self-supervised pretraining (the headline)

### 3.1.1 Data
- **Stitched videos.** Each clip has a uid of the form `<video>-<segment>` (iSign) or `<video>_<sentence>` (How2Sign). Order a video's clips by segment number and concatenate their cached pose frames. The result is untrimmed story videos of roughly 5,000 frames each for iSign (about 26 clips × 218 frames).
- **Gaps:** segments filtered out of the dataset leave gaps. Mark each gap with a boundary flag, the same mechanism as the Phase 2 context model.
- **Pretraining corpus: train splits only**, so no test poses are seen even without labels. iSign has about 3,860 train videos; How2Sign about 31k sentence clips (videos to be counted); PHOENIX 7k clips (broadcasts to be counted).
- **Unlabelled extras:** the iSign pose directory holds about 127k clips against 111.6k labelled ones. Check whether the roughly 15k unlabelled clips belong to train videos; if so they can be added.
- **Training window:** random 4,096-frame windows of the stitched videos (about 164 s at 25 fps). Clip-level runs use the same frames, cut at segment boundaries.

### 3.1.2 Objective
1. **Masked frame modelling (main):**
   - mask contiguous spans of 8–32 frames (about 30% of frames in total);
   - predict the masked keypoint coordinates with an L1 loss;
   - also a **hand-focused** variant: mask the hand keypoints only in some spans. Hands carry most lexical content.
2. **Next-segment prediction (optional, for the forward scan):** predict a pooled summary of the next clip from the state at a clip boundary. This trains the "carry the story" ability that context failed to learn from labels.

The **bidirectional Mamba-2 encoder** is kept as is: the same 5 layers and 34 M parameters, plus a small reconstruction head that is discarded after pretraining.

### 3.1.3 Fine-tuning
- **The standard Phase 1 recipe:** clip-level translation, same decoder, same schedule, same seeds, learning rate re-chosen from {1e-4, 3e-4} on validation.
- **Two settings:**
  - **full fine-tuning;**
  - **frozen encoder**, which shows how much meaning the pretraining alone put into the representation.
- **Also:** the context model (2 previous clips) on top of the pretrained encoder. Pretraining may be what makes context usable.

### 3.1.4 Experiment grid (iSign first, 3 seeds per cell)
| ID | Encoder | Pretraining | Question |
|---|---|---|---|
| P0 | Mamba | none (= Phase 2 full memory, 21.36) | baseline |
| P1 | Mamba | masked frames on **clips** | does pretraining help? (H1) |
| **P2** | Mamba | masked frames on **4,096-frame stitched videos**, same frames and steps as P1 | does *long context* help? (H2) |
| P2-hand | Mamba | as P2, hand-focused masking | do hands matter for pretraining? |
| T0 | Transformer | none (19.06) | baseline |
| T1 | Transformer | masked frames on **clips** | pretraining with attention |
| T2 | Transformer | masked frames on 4,096-frame windows, *if it fits*; else local-window attention | can attention use long context at all, and at what cost? (H3) |
| P2-scale | Mamba | P2 on 25 / 50 / 100% of the pretraining videos | data-scaling curve |

**Measured for every cell:**
- **Quality:** test chrF2 and BLEU-4 (beam 5, length penalty chosen on validation).
- **Content (the bottleneck metrics):** content-word F1, the no-content-overlap rate, name recall.
- **Length breakdown:** the gain by clip length.
- **Cost:** pretraining GPU-hours, peak memory, throughput.

**Go/no-go gate (after seed 42 of P1 and P2):**
- **Continue with Stage 3.1** if P2 > P0 by ≥ 0.5 test chrF2 *and* content-word F1 improves.
- **Make long context the headline** if P2 > P1 as well.
- **If neither holds,** fall back to Stage 3.3 as the method.

### 3.1.5 Compute (estimate, to be remeasured)
- **Pretraining corpus:** about 22M iSign train frames, plus How2Sign and PHOENIX for Stage 3.2.
- **Per pretraining run:** about 5 epochs at about 24k frames per step, roughly 4.5k steps, which is about 6–10 GPU-hours on one H200 for the 34 M Mamba.
- **Fine-tuning:** as in Phase 1, about 3–4 h per iSign run.
- **Stage 3.1 total:** about 4 pretraining runs + about 20 fine-tunes, roughly 100–120 GPU-hours, or 1.5–2 days on 4 GPUs.

---

### 3.1.6 First gate result (iSign, Mamba, seed 42; 5 epochs of iSign-only pretraining, about 8 min)
| | Test chrF2 | BLEU-4 | Content-word F1 | vs P0 |
|---|---|---|---|---|
| P0 no pretraining | 21.26 | 4.10 | 14.97 | — |
| P1 clip pretraining | 21.74 | 4.19 | 15.18 | chrF2 p = 0.001, BLEU-4 n.s. |
| P2 long-context pretraining | 21.69 | **4.32** | **15.50** | chrF2 p = 0.001, BLEU-4 p = 0.014 |
| P2-hand | 21.70 | 4.23 | 15.42 | chrF2 p = 0.001, BLEU-4 n.s. |

Pretraining helps modestly. P2 vs P1 is not significant (chrF2 p = 0.25, BLEU-4 p = 0.10). The run was far below the planned
budget (loss still falling, no data beyond the fine-tuning set), so it cannot separate "long context doesn't help" from
"pretraining was too small".

### 3.1.7 Track A: pretraining at a realistic scale (`scripts/run_pretrain_large.sh`)
- **Pretraining:** 50 epochs on the pooled train poses of iSign + How2Sign + PHOENIX (no text, so How2Sign's target-text
  issue doesn't matter). About 1.5–2 h per run. One pretraining per seed.
- **PART=gate:** P2L vs P1L (Mamba), seed 42. **Decide here:** P2L > P1L clearly → long-context headline;
  P1L ≈ P2L > P0 → "pretraining helps"; both ≈ P0 → drop.
- **PART=seeds:** P2L/P1L seeds 13/1337; Transformer controls T2L (long, or 2,048 frames if 4,096 runs out of memory) and
  T1L, 3 seeds. Baselines: P0 (phase 2 mamba_padfix) and T0 (phase-1 Transformer), both 3 seeds.
- **PART=ctx:** the 2-previous-clips context model fine-tuned from each P2L encoder, vs phase 2's mamba_ctx_k2 without
  pretraining (3 seeds). Tests whether story-level pretraining makes context usable.
- Length penalty fixed at 1.0 (validation tuning gave +0.01 BLEU-4 on average; choosing by chrF2 only lengthens outputs).

## 4. Stage 3.2: one encoder, three sign languages (secondary ablation; partly prior work, §11)
| ID | Pretraining data | Fine-tuned on |
|---|---|---|
| M0 | own language only (= P2 per dataset) | iSign, How2Sign, PHOENIX separately |
| M1 | iSign + How2Sign + PHOENIX together, with a learned language embedding | each dataset separately |
| M2 | iSign only | How2Sign and PHOENIX (cross-lingual transfer) |

**The question:** do sign languages share low-level motion structure that transfers? The prediction is that the gain shows up mainly on PHOENIX and How2Sign. The analysis is per-dataset gains, plus whether the hand-focused objective transfers better than the full-body one.

---

## 5. Stage 3.3: Mamba + Δ compression as an LLM front end (fallback or parallel)
- **Architecture:** pretrained (or Phase 2) Mamba encoder, Δ-weighted averaging (about 13 tokens per clip), a 2-layer MLP projector, then an LLM. The lab already runs Qwen/Gemma/Phi translation (`qwen-3-8bVT.py`), so the comparison point exists.
- **LLM training:** frozen LLM first, then LoRA. Prompt: "Translate this sign language: <tokens>".
- **Controls:** the same LLM fed plain-averaged tokens; the same LLM fed all frames at stride 4 (the lab's existing input density). Compare against SignLLM-style discrete token compression, the closest prior work (§11).
- **Measured:** chrF2 and BLEU-4, **plus the content metrics.** The known risk is that an LLM hallucinates *more fluently*, and content-word F1 and the no-overlap rate catch that.
- **Why it's interesting:** compression stops being optional. LLM prompt cost scales with tokens, so 13 tokens per clip is what makes story-level LLM translation affordable.

---

## 6. Stage 3.4: lexical grounding loss (cheap add-on, any encoder)
- **Method:** an auxiliary multi-label head on the encoder's pooled output predicts the bag of content words in the target sentence (stop-words removed, vocabulary capped at about 3k), using binary cross-entropy weighted 0.1–0.5.
- **What it targets:** the "fluent but wrong" failure directly, by forcing word identity into the encoder.
- **Tested:** on P0, and on the best Stage 3.1 model.
- **Measured:** the content metrics first, chrF2 second.

---

## 7. Order of work
| Step | Work | Training needed | Gate / output |
|---|---|---|---|
| 1 | Stage 3.0 (MBR, selective translation) + Phase 2 fixes | none / reruns | corrected baselines; first Phase 3 results within days |
| 2 | Build the stitched-video loader + masked-frame pretraining; pretrain P1 and P2 (seed 42) | yes | **go/no-go gate** (§3.1.4) |
| 3 | Full Stage 3.1 grid, 3 seeds; Transformer controls; cost curves | yes | headline table |
| 4 | Stage 3.2 (multilingual, secondary) | yes | transfer ablation |
| 5 | Stage 3.3 (LLM front end) if the gate fails, or as an extension | yes | fallback headline |
| 6 | Stage 3.4 on the best model | yes | add-on |

## 8. Paper this leads to
**If H1 and H2 hold:**

> *"Long-context self-supervised state-space models for sign language translation."*

1. Pretraining on untrimmed sign video with an SSM encoder, which attention can't afford.
2. It fixes content recognition (the bottleneck the error analysis identified).
3. A secondary ablation: whether it transfers across sign languages.
4. The Phase 1 benchmark and Phase 2 compression/quantization as supporting results.

**If only H1 holds:** pretraining as the method, plus the honest finding that long context did not add.

**Either way, Stage 3.0** adds consensus decoding and selective translation.

## 9. Code layout (implemented 2 October 2026)
`phase 3/` holds only new code and **imports phase 2's `src/`** (models, data, evaluation) through `p3/paths.py`, so nothing is copied twice. Phase 2's `src/train.py` gained one backward-compatible option, `--init-encoder`.

| File | What it does |
|---|---|
| `p3/nbest.py`, `scripts/nbest_decode.py` | beam-5 n-best lists with raw scores (val + test), re-rankable at any length penalty; identical to the existing beam search (tested) |
| `p3/chrf.py`, `p3/consensus.py`, `scripts/mbr.py` | consensus (MBR) decoding with a fast cached chrF utility; singles, MBR within one seed, over seeds' top-1, over the pooled n-best, oracle; paired bootstrap |
| `scripts/selective.py` | 6 confidence signals + a combined one, directions fixed on validation; Spearman, failure AUROC, coverage–quality curves vs oracle and random |
| `p3/stitch.py` | clips grouped into source videos by uid (iSign, How2Sign and PHOENIX formats), stitched in segment order; `clip` and `long` (4,096-frame windows, random phase) modes over exactly the same frames |
| `p3/masking.py` | span masking (30% of frames, spans of 8–32), a share of spans hiding only the hand keypoints |
| `p3/pretrain_model.py`, `scripts/pretrain.py` | masked-frame pretraining of the translation model's own front end + encoder; saves `encoder_init.pt` for `--init-encoder`; `--stats-only` prints the stitching statistics |
| `scripts/run_stage30.sh` | `PART=decode` (GPU queue of n-best jobs) then `PART=analyze` (CPU: MBR + selective per dataset × encoder) |
| `scripts/run_pretrain_gate.sh` | the go/no-go gate: P2, P1, P2-hand, each as pretrain → fine-tune → test decode → val-chosen length penalty |
| `scripts/test_stage30.py`, `scripts/test_pretrain.py` | CPU tests (stand-ins for mamba_ssm and SentencePiece) |

## 10. Risks
| Risk | Mitigation |
|---|---|
| Pretraining helps chrF2 only through fluency, not content | content metrics are primary; the frozen-encoder fine-tune tests representation quality directly |
| Gaps between stitched segments make "long context" noisy | boundary flags; compare against clip-level pretraining on the *same* frames (P1 vs P2) |
| The Transformer long-context control doesn't fit in memory | report that as the finding (with measured memory), plus a local-window attention variant |
| Little unlabelled data beyond the train splits | the data-scaling curve (P2-scale) shows whether more data would help; the multilingual pool adds about 38k clips |
| The prior-art check finds the headline already done | §11 found none for Stage 3.1; repeat on Google Scholar before submission |

## 11. Novelty check (literature check, 2 October 2026)
Only papers actually opened are cited. Web search isn't exhaustive, so do one Google Scholar pass before submission.

| Stage | Closest prior work | Verdict |
|---|---|---|
| **3.1 Long-context SSM pretraining** | Every self-supervised sign encoder found pretrains **Transformers on clips**: [SSVP-SLT](https://arxiv.org/html/2402.09611v2) (≤ 128 frames), [SignDino](https://arxiv.org/html/2609.06296) (64–96), [MSLU](https://arxiv.org/html/2408.08544) (≤ 256), [SHuBERT](https://arxiv.org/html/2411.16765) (≤ 1,500 frames per batch, input unit unclear). SSMs in sign language appear only in recognition ([PhonSSM](https://arxiv.org/abs/2604.08761)) or as an aggregator ([EvSLT](https://arxiv.org/abs/2408.10488)). Context-aware SLT ([Jang et al., CVPR 2025](https://arxiv.org/abs/2501.09754)) is supervised and passes context as text. | **No prior work found.** Word the claim as *explicit story-level (5k+ frame) context*. It still has to be shown to help sentence-level SLT. |
| 3.2 Multilingual pretraining | Supervised multilingual pose SLT already covers ISL/ASL/DGS ([Scaling SLT, NeurIPS 2024](https://arxiv.org/html/2407.11855); [WSLP 2025](https://aclanthology.org/2025.wslp-main.5/)). Self-supervised multilingual masked pose modelling exists in MSLU (ASL/BSL/CSL/DGS, no ISL). | **Partly done.** Demoted to a secondary ablation (§4). |
| 3.3 Mamba + Δ compression → LLM | LLM decoders for SLT are established ([SignLLM, CVPR 2024](https://arxiv.org/abs/2404.00925), [SpaMo](https://arxiv.org/abs/2408.10593), [Uni-Sign](https://arxiv.org/abs/2501.15187), [SignGPT](https://arxiv.org/abs/2609.21709)), including pose input. SignLLM's discrete token compression is the closest precedent. | **New only in combination:** no SSM front end, and no about-13-continuous-token Δ compression, has been found. It must be compared with SignLLM, framed as an efficiency/accuracy trade-off. |
| 3.0a Consensus (MBR) decoding | No SLT paper uses MBR (a search summary's claim about DiffSLT is false; checked in the PDF). The nearest analogue is MBR for ASR and speech translation ([Jinnai 2025](https://arxiv.org/abs/2510.19471)). | **Unexplored in SLT.** A low-novelty method, but a legitimate empirical contribution. |
| 3.0b Selective translation | Hallucination and reliability scoring for SLT exists ([Grounding or Guessing?, ICLR 2026](https://arxiv.org/abs/2510.18439), AUROC on PHOENIX/CSL-Daily). | **Reliability prediction is not new and must be cited.** What's new is abstention with **coverage–quality curves**, and **seed disagreement / MBR consensus** as the confidence signal. |

Also cite: [PoseStitch-SLT (EMNLP 2025)](https://arxiv.org/abs/2511.00270), which pretrains on synthetic stitched templates for iSign and How2Sign. Its "stitching" is a different idea (synthetic sentences, not real untrimmed stories), but reviewers will ask, so position against it explicitly.
