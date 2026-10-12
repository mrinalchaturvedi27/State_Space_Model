# Experiment queue

Priority-ordered checklist for the remaining experiments from the expert review (`RESEARCH_PROGRAM_2026-10-08.md`) and the diagnostics. Server jobs run from one shared queue, `phase 3/scripts/run_queue.sh`, so higher-priority jobs always start first. Finished runs are skipped, so the script is safe to re-run.

## Where we stand

### iSign (test, beam 5, 3 seeds; video-grouped 95% intervals)
| Encoder | chrF2 | BLEU-4 | vs Mamba, chrF2 | vs Mamba, BLEU-4 | vs Mamba, >256-frame clips |
|---|---|---|---|---|---|
| Transformer | 19.06 | 2.90 | −2.30 [−2.55, −2.05] | −1.10 | −2.60 |
| + conv stem | 19.70 | 3.15 | −1.66 | −0.85 | −1.93 |
| + relative positions | 19.78 | 3.37 | −1.58 | −0.64 | −1.82 |
| TCN | 20.95 | 3.92 | −0.41 [−0.57, −0.27] | −0.08 (n.s.) | −0.68 |
| Mamba in 5-second windows | 21.19 | 3.79 | −0.17 [−0.34, −0.02] | −0.22 | **−0.58** [−0.83, −0.36] |
| **Mamba, selectivity removed** | **21.79** | **4.31** | **+0.43 [+0.28, +0.57]** | **+0.31 [+0.17, +0.45]** | +0.46 |
| Mamba | 21.36 | 4.00 | — | — | — |

Single-seed arms: local-attention Transformer 20.00 / 3.54; whole-clip TCN 20.93 / 4.13 (closes the TCN's long-clip deficit, −0.08).

### PHOENIX (80-epoch budget, every run converged and early-stopped; 3 seeds)
| Encoder | chrF2 | BLEU-4 | vs Mamba, chrF2 | vs Mamba, BLEU-4 | Best epoch |
|---|---|---|---|---|---|
| Transformer | 32.25 | 10.76 | −2.19 [−2.75, −1.56] | −0.35 (n.s.) | 30–42 |
| TCN | 33.63 | 11.70 | −0.81 [−1.51, −0.15] | +0.59 [+0.01, +1.05] | 39–49 |
| **Mamba, selectivity removed** | 34.23 | **12.10** | −0.20 (n.s.) | **+0.99 [+0.37, +1.54]** | 24–47 |
| Mamba | 34.44 | 11.11 | — | — | 22–26 |

PHOENIX clips are all ≤256 frames, so the long-clip question can't be asked there.

### What the results say
1. **Local temporal modelling explains most of the gap.** A TCN closes 82% (iSign) and 63% (PHOENIX) of the chrF2 gap, and all of the BLEU-4 gap.
2. **Long clips also need context beyond about 5 seconds.** Windowed Mamba loses specifically on long iSign clips (long-minus-short −0.92 [−1.32, −0.54]), and a TCN reaching the whole clip recovers the TCN's long-clip deficit.
3. **Input-dependent selectivity isn't needed; removing it helps.** A time-invariant (non-selective) Mamba is the best encoder on iSign (+0.43 chrF2, +0.31 BLEU-4, the same at every length). On PHOENIX it ties on chrF2 and is better on BLEU-4.
4. **Mamba converges fastest** (best epochs 22–26 on PHOENIX, against 30–49 for the others).

A time-invariant SSM is equivalent to a long learned convolution. **The winning inductive bias is dense temporal filtering over the whole clip, not content-dependent selection.**

**Other results:**
- The word-level (bag-of-words) loss hurts both encoders; dropped.
- Resources: `phase 3/results/profile/profile.csv`.
- Headline tables: `phase 3/results/compare/`.

## Checklist

### Done
- [x] P0 tests and GPU smoke test.
- [x] P1 `confirm`: TCN, conv stem and relative positions, 3 seeds.
- [x] P2 `select`: windowed Mamba, non-selective Mamba, whole-clip TCN, local-attention Transformer.
- [x] P3 `phoenix`: TCN, 3 seeds, plus order diagnostics. Inconclusive because of the epoch cap; P8 reruns it.
- [x] P4 `profile`.
- [x] P6 `bow`: dropped.
- [x] E3 length-gap analysis with video-grouped bootstrap: `phase 3/results/length_gap/`.

### Done since
- [x] P7 `select3` (3 seeds): selectivity removed is better; windowing hurts long clips.
- [x] P8 `phoenix_long`: converged comparison on PHOENIX.

### Now
#### P9 `how2sign`: the third dataset, with corrected targets (12 runs)
The original How2Sign runs trained on lowercased text with `<UNKNOWN>` tokens but were scored against cased references. `how2sign_cased` uses one consistently formatted column for train, val and test.
- [x] **Step 1:** `check_text_columns.py` (2026-10-12): train `text` is lowercase with 22.8% `<UNKNOWN>`, train `SENTENCE` is cased; val/test have only `text`, already cased. The configs now use `train_text_col: SENTENCE` and `text_col: text`, and the runner refuses to train unless the built cache passes the same check for all three splits.
- [ ] **Step 2:** `H2S_TEXT_CHECKED=1 PARTS="how2sign"`. It builds the cache once (CPU, about 40 min), then trains Transformer, TCN, Mamba and non-selective Mamba, 3 seeds each, with the 80-epoch budget. Seed 42 also gets order diagnostics on short and long clips.
- **What it decides:** How2Sign has the longest clips, so it's the strongest test of point 2 (context beyond ~5 s) and a third-language check of points 1 and 3.

#### After the 12 October review (`REVIEW_FEEDBACK_2026-10-12.md`)
Fixed on my side, with every interval recomputed:
- [x] Source-video grouping: iSign dictionary clips (`_d`, `_e1`–`_e4`) are now grouped by video; 216 test videos, not 260.
- [x] Content F1 is now clipped-multiset; number F1 and negation are scored separately.

Queued behind How2Sign, appended to the running queue:
- [ ] **P10 `windows`, boundary vs context.**
  - Window diagnostic (inference) on the window-trained checkpoints (3 seeds), fixed Mamba and input-independent Mamba.
  - Train windows of 64 and 256, and 128 with overlap (seed 42).
- [ ] **P11 `s4d`:** independent LTI baseline, 3 seeds.
- [ ] **P12 `tcn_wide3`:** wide-receptive-field TCN, seeds 13 and 1337.

CPU checks, run now on the server:
- [ ] E1 partner re-check (`check_video_partners.py`).
- [ ] Learned retention times (`inspect_retention.py`).

### My side (CPU, after each push)
- [x] `length_gap.py` and `compare_encoders.py` with corrected grouping and metrics.
- [x] `REVIEW_RESPONSE.md` revised (claim scoped, labels fixed, parameter bookkeeping, §0 changes).
- [ ] Analyse P9–P12; update the response.
- [ ] Draft the paper.

## Not queued, and why
| Review item | Reason |
|---|---|
| E4: frozen lexical discrimination and reranking | Its trigger (weak video dependence) didn't happen; E1 showed strong dependence |
| E6: lexical loss | Done; hurts both encoders at weight 0.3 |
| E7: selective translation vs published confidence, plus human ratings | A separate paper; needs annotators |
| E8: pooling, E9: RGB / hand crops, E10: context audit | Low priority for the current claim |
| E11–E13: more pretraining, time-aware SSM, streaming | The review says to stop these |
| Human evaluation (150–200 clips) | Needs ISL signers; plan it separately if available |

## Commands

1. Update the code. The How2Sign queue keeps running; the running jobs already loaded their code.

```bash
cd /scratch/home/student1/Sign_Language/Projects/State_Space_Model
git pull --ff-only
conda activate ssm-slt
```

2. Tests (CPU):

```bash
cd "phase 2" && python scripts/test_selectivity.py && cd ..
cd "phase 3" && python scripts/test_diagnostics.py && cd ..
```

3. Append the review experiments to the running queue. This only works once the How2Sign queue has listed its jobs (after its ~40-minute cache build):

```bash
APPEND=1 PARTS="windows s4d tcn_wide3" bash "phase 3/scripts/run_queue.sh"
```

4. CPU checks (seconds to minutes; they don't need a GPU):

```bash
cd "phase 3"
python scripts/check_video_partners.py --cache-dir ../cache --dataset isign --video-dir results/diagnostics/screen/video
for s in 42 13 1337; do python scripts/inspect_retention.py --checkpoint "../phase 2/results/isign/mamba_nonselective/lr0.0003_s$s/checkpoints/best.pt" --out results/retention/isign_mamba_nonselective_s$s.csv; done
python scripts/inspect_retention.py --checkpoint "../phase 2/results/isign/mamba_padfix/lr0.0003_s42/checkpoints/best.pt" --out results/retention/isign_mamba_padfix_s42.csv
cd ..
```

5. Progress: `tail -f logs/queue.log`. Pending jobs: `cut -f1 logs/queue_jobs.txt`.

6. Push results (partway or at the end):

```bash
git add "phase 2/results" "phase 2/results_long" "phase 3/results"
git commit -m "Queue results"
git push
```
