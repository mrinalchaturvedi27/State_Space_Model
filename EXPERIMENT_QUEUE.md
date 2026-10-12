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
- [ ] **Step 1 (you, minutes):** run `check_text_columns.py` and paste the output. I'll confirm the column, or change `text_col` in both `configs/data/how2sign_cased.yaml` and `phase 2/configs/data/how2sign_cased.yaml`.
- [ ] **Step 2:** `H2S_TEXT_CHECKED=1 PARTS="how2sign"`. It builds the cache once (CPU, about 40 min), then trains Transformer, TCN, Mamba and non-selective Mamba, 3 seeds each, with the 80-epoch budget. Seed 42 also gets order diagnostics on short and long clips.
- **What it decides:** How2Sign has the longest clips, so it's the strongest test of point 2 (context beyond ~5 s) and a third-language check of points 1 and 3.

### My side (CPU, after each push)
- [x] `length_gap.py` for the P7 arms (3 seeds); `compare_encoders.py` headline tables.
- [ ] Video-grouped bootstrap for every headline comparison in REPORT.md.
- [ ] Draft the paper: "Temporal inductive bias, not selectivity: what makes state-space encoders win in pose-based SLT".

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

Step 1, the target check (no GPU, minutes). Paste me the whole output:

```bash
cd /scratch/home/student1/Sign_Language/Projects/State_Space_Model
git pull --ff-only
conda activate ssm-slt
python scripts/check_text_columns.py --data configs/data/how2sign.yaml --cache-dir cache
```

Step 2, only after I've confirmed the column:

```bash
rm -f logs/queue.log logs/queue_jobs.txt
tmux new -s queue
H2S_TEXT_CHECKED=1 PARTS="how2sign" GPUS="0 1 2 3" bash "phase 3/scripts/run_queue.sh"
```

Progress: `tail -f logs/queue.log`. Pending jobs: `cut -f1 logs/queue_jobs.txt`.

To push results (as many times as you like while the queue runs):

```bash
git add "phase 2/results_long" "phase 3/results"
git commit -m "How2Sign results"
git push
```

**Expected time:** about 40 min for the cache, then 12 runs on 4 GPUs, roughly 1–1.5 days.
