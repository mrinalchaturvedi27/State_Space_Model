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
#### P5 How2Sign target validity (manual; needed before any How2Sign claim)
- [ ] `python scripts/check_text_columns.py --data configs/data/how2sign.yaml`. Paste the output so the correct target column can be confirmed.
- [ ] If confirmed: retrain Transformer, TCN, Mamba and non-selective Mamba on the corrected targets. This is the third dataset, and the longest clips.

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

```bash
cd /scratch/home/student1/Sign_Language/Projects/State_Space_Model
git pull --ff-only
conda activate ssm-slt
```

```bash
rm -f logs/queue.log logs/queue_jobs.txt
```

```bash
tmux new -s queue
PARTS="select3 phoenix_long" GPUS="0 1 2 3" bash "phase 3/scripts/run_queue.sh"
```

Progress: `tail -f logs/queue.log`. Pending jobs: `cut -f1 logs/queue_jobs.txt`.

To push results (as many times as you like while the queue runs):

```bash
git add "phase 2/results" "phase 2/results_long" "phase 3/results"
git commit -m "Queue results"
git push
```

**Expected time:** 16 jobs on 4 GPUs. The 4 iSign runs take about 4–6 h each. The 12 PHOENIX runs are shorter, but up to twice as long as before because of the 80-epoch budget. About 1 day in total.
