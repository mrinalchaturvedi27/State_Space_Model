# Experiment queue

Priority-ordered checklist for the remaining experiments from the expert review (`RESEARCH_PROGRAM_2026-10-08.md`) and the diagnostics. Server jobs run from one shared queue, `phase 3/scripts/run_queue.sh`, so higher-priority jobs always start first. Finished runs are skipped, so the script is safe to re-run.

## Where we stand

### iSign (test, beam 5; video-grouped 95% intervals)
| Encoder | Seeds | chrF2 | BLEU-4 | vs Mamba, chrF2 | vs Mamba, >256-frame clips |
|---|---|---|---|---|---|
| Transformer | 3 | 19.06 | 2.90 | −2.30 [−2.55, −2.05] | −2.60 |
| + conv stem | 3 | 19.70 | 3.15 | −1.66 | −1.93 |
| + relative positions | 3 | 19.78 | 3.37 | −1.58 | −1.82 |
| Local-attention Transformer (121 frames) | 1 | 20.00 | 3.54 | −1.26 | −1.16 |
| **TCN** | 3 | **20.95** | **3.92** | **−0.41 [−0.57, −0.27]** | −0.68 |
| TCN reaching the whole clip | 1 | 20.93 | 4.13 | −0.33 | −0.08 |
| **Mamba in 5-second windows** | 1 | 21.27 | 3.80 | **+0.01** [−0.21, +0.24] | −0.09 |
| **Mamba, selectivity removed** | 1 | **21.66** | 4.08 | **+0.40** [+0.18, +0.62] | **+0.88** |
| Mamba | 3 | 21.36 | 4.00 | — | — |

**What the results say:**
1. **A TCN closes 82% of the chrF2 gap and 93% of the BLEU gap** (3 seeds). Its remaining deficit is on long clips: the long-minus-short difference is +0.83 [+0.44, +1.26].
2. **Mamba doesn't need memory beyond about 5 seconds.** The windowed version loses nothing.
3. **Mamba doesn't need input-dependent selectivity.** The fixed (time-invariant) version is slightly better (1 seed).
4. **The likely mechanism:** dense temporal recurrence over a few seconds of motion. The Transformer barely uses frame order; shuffling every frame costs it −0.35 chrF2, against −4.21 for Mamba.

The earlier hypothesis, "selective long-range memory explains the length-dependent part", is **rejected** by points 2 and 3.

### PHOENIX: not conclusive yet
- **3-seed scores:** Transformer 31.87 / 10.64, TCN 32.46 / 10.77, Mamba 33.94 / 11.68.
- **The problem:** the Transformer and TCN were still improving at the 40-epoch cap (best epochs 36–39), while Mamba converged by about epoch 30. P8 reruns PHOENIX with an 80-epoch budget.

### Other results
- **Word-level (bag-of-words) loss, E6:** hurts both encoders (Transformer −4.15, Mamba −1.21 chrF2). The Transformer + loss run was also still improving at the cap, so the interaction isn't interpretable. **Dropped.**
- **Resources** (`phase 3/results/profile/profile.csv`):

| | Training step | Peak memory | Batch-1 encoder time |
|---|---|---|---|
| Transformer | 66 ms | 4.4 GB | 3–5 ms |
| TCN | 91 ms | 4.8 GB | 3.5 ms |
| Mamba | 140 ms | 3.9 GB | 15 ms |

## Checklist

### Done
- [x] P0 tests and GPU smoke test.
- [x] P1 `confirm`: TCN, conv stem and relative positions, 3 seeds.
- [x] P2 `select`: windowed Mamba, non-selective Mamba, whole-clip TCN, local-attention Transformer.
- [x] P3 `phoenix`: TCN, 3 seeds, plus order diagnostics. Inconclusive because of the epoch cap; P8 reruns it.
- [x] P4 `profile`.
- [x] P6 `bow`: dropped.
- [x] E3 length-gap analysis with video-grouped bootstrap: `phase 3/results/length_gap/`.

### Now
#### P7 `select3`: are "no selectivity needed" and "no long memory needed" real? (iSign, 4 runs)
- [ ] `mamba_nonselective` and `mamba_window`, seeds 13 and 1337.
- **Decision:** if non-selective Mamba ≥ Mamba on the 3-seed mean, "selectivity isn't needed" becomes the headline.

#### P8 `phoenix_long`: PHOENIX with an 80-epoch budget (12 short runs)
- [ ] Transformer, TCN, Mamba and non-selective Mamba, seeds 42, 13 and 1337.
- **Settings:** `configs/train/base_f3_long.yaml`, same early stopping. Results go to `phase 2/results_long/`, so the 40-epoch runs stay intact.
- **Decision:** does TCN ≈ Mamba and non-selective ≥ Mamba also hold on a second language once every encoder has converged?

#### P5 How2Sign target validity (manual; needed before any How2Sign claim)
- [ ] `python scripts/check_text_columns.py --data configs/data/how2sign.yaml`. Paste the output so the correct target column can be confirmed.
- [ ] If confirmed: `H2S_TEXT_CHECKED=1 PART=how2sign bash scripts/run_fixes.sh` (retrains **both** encoders on the cased targets).

### My side (CPU, after each push)
- [ ] `length_gap.py` for the P7 arms (3 seeds) and the PHOENIX reruns.
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
