# Experiment queue

Priority-ordered checklist for the remaining experiments from the expert review (`RESEARCH_PROGRAM_2026-10-08.md`) and the diagnostics. Server jobs run from one shared queue, `phase 3/scripts/run_queue.sh`, so higher-priority jobs always start first. Finished runs are skipped, so the script is safe to re-run.

## Where we stand (iSign, test, beam 5)

| Encoder | chrF2 | BLEU-4 | Seeds |
|---|---|---|---|
| Transformer | 18.94 | 2.97 | 1 (seed 42) |
| + convolutional stem | 19.65 | 3.07 | 1 |
| + relative positions | 19.74 | 3.26 | 1 |
| **TCN (convolutions only)** | **21.10** | **3.94** | 1 |
| Mamba (padding fixed) | 21.26 | 4.10 | 1 |

- **Diagnostics:** the Transformer barely uses frame order. Shuffling every frame costs it −0.35 chrF2, against −4.21 for Mamba.
- **E3, done** (video-grouped bootstrap, 2,000 draws):
  - The Mamba − Transformer gap grows with clip length, from +1.77 (≤128 frames) to +2.60 chrF2 (>256 frames).
  - The long-minus-short difference is +0.83 [+0.36, +1.23] over 3 seeds.
  - The growth holds at fixed sentence length for medium and long references.
- **The TCN − Transformer gap is flat across lengths** (difference −0.15).
- **So the length-dependent part of Mamba's advantage is exactly what the TCN lacks.** Mamba − TCN by length: −0.23 / +0.13 / +0.44, with a long-minus-short difference of +0.67 [0.00, +1.40] (seed 42 only). It is concentrated in long clips with long sentences: +0.73 [+0.23, +1.24].

**Hypothesis under test:**
1. Local temporal modelling explains most of the gap.
2. Selective long-range memory explains the length-dependent rest.

## Checklist

### P0: before anything (server, about 10 min)
- [ ] `git pull --ff-only`, then run the CPU tests:
  - `phase 2/scripts/test_controls.py`
  - `phase 2/scripts/test_selectivity.py`
  - `phase 3/scripts/test_diagnostics.py`
- [ ] GPU smoke test of the new arms. It runs 20 steps each and is the first time `mamba_nonselective` and `mamba_window` run on the real Mamba2 kernels:
  `SMOKE=1 PARTS="select phoenix" GPUS="0" bash "phase 3/scripts/run_queue.sh"`

### P1 `confirm`: does TCN ≈ Mamba hold? (iSign, 6 runs)
- [ ] TCN, seeds 13 and 1337.
- [ ] Conv stem and relative positions, seeds 13 and 1337.
- **Decision:** if the TCN stays within about 0.3 chrF2 of Mamba on the 3-seed mean, "local temporal modelling explains most of the gap" is confirmed on iSign.

### P2 `select`: what is Mamba's long-clip lead? (iSign, seed 42, 4 runs, each with order diagnostics on short and long clips)
| Arm | Change | Prediction if selective long-range memory is the cause |
|---|---|---|
| [ ] `mamba_window` | Mamba run on separate 128-frame windows (5.1 s) | Loses the long-clip lead over the TCN |
| [ ] `mamba_nonselective` | B, C, Δ fixed (learned constants, not computed from the input) | Loses the long-clip lead |
| [ ] `tcn_wide` | TCN reaching the whole clip (513 frames) | Does **not** catch up on long clips |
| [ ] `transformer_local` | Attention limited to ±10 frames per layer (121-frame reach, like the TCN), no convolution | Separates "local" from "convolutional" |
- **Analysis:** `length_gap.py` for each arm against the TCN and Mamba (interaction on >256 vs ≤128 frames).
- **Decision:** if `mamba_window` and `mamba_nonselective` both lose the long-clip lead and `tcn_wide` doesn't gain it, run P7.

### P3 `phoenix`: second dataset (base_f3 schedule)
- [ ] TCN, seeds 42, 13 and 1337. Seed 42 also gets order diagnostics.
- [ ] Order diagnostics, 2,000 clips, for the existing Transformer (`results_f3`) and Mamba (padding fixed).
- **Compare against** Transformer 31.87 / 10.64 and Mamba 33.58 / 11.77 (3-seed means).

### P4 `profile`: resource table (one job, minutes)
- [ ] For all 9 encoders: parameters, training step time and peak memory at the token budget, and batch-1 encoder latency at 128, 256 and 512 frames. Output: `phase 3/results/profile/profile.csv`.

### P5 How2Sign target validity (manual; needed before any How2Sign claim)
- [ ] `python scripts/check_text_columns.py --data configs/data/how2sign.yaml`. Paste the output so the correct target column can be confirmed.
- [ ] If confirmed: `H2S_TEXT_CHECKED=1 PART=how2sign bash scripts/run_fixes.sh` (retrains **both** encoders on the cased targets).

### P6 `bow`: does a word-level training signal change which encoder wins? (E6, PHOENIX, seed 42)
- [ ] Transformer (phase-2 harness baseline cell; overwrites the stale Phase 1 copy at that path).
- [ ] Transformer + bag-of-words loss.
- [ ] Mamba + bag-of-words loss. The plain Mamba cell already exists.
- **Statistic:** (Mamba_bow − Mamba) − (Transformer_bow − Transformer), not the two gains separately.

### P7 `select3`: conditional, only if P2 supports the hypothesis
- [ ] `mamba_window` and `mamba_nonselective`, seeds 13 and 1337 on iSign: `PARTS="select3"`. To add PHOENIX, ask me and I'll extend the part.

### My side (CPU, after each push)
- [x] E3 length-gap analysis with video-grouped bootstrap: `phase 3/results/length_gap/`.
- [ ] Re-run `length_gap.py` with 3 seeds when P1 lands, and for each P2 arm.
- [ ] Re-run the video-grouped bootstrap for every headline comparison in REPORT.md (the review flags clip-level bootstrap as overconfident).
- [ ] Draft paper sections: setup, benchmark, diagnostics, controls ladder.

## Not queued, and why
| Review item | Reason |
|---|---|
| E4: frozen lexical discrimination and reranking | Its trigger (weak video dependence) didn't happen; E1 showed strong dependence |
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
cd "phase 2"
python scripts/test_controls.py
python scripts/test_selectivity.py
cd ..
```

```bash
SMOKE=1 PARTS="select phoenix" GPUS="0" bash "phase 3/scripts/run_queue.sh"
```

```bash
tmux new -s queue
PARTS="confirm select phoenix profile bow" GPUS="0 1 2 3" bash "phase 3/scripts/run_queue.sh"
```

Progress: `tail -f logs/queue.log`. Pending jobs: `cut -f1 logs/queue_jobs.txt`.

To push results (as many times as you like while the queue runs):

```bash
git add "phase 2/results" "phase 3/results"
git commit -m "Queue results"
git push
```

**Expected time:** 19 jobs on 4 GPUs is about 1.5–2 days. An iSign run takes about 4–6 h; PHOENIX runs are shorter.
