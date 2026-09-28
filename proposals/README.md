# Proposals — candidate architectures and the evidence for each

Candidate Phase 2 architectures, each tied to a premise that can be tested before any expensive
training. The Phase 2 plan itself (C1 discourse memory, C2 Δ vs kinematics) is in `../PHASE2_PLAN.md`;
this directory only holds the new candidates and their evidence tests.

```
proposals/
  README.md                      this file: proposals, premises, evidence, verdicts
  evidence/                      local tests, run on the pushed prediction CSVs / metrics.xlsx
    common.py                    loaders + metrics matching slt_reporting.compute_metrics
    e1_seed_specialisation.py    R1: do seeds act like signer experts?
    e2_gate_curves.py            A: is Δ placement better than uniform, above seed noise?
    e3_error_profile.py          A: are errors on the language side or on recognising signs?
    server/e4_articulators.py    B: hand/face asynchrony + per-head Δ specialisation (needs GPU + cache)
  results/                       reports written by the tests (e1–e3 now, e4 once run on the server)
```

Local run (needs numpy, sacrebleu, openpyxl): `cd proposals/evidence && python e1_seed_specialisation.py`
(same for e2, e3). E2 picks up round-2 runs automatically once they are pushed.

## Status (2026-09-28)

| id | proposal | premise tested | evidence | verdict |
|---|---|---|---|---|
| R1 | MoE with experts as "signers" (from the ensemble gain) | seeds specialise by video | E1 | **rejected** |
| A | Δ-segmented Mamba encoder → pretrained LLM decoder | (a) Δ places memory tokens better than uniform | E2 | **supported, 1 seed**; confirm with round 2 |
| | | (b) the decoder / language side is the bottleneck | E3 | **not supported** — reframe A |
| B | articulator-factored scans (hands / face / body) | channels are asynchronous; shared Δ under-uses the face | E4 | **pending** (server) |

## R1 — MoE as separate signers: rejected

Test (E1, iSign, 3 seeds, lr 3e-4): for each sentence, which seed scores best (sentence chrF2), and do
those winners cluster by video (a signer proxy) more than a permutation null?

| | Mamba test | Mamba val | TF test | TF val |
|---|---|---|---|---|
| winner concentration, observed vs null | 0.3660 vs 0.3652 | 0.3539 vs 0.3552 | 0.3669 vs 0.3643 | 0.3584 vs 0.3557 |
| permutation p | 0.30 | 0.80 | 0.047 | 0.050 |
| r of sentence chrF2 between seeds | 0.66–0.67 | 0.65–0.67 | 0.62–0.64 | 0.56–0.63 |

- The winning seed does not depend on the video for Mamba. The TF effect is borderline and tiny
  (+0.003 on a 0.33–1 scale). Seeds are not signer experts; their errors are just partly independent.
- Mamba test BLEU-4: seed mean 3.73, random pick per sentence 3.77, **ensemble 4.91**, oracle best-of-3
  5.61. The logit-averaging ensemble already gets 63% of the way to an oracle that peeks at the
  reference, so a learned router over seeds has at most ~0.7 BLEU of headroom, and only with a perfect router.
- Side finding: Mamba s13 wins 41.7% of test sentences but only 33.9% of val, and its test chrF2
  (22.27) is 1.6 above the other seeds while on val it is level. That matches F5 in PHASE2_PLAN.md
  (s13 `test_best_beam5.csv` decoded with different settings). Regenerate it before the paper table.

## A — Δ-segmented encoder feeding an LLM decoder

**(a) Δ placement (E2).** Seed noise, phase-1 iSign best val chrF2: Mamba sd **0.14**, TF sd 0.19
(much tighter than the ~0.5 assumed in PHASE2_PLAN.md).

| run (s42) | epochs | best val chrF2 (epoch) | slope, last 5 epochs | kept_frac |
|---|---|---|---|---|
| mamba_padfix | 29 | 21.06 (20) | +0.011 | 1.00 |
| mamba_banks | 32 | 21.06 (23) | +0.061 | 1.00 |
| mamba_pool_matched | 40 | 20.36 (34) | −0.008 | 0.07 |
| mamba_uniform | 28 | 19.63 (19) | −0.013 | 0.07 |

- pool_matched beats uniform in 17 of 18 matched epochs from epoch 10. The mean gap at matched epochs
  is **+0.36 chrF2**, which is the fair number: +0.73 best-vs-best compares a 40-epoch run with a
  run that early-stopped at 28.
- Correction to the gate commit message: pool_matched was **not** still improving at the cap. It
  peaked at epoch 34 and the last-5-epoch slope is flat.
- Cost of compression: keeping 7% of frames costs 0.7 chrF2 against padfix, with the current decoder.
- Horizon banks: no gain (21.06 = 21.06). Drop them.
- Needed: round 2 (`phase 2/scripts/run_phase2_round2.sh`), which already runs seeds 13 and 1337
  for both arms and the CIF-style pool_avg vs uniform_avg pair. Rerun E2 after pushing.

**(b) Is the language side the bottleneck? (E3, iSign test)** Unigram precision / recall:

| system | function words P / R | content words P / R | distinct outputs |
|---|---|---|---|
| Mamba s42 | 26.0 / 25.7 | 16.8 / 13.5 | 99.2% |
| Mamba ensemble | 27.3 / 31.0 | 20.1 / 16.7 | 98.0% |
| TF s42 | 24.3 / 25.4 | 12.1 / 9.7 | 96.9% |
| TF ensemble | 25.8 / 29.5 | 14.7 / 12.3 | 97.0% |

- Outputs are varied, sentence-like and not generic (≤2% are among the 20 most frequent outputs).
  They fail on **content words**, which is a recognition problem. An LLM decoder alone does not fix that.
- Mamba's win over TF is mostly content words (P 16.8 vs 12.1; function words 26.0 vs 24.3). That is
  a Phase 1 finding worth a line in the paper: the SSM recognises more lexical content.
- This checks fluency only indirectly (diversity plus reading samples), not with a language-model score.

**Reframed A.** The core contribution is the Δ-driven adaptive-rate connector, which is SSM-specific
and cuts memory ~14× so that multi-clip context (C1) stays cheap. The LLM decoder is a secondary
experiment. Its possible benefit is lexical priors (names, domain vocabulary), not fluency, so
it has to be measured against the same connector with the current decoder rather than assumed.

## B — articulator-factored scans: pending

E4 tests two things. Part 1: are hand speed and non-rigid face speed (head motion removed) out of
sync (lagged ρ, peak lag)? Part 2: does any head of the shared scan track the face once hands are
partialled out? B is worth building only if the channels are asynchronous **and** no head tracks the face.

On t3ihpc07, from `phase 2/`:

```
python ../proposals/evidence/server/e4_articulators.py --data configs/data/isign.yaml \
    --model configs/model/mamba_banks.yaml \
    --checkpoint results/isign/mamba_banks/lr0.0003_s42/checkpoints/best.pt \
    --cache-dir ../cache --max-clips 2000 --out ../proposals/results/e4_isign
```

Limitation: speed is a crude proxy for non-manual markers. An eyebrow raise held across a clause
barely moves, so part 1 can under-state asynchrony.

## Next

1. Server: run round 2 (if not already running) and E4; push `results/` and `proposals/results/e4_isign/`.
2. Local: rerun E2 on the round-2 metrics. A stands if the matched-epoch gap stays positive on 3
   seeds and exceeds ~2× seed sd (≈0.3 chrF2).
3. Regenerate the Mamba s13 test decode (F5) and rerun E1 and E3.
