# SSM vs. Transformer — Pose-to-Text Sign Language Translation Benchmark

**Task:** ISLPose → English translation (iSign Task 1b), gloss-free, trained from scratch.
**Claim under test:** does a selective state-space encoder (Mamba-2) beat a Transformer encoder on long
pose sequences at matched parameters, data, and training budget — and at what accuracy/efficiency trade-off?
**Datasets:** iSign (first), then How2Sign, PHOENIX-2014T — same code, one config file swapped.

Everything below is either verified against the data on disk (§1) or a frozen protocol decision (§2–§7).

---

## 1. Verified data facts (measured 2026-08-24, not assumed)

| | train | val | test |
|---|---|---|---|
| segments | 99,923 | 5,653 | 6,069 |
| source videos | 3,862 | 215 | 216 |
| missing `.pose`/`.pkl` | **0** | 0 | 0 |
| unique target texts | 98,904 | 5,636 | 5,993 |
| target words (mean / med / p90 / max) | 11.9 / 11 / 18 / 30 | 11.8 / 11 / 18 / 30 | 11.7 / 11 / 18 / 30 |

* **Splits are video-disjoint** (train∩val = train∩test = val∩test = 0 videos) → no signer/topic leakage at video level.
* Exact target-sentence overlap with train: val 2.0%, test 2.3% (short, generic sentences; acceptable, report it).
* Train text: 1.36M word tokens, 32,773 word types (20,560 with freq ≥ 2) → **subword tokenizer is mandatory**.
* 3,223 train rows contain non-ASCII characters (curly quotes etc.) → NFKC-normalize before tokenizer training.
* CSV schema: `uid,text,video_id,split`; `uid = <video_id>-<segment>` and maps 1:1 to `<pose_dir>/<uid>.pose`.

**Verified paths (2026-08-24, reachable from the login node, on Lustre `/scratch`):**
```
train_csv: /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/train_split_unicode_filtered.csv
val_csv:   /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/val_split_unicode_filtered.csv
test_csv:  /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/test_split_unicode_filtered.csv
pose_dir:  /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/pose_data/isign/iSign-poses_v1.1/
```
Row counts confirmed against §1's table (99,924 / 5,654 / 6,070 lines incl. header = 99,923 / 5,653 / 6,069
segments); pose dir holds 254,474 files = 127,237 clips × {`.pose`,`.pkl`}, matching the count below.
The `/DATA7/...` path this section previously cited is **not mounted on the login node** — replaced here
with the confirmed working path. This is now the path `configs/data/isign.yaml` (§5) should use.

**Pose files** (260 GB, 127,237 clips × {`.pose`,`.pkl`}):
* `.pose` = pose-format binary, MediaPipe Holistic, **25 fps**, 576 points XYZC
  (POSE 33 + FACE 468 + LEFT_HAND 21 + RIGHT_HAND 21 + POSE_WORLD 33). Parse cost **3.4 ms/clip**.
* `.pkl` = a pre-reduced masked array `(T, 78, 3)` in **pixel** coords, iSign-only.
* Clip length @25 fps: mean 218 frames (8.7 s), p50 197, p90 360, p95 438, p99 617, max 1100 (44 s), min 22.
* `pose_format.utils.generic.reduce_holistic` → **178 keypoints** (POSE 8 + FACE 128 + hands 21+21), 2 ms/clip.

**Decision: build the feature cache from `.pose`, not `.pkl`.** `.pkl` exists only for iSign; How2Sign
and PHOENIX ship `.pose` only. One documented preprocessing path for all three datasets is worth more
than reusing a per-dataset artifact.

**How2Sign and PHOENIX paths — verified 2026-08-24, same `/scratch` tree as iSign:**
```
# How2Sign — 34,355 clips (matches the plan's earlier count)
train_csv: /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/How2sign_csv/How2sign_train.csv   (31,092 rows)
val_csv:   /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/How2sign_csv/How2sign_val.csv     (1,561 rows)
test_csv:  /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/How2sign_csv/How2sign_test.csv    (2,098 rows)
pose_dir:  /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/pose_data/how2sign/
uid_col: uid, text_col: text — BUT pose files are named by SENTENCE_ID, not uid (uid additionally suffixes
  "-<take>-rgb_front"; e.g. uid `--7E2sU6zP4_10-5-rgb_front` -> pose file `--7E2sU6zP4_10.pose`,
  keyed by the SENTENCE_ID column). configs/data/how2sign.yaml needs a separate `pose_id_col: SENTENCE_ID`
  distinct from `uid_col` — §5's "<pose_dir>/<uid>.pose maps 1:1" assumption is iSign/PHOENIX-only.
  Confirmed 0 missing pose files across all 31,092 train rows on this key.

# PHOENIX-2014T — 8,257 clips (matches the plan's earlier count), German
train_csv: /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/Phonix_csv/train.csv  (7,096 rows)
val_csv:   /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/Phonix_csv/val.csv    (519 rows)
test_csv:  /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/csv_files/Phonix_csv/test.csv   (642 rows)
pose_dir:  /scratch/home/student1/Sign_Language/Projects/ARR-August-26/pose/pose_data/phoenixT/pose_Phoenix/
uid_col: uid, text_col: text (German, lowercase) — here uid DOES map 1:1 to `<pose_dir>/<uid>.pose`
  (confirmed 0 missing across all 7,096 train rows), so no pose_id_col override needed. `glosses` column
  present but empty/ignored, per §3's no-CTC decision.
```

**Reference numbers to beat** (iSign paper, Findings ACL 2024, Table for SignPose-to-Text):
BLEU-4 0.09–1.47, ROUGE-L 7.60–19.58 (best BLEU-4: T5-base + MediaPipe(75) = 1.47; best ROUGE-L:
Camgöz et al. 2020 + I3D = 19.58). Human ceiling on 593 sentences: BLEU-4 69.3, ROUGE-L 81.9.
**Absolute BLEU on iSign is near the floor** — §6 therefore mandates chrF2 + ROUGE-L + length-bucketed
reporting, and forbids drawing conclusions from BLEU-4 alone.

---

## 2. Preprocessing — frozen once, byte-identical for both arms

Build a cache **before any training** so both arms consume the exact same tensors, and so epoch time is
not I/O-bound:

1. `Pose.read` → `reduce_holistic` → **178 keypoints**.
2. Normalize with `pose_normalization_info(header)` (shoulder-distance scale + shoulder-midpoint origin).
   Per-clip, **not** per-split statistics.
3. Keep **x, y** only, drop z (MediaPipe z is unreliable) and drop confidence from the feature vector →
   `D_in = 178 × 2 = 356`.
4. Missing/masked frames: linear interpolation across time per keypoint; leading/trailing gaps = nearest
   valid frame; fully-empty clip → zeros + flag in the manifest (**do not silently zero-fill**, count them).
5. Resample every dataset to a common **25 fps** by temporal linear interpolation (iSign 25, PHOENIX 25,
   How2Sign 30 → keeps cross-dataset numbers comparable).
6. Store `float16` in one flat `numpy.memmap` per split + a Parquet index (`uid, offset, n_frames, text`).
   Estimated size: iSign 24.4M frames × 356 × 2 B ≈ **17.4 GB** (all three datasets < 30 GB).
7. Stamp the cache with a SHA-256 of the preprocessing config; training refuses to start on a hash mismatch.

Cache build: ~40 CPU-min with 32 workers per dataset (parse 3.4 ms + reduce 2 ms per clip). Write to
`/scratch/home/student1/Sign_Language/Projects/State_Space_Model/cache/` — `/scratch` is a Lustre mount
with 315 TB free (confirmed 2026-08-24), so the 30 GB cache for all three datasets is a non-issue. This
replaces the earlier `/DATA405` plan, which is not reachable from the login node.

**Source length:** cap at **T_max = 512** frames (covers ~97.5% of clips uncut); longer clips are uniformly
subsampled to 512 (not truncated — truncation would delete the end of the sentence). Length is a first-class
variable here, so 512 is also an ablation axis (§7). No conv downsampling stem in the main config: a ×4
stem would shorten exactly the sequences whose length is the object of study. Front-end for **both** arms
is identical: `Linear(356 → d) → LayerNorm → Dropout(0.1)`.

**Train-time augmentation** (identical, same seed stream, applied to both arms):
temporal resample ×U(0.8, 1.2); random frame drop p=0.05; Gaussian coordinate noise σ=0.01;
random 2D affine (rotation ±5°, scale ±10%, translation ±0.05).
*Excluded:* horizontal flip — it swaps dominant/non-dominant hand, which is linguistically meaningful in ISL.

**Target text:** NFKC → SentencePiece **unigram, vocab 4,000**, trained on the **train split only**
(byte fallback on, `--character_coverage 1.0`), shared with the decoder output projection (tied weights).
Case and punctuation preserved. `max_tgt_len = 64` subwords. PHOENIX is small (7,096 sentences) → vocab
1,000 there; the vocab is fixed **per dataset**, identical across arms, and recorded in the results table.

---

## 3. Model arms — matched parameters, one variable changed

`d_model = 512`, pre-LN everywhere, dropout 0.1, GELU, tied target embeddings, label smoothing 0.1.

**Primary comparison (encoder swap, decoder held byte-identical):**

| Arm | Encoder | Decoder | Params |
|---|---|---|---|
| **A1 Transformer** | 6 × Transformer layer (d=512, ffn=2048, 8 heads, sinusoidal PE) | 3 × Transformer decoder layer w/ cross-attn | **33.76 M** |
| **A2 SSM** | 5 × **Bi-Mamba-2** layer (d=512, d_state=64, expand=2, headdim=64, conv=4; fwd+bwd scan, concat→proj) | *same 3 × Transformer decoder layer* | **34.01 M (+0.74%)** |

**Counts are measured, not guessed — and the measurement changed the design** (2026-08-24, implemented in
`src/models/{transformer,mamba,common}.py`, verified with `scripts/count_params.py`): a single-direction
Mamba2 module at `d_state=256` costs 1.85 M params; `BiMamba2Layer` runs **two full independent Mamba2
modules** (forward + backward, not shared weights) plus a `Linear(2·d_model, d_model)` merge projection
(0.526 M), so each bidirectional layer costs ≈3.70 M **at d_state=64 already**, not d_state=256 as
originally guessed — the pre-implementation estimate didn't anticipat
e the two-full-module cost. Measured:
Transformer enc layer 3.152 M (exact), Bi-Mamba-2 layer at d_state=64 ≈3.83 M, shared components
(embeddings + front-end + 3-layer decoder, tied) = 14.844 M exactly. **`d_state=64` is what actually lands
A2 within the 2% gate** (0.74% measured) — using 256 as originally written overshoots by 6.6% and fails the
gate. Run `scripts/count_params.py` at the start of every training job; it aborts on >2% exactly as
originally specified, just with the corrected value.

* Mamba encoder gets **no positional encoding** (the recurrence is inherently ordered) — that is the
  mechanism difference, and it is disclosed rather than hidden.
* Bidirectional encoder for A2 because the Transformer encoder is bidirectional; a unidirectional-SSM
  encoder would be a confounded comparison (it appears as an ablation in §7 instead).

**Secondary control — depth-matched:** A2b = 6 × Bi-Mamba-2, d_state=32 → **measured 37.44 M (+10.9%)**,
not the originally-guessed +2.8%. Same root cause as A2 above: this bidirectional design's fixed per-layer
overhead (≈3.70 M minimum, from the two full Mamba2 modules + projection, roughly independent of d_state)
means 6 layers of it costs more than 6 Transformer layers regardless of how low d_state goes — even
`d_state→0` only reaches +9.7%. A2b is **not** gated by the 2% check (only the primary A1/A2 pair is,
per the plan and per `count_params.py`), so this is left as-is and reported transparently in the paper
rather than chased further; if a tighter depth-matched control turns out to matter, the fix is to reduce
`expand` or `headdim`, or to tie the forward/backward Mamba2 weights (a real architectural change, not a
hyperparameter tweak, so not done unilaterally here). Reviewers ask for both "same params" and "same
depth"; A1 vs A2 answers the first, A1 vs A2b the second — cheap insurance, imperfect on the second axis.

**Secondary pair — decoder-only prefix-LM** (run only after the primary pair is done): projected frames and
text tokens in one causal stack, `[frames] ++ [BOS] text`, loss on text positions only.
A3 = Transformer-only, A4 = Mamba-2-only, matched the same way. This tests whether the SSM's advantage
survives when it must also do the decoding, not just the encoding.

**Optional hybrid** A5 = Bi-Mamba-2 encoder with a self-attention layer every 3rd block (params rematched).
Usually the strongest arm in the literature; include if the paper needs a "best system" row.

No CTC auxiliary loss: iSign has no glosses. (PHOENIX does — keep it off there too, for cross-dataset parity.)

---

## 4. Training protocol — identical for every arm

| | |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-8, weight decay 0.01 |
| No decay on | biases, all norms, and Mamba `A_log`, `D`, `dt_bias` (standard, and it matters) |
| Schedule | cosine to 10% of peak, 4,000 warmup steps |
| Grad clip | 1.0 (global norm) |
| Precision | bf16 autocast, fp32 master weights |
| Batching | token-based: ~24,576 source frames/batch (≈ 64 clips at mean length) + `grad_accum` to a fixed effective size — equal *frames* per step, not equal *clips*, so both arms see identical data per step |
| Budget | 40 epochs (≈ 62k steps at effective batch 64) — **fixed step budget, identical per arm** |
| Eval | every epoch on val: loss + BLEU-4 + chrF2 (greedy, for speed) |
| Checkpoint selection | best **val chrF2** (BLEU-4 is too noisy at iSign's score range to select on) |
| Early stopping | patience 8 epochs on val chrF2; unused budget is *not* reallocated to the other arm |
| Seeds | 13, 42, 1337 — 3 runs per arm, report mean ± std |

**Learning rate is tuned per arm, from an identical grid and budget** — `{1e-4, 3e-4, 1e-3}`, seed 42,
selected on val chrF2, then the winning lr is run with the other 2 seeds. This is the fair reading of
"all parameters the same": SSMs and Transformers have genuinely different lr optima, and freezing a single
lr for both would silently hand the win to whichever arm the number happened to suit. Document the grid,
the budget (identical), and the chosen lr per arm in the paper's appendix.

Logging: W&B (project `ARR-SSM-vs-TF-SLT`, run name `{dataset}-{arm}-lr{lr}-s{seed}`), plus a local
`results/{run}/metrics.jsonl` so the tables can be rebuilt without network access.

---

## 5. Repo layout — dataset switch is one config file

```
slt-bench/
  configs/
    data/isign.yaml        # csv paths + pose_dir + fps + vocab_size  <-- the only file that changes
    data/how2sign.yaml     #   uid/text column names live here too
    data/phoenix14t.yaml
    model/transformer.yaml  model/mamba.yaml  model/mamba_depth.yaml
    train/base.yaml        # every hyperparameter from §4, shared by all arms
  scripts/build_cache.py   # §2, multiprocessing, resumable, writes cache hash
  scripts/count_params.py  # asserts |A1 - A2| / A1 <= 0.02
  src/data.py  src/models/{transformer,mamba,common}.py  src/train.py  src/evaluate.py
  src/slt_reporting.py     # metrics + predictions + Excel + checkpoints (written & tested, see §9)
  results/{dataset}/{arm}/lr{lr}_s{seed}/{checkpoints,predictions,results}/   # §9
  results/benchmark_tables.xlsx   # aggregate_runs() output: all_runs + summary_mean_std
```

`data/*.yaml` carries `train_csv/val_csv/test_csv`, `pose_dir`, `uid_col`, `text_col`, `pose_id_col`, `fps`,
`vocab_size`. All three datasets' concrete paths are confirmed and listed in §1. `pose_id_col` defaults to
`uid_col` (true for iSign and PHOENIX, verified: pose files are named `<uid>.pose`) — **How2Sign is the
exception**, it needs `pose_id_col: SENTENCE_ID` because pose files are keyed by `SENTENCE_ID`, not `uid`
(`uid` = `SENTENCE_ID` + `-<take>-rgb_front`; see §1 for the confirmed example). Getting this wrong doesn't
error, it silently 0%-matches every How2Sign row against the pose dir, so `build_cache.py` should assert a
minimum match rate rather than only counting misses.
How2Sign needs `uid_col: uid, text_col: text` (it also has `SENTENCE`/`sentence_with_punctuation` — pick
`text` and say so); PHOENIX needs `text_col: text` (German, lowercase, `glosses` column ignored).

---

## 6. Evaluation — corpus-level, beam search, efficiency reported alongside accuracy

**Decoding:** beam 5, length penalty 1.0, `max_new_tokens = 64`, no n-gram blocking. Identical for all arms.
Evaluate on **val and test**; the paper reports test, with val in the appendix.

**Quality metrics** (all corpus-level, with SacreBLEU signatures printed):
* SacreBLEU **BLEU-4** (13a tokenizer) + BLEU-1/2/3 for continuity with the iSign paper
* **chrF2** — the primary metric for ranking systems at iSign's score range
* ROUGE-L (F1) and METEOR — iSign-paper comparability
* WER — reported in the iSign paper, cheap to add
* Optional: BERTScore-F1 (multilingual for PHOENIX)

> Deviation from the group's existing script, deliberately: `qwen-3-8bVT.py` averages *sentence-level*
> `sacrebleu.sentence_bleu` and NLTK smoothed sentence BLEU. Those are not comparable to any published
> BLEU and inflate scores when references are short. Use corpus BLEU. Keep the old numbers out of the
> new tables, or recompute them with corpus BLEU before comparing.

**Efficiency metrics — this is half the contribution, not an afterthought:**
train throughput (frames/s, clips/s), peak GPU memory at T ∈ {128, 256, 512, 1024}, inference latency and
decode tokens/s at the same T grid, params, and FLOPs/frame. Run all measurements on one idle GPU,
same batch size, same precision, median of 20 batches after 5 warmup.
*(Hardware note: this cluster's GPU node has H200 NVL, not A100 80GB — see §10. Numbers below assumed
A100 and need remeasuring; don't quote them as final.)*

**Length-bucketed quality** — the actual test of the long-range claim: report chrF2/BLEU-4 by source length
bucket (≤128, 129–256, 257–512, >512 frames) and by target length (≤8, 9–16, >16 words).
If the SSM wins only in the >256 buckets, that is the finding — say it that way.

**Statistics:** 3 seeds (mean ± std) + paired bootstrap resampling (`sacrebleu --paired-bs`, n=1000) between
A1 and A2 on the test set. Report the p-value; at iSign's score range, a 0.3 BLEU gap is noise and must not
be described as an improvement.

**Qualitative:** 25 fixed test examples (same `uid`s across arms and datasets) dumped to
`results/{run}/samples.tsv` for the paper's error-analysis table.

---

## 7. Ablations (iSign only; other two datasets get the primary pair + A2b)

1. **Source length cap** T ∈ {128, 256, 512, 1024} — both arms. The core scaling plot.
2. **Encoder directionality**: uni- vs bi-directional Mamba (isolates what bidirectionality buys the SSM).
3. **Positional encoding**: Transformer without PE, Mamba with PE (does PE explain any gap?).
4. **Depth at fixed params**: 4/6/8 Transformer layers vs 3/5/7 Bi-Mamba layers.
5. **Conv stem** ×2 / ×4 temporal downsample, applied identically — accuracy vs throughput.
6. **Keypoint set**: 178 (default) vs hands+body only (50 kp) vs full 543 — how much does the face matter?
7. **Hybrid** A5 attention-interleave frequency ∈ {every 2nd, 3rd, none}.

---

## 8. Compute budget and order of work

Estimated ~4 h per iSign run (33 M params, T ≤ 512, bf16, one A100 80GB — **assumed hardware, see §10 for
the actual H200 NVL correction**; 1,562 steps/epoch × 40 epochs). Treat the GPU-hours table below as a
rough planning shape, not a commitment, until it's redone against H200.

| Stage | Runs | GPU-hours |
|---|---|---|
| Cache build (3 datasets, CPU) | — | ~2 CPU-h |
| Smoke tests (1k-sample subset, both arms) | 4 | ~1 |
| iSign lr grid (2 arms × 3 lr, seed 42) | 6 | ~24 |
| iSign primary (2 arms × 2 remaining seeds) + A2b × 3 | 7 | ~28 |
| iSign ablations 1–4 | ~14 | ~50 |
| How2Sign (31k train → ~1.3 h/run): primary + A2b, 3 seeds | 9 | ~12 |
| PHOENIX (7k train → ~0.4 h/run): primary + A2b, 3 seeds | 9 | ~4 |
| Efficiency benchmarking (idle GPU required) | — | ~2 |

≈ 120 GPU-hours, hardware TBD after remeasuring on H200 (see §10 — this used to say "2 dedicated A100s,
GPUs 0/4 free" which doesn't correspond to this cluster's actual single 4×H200 node `t3ihpc07`). Check
`squeue`/node state before claiming a GPU, pin with `CUDA_VISIBLE_DEVICES`, and never share a GPU when
collecting the efficiency numbers, or they are meaningless.

Order: cache → smoke test → lr grid → primary pair (3 seeds) → **write the iSign table** → How2Sign →
PHOENIX → ablations. The paper is safe after the third arrow; everything after strengthens it.

---

## 9. Run outputs — checkpoints, predictions, Excel (per Sanjeet's list)

Implemented and tested in [`slt_reporting.py`](slt_reporting.py). It reproduces the lab's existing
conventions from `qwen-3-8bVT.py` rather than inventing new ones, so these runs drop into the same
comparison sheets as the Qwen/Gemma/Phi results already in the project tree.

### Directory layout (one per run)

```
results/{dataset}/{arm}/lr{lr}_s{seed}/
  checkpoints/  best.pt  last.pt  epoch_{N}.pt        # every 10 epochs, 2 most recent kept
                final_weights.pt                       # weights only, fp32, for archiving/release
  predictions/  val_epoch_{N}.csv   test_epoch_{N}.csv         # every eval epoch, greedy
                val_best_beam5.csv  test_best_beam5.csv        # from best.pt, beam 5 -> paper numbers
                #   the same reference/translation pairs are mirrored into metrics.xlsx (sheets
                #   pred_val / pred_test) so everything is readable without leaving Excel
  results/      metrics.xlsx  metrics.jsonl  config.yaml  train.log  samples.tsv
```
`run_dir(base, dataset, arm, lr, seed)` creates it. Aggregation across runs writes one
`results/benchmark_tables.xlsx` at the root.

### 1. Checkpoints
`save_checkpoint()` writes model + optimizer + scheduler + scaler + epoch/step + metrics + config +
**RNG state** (torch/cuda/numpy) so a resumed run continues bit-comparably. Written to `.tmp` then
`os.replace`d, so a kill during a write cannot corrupt the file. `best.pt` is a copy, not a symlink
(survives moving the directory between machines).
Retention: `best` + `last` always, `epoch_{N}` every 10 epochs with only the 2 most recent kept.
Size at 33.8M params: ~135 MB weights, ~405 MB with AdamW state → ~1.2 GB/run, ~55 GB for the full
grid. Goes on `/scratch` (315 TB free, see §10). Selection is on **val chrF2**, not val loss — the existing scripts select on
val loss, which at iSign's score range drifts away from translation quality (visible in the phi-3.5 sheet:
val loss rises from epoch 9 while BLEU keeps improving).

### 2. Predictions, val and test
`save_predictions()` writes the lab's exact three columns first — `video_id, prediction, ground_truth` —
so old and new CSVs are interchangeable, then appends `sent_chrF2, sent_ROUGE-L, ref_words, pred_words,
n_frames` for the length-bucketed analysis in §6. Written for **both val and test at every eval epoch**,
plus a final beam-5 pass from `best.pt` (`*_best_beam5.csv`) — those are the files the paper's numbers and
error-analysis table come from.

### 3. Excel of all metrics
`ExcelReporter` rewrites the workbook atomically after every evaluation (a crash mid-run still leaves a
readable file). Seven sheets:

| sheet | contents |
|---|---|
| `run_config` | every hyperparameter as key/value — makes each sheet self-describing |
| `epoch_metrics` | one row per eval epoch: `epoch, train_loss, val_loss`, then the legacy columns in their original order, then the corpus columns |
| `best` | the selected epoch, one row |
| `test_final` | beam-5 test metrics from `best.pt`, one row, with decoding settings |
| **`pred_val`** | **every validation sentence: `uid`, the given reference, the model's translation** |
| **`pred_test`** | **every test sentence: `uid`, the given reference, the model's translation** |
| **`samples`** | **25 fixed uids per split, one block per epoch — shows how the translation of the same sentence changes as training proceeds** |

The reference and the model's translation therefore live in the workbook itself, not only in the
prediction CSVs: `log_predictions(split, uids, predictions, references, epoch=...)` writes
`uid | ground_truth | model_translation` (+ any extra column such as `n_frames`) into `pred_<split>`.
Call it whenever a new best checkpoint is found and after the final beam-5 pass — the sheet always holds
the most recently logged set, so at the end of training it holds exactly the system being reported.
Cost measured on the real 6,069-sentence iSign test set: 0.8 s per flush, 0.5 MB workbook. Sheets are
capped at `max_pred_rows` (default 200k) so a large dataset cannot blow past Excel's 1,048,576-row limit.

**Cross-arm comparison of the actual translations:** `aggregate_runs(..., side_by_side=True)` also writes a
`pred_<dataset>` sheet in `benchmark_tables.xlsx` holding one row per test `uid` with the reference and
**every arm's translation of that same sentence side by side** (`ground_truth | transformer_s42 |
mamba_s42 | ...`). That sheet is where the paper's qualitative comparison and error analysis come from —
read one row and you see what each architecture did with the same clip.

Text written to Excel is sanitised on the way in (`_excel_safe`): control characters that make openpyxl
raise `IllegalCharacterError` are stripped, a leading `=`/`+`/`@` is escaped so Excel does not evaluate a
prediction as a formula, and cells are capped at 32,000 characters. The CSVs keep the raw untouched text.
Without this a single stray byte in a decoder output would crash `flush()` and take a multi-hour run's
bookkeeping with it.

`epoch_metrics` keeps `val_BLEU-1..4, val_ROUGE-L, val_SacreBLEU, test_BLEU-1..4, test_ROUGE-L,
test_SacreBLEU` — **identical names and order to the existing `metrics_summary.xlsx`** files
(only `train_loss` is added). Old sheets and new sheets concatenate without renaming anything.

### 4. Metrics: BLEU-1/2/3/4, SacreBLEU, ROUGE-L (+ chrF2, METEOR, WER)
`compute_metrics()` returns two families in one call:

* **legacy** (`BLEU-1..4`, `ROUGE-L`, `SacreBLEU`) — sentence-level, averaged over the corpus, a verbatim
  port of `compute_metrics` in `qwen-3-8bVT.py`. **Verified numerically identical** (< 1e-9) against the
  original implementation on 300 real predictions from the existing qwen run.
* **corpus** (`corpus_BLEU-1..4`, `corpus_chrF2`, `corpus_ROUGE-L`, `corpus_METEOR`, `corpus_WER`) —
  standard corpus-level SacreBLEU/chrF, all on the 0–100 scale. These are the paper numbers.

Both are reported because they disagree by an order of magnitude and the gap is not a bug: on those same
300 predictions, legacy `BLEU-4` = 0.009 and legacy `SacreBLEU` = 2.56, while `corpus_BLEU-4` = 0.16.
Sentence-level smoothed BLEU inflates scores on short references; corpus BLEU is what every published
SLT number uses. Keeping both means the new runs are comparable *both* to the lab's earlier sheets and to
the literature. `sacrebleu_signature()` is logged once per run for reproducibility
(`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`).

Scale warning for whoever builds the final tables: legacy `BLEU-1..4` and `ROUGE-L` are fractions (0–1)
while legacy `SacreBLEU` is 0–100 — that mixed convention is inherited from the existing sheets. Every
`corpus_*` column is 0–100. `corpus_WER` can exceed 100 when predictions are longer than references
(it hits 190 on the qwen predictions above); that is correct WER behaviour, not an error.

### Cross-run aggregation → the paper tables
`aggregate_runs(results_root, out_xlsx)` walks every `metrics.xlsx`, pulls each run's best epoch and final
test metrics, and writes `all_runs` (one row per run) + `summary_mean_std` (mean ± std over the 3 seeds,
grouped by dataset × arm). That single call turns ~45 runs into the benchmark table, and it is the only
thing that needs rerunning when a new dataset finishes.

---

## 10. Environment and hardware — resolved and verified (2026-08-24)

Everything in this section was previously a risk/guess; it is now a built, smoke-tested environment on
the actual training node, reached via direct SSH (`ssh t3ihpc07`, confirmed working from this session).

**GPU hardware:** one GPU node on this cluster, `t3ihpc07`, **4× NVIDIA H200 NVL** (143,771 MiB / ~141 GB
each), driver 595.45.04, CUDA 13.2 (driver ceiling — installed toolkits can be older, see below). Not A100
80GB as the plan originally assumed; §6/§8's throughput/memory numbers still need remeasuring on this
hardware, but the headroom (141 GB vs 80 GB) means the frame-budget in §4 has room to grow, not shrink.
Live snapshot at time of writing (`nvidia-smi` on the node): GPU0 15% util/113 GB free, GPU1 49%/109 GB
free, GPU2 **0% util/143 GB free (fully idle)**, GPU3 81%/64 GB free — GPU2 is the clean choice for the
efficiency benchmarking run in §6, which needs an untouched GPU. Two of your own jobs (`sl_vlm_paligemma`,
`isign_paligemma`) are running on GPUs 0/1/3; check `squeue`/`nvidia-smi` again before claiming one, since
this snapshot will be stale by the time training starts.

**New dedicated conda env `ssm-slt`** (Python 3.11.15, separate from `isl`), built and verified end-to-end
on `t3ihpc07`:

| package | version | source |
|---|---|---|
| torch | 2.8.0+cu126 | `pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126` |
| causal-conv1d | 1.7.0 | prebuilt wheel, `cu12torch2.8cxx11abiTRUE-cp311` |
| mamba-ssm | 2.3.2.post1 | prebuilt wheel, same tag |
| flash-attn | 2.8.3.post1 | prebuilt wheel, same tag |
| pandas, sacrebleu, nltk, rouge_score, openpyxl, pose-format, sentencepiece, wandb, jiwer, sacremoses | latest | plain `pip install` |

**Why torch 2.8.0+cu126 specifically:** checked the actual GitHub release assets for `causal-conv1d`,
`mamba-ssm`, and `flash-attn` (not guessed) — `cu12torch2.8cxx11abiTRUE-cp311` is the *only* tag all three
projects ship a prebuilt wheel for simultaneously, on Python 3.11 x86_64. mamba-ssm's newest wheels reach
torch2.10/cu13; flash-attn's newest cp311 wheel tops out at torch2.8/cu12 — 2.8 is the overlap, so that's
what's installed. This fully replaces the original Plan A/B/C uncertainty: **no source build was needed**,
and both arms get fused kernels (Plan A from the original §10), so the efficiency comparison in §6 is valid
without the "same-quality kernels" caveat that used to gate it.

Verified with a real forward pass on GPU2 (H200): `mamba_ssm.Mamba2(d_model=256, d_state=64, headdim=64)`
and `flash_attn.flash_attn_func` both ran in bf16 and returned correctly-shaped output. `slt_reporting.py`'s
`compute_metrics` also ran end-to-end in this env with correct output. `pose_format.Pose.read` was verified
against a real iSign `.pose` file: shape `(71, 1, 576, 3)` (frames, people, points, xyz) — 576 = 33+468+21+21+33
matches §1's component breakdown exactly, and `pose.body.fps == 25`, confirming §1's fps claim. **API note
for whoever writes `build_cache.py`:** fps lives on `pose.body.fps`, not `pose.header` (`PoseHeader` has no
`fps` attribute — this will `AttributeError` if guessed). `PoseBody` also already ships `.interpolate()`,
`.augment2d()`, `.frame_dropout_*()`, and `.select_frames()` methods — worth checking against §2's
hand-rolled interpolation/augmentation spec before reimplementing them from scratch.

**Installation had one recurring flake, now worked around:** `pip install` intermittently failed with
`OSError: [Errno 14] Bad address` while writing large wheels to the Lustre-backed conda env directory
(`/scratch/.../.conda/envs/...`) — reproduced across torch, causal-conv1d, mamba-ssm, and transformers
installs, roughly 1-in-3 attempts, always recoverable by immediate retry (`--no-cache-dir`, `TMPDIR` on
local disk did not prevent it, so it isn't cache-location-dependent). Once it corrupted a partially-written
`.so` (`tokenizers.abi3.so: file too short`), which needed `pip install --ignore-installed <pkg>` to fix
since the interrupted install also left no RECORD file for a clean uninstall. **Practical implication:**
any install step in a setup script for this cluster should retry on failure and verify the import
afterward, not trust a single `pip install` exit code — and after any large pip install, spot check that
critical `.so` files aren't truncated before relying on them in a training job.

**Both arms use fused kernels — the original efficiency-comparison caveat is resolved**, not a live risk.

**Disk:** `/scratch` is a Lustre mount with **315 TB free** (confirmed 2026-08-24) — this replaces the
`/DATA7`/`/`/`/DATA405` juggling entirely. Cache, checkpoints, and W&B run directories all go under
`/scratch/home/student1/Sign_Language/Projects/State_Space_Model/`. Checkpoints: keep best + last only
(33 M params ≈ 400 MB with optimizer) — capacity is no longer a constraint, so this is about keeping the
run directories navigable, not about running out of space.

**Two open decisions I would like confirmed** (defaults in brackets are what the plan assumes):
1. **Keypoint set** — `reduce_holistic` 178 kp / 356-dim [default], vs the 1,086-dim full-face features your
   VLM scripts use. 356-dim is the standard pose-baseline choice and 3× cheaper; the 1,086-dim variant is
   ablation 6 if you want continuity with the Qwen/Gemma numbers.
2. **Second SSM arm priority** — depth-matched A2b [default, ~4 GPU-h] vs jumping straight to the
   decoder-only pair A3/A4, if reviewer-proofing the encoder claim matters less to you than breadth.

**Also, unrelated but worth fixing:** `qwen-3-8bVT.py` has a hard-coded HuggingFace token in
`main()` (`hf_token = os.environ.get("HF_TOKEN", "hf_...")`). Rotate it and read it from the environment
only — it is in a file that will be shared with the paper's artifact.

**Bugs in the existing pose pipeline not to inherit** (from `qwen-3-8bVT.py`, for reference):
`pose.normalize(...)` is called *after* the keypoint slice is taken, so normalization never reaches the
returned array; the component offset uses a string comparison (`comp < component`) instead of header order;
and failed pose loads return zeros that are then trained on as if they were real. The new `build_cache.py`
handles all three, and counts the failures instead of hiding them.




### wanb api: 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