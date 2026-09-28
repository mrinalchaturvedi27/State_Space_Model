# Next ideas — boundary signals borrowed from systems that already solve this

Our task: turn a dense, continuous pose stream (25 fps) into a short sequence of meaningful units,
then into text. Several natural and engineered systems solve the same compression problem efficiently.
Three of them map directly onto the phase-2 pooling framework (`src/models/scope.py`,
`src/models/mamba_pool.py`) and are ranked below by value for the paper.

**Start only after round 2** (`scripts/run_phase2_round2.sh`) confirms the Δ-vs-uniform gap on seeds
13/1337. Each idea is a new arm in the same framework and must pass the same control as the gate:
**same number of memory tokens as `mamba_uniform`, only the choice of frames differs.**

---

## 0. What these ideas can and cannot buy

Gate numbers (iSign, s42, best val chrF2; details in `../proposals/README.md`):

| run | best val chrF2 | frames kept |
|---|---|---|
| `mamba_padfix` | 21.06 | 100% |
| `mamba_pool_matched` | 20.36 | 7% |
| `mamba_uniform` | 19.63 | 7% |

Seed sd of best val chrF2 is 0.14 (phase-1 Mamba, 3 seeds).

- **Ideas 1 and 3 only change which frames are kept.** At best they close the 0.7 chrF2 gap to
  full memory; they cannot beat `mamba_padfix`.
- **Errors are in recognising the signs, not in the language** (`../proposals/results/e3_error_profile.md`:
  content-word precision 17–20% vs function words 26–27%, outputs ~98% distinct). Choosing better
  frames does not fix recognition.
- So the claim for 1 and 3 is "better quality at a fixed token budget". That only counts where the
  budget matters: C1's multi-clip context, or an LLM decoder (`../proposals/README.md`, proposal A).
  Present them as parts of C1 / A, not as standalone contributions.
- **Idea 2 is the only one that changes what goes into the state**, so it is the only one that could
  beat `mamba_padfix` itself.

---

## Control needed by every idea: hand-speed pooling

**Arm.** `mamba_pool_speed`: keep frames at equal quantiles of cumulative **hand speed** (mean
hand-keypoint displacement per frame, from the raw input), same matched count as `mamba_uniform`,
no learned parameters. Reuses `delta_quantile_ends(speed, valid, n_keep)` with speed in place of Δ.

**Why.**
- For Δ: if hand-speed pooling matches `mamba_pool_matched`, the gate result means "sample where the
  hands move" — level-crossing sampling — and Δ learned nothing a speed measurement doesn't give.
- For surprise (idea 1): at 25 fps, next-frame prediction error is mostly a function of speed, so
  surprise will likely land close to this arm too.

**Add it to round 2 now.** It decides which branch to take below.

---

## 1. Surprise-based boundaries

**Analogy.** Event segmentation theory (Zacks et al.): the brain continuously predicts what comes next.
A spike in prediction error marks an event boundary, and that's when memory gets updated.

**Why.** It has a clear cognitive-science story and slots straight into the existing pooling code as a
new boundary signal to compare against Δ. Novelty is limited to the application (see prior art).

**Mechanism.**
- A small head predicts the next frame from the forward scan's output at the pooling layer.
- **Target: the raw normalised keypoints** (`Linear(512→356)`), not the front-end output. Predicting
  the front-end output lets the front end make its own features easy to predict (a moving target).
  If a latent target is wanted anyway, stop gradients through it and take it **before dropout**, or
  the surprise signal carries dropout noise during training.
- **Surprise = error after constant-velocity extrapolation**, not after a plain next-frame prediction:
  the head predicts the residual `x_t − (2·x_{t−1} − x_{t−2})`, and
  `surprise_t = ‖pred_{t−1} − (x_t − 2·x_{t−1} + x_{t−2})‖²`. This measures a *change* in motion
  (closer to an event) instead of speed, so it is not a restatement of the hand-speed control.
- Train it with an auxiliary MSE loss at a small weight, e.g. 0.1, and report the weight in the paper.
- Pool with the existing quantile code: `delta_quantile_ends(surprise, valid, n_keep)` in place of Δ.
  Pick-one-frame and averaging variants both work unchanged.

**Arm.** `mamba_pool_surprise` (matched count), compared with `mamba_pool_matched` (Δ),
`mamba_pool_speed` and `mamba_uniform`.

**Cost.** The head adds one `Linear(512→356)` ≈ 0.18 M parameters (+0.6%). `count_params.py`'s
"phase-2 arms add no parameters" check needs an explicit exception for it, and the parameter
difference must be reported.

**Prior art.** Prediction-error event segmentation is established for video:
- Aakur & Sarkar, *A Perceptual Prediction Framework for Self Supervised Event Segmentation*, CVPR 2019.
- SUNTA, *Hierarchical Video Prediction with Surprise-based Chunking*, arXiv 2607.02087 (2026).
- EM-LLM (Fountas et al., 2024): surprise-based event segmentation for LLM long-context memory.

Not found for sign language translation. The claim is the application (surprise-timed memory
tokens for SLT), not the mechanism.

---

## 2. Confidence-gated Δ

**Analogy.** The Kalman filter. A linear dynamical system (which is what an SSM is) updates its state by a
*gain*: how much to trust the new observation. When the sensor is noisy, the gain drops.

**Why.** It's a small change with an obvious motivation: don't write bad frames into memory. It uses
data we already have but discard, and it is the only idea here aimed at recognition rather than at
frame choice. The analogy holds in Mamba-2: a small Δ both writes less and forgets less
(`exp(−Δ·A)` → 1), which is the right behaviour for a frame you don't trust.

**Check the data first (cheap, before any cache rebuild).** On the server, from the existing index
parquet: the distribution of per-clip `frac_missing`, and its relation to per-sentence chrF2 (join on
uid with `results/isign/*/lr0.0003_s*/predictions/val_best_beam5.csv`). If hand keypoints are rarely
interpolated in iSign, there is nothing to gate: stop here.

**Mechanism.**
- Δ plays the gain's role. Add a per-frame reliability term to the dt slice of `in_proj`, the same
  wrapping trick as the motion-gated Δ in `PHASE2_PLAN.md` (C2 step 2), so the fused kernels are kept:
  `dt_raw ← dt_raw + w_c · c_t`, with `w_c` initialised at 0 so training starts from the baseline.
- Apply it to **both** the forward and backward scans.
- `c_t` = the fraction of hand keypoints that are observed (not interpolated), plus the mean MediaPipe
  confidence of the hands.

**Data change (required).** `scripts/build_cache.py` currently drops the confidence channel and keeps only
the per-clip missing fraction. The per-frame mask it computes (`_interpolate_missing`) must be written
to the cache as an extra channel. That means a cache rebuild (~40 CPU-min per dataset) with a new
config hash, and phase-1 runs are unaffected.

**Control.** Feed the same `c_t` as a plain input feature (front-end concat) to a non-gated arm, so the
gain is attributed to *gating*, not to extra information. The same reasoning as the kinematics control in C2.

**Evaluation.** Besides clean val/test, evaluate with hand keypoints dropped (set missing and
interpolated as in the cache) on a fixed random 10 / 20 / 40% of frames. Gating should degrade
gracefully where the ungated arm and the concat control don't. This shows the effect more clearly
than a small change in clean BLEU.

---

## 3. Adaptive threshold for integrate-and-fire pooling

**Analogy.** Integrate-and-fire neurons: the neuron accumulates input, fires at a threshold and resets.
The threshold also adapts after each spike, which keeps the firing rate steady.

**Why.** It turns the pooling rate into a controllable target, and it's what makes the free-rate variant
(`mamba_pool`) usable. Today its fixed threshold lets the kept-frame count drift as Δ trains. It is the
cheapest idea here: no new parameters, no data change.

**Two versions, simplest first.**
- **Global rate target:** `threshold = stride × EMA(mean Δ over real frames)`, updated each training
  step and frozen at eval (a buffer, like BatchNorm's running mean). The *average* rate is pinned to
  1/stride, but each clip keeps its own rate: dense signing gets more tokens, holds get fewer. This
  is the one thing the matched arm cannot express.
- **Spike-frequency adaptation:** after each kept frame, raise the threshold by a factor and let it
  decay back. This gives local rate control. It needs a sequential loop or a small custom scan, so do
  it only if the global version helps.

**Arm.** `mamba_pool_adaptive`. Compare it with `mamba_pool_matched` *at the same average `kept_frac`*.
The question is whether letting the rate vary per clip helps beyond choosing positions.

**Logging.** Report `kept_frac` on **train and val** (both already logged per epoch). The threshold is
frozen at eval, so the val rate can drift from the train rate; expect this most on How2Sign and PHOENIX,
whose clip lengths differ from iSign's.

---

## Order of work

1. Round 2 + `mamba_pool_speed`, seeds 13 / 42 / 1337.
2. **If Δ beats hand speed** (gap above ~2× seed sd ≈ 0.3 chrF2): idea 3, then idea 1 with the fixes above.
3. Idea 2's `frac_missing` check, in parallel (CPU only).
4. **If hand speed ties Δ:** the gate result is about kinematics, not learned boundaries. Drop the
   "Δ learns better boundaries" story and build on C2's motion-gated Δ and idea 2.

---

## Other analogies (motivation and citations, not arms yet)

| Source | Idea | Relevance |
|---|---|---|
| Level-crossing sampling (in electronics, analog-to-digital conversion) | sample when the signal has changed by a fixed amount | the signal-processing name for `mamba_pool_matched` (equal steps of cumulative Δ); `mamba_pool_speed` is the same with speed as the signal |
| Event cameras, the retina | signal only changes | argues for velocity/acceleration inputs (C2 step 2) |
| Video compression (MPEG) | keyframes at scene changes plus cheap differences | memory token = boundary state + within-segment motion summary; a richer `mamba_pool_avg` |
| Speech perception rhythms (Giraud & Poeppel) | nested chunking at syllable and phrase rates | a segment-length prior at ~1–2 signs/s (the horizon banks gave no gain on their own: 21.06 = 21.06) |
| Movement-Hold model (Liddell & Johnson) | signs = holds + movements; blinks and nods mark boundaries | check whether Δ reacts to face/head motion (`analyze_delta.py`, and `../proposals/evidence/server/e4_articulators.py` with head motion removed) |
| Simultaneous interpreters | 2–3 s lag, chunked, context-heavy | motivates C1 (context across clips) and streaming |

These analogies motivate experiments; they don't prove anything. Every claim still needs the
matched-count control, and every boundary signal also needs the hand-speed control.
