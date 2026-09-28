# Next ideas — boundary signals borrowed from systems that already solve this

Our task: turn a dense, continuous pose stream (25 fps) into a short sequence of meaningful units,
then into text. Several natural and engineered systems solve the same compression problem efficiently.
Three of them map directly onto the phase-2 pooling framework (`src/models/scope.py`,
`src/models/mamba_pool.py`) and are ranked below by value for the paper.

**Start only after round 2** (`scripts/run_phase2_round2.sh`) confirms the Δ-vs-uniform gap on seeds
13/1337. Each idea is a new arm in the same framework and must pass the same control as the gate:
**same number of memory tokens as `mamba_uniform`, only the choice of frames differs.**

---

## 1. Surprise-based boundaries

**Analogy.** Event segmentation theory (Zacks et al.): the brain continuously predicts what comes next.
A spike in prediction error marks an event boundary, and that's when memory gets updated.

**Why first.** It's the most novel of the three for sign language translation, it has a clear
cognitive-science story, and it slots straight into the existing pooling code as a new boundary signal to
compare against Δ.

**Mechanism.**
- A small head predicts the next frame's front-end features from the forward scan's output at the
  pooling layer: `pred_t = W · h_fwd_t`, and `surprise_t = ‖pred_{t-1} − x_t‖²`.
- Train it with an auxiliary MSE loss at a small weight, e.g. 0.1, and report the weight in the paper.
- Pool with the existing quantile code: `delta_quantile_ends(surprise, valid, n_keep)` in place of Δ.
  Pick-one-frame and averaging variants both work unchanged.

**Arm.** `mamba_pool_surprise` (matched count), compared with `mamba_pool_matched` (Δ) and
`mamba_uniform`.

**Cost.** The head adds one `Linear(512→512)` ≈ 0.26 M parameters (+0.8%). `count_params.py`'s
"phase-2 arms add no parameters" check needs an explicit exception for it, and the parameter
difference must be reported.

**Prior art to check.** EM-LLM (2024) uses surprise-based event segmentation for LLM long-context memory.
Verify it, and search for surprise or prediction-error segmentation in sign language or video.

---

## 2. Confidence-gated Δ

**Analogy.** The Kalman filter. A linear dynamical system (which is what an SSM is) updates its state by a
*gain*: how much to trust the new observation. When the sensor is noisy, the gain drops.

**Why.** It's a small change with an obvious motivation: don't write bad frames into memory. And it
uses data we already have but discard.

**Mechanism.**
- Δ plays the gain's role. Add a per-frame reliability term to the dt slice of `in_proj`, the same
  wrapping trick as the motion-gated Δ in `PHASE2_PLAN.md` (C2 step 2), so the fused kernels are kept:
  `dt_raw ← dt_raw + w_c · c_t`, with `w_c` initialised at 0 so training starts from the baseline.
- `c_t` = the fraction of hand keypoints that are observed (not interpolated), plus the mean MediaPipe
  confidence of the hands.

**Data change (required).** `scripts/build_cache.py` currently drops the confidence channel and keeps only
the per-clip missing fraction. The per-frame mask it computes (`_interpolate_missing`) must be written
to the cache as an extra channel. That means a cache rebuild (~40 CPU-min per dataset) with a new
config hash, and phase-1 runs are unaffected.

**Control.** Feed the same `c_t` as a plain input feature (front-end concat) to a non-gated arm, so the
gain is attributed to *gating*, not to extra information. The same reasoning as the kinematics control in C2.

---

## 3. Adaptive threshold for integrate-and-fire pooling

**Analogy.** Integrate-and-fire neurons: the neuron accumulates input, fires at a threshold and resets.
The threshold also adapts after each spike, which keeps the firing rate steady.

**Why.** It turns the pooling rate into a controllable target, and it's what makes the free-rate variant
(`mamba_pool`) usable. Today its fixed threshold lets the kept-frame count drift as Δ trains.

**Two versions, simplest first.**
- **Global rate target:** `threshold = stride × EMA(mean Δ over real frames)`, updated each training
  step and frozen at eval (a buffer, like BatchNorm's running mean). The *average* rate is pinned to
  1/stride, but each clip keeps its own rate: dense signing gets more tokens, holds get fewer. This
  is the one thing the matched arm cannot express.
- **Spike-frequency adaptation:** after each kept frame, raise the threshold by a factor and let it
  decay back. This gives local rate control. It needs a sequential loop or a small custom scan, so do
  it only if the global version helps.

**Arm.** `mamba_pool_adaptive`. Compare it with `mamba_pool_matched` *at the same average `kept_frac`*
(already logged per epoch). The question is whether letting the rate vary per clip helps beyond choosing
positions.

---

## Other analogies (motivation and citations, not arms yet)

| Source | Idea | Relevance |
|---|---|---|
| Level-crossing sampling (in electronics, analog-to-digital conversion) | sample when the signal has changed by a fixed amount | the signal-processing name for `mamba_pool_matched`; use it to frame the method |
| Event cameras, the retina | signal only changes | argues for velocity/acceleration inputs (C2 step 2) |
| Video compression (MPEG) | keyframes at scene changes plus cheap differences | memory token = boundary state + within-segment motion summary; a richer `mamba_pool_avg` |
| Speech perception rhythms (Giraud & Poeppel) | nested chunking at syllable and phrase rates | supports the multi-timescale heads; a segment-length prior at ~1–2 signs/s |
| Movement-Hold model (Liddell & Johnson) | signs = holds + movements; blinks and nods mark boundaries | check whether Δ reacts to face/head motion (`analyze_delta.py` already splits face vs hands) |
| Simultaneous interpreters | 2–3 s lag, chunked, context-heavy | motivates C1 (context across clips) and streaming |

These analogies motivate experiments; they don't prove anything. Every claim still needs the
matched-count control.
