#!/bin/bash
# Post-hoc memory quantization, no training: quality vs bytes for the decoder memory of
# full memory (mamba_padfix), plain averaging (mamba_uniform_avg) and Δ-weighted averaging
# (mamba_pool_avg). Decode-only jobs, run one after another on one GPU; fine to share a GPU
# with a training queue.
#
# Done already (2026-09-30): iSign greedy sweep (3 models; 3 seeds for the averaging models),
# iSign beam-5 at fp16 / turbo:3 / naive:3 for seed 42.
#
# PART=greedy  How2Sign + PHOENIX greedy sweep, 3 models x 3 seeds (skips missing checkpoints)
# PART=beam    iSign beam-5, Δ-avg and full memory, 3 seeds, at fp16 / turbo:3 / naive:3 /
#              turbo:2 / naive:2 -- the paper's table without a seed-42-only caveat
# PART=all     both, greedy first (default)
# PART=isign   the original iSign greedy sweep (kept for reproducing it)
#
# Results: phase 2/results/<dataset>/<model>/lr0.0003_s<seed>/quant/summary.csv (+ prediction CSVs).
#
# Usage (on t3ihpc07):
#   GPU=3 PART=greedy bash "phase 2/quant/run_quant.sh"     # in one tmux window
#   GPU=2 PART=beam   bash "phase 2/quant/run_quant.sh"     # in another, in parallel
#   Quick check first:  GPU=3 MAX_CLIPS=300 PART=greedy bash "phase 2/quant/run_quant.sh"
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPU="${GPU:-0}"
PART="${PART:-all}"
MAX_CLIPS="${MAX_CLIPS:-0}"
LOG="${PROJECT_DIR}/logs/phase2_quant_${PART}.log"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PHASE2_DIR}"  # run_dir paths below are relative to phase 2, whatever the caller's directory

SWEEP="fp16,turbo:8,turbo:4,turbo:3,turbo:2,naive:8,naive:4,naive:3,naive:2,turbo_prod:4,turbo_prod:3,turbo_prod:2"
BEAM="fp16,turbo:3,naive:3,turbo:2,naive:2"
MODELS="mamba_pool_avg mamba_uniform_avg mamba_padfix"

run() {  # dataset model seed decode settings
  local data="$1" model="$2" seed="$3" decode="$4" settings="$5"
  local run_dir="results/${data}/${model}/lr0.0003_s${seed}"
  local out="${run_dir}/quant"
  [ "${MAX_CLIPS}" = "0" ] || out="${run_dir}/quant_smoke"
  if [ ! -f "${run_dir}/checkpoints/best.pt" ]; then
    echo "SKIP ${data} ${model} s${seed}: no ${run_dir}/checkpoints/best.pt" | tee -a "${LOG}"
    return 0
  fi
  echo "[GPU${GPU}] ${data} ${model} s${seed} ${decode} $(date)" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python quant/quant_eval.py --data "configs/data/${data}.yaml" --model "configs/model/${model}.yaml" \
      --checkpoint "${run_dir}/checkpoints/best.pt" --cache-dir "${PROJECT_DIR}/cache" \
      --split test --decode "${decode}" --settings "${settings}" --max-clips "${MAX_CLIPS}" \
      --out "${out}" 2>&1 | tee -a "${LOG}"
}

echo "=== quant ${PART} $(date) ===" | tee -a "${LOG}"
if [ "${PART}" = "isign" ]; then
  for m in ${MODELS}; do run isign "$m" 42 greedy "${SWEEP}"; done
  for s in 13 1337; do for m in mamba_pool_avg mamba_uniform_avg; do run isign "$m" "$s" greedy "${SWEEP}"; done; done
fi
if [ "${PART}" = "greedy" ] || [ "${PART}" = "all" ]; then
  for d in phoenix14t how2sign; do
    for s in 42 13 1337; do for m in ${MODELS}; do run "$d" "$m" "$s" greedy "${SWEEP}"; done; done
  done
fi
if [ "${PART}" = "beam" ] || [ "${PART}" = "all" ]; then
  for s in 42 13 1337; do for m in mamba_pool_avg mamba_padfix; do run isign "$m" "$s" beam "${BEAM}"; done; done
fi
echo "=== quant ${PART} finished $(date) ===" | tee -a "${LOG}"
