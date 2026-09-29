#!/bin/bash
# Post-hoc memory quantization on iSign test, no training: quality vs bytes for the decoder
# memory of three trained models -- full memory (mamba_padfix), plain averaging
# (mamba_uniform_avg) and Δ-weighted averaging (mamba_pool_avg). Decode-only jobs, run one
# after another on one GPU; fine to share a GPU with a training queue.
#
#   1. greedy sweep, seed 42, all three models: fp16 and 8/4/3/2 bits for turbo and naive,
#      plus turbo_prod at 4/3/2 (one process per model, the checkpoint is loaded once)
#   2. greedy sweep on seeds 13/1337 for the two averaging models (the ones with 3 seeds)
#   3. beam-5 confirmation at fp16, turbo:3, naive:3 for the three seed-42 models
#
# Results: phase 2/results/isign/<model>/lr0.0003_s<seed>/quant/summary.csv (+ prediction CSVs).
#
# Usage (on t3ihpc07):  GPU=3 bash "phase 2/quant/run_quant.sh"
#   Quick check first:  GPU=3 MAX_CLIPS=300 bash "phase 2/quant/run_quant.sh"   (writes quant_smoke/)
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPU="${GPU:-0}"
MAX_CLIPS="${MAX_CLIPS:-0}"
LOG="${PROJECT_DIR}/logs/phase2_quant.log"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PHASE2_DIR}"  # run_dir paths below are relative to phase 2, whatever the caller's directory

SWEEP="fp16,turbo:8,turbo:4,turbo:3,turbo:2,naive:8,naive:4,naive:3,naive:2,turbo_prod:4,turbo_prod:3,turbo_prod:2"
BEAM="fp16,turbo:3,naive:3"

run() {  # model seed decode settings
  local model="$1" seed="$2" decode="$3" settings="$4"
  local run_dir="results/isign/${model}/lr0.0003_s${seed}"
  local out="${run_dir}/quant"
  [ "${MAX_CLIPS}" = "0" ] || out="${run_dir}/quant_smoke"
  if [ ! -f "${run_dir}/checkpoints/best.pt" ]; then
    echo "SKIP ${model} s${seed}: no ${run_dir}/checkpoints/best.pt" | tee -a "${LOG}"
    return 0
  fi
  echo "[GPU${GPU}] ${model} s${seed} ${decode} $(date)" | tee -a "${LOG}"
  (cd "${PHASE2_DIR}" && CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python quant/quant_eval.py --data configs/data/isign.yaml --model "configs/model/${model}.yaml" \
      --checkpoint "${run_dir}/checkpoints/best.pt" --cache-dir "${PROJECT_DIR}/cache" \
      --split test --decode "${decode}" --settings "${settings}" --max-clips "${MAX_CLIPS}" \
      --out "${out}") 2>&1 | tee -a "${LOG}"
}

echo "=== quant sweep $(date) ===" | tee -a "${LOG}"
for m in mamba_pool_avg mamba_uniform_avg mamba_padfix; do run "$m" 42 greedy "${SWEEP}"; done
for s in 13 1337; do
  for m in mamba_pool_avg mamba_uniform_avg; do run "$m" "$s" greedy "${SWEEP}"; done
done
for m in mamba_pool_avg mamba_uniform_avg mamba_padfix; do run "$m" 42 beam "${BEAM}"; done
echo "=== quant sweep finished $(date) ===" | tee -a "${LOG}"
