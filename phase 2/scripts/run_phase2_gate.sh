#!/bin/bash
# Phase-2 gate on iSign, seed 42, lr 3e-4 (the phase-1 Mamba selection).
# Four arms in parallel, one GPU each. Each arm differs from the one above it by one change:
#
#   mamba_padfix        phase-1 Mamba + padding fix             (the gate's baseline)
#   mamba_banks         + horizon-bank init                     (does the init alone help?)
#   mamba_uniform       + keep every 16th frame for the decoder (cost of a shorter memory)
#   mamba_pool_matched  same frame count, positions chosen by Δ (does Δ pick better frames?)
#
# Read val corpus chrF2 and train/val kept_frac per arm from results/isign/<arm>/lr0.0003_s42/.
# The Δ claim is mamba_pool_matched vs mamba_uniform; kept_frac must be equal for those two.
#
# Usage (on t3ihpc07, from anywhere):
#   bash "phase 2/scripts/run_phase2_gate.sh"                    # GPUs 0 1 2 3
#   GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_gate.sh" --max-steps 20 --no-wandb   # smoke test
# Extra arguments are passed to every train.py call. Check nvidia-smi first: set GPUS to idle ones.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
ARMS=(mamba_padfix mamba_banks mamba_uniform mamba_pool_matched)
LR=3e-4
SEED=42

if [ "${#GPUS[@]}" -lt "${#ARMS[@]}" ]; then
  echo "need ${#ARMS[@]} GPUs for ${#ARMS[@]} arms, got GPUS='${GPUS[*]}'" >&2
  exit 1
fi

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${PHASE2_DIR}"
echo "=== parameter check $(date) ==="
conda run --no-capture-output -n "${CONDA_ENV}" python scripts/count_params.py

pids=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"
  gpu="${GPUS[$i]}"
  log="${LOG_DIR}/phase2_gate_${arm}_s${SEED}.log"
  echo "[GPU${gpu}] ${arm} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python src/train.py \
      --data configs/data/isign.yaml \
      --model "configs/model/${arm}.yaml" \
      --cache-dir "${PROJECT_DIR}/cache" \
      --results-dir "${PHASE2_DIR}/results" \
      --lr "${LR}" --seed "${SEED}" "$@" > "${log}" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${ARMS[$i]} (see ${LOG_DIR}/phase2_gate_${ARMS[$i]}_s${SEED}.log)" >&2
    status=1
  fi
done
echo "=== gate finished $(date), status ${status} ==="
exit "${status}"
