#!/bin/bash
# Beam-5 val + test decode from best.pt for the four gate arms (iSign, s42), one after another
# on one GPU. Fills each run's test_final sheet and writes predictions/*_best_beam5.csv.
#
#   GPU=2 bash "phase 2/scripts/eval_phase2_gate.sh"
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPU="${GPU:-0}"
ARMS=(${ARMS:-mamba_padfix mamba_banks mamba_uniform mamba_pool_matched})
SEED="${SEED:-42}"

cd "${PHASE2_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
for arm in "${ARMS[@]}"; do
  ckpt="results/isign/${arm}/lr0.0003_s${SEED}/checkpoints/best.pt"
  log="${PROJECT_DIR}/logs/phase2_eval_${arm}_s${SEED}.log"
  echo "[GPU${GPU}] ${arm} -> ${log}"
  CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python src/evaluate.py \
      --data configs/data/isign.yaml \
      --model "configs/model/${arm}.yaml" \
      --checkpoint "${ckpt}" \
      --cache-dir "${PROJECT_DIR}/cache" > "${log}" 2>&1
done
echo "done $(date)"
