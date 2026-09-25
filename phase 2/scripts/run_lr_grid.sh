#!/bin/bash
# LR grid search: 2 arms × 3 learning rates × seed 42 = 6 runs
# Transformer on GPU 1, Mamba on GPU 2 (in parallel)
# Per BENCHMARK_PLAN.md §8

set -euo pipefail

PROJECT_DIR="/scratch/home/student1/Sign_Language/Projects/State_Space_Model"
CONDA_ENV="ssm-slt"
WANDB_PROJECT="ARR-SSM-vs-TF-SLT"
SEED=42
LRS=(1e-4 3e-4 1e-3)

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "  LR Grid Search — $(date)"
echo "=========================================="

# --- Transformer arm on GPU 1 ---
(
  for LR in "${LRS[@]}"; do
    echo "[GPU1/transformer] Starting lr=${LR} seed=${SEED} at $(date)"
    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n ${CONDA_ENV} \
      python "${PROJECT_DIR}/src/train.py" \
        --data "${PROJECT_DIR}/configs/data/isign.yaml" \
        --model "${PROJECT_DIR}/configs/model/transformer.yaml" \
        --cache-dir "${PROJECT_DIR}/cache" \
        --results-dir "${PROJECT_DIR}/results" \
        --lr ${LR} --seed ${SEED} \
        --wandb-project "${WANDB_PROJECT}" \
      2>&1 | tee "${LOG_DIR}/transformer_lr${LR}_s${SEED}.log"
    echo "[GPU1/transformer] Finished lr=${LR} at $(date)"
    echo ""
  done
  echo "[GPU1/transformer] ALL DONE at $(date)"
) &
PID_TF=$!

# --- Mamba arm on GPU 2 ---
(
  for LR in "${LRS[@]}"; do
    echo "[GPU2/mamba] Starting lr=${LR} seed=${SEED} at $(date)"
    CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ${CONDA_ENV} \
      python "${PROJECT_DIR}/src/train.py" \
        --data "${PROJECT_DIR}/configs/data/isign.yaml" \
        --model "${PROJECT_DIR}/configs/model/mamba.yaml" \
        --cache-dir "${PROJECT_DIR}/cache" \
        --results-dir "${PROJECT_DIR}/results" \
        --lr ${LR} --seed ${SEED} \
        --wandb-project "${WANDB_PROJECT}" \
      2>&1 | tee "${LOG_DIR}/mamba_lr${LR}_s${SEED}.log"
    echo "[GPU2/mamba] Finished lr=${LR} at $(date)"
    echo ""
  done
  echo "[GPU2/mamba] ALL DONE at $(date)"
) &
PID_MAMBA=$!

echo "Transformer PID: ${PID_TF} (GPU 1)"
echo "Mamba PID:       ${PID_MAMBA} (GPU 2)"
echo "Logs in: ${LOG_DIR}/"
echo ""

# Wait for both
wait ${PID_TF}
TF_EXIT=$?
wait ${PID_MAMBA}
MAMBA_EXIT=$?

echo "=========================================="
echo "  LR Grid Complete — $(date)"
echo "  Transformer exit: ${TF_EXIT}"
echo "  Mamba exit:       ${MAMBA_EXIT}"
echo "=========================================="
