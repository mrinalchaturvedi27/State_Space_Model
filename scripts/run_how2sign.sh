#!/bin/bash
# How2Sign benchmarks: primary (Transformer & Mamba) + A2b (Mamba Depth), 3 seeds
# Dataset: How2Sign
# Per BENCHMARK_PLAN.md §8

set -euo pipefail

PROJECT_DIR="/scratch/home/student1/Sign_Language/Projects/State_Space_Model"
CONDA_ENV="ssm-slt"
WANDB_PROJECT="ARR-SSM-vs-TF-SLT"
LR=3e-4
SEEDS=(42 13 1337)

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "  How2Sign Benchmarking — $(date)"
echo "=========================================="

# We will run the 3 models on 3 separate GPUs in parallel, 
# and iterate through the 3 seeds sequentially on each GPU.

# --- Transformer arm on GPU 1 ---
(
  for SEED in "${SEEDS[@]}"; do
    echo "[GPU1/transformer] Starting lr=${LR} seed=${SEED} at $(date)"
    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n ${CONDA_ENV} \
      python "${PROJECT_DIR}/src/train.py" \
        --data "${PROJECT_DIR}/configs/data/how2sign.yaml" \
        --model "${PROJECT_DIR}/configs/model/transformer.yaml" \
        --cache-dir "${PROJECT_DIR}/cache" \
        --results-dir "${PROJECT_DIR}/results" \
        --lr ${LR} --seed ${SEED} \
        --wandb-project "${WANDB_PROJECT}" \
      2>&1 | tee "${LOG_DIR}/how2sign_transformer_lr${LR}_s${SEED}.log"
    echo "[GPU1/transformer] Finished lr=${LR} seed=${SEED} at $(date)"
    echo ""
  done
  echo "[GPU1/transformer] ALL DONE at $(date)"
) &
PID_TF=$!

# --- Mamba arm on GPU 2 ---
(
  for SEED in "${SEEDS[@]}"; do
    echo "[GPU2/mamba] Starting lr=${LR} seed=${SEED} at $(date)"
    CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ${CONDA_ENV} \
      python "${PROJECT_DIR}/src/train.py" \
        --data "${PROJECT_DIR}/configs/data/how2sign.yaml" \
        --model "${PROJECT_DIR}/configs/model/mamba.yaml" \
        --cache-dir "${PROJECT_DIR}/cache" \
        --results-dir "${PROJECT_DIR}/results" \
        --lr ${LR} --seed ${SEED} \
        --wandb-project "${WANDB_PROJECT}" \
      2>&1 | tee "${LOG_DIR}/how2sign_mamba_lr${LR}_s${SEED}.log"
    echo "[GPU2/mamba] Finished lr=${LR} seed=${SEED} at $(date)"
    echo ""
  done
  echo "[GPU2/mamba] ALL DONE at $(date)"
) &
PID_MAMBA=$!



echo "Transformer PID:   ${PID_TF} (GPU 1)"
echo "Mamba PID:         ${PID_MAMBA} (GPU 2)"
echo "Mamba Depth PID:   ${PID_MAMBA_DEPTH} (GPU 3)"
echo "Logs in: ${LOG_DIR}/"
echo ""

# Wait for all
wait ${PID_TF}
TF_EXIT=$?
wait ${PID_MAMBA}
MAMBA_EXIT=$?
wait ${PID_MAMBA_DEPTH}
MAMBA_DEPTH_EXIT=$?

echo "=========================================="
echo "  How2Sign Benchmarking Complete — $(date)"
echo "  Transformer exit: ${TF_EXIT}"
echo "  Mamba exit:       ${MAMBA_EXIT}"
echo "  Mamba Depth exit: ${MAMBA_DEPTH_EXIT}"
echo "=========================================="
