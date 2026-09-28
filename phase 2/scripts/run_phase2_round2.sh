#!/bin/bash
# Phase-2 round 2 on iSign, lr 3e-4. Ten runs, queued over the GPUs in priority order:
#
#   wave 1  confirm the gate: mamba_uniform + mamba_pool_matched, seeds 13 and 1337
#   wave 2  CIF-style Δ averaging vs plain averaging (s42); compression 1/8 (s42)
#   wave 3  compression 1/32 (s42)
#
# Job i runs on GPUS[i % n_gpus]; each GPU works through its own jobs one after another, so
# with 4 GPUs wave 1 finishes first. Results: phase 2/results/isign/<arm or tag>/lr0.0003_s<seed>/.
#
# Usage (on t3ihpc07, in tmux):
#   GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_round2.sh"
#   GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_round2.sh" --max-steps 20 --no-wandb --results-dir results_smoke
# Extra arguments are passed to every train.py call.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
LR=3e-4

# "config seed", highest priority first.
JOBS=(
  "mamba_uniform 13"            "mamba_uniform 1337"
  "mamba_pool_matched 13"       "mamba_pool_matched 1337"
  "mamba_uniform_avg 42"        "mamba_pool_avg 42"
  "mamba_uniform_s8 42"         "mamba_pool_matched_s8 42"
  "mamba_uniform_s32 42"        "mamba_pool_matched_s32 42"
)

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
cd "${PHASE2_DIR}"
echo "=== parameter check $(date) ==="
conda run --no-capture-output -n "${CONDA_ENV}" python scripts/count_params.py

run_queue() {
  local gpu="$1"; shift
  local job cfg seed log
  for job in "$@"; do
    read -r cfg seed <<< "${job}"
    log="${LOG_DIR}/phase2_r2_${cfg}_s${seed}.log"
    echo "[GPU${gpu}] start ${cfg} s${seed} $(date) -> ${log}"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        python src/train.py \
          --data configs/data/isign.yaml \
          --model "configs/model/${cfg}.yaml" \
          --cache-dir "${PROJECT_DIR}/cache" \
          --results-dir "${PHASE2_DIR}/results" \
          --lr "${LR}" --seed "${seed}" ${EXTRA[@]+"${EXTRA[@]}"} > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${cfg} s${seed} $(date)"
    else
      echo "[GPU${gpu}] FAILED ${cfg} s${seed}, see ${log}" >&2
    fi
  done
}

EXTRA=("$@")
pids=()
n=${#GPUS[@]}
for g in "${!GPUS[@]}"; do
  queue=()
  for j in "${!JOBS[@]}"; do
    if (( j % n == g )); then queue+=("${JOBS[$j]}"); fi
  done
  run_queue "${GPUS[$g]}" "${queue[@]}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== round 2 finished $(date) ==="
