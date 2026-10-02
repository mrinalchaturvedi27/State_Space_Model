#!/bin/bash
# Stage 3.1 go/no-go gate (phase 3/PHASE3_PLAN.md §3.1.4), iSign, Mamba (mamba_padfix), seed 42.
# Each job = one pipeline: masked-frame pretraining -> fine-tune with --init-encoder -> beam-5 test
# decode -> length penalty chosen on validation. Baseline P0 = phase 2's mamba_padfix (no pretraining).
#   P1       pretraining on single clips (control: same frames, no long context)
#   P2       pretraining on 4,096-frame windows of stitched story videos
#   P2hand   P2 with hand-focused masking (hand-frac 0.6 instead of 0.3)
# Pretrained encoders: "phase 3/results/pretrain/<EXP>/"; fine-tunes: "phase 3/results/finetune/<EXP>/".
# Gate: P2 > P0 (21.26 test chrF2 at s42) by >= 0.5 with better content-word F1 -> continue; P2 > P1 too
# -> long context is the headline.
#
# Usage (on t3ihpc07, in tmux):
#   python "phase 3/scripts/pretrain.py" --data configs/data/isign.yaml --model configs/model/mamba_padfix.yaml \
#       --cache-dir cache --stats-only --out /tmp/x          # stitching statistics first (CPU, seconds)
#   GPUS="0 1 2" bash "phase 3/scripts/run_pretrain_gate.sh"
#   SMOKE=1 GPUS="0" bash "phase 3/scripts/run_pretrain_gate.sh"      # 20 pretrain + 20 fine-tune steps
# Logs: logs/gate_<EXP>.log; progress: logs/gate_queue.log.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/gate_queue.txt"
LOCK="${LOG_DIR}/gate_queue.lock"
mkdir -p "${LOG_DIR}"

PHASE3_DIR="${PROJECT_DIR}/phase 3"
PT_EXTRA=""; FT_EXTRA=""; PT_ROOT="results/pretrain"; FT_ROOT="results/finetune"
if [ "${SMOKE}" = "1" ]; then
  PT_EXTRA="--max-steps 20 --epochs 1"; FT_EXTRA="--max-steps 20 --no-wandb"
  PT_ROOT="results_smoke/pretrain"; FT_ROOT="results_smoke/finetune"
fi

gate_job() {  # EXP mode hand-frac
  local exp="$1" mode="$2" hand="$3"
  local pt="${PHASE3_DIR}/${PT_ROOT}/${exp}"
  local ft="${PHASE3_DIR}/${FT_ROOT}/${exp}"
  local run="${ft}/isign/mamba_padfix/lr0.0003_s42"
  if [ "${SMOKE}" != "1" ] && [ -f "${run}/lp_tune/chosen.csv" ]; then echo "skip ${exp}: done" >&2; return 0; fi
  local cmd="cd \"${PHASE3_DIR}\" && python scripts/pretrain.py --data configs/data/isign.yaml \
--model configs/model/mamba_padfix.yaml --cache-dir \"${PROJECT_DIR}/cache\" --mode ${mode} --hand-frac ${hand} \
--seed 42 --out \"${pt}\" ${PT_EXTRA} && cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/isign.yaml \
--model configs/model/mamba_padfix.yaml --cache-dir \"${PROJECT_DIR}/cache\" --results-dir \"${ft}\" \
--lr 3e-4 --seed 42 --init-encoder \"${pt}/encoder_init.pt\" ${FT_EXTRA}"
  if [ "${SMOKE}" != "1" ]; then
    cmd="${cmd} && python src/evaluate.py --data configs/data/isign.yaml --model configs/model/mamba_padfix.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --checkpoint \"${run}/checkpoints/best.pt\" && python scripts/lp_tune.py \
--data configs/data/isign.yaml --model configs/model/mamba_padfix.yaml --checkpoint \"${run}/checkpoints/best.pt\" \
--cache-dir \"${PROJECT_DIR}/cache\""
  fi
  printf '%s\t%s\n' "${exp}" "${cmd}"
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  { gate_job P2 long 0.3; gate_job P1 clip 0.3; gate_job P2hand long 0.6; } > "${QUEUE}"
  echo "queued $(wc -l < "${QUEUE}") jobs in ${QUEUE}"
fi

next_job() {  # atomically pop the first queue line
  flock "${LOCK}" bash -c 'l=$(head -n 1 "$0"); if [ -n "$l" ]; then sed -i 1d "$0"; fi; printf "%s" "$l"' "${QUEUE}"
}

worker() {
  local gpu="$1" line name cmd log
  while true; do
    line="$(next_job)"
    [ -z "${line}" ] && break
    name="${line%%$'\t'*}"
    cmd="${line#*$'\t'}"
    log="${LOG_DIR}/gate_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/gate_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/gate_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/gate_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== gate finished $(date) ===" | tee -a "${LOG_DIR}/gate_queue.log"
