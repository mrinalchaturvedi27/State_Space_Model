#!/bin/bash
# Round 5: one shared job queue over the GPUs (same runner as rounds 3-4). Priority order:
#
#   1. C1 confirmation (iSign): mamba_ctx_k0 / k2 / k2_random, seeds 13 and 1337; and does the
#      gain grow with context? mamba_ctx_k4 / k8, seed 42 (configs/train/base_ctx_long.yaml)
#   2. Δ-weighted averaging on How2Sign and PHOENIX-2014T: mamba_uniform_avg / pool_random_avg /
#      pool_avg, 3 seeds each, F3 schedule (configs/train/base_f3.yaml)
#   3. full-memory baseline mamba_padfix: iSign seeds 13/1337; How2Sign + PHOENIX 3 seeds (F3)
#   4. mamba_pool_avg_jitter (iSign, 3 seeds): does a training-time shift add to Δ averaging?
# Every training job is followed by its beam-5 val/test decode.
#
# Usage (on t3ihpc07, in tmux):
#   GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_round5.sh"
#   SMOKE=1 GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_round5.sh"   # 20 steps, no decodes
# Per-job logs: logs/phase2_r5_<job>.log; progress: logs/phase2_r5_queue.log.
# Jobs left unstarted if interrupted: logs/phase2_r5_queue.txt (delete it to start over).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/phase2_r5_queue.txt"
LOCK="${LOG_DIR}/phase2_r5_queue.lock"
mkdir -p "${LOG_DIR}"

P2_RESULTS="results"
TRAIN_EXTRA=""
if [ "${SMOKE}" = "1" ]; then
  P2_RESULTS="results_smoke"
  TRAIN_EXTRA="--max-steps 20 --no-wandb"
fi

# One line per job: <name><TAB><shell command run from PROJECT_DIR>.
p2_train() {  # config seed [dataset] [train config] -- training, then (unless SMOKE) its decode
  local cfg="$1" seed="$2" data="${3:-isign}" train="${4:-base}"
  local cmd="cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/${data}.yaml \
--model configs/model/${cfg}.yaml --train configs/train/${train}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --results-dir \"${PHASE2_DIR}/${P2_RESULTS}\" \
--lr 3e-4 --seed ${seed} ${TRAIN_EXTRA}"
  [ "${SMOKE}" = "1" ] || cmd="${cmd} && $(p2_eval_cmd "${cfg}" "${seed}" "${data}")"
  printf '%s\t%s\n' "train_${data}_${cfg}_s${seed}" "${cmd}"
}
p2_eval_cmd() {  # config seed dataset -- results dir name is the config name (its tag or arm)
  printf '%s' "cd \"${PHASE2_DIR}\" && python src/evaluate.py --data configs/data/$3.yaml \
--model configs/model/$1.yaml --cache-dir \"${PROJECT_DIR}/cache\" \
--checkpoint ${P2_RESULTS}/$3/$1/lr0.0003_s$2/checkpoints/best.pt"
}
if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    for s in 13 1337; do
      for c in mamba_ctx_k2 mamba_ctx_k2_random mamba_ctx_k0; do p2_train "$c" "$s"; done
    done
    p2_train mamba_ctx_k4 42 isign base_ctx_long
    p2_train mamba_ctx_k8 42 isign base_ctx_long
    for d in phoenix14t how2sign; do
      for s in 13 42 1337; do
        for c in mamba_pool_avg mamba_pool_random_avg mamba_uniform_avg; do p2_train "$c" "$s" "$d" base_f3; done
      done
    done
    for s in 13 1337; do p2_train mamba_padfix "$s"; done
    for d in phoenix14t how2sign; do
      for s in 13 42 1337; do p2_train mamba_padfix "$s" "$d" base_f3; done
    done
    for s in 13 42 1337; do p2_train mamba_pool_avg_jitter "$s"; done
  } > "${QUEUE}"
  echo "queued $(wc -l < "${QUEUE}") jobs in ${QUEUE}"
fi

echo "=== parameter check $(date) ===" | tee -a "${LOG_DIR}/phase2_r5_queue.log"
(cd "${PHASE2_DIR}" && conda run --no-capture-output -n "${CONDA_ENV}" python scripts/count_params.py)

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
    log="${LOG_DIR}/phase2_r5_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/phase2_r5_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/phase2_r5_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/phase2_r5_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== round 5 finished $(date) ===" | tee -a "${LOG_DIR}/phase2_r5_queue.log"
