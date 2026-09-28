#!/bin/bash
# Round 3: one shared job queue over the GPUs. Each GPU takes the next job when it is free,
# so short test decodes fill the gaps between trainings. Priority order:
#
#   1. random-spacing control (decides whether Δ carries information): mamba_pool_random at
#      1/16 and 1/32, seeds 13/42/1337, each followed by its beam-5 val/test decode
#   2. beam-5 val/test decodes for the round-2 runs that have none yet (10)
#   3. seeds 13/1337 for the 1/32 pair and the averaging pair, each + decode
#   4. How2Sign phase-1 rerun (transformer + mamba, 3 seeds) with the F3 warmup fix
#      (configs/train/base_f3.yaml), into results_f3/ so the original runs stay intact, + decode
#
# Usage (on t3ihpc07, in tmux):
#   GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_round3.sh"
#   SMOKE=1 GPUS="0 1 2 3" bash "phase 2/scripts/run_phase2_round3.sh"   # 20 steps, no decodes
# Per-job logs: logs/phase2_r3_<job>.log; progress: logs/phase2_r3_queue.log.
# Jobs left unstarted if interrupted: logs/phase2_r3_queue.txt (delete it to start over).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/phase2_r3_queue.txt"
LOCK="${LOG_DIR}/phase2_r3_queue.lock"
mkdir -p "${LOG_DIR}"

P2_RESULTS="results"
P1_RESULTS="results_f3"
TRAIN_EXTRA=""
if [ "${SMOKE}" = "1" ]; then
  P2_RESULTS="results_smoke"
  P1_RESULTS="results_smoke_f3"
  TRAIN_EXTRA="--max-steps 20 --no-wandb"
fi

# One line per job: <name><TAB><shell command run from PROJECT_DIR>.
p2_train() {  # config seed -- phase-2 iSign training, then (unless SMOKE) its decode
  local cfg="$1" seed="$2"
  local cmd="cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/isign.yaml \
--model configs/model/${cfg}.yaml --cache-dir \"${PROJECT_DIR}/cache\" \
--results-dir \"${PHASE2_DIR}/${P2_RESULTS}\" --lr 3e-4 --seed ${seed} ${TRAIN_EXTRA}"
  [ "${SMOKE}" = "1" ] || cmd="${cmd} && $(p2_eval_cmd "${cfg}" "${seed}")"
  printf '%s\t%s\n' "train_${cfg}_s${seed}" "${cmd}"
}
p2_eval_cmd() {  # config seed -- results dir name is the config name (its tag or arm)
  printf '%s' "cd \"${PHASE2_DIR}\" && python src/evaluate.py --data configs/data/isign.yaml \
--model configs/model/$1.yaml --cache-dir \"${PROJECT_DIR}/cache\" \
--checkpoint ${P2_RESULTS}/isign/$1/lr0.0003_s$2/checkpoints/best.pt"
}
p2_eval() {
  [ "${SMOKE}" = "1" ] && return 0
  printf '%s\t%s\n' "eval_$1_s$2" "$(p2_eval_cmd "$1" "$2")"
}
h2s_f3() {  # arm seed -- phase-1 code and configs, How2Sign, F3 schedule
  local arm="$1" seed="$2"
  local cmd="cd \"${PROJECT_DIR}\" && python src/train.py --data configs/data/how2sign.yaml \
--model configs/model/${arm}.yaml --train configs/train/base_f3.yaml --cache-dir cache \
--results-dir ${P1_RESULTS} --lr 3e-4 --seed ${seed} ${TRAIN_EXTRA}"
  [ "${SMOKE}" = "1" ] || cmd="${cmd} && python src/evaluate.py --data configs/data/how2sign.yaml \
--model configs/model/${arm}.yaml --cache-dir cache \
--checkpoint ${P1_RESULTS}/how2sign/${arm}/lr0.0003_s${seed}/checkpoints/best.pt"
  printf '%s\t%s\n' "h2s_f3_${arm}_s${seed}" "${cmd}"
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    for s in 13 42 1337; do p2_train mamba_pool_random "$s"; done
    for s in 13 42 1337; do p2_train mamba_pool_random_s32 "$s"; done
    for s in 13 1337; do p2_eval mamba_uniform "$s"; p2_eval mamba_pool_matched "$s"; done
    for c in mamba_uniform_avg mamba_pool_avg mamba_uniform_s8 mamba_pool_matched_s8 \
             mamba_uniform_s32 mamba_pool_matched_s32; do p2_eval "$c" 42; done
    for s in 13 1337; do
      for c in mamba_uniform_s32 mamba_pool_matched_s32 mamba_uniform_avg mamba_pool_avg; do
        p2_train "$c" "$s"
      done
    done
    for s in 13 42 1337; do h2s_f3 transformer "$s"; h2s_f3 mamba "$s"; done
  } > "${QUEUE}"
  echo "queued $(wc -l < "${QUEUE}") jobs in ${QUEUE}"
fi

# Keep How2Sign-rerun outputs pushable like results/: only .xlsx/.csv, never checkpoints.
mkdir -p "${PROJECT_DIR}/${P1_RESULTS}"
cp -n "${PROJECT_DIR}/results/.gitignore" "${PROJECT_DIR}/${P1_RESULTS}/.gitignore" 2>/dev/null || true

echo "=== parameter check $(date) ===" | tee -a "${LOG_DIR}/phase2_r3_queue.log"
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
    log="${LOG_DIR}/phase2_r3_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/phase2_r3_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/phase2_r3_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/phase2_r3_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== round 3 finished $(date) ===" | tee -a "${LOG_DIR}/phase2_r3_queue.log"
