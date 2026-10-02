#!/bin/bash
# Length penalty chosen on validation, then test decoded once (scripts/lp_tune.py), for every
# model in REPORT.md. Decode-only, no training; one shared queue over the GPUs. Order:
#   1. phase-1 benchmark: transformer + mamba on iSign, How2Sign (results_f3) and PHOENIX, 3 seeds
#   2. phase-2 iSign: padfix, pool_avg, uniform_avg, pool_random_avg, transformer_uniform_avg,
#      ctx_k0 / ctx_k2 / ctx_k2_random, 3 seeds
#   3. phase-2 How2Sign + PHOENIX: padfix, pool_avg, uniform_avg, pool_random_avg, 3 seeds
# A job is skipped when its checkpoint is missing, or when <run dir>/lp_tune/chosen.csv exists
# (so a rerun resumes). Each job writes <run dir>/lp_tune/{summary,chosen}.csv + test predictions.
#
# Usage (on t3ihpc07, in tmux):
#   GPUS="0 1 2 3" bash "phase 2/scripts/run_lp_tune.sh"
#   MAX_CLIPS=50 GPUS="0" bash "phase 2/scripts/run_lp_tune.sh"   # quick check, writes lp_tune_smoke/
# Per-job logs: logs/lp_tune_<job>.log; progress: logs/lp_tune_queue.log.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/lp_tune_queue.txt"
LOCK="${LOG_DIR}/lp_tune_queue.lock"
mkdir -p "${LOG_DIR}"

MAX_CLIPS="${MAX_CLIPS:-0}"
SUBDIR="lp_tune"
[ "${MAX_CLIPS}" = "0" ] || SUBDIR="lp_tune_smoke"

# One line per job: <name><TAB><shell command run from PROJECT_DIR>.
lp_job() {  # dataset config run-dir (relative to PROJECT_DIR)
  local data="$1" cfg="$2" run="$3"
  local ckpt="${PROJECT_DIR}/${run}/checkpoints/best.pt"
  [ -f "${ckpt}" ] || { echo "skip ${run}: no checkpoint" >&2; return 0; }
  [ -f "${PROJECT_DIR}/${run}/${SUBDIR}/chosen.csv" ] && { echo "skip ${run}: already tuned" >&2; return 0; }
  local name
  name="$(echo "${run}" | tr '/ ' '__')"
  printf '%s\t%s\n' "${name}" "cd \"${PHASE2_DIR}\" && python scripts/lp_tune.py \
--data configs/data/${data}.yaml --model configs/model/${cfg}.yaml --checkpoint \"${ckpt}\" \
--cache-dir \"${PROJECT_DIR}/cache\" --max-clips ${MAX_CLIPS} --out \"${PROJECT_DIR}/${run}/${SUBDIR}\""
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    for s in 13 42 1337; do
      for a in transformer mamba; do
        lp_job isign "$a" "results/isign/$a/lr0.0003_s$s"
        lp_job how2sign "$a" "results_f3/how2sign/$a/lr0.0003_s$s"
        lp_job phoenix14t "$a" "results/phoenix14t/$a/lr0.0003_s$s"
      done
    done
    for s in 13 42 1337; do
      for c in mamba_padfix mamba_pool_avg mamba_uniform_avg mamba_pool_random_avg transformer_uniform_avg \
               mamba_ctx_k2 mamba_ctx_k0 mamba_ctx_k2_random; do
        lp_job isign "$c" "phase 2/results/isign/$c/lr0.0003_s$s"
      done
    done
    for d in how2sign phoenix14t; do
      for s in 13 42 1337; do
        for c in mamba_padfix mamba_pool_avg mamba_uniform_avg mamba_pool_random_avg; do
          lp_job "$d" "$c" "phase 2/results/$d/$c/lr0.0003_s$s"
        done
      done
    done
  } > "${QUEUE}"
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
    log="${LOG_DIR}/lp_tune_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/lp_tune_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/lp_tune_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/lp_tune_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== lp tuning finished $(date) ===" | tee -a "${LOG_DIR}/lp_tune_queue.log"
