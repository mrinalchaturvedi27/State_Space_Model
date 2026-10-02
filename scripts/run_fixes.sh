#!/bin/bash
# Fixes from ABLATION_AND_ERROR_ANALYSIS.md, phase-1 code (Transformer vs Mamba), F3 schedule.
# Each job trains, decodes test with beam 5, then picks the length penalty on validation
# (phase 2/scripts/lp_tune.py). Results go to results_f3/<dataset>/<arm>/lr0.0003_s<seed>/.
#
#   PART=phoenix   PHOENIX-2014T, transformer + mamba x 3 seeds: the phase-1 PHOENIX runs used the
#                  default 4,000-step warmup and were under-trained (Mamba 31.72 -> 33.94 with F3)
#   PART=how2sign  How2Sign with one consistently cased target column (configs/data/how2sign_cased.yaml):
#                  builds that cache first (CPU, ~40 min), then transformer + mamba x 3 seeds.
#                  Refuses to run unless H2S_TEXT_CHECKED=1, i.e. after scripts/check_text_columns.py
#                  confirmed the mismatch and the text_col in how2sign_cased.yaml.
#   PART=all       both (PHOENIX jobs first)
#
# Usage (on t3ihpc07, in tmux):
#   PART=phoenix GPUS="0 1 2" bash scripts/run_fixes.sh
#   H2S_TEXT_CHECKED=1 PART=all GPUS="0 1 2 3" bash scripts/run_fixes.sh
#   SMOKE=1 PART=phoenix GPUS="0" bash scripts/run_fixes.sh      # 20 steps, no decodes
# Per-job logs: logs/fixes_<job>.log; progress: logs/fixes_queue.log.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/fixes_queue.txt"
LOCK="${LOG_DIR}/fixes_queue.lock"
mkdir -p "${LOG_DIR}"

PART="${PART:-phoenix}"
RESULTS="results_f3"
TRAIN_EXTRA=""
if [ "${SMOKE}" = "1" ]; then
  RESULTS="results_smoke_f3"
  TRAIN_EXTRA="--max-steps 20 --no-wandb"
fi

# One line per job: <name><TAB><shell command run from PROJECT_DIR>.
p1_job() {  # dataset arm seed -- phase-1 training (F3), then beam-5 test decode + val-chosen length penalty
  local data="$1" arm="$2" seed="$3"
  local run="${RESULTS}/${data}/${arm}/lr0.0003_s${seed}"
  if [ "${SMOKE}" != "1" ] && [ -f "${PROJECT_DIR}/${run}/lp_tune/chosen.csv" ]; then
    echo "skip ${run}: already done" >&2
    return 0
  fi
  local cmd="cd \"${PROJECT_DIR}\" && python src/train.py --data configs/data/${data}.yaml \
--model configs/model/${arm}.yaml --train configs/train/base_f3.yaml --cache-dir cache \
--results-dir ${RESULTS} --lr 3e-4 --seed ${seed} ${TRAIN_EXTRA}"
  if [ "${SMOKE}" != "1" ]; then
    cmd="${cmd} && python src/evaluate.py --data configs/data/${data}.yaml --model configs/model/${arm}.yaml \
--cache-dir cache --checkpoint ${run}/checkpoints/best.pt && cd \"${PHASE2_DIR}\" && python scripts/lp_tune.py \
--data configs/data/${data}.yaml --model configs/model/${arm}.yaml \
--checkpoint \"${PROJECT_DIR}/${run}/checkpoints/best.pt\" --cache-dir \"${PROJECT_DIR}/cache\""
  fi
  printf '%s\t%s\n' "${data}_${arm}_s${seed}" "${cmd}"
}

if [ "${PART}" = "how2sign" ] || [ "${PART}" = "all" ]; then
  if [ "${H2S_TEXT_CHECKED:-0}" != "1" ]; then
    echo "Run first: python scripts/check_text_columns.py --data configs/data/how2sign.yaml --cache-dir cache" >&2
    echo "then confirm text_col in configs/data/how2sign_cased.yaml and rerun with H2S_TEXT_CHECKED=1." >&2
    exit 1
  fi
  if [ ! -f "${PROJECT_DIR}/cache/how2sign_cased/cache_manifest.json" ]; then
    echo "=== building cache/how2sign_cased $(date) (CPU) ===" | tee -a "${LOG_DIR}/fixes_queue.log"
    (cd "${PROJECT_DIR}" && conda run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/build_cache.py --data configs/data/how2sign_cased.yaml --out-dir cache --workers 32) \
      > "${LOG_DIR}/fixes_build_cache_how2sign_cased.log" 2>&1
  fi
fi

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    if [ "${PART}" = "phoenix" ] || [ "${PART}" = "all" ]; then
      for s in 13 42 1337; do for a in transformer mamba; do p1_job phoenix14t "$a" "$s"; done; done
    fi
    if [ "${PART}" = "how2sign" ] || [ "${PART}" = "all" ]; then
      for s in 13 42 1337; do for a in transformer mamba; do p1_job how2sign_cased "$a" "$s"; done; done
    fi
  } > "${QUEUE}"
  echo "queued $(wc -l < "${QUEUE}") jobs in ${QUEUE}"
fi

mkdir -p "${PROJECT_DIR}/${RESULTS}"
cp -n "${PROJECT_DIR}/results/.gitignore" "${PROJECT_DIR}/${RESULTS}/.gitignore" 2>/dev/null || true

echo "=== parameter check $(date) ===" | tee -a "${LOG_DIR}/fixes_queue.log"
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
    log="${LOG_DIR}/fixes_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/fixes_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/fixes_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/fixes_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== fixes finished $(date) ===" | tee -a "${LOG_DIR}/fixes_queue.log"
