#!/bin/bash
# Stage 3.0 (phase 3/PHASE3_PLAN.md), no training.
#   PART=decode   GPU queue: beam-5 n-best lists (val + test) for every trained model below, written
#                 to <run dir>/nbest/ (uses the length penalty lp_tune chose on validation, else 1.0)
#   PART=analyze  CPU: consensus (MBR) decoding + selective translation for each dataset x encoder,
#                 3 seeds pooled, into "phase 3/results/stage30/<dataset>_<arm>/"
# Models: phase-1 Transformer + Mamba on iSign (results/), How2Sign (results_f3/), PHOENIX
# (results_f3/ when the F3 reruns exist, else results/); phase-2 iSign full memory and Δ averaging.
#
# Usage (on t3ihpc07):
#   PART=decode GPUS="0 0 1 1" bash "phase 3/scripts/run_stage30.sh"     # two workers per GPU is fine
#   PART=analyze bash "phase 3/scripts/run_stage30.sh"
# Logs: logs/stage30_*.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/stage30_queue.txt"
LOCK="${LOG_DIR}/stage30_queue.lock"
mkdir -p "${LOG_DIR}"

PART="${PART:-decode}"
PHASE3_DIR="${PROJECT_DIR}/phase 3"
PHOENIX_ROOT="results"
[ -d "${PROJECT_DIR}/results_f3/phoenix14t" ] && PHOENIX_ROOT="results_f3"

# dataset root arm-config arm-dir : one entry per encoder (x 3 seeds)
SETS=(
  "isign results transformer transformer" "isign results mamba mamba"
  "how2sign results_f3 transformer transformer" "how2sign results_f3 mamba mamba"
  "phoenix14t ${PHOENIX_ROOT} transformer transformer" "phoenix14t ${PHOENIX_ROOT} mamba mamba"
  "isign phase_2/results mamba_padfix mamba_padfix" "isign phase_2/results mamba_pool_avg mamba_pool_avg"
)
run_root() { echo "${1//phase_2/phase 2}"; }

if [ "${PART}" = "analyze" ]; then
  cd "${PHASE3_DIR}"
  for set in "${SETS[@]}"; do
    read -r data root cfg arm <<< "${set}"
    root="$(run_root "${root}")"
    lang=en; [ "${data}" = "phoenix14t" ] && lang=de
    dirs=()
    for s in 13 42 1337; do
      d="${PROJECT_DIR}/${root}/${data}/${arm}/lr0.0003_s${s}/nbest"
      [ -f "${d}/test.csv" ] && [ -f "${d}/val.csv" ] && dirs+=("${d}")
    done
    if [ "${#dirs[@]}" -lt 2 ]; then echo "skip ${data} ${arm}: ${#dirs[@]} n-best dirs"; continue; fi
    out="results/stage30/${data}_${arm}"
    echo "=== ${data} ${arm} (${#dirs[@]} seeds) $(date)"
    conda run --no-capture-output -n "${CONDA_ENV}" python scripts/mbr.py --runs "${dirs[@]}" --split test --out "${out}" \
      2>&1 | tee "${LOG_DIR}/stage30_mbr_${data}_${arm}.log"
    conda run --no-capture-output -n "${CONDA_ENV}" python scripts/selective.py --runs "${dirs[@]}" --lang "${lang}" --out "${out}" \
      2>&1 | tee "${LOG_DIR}/stage30_selective_${data}_${arm}.log"
  done
  exit 0
fi

nb_job() {  # data root cfg arm seed
  local data="$1" root="$2" cfg="$3" arm="$4" seed="$5"
  local run="${PROJECT_DIR}/$(run_root "${root}")/${data}/${arm}/lr0.0003_s${seed}"
  [ -f "${run}/checkpoints/best.pt" ] || { echo "skip ${run}: no checkpoint" >&2; return 0; }
  [ -f "${run}/nbest/test.csv" ] && { echo "skip ${run}: n-best exists" >&2; return 0; }
  printf '%s\t%s\n' "nbest_${data}_${arm}_${root//\//_}_s${seed}" "cd \"${PHASE3_DIR}\" && python scripts/nbest_decode.py \
--data configs/data/${data}.yaml --model configs/model/${cfg}.yaml --checkpoint \"${run}/checkpoints/best.pt\" \
--cache-dir \"${PROJECT_DIR}/cache\""
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    for s in 42 13 1337; do
      for set in "${SETS[@]}"; do read -r data root cfg arm <<< "${set}"; nb_job "${data}" "${root}" "${cfg}" "${arm}" "${s}"; done
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
    log="${LOG_DIR}/stage30_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/stage30_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/stage30_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/stage30_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== stage 3.0 decoding finished $(date) ===" | tee -a "${LOG_DIR}/stage30_queue.log"
