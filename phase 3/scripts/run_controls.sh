#!/bin/bash
# Diagnostic program step 4 (RESEARCH_PROGRAM_2026-10-08.md): which missing ingredient explains Mamba's
# advantage? The screen showed the Transformer is nearly order-blind (all frames shuffled: -0.35 chrF2
# on iSign) while Mamba depends on temporal order (-4.21).
#   PART=longorder  no training: the order / density diagnostic on LONG clips (257-512 frames, where the
#                   length gap is largest) for the existing Transformer and corrected Mamba, iSign seed 42
#   PART=train      iSign seed 42, parameter-matched controls, each: train -> beam-5 test decode ->
#                   order / density diagnostics on short (<=256) and long (257-512) clips:
#                     transformer_convstem  conv stem (receptive field 9 frames) + Transformer
#                     transformer_relpos    ALiBi relative positions instead of absolute sinusoids
#                     tcn                   attention-free dilated temporal conv (receptive field 121)
#   PART=seeds      seeds 13 / 1337 for the arms in ARMS (default: all three), after the screen
# Baselines (seed 42): results/isign/transformer, phase 2 results mamba_padfix; their short-clip
# diagnostics are in "phase 3/results/diagnostics/screen".
#
# Usage (on t3ihpc07):
#   GPUS="2" PART=longorder bash "phase 3/scripts/run_controls.sh"
#   GPUS="0 1 3" PART=train bash "phase 3/scripts/run_controls.sh"
#   SMOKE=1 GPUS="0" PART=train bash "phase 3/scripts/run_controls.sh"   # 20 steps, no decodes/diagnostics
# Logs: logs/controls_<job>.log; progress: logs/controls_queue.log. Each PART has its own queue file.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/controls_queue.txt"
LOCK="${LOG_DIR}/controls_queue.lock"
mkdir -p "${LOG_DIR}"

PHASE3_DIR="${PROJECT_DIR}/phase 3"
PART="${PART:-train}"
QUEUE="${LOG_DIR}/controls_${PART}_queue.txt"
LOCK="${LOG_DIR}/controls_${PART}_queue.lock"
ARMS="${ARMS:-transformer_convstem transformer_relpos tcn}"
RESULTS="${PHASE2_DIR}/results"; TRAIN_EXTRA=""
if [ "${SMOKE}" = "1" ]; then RESULTS="${PHASE2_DIR}/results_smoke"; TRAIN_EXTRA="--max-steps 20 --no-wandb"; fi
DIAG="results/diagnostics"

diag_cmds() {  # cfg checkpoint tag -> both length ranges of the order/density diagnostic
  local cfg="$1" ck="$2" tag="$3"
  printf '%s' "cd \"${PHASE3_DIR}\" && python scripts/diag_density.py --data configs/data/isign.yaml \
--model configs/model/${cfg}.yaml --checkpoint \"${ck}\" --cache-dir \"${PROJECT_DIR}/cache\" --n 2000 \
--out ${DIAG}/controls/density_short/isign_${tag} && python scripts/diag_density.py --data configs/data/isign.yaml \
--model configs/model/${cfg}.yaml --checkpoint \"${ck}\" --cache-dir \"${PROJECT_DIR}/cache\" --n 2000 \
--min-frames 257 --max-frames 512 --out ${DIAG}/controls/density_long/isign_${tag}"
}

train_job() {  # arm seed
  local arm="$1" seed="$2"
  local run="${RESULTS}/isign/${arm}/lr0.0003_s${seed}"
  if [ "${SMOKE}" != "1" ] && [ -f "${run}/predictions/test_best_beam5.csv" ]; then echo "skip ${arm} s${seed}: done" >&2; return 0; fi
  local cmd="cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/isign.yaml --model configs/model/${arm}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --results-dir \"${RESULTS}\" --lr 3e-4 --seed ${seed} ${TRAIN_EXTRA}"
  if [ "${SMOKE}" != "1" ]; then
    cmd="${cmd} && python src/evaluate.py --data configs/data/isign.yaml --model configs/model/${arm}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --checkpoint \"${run}/checkpoints/best.pt\""
    [ "${seed}" = "42" ] && cmd="${cmd} && $(diag_cmds "${arm}" "${run}/checkpoints/best.pt" "${arm}")"
  fi
  printf '%s\t%s\n' "${arm}_s${seed}" "${cmd}"
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    case "${PART}" in
      longorder)
        for pair in "transformer|${PROJECT_DIR}/results/isign/transformer/lr0.0003_s42" \
                    "mamba_padfix|${PHASE2_DIR}/results/isign/mamba_padfix/lr0.0003_s42"; do
          cfg="${pair%%|*}"; run="${pair#*|}"
          printf '%s\t%s\n' "longorder_${cfg}" "cd \"${PHASE3_DIR}\" && python scripts/diag_density.py \
--data configs/data/isign.yaml --model configs/model/${cfg}.yaml --checkpoint \"${run}/checkpoints/best.pt\" \
--cache-dir \"${PROJECT_DIR}/cache\" --n 2000 --min-frames 257 --max-frames 512 --out ${DIAG}/controls/density_long/isign_${cfg}"
        done ;;
      train) for a in ${ARMS}; do train_job "$a" 42; done ;;
      seeds) for s in 13 1337; do for a in ${ARMS}; do train_job "$a" "$s"; done; done ;;
      *) echo "unknown PART=${PART}" >&2; exit 1 ;;
    esac
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
    log="${LOG_DIR}/controls_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/controls_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/controls_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/controls_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== controls finished $(date) ===" | tee -a "${LOG_DIR}/controls_queue.log"
