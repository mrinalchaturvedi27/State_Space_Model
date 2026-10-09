#!/bin/bash
# The experiment queue (EXPERIMENT_QUEUE.md), in priority order. One shared job list, popped by one
# worker per GPU, so higher-priority jobs always start first. Finished runs (test predictions present)
# are skipped, so the script can be re-run after a crash or with more parts.
#   P1 confirm      iSign seeds 13/1337: tcn (the result to confirm), then conv stem and relative positions
#   P2 select       iSign seed 42: mamba_window, mamba_nonselective, tcn_wide, transformer_local
#                   (does Mamba's long-clip lead need selective, long-range memory?)
#   P3 phoenix      PHOENIX (base_f3) tcn seeds 42/13/1337; order diagnostics for tcn, Transformer, Mamba
#   P4 profile      parameters, training step time/memory, batch-1 encoder latency (one job, minutes)
#   P6 bow          E6 on PHOENIX seed 42: Transformer and Mamba, each with the lexical (bag-of-words) loss
#   P7 select3      conditional, only after P2 says so: seeds 13/1337 for ARMS7 (default mamba_window
#                   mamba_nonselective) on iSign
# P5 (How2Sign target check + retrain) is manual; see EXPERIMENT_QUEUE.md.
# Each iSign seed-42 training job also runs the order/density diagnostic on short and long clips.
#
# Usage (on t3ihpc07):
#   PARTS="confirm select phoenix profile bow" GPUS="0 1 2 3" bash "phase 3/scripts/run_queue.sh"
#   SMOKE=1 PARTS="select phoenix" GPUS="0" bash "phase 3/scripts/run_queue.sh"   # 20 steps, no decode
# Logs: logs/queue_<job>.log; progress: logs/queue.log; pending jobs: logs/queue_jobs.txt
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/queue_jobs.txt"
LOCK="${LOG_DIR}/queue_jobs.lock"
mkdir -p "${LOG_DIR}"

PHASE3_DIR="${PROJECT_DIR}/phase 3"
PARTS="${PARTS:-confirm select phoenix profile bow}"
ARMS7="${ARMS7:-mamba_window mamba_nonselective}"
RESULTS="${PHASE2_DIR}/results"; TRAIN_EXTRA=""
if [ "${SMOKE}" = "1" ]; then
  RESULTS="${PHASE2_DIR}/results_smoke"; TRAIN_EXTRA="--max-steps 20 --no-wandb"
  QUEUE="${LOG_DIR}/queue_smoke_jobs.txt"; LOCK="${LOG_DIR}/queue_smoke_jobs.lock"
fi
DIAG="results/diagnostics/queue"

density_cmd() {  # data cfg checkpoint tag range(short|long) n [sample_file]
  local data="$1" cfg="$2" ck="$3" tag="$4" range="$5" n="$6" sample="${7:-}"
  local lim="" ; [ "${range}" = "long" ] && lim="--min-frames 257 --max-frames 512"
  local smp="" ; [ -n "${sample}" ] && smp="--sample-file \"${sample}\""
  printf '%s' "cd \"${PHASE3_DIR}\" && python scripts/diag_density.py --data configs/data/${data}.yaml \
--model configs/model/${cfg}.yaml --checkpoint \"${ck}\" --cache-dir \"${PROJECT_DIR}/cache\" --n ${n} ${lim} ${smp} \
--out ${DIAG}/density_${range}/${data}_${tag}"
}

job() {  # cfg seed data train diag(none|both|short) -> one queue line, or nothing if already done
  local cfg="$1" seed="$2" data="$3" train="$4" diag="$5"
  local y="${PHASE2_DIR}/configs/model/${cfg}.yaml" tag
  [ -f "${y}" ] || { echo "missing model config ${y}" >&2; exit 1; }
  tag="$(awk '$1 == "tag:" {print $2}' "${y}")"; tag="${tag:-$(awk '$1 == "arm:" {print $2}' "${y}")}"
  local run="${RESULTS}/${data}/${tag}/lr0.0003_s${seed}"
  if [ "${SMOKE}" != "1" ] && [ -f "${run}/predictions/test_best_beam5.csv" ]; then echo "skip ${data} ${tag} s${seed}: done" >&2; return 0; fi
  local cmd="cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/${data}.yaml --model configs/model/${cfg}.yaml \
--train configs/train/${train}.yaml --cache-dir \"${PROJECT_DIR}/cache\" --results-dir \"${RESULTS}\" --lr 3e-4 --seed ${seed} ${TRAIN_EXTRA}"
  if [ "${SMOKE}" != "1" ]; then
    cmd="${cmd} && python src/evaluate.py --data configs/data/${data}.yaml --model configs/model/${cfg}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --checkpoint \"${run}/checkpoints/best.pt\""
    local n=2000
    case "${diag}" in
      both)  cmd="${cmd} && $(density_cmd "${data}" "${cfg}" "${run}/checkpoints/best.pt" "${tag}" short ${n}) && $(density_cmd "${data}" "${cfg}" "${run}/checkpoints/best.pt" "${tag}" long ${n})" ;;
      short) cmd="${cmd} && $(density_cmd "${data}" "${cfg}" "${run}/checkpoints/best.pt" "${tag}" short ${n})" ;;
    esac
  fi
  printf '%s\t%s\n' "${data}_${tag}_s${seed}" "${cmd}"
}

diag_only() {  # name cmd -- for existing checkpoints; skipped when its summary exists
  local name="$1" out="$2" cmd="$3"
  if [ "${SMOKE}" = "1" ] || [ -f "${PHASE3_DIR}/${out}/summary.csv" ]; then return 0; fi
  printf '%s\t%s\n' "${name}" "${cmd}"
}

build() {
  for part in ${PARTS}; do
    case "${part}" in
      confirm)
        for s in 13 1337; do job tcn "$s" isign base none; done
        for a in transformer_convstem transformer_relpos; do for s in 13 1337; do job "$a" "$s" isign base none; done; done ;;
      select)
        for a in mamba_window mamba_nonselective tcn_wide transformer_local; do job "$a" 42 isign base both; done ;;
      phoenix)
        for s in 42 13 1337; do job tcn "$s" phoenix14t base_f3 "$([ "$s" = 42 ] && echo short || echo none)"; done
        for pair in "transformer|${PROJECT_DIR}/results_f3/phoenix14t/transformer/lr0.0003_s42" \
                    "mamba_padfix|${PHASE2_DIR}/results/phoenix14t/mamba_padfix/lr0.0003_s42"; do
          c="${pair%%|*}"; r="${pair#*|}"
          diag_only "diag_phoenix14t_${c}" "${DIAG}/density_short/phoenix14t_${c}" \
            "$(density_cmd phoenix14t "${c}" "${r}/checkpoints/best.pt" "${c}" short 2000)"
        done ;;
      profile)
        [ "${SMOKE}" = "1" ] || printf '%s\t%s\n' "profile" "cd \"${PHASE3_DIR}\" && python scripts/profile_encoders.py \
--models transformer transformer_convstem transformer_relpos transformer_local tcn tcn_wide mamba_padfix mamba_nonselective mamba_window \
--out results/profile/profile.csv" ;;
      bow)
        job transformer 42 phoenix14t base_f3 none   # phase-2 harness baseline cell (skipped if present)
        for a in transformer_bow mamba_padfix_bow; do job "$a" 42 phoenix14t base_f3 none; done ;;
      select3)
        for s in 13 1337; do for a in ${ARMS7}; do job "$a" "$s" isign base none; done; done ;;
      *) echo "unknown part ${part}" >&2; exit 1 ;;
    esac
  done
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE} (delete it to rebuild from PARTS)"
else
  build > "${QUEUE}"
  echo "queued $(wc -l < "${QUEUE}") jobs in ${QUEUE}:"; cut -f1 "${QUEUE}" | sed 's/^/  /'
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
    log="${LOG_DIR}/queue_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== queue finished $(date) ===" | tee -a "${LOG_DIR}/queue.log"
