#!/bin/bash
# Track A (phase 3/PHASE3_PLAN.md): pretraining at a realistic scale. The first gate pretrained for
# 5 epochs (~8 min) on iSign only and gave +0.4-0.5 chrF2; this tests the idea properly.
# Pretraining: 50 epochs (EPOCHS) on the pooled train poses of iSign + How2Sign + PHOENIX (no text
# used, so How2Sign's target-text issue does not matter), then fine-tuning on iSign as before.
# Each job: pretrain (skipped if its encoder exists) -> fine-tune with --init-encoder -> beam-5 test
# decode (length penalty 1.0, the fixed protocol).
#
#   PART=gate   P2L + P1L (Mamba, long vs clip pretraining), seed 42          <- run first, decide
#   PART=seeds  P2L + P1L seeds 13/1337; Transformer controls T2L + T1L seeds 42/13/1337
#   PART=ctx    the 2-previous-clips context model fine-tuned from each P2L encoder (needs PART=gate/seeds)
# Baselines already exist: P0 = phase 2 mamba_padfix (3 seeds), T0 = phase-1 transformer (3 seeds),
# context without pretraining = phase 2 mamba_ctx_k2 (3 seeds).
# Outputs: "phase 3/results/pretrain/<EXP>_s<seed>/", "phase 3/results/finetune/<EXP>/isign/<model>/lr0.0003_s<seed>/".
#
# Usage (on t3ihpc07, in tmux):
#   PART=gate GPUS="1 2" bash "phase 3/scripts/run_pretrain_large.sh"
#   PART=seeds GPUS="0 1 2 3" bash "phase 3/scripts/run_pretrain_large.sh"
#   T2_WINDOW=2048 ... if the Transformer's 4,096-frame pretraining runs out of memory
#   SMOKE=1 PART=gate GPUS="0" bash "phase 3/scripts/run_pretrain_large.sh"
# Logs: logs/large_<job>.log; progress: logs/large_queue.log. Each PART has its own queue file.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE2_DIR="${PROJECT_DIR}/phase 2"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPUS=(${GPUS:-0 1 2 3})
SMOKE="${SMOKE:-0}"
LOG_DIR="${PROJECT_DIR}/logs"
QUEUE="${LOG_DIR}/large_queue.txt"
LOCK="${LOG_DIR}/large_queue.lock"
mkdir -p "${LOG_DIR}"

PHASE3_DIR="${PROJECT_DIR}/phase 3"
PART="${PART:-gate}"
QUEUE="${LOG_DIR}/large_${PART}_queue.txt"
LOCK="${LOG_DIR}/large_${PART}_queue.lock"
EPOCHS="${EPOCHS:-50}"
T2_WINDOW="${T2_WINDOW:-4096}"
PT_DATA="configs/data/isign.yaml configs/data/how2sign.yaml configs/data/phoenix14t.yaml"
PT_EXTRA="--epochs ${EPOCHS}"; FT_EXTRA=""; PT_ROOT="results/pretrain"; FT_ROOT="results/finetune"
if [ "${SMOKE}" = "1" ]; then
  PT_EXTRA="--max-steps 20 --epochs 1"; FT_EXTRA="--max-steps 20 --no-wandb"
  PT_ROOT="results_smoke/pretrain"; FT_ROOT="results_smoke/finetune"
fi

job() {  # EXP seed pretrain-model mode window finetune-model results-name [init-from-EXP]
  local exp="$1" seed="$2" ptm="$3" mode="$4" win="$5" ftm="$6" rname="$7" from="${8:-}"
  local src="${from:-$exp}"
  local pt="${PHASE3_DIR}/${PT_ROOT}/${src}_s${seed}"
  local ft="${PHASE3_DIR}/${FT_ROOT}/${exp}"
  local run="${ft}/isign/${rname}/lr0.0003_s${seed}"
  if [ "${SMOKE}" != "1" ] && [ -f "${run}/predictions/test_best_beam5.csv" ]; then echo "skip ${exp} s${seed}: done" >&2; return 0; fi
  local cmd=""
  if [ -z "${from}" ]; then
    cmd="cd \"${PHASE3_DIR}\" && ( [ -f \"${pt}/done.json\" ] || python scripts/pretrain.py --data ${PT_DATA} \
--model configs/model/${ptm}.yaml --cache-dir \"${PROJECT_DIR}/cache\" --mode ${mode} --window ${win} --seed ${seed} \
--out \"${pt}\" ${PT_EXTRA} ) && "
  else
    cmd="test -f \"${pt}/encoder_init.pt\" && "
  fi
  cmd="${cmd}cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/isign.yaml --model configs/model/${ftm}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --results-dir \"${ft}\" --lr 3e-4 --seed ${seed} --init-encoder \"${pt}/encoder_init.pt\" ${FT_EXTRA}"
  if [ "${SMOKE}" != "1" ]; then
    cmd="${cmd} && python src/evaluate.py --data configs/data/isign.yaml --model configs/model/${ftm}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --checkpoint \"${run}/checkpoints/best.pt\""
  fi
  printf '%s\t%s\n' "${exp}_s${seed}" "${cmd}"
}

if [ -s "${QUEUE}" ]; then
  echo "resuming: $(wc -l < "${QUEUE}") jobs left in ${QUEUE}"
else
  {
    case "${PART}" in
      gate)  job P2L 42 mamba_padfix long 4096 mamba_padfix mamba_padfix
             job P1L 42 mamba_padfix clip 4096 mamba_padfix mamba_padfix ;;
      seeds) for s in 13 1337; do
               job P2L "$s" mamba_padfix long 4096 mamba_padfix mamba_padfix
               job P1L "$s" mamba_padfix clip 4096 mamba_padfix mamba_padfix
             done
             for s in 42 13 1337; do
               job T2L "$s" transformer long "${T2_WINDOW}" transformer transformer
               job T1L "$s" transformer clip 4096 transformer transformer
             done ;;
      ctx)   for s in 42 13 1337; do job P2L_ctx "$s" "" "" "" mamba_ctx_k2 mamba_ctx_k2 P2L; done ;;
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
    log="${LOG_DIR}/large_${name}.log"
    echo "[GPU${gpu}] start ${name} $(date)" | tee -a "${LOG_DIR}/large_queue.log"
    if CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
        bash -c "${cmd}" > "${log}" 2>&1; then
      echo "[GPU${gpu}] done  ${name} $(date)" | tee -a "${LOG_DIR}/large_queue.log"
    else
      echo "[GPU${gpu}] FAILED ${name} $(date), see ${log}" | tee -a "${LOG_DIR}/large_queue.log" >&2
    fi
  done
}

pids=()
for g in "${GPUS[@]}"; do
  worker "${g}" &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== track A finished $(date) ===" | tee -a "${LOG_DIR}/large_queue.log"
