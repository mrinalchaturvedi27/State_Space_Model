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
#   P7 select3      seeds 13/1337 for ARMS7 (default mamba_window mamba_nonselective) on iSign
#   P8 phoenix_long PHOENIX with an 80-epoch budget (base_f3_long; results in "phase 2/results_long"):
#                   transformer, tcn, mamba_padfix, mamba_nonselective, seeds 42/13/1337
#   P9 how2sign     How2Sign with one consistently cased target (configs/data/how2sign_cased.yaml), the third
#                   dataset and the longest clips: builds that cache first (CPU, ~40 min, once), then the same
#                   four encoders x 3 seeds with the 80-epoch budget into "phase 2/results_long"; seed 42 also
#                   gets the order diagnostic on short and long clips. Refuses to run unless
#                   H2S_TEXT_CHECKED=1, i.e. after scripts/check_text_columns.py confirmed text_col.
#   After the 2026-10-12 review:
#   P10 windows     boundary vs context on iSign: the window diagnostic (inference) on the existing
#                   mamba_window (3 seeds), mamba_padfix and mamba_nonselective checkpoints, then training
#                   mamba_window64, mamba_window256, mamba_window128_overlap (seed 42, + order diagnostics)
#   P11 s4d         independent LTI baseline: S4D encoder, iSign, seeds 42/13/1337 (42 + order diagnostics)
#   P12 tcn_wide3   wide-receptive-field TCN, iSign seeds 13/1337
#   Before the paper:
#   P13 lr_sweep    learning rate 1e-4 and 1e-3 (seed 42, iSign, base.yaml) for tcn, s4d, mamba_nonselective,
#                   matching phase 1's sweep of transformer and mamba (3e-4 was best on val for both)
#   P14 h2s_tcn_wide wide-receptive-field TCN on How2Sign (cased, 80-epoch budget), 3 seeds (42 + diagnostics)
# APPEND=1 adds the PARTS' jobs to the end of an already running queue and exits (the running workers
# pick them up); it refuses if no running queue has listed its jobs yet.
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
LONG_RESULTS="${PHASE2_DIR}/results_long"; [ "${SMOKE}" = "1" ] && LONG_RESULTS="${PHASE2_DIR}/results_long_smoke"

density_cmd() {  # data cfg checkpoint tag range(short|long) n [sample_file]
  local data="$1" cfg="$2" ck="$3" tag="$4" range="$5" n="$6" sample="${7:-}"
  local lim="" ; [ "${range}" = "long" ] && lim="--min-frames 257 --max-frames 512"
  local smp="" ; [ -n "${sample}" ] && smp="--sample-file \"${sample}\""
  printf '%s' "cd \"${PHASE3_DIR}\" && python scripts/diag_density.py --data configs/data/${data}.yaml \
--model configs/model/${cfg}.yaml --checkpoint \"${ck}\" --cache-dir \"${PROJECT_DIR}/cache\" --n ${n} ${lim} ${smp} \
--out ${DIAG}/density_${range}/${data}_${tag}"
}

job() {  # cfg seed data train diag(none|both|short) [results root] [lr] -> one queue line, or nothing if done
  local cfg="$1" seed="$2" data="$3" train="$4" diag="$5"
  local RESULTS="${6:-${RESULTS}}" lr="${7:-0.0003}"
  local y="${PHASE2_DIR}/configs/model/${cfg}.yaml" tag
  [ -f "${y}" ] || { echo "missing model config ${y}" >&2; exit 1; }
  tag="$(awk '$1 == "tag:" {print $2}' "${y}")"; tag="${tag:-$(awk '$1 == "arm:" {print $2}' "${y}")}"
  local run="${RESULTS}/${data}/${tag}/lr${lr}_s${seed}" name="${data}_${tag}_s${seed}"
  [ "${lr}" = "0.0003" ] || name="${data}_${tag}_lr${lr}_s${seed}"
  if [ "${SMOKE}" != "1" ] && [ -f "${run}/predictions/test_best_beam5.csv" ]; then echo "skip ${name}: done" >&2; return 0; fi
  local cmd="cd \"${PHASE2_DIR}\" && python src/train.py --data configs/data/${data}.yaml --model configs/model/${cfg}.yaml \
--train configs/train/${train}.yaml --cache-dir \"${PROJECT_DIR}/cache\" --results-dir \"${RESULTS}\" --lr ${lr} --seed ${seed} ${TRAIN_EXTRA}"
  if [ "${SMOKE}" != "1" ]; then
    cmd="${cmd} && python src/evaluate.py --data configs/data/${data}.yaml --model configs/model/${cfg}.yaml \
--cache-dir \"${PROJECT_DIR}/cache\" --checkpoint \"${run}/checkpoints/best.pt\""
    local n=2000
    case "${diag}" in
      both)  cmd="${cmd} && $(density_cmd "${data}" "${cfg}" "${run}/checkpoints/best.pt" "${tag}" short ${n}) && $(density_cmd "${data}" "${cfg}" "${run}/checkpoints/best.pt" "${tag}" long ${n})" ;;
      short) cmd="${cmd} && $(density_cmd "${data}" "${cfg}" "${run}/checkpoints/best.pt" "${tag}" short ${n})" ;;
    esac
  fi
  printf '%s\t%s\n' "${name}" "${cmd}"
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
      phoenix_long)
        for s in 42 13 1337; do for a in mamba_nonselective tcn transformer mamba_padfix; do
          job "$a" "$s" phoenix14t base_f3_long none "${LONG_RESULTS}"; done; done ;;
      how2sign)
        for s in 42 13 1337; do for a in mamba_nonselective tcn transformer mamba_padfix; do
          job "$a" "$s" how2sign_cased base_f3_long "$([ "$s" = 42 ] && echo both || echo none)" "${LONG_RESULTS}"; done; done ;;
      windows)
        WS_S="results/diagnostics/queue/density_short/sample_density_isign_0-256.csv"
        WS_L="results/diagnostics/queue/density_long/sample_density_isign_257-512.csv"
        for pair in "mamba_window|s42" "mamba_window|s13" "mamba_window|s1337" "mamba_padfix|s42" "mamba_nonselective|s42"; do
          c="${pair%%|*}"; sd="${pair#*|}"; r="${PHASE2_DIR}/results/isign/${c}/lr0.0003_${sd}"
          diag_only "window_isign_${c}_${sd}" "results/diagnostics/window/isign_${c}_${sd}" \
            "cd \"${PHASE3_DIR}\" && python scripts/diag_window.py --data configs/data/isign.yaml --model configs/model/${c}.yaml \
--checkpoint \"${r}/checkpoints/best.pt\" --cache-dir \"${PROJECT_DIR}/cache\" --sample-short ${WS_S} --sample-long ${WS_L} \
--out results/diagnostics/window/isign_${c}_${sd}"
        done
        for a in mamba_window128_overlap mamba_window64 mamba_window256; do job "$a" 42 isign base both; done ;;
      s4d)
        for s in 42 13 1337; do job s4d "$s" isign base "$([ "$s" = 42 ] && echo both || echo none)"; done ;;
      tcn_wide3)
        for s in 13 1337; do job tcn_wide "$s" isign base none; done ;;
      lr_sweep)
        for lr in 0.0001 0.001; do for a in tcn s4d mamba_nonselective; do job "$a" 42 isign base none "${RESULTS}" "${lr}"; done; done ;;
      h2s_tcn_wide)
        for s in 42 13 1337; do
          job tcn_wide "$s" how2sign_cased base_f3_long "$([ "$s" = 42 ] && echo both || echo none)" "${LONG_RESULTS}"; done ;;
      select3)
        for s in 13 1337; do for a in ${ARMS7}; do job "$a" "$s" isign base none; done; done ;;
      *) echo "unknown part ${part}" >&2; exit 1 ;;
    esac
  done
}

case " ${PARTS} " in *" how2sign "*)
  if [ "${H2S_TEXT_CHECKED:-0}" != "1" ]; then
    echo "Run first: python scripts/check_text_columns.py --data configs/data/how2sign.yaml --cache-dir cache" >&2
    echo "then confirm text_col in configs/data/how2sign_cased.yaml and rerun with H2S_TEXT_CHECKED=1." >&2
    exit 1
  fi
  if [ "${SMOKE}" != "1" ] && [ ! -f "${PROJECT_DIR}/cache/how2sign_cased/cache_manifest.json" ]; then
    echo "=== building cache/how2sign_cased $(date) (CPU, ~40 min) ===" | tee -a "${LOG_DIR}/queue.log"
    (cd "${PROJECT_DIR}" && conda run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/build_cache.py --data configs/data/how2sign_cased.yaml --out-dir cache --workers 32) \
      > "${LOG_DIR}/queue_build_cache_how2sign_cased.log" 2>&1
    echo "=== cache built $(date) ===" | tee -a "${LOG_DIR}/queue.log"
  fi
  if [ "${SMOKE}" != "1" ]; then  # refuse to train unless train and val/test targets look alike
    (cd "${PROJECT_DIR}" && conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import re, sys, pandas as pd
unk = re.compile(r'<\s*unk(nown)?\s*>', re.I)
for split in ('train', 'val', 'test'):
    t = pd.read_parquet(f'cache/how2sign_cased/{split}_index.parquet')['text'].astype(str)
    up = 100 * t.map(lambda s: any(c.isupper() for c in unk.sub('', s))).mean()
    un = 100 * t.map(lambda s: bool(unk.search(s))).mean()
    print(f'how2sign_cased {split}: {len(t)} clips, uppercase {up:.1f}%, <UNKNOWN> {un:.1f}%')
    if up < 90 or un > 1:
        sys.exit(f'target check FAILED for {split}: fix text_col / train_text_col and rebuild cache/how2sign_cased')
") 2>&1 | tee -a "${LOG_DIR}/queue.log"
    [ "${PIPESTATUS[0]}" = "0" ] || exit 1
  fi ;;
esac

if [ "${APPEND:-0}" = "1" ]; then
  if [ ! -s "${QUEUE}" ]; then
    echo "APPEND=1 needs a running queue with pending jobs in ${QUEUE}; none found (wait until it has listed" >&2
    echo "its jobs, or start these parts normally without APPEND)." >&2
    exit 1
  fi
  tmp="$(mktemp)"; build > "${tmp}"
  flock "${LOCK}" bash -c 'cat "$0" >> "$1"' "${tmp}" "${QUEUE}"
  echo "appended $(wc -l < "${tmp}") jobs to ${QUEUE}:"; cut -f1 "${tmp}" | sed 's/^/  /'; rm -f "${tmp}"
  exit 0
fi

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
