#!/bin/bash
# Diagnostic program steps 1-3 (RESEARCH_PROGRAM_2026-10-08.md), existing checkpoints, dev data only.
#   PART=check   PHOENIX implementation check: small samples, all three diagnostics (minutes)
#   PART=screen  iSign scientific screen, seed 42: Transformer, phase-1 Mamba (backward-scan leak) and
#                mamba_padfix (fixed). Padding invariance on the real checkpoints, video dependence
#                (1,000 clips, 200 free-decoded), sampling density / order (2,000 clips <= 256 frames).
# Runs one model at a time on one GPU. Outputs: "phase 3/results/diagnostics/<part>/{padding,video,density}/".
# The audit (step 1a) runs on CPU anywhere:  python "phase 3/scripts/diag_audit.py" --root . --out AUDIT_runs.csv
#
#   GPU=2 PART=check  bash "phase 3/scripts/run_diagnostics.sh"
#   GPU=2 PART=screen bash "phase 3/scripts/run_diagnostics.sh"
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/home/student1/Sign_Language/Projects/State_Space_Model}"
PHASE3_DIR="${PROJECT_DIR}/phase 3"
CONDA_ENV="${CONDA_ENV:-ssm-slt}"
GPU="${GPU:-0}"
PART="${PART:-check}"
LOG="${PROJECT_DIR}/logs/diag_${PART}.log"
mkdir -p "${PROJECT_DIR}/logs"
cd "${PHASE3_DIR}"

if [ "${PART}" = "check" ]; then
  DATA=phoenix14t; CROSS=isign; NV=100; ND=20; NDEN=200
  MODELS=("transformer|${PROJECT_DIR}/results_f3/phoenix14t/transformer/lr0.0003_s42"
          "mamba_padfix|${PROJECT_DIR}/phase 2/results/phoenix14t/mamba_padfix/lr0.0003_s42")
else
  DATA=isign; CROSS=how2sign; NV=1000; ND=200; NDEN=2000
  MODELS=("transformer|${PROJECT_DIR}/results/isign/transformer/lr0.0003_s42"
          "mamba|${PROJECT_DIR}/results/isign/mamba/lr0.0003_s42"
          "mamba_padfix|${PROJECT_DIR}/phase 2/results/isign/mamba_padfix/lr0.0003_s42")
fi
OUT="results/diagnostics/${PART}"
py() { CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" python "$@" 2>&1 | tee -a "${LOG}"; }

echo "=== diagnostics ${PART} on ${DATA} $(date)" | tee -a "${LOG}"
for entry in "${MODELS[@]}"; do
  cfg="${entry%%|*}"; run="${entry#*|}"; ck="${run}/checkpoints/best.pt"
  [ -f "${ck}" ] || { echo "SKIP ${cfg}: no ${ck}" | tee -a "${LOG}"; continue; }
  echo "--- ${cfg}: padding" | tee -a "${LOG}"
  py scripts/diag_padding.py --data "configs/data/${DATA}.yaml" --model "configs/model/${cfg}.yaml" \
     --checkpoint "${ck}" --cache-dir "${PROJECT_DIR}/cache" --out "${OUT}/padding/${DATA}_${cfg}.csv"
  echo "--- ${cfg}: video dependence" | tee -a "${LOG}"
  py scripts/diag_video.py --data "configs/data/${DATA}.yaml" --model "configs/model/${cfg}.yaml" \
     --checkpoint "${ck}" --cross-data "configs/data/${CROSS}.yaml" --cache-dir "${PROJECT_DIR}/cache" \
     --n "${NV}" --n-decode "${ND}" --out "${OUT}/video/${DATA}_${cfg}"
  echo "--- ${cfg}: sampling density / order" | tee -a "${LOG}"
  py scripts/diag_density.py --data "configs/data/${DATA}.yaml" --model "configs/model/${cfg}.yaml" \
     --checkpoint "${ck}" --cache-dir "${PROJECT_DIR}/cache" --n "${NDEN}" --out "${OUT}/density/${DATA}_${cfg}"
done
echo "=== done $(date)" | tee -a "${LOG}"
