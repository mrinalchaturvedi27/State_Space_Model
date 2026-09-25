#!/bin/bash
# Phase-2 gate on iSign, seed 42, lr 3e-4 (the phase-1 Mamba selection).
# Runs the Δ-pooling arm, then the uniform-stride control.
# Compare val corpus chrF2 against the phase-1 Mamba run at the same lr and seed.
# Pass extra train.py flags after the script name, e.g. --max-steps 20 --no-wandb.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== mamba_pool (delta) ==="
python src/train.py --data configs/data/isign.yaml --model configs/model/mamba_pool.yaml \
    --lr 3e-4 --seed 42 "$@"

echo "=== mamba_uniform (control) ==="
python src/train.py --data configs/data/isign.yaml --model configs/model/mamba_uniform.yaml \
    --lr 3e-4 --seed 42 "$@"
