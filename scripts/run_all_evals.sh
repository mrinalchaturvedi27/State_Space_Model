#!/bin/bash
set -euo pipefail

cd /scratch/home/student1/Sign_Language/Projects/State_Space_Model

for ds in phoenix14t how2sign isign; do
  for arm in transformer mamba; do
    for seed in 42 13 1337; do
      ckpt="results/${ds}/${arm}/lr0.0003_s${seed}/checkpoints/best.pt"
      if [ -f "$ckpt" ]; then
        echo "Evaluating $ckpt..."
        conda run --no-capture-output -n ssm-slt python src/evaluate.py \
          --data configs/data/${ds}.yaml \
          --model configs/model/${arm}.yaml \
          --checkpoint "$ckpt" \
          --splits test
      else
        echo "Missing checkpoint: $ckpt"
      fi
    done
  done
done

echo "Evaluation complete! Running aggregation..."
conda run --no-capture-output -n ssm-slt python -c "import sys; sys.path.insert(0, 'src'); from slt_reporting import aggregate_runs; aggregate_runs('results', 'results/benchmark_tables.xlsx', side_by_side=True)"
echo "Aggregation complete."
