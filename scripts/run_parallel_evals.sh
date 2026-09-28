#!/bin/bash
set -euo pipefail

cd /scratch/home/student1/Sign_Language/Projects/State_Space_Model

gpu_id=0
for ds in phoenix14t how2sign isign; do
  for arm in transformer mamba; do
    for seed in 42 13 1337; do
      ckpt="results/${ds}/${arm}/lr0.0003_s${seed}/checkpoints/best.pt"
      if [ -f "$ckpt" ]; then
        echo "Evaluating $ckpt on GPU $gpu_id..."
        CUDA_VISIBLE_DEVICES=$gpu_id conda run --no-capture-output -n ssm-slt python src/evaluate.py \
          --data configs/data/${ds}.yaml \
          --model configs/model/${arm}.yaml \
          --checkpoint "$ckpt" \
          --splits test > "logs/eval_${ds}_${arm}_${seed}.log" 2>&1 &
        
        gpu_id=$(( (gpu_id + 1) % 4 ))
      fi
    done
  done
done

echo "Waiting for all background evaluations to finish..."
wait
echo "All evaluations complete! Running aggregation..."
conda run --no-capture-output -n ssm-slt python -c "import sys; sys.path.insert(0, 'src'); from slt_reporting import aggregate_runs; aggregate_runs('results', 'results/benchmark_tables.xlsx', side_by_side=True)"
echo "Aggregation complete."
