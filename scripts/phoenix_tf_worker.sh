#!/bin/bash
PROJECT_DIR="/scratch/home/student1/Sign_Language/Projects/State_Space_Model"
ssh t3ihpc07 "cd ${PROJECT_DIR} && export WANDB_API_KEY=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 && for SEED in 42 13 1337; do echo '[GPU1/transformer] Starting seed='\$SEED; CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n ssm-slt python src/train.py --data configs/data/phoenix14t.yaml --model configs/model/transformer.yaml --cache-dir cache --results-dir results --lr 3e-4 --seed \$SEED --wandb-project ARR-SSM-vs-TF-SLT 2>&1 | tee logs/phoenix_transformer_lr3e-4_s\${SEED}.log; done"
sleep 86400
