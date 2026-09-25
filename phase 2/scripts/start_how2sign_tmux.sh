#!/bin/bash
# How2Sign benchmarks: primary (Transformer & Mamba), 3 seeds via tmux

PROJECT_DIR="/scratch/home/student1/Sign_Language/Projects/State_Space_Model"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Create worker script for Transformer
cat << 'EOF' > ${PROJECT_DIR}/scripts/h2s_tf_worker.sh
#!/bin/bash
PROJECT_DIR="/scratch/home/student1/Sign_Language/Projects/State_Space_Model"
ssh t3ihpc07 "cd ${PROJECT_DIR} && export WANDB_API_KEY=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 && for SEED in 42 13 1337; do echo '[GPU1/transformer] Starting seed='\$SEED; CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n ssm-slt python src/train.py --data configs/data/how2sign.yaml --model configs/model/transformer.yaml --cache-dir cache --results-dir results --lr 3e-4 --seed \$SEED --wandb-project ARR-SSM-vs-TF-SLT 2>&1 | tee logs/how2sign_transformer_lr3e-4_s\${SEED}.log; done"
sleep 86400
EOF
chmod +x ${PROJECT_DIR}/scripts/h2s_tf_worker.sh

# Create worker script for Mamba
cat << 'EOF' > ${PROJECT_DIR}/scripts/h2s_mamba_worker.sh
#!/bin/bash
PROJECT_DIR="/scratch/home/student1/Sign_Language/Projects/State_Space_Model"
ssh t3ihpc07 "cd ${PROJECT_DIR} && export WANDB_API_KEY=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 && for SEED in 42 13 1337; do echo '[GPU2/mamba] Starting seed='\$SEED; CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ssm-slt python src/train.py --data configs/data/how2sign.yaml --model configs/model/mamba.yaml --cache-dir cache --results-dir results --lr 3e-4 --seed \$SEED --wandb-project ARR-SSM-vs-TF-SLT 2>&1 | tee logs/how2sign_mamba_lr3e-4_s\${SEED}.log; done"
sleep 86400
EOF
chmod +x ${PROJECT_DIR}/scripts/h2s_mamba_worker.sh

echo "Killing any existing how2sign tmux sessions..."
tmux kill-session -t how2sign_tf 2>/dev/null || true
tmux kill-session -t how2sign_mamba 2>/dev/null || true

echo "Starting tmux sessions: how2sign_tf, how2sign_mamba"
tmux new-session -d -s how2sign_tf "${PROJECT_DIR}/scripts/h2s_tf_worker.sh"
tmux new-session -d -s how2sign_mamba "${PROJECT_DIR}/scripts/h2s_mamba_worker.sh"

echo "Done!"
