#!/bin/bash

set -euo pipefail

cd /workspace/mistral-medqa-abstention
mkdir -p logs

echo "========================================"
echo "STARTING warm-start LR = 1e-5"
echo "========================================"

python scripts/phase2_learned_abstention/train_warmstart.py \
  --lr 1e-5 \
  --run-name lr1e5 \
  > logs/warmstart_lr1e5.log 2>&1

echo "LR 1e-5 FINISHED"


echo "========================================"
echo "STARTING warm-start LR = 3e-5"
echo "========================================"

python scripts/phase2_learned_abstention/train_warmstart.py \
  --lr 3e-5 \
  --run-name lr3e5 \
  > logs/warmstart_lr3e5.log 2>&1

echo "LR 3e-5 FINISHED"


echo "========================================"
echo "STARTING warm-start LR = 5e-5"
echo "========================================"

python scripts/phase2_learned_abstention/train_warmstart.py \
  --lr 5e-5 \
  --run-name lr5e5 \
  > logs/warmstart_lr5e5.log 2>&1

echo "LR 5e-5 FINISHED"

echo "ALL WARM-START RUNS FINISHED"
