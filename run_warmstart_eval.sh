#!/bin/bash

set -euo pipefail

cd /workspace/mistral-medqa-abstention
mkdir -p logs

echo "========================================"
echo "EVALUATING warm-start LR = 1e-5"
echo "========================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/warmstart/lr1e5/policy \
  --run-name lr1e5 \
  > logs/warmstart_eval_lr1e5.log 2>&1

echo "LR 1e-5 EVAL FINISHED"


echo "========================================"
echo "EVALUATING warm-start LR = 3e-5"
echo "========================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/warmstart/lr3e5/policy \
  --run-name lr3e5 \
  > logs/warmstart_eval_lr3e5.log 2>&1

echo "LR 3e-5 EVAL FINISHED"


echo "========================================"
echo "EVALUATING warm-start LR = 5e-5"
echo "========================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/warmstart/lr5e5/policy \
  --run-name lr5e5 \
  > logs/warmstart_eval_lr5e5.log 2>&1

echo "LR 5e-5 EVAL FINISHED"

echo "ALL WARM-START DEV EVALUATIONS FINISHED"
