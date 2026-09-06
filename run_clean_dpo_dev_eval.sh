#!/bin/bash

set -euo pipefail

cd /workspace/mistral-medqa-abstention
mkdir -p logs

echo "========================================"
echo "EVALUATING DPO ratio 1:1"
echo "========================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/dpo/ratio_1to1/final/policy \
  --run-name dpo_ratio_1to1 \
  > logs/dpo_dev_ratio_1to1.log 2>&1

echo "DPO ratio 1:1 DEV EVAL FINISHED"


echo "========================================"
echo "EVALUATING DPO ratio 1.5:1"
echo "========================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/dpo/ratio_1p5to1/final/policy \
  --run-name dpo_ratio_1p5to1 \
  > logs/dpo_dev_ratio_1p5to1.log 2>&1

echo "DPO ratio 1.5:1 DEV EVAL FINISHED"


echo "========================================"
echo "EVALUATING DPO ratio 2:1"
echo "========================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/dpo/ratio_2to1/final/policy \
  --run-name dpo_ratio_2to1 \
  > logs/dpo_dev_ratio_2to1.log 2>&1

echo "DPO ratio 2:1 DEV EVAL FINISHED"

echo "ALL CLEAN DPO DEV EVALUATIONS FINISHED"
