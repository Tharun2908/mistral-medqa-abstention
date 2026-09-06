#!/bin/bash

set -euo pipefail

cd /workspace/mistral-medqa-abstention
mkdir -p logs

echo "========================================"
echo "STARTING DPO ratio 1:1"
echo "========================================"

python scripts/phase2_learned_abstention/train_clean_dpo.py \
  --pair-set ratio_1to1 \
  --run-name ratio_1to1 \
  > logs/dpo_ratio_1to1.log 2>&1

echo "DPO ratio 1:1 FINISHED"


echo "========================================"
echo "STARTING DPO ratio 1.5:1"
echo "========================================"

python scripts/phase2_learned_abstention/train_clean_dpo.py \
  --pair-set ratio_1p5to1 \
  --run-name ratio_1p5to1 \
  > logs/dpo_ratio_1p5to1.log 2>&1

echo "DPO ratio 1.5:1 FINISHED"


echo "========================================"
echo "STARTING DPO ratio 2:1"
echo "========================================"

python scripts/phase2_learned_abstention/train_clean_dpo.py \
  --pair-set ratio_2to1 \
  --run-name ratio_2to1 \
  > logs/dpo_ratio_2to1.log 2>&1

echo "DPO ratio 2:1 FINISHED"

echo "ALL CLEAN DPO RATIO RUNS FINISHED"
