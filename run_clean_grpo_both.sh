#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate medqa-grpo

cd /workspace/mistral-medqa-abstention

mkdir -p logs

echo "============================================================"
echo "STARTING GRPO ARM A"
date
echo "============================================================"

python scripts/phase2_learned_abstention/train_clean_grpo.py \
  --arm A \
  --run-name main \
  > logs/grpo_arm_a.log 2>&1

echo "============================================================"
echo "ARM A FINISHED"
date
echo "============================================================"

echo "============================================================"
echo "STARTING GRPO ARM B"
date
echo "============================================================"

python scripts/phase2_learned_abstention/train_clean_grpo.py \
  --arm B \
  --run-name main \
  > logs/grpo_arm_b.log 2>&1

echo "============================================================"
echo "ARM B FINISHED"
date
echo "============================================================"

echo "============================================================"
echo "ALL CLEAN GRPO RUNS FINISHED"
date
echo "============================================================"
