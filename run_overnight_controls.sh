#!/bin/bash
set -euo pipefail

cd /workspace/mistral-medqa-abstention
mkdir -p logs

echo "============================================================"
echo "OVERNIGHT CLEAN CONTROL RUNS"
echo "Started: $(date)"
echo "============================================================"


echo
echo "============================================================"
echo "1/6 CONTINUE-SFT MATCHED-COMPUTE CONTROL"
echo "Started: $(date)"
echo "============================================================"

python scripts/phase2_learned_abstention/train_continue_sft_control.py \
  --run-name main

echo "Continue-SFT training finished: $(date)"


echo
echo "============================================================"
echo "2/6 CONTINUE-SFT DEV EVALUATION — CHECKPOINT 1000"
echo "Started: $(date)"
echo "============================================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/continue_sft_control/main/checkpoints/checkpoint-1000 \
  --run-name continue_sft_step1000

echo "Continue-SFT dev evaluation finished: $(date)"


echo
echo "============================================================"
echo "3/6 SUPERVISED 5-WAY — SEED 43"
echo "Started: $(date)"
echo "============================================================"

python scripts/phase2_learned_abstention/train_supervised_5way_seed43.py \
  --run-name seed43

echo "Supervised seed43 training finished: $(date)"


echo
echo "============================================================"
echo "4/6 SUPERVISED 5-WAY SEED43 — DEV EVALUATION"
echo "Started: $(date)"
echo "============================================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/supervised_5way/seed43/final \
  --run-name supervised_5way_seed43

echo "Supervised seed43 evaluation finished: $(date)"


echo
echo "============================================================"
echo "5/6 DPO 2:1 — SEED 43"
echo "Started: $(date)"
echo "============================================================"

python scripts/phase2_learned_abstention/train_clean_dpo_seed43.py \
  --pair-set ratio_2to1 \
  --run-name seed43

echo "DPO seed43 training finished: $(date)"


echo
echo "============================================================"
echo "6/6 DPO SEED43 — DEV EVALUATION"
echo "Started: $(date)"
echo "============================================================"

python scripts/phase2_learned_abstention/eval_warmstart_dev.py \
  --adapter results/clean_protocol/learned_abstention/dpo/seed43/final/policy \
  --run-name dpo_ratio_2to1_seed43

echo "DPO seed43 evaluation finished: $(date)"


echo
echo "============================================================"
echo "ALL OVERNIGHT RUNS FINISHED"
echo "Finished: $(date)"
echo "============================================================"
