#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate medqa-grpo

cd /workspace/mistral-medqa-abstention

ROOT="results/clean_protocol/learned_abstention/grpo/posthoc_arm_b_reward03/main"
EVAL="scripts/phase2_learned_abstention/eval_warmstart_dev.py"

for STEP in 50 100 150 200 250; do
    echo
    echo "================================================================================"
    echo "POSTHOC E=+0.3 — CHECKPOINT ${STEP}"
    echo "================================================================================"

    python "$EVAL" \
      --adapter "${ROOT}/checkpoints/checkpoint-${STEP}" \
      --run-name "posthoc_reward03_checkpoint_${STEP}"
done

echo
echo "================================================================================"
echo "POSTHOC E=+0.3 — FINAL"
echo "================================================================================"

python "$EVAL" \
  --adapter "${ROOT}/final" \
  --run-name "posthoc_reward03_final"

echo
echo "POSTHOC REWARD03 DEV SWEEP COMPLETE"
