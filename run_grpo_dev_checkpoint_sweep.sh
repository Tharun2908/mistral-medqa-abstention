#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate medqa-grpo

cd /workspace/mistral-medqa-abstention

mkdir -p logs

EVAL="scripts/phase2_learned_abstention/eval_warmstart_dev.py"

A_ROOT="results/clean_protocol/learned_abstention/grpo/arm_a_answer_only/main"
B_ROOT="results/clean_protocol/learned_abstention/grpo/arm_b_abstention/main"

run_eval () {
    ADAPTER="$1"
    NAME="$2"

    echo "============================================================"
    echo "DEV EVAL: $NAME"
    date
    echo "Adapter: $ADAPTER"
    echo "============================================================"

    python "$EVAL" \
      --adapter "$ADAPTER" \
      --run-name "$NAME"

    echo "FINISHED: $NAME"
    echo
}

for STEP in 50 100 150 200 250; do
    run_eval \
      "$A_ROOT/checkpoints/checkpoint-$STEP" \
      "grpo_arm_a_checkpoint_${STEP}"
done

run_eval \
  "$A_ROOT/final" \
  "grpo_arm_a_final"

for STEP in 50 100 150 200 250; do
    run_eval \
      "$B_ROOT/checkpoints/checkpoint-$STEP" \
      "grpo_arm_b_checkpoint_${STEP}"
done

run_eval \
  "$B_ROOT/final" \
  "grpo_arm_b_final"

echo "============================================================"
echo "ALL GRPO DEV CHECKPOINT EVALUATIONS FINISHED"
date
echo "============================================================"
