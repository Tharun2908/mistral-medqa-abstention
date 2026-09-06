#!/bin/bash

set -euo pipefail

cd /workspace/mistral-medqa-abstention

mkdir -p logs

for fold in 1 2 3 4
do
    echo "========================================"
    echo "STARTING OOF FOLD ${fold}"
    echo "========================================"

    python scripts/phase1_sft_posthoc/generate_sft_oof_fold.py \
        --fold ${fold} \
        > logs/oof_fold${fold}.log 2>&1

    echo "FOLD ${fold} FINISHED"
done

echo "ALL OOF FOLDS FINISHED"
