"""
build_clean_source_pools.py
---------------------------

Create deterministic, disjoint source pools for the clean learned-abstention
experiments.

Input:
    Clean 5-fold OOF SFT predictions over all 10,178 MedQA training examples.

Outputs:
    1. warmstart_data.json
    2. dpo_candidate_pools.json

Important:
    - Uses MedQA TRAIN OOF predictions only.
    - No MedQA dev/test examples are used.
    - Warm-start and DPO source examples are disjoint.
    - DPO pair ratios are NOT chosen here.
"""

import json
import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "oof_sft"
    / "sft_train_oof_predictions.json"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
)

WARMSTART_FILE = OUTPUT_DIR / "warmstart_data.json"
DPO_POOL_FILE = OUTPUT_DIR / "dpo_candidate_pools.json"

SEED = 42

WARMSTART_WRONG = 800
WARMSTART_CORRECT = 1500

ANSWER_SET = ["A", "B", "C", "D"]

ABSTAIN = " I cannot answer confidently."
ANSWER_TEMPLATE = " The answer is {}."


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

def build_prompt(question, options):
    option_lines = "\n".join(
        f"{k}: {options[k]}"
        for k in ANSWER_SET
    )

    return (
        f"Question: {question}\n\n"
        f"Options:\n"
        f"{option_lines}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def source_record(row):
    """
    Keep all metadata needed to build future DPO pairs.
    """

    return {
        "train_index": row["train_index"],
        "id": row["id"],
        "question": row["question"],
        "options": row["options"],
        "answer_idx": row["answer_idx"],
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "oof_fold": row["oof_fold"],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print("BUILD CLEAN LEARNED-ABSTENTION SOURCE POOLS")
    print("=" * 72)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("protocol") != "clean_oof_v1":
        raise RuntimeError(
            f"Unexpected protocol: {data.get('protocol')}"
        )

    predictions = data["predictions"]

    if len(predictions) != 10178:
        raise RuntimeError(
            f"Expected 10178 OOF predictions, got {len(predictions)}"
        )

    # --------------------------------------------------------------
    # Basic integrity
    # --------------------------------------------------------------

    indices = [int(p["train_index"]) for p in predictions]

    if len(set(indices)) != 10178:
        raise RuntimeError("OOF predictions contain duplicate train indices.")

    malformed = [
        p for p in predictions
        if p.get("prediction") not in ANSWER_SET
    ]

    if malformed:
        raise RuntimeError(
            f"Found {len(malformed)} malformed predictions."
        )

    correct = [
        p for p in predictions
        if p["prediction"] == p["answer_idx"]
    ]

    wrong = [
        p for p in predictions
        if p["prediction"] != p["answer_idx"]
    ]

    print(f"\nCorrect OOF predictions : {len(correct)}")
    print(f"Wrong OOF predictions   : {len(wrong)}")

    if len(correct) != 5219 or len(wrong) != 4959:
        raise RuntimeError(
            "Correct/wrong counts do not match the combined OOF artifact."
        )

    # --------------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------------

    rng = random.Random(SEED)

    correct = correct.copy()
    wrong = wrong.copy()

    rng.shuffle(correct)
    rng.shuffle(wrong)

    # --------------------------------------------------------------
    # Reserve warm-start examples
    # --------------------------------------------------------------

    warm_wrong = wrong[:WARMSTART_WRONG]

    dpo_wrong = wrong[WARMSTART_WRONG:]

    warm_correct = correct[:WARMSTART_CORRECT]

    dpo_correct = correct[WARMSTART_CORRECT:]

    print("\nSource allocation:")
    print(f"  Warm-start wrong   : {len(warm_wrong)}")
    print(f"  Warm-start correct : {len(warm_correct)}")
    print(f"  DPO wrong pool     : {len(dpo_wrong)}")
    print(f"  DPO correct pool   : {len(dpo_correct)}")

    # --------------------------------------------------------------
    # Strict disjointness checks
    # --------------------------------------------------------------

    warm_indices = {
        int(x["train_index"])
        for x in warm_wrong + warm_correct
    }

    dpo_indices = {
        int(x["train_index"])
        for x in dpo_wrong + dpo_correct
    }

    overlap = warm_indices & dpo_indices

    if overlap:
        raise RuntimeError(
            f"Warm-start / DPO overlap detected: "
            f"{sorted(overlap)[:20]}"
        )

    if len(warm_indices) != (
        WARMSTART_WRONG + WARMSTART_CORRECT
    ):
        raise RuntimeError(
            "Duplicate source examples inside warm-start pool."
        )

    if len(dpo_indices) != (
        len(dpo_wrong) + len(dpo_correct)
    ):
        raise RuntimeError(
            "Duplicate source examples inside DPO pool."
        )

    if len(warm_indices | dpo_indices) != 10178:
        raise RuntimeError(
            "Source allocation does not cover all training examples."
        )

    print("\nWarm-start / DPO disjointness: PASS")
    print("Full train coverage            : PASS")

    # --------------------------------------------------------------
    # Build warm-start SFT dataset
    # --------------------------------------------------------------

    warmstart = []

    for row in warm_wrong:

        warmstart.append(
            {
                "train_index": row["train_index"],
                "id": row["id"],
                "prompt": build_prompt(
                    row["question"],
                    row["options"],
                ),
                "completion": ABSTAIN,
                "type": "abstain",
                "source_oof_confidence": row["confidence"],
                "source_oof_fold": row["oof_fold"],
            }
        )

    for row in warm_correct:

        warmstart.append(
            {
                "train_index": row["train_index"],
                "id": row["id"],
                "prompt": build_prompt(
                    row["question"],
                    row["options"],
                ),
                "completion": ANSWER_TEMPLATE.format(
                    row["answer_idx"]
                ),
                "type": "answer",
                "source_oof_confidence": row["confidence"],
                "source_oof_fold": row["oof_fold"],
            }
        )

    rng.shuffle(warmstart)

    # --------------------------------------------------------------
    # DPO candidate source pools
    # --------------------------------------------------------------

    dpo_pool_output = {
        "protocol": "clean_oof_v1",
        "seed": SEED,

        "description": (
            "Disjoint OOF-derived source pools. "
            "No DPO pair ratio has been selected yet."
        ),

        "wrong_pool": [
            source_record(x)
            for x in dpo_wrong
        ],

        "correct_pool": [
            source_record(x)
            for x in dpo_correct
        ],
    }

    # --------------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------------

    wrong_conf = np.array(
        [x["confidence"] for x in dpo_wrong],
        dtype=float,
    )

    correct_conf = np.array(
        [x["confidence"] for x in dpo_correct],
        dtype=float,
    )

    print("\nDPO candidate-pool confidence:")
    print(
        f"  Wrong pool mean   : "
        f"{wrong_conf.mean():.4f}"
    )
    print(
        f"  Correct pool mean : "
        f"{correct_conf.mean():.4f}"
    )

    print("\nWarm-start composition:")
    print(
        f"  abstain : {WARMSTART_WRONG}"
    )
    print(
        f"  answer  : {WARMSTART_CORRECT}"
    )
    print(
        f"  total   : {len(warmstart)}"
    )
    print(
        f"  abstain fraction : "
        f"{WARMSTART_WRONG / len(warmstart):.4f}"
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        WARMSTART_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "protocol": "clean_oof_v1",
                "seed": SEED,
                "n_examples": len(warmstart),
                "n_abstain": WARMSTART_WRONG,
                "n_answer": WARMSTART_CORRECT,
                "examples": warmstart,
            },
            f,
            indent=2,
        )

    with open(
        DPO_POOL_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dpo_pool_output,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(WARMSTART_FILE)
    print(DPO_POOL_FILE)

    print("\nSOURCE POOL BUILD: PASS")


if __name__ == "__main__":
    main()
