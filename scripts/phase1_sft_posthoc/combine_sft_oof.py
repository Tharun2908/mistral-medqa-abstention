"""
combine_sft_oof.py
------------------

Combine the five clean MedQA SFT OOF prediction files.

Checks:
    - all 5 fold files exist
    - total predictions = 10,178
    - every original train_index appears exactly once
    - no duplicate train indices
    - no missing train indices
    - each prediction belongs to the expected fold
    - outputs sorted by original MedQA training index

Output:
    results/clean_protocol/oof_sft/sft_train_oof_predictions.json
"""

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

OOF_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "oof_sft"
)

OUTPUT_FILE = (
    OOF_ROOT
    / "sft_train_oof_predictions.json"
)

N_FOLDS = 5
EXPECTED_TOTAL = 10178


def main():

    print("=" * 72)
    print("COMBINE CLEAN SFT OOF PREDICTIONS")
    print("=" * 72)

    all_predictions = []
    fold_summary = []

    # --------------------------------------------------------------
    # Load each fold
    # --------------------------------------------------------------

    for fold in range(N_FOLDS):

        path = (
            OOF_ROOT
            / f"fold_{fold}"
            / "predictions.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing fold file:\n{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if data["fold"] != fold:
            raise RuntimeError(
                f"Fold metadata mismatch: "
                f"expected {fold}, got {data['fold']}"
            )

        if data.get("smoke"):
            raise RuntimeError(
                f"Fold {fold} is marked as smoke run!"
            )

        rows = data["predictions"]

        # Every prediction in this file should say it belongs
        # to the same fold.
        bad_fold_rows = [
            r for r in rows
            if r["oof_fold"] != fold
        ]

        if bad_fold_rows:
            raise RuntimeError(
                f"Fold {fold} contains predictions "
                f"tagged with another fold."
            )

        calculated_accuracy = np.mean(
            [
                int(r["is_correct"])
                for r in rows
            ]
        )

        stored_accuracy = data["oof_accuracy"]

        if not np.isclose(
            calculated_accuracy,
            stored_accuracy,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Fold {fold} accuracy mismatch: "
                f"stored={stored_accuracy}, "
                f"calculated={calculated_accuracy}"
            )

        fold_summary.append(
            {
                "fold": fold,
                "n_examples": len(rows),
                "oof_accuracy": stored_accuracy,
                "best_checkpoint": data.get(
                    "best_checkpoint"
                ),
                "best_eval_loss": data.get(
                    "best_eval_loss"
                ),
            }
        )

        all_predictions.extend(rows)

        print(
            f"Fold {fold}: "
            f"n={len(rows):4d}  "
            f"accuracy={stored_accuracy:.4f}  "
            f"best_eval_loss="
            f"{data.get('best_eval_loss')}"
        )

    # --------------------------------------------------------------
    # Global integrity
    # --------------------------------------------------------------

    print("\n" + "-" * 72)
    print("GLOBAL INTEGRITY CHECK")
    print("-" * 72)

    total = len(all_predictions)

    print(
        f"Total predictions       : "
        f"{total}"
    )

    if total != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} predictions, "
            f"got {total}"
        )

    indices = [
        int(r["train_index"])
        for r in all_predictions
    ]

    unique_indices = set(indices)

    print(
        f"Unique train indices    : "
        f"{len(unique_indices)}"
    )

    duplicates = (
        total
        - len(unique_indices)
    )

    print(
        f"Duplicate indices       : "
        f"{duplicates}"
    )

    if duplicates != 0:
        from collections import Counter

        counts = Counter(indices)

        dup_ids = [
            idx
            for idx, count in counts.items()
            if count > 1
        ]

        raise RuntimeError(
            f"Duplicate train indices found: "
            f"{dup_ids[:20]}"
        )

    expected_indices = set(
        range(EXPECTED_TOTAL)
    )

    missing = sorted(
        expected_indices
        - unique_indices
    )

    extra = sorted(
        unique_indices
        - expected_indices
    )

    print(
        f"Missing indices         : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected indices      : "
        f"{len(extra)}"
    )

    if missing:
        raise RuntimeError(
            f"Missing train indices: "
            f"{missing[:20]}"
        )

    if extra:
        raise RuntimeError(
            f"Unexpected train indices: "
            f"{extra[:20]}"
        )

    # --------------------------------------------------------------
    # Sort into original MedQA train order
    # --------------------------------------------------------------

    all_predictions.sort(
        key=lambda r: int(
            r["train_index"]
        )
    )

    # Strong final ordering check.
    for expected_idx, row in enumerate(
        all_predictions
    ):

        actual_idx = int(
            row["train_index"]
        )

        if actual_idx != expected_idx:
            raise RuntimeError(
                f"Ordering failed at position "
                f"{expected_idx}: got {actual_idx}"
            )

    print(
        "Original train ordering : PASS"
    )

    # --------------------------------------------------------------
    # Combined metrics
    # --------------------------------------------------------------

    total_correct = sum(
        int(r["is_correct"])
        for r in all_predictions
    )

    combined_accuracy = (
        total_correct / total
    )

    confidences = np.array(
        [
            float(r["confidence"])
            for r in all_predictions
        ]
    )

    correct_conf = np.array(
        [
            float(r["confidence"])
            for r in all_predictions
            if r["is_correct"]
        ]
    )

    wrong_conf = np.array(
        [
            float(r["confidence"])
            for r in all_predictions
            if not r["is_correct"]
        ]
    )

    print("\n" + "-" * 72)
    print("COMBINED OOF SUMMARY")
    print("-" * 72)

    print(
        f"Correct               : "
        f"{total_correct} / {total}"
    )

    print(
        f"OOF accuracy          : "
        f"{combined_accuracy:.4f} "
        f"({combined_accuracy * 100:.2f}%)"
    )

    print(
        f"Mean confidence       : "
        f"{confidences.mean():.4f}"
    )

    print(
        f"Mean conf | correct   : "
        f"{correct_conf.mean():.4f}"
    )

    print(
        f"Mean conf | wrong     : "
        f"{wrong_conf.mean():.4f}"
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output = {
        "protocol": "clean_oof_v1",

        "n_folds": N_FOLDS,

        "n_examples": total,

        "oof_accuracy": combined_accuracy,

        "correct": total_correct,

        "wrong": total - total_correct,

        "mean_confidence": float(
            confidences.mean()
        ),

        "mean_confidence_correct": float(
            correct_conf.mean()
        ),

        "mean_confidence_wrong": float(
            wrong_conf.mean()
        ),

        "fold_summary": fold_summary,

        "integrity": {
            "expected_total": EXPECTED_TOTAL,
            "unique_train_indices":
                len(unique_indices),
            "duplicates": duplicates,
            "missing": len(missing),
            "unexpected": len(extra),
            "original_order_restored": True,
        },

        "predictions": all_predictions,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        "\nIntegrity status       : PASS"
    )

    print(
        f"\nSaved ->\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
