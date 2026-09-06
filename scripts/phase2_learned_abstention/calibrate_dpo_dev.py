"""
calibrate_dpo_dev.py
--------------------

Freeze P(E) thresholds for the selected clean DPO model using MedQA DEV only.

Selected model:
    warm-start LR = 5e-6
    DPO ratio     = 2:1
    DPO LR        = 5e-6
    beta          = 0.1
    epochs        = 2
    seed          = 42

Threshold selection:
    - Uses ONLY the ranking/distribution of P(E).
    - Does NOT use correctness labels to choose thresholds.
    - Answer when P(E) <= threshold.
    - Abstain otherwise.

Target coverages:
    30%, 40%, 50%, 60%

The resulting numeric thresholds will later be applied literally and unchanged
to the locked MedQA test set.
"""

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart_eval"
    / "dpo_ratio_2to1.json"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "dpo_ratio_2to1_dev_calibration.json"
)

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]


def main():

    print("=" * 72)
    print("DPO DEV P(E) CALIBRATION")
    print("=" * 72)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data.get("split") != "dev":
        raise RuntimeError(
            f"Expected DEV artifact, "
            f"got split={data.get('split')}"
        )

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"Expected 1272 dev rows, "
            f"got {len(rows)}"
        )

    pe = np.array(
        [
            float(row["p_abstain"])
            for row in rows
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(pe)):
        raise RuntimeError(
            "Non-finite P(E) values found."
        )

    print(
        f"\nLoaded dev examples : "
        f"{len(rows)}"
    )

    print(
        f"Mean P(E)           : "
        f"{pe.mean():.6f}"
    )

    print(
        f"Min P(E)            : "
        f"{pe.min():.6f}"
    )

    print(
        f"Max P(E)            : "
        f"{pe.max():.6f}"
    )

    sorted_pe = np.sort(pe)

    calibration = []

    print(
        "\n" + "-" * 72
    )

    print(
        "FROZEN COVERAGE THRESHOLDS"
    )

    print(
        "-" * 72
    )

    for target in TARGET_COVERAGES:

        # Number of examples we intend to answer.
        k = int(
            round(
                target
                * len(rows)
            )
        )

        k = max(
            1,
            min(
                k,
                len(rows),
            ),
        )

        # Rank-derived threshold.
        #
        # No correctness labels are involved.
        threshold = float(
            sorted_pe[k - 1]
        )

        # Deployment rule:
        # answer if P(E) <= threshold
        answered = [
            row
            for row in rows
            if float(
                row["p_abstain"]
            ) <= threshold
        ]

        n_answered = len(
            answered
        )

        actual_coverage = (
            n_answered
            / len(rows)
        )

        n_correct = sum(
            int(
                row["wouldbe_correct"]
            )
            for row in answered
        )

        n_wrong = (
            n_answered
            - n_correct
        )

        answered_accuracy = (
            n_correct
            / n_answered
            if n_answered
            else float("nan")
        )

        dataset_wrong_rate = (
            n_wrong
            / len(rows)
        )

        n_abstained = (
            len(rows)
            - n_answered
        )

        # Same fixed utility used throughout project:
        #
        # correct = +1
        # abstain = +0.3
        # wrong   = -2
        mean_utility = (
            (
                n_correct * 1.0
                + n_abstained * 0.3
                + n_wrong * -2.0
            )
            / len(rows)
        )

        record = {
            "target_coverage":
                target,

            "pe_threshold":
                threshold,

            "rule":
                "answer_if_p_abstain_lte_threshold",

            "actual_coverage":
                actual_coverage,

            "n_answered":
                n_answered,

            "n_abstained":
                n_abstained,

            "n_correct_answered":
                n_correct,

            "n_wrong_answered":
                n_wrong,

            "answered_accuracy":
                answered_accuracy,

            "dataset_wrong_rate":
                dataset_wrong_rate,

            "mean_utility":
                mean_utility,
        }

        calibration.append(
            record
        )

        print(
            f"\nTarget coverage      : "
            f"{target:.0%}"
        )

        print(
            f"P(E) threshold       : "
            f"{threshold:.6f}"
        )

        print(
            f"Actual coverage      : "
            f"{actual_coverage:.2%}"
        )

        print(
            f"Answered examples    : "
            f"{n_answered}"
        )

        print(
            f"Answered accuracy    : "
            f"{answered_accuracy:.2%}"
        )

        print(
            f"Dataset wrong rate   : "
            f"{dataset_wrong_rate:.2%}"
        )

        print(
            f"Mean utility         : "
            f"{mean_utility:.4f}"
        )

    # --------------------------------------------------------------
    # Save frozen calibration artifact
    # --------------------------------------------------------------

    output = {
        "protocol":
            "clean_dpo_dev_calibration_v1",

        "split":
            "dev",

        "model_selection": {
            "warmstart_lr":
                5e-6,

            "dpo_ratio":
                "2:1",

            "dpo_lr":
                5e-6,

            "beta":
                0.1,

            "epochs":
                2,

            "seed":
                42,
        },

        "threshold_selection": {
            "uses_correctness_labels":
                False,

            "selection_basis":
                (
                    "rank of P(E) on "
                    "official MedQA dev"
                ),

            "deployment_rule":
                (
                    "answer if "
                    "P(E) <= threshold"
                ),

            "test_policy":
                (
                    "apply these exact "
                    "numeric thresholds "
                    "unchanged to test"
                ),
        },

        "n_dev":
            len(rows),

        "target_coverages":
            TARGET_COVERAGES,

        "calibration":
            calibration,
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
        "\n" + "=" * 72
    )

    print(
        "DPO DEV CALIBRATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Saved frozen thresholds ->\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
