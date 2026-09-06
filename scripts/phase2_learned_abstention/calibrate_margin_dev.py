"""
calibrate_margin_dev.py
-----------------------

Calibrate learned-abstention models on official MedQA DEV using the
abstention margin:

    margin = score(E) - max(score(A), score(B), score(C), score(D))

Interpretation:
    margin > 0  -> natural E/abstain decision
    margin <= 0 -> natural answer decision

For fixed target coverage:
    answer examples with the LOWEST margins.

No correctness labels are used to choose thresholds.
Correctness is used only after threshold selection to report performance.

Models:
    - frozen DPO 2:1
    - supervised 5-way baseline

Also prints the already-frozen post-hoc SFT results for matched-coverage
comparison.

Official MedQA test is never touched.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart_eval"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
)

SFT_CALIBRATION = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "phase1_sft"
    / "sft_dev_calibration.json"
)

MODELS = {
    "dpo_ratio_2to1": (
        EVAL_ROOT
        / "dpo_ratio_2to1.json"
    ),

    "supervised_5way": (
        EVAL_ROOT
        / "supervised_5way.json"
    ),
}

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]

ANSWER_SET = [
    "A",
    "B",
    "C",
    "D",
]


def compute_margin(row):

    scores = row["scores"]

    answer_max = max(
        float(scores[label])
        for label in ANSWER_SET
    )

    e_score = float(
        scores["E"]
    )

    return (
        e_score
        - answer_max
    )


def calibrate_model(
    model_name,
    input_file,
):

    with open(
        input_file,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data.get("split") != "dev":
        raise RuntimeError(
            f"{model_name}: expected dev split, "
            f"got {data.get('split')}"
        )

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"{model_name}: expected 1272 rows, "
            f"got {len(rows)}"
        )

    # --------------------------------------------------------------
    # Margins
    # --------------------------------------------------------------

    margins = np.array(
        [
            compute_margin(row)
            for row in rows
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(margins)
    ):
        raise RuntimeError(
            f"{model_name}: non-finite margins."
        )

    wrong = np.array(
        [
            int(
                not row[
                    "wouldbe_correct"
                ]
            )
            for row in rows
        ],
        dtype=int,
    )

    margin_auroc = roc_auc_score(
        wrong,
        margins,
    )

    mean_correct = float(
        margins[
            wrong == 0
        ].mean()
    )

    mean_wrong = float(
        margins[
            wrong == 1
        ].mean()
    )

    # --------------------------------------------------------------
    # Verify natural decision equivalence
    # --------------------------------------------------------------

    natural_from_margin = (
        margins > 0
    )

    natural_from_saved = np.array(
        [
            bool(row["abstain"])
            for row in rows
        ],
        dtype=bool,
    )

    mismatches = int(
        np.sum(
            natural_from_margin
            != natural_from_saved
        )
    )

    if mismatches != 0:
        raise RuntimeError(
            f"{model_name}: margin decision "
            f"mismatch on {mismatches} rows."
        )

    natural_coverage = float(
        np.mean(
            margins <= 0
        )
    )

    # --------------------------------------------------------------
    # Coverage calibration
    # --------------------------------------------------------------

    sorted_margin = np.sort(
        margins
    )

    calibration = []

    for target in TARGET_COVERAGES:

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

        # Lowest margins = strongest answer preference.
        threshold = float(
            sorted_margin[
                k - 1
            ]
        )

        answered_indices = [
            i
            for i, margin
            in enumerate(margins)
            if margin <= threshold
        ]

        n_answered = len(
            answered_indices
        )

        actual_coverage = (
            n_answered
            / len(rows)
        )

        n_correct = sum(
            int(
                rows[i][
                    "wouldbe_correct"
                ]
            )
            for i in answered_indices
        )

        n_wrong = (
            n_answered
            - n_correct
        )

        answered_accuracy = (
            n_correct
            / n_answered
        )

        dataset_wrong_rate = (
            n_wrong
            / len(rows)
        )

        n_abstained = (
            len(rows)
            - n_answered
        )

        mean_utility = (
            (
                n_correct * 1.0
                + n_abstained * 0.3
                + n_wrong * -2.0
            )
            / len(rows)
        )

        calibration.append(
            {
                "target_coverage":
                    target,

                "margin_threshold":
                    threshold,

                "rule":
                    (
                        "answer_if_"
                        "abstention_margin_lte_"
                        "threshold"
                    ),

                "actual_coverage":
                    actual_coverage,

                "n_answered":
                    n_answered,

                "n_abstained":
                    n_abstained,

                "answered_accuracy":
                    answered_accuracy,

                "dataset_wrong_rate":
                    dataset_wrong_rate,

                "mean_utility":
                    mean_utility,
            }
        )

    output = {
        "protocol":
            "clean_margin_dev_calibration_v1",

        "split":
            "dev",

        "model":
            model_name,

        "score_definition":
            (
                "score_E - max("
                "score_A, score_B, "
                "score_C, score_D)"
            ),

        "selection": {
            "uses_correctness_labels":
                False,

            "answer_rule":
                (
                    "abstention_margin "
                    "<= threshold"
                ),

            "test_policy":
                (
                    "apply exact frozen "
                    "numeric thresholds "
                    "unchanged to test"
                ),
        },

        "diagnostic": {
            "mean_margin_correct":
                mean_correct,

            "mean_margin_wrong":
                mean_wrong,

            "auroc_margin_as_wrongness":
                margin_auroc,

            "natural_coverage":
                natural_coverage,

            "natural_decision_mismatches":
                mismatches,
        },

        "calibration":
            calibration,
    }

    output_file = (
        OUTPUT_ROOT
        / f"{model_name}_margin_dev_calibration.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    return output, output_file


def load_sft():

    with open(
        SFT_CALIBRATION,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if "operating_points" in data:
        return data["operating_points"]

    raise RuntimeError(
        "Could not find SFT operating points."
    )


def main():

    print("=" * 80)
    print("DEV ABSTENTION-MARGIN CALIBRATION")
    print("=" * 80)

    results = {}

    for name, path in MODELS.items():

        output, output_file = (
            calibrate_model(
                name,
                path,
            )
        )

        results[name] = output

        d = output[
            "diagnostic"
        ]

        print(
            f"\n{name}"
        )

        print(
            f"  Mean margin | correct : "
            f"{d['mean_margin_correct']:.6f}"
        )

        print(
            f"  Mean margin | wrong   : "
            f"{d['mean_margin_wrong']:.6f}"
        )

        print(
            f"  Margin AUROC -> wrong : "
            f"{d['auroc_margin_as_wrongness']:.4f}"
        )

        print(
            f"  Natural coverage      : "
            f"{d['natural_coverage']:.4f}"
        )

        print(
            f"  Decision check        : PASS"
        )

        print(
            f"  Saved -> {output_file}"
        )

    # --------------------------------------------------------------
    # Frozen thresholds
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FROZEN MARGIN THRESHOLDS"
    )

    print(
        "=" * 80
    )

    for name, output in (
        results.items()
    ):

        print(
            f"\n{name}"
        )

        for row in (
            output["calibration"]
        ):

            print(
                f"  {row['target_coverage']:.0%} "
                f"coverage -> "
                f"threshold="
                f"{row['margin_threshold']:.6f}  "
                f"actual="
                f"{row['actual_coverage']:.2%}  "
                f"acc="
                f"{row['answered_accuracy']:.2%}  "
                f"wrong="
                f"{row['dataset_wrong_rate']:.2%}  "
                f"utility="
                f"{row['mean_utility']:.4f}"
            )

    # --------------------------------------------------------------
    # Matched-coverage comparison
    # --------------------------------------------------------------

    sft = load_sft()

    sft_by_target = {
        round(
            float(
                row["target_coverage"]
            ),
            2,
        ):
        row
        for row in sft
    }

    dpo_by_target = {
        round(
            row["target_coverage"],
            2,
        ):
        row
        for row in results[
            "dpo_ratio_2to1"
        ]["calibration"]
    }

    sup_by_target = {
        round(
            row["target_coverage"],
            2,
        ):
        row
        for row in results[
            "supervised_5way"
        ]["calibration"]
    }

    print(
        "\n" + "=" * 80
    )

    print(
        "MATCHED-COVERAGE DEV COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        f"{'cov':>6}"
        f"{'SFT acc':>12}"
        f"{'DPO acc':>12}"
        f"{'SUP acc':>12}"
        f"{'SFT wrong':>12}"
        f"{'DPO wrong':>12}"
        f"{'SUP wrong':>12}"
    )

    print(
        "-" * 78
    )

    for target in (
        TARGET_COVERAGES
    ):

        key = round(
            target,
            2,
        )

        s = sft_by_target[key]
        d = dpo_by_target[key]
        u = sup_by_target[key]

        print(
            f"{target:>6.0%}"
            f"{s['answered_accuracy']:>12.4f}"
            f"{d['answered_accuracy']:>12.4f}"
            f"{u['answered_accuracy']:>12.4f}"
            f"{s['dataset_wrong_rate']:>12.4f}"
            f"{d['dataset_wrong_rate']:>12.4f}"
            f"{u['dataset_wrong_rate']:>12.4f}"
        )

    print(
        "\nMARGIN CALIBRATION: PASS"
    )


if __name__ == "__main__":
    main()
