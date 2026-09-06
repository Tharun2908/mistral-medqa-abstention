"""
bootstrap_correct_only_dev.py
-----------------------------

Final paired-bootstrap extension including the correct-only SFT control.

No thresholds are re-selected here.

Frozen thresholds are loaded from:
1. dev_full_completion_paired_bootstrap.json
   - original SFT
   - continue-SFT
   - DPO 2:1
   - supervised 5-way

2. correct_only_sft_step1000_dev_calibration.json
   - correct-only SFT

Metrics are evaluated on the same 1,272 MedQA DEV questions.

Bootstrap:
    10,000 paired resamples
    same question indices across models
    fixed deployment decisions from the frozen thresholds

Primary comparisons:
    correct_only - supervised_5way
    correct_only - continue_sft
    correct_only - original_sft
    correct_only - dpo_2to1

Official test is never touched.
"""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
)

EVAL_ROOT = (
    ROOT
    / "warmstart_eval"
)

PREVIOUS_BOOTSTRAP = (
    ROOT
    / "dev_full_completion_paired_bootstrap.json"
)

CORRECT_ONLY_CALIBRATION = (
    ROOT
    / "correct_only_sft_step1000_dev_calibration.json"
)

OUTPUT_FILE = (
    ROOT
    / "dev_correct_only_paired_bootstrap.json"
)


MODEL_FILES = {
    "original_sft":
        EVAL_ROOT / "original_sft.json",

    "continue_sft":
        EVAL_ROOT / "continue_sft_step1000.json",

    "correct_only":
        EVAL_ROOT / "correct_only_sft_step1000.json",

    "dpo_2to1":
        EVAL_ROOT / "dpo_ratio_2to1.json",

    "supervised_5way":
        EVAL_ROOT / "supervised_5way.json",
}


TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]

ANSWER_LABELS = [
    "A",
    "B",
    "C",
    "D",
]

N_BOOTSTRAP = 10_000

BOOTSTRAP_SEED = 20260906


PAIRWISE = [
    (
        "correct_only",
        "supervised_5way",
    ),
    (
        "correct_only",
        "continue_sft",
    ),
    (
        "correct_only",
        "original_sft",
    ),
    (
        "correct_only",
        "dpo_2to1",
    ),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def softmax(values):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x - np.max(x)

    e = np.exp(x)

    return e / e.sum()


def answer_confidence(row):

    scores = row["scores"]

    vals = [
        float(scores[label])
        for label in ANSWER_LABELS
    ]

    return float(
        np.max(
            softmax(vals)
        )
    )


def abstention_margin(row):

    scores = row["scores"]

    best_answer = max(
        float(scores[label])
        for label in ANSWER_LABELS
    )

    return (
        float(scores["E"])
        - best_answer
    )


def get_correct(row):

    return int(
        bool(
            row["wouldbe_correct"]
        )
    )


def load_eval(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data.get("split") != "dev":
        raise RuntimeError(
            f"{path}: expected DEV split"
        )

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"{path}: expected 1272 rows, "
            f"got {len(rows)}"
        )

    return rows


def metrics(
    correct,
    answered,
):

    n = len(correct)

    n_answered = int(
        answered.sum()
    )

    n_abstained = (
        n - n_answered
    )

    if n_answered == 0:

        return {
            "coverage": 0.0,
            "answered_accuracy": np.nan,
            "dataset_wrong_rate": 0.0,
            "utility": 0.3,
        }

    n_correct = int(
        correct[
            answered
        ].sum()
    )

    n_wrong = (
        n_answered
        - n_correct
    )

    return {
        "coverage":
            float(
                n_answered / n
            ),

        "answered_accuracy":
            float(
                n_correct
                / n_answered
            ),

        "dataset_wrong_rate":
            float(
                n_wrong / n
            ),

        "utility":
            float(
                (
                    n_correct * 1.0
                    + n_abstained * 0.3
                    + n_wrong * -2.0
                )
                / n
            ),
    }


def ci95(values):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    return (
        float(
            np.percentile(
                x,
                2.5,
            )
        ),
        float(
            np.percentile(
                x,
                97.5,
            )
        ),
    )


# ---------------------------------------------------------------------
# Frozen thresholds
# ---------------------------------------------------------------------

def load_frozen_thresholds():

    with open(
        PREVIOUS_BOOTSTRAP,
        "r",
        encoding="utf-8",
    ) as f:
        previous = json.load(f)

    thresholds = {}

    for target in TARGET_COVERAGES:

        key = str(target)

        thresholds[target] = {
            "original_sft":
                float(
                    previous[
                        "thresholds"
                    ][key][
                        "original_sft"
                    ]
                ),

            "continue_sft":
                float(
                    previous[
                        "thresholds"
                    ][key][
                        "continue_sft"
                    ]
                ),

            "dpo_2to1":
                float(
                    previous[
                        "thresholds"
                    ][key][
                        "dpo_2to1"
                    ]
                ),

            "supervised_5way":
                float(
                    previous[
                        "thresholds"
                    ][key][
                        "supervised_5way"
                    ]
                ),
        }

    with open(
        CORRECT_ONLY_CALIBRATION,
        "r",
        encoding="utf-8",
    ) as f:
        correct_only = json.load(f)

    op_by_target = {
        float(
            op["target_coverage"]
        ): op
        for op
        in correct_only[
            "operating_points"
        ]
    }

    for target in TARGET_COVERAGES:

        thresholds[
            target
        ][
            "correct_only"
        ] = float(
            op_by_target[
                target
            ][
                "threshold"
            ]
        )

    return thresholds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 92)
    print("CORRECT-ONLY SFT — FINAL PAIRED BOOTSTRAP")
    print("=" * 92)

    rows_by_model = {
        name:
            load_eval(path)

        for name, path
        in MODEL_FILES.items()
    }

    # -------------------------------------------------------------
    # Exact alignment check
    # -------------------------------------------------------------

    reference_ids = [
        row["id"]
        for row
        in rows_by_model[
            "original_sft"
        ]
    ]

    for name, rows in (
        rows_by_model.items()
    ):

        ids = [
            row["id"]
            for row in rows
        ]

        if ids != reference_ids:

            raise RuntimeError(
                f"Alignment failed: {name}"
            )

    print(
        "\nAlignment check: PASS using id"
    )

    thresholds = (
        load_frozen_thresholds()
    )

    # -------------------------------------------------------------
    # Build arrays
    # -------------------------------------------------------------

    model_data = {}

    for name, rows in (
        rows_by_model.items()
    ):

        correct = np.asarray(
            [
                get_correct(row)
                for row in rows
            ],
            dtype=int,
        )

        if name in {
            "original_sft",
            "continue_sft",
            "correct_only",
        }:

            score = np.asarray(
                [
                    answer_confidence(row)
                    for row in rows
                ],
                dtype=float,
            )

            direction = "high"

        else:

            score = np.asarray(
                [
                    abstention_margin(row)
                    for row in rows
                ],
                dtype=float,
            )

            direction = "low"

        model_data[name] = {
            "correct":
                correct,

            "score":
                score,

            "direction":
                direction,
        }

    print(
        "\nWould-be answer accuracy:"
    )

    for name, d in (
        model_data.items()
    ):

        print(
            f"  {name:20}"
            f"{d['correct'].mean():.4f}"
        )

    # -------------------------------------------------------------
    # Freeze decisions
    # -------------------------------------------------------------

    decisions = {}

    observed = {}

    print(
        "\n" + "=" * 92
    )

    print(
        "FROZEN DEV OPERATING POINTS"
    )

    print(
        "=" * 92
    )

    for target in (
        TARGET_COVERAGES
    ):

        decisions[target] = {}
        observed[target] = {}

        print(
            f"\nTarget coverage: "
            f"{target:.0%}"
        )

        print(
            f"{'model':20}"
            f"{'threshold':>14}"
            f"{'coverage':>11}"
            f"{'ans_acc':>11}"
            f"{'wrong':>11}"
            f"{'utility':>11}"
        )

        print(
            "-" * 79
        )

        for name, d in (
            model_data.items()
        ):

            threshold = (
                thresholds[
                    target
                ][name]
            )

            if (
                d["direction"]
                == "high"
            ):

                answered = (
                    d["score"]
                    >= threshold
                )

            else:

                answered = (
                    d["score"]
                    <= threshold
                )

            decisions[
                target
            ][name] = answered

            m = metrics(
                d["correct"],
                answered,
            )

            observed[
                target
            ][name] = m

            print(
                f"{name:20}"
                f"{threshold:14.6f}"
                f"{m['coverage']:11.4f}"
                f"{m['answered_accuracy']:11.4f}"
                f"{m['dataset_wrong_rate']:11.4f}"
                f"{m['utility']:11.4f}"
            )

    # -------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------

    rng = (
        np.random.default_rng(
            BOOTSTRAP_SEED
        )
    )

    n = 1272

    bootstrap = {
        target: {
            f"{a}_minus_{b}": {
                "answered_accuracy": [],
                "dataset_wrong_rate": [],
                "utility": [],
            }
            for a, b
            in PAIRWISE
        }
        for target
        in TARGET_COVERAGES
    }

    print(
        "\nRunning "
        f"{N_BOOTSTRAP:,} paired bootstrap replicates..."
    )

    for _ in range(
        N_BOOTSTRAP
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        for target in (
            TARGET_COVERAGES
        ):

            rep_metrics = {}

            for name, d in (
                model_data.items()
            ):

                rep_metrics[
                    name
                ] = metrics(
                    d["correct"][idx],

                    decisions[
                        target
                    ][name][idx],
                )

            for a, b in PAIRWISE:

                key = (
                    f"{a}_minus_{b}"
                )

                for metric_name in [
                    "answered_accuracy",
                    "dataset_wrong_rate",
                    "utility",
                ]:

                    bootstrap[
                        target
                    ][key][
                        metric_name
                    ].append(

                        rep_metrics[
                            a
                        ][metric_name]

                        -

                        rep_metrics[
                            b
                        ][metric_name]
                    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 92
    )

    print(
        "PAIRED DIFFERENCES — ANSWERED ACCURACY"
    )

    print(
        "Positive = first model has higher answered accuracy"
    )

    print(
        "=" * 92
    )

    output_pairwise = {}

    for target in (
        TARGET_COVERAGES
    ):

        print(
            f"\nCoverage target "
            f"{target:.0%}"
        )

        output_pairwise[
            str(target)
        ] = {}

        for a, b in PAIRWISE:

            key = (
                f"{a}_minus_{b}"
            )

            point = (
                observed[
                    target
                ][a][
                    "answered_accuracy"
                ]
                -
                observed[
                    target
                ][b][
                    "answered_accuracy"
                ]
            )

            values = (
                bootstrap[
                    target
                ][key][
                    "answered_accuracy"
                ]
            )

            lo, hi = (
                ci95(values)
            )

            p_positive = float(
                np.mean(
                    np.asarray(
                        values
                    )
                    > 0
                )
            )

            print(
                f"  {a:20} - "
                f"{b:20} "
                f"delta={point:+.4f}  "
                f"95% CI "
                f"[{lo:+.4f}, {hi:+.4f}]  "
                f"P(delta>0)="
                f"{p_positive:.3f}"
            )

            output_pairwise[
                str(target)
            ][key] = {
                "point":
                    float(point),

                "ci95_low":
                    lo,

                "ci95_high":
                    hi,

                "bootstrap_probability_positive":
                    p_positive,
            }

    # -------------------------------------------------------------
    # Also print wrong-rate comparison for correct-only vs 5-way.
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 92
    )

    print(
        "CORRECT-ONLY MINUS SUPERVISED 5-WAY — DATASET WRONG RATE"
    )

    print(
        "Negative = correct-only makes fewer dataset-level errors"
    )

    print(
        "=" * 92
    )

    for target in (
        TARGET_COVERAGES
    ):

        key = (
            "correct_only_minus_"
            "supervised_5way"
        )

        point = (
            observed[
                target
            ][
                "correct_only"
            ][
                "dataset_wrong_rate"
            ]
            -
            observed[
                target
            ][
                "supervised_5way"
            ][
                "dataset_wrong_rate"
            ]
        )

        lo, hi = ci95(
            bootstrap[
                target
            ][key][
                "dataset_wrong_rate"
            ]
        )

        print(
            f"{target:.0%}: "
            f"delta={point:+.4f}  "
            f"95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]"
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    output = {
        "protocol":
            "correct_only_final_paired_bootstrap_v1",

        "split":
            "dev",

        "n_dev":
            1272,

        "bootstrap_replicates":
            N_BOOTSTRAP,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "threshold_policy":
            (
                "reuse previously frozen "
                "full-dev numeric thresholds; "
                "no threshold selection inside bootstrap"
            ),

        "would_be_accuracy": {
            name:
                float(
                    d["correct"].mean()
                )

            for name, d
            in model_data.items()
        },

        "thresholds": {
            str(target): {
                name:
                    float(value)

                for name, value
                in thresholds[
                    target
                ].items()
            }

            for target
            in TARGET_COVERAGES
        },

        "observed": {
            str(target):
                observed[target]

            for target
            in TARGET_COVERAGES
        },

        "pairwise_answered_accuracy":
            output_pairwise,

        "official_test_used":
            False,
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
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nCORRECT-ONLY PAIRED BOOTSTRAP: PASS"
    )


if __name__ == "__main__":
    main()
