"""
bootstrap_dev_comparison.py
---------------------------

Clean paired-bootstrap comparison on official MedQA DEV.

IMPORTANT
---------
This analysis uses ONE consistent full-completion evaluation space:

Original SFT:
    score = max softmax(A,B,C,D full-completion scores)
    answer if score >= threshold

Continue-SFT:
    score = max softmax(A,B,C,D full-completion scores)
    answer if score >= threshold

DPO 2:1:
    score = E score - max(A,B,C,D scores)
    answer if margin <= threshold

Supervised 5-way:
    score = E score - max(A,B,C,D scores)
    answer if margin <= threshold

Thresholds are selected on the full official dev set using score rank ONLY.
Correctness labels are NOT used to choose thresholds.

Then the selected numeric thresholds are held fixed during 10,000 paired
bootstrap resamples.

The bootstrap therefore estimates uncertainty around the already-selected
dev operating points rather than re-selecting thresholds inside each sample.

Official MedQA test is never touched.
"""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

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

MODEL_FILES = {
    "original_sft": EVAL_ROOT / "original_sft.json",
    "continue_sft": EVAL_ROOT / "continue_sft_step1000.json",
    "dpo_2to1": EVAL_ROOT / "dpo_ratio_2to1.json",
    "supervised_5way": EVAL_ROOT / "supervised_5way.json",
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
    ("continue_sft", "supervised_5way"),
    ("continue_sft", "original_sft"),
    ("continue_sft", "dpo_2to1"),
    ("supervised_5way", "dpo_2to1"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def softmax(values):
    x = np.asarray(values, dtype=float)
    x = x - np.max(x)

    e = np.exp(x)

    return e / e.sum()


def answer_confidence(row):
    """
    Full-completion A-D confidence.
    """

    scores = row["scores"]

    values = [
        float(scores[label])
        for label in ANSWER_LABELS
    ]

    probs = softmax(values)

    return float(np.max(probs))


def abstention_margin(row):
    """
    E score - best A-D score.
    """

    scores = row["scores"]

    best_answer = max(
        float(scores[label])
        for label in ANSWER_LABELS
    )

    e_score = float(scores["E"])

    return (
        e_score
        - best_answer
    )


def get_correct(row):
    return int(
        bool(
            row["wouldbe_correct"]
        )
    )


def load_rows(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        x = json.load(f)

    if x.get("split") != "dev":
        raise RuntimeError(
            f"{path}: expected dev split, "
            f"got {x.get('split')}"
        )

    rows = x["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"{path}: expected 1272 rows, "
            f"got {len(rows)}"
        )

    return rows


# ---------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------

def verify_alignment(all_rows):

    names = list(all_rows.keys())

    reference = all_rows[
        names[0]
    ]

    # Try explicit IDs first.
    candidate_ids = [
        "id",
        "idx",
        "example_id",
        "question_id",
    ]

    for key in candidate_ids:

        if all(
            key in rows[0]
            for rows in all_rows.values()
        ):

            ref_ids = [
                str(row[key])
                for row in reference
            ]

            if len(set(ref_ids)) != len(ref_ids):
                continue

            for name in names[1:]:

                ids = [
                    str(row[key])
                    for row in all_rows[name]
                ]

                if ids != ref_ids:
                    raise RuntimeError(
                        f"Row alignment mismatch "
                        f"for {name} using key={key}"
                    )

            print(
                f"Alignment check: PASS "
                f"using key '{key}'"
            )

            return

    # Otherwise verify same ordering with whatever stable fields exist.
    possible_fields = [
        "question",
        "gold_answer",
        "gold",
        "answer_idx",
    ]

    common_fields = [
        field
        for field in possible_fields
        if all(
            field in rows[0]
            for rows in all_rows.values()
        )
    ]

    if not common_fields:
        raise RuntimeError(
            "Could not verify paired row alignment. "
            "No shared ID or stable example field found."
        )

    for i in range(
        len(reference)
    ):

        for field in common_fields:

            ref_value = reference[i][field]

            for name in names[1:]:

                if (
                    all_rows[name][i][field]
                    != ref_value
                ):
                    raise RuntimeError(
                        f"Alignment mismatch at row {i}, "
                        f"field={field}, model={name}"
                    )

    print(
        "Alignment check: PASS "
        f"using ordered fields {common_fields}"
    )


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def select_threshold(
    scores,
    target_coverage,
    direction,
):

    n = len(scores)

    k = int(
        round(
            target_coverage * n
        )
    )

    k = max(
        1,
        min(k, n),
    )

    if direction == "high":

        ranked = np.sort(
            scores
        )[::-1]

        return float(
            ranked[k - 1]
        )

    if direction == "low":

        ranked = np.sort(
            scores
        )

        return float(
            ranked[k - 1]
        )

    raise ValueError(direction)


def answered_mask(
    scores,
    threshold,
    direction,
):

    if direction == "high":
        return (
            scores >= threshold
        )

    if direction == "low":
        return (
            scores <= threshold
        )

    raise ValueError(direction)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def metrics_from_mask(
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
        correct[answered].sum()
    )

    n_wrong = (
        n_answered
        - n_correct
    )

    coverage = (
        n_answered / n
    )

    answered_accuracy = (
        n_correct
        / n_answered
    )

    dataset_wrong_rate = (
        n_wrong / n
    )

    utility = (
        (
            n_correct * 1.0
            + n_abstained * 0.3
            + n_wrong * -2.0
        )
        / n
    )

    return {
        "coverage":
            float(coverage),

        "answered_accuracy":
            float(answered_accuracy),

        "dataset_wrong_rate":
            float(dataset_wrong_rate),

        "utility":
            float(utility),
    }


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 88)
    print("CLEAN FULL-COMPLETION DEV COMPARISON + PAIRED BOOTSTRAP")
    print("=" * 88)

    all_rows = {
        name: load_rows(path)
        for name, path
        in MODEL_FILES.items()
    }

    verify_alignment(
        all_rows
    )

    # --------------------------------------------------------------
    # Build score/correct arrays.
    # --------------------------------------------------------------

    model_data = {}

    for name, rows in (
        all_rows.items()
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
        }:

            scores = np.asarray(
                [
                    answer_confidence(row)
                    for row in rows
                ],
                dtype=float,
            )

            direction = "high"

            score_name = (
                "full_completion_AD_confidence"
            )

        else:

            scores = np.asarray(
                [
                    abstention_margin(row)
                    for row in rows
                ],
                dtype=float,
            )

            direction = "low"

            score_name = (
                "abstention_margin"
            )

        if not np.all(
            np.isfinite(scores)
        ):
            raise RuntimeError(
                f"{name}: non-finite scores"
            )

        model_data[name] = {
            "scores":
                scores,

            "correct":
                correct,

            "direction":
                direction,

            "score_name":
                score_name,
        }

    # --------------------------------------------------------------
    # Accuracy sanity check.
    # --------------------------------------------------------------

    print("\nWould-be full-completion answer accuracy:")

    for name, d in (
        model_data.items()
    ):

        print(
            f"  {name:20} "
            f"{d['correct'].mean():.4f}"
        )

    # --------------------------------------------------------------
    # Select thresholds on full DEV.
    # --------------------------------------------------------------

    thresholds = {}

    observed = {}

    print(
        "\n" + "=" * 88
    )

    print(
        "FULL-COMPLETION DEV OPERATING POINTS"
    )

    print(
        "=" * 88
    )

    for target in (
        TARGET_COVERAGES
    ):

        thresholds[target] = {}
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
                select_threshold(
                    d["scores"],
                    target,
                    d["direction"],
                )
            )

            thresholds[
                target
            ][name] = threshold

            answered = (
                answered_mask(
                    d["scores"],
                    threshold,
                    d["direction"],
                )
            )

            m = metrics_from_mask(
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

    # --------------------------------------------------------------
    # Freeze decisions before bootstrap.
    # --------------------------------------------------------------

    decision_masks = {}

    for target in (
        TARGET_COVERAGES
    ):

        decision_masks[
            target
        ] = {}

        for name, d in (
            model_data.items()
        ):

            decision_masks[
                target
            ][name] = (
                answered_mask(
                    d["scores"],
                    thresholds[target][name],
                    d["direction"],
                )
            )

    # --------------------------------------------------------------
    # Paired bootstrap.
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 88
    )

    print(
        f"PAIRED BOOTSTRAP: "
        f"{N_BOOTSTRAP:,} REPLICATES"
    )

    print(
        "=" * 88
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = 1272

    bootstrap_results = {
        target: {
            name: {
                "answered_accuracy": [],
                "dataset_wrong_rate": [],
                "utility": [],
            }
            for name in model_data
        }
        for target in TARGET_COVERAGES
    }

    pairwise_results = {
        target: {
            f"{a}_minus_{b}": {
                "answered_accuracy": [],
                "dataset_wrong_rate": [],
                "utility": [],
            }
            for a, b in PAIRWISE
        }
        for target in TARGET_COVERAGES
    }

    for _ in range(
        N_BOOTSTRAP
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        replicate_metrics = {}

        for target in (
            TARGET_COVERAGES
        ):

            replicate_metrics[
                target
            ] = {}

            for name, d in (
                model_data.items()
            ):

                correct_b = (
                    d["correct"][idx]
                )

                answered_b = (
                    decision_masks[
                        target
                    ][name][idx]
                )

                m = metrics_from_mask(
                    correct_b,
                    answered_b,
                )

                replicate_metrics[
                    target
                ][name] = m

                for metric in [
                    "answered_accuracy",
                    "dataset_wrong_rate",
                    "utility",
                ]:

                    bootstrap_results[
                        target
                    ][name][metric].append(
                        m[metric]
                    )

            for a, b in PAIRWISE:

                key = (
                    f"{a}_minus_{b}"
                )

                ma = (
                    replicate_metrics[
                        target
                    ][a]
                )

                mb = (
                    replicate_metrics[
                        target
                    ][b]
                )

                for metric in [
                    "answered_accuracy",
                    "dataset_wrong_rate",
                    "utility",
                ]:

                    pairwise_results[
                        target
                    ][key][metric].append(
                        ma[metric]
                        - mb[metric]
                    )

    # --------------------------------------------------------------
    # Individual model CIs.
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 88
    )

    print(
        "ANSWERED-ACCURACY 95% BOOTSTRAP CIs"
    )

    print(
        "=" * 88
    )

    for target in (
        TARGET_COVERAGES
    ):

        print(
            f"\nCoverage target "
            f"{target:.0%}"
        )

        for name in model_data:

            point = (
                observed[
                    target
                ][name][
                    "answered_accuracy"
                ]
            )

            lo, hi = percentile_ci(
                bootstrap_results[
                    target
                ][name][
                    "answered_accuracy"
                ]
            )

            print(
                f"  {name:20} "
                f"{point:.4f}  "
                f"95% CI "
                f"[{lo:.4f}, {hi:.4f}]"
            )

    # --------------------------------------------------------------
    # Pairwise delta CIs.
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 88
    )

    print(
        "PAIRED DIFFERENCES — ANSWERED ACCURACY"
    )

    print(
        "Positive delta means the first model is better."
    )

    print(
        "=" * 88
    )

    summary_pairwise = {}

    for target in (
        TARGET_COVERAGES
    ):

        print(
            f"\nCoverage target "
            f"{target:.0%}"
        )

        summary_pairwise[
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

            values = np.asarray(
                pairwise_results[
                    target
                ][key][
                    "answered_accuracy"
                ],
                dtype=float,
            )

            values = values[
                np.isfinite(values)
            ]

            lo, hi = (
                percentile_ci(
                    values
                )
            )

            probability_positive = (
                float(
                    np.mean(
                        values > 0
                    )
                )
            )

            print(
                f"  {a:20} - "
                f"{b:20} "
                f"delta={point:+.4f}  "
                f"95% CI "
                f"[{lo:+.4f}, {hi:+.4f}]  "
                f"P(delta>0)="
                f"{probability_positive:.3f}"
            )

            summary_pairwise[
                str(target)
            ][key] = {
                "point_difference":
                    float(point),

                "ci95_low":
                    lo,

                "ci95_high":
                    hi,

                "bootstrap_probability_positive":
                    probability_positive,
            }

    # --------------------------------------------------------------
    # Save everything.
    # --------------------------------------------------------------

    serializable_thresholds = {
        str(target): {
            name:
                float(value)
            for name, value
            in thresholds[
                target
            ].items()
        }
        for target in TARGET_COVERAGES
    }

    output = {
        "protocol":
            "clean_dev_full_completion_paired_bootstrap_v1",

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
                "select thresholds once on full dev "
                "using score rank only; hold exact "
                "thresholds fixed during paired bootstrap"
            ),

        "models": {
            name: {
                "score":
                    d["score_name"],

                "direction":
                    d["direction"],

                "would_be_accuracy":
                    float(
                        d[
                            "correct"
                        ].mean()
                    ),
            }
            for name, d
            in model_data.items()
        },

        "thresholds":
            serializable_thresholds,

        "observed_operating_points": {
            str(target):
                observed[target]
            for target
            in TARGET_COVERAGES
        },

        "pairwise_answered_accuracy":
            summary_pairwise,
    }

    output_file = (
        OUTPUT_ROOT
        / "dev_full_completion_paired_bootstrap.json"
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

    print(
        "\nSaved:"
    )

    print(
        output_file
    )

    print(
        "\nPAIRED BOOTSTRAP: PASS"
    )


if __name__ == "__main__":
    main()
