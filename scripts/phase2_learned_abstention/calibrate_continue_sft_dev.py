"""
Calibrate the matched-compute continue-SFT control on official MedQA DEV.

Score:
    confidence = softmax over A/B/C/D full-completion scores,
                 taking the maximum probability.

Selection:
    Higher confidence -> answer.
    Lower confidence  -> abstain.

Thresholds are chosen by confidence rank only.
Correctness labels are used only AFTER threshold selection to report metrics.

Official test is never touched.
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
    / "continue_sft_step1000.json"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "continue_sft_step1000_dev_calibration.json"
)

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]

ANSWER_LABELS = ["A", "B", "C", "D"]


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def get_answer_confidence(row):
    """
    Convert A-D mean completion log-scores into a normalized
    answer-only probability distribution and return max probability.
    """

    scores = row["scores"]

    answer_scores = [
        float(scores[label])
        for label in ANSWER_LABELS
    ]

    probs = softmax(answer_scores)

    return float(np.max(probs))


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("split") != "dev":
        raise RuntimeError(
            f"Expected dev split, got {data.get('split')}"
        )

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"Expected 1272 dev examples, got {len(rows)}"
        )

    confidences = np.array(
        [
            get_answer_confidence(row)
            for row in rows
        ],
        dtype=float,
    )

    correctness = np.array(
        [
            int(row["wouldbe_correct"])
            for row in rows
        ],
        dtype=int,
    )

    if not np.all(np.isfinite(confidences)):
        raise RuntimeError(
            "Found non-finite confidence values."
        )

    print("=" * 80)
    print("CONTINUE-SFT DEV CONFIDENCE CALIBRATION")
    print("=" * 80)

    print(f"\nExamples              : {len(rows)}")
    print(
        f"Would-be accuracy     : "
        f"{correctness.mean():.4f}"
    )

    print(
        f"Mean confidence       : "
        f"{confidences.mean():.4f}"
    )

    print(
        f"Mean conf | correct   : "
        f"{confidences[correctness == 1].mean():.4f}"
    )

    print(
        f"Mean conf | wrong     : "
        f"{confidences[correctness == 0].mean():.4f}"
    )

    # Highest confidence first.
    sorted_conf = np.sort(confidences)[::-1]

    operating_points = []

    print("\n" + "=" * 80)
    print("FROZEN CONTINUE-SFT THRESHOLDS")
    print("=" * 80)

    for target in TARGET_COVERAGES:

        k = int(
            round(
                target * len(rows)
            )
        )

        k = max(
            1,
            min(k, len(rows)),
        )

        threshold = float(
            sorted_conf[k - 1]
        )

        # Literal deployment rule.
        answered = (
            confidences >= threshold
        )

        n_answered = int(
            answered.sum()
        )

        actual_coverage = (
            n_answered / len(rows)
        )

        n_correct = int(
            correctness[answered].sum()
        )

        n_wrong = (
            n_answered - n_correct
        )

        n_abstained = (
            len(rows) - n_answered
        )

        answered_accuracy = (
            n_correct / n_answered
            if n_answered > 0
            else float("nan")
        )

        dataset_wrong_rate = (
            n_wrong / len(rows)
        )

        mean_utility = (
            (
                n_correct * 1.0
                + n_abstained * 0.3
                + n_wrong * -2.0
            )
            / len(rows)
        )

        result = {
            "target_coverage": target,
            "confidence_threshold": threshold,
            "rule": "answer_if_confidence_gte_threshold",
            "actual_coverage": actual_coverage,
            "n_answered": n_answered,
            "n_abstained": n_abstained,
            "answered_accuracy": answered_accuracy,
            "dataset_wrong_rate": dataset_wrong_rate,
            "mean_utility": mean_utility,
        }

        operating_points.append(result)

        print(
            f"{target:.0%} coverage -> "
            f"threshold={threshold:.6f}  "
            f"actual={actual_coverage:.2%}  "
            f"acc={answered_accuracy:.2%}  "
            f"wrong={dataset_wrong_rate:.2%}  "
            f"utility={mean_utility:.4f}"
        )

    output = {
        "protocol":
            "clean_continue_sft_dev_calibration_v1",

        "selection_split":
            "dev",

        "n_dev":
            len(rows),

        "model":
            "continue_sft_step1000",

        "would_be_accuracy":
            float(correctness.mean()),

        "score_definition":
            (
                "max softmax probability over "
                "A/B/C/D full-completion mean log-scores"
            ),

        "threshold_selection":
            "confidence_rank_only_no_label_based_selection",

        "test_policy":
            (
                "apply exact frozen numeric thresholds "
                "unchanged to official test"
            ),

        "operating_points":
            operating_points,
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

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\nCONTINUE-SFT CALIBRATION: PASS")


if __name__ == "__main__":
    main()
