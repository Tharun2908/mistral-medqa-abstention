"""
calibrate_sft_dev.py
--------------------

Choose SFT selective-prediction thresholds using ONLY the official MedQA
development split.

Threshold selection uses confidence ranks only -- no correctness labels are
used to choose the threshold.

Target coverage levels:
    30%, 40%, 50%, 60%

The resulting thresholds are frozen and later applied unchanged to test.
"""

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "phase1_sft"
    / "sft_dev_predictions.json"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "phase1_sft"
    / "sft_dev_calibration.json"
)

TARGET_COVERAGES = [0.30, 0.40, 0.50, 0.60]


def threshold_for_target_coverage(confidences, target_coverage):
    """
    Pick threshold using confidence values ONLY.

    Example:
        target coverage = 0.50
        -> find the confidence of the 50%-highest-ranked example
        -> answer examples with confidence >= that threshold

    Ties can make achieved coverage slightly different from the target.
    """

    n = len(confidences)

    n_target = int(round(target_coverage * n))
    n_target = max(1, min(n_target, n))

    sorted_conf = np.sort(confidences)[::-1]

    threshold = float(sorted_conf[n_target - 1])

    return threshold


def evaluate_threshold(predictions, threshold):
    answered = [
        p for p in predictions
        if p["confidence"] >= threshold
    ]

    total = len(predictions)
    n_answered = len(answered)

    correct = sum(
        int(p["is_correct"])
        for p in answered
    )

    wrong = n_answered - correct

    coverage = n_answered / total

    answered_accuracy = (
        correct / n_answered
        if n_answered > 0
        else float("nan")
    )

    dataset_wrong_rate = wrong / total

    return {
        "threshold": threshold,
        "coverage": coverage,
        "answered": n_answered,
        "correct": correct,
        "wrong": wrong,
        "answered_accuracy": answered_accuracy,
        "dataset_wrong_rate": dataset_wrong_rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help="Destination JSON; use a separate path to preserve committed results.",
    )
    args = parser.parse_args()
    output_file = args.output

    print("=" * 72)
    print("SFT DEV CALIBRATION — CLEAN PROTOCOL")
    print("=" * 72)

    print(f"\nLoading:\n{INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("split") != "dev":
        raise RuntimeError(
            f"Expected dev predictions, got split={data.get('split')}"
        )

    predictions = data["predictions"]

    if len(predictions) != 1272:
        raise RuntimeError(
            f"Expected 1272 dev examples, got {len(predictions)}"
        )

    confidences = np.array(
        [p["confidence"] for p in predictions],
        dtype=np.float64,
    )

    print(f"\nExamples: {len(predictions)}")
    print(f"Base dev accuracy: {data['accuracy']:.4f}")

    calibration_rows = []

    print("\n" + "-" * 92)
    print(
        f"{'Target':>10} "
        f"{'Threshold':>12} "
        f"{'Actual Cov':>12} "
        f"{'Answered Acc':>14} "
        f"{'Wrong Rate':>12} "
        f"{'N Answered':>12}"
    )
    print("-" * 92)

    for target in TARGET_COVERAGES:

        threshold = threshold_for_target_coverage(
            confidences,
            target,
        )

        metrics = evaluate_threshold(
            predictions,
            threshold,
        )

        row = {
            "target_coverage": target,
            **metrics,
        }

        calibration_rows.append(row)

        print(
            f"{target * 100:>9.0f}% "
            f"{threshold:>12.6f} "
            f"{metrics['coverage'] * 100:>11.2f}% "
            f"{metrics['answered_accuracy'] * 100:>13.2f}% "
            f"{metrics['dataset_wrong_rate'] * 100:>11.2f}% "
            f"{metrics['answered']:>12}"
        )

    output = {
        "protocol": "clean_v1",
        "selection_split": "dev",
        "n_dev": len(predictions),
        "sft_dev_accuracy": data["accuracy"],
        "threshold_selection": (
            "confidence_rank_only_no_label_based_selection"
        ),
        "target_coverages": TARGET_COVERAGES,
        "operating_points": calibration_rows,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)

    print("-" * 92)

    print(f"\nSaved -> {output_file}")

    print(
        "\nThese thresholds are now the frozen SFT thresholds "
        "for future test evaluation."
    )


if __name__ == "__main__":
    main()
