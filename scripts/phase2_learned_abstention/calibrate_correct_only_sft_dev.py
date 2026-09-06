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
    / "correct_only_sft_step1000.json"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "correct_only_sft_step1000_dev_calibration.json"
)

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


def softmax(values):
    x = np.asarray(values, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def get_confidence(row):
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
            f"Expected 1272 rows, got {len(rows)}"
        )

    confidence = np.asarray(
        [
            get_confidence(row)
            for row in rows
        ],
        dtype=float,
    )

    correct = np.asarray(
        [
            int(row["wouldbe_correct"])
            for row in rows
        ],
        dtype=int,
    )

    print("=" * 80)
    print("CORRECT-ONLY SFT DEV CALIBRATION")
    print("=" * 80)

    print(
        f"\nWould-be accuracy   : "
        f"{correct.mean():.4f}"
    )

    print(
        f"Mean confidence     : "
        f"{confidence.mean():.4f}"
    )

    print(
        f"Mean conf | correct : "
        f"{confidence[correct == 1].mean():.4f}"
    )

    print(
        f"Mean conf | wrong   : "
        f"{confidence[correct == 0].mean():.4f}"
    )

    ranked = np.sort(
        confidence
    )[::-1]

    operating_points = []

    print("\n" + "=" * 80)
    print("MATCHED-COVERAGE OPERATING POINTS")
    print("=" * 80)

    for target in TARGET_COVERAGES:

        k = int(
            round(
                target * len(rows)
            )
        )

        threshold = float(
            ranked[k - 1]
        )

        answered = (
            confidence >= threshold
        )

        n_answered = int(
            answered.sum()
        )

        n_correct = int(
            correct[answered].sum()
        )

        n_wrong = (
            n_answered
            - n_correct
        )

        n_abstained = (
            len(rows)
            - n_answered
        )

        coverage = (
            n_answered
            / len(rows)
        )

        answered_accuracy = (
            n_correct
            / n_answered
        )

        wrong_rate = (
            n_wrong
            / len(rows)
        )

        utility = (
            (
                n_correct * 1.0
                + n_abstained * 0.3
                + n_wrong * -2.0
            )
            / len(rows)
        )

        operating_points.append(
            {
                "target_coverage":
                    target,

                "threshold":
                    threshold,

                "actual_coverage":
                    coverage,

                "answered_accuracy":
                    answered_accuracy,

                "dataset_wrong_rate":
                    wrong_rate,

                "utility":
                    utility,
            }
        )

        print(
            f"{target:.0%} -> "
            f"threshold={threshold:.6f}  "
            f"coverage={coverage:.2%}  "
            f"acc={answered_accuracy:.2%}  "
            f"wrong={wrong_rate:.2%}  "
            f"utility={utility:.4f}"
        )

    output = {
        "protocol":
            "correct_only_sft_dev_calibration_v1",

        "split":
            "dev",

        "n_dev":
            len(rows),

        "would_be_accuracy":
            float(correct.mean()),

        "score":
            (
                "max softmax probability over "
                "A-D full-completion scores"
            ),

        "operating_points":
            operating_points,

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

    print("\nSaved:")
    print(OUTPUT_FILE)

    print(
        "\nCORRECT-ONLY CALIBRATION: PASS"
    )


if __name__ == "__main__":
    main()
