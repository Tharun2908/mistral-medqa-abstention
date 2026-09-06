import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart_eval"
    / "grpo_common_init.json"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo_common_init_dev_scores.json"
)

ANSWER_LABELS = ["A", "B", "C", "D"]

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def answer_confidence(row):
    scores = row["scores"]

    vals = [
        float(scores[label])
        for label in ANSWER_LABELS
    ]

    probs = softmax(vals)

    return float(np.max(probs))


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


def metrics(correct, answered):

    n = len(correct)

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
        n - n_answered
    )

    return {
        "coverage":
            float(n_answered / n),

        "answered_accuracy":
            float(
                n_correct / n_answered
            ),

        "dataset_wrong_rate":
            float(n_wrong / n),

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


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"Expected 1272 rows, got {len(rows)}"
        )

    correct = np.asarray(
        [
            int(row["wouldbe_correct"])
            for row in rows
        ],
        dtype=int,
    )

    wrong = 1 - correct

    confidence = np.asarray(
        [
            answer_confidence(row)
            for row in rows
        ],
        dtype=float,
    )

    margin = np.asarray(
        [
            abstention_margin(row)
            for row in rows
        ],
        dtype=float,
    )

    # Higher wrongness score should mean more likely wrong.
    confidence_wrongness = (
        -confidence
    )

    confidence_auc = float(
        roc_auc_score(
            wrong,
            confidence_wrongness,
        )
    )

    margin_auc = float(
        roc_auc_score(
            wrong,
            margin,
        )
    )

    print("=" * 84)
    print("GRPO COMMON INIT — DEV SCORE DIAGNOSTICS")
    print("=" * 84)

    print(
        f"\nWould-be answer accuracy : "
        f"{correct.mean():.4f}"
    )

    print(
        f"A-D confidence -> wrong AUROC : "
        f"{confidence_auc:.4f}"
    )

    print(
        f"E margin -> wrong AUROC        : "
        f"{margin_auc:.4f}"
    )

    print(
        f"Mean margin | correct          : "
        f"{margin[correct == 1].mean():.4f}"
    )

    print(
        f"Mean margin | wrong            : "
        f"{margin[correct == 0].mean():.4f}"
    )

    results = {
        "confidence": {},
        "margin": {},
    }

    # ----------------------------------------------------------
    # Arm-A style:
    # high confidence -> answer
    # ----------------------------------------------------------

    ranked_conf = np.sort(
        confidence
    )[::-1]

    print("\n" + "=" * 84)
    print("A-D CONFIDENCE OPERATING POINTS")
    print("=" * 84)

    for target in TARGET_COVERAGES:

        k = int(
            round(
                target * len(rows)
            )
        )

        threshold = float(
            ranked_conf[k - 1]
        )

        answered = (
            confidence >= threshold
        )

        m = metrics(
            correct,
            answered,
        )

        results[
            "confidence"
        ][str(target)] = {
            "threshold":
                threshold,
            **m,
        }

        print(
            f"{target:.0%} -> "
            f"thr={threshold:.6f}  "
            f"cov={m['coverage']:.2%}  "
            f"acc={m['answered_accuracy']:.2%}  "
            f"wrong={m['dataset_wrong_rate']:.2%}  "
            f"utility={m['utility']:.4f}"
        )

    # ----------------------------------------------------------
    # Arm-B style:
    # low margin -> answer
    # ----------------------------------------------------------

    ranked_margin = np.sort(
        margin
    )

    print("\n" + "=" * 84)
    print("E-MARGIN OPERATING POINTS")
    print("=" * 84)

    for target in TARGET_COVERAGES:

        k = int(
            round(
                target * len(rows)
            )
        )

        threshold = float(
            ranked_margin[k - 1]
        )

        answered = (
            margin <= threshold
        )

        m = metrics(
            correct,
            answered,
        )

        results[
            "margin"
        ][str(target)] = {
            "threshold":
                threshold,
            **m,
        }

        print(
            f"{target:.0%} -> "
            f"thr={threshold:.6f}  "
            f"cov={m['coverage']:.2%}  "
            f"acc={m['answered_accuracy']:.2%}  "
            f"wrong={m['dataset_wrong_rate']:.2%}  "
            f"utility={m['utility']:.4f}"
        )

    output = {
        "protocol":
            "grpo_common_init_dev_scores_v1",

        "split":
            "dev",

        "n_dev":
            len(rows),

        "would_be_accuracy":
            float(correct.mean()),

        "confidence_wrongness_auroc":
            confidence_auc,

        "margin_wrongness_auroc":
            margin_auc,

        "operating_points":
            results,

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
        "\nCOMMON INIT SCORE CHECK: PASS"
    )


if __name__ == "__main__":
    main()
