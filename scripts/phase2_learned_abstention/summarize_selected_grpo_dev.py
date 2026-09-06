import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

EVAL_ROOT = (
    ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart_eval"
)

OUT = (
    ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo"
    / "selected_dev_operating_points.json"
)

ANSWER = ["A", "B", "C", "D"]

COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]


def load(name):
    with open(
        EVAL_ROOT / f"{name}.json",
        encoding="utf-8",
    ) as f:
        x = json.load(f)

    rows = x["rows"]

    assert len(rows) == 1272

    return rows


def softmax(x):
    x = np.asarray(x, dtype=float)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def confidence(row):
    return float(
        max(
            softmax([
                float(row["scores"][x])
                for x in ANSWER
            ])
        )
    )


def margin(row):
    s = row["scores"]

    return float(
        s["E"]
        - max(
            float(s[x])
            for x in ANSWER
        )
    )


def base_arrays(rows):

    correct = np.asarray(
        [
            int(r["wouldbe_correct"])
            for r in rows
        ],
        dtype=int,
    )

    return correct


def operating_metrics(
    correct,
    answered,
):

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
                    n_correct
                    + 0.3 * n_abstained
                    - 2.0 * n_wrong
                )
                / n
            ),
    }


def summarize_confidence(rows):

    correct = base_arrays(rows)

    score = np.asarray(
        [
            confidence(r)
            for r in rows
        ]
    )

    wrong = 1 - correct

    auc = float(
        roc_auc_score(
            wrong,
            -score,
        )
    )

    ordered = np.sort(
        score
    )[::-1]

    ops = {}

    for cov in COVERAGES:

        k = int(
            round(
                cov * len(rows)
            )
        )

        threshold = float(
            ordered[k - 1]
        )

        answered = (
            score >= threshold
        )

        ops[str(cov)] = {
            "threshold":
                threshold,

            **operating_metrics(
                correct,
                answered,
            ),
        }

    return {
        "wouldbe_accuracy":
            float(correct.mean()),

        "wrongness_auroc":
            auc,

        "score":
            "A-D confidence",

        "answer_rule":
            "confidence >= threshold",

        "operating_points":
            ops,
    }


def summarize_margin(rows):

    correct = base_arrays(rows)

    score = np.asarray(
        [
            margin(r)
            for r in rows
        ]
    )

    wrong = 1 - correct

    auc = float(
        roc_auc_score(
            wrong,
            score,
        )
    )

    # Lower margin = safer answer.
    ordered = np.sort(
        score
    )

    ops = {}

    for cov in COVERAGES:

        k = int(
            round(
                cov * len(rows)
            )
        )

        threshold = float(
            ordered[k - 1]
        )

        answered = (
            score <= threshold
        )

        ops[str(cov)] = {
            "threshold":
                threshold,

            **operating_metrics(
                correct,
                answered,
            ),
        }

    return {
        "wouldbe_accuracy":
            float(correct.mean()),

        "wrongness_auroc":
            auc,

        "score":
            "E margin",

        "answer_rule":
            "margin <= threshold",

        "operating_points":
            ops,
    }


def print_result(
    title,
    result,
):

    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

    print(
        f"WB accuracy      : "
        f"{result['wouldbe_accuracy']:.4f}"
    )

    print(
        f"Wrongness AUROC  : "
        f"{result['wrongness_auroc']:.4f}"
    )

    print()

    for cov in COVERAGES:

        r = result[
            "operating_points"
        ][str(cov)]

        print(
            f"{cov:.0%} -> "
            f"thr={r['threshold']:.6f}  "
            f"cov={r['coverage']:.2%}  "
            f"acc={r['answered_accuracy']:.2%}  "
            f"wrong={r['dataset_wrong_rate']:.2%}  "
            f"utility={r['utility']:.4f}"
        )


def main():

    arm_a = summarize_confidence(
        load(
            "grpo_arm_a_checkpoint_50"
        )
    )

    arm_b = summarize_margin(
        load(
            "grpo_arm_b_checkpoint_50"
        )
    )

    print("=" * 88)
    print("SELECTED GRPO DEV OPERATING POINTS")
    print("=" * 88)

    print(
        "Official TEST touched: NO"
    )

    print_result(
        "ARM A — CHECKPOINT 50",
        arm_a,
    )

    print_result(
        "ARM B — CHECKPOINT 50",
        arm_b,
    )

    result = {
        "protocol":
            "clean_grpo_selected_dev_thresholds_v1",

        "split":
            "dev",

        "test_used":
            False,

        "arm_a": {
            "checkpoint":
                "checkpoint-50",

            "adapter":
                str(
                    ROOT
                    / "results"
                    / "clean_protocol"
                    / "learned_abstention"
                    / "grpo"
                    / "arm_a_answer_only"
                    / "main"
                    / "checkpoints"
                    / "checkpoint-50"
                ),

            **arm_a,
        },

        "arm_b": {
            "checkpoint":
                "checkpoint-50",

            "adapter":
                str(
                    ROOT
                    / "results"
                    / "clean_protocol"
                    / "learned_abstention"
                    / "grpo"
                    / "arm_b_abstention"
                    / "main"
                    / "checkpoints"
                    / "checkpoint-50"
                ),

            **arm_b,
        },
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"\nSaved: {OUT}"
    )

    print(
        "\nDEV THRESHOLDS FROZEN — TEST STILL LOCKED"
    )


if __name__ == "__main__":
    main()
