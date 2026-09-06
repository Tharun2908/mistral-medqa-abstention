"""
FINAL LOCKED TEST evaluation for selected GRPO arms.

IMPORTANT:
- Checkpoints were selected on DEV.
- Coverage thresholds were selected/frozen on DEV.
- TEST is used only for final evaluation.
- No threshold, checkpoint, or hyperparameter is selected from TEST.

Uses exactly the same full-sentence A/B/C/D/E scoring implementation
as eval_warmstart_dev.py.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.common.medqa_data import load_medqa

from scripts.phase2_learned_abstention.eval_warmstart_dev import (
    build_prompt,
    load_model,
    score_completion_batch,
)


ANSWER_LABELS = ["A", "B", "C", "D"]

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]


FROZEN_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo"
    / "selected_dev_operating_points.json"
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "locked_test"
    / "grpo"
)


def normalize_gold(value):

    if isinstance(value, int):
        if 0 <= value <= 3:
            return ANSWER_LABELS[value]

    text = str(value).strip().upper()

    if text in ANSWER_LABELS:
        return text

    if text in {"0", "1", "2", "3"}:
        return ANSWER_LABELS[int(text)]

    raise ValueError(
        f"Cannot normalize gold: {value!r}"
    )


def softmax(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    x = x - np.max(x)

    e = np.exp(x)

    return e / e.sum()


def answer_confidence(scores):

    vals = [
        float(scores[x])
        for x in ANSWER_LABELS
    ]

    return float(
        np.max(
            softmax(vals)
        )
    )


def abstention_margin(scores):

    best_answer = max(
        float(scores[x])
        for x in ANSWER_LABELS
    )

    return float(
        scores["E"]
        - best_answer
    )


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
        n
        - n_answered
    )

    answered_accuracy = (
        n_correct / n_answered
        if n_answered > 0
        else float("nan")
    )

    return {
        "coverage":
            float(
                n_answered / n
            ),

        "answered_accuracy":
            float(
                answered_accuracy
            ),

        "dataset_wrong_rate":
            float(
                n_wrong / n
            ),

        "utility":
            float(
                (
                    n_correct
                    + 0.3 * n_abstained
                    - 2.0 * n_wrong
                )
                / n
            ),

        "n_answered":
            n_answered,

        "n_correct_answered":
            n_correct,

        "n_wrong_answered":
            n_wrong,

        "n_abstained":
            n_abstained,
    }


def evaluate_adapter(
    name,
    adapter,
    score_type,
    frozen_thresholds,
    test,
):

    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)

    print(
        f"Adapter:\n{adapter}"
    )

    print(
        f"Deployment score: {score_type}"
    )

    model, tokenizer = load_model(
        adapter
    )

    rows = []

    for i, ex in enumerate(test):

        prompt = build_prompt(
            ex["question"],
            ex["options"],
        )

        scores = score_completion_batch(
            model,
            tokenizer,
            prompt,
        )

        gold = normalize_gold(
            ex["answer_idx"]
        )

        would_be = max(
            ANSWER_LABELS,
            key=lambda x: scores[x],
        )

        natural = max(
            [
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
            key=lambda x: scores[x],
        )

        correct = int(
            would_be == gold
        )

        confidence = (
            answer_confidence(
                scores
            )
        )

        margin = (
            abstention_margin(
                scores
            )
        )

        rows.append(
            {
                "id":
                    str(ex["id"]),

                "gold":
                    gold,

                "wouldbe_answer":
                    would_be,

                "wouldbe_correct":
                    correct,

                "natural_decision":
                    natural,

                "scores":
                    {
                        k: float(v)
                        for k, v
                        in scores.items()
                    },

                "answer_confidence":
                    confidence,

                "abstention_margin":
                    margin,
            }
        )

        if (
            (i + 1) % 100 == 0
            or
            (i + 1) == len(test)
        ):

            print(
                f"  scored "
                f"{i + 1}/{len(test)}"
            )

    correct = np.asarray(
        [
            r["wouldbe_correct"]
            for r in rows
        ],
        dtype=int,
    )

    wrong = 1 - correct

    confidence = np.asarray(
        [
            r["answer_confidence"]
            for r in rows
        ],
        dtype=float,
    )

    margin = np.asarray(
        [
            r["abstention_margin"]
            for r in rows
        ],
        dtype=float,
    )

    confidence_auc = float(
        roc_auc_score(
            wrong,
            -confidence,
        )
    )

    margin_auc = float(
        roc_auc_score(
            wrong,
            margin,
        )
    )

    natural_coverage = float(
        np.mean(
            [
                r["natural_decision"]
                != "E"
                for r in rows
            ]
        )
    )

    results = {
        "name":
            name,

        "adapter":
            adapter,

        "n_test":
            len(rows),

        "wouldbe_accuracy":
            float(
                correct.mean()
            ),

        "confidence_wrongness_auroc":
            confidence_auc,

        "margin_wrongness_auroc":
            margin_auc,

        "natural_coverage":
            natural_coverage,

        "score_type":
            score_type,

        "operating_points":
            {},
    }

    print(
        f"\nWould-be accuracy             : "
        f"{results['wouldbe_accuracy']:.4f}"
    )

    print(
        f"Confidence -> wrong AUROC     : "
        f"{confidence_auc:.4f}"
    )

    print(
        f"E margin -> wrong AUROC       : "
        f"{margin_auc:.4f}"
    )

    print(
        f"Natural coverage              : "
        f"{natural_coverage:.4f}"
    )

    print(
        "\nFROZEN DEV THRESHOLDS APPLIED TO TEST"
    )

    for cov in TARGET_COVERAGES:

        key = str(cov)

        threshold = float(
            frozen_thresholds[
                key
            ]["threshold"]
        )

        if score_type == "confidence":

            answered = (
                confidence
                >= threshold
            )

        elif score_type == "margin":

            answered = (
                margin
                <= threshold
            )

        else:

            raise ValueError(
                score_type
            )

        m = operating_metrics(
            correct,
            answered,
        )

        results[
            "operating_points"
        ][key] = {
            "target_dev_coverage":
                cov,

            "frozen_dev_threshold":
                threshold,

            **m,
        }

        print(
            f"{cov:.0%} target -> "
            f"thr={threshold:.6f}  "
            f"test_cov={m['coverage']:.2%}  "
            f"acc={m['answered_accuracy']:.2%}  "
            f"wrong={m['dataset_wrong_rate']:.2%}  "
            f"utility={m['utility']:.4f}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_ROOT
        / f"{name}.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "protocol":
                    "clean_locked_test_v1",

                "split":
                    "test",

                "threshold_source":
                    "official_dev",

                "checkpoint_source":
                    "official_dev",

                "results":
                    results,

                "rows":
                    rows,
            },
            f,
            indent=2,
        )

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

    return results


def main():

    print("=" * 92)
    print("FINAL LOCKED TEST — SELECTED GRPO")
    print("=" * 92)

    print(
        "\nWARNING: OFFICIAL TEST IS NOW BEING ACCESSED."
    )

    print(
        "All checkpoints and thresholds are already frozen."
    )

    with open(
        FROZEN_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        frozen = json.load(f)

    # ----------------------------------------------------------
    # Integrity checks
    # ----------------------------------------------------------

    assert (
        frozen["arm_a"]["checkpoint"]
        == "checkpoint-50"
    )

    assert (
        frozen["arm_b"]["checkpoint"]
        == "checkpoint-50"
    )

    arm_a_adapter = (
        frozen["arm_a"]["adapter"]
    )

    arm_b_adapter = (
        frozen["arm_b"]["adapter"]
    )

    # ----------------------------------------------------------
    # FIRST AND ONLY intended final TEST load
    # ----------------------------------------------------------

    test = load_medqa(
        "test",
        allow_test=True,
    )

    if len(test) != 1273:

        raise RuntimeError(
            f"Expected 1273 TEST examples, "
            f"got {len(test)}"
        )

    print(
        f"\nOfficial TEST examples: "
        f"{len(test)}"
    )

    print(
        "No test-derived selection will occur."
    )

    # ----------------------------------------------------------
    # Arm A
    # ----------------------------------------------------------

    arm_a = evaluate_adapter(
        name="grpo_arm_a_checkpoint_50",

        adapter=arm_a_adapter,

        score_type="confidence",

        frozen_thresholds=(
            frozen[
                "arm_a"
            ][
                "operating_points"
            ]
        ),

        test=test,
    )

    # ----------------------------------------------------------
    # Arm B
    # ----------------------------------------------------------

    arm_b = evaluate_adapter(
        name="grpo_arm_b_checkpoint_50",

        adapter=arm_b_adapter,

        score_type="margin",

        frozen_thresholds=(
            frozen[
                "arm_b"
            ][
                "operating_points"
            ]
        ),

        test=test,
    )

    summary = {
        "protocol":
            "clean_locked_grpo_test_v1",

        "split":
            "test",

        "n_test":
            1273,

        "selection_split":
            "dev",

        "threshold_split":
            "dev",

        "no_test_tuning":
            True,

        "arm_a":
            arm_a,

        "arm_b":
            arm_b,
    }

    with open(
        OUTPUT_ROOT
        / "grpo_locked_test_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 92)

    print(
        "LOCKED GRPO TEST COMPLETE"
    )

    print(
        "NO TEST-BASED RETUNING PERMITTED"
    )

    print(
        f"Saved:\n"
        f"{OUTPUT_ROOT / 'grpo_locked_test_summary.json'}"
    )


if __name__ == "__main__":
    main()
