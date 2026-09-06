"""
Post-hoc TEST evaluation of common-init DPO 2:1.

IMPORTANT:
- This experiment was motivated AFTER the original locked TEST had been opened.
- Therefore it is a POST-HOC robustness/control follow-up.
- Model training/checkpoint selection used only TRAIN-derived internal validation.
- Deployment thresholds were frozen on official DEV before this TEST evaluation.
- TEST is not used to select or retune anything here.

Deployment score:
    abstention_margin =
        score(E) - max(score(A), score(B), score(C), score(D))

Answer rule:
    answer if abstention_margin <= frozen DEV threshold
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
ALL_LABELS = ["A", "B", "C", "D", "E"]


ADAPTER = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "dpo"
    / "common_init_ratio_2to1"
    / "final"
    / "policy"
)


DEV_CALIBRATION = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "dpo_common_init_ratio_2to1_margin_dev_calibration.json"
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "posthoc_test"
    / "dpo_common_init_ratio_2to1"
)


OUTPUT_FILE = (
    OUTPUT_ROOT
    / "dpo_common_init_ratio_2to1_test.json"
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

    utility = (
        n_correct
        + 0.3 * n_abstained
        - 2.0 * n_wrong
    ) / n

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
                utility
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


def main():

    print("=" * 92)
    print("POST-HOC DPO COMMON-INIT TEST EVALUATION")
    print("=" * 92)

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a POST-HOC robustness/control follow-up."
    )

    print(
        "Original locked TEST had already been opened "
        "before this experiment was motivated."
    )

    print(
        "\nNo TEST-based checkpoint or threshold "
        "selection is performed."
    )

    # ----------------------------------------------------------
    # Load frozen DEV calibration
    # ----------------------------------------------------------

    with open(
        DEV_CALIBRATION,
        "r",
        encoding="utf-8",
    ) as f:
        calibration = json.load(f)

    if calibration.get("split") != "dev":
        raise RuntimeError(
            "Expected frozen DEV calibration."
        )

    frozen_thresholds = {
        key:
            float(value["threshold"])
        for key, value
        in calibration[
            "operating_points"
        ].items()
    }

    print(
        "\nFrozen DEV thresholds:"
    )

    for key in [
        "0.3",
        "0.4",
        "0.5",
        "0.6",
    ]:

        print(
            f"  {key}: "
            f"{frozen_thresholds[key]:.6f}"
        )

    # ----------------------------------------------------------
    # Load official TEST
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
        f"\nAdapter:\n{ADAPTER}"
    )

    # ----------------------------------------------------------
    # Score TEST
    # ----------------------------------------------------------

    model, tokenizer = load_model(
        str(ADAPTER)
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
            key=lambda x:
                scores[x],
        )

        natural = max(
            ALL_LABELS,
            key=lambda x:
                scores[x],
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

                "answer_confidence":
                    confidence,

                "abstention_margin":
                    margin,

                "scores": {
                    k:
                        float(v)
                    for k, v
                    in scores.items()
                },
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

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    correct = np.asarray(
        [
            r["wouldbe_correct"]
            for r in rows
        ],
        dtype=int,
    )

    wrong = (
        1
        - correct
    )

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

    wouldbe_accuracy = float(
        correct.mean()
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

    print(
        "\n" + "=" * 92
    )

    print(
        "POST-HOC TEST SUMMARY"
    )

    print(
        "=" * 92
    )

    print(
        f"Would-be accuracy             : "
        f"{wouldbe_accuracy:.4f}"
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

    # ----------------------------------------------------------
    # Apply exact frozen DEV thresholds
    # ----------------------------------------------------------

    operating_points = {}

    print(
        "\nFROZEN DEV THRESHOLDS -> "
        "POST-HOC TEST"
    )

    for key in [
        "0.3",
        "0.4",
        "0.5",
        "0.6",
    ]:

        target = float(key)

        threshold = (
            frozen_thresholds[key]
        )

        answered = (
            margin
            <= threshold
        )

        metrics = operating_metrics(
            correct,
            answered,
        )

        operating_points[key] = {
            "target_dev_coverage":
                target,

            "frozen_dev_threshold":
                threshold,

            **metrics,
        }

        print(
            f"{target:.0%} target -> "
            f"thr={threshold:.6f}  "
            f"test_cov="
            f"{metrics['coverage']:.2%}  "
            f"acc="
            f"{metrics['answered_accuracy']:.2%}  "
            f"wrong="
            f"{metrics['dataset_wrong_rate']:.2%}  "
            f"utility="
            f"{metrics['utility']:.4f}"
        )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "protocol":
            "posthoc_dpo_common_init_test_v1",

        "split":
            "test",

        "status":
            "posthoc_followup_after_original_test_open",

        "original_locked_test_already_seen":
            True,

        "test_used_for_selection":
            False,

        "interpretation_constraint":
            (
                "Robustness/control follow-up only; "
                "not part of the original preregistered "
                "one-look locked-test comparison."
            ),

        "adapter":
            str(ADAPTER),

        "dev_calibration_file":
            str(DEV_CALIBRATION),

        "deployment_score":
            (
                "score(E) - max(score(A-D))"
            ),

        "answer_rule":
            (
                "abstention_margin <= "
                "frozen DEV threshold"
            ),

        "n_test":
            len(rows),

        "wouldbe_accuracy":
            wouldbe_accuracy,

        "confidence_wrongness_auroc":
            confidence_auc,

        "margin_wrongness_auroc":
            margin_auc,

        "natural_coverage":
            natural_coverage,

        "operating_points":
            operating_points,

        "rows":
            rows,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

    print(
        "\n" + "=" * 92
    )

    print(
        "POST-HOC DPO TEST COMPLETE"
    )

    print(
        "=" * 92
    )

    print(
        "\nNO TEST-BASED RETUNING PERMITTED"
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
