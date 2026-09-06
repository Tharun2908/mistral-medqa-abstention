"""
Final clean MedQA locked-TEST comparison.

All model/checkpoint/score/threshold choices were frozen on DEV
before these TEST evaluations.

Models evaluated here:
    1. Original SFT
    2. Continue-SFT
    3. Correct-only SFT
    4. Supervised 5-way
    5. DPO 2:1
    6. GRPO common initialization

Already-evaluated locked TEST results are reused for:
    7. GRPO Arm A
    8. GRPO Arm B

IMPORTANT:
    - No TEST-derived thresholding.
    - No TEST-derived checkpoint selection.
    - No hyperparameter changes.
    - All full-completion scores use the same A/B/C/D/E evaluator.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.common.medqa_data import load_medqa

from scripts.phase2_learned_abstention.eval_warmstart_dev import (
    build_prompt,
    load_model,
    score_completion_batch,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "locked_test"
    / "final_comparison"
)

GRPO_TEST_SUMMARY = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "locked_test"
    / "grpo"
    / "grpo_locked_test_summary.json"
)


ANSWER_LABELS = [
    "A",
    "B",
    "C",
    "D",
]

ALL_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
]

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]


# ---------------------------------------------------------------------
# FROZEN configurations
# ---------------------------------------------------------------------

METHODS = {

    # ----------------------------------------------------------
    # 1. Original SFT
    #
    # IMPORTANT:
    # These are the corrected FULL-COMPLETION A-D confidence
    # thresholds, NOT the historical next-token thresholds.
    # ----------------------------------------------------------

    "original_sft": {

        "adapter":
            "Primeinvincible/mistral-medqa-lora-v3",

        "checkpoint_selection":
            "original frozen SFT adapter",

        "score_type":
            "confidence",

        "score_definition":
            (
                "max softmax probability over A/B/C/D "
                "full-completion mean log-scores"
            ),

        "threshold_source":
            (
                "dev_full_completion_paired_bootstrap.json"
            ),

        "thresholds": {
            "0.3": 0.3021589736677482,
            "0.4": 0.2923220633867815,
            "0.5": 0.2854659852747448,
            "0.6": 0.2785008328807732,
        },
    },


    # ----------------------------------------------------------
    # 2. Continue-SFT
    # ----------------------------------------------------------

    "continue_sft": {

        "adapter":
            str(
                REPO_ROOT
                / "results"
                / "clean_protocol"
                / "learned_abstention"
                / "continue_sft_control"
                / "main"
                / "checkpoints"
                / "checkpoint-1000"
            ),

        "checkpoint_selection":
            "matched-compute checkpoint-1000",

        "score_type":
            "confidence",

        "score_definition":
            (
                "max softmax probability over A/B/C/D "
                "full-completion mean log-scores"
            ),

        "threshold_source":
            "continue_sft_step1000_dev_calibration.json",

        "thresholds": {
            "0.3": 0.34801787636206843,
            "0.4": 0.331095533599246,
            "0.5": 0.3182072478715298,
            "0.6": 0.30646372555930074,
        },
    },


    # ----------------------------------------------------------
    # 3. Correct-only SFT
    #
    # final/ is the deliberately stopped step-1000 state.
    # ----------------------------------------------------------

    "correct_only_sft": {

        "adapter":
            str(
                REPO_ROOT
                / "results"
                / "clean_protocol"
                / "learned_abstention"
                / "correct_only_sft_control"
                / "main"
                / "final"
            ),

        "checkpoint_selection":
            "optimizer-step matched exact step-1000 final",

        "score_type":
            "confidence",

        "score_definition":
            (
                "max softmax probability over A/B/C/D "
                "full-completion mean log-scores"
            ),

        "threshold_source":
            "correct_only_sft_step1000_dev_calibration.json",

        "thresholds": {
            "0.3": 0.5008164101831156,
            "0.4": 0.4713761710744318,
            "0.5": 0.44182732886073406,
            "0.6": 0.41114522540304105,
        },
    },


    # ----------------------------------------------------------
    # 4. Supervised 5-way
    #
    # trainer_state confirmed checkpoint-1000 was best.
    # ----------------------------------------------------------

    "supervised_5way": {

        "adapter":
            str(
                REPO_ROOT
                / "results"
                / "clean_protocol"
                / "learned_abstention"
                / "supervised_5way"
                / "main"
                / "checkpoints"
                / "checkpoint-1000"
            ),

        "checkpoint_selection":
            (
                "internal validation best checkpoint-1000; "
                "best_metric=0.13248348236083984"
            ),

        "score_type":
            "margin",

        "score_definition":
            (
                "score(E) - max(score(A), score(B), "
                "score(C), score(D))"
            ),

        "threshold_source":
            "supervised_5way_margin_dev_calibration.json",

        "thresholds": {
            "0.3": -0.0319032222032547,
            "0.4": 0.02521248161792755,
            "0.5": 0.07165184617042542,
            "0.6": 0.12055434286594391,
        },
    },


    # ----------------------------------------------------------
    # 5. DPO 2:1
    #
    # trainer_state confirmed checkpoint-500 was best.
    # ----------------------------------------------------------

    "dpo_2to1": {

        "adapter":
            str(
                REPO_ROOT
                / "results"
                / "clean_protocol"
                / "learned_abstention"
                / "dpo"
                / "ratio_2to1"
                / "checkpoints"
                / "checkpoint-500"
                / "policy"
            ),

        "checkpoint_selection":
            (
                "internal preference validation best "
                "checkpoint-500; "
                "best_metric=0.5920783281326294"
            ),

        "score_type":
            "margin",

        "score_definition":
            (
                "score(E) - max(score(A), score(B), "
                "score(C), score(D))"
            ),

        "threshold_source":
            "dpo_ratio_2to1_margin_dev_calibration.json",

        "thresholds": {
            "0.3": -0.5526352226734161,
            "0.4": -0.4132750630378723,
            "0.5": -0.28371188044548035,
            "0.6": -0.16651898622512817,
        },
    },


    # ----------------------------------------------------------
    # 6. GRPO common initialization
    #
    # Headline deployment uses its confidence policy.
    # ----------------------------------------------------------

    "grpo_common_init": {

        "adapter":
            str(
                REPO_ROOT
                / "results"
                / "clean_protocol"
                / "learned_abstention"
                / "grpo_common_warmstart"
                / "main"
                / "policy"
            ),

        "checkpoint_selection":
            "frozen common initialization before GRPO",

        "score_type":
            "confidence",

        "score_definition":
            (
                "max softmax probability over A/B/C/D "
                "full-completion mean log-scores"
            ),

        "threshold_source":
            "grpo_common_init_dev_scores.json",

        "thresholds": {
            "0.3": 0.39002953177939786,
            "0.4": 0.36998360497002253,
            "0.5": 0.35261813757151017,
            "0.6": 0.3365014738198169,
        },
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_gold(value):

    if isinstance(value, int):

        if 0 <= value <= 3:
            return ANSWER_LABELS[value]

    text = str(value).strip().upper()

    if text in ANSWER_LABELS:
        return text

    if text in {
        "0",
        "1",
        "2",
        "3",
    }:

        return ANSWER_LABELS[
            int(text)
        ]

    raise ValueError(
        f"Cannot normalize gold: {value!r}"
    )


def softmax(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    x = (
        x
        - np.max(x)
    )

    e = np.exp(x)

    return (
        e
        / e.sum()
    )


def answer_confidence(scores):

    values = [
        float(scores[label])
        for label in ANSWER_LABELS
    ]

    probs = softmax(
        values
    )

    return float(
        np.max(probs)
    )


def abstention_margin(scores):

    best_answer = max(
        float(scores[label])
        for label in ANSWER_LABELS
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
        correct[
            answered
        ].sum()
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


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------

def evaluate_method(
    name,
    config,
    test,
):

    output_file = (
        OUTPUT_ROOT
        / f"{name}.json"
    )

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print(
        f"Adapter:\n"
        f"{config['adapter']}"
    )

    print(
        f"Score type: "
        f"{config['score_type']}"
    )

    model, tokenizer = (
        load_model(
            config["adapter"]
        )
    )

    rows = []

    for i, ex in enumerate(test):

        prompt = build_prompt(
            ex["question"],
            ex["options"],
        )

        scores = (
            score_completion_batch(
                model,
                tokenizer,
                prompt,
            )
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
                    int(
                        would_be == gold
                    ),

                "natural_decision":
                    natural,

                "answer_confidence":
                    confidence,

                "abstention_margin":
                    margin,

                "scores": {
                    label:
                        float(
                            scores[label]
                        )
                    for label
                    in ALL_LABELS
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

    correct = np.asarray(
        [
            row[
                "wouldbe_correct"
            ]
            for row in rows
        ],
        dtype=int,
    )

    wrong = (
        1
        - correct
    )

    confidence = np.asarray(
        [
            row[
                "answer_confidence"
            ]
            for row in rows
        ],
        dtype=float,
    )

    margin = np.asarray(
        [
            row[
                "abstention_margin"
            ]
            for row in rows
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
                row[
                    "natural_decision"
                ]
                != "E"
                for row in rows
            ]
        )
    )

    if (
        config["score_type"]
        == "confidence"
    ):

        deployment_auc = (
            confidence_auc
        )

    elif (
        config["score_type"]
        == "margin"
    ):

        deployment_auc = (
            margin_auc
        )

    else:

        raise ValueError(
            config["score_type"]
        )

    result = {
        "method":
            name,

        "adapter":
            config["adapter"],

        "checkpoint_selection":
            config[
                "checkpoint_selection"
            ],

        "score_type":
            config[
                "score_type"
            ],

        "score_definition":
            config[
                "score_definition"
            ],

        "threshold_source":
            config[
                "threshold_source"
            ],

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

        "deployment_wrongness_auroc":
            deployment_auc,

        "natural_coverage":
            natural_coverage,

        "operating_points":
            {},
    }

    print(
        f"\nWould-be accuracy          : "
        f"{result['wouldbe_accuracy']:.4f}"
    )

    print(
        f"Confidence wrong AUROC     : "
        f"{confidence_auc:.4f}"
    )

    print(
        f"E-margin wrong AUROC       : "
        f"{margin_auc:.4f}"
    )

    print(
        f"DEPLOYMENT wrong AUROC     : "
        f"{deployment_auc:.4f}"
    )

    print(
        f"Natural coverage           : "
        f"{natural_coverage:.4f}"
    )

    print(
        "\nFROZEN DEV THRESHOLDS -> TEST"
    )

    for target in TARGET_COVERAGES:

        key = str(
            target
        )

        threshold = float(
            config[
                "thresholds"
            ][key]
        )

        if (
            config["score_type"]
            == "confidence"
        ):

            answered = (
                confidence
                >= threshold
            )

        else:

            answered = (
                margin
                <= threshold
            )

        metrics = (
            operating_metrics(
                correct,
                answered,
            )
        )

        result[
            "operating_points"
        ][key] = {
            "target_dev_coverage":
                target,

            "frozen_dev_threshold":
                threshold,

            **metrics,
        }

        print(
            f"{target:.0%} target -> "
            f"test_cov="
            f"{metrics['coverage']:.2%}  "
            f"acc="
            f"{metrics['answered_accuracy']:.2%}  "
            f"wrong="
            f"{metrics['dataset_wrong_rate']:.2%}  "
            f"utility="
            f"{metrics['utility']:.4f}"
        )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "protocol":
                    "clean_final_locked_test_v1",

                "split":
                    "test",

                "checkpoint_source":
                    "pre-test frozen",

                "threshold_source":
                    "official dev",

                "no_test_selection":
                    True,

                "result":
                    result,

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

    return result


# ---------------------------------------------------------------------
# GRPO result reuse
# ---------------------------------------------------------------------

def load_existing_grpo():

    if not GRPO_TEST_SUMMARY.exists():

        raise FileNotFoundError(
            f"Missing existing locked GRPO results:\n"
            f"{GRPO_TEST_SUMMARY}"
        )

    with open(
        GRPO_TEST_SUMMARY,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data


def normalize_grpo_result(
    method_name,
    raw,
):

    score_type = (
        raw["score_type"]
    )

    if score_type == "confidence":

        deployment_auc = (
            raw[
                "confidence_wrongness_auroc"
            ]
        )

    else:

        deployment_auc = (
            raw[
                "margin_wrongness_auroc"
            ]
        )

    return {
        "method":
            method_name,

        "adapter":
            raw["adapter"],

        "score_type":
            score_type,

        "wouldbe_accuracy":
            raw[
                "wouldbe_accuracy"
            ],

        "confidence_wrongness_auroc":
            raw[
                "confidence_wrongness_auroc"
            ],

        "margin_wrongness_auroc":
            raw[
                "margin_wrongness_auroc"
            ],

        "deployment_wrongness_auroc":
            deployment_auc,

        "natural_coverage":
            raw[
                "natural_coverage"
            ],

        "operating_points":
            raw[
                "operating_points"
            ],
    }


# ---------------------------------------------------------------------
# Final tables
# ---------------------------------------------------------------------

def print_summary_table(results):

    print("\n" + "=" * 116)
    print("FINAL LOCKED TEST — MODEL SUMMARY")
    print("=" * 116)

    print(
        f"{'method':24}"
        f"{'WB acc':>10}"
        f"{'deploy AUC':>13}"
        f"{'conf AUC':>11}"
        f"{'margin AUC':>12}"
        f"{'nat cov':>10}"
        f"{'score':>12}"
    )

    print("-" * 116)

    for r in results:

        print(
            f"{r['method']:24}"
            f"{r['wouldbe_accuracy']:10.4f}"
            f"{r['deployment_wrongness_auroc']:13.4f}"
            f"{r['confidence_wrongness_auroc']:11.4f}"
            f"{r['margin_wrongness_auroc']:12.4f}"
            f"{r['natural_coverage']:10.4f}"
            f"{r['score_type']:>12}"
        )


def print_coverage_table(
    results,
    target,
):

    key = str(
        target
    )

    print("\n" + "=" * 116)

    print(
        f"FROZEN DEV {target:.0%} TARGET "
        f"-> ACTUAL LOCKED TEST"
    )

    print("=" * 116)

    print(
        f"{'method':24}"
        f"{'test cov':>11}"
        f"{'ans acc':>11}"
        f"{'wrong rate':>13}"
        f"{'utility':>11}"
    )

    print("-" * 116)

    for r in results:

        op = (
            r[
                "operating_points"
            ][key]
        )

        print(
            f"{r['method']:24}"
            f"{op['coverage']:11.2%}"
            f"{op['answered_accuracy']:11.2%}"
            f"{op['dataset_wrong_rate']:13.2%}"
            f"{op['utility']:11.4f}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 100)
    print("FINAL LOCKED MEDQA TEST COMPARISON")
    print("=" * 100)

    print(
        "\nAll checkpoints and thresholds "
        "were frozen before TEST evaluation."
    )

    print(
        "No TEST-based retuning or selection "
        "is performed."
    )

    test = load_medqa(
        "test",
        allow_test=True,
    )

    if len(test) != 1273:

        raise RuntimeError(
            f"Expected 1273 TEST rows, "
            f"got {len(test)}"
        )

    print(
        f"\nOfficial TEST rows: "
        f"{len(test)}"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # ----------------------------------------------------------
    # Evaluate six frozen baselines
    # ----------------------------------------------------------

    for name, config in (
        METHODS.items()
    ):

        result = (
            evaluate_method(
                name,
                config,
                test,
            )
        )

        results.append(
            result
        )

    # ----------------------------------------------------------
    # Reuse locked GRPO A/B results
    # ----------------------------------------------------------

    grpo = load_existing_grpo()

    arm_a = normalize_grpo_result(
        "grpo_arm_a",
        grpo["arm_a"],
    )

    arm_b = normalize_grpo_result(
        "grpo_arm_b",
        grpo["arm_b"],
    )

    results.append(
        arm_a
    )

    results.append(
        arm_b
    )

    # ----------------------------------------------------------
    # Print final comparison
    # ----------------------------------------------------------

    print_summary_table(
        results
    )

    for target in (
        TARGET_COVERAGES
    ):

        print_coverage_table(
            results,
            target,
        )

    # ----------------------------------------------------------
    # Save final summary
    # ----------------------------------------------------------

    summary = {
        "protocol":
            "clean_final_locked_test_comparison_v1",

        "split":
            "test",

        "n_test":
            1273,

        "selection_split":
            "dev_or_internal_train_validation",

        "threshold_split":
            "dev",

        "no_test_tuning":
            True,

        "methods":
            results,
    }

    output_file = (
        OUTPUT_ROOT
        / "final_locked_test_comparison.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 100)

    print(
        "FINAL LOCKED TEST COMPARISON COMPLETE"
    )

    print(
        "NO TEST-BASED RETUNING PERMITTED"
    )

    print(
        f"\nSaved:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
