"""
build_supervised_5way_data.py
-----------------------------

Build the clean supervised 5-way abstention baseline dataset.

Targets:
    A/B/C/D : answer when the clean OOF SFT prediction was correct
    E       : abstain when the clean OOF SFT prediction was wrong

Why OOF matters:
    Each correctness decision was produced by an SFT model that did not
    train on that example.

Output space matches the DPO experiment exactly:
    A -> " The answer is A."
    B -> " The answer is B."
    C -> " The answer is C."
    D -> " The answer is D."
    E -> " I cannot answer confidently."

A fixed, stratified 90/10 internal train/validation split is created from
MedQA TRAIN only. Official MedQA dev/test are untouched.
"""

import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "oof_sft"
    / "sft_train_oof_predictions.json"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "supervised_5way_data.json"
)

SEED = 42

ANSWER_SET = ["A", "B", "C", "D"]

COMPLETIONS = {
    "A": " The answer is A.",
    "B": " The answer is B.",
    "C": " The answer is C.",
    "D": " The answer is D.",
    "E": " I cannot answer confidently.",
}


def build_prompt(question, options):

    option_lines = "\n".join(
        f"{letter}: {options[letter]}"
        for letter in ANSWER_SET
    )

    return (
        f"Question: {question}\n\n"
        f"Options:\n"
        f"{option_lines}\n\n"
        f"Answer:"
    )


def main():

    print("=" * 72)
    print("BUILD CLEAN SUPERVISED 5-WAY DATA")
    print("=" * 72)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data.get("protocol") != "clean_oof_v1":
        raise RuntimeError(
            f"Unexpected source protocol: "
            f"{data.get('protocol')}"
        )

    rows = data["predictions"]

    if len(rows) != 10178:
        raise RuntimeError(
            f"Expected 10178 OOF examples, "
            f"got {len(rows)}"
        )

    # --------------------------------------------------------------
    # Integrity
    # --------------------------------------------------------------

    indices = [
        int(row["train_index"])
        for row in rows
    ]

    if len(set(indices)) != 10178:
        raise RuntimeError(
            "Duplicate train_index values found."
        )

    if set(indices) != set(range(10178)):
        raise RuntimeError(
            "OOF train indices do not cover 0..10177 exactly."
        )

    # --------------------------------------------------------------
    # Build supervised targets
    # --------------------------------------------------------------

    examples = []

    for row in rows:

        prediction = row["prediction"]
        gold = row["answer_idx"]

        if prediction not in ANSWER_SET:
            raise RuntimeError(
                f"Malformed prediction: {prediction}"
            )

        if gold not in ANSWER_SET:
            raise RuntimeError(
                f"Malformed gold label: {gold}"
            )

        is_correct = (
            prediction == gold
        )

        if bool(row["is_correct"]) != is_correct:
            raise RuntimeError(
                f"is_correct mismatch for "
                f"train_index={row['train_index']}"
            )

        # Correct OOF prediction -> explicitly answer.
        # Wrong OOF prediction   -> explicitly abstain.
        target_label = (
            gold
            if is_correct
            else "E"
        )

        examples.append(
            {
                "train_index":
                    int(row["train_index"]),

                "id":
                    row["id"],

                "question":
                    row["question"],

                "options":
                    row["options"],

                "gold_answer":
                    gold,

                "oof_prediction":
                    prediction,

                "oof_correct":
                    is_correct,

                "oof_confidence":
                    float(row["confidence"]),

                "oof_fold":
                    int(row["oof_fold"]),

                "target_label":
                    target_label,

                "prompt":
                    build_prompt(
                        row["question"],
                        row["options"],
                    ),

                "completion":
                    COMPLETIONS[target_label],
            }
        )

    # --------------------------------------------------------------
    # Verify counts
    # --------------------------------------------------------------

    n_answer = sum(
        x["target_label"] != "E"
        for x in examples
    )

    n_abstain = sum(
        x["target_label"] == "E"
        for x in examples
    )

    if n_answer != 5219:
        raise RuntimeError(
            f"Expected 5219 answer targets, "
            f"got {n_answer}"
        )

    if n_abstain != 4959:
        raise RuntimeError(
            f"Expected 4959 abstain targets, "
            f"got {n_abstain}"
        )

    class_counts = Counter(
        x["target_label"]
        for x in examples
    )

    print(
        f"\nTotal examples : "
        f"{len(examples)}"
    )

    print(
        f"Answer targets : "
        f"{n_answer}"
    )

    print(
        f"Abstain target : "
        f"{n_abstain}"
    )

    print(
        "\n5-way label distribution:"
    )

    for label in [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]:
        print(
            f"  {label}: "
            f"{class_counts[label]}"
        )

    # --------------------------------------------------------------
    # Fixed stratified internal split
    # --------------------------------------------------------------

    all_indices = list(
        range(len(examples))
    )

    labels = [
        x["target_label"]
        for x in examples
    ]

    train_idx, val_idx = (
        train_test_split(
            all_indices,
            test_size=0.10,
            random_state=SEED,
            stratify=labels,
        )
    )

    train = [
        examples[i]
        for i in train_idx
    ]

    val = [
        examples[i]
        for i in val_idx
    ]

    # --------------------------------------------------------------
    # Split integrity
    # --------------------------------------------------------------

    train_sources = {
        x["train_index"]
        for x in train
    }

    val_sources = {
        x["train_index"]
        for x in val
    }

    if train_sources & val_sources:
        raise RuntimeError(
            "Internal train/val overlap detected."
        )

    if len(
        train_sources
        | val_sources
    ) != 10178:
        raise RuntimeError(
            "Internal split does not recover "
            "all 10178 examples."
        )

    train_counts = Counter(
        x["target_label"]
        for x in train
    )

    val_counts = Counter(
        x["target_label"]
        for x in val
    )

    print(
        "\nInternal split:"
    )

    print(
        f"  train : {len(train)}"
    )

    print(
        f"  val   : {len(val)}"
    )

    print(
        "\nTrain label counts:"
    )

    for label in [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]:
        print(
            f"  {label}: "
            f"{train_counts[label]}"
        )

    print(
        "\nVal label counts:"
    )

    for label in [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]:
        print(
            f"  {label}: "
            f"{val_counts[label]}"
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "protocol":
            "clean_supervised_5way_v1",

        "source_protocol":
            "clean_oof_v1",

        "seed":
            SEED,

        "target_definition": {
            "correct_oof_prediction":
                "gold answer A/B/C/D",

            "wrong_oof_prediction":
                "E",

            "E_completion":
                COMPLETIONS["E"],
        },

        "n_total":
            len(examples),

        "n_answer_targets":
            n_answer,

        "n_abstain_targets":
            n_abstain,

        "class_counts":
            dict(class_counts),

        "split": {
            "strategy":
                "stratified_90_10",

            "stratification":
                "target_label",

            "train_n":
                len(train),

            "val_n":
                len(val),
        },

        "train":
            train,

        "val":
            val,
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
        "\nTrain/val overlap : 0"
    )

    print(
        "Full coverage     : PASS"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SUPERVISED 5-WAY DATA BUILD: PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Saved ->\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
