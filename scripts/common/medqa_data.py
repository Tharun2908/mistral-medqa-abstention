"""
medqa_data.py
-------------

Single source of truth for MedQA data splits.

Research protocol:
    train -> parameter learning
    dev   -> checkpoint / threshold / hyperparameter selection
    test  -> final locked evaluation only

The test split is deliberately blocked unless allow_test=True is passed.
"""

from datasets import load_dataset

DATASET_ID = "GBaker/MedQA-USMLE-4-options-hf"

LETTERS = ["A", "B", "C", "D"]

EXPECTED_SIZES = {
    "train": 10178,
    "dev": 1272,
    "test": 1273,
}


def _normalize(example):
    """
    Convert the HF multiple-choice schema:

        sent1
        ending0 ... ending3
        label: 0..3

    into the schema already used throughout this project:

        question
        options = {A, B, C, D}
        answer_idx = A/B/C/D
    """

    label = int(example["label"])

    return {
        "id": example["id"],
        "question": example["sent1"],
        "options": {
            "A": example["ending0"],
            "B": example["ending1"],
            "C": example["ending2"],
            "D": example["ending3"],
        },
        "answer_idx": LETTERS[label],
    }


def load_medqa(split: str, *, allow_test: bool = False):
    """
    Load one official MedQA partition.

    Valid names:
        train
        dev
        validation
        test

    Test access is intentionally locked unless allow_test=True.
    """

    requested = split.lower().strip()

    aliases = {
        "train": "train",
        "dev": "dev",
        "validation": "dev",
        "test": "test",
    }

    if requested not in aliases:
        raise ValueError(
            f"Unknown split '{split}'. "
            "Use one of: train, dev, validation, test."
        )

    logical_split = aliases[requested]

    if logical_split == "test" and not allow_test:
        raise RuntimeError(
            "\nTEST SPLIT IS LOCKED.\n"
            "Do not use MedQA test for development, checkpoint selection, "
            "threshold tuning, warm-start gating, or debugging.\n\n"
            "Use:\n"
            "    load_medqa('dev')\n\n"
            "Only final frozen evaluation may call:\n"
            "    load_medqa('test', allow_test=True)\n"
        )

    # Load the official 3-way dataset.
    raw = load_dataset(DATASET_ID)

    # Hugging Face exposes the original dev partition as "validation".
    if logical_split == "dev":
        if "validation" in raw:
            hf_split = "validation"
        elif "dev" in raw:
            hf_split = "dev"
        else:
            raise RuntimeError(
                f"No dev/validation split found. Available: {list(raw.keys())}"
            )
    else:
        hf_split = logical_split

    ds = raw[hf_split]

    # Convert to the schema used by our existing project.
    original_columns = ds.column_names

    ds = ds.map(
        _normalize,
        remove_columns=original_columns,
        desc=f"Normalizing MedQA {logical_split}",
    )

    expected = EXPECTED_SIZES[logical_split]

    if len(ds) != expected:
        raise RuntimeError(
            f"Unexpected {logical_split} size: {len(ds)}. "
            f"Expected {expected}. Dataset may have changed."
        )

    return ds