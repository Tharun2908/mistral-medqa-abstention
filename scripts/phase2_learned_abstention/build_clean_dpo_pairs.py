"""
build_clean_dpo_pairs.py
------------------------

Build controlled DPO preference datasets from the clean OOF-derived
candidate pools.

Protocol:
    - MedQA TRAIN only.
    - Source examples come from 5-fold OOF SFT predictions.
    - Warm-start examples are completely excluded.
    - Every DPO configuration has exactly 2,400 pairs.
    - Only the abstain:answer ratio changes.
    - Wrong examples are sampled confidence-weighted.
    - Correct examples are sampled uniformly.
    - Sampling is nested across ratios to reduce sampling noise.
    - Each configuration gets an exact stratified 90/10 train/val split.

Ratios:
    1:1   -> 1200 abstain + 1200 answer
    1.5:1 -> 1440 abstain +  960 answer
    2:1   -> 1600 abstain +  800 answer

Outputs:
    dpo_pairs_ratio_1to1.json
    dpo_pairs_ratio_1p5to1.json
    dpo_pairs_ratio_2to1.json
"""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
)

POOL_FILE = DATA_DIR / "dpo_candidate_pools.json"
WARMSTART_FILE = DATA_DIR / "warmstart_data.json"

SEED = 42
VAL_FRAC = 0.10

ANSWER_SET = ["A", "B", "C", "D"]

ABSTAIN = " I cannot answer confidently."
ANSWER_TEMPLATE = " The answer is {}."


RATIO_CONFIGS = {
    "ratio_1to1": {
        "n_abstain": 1200,
        "n_answer": 1200,
    },
    "ratio_1p5to1": {
        "n_abstain": 1440,
        "n_answer": 960,
    },
    "ratio_2to1": {
        "n_abstain": 1600,
        "n_answer": 800,
    },
}


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Pair creation
# ---------------------------------------------------------------------

def make_abstain_pair(row):

    if row["prediction"] == row["answer_idx"]:
        raise RuntimeError(
            "Abstain pair source is not actually wrong."
        )

    return {
        "prompt": build_prompt(
            row["question"],
            row["options"],
        ),

        "chosen": ABSTAIN,

        "rejected": ANSWER_TEMPLATE.format(
            row["prediction"]
        ),

        "type": "abstain",

        "source_train_index": int(
            row["train_index"]
        ),

        "source_id": row["id"],

        "source_prediction":
            row["prediction"],

        "source_gold":
            row["answer_idx"],

        "source_oof_confidence":
            float(row["confidence"]),

        "source_oof_fold":
            int(row["oof_fold"]),
    }


def make_answer_pair(row):

    if row["prediction"] != row["answer_idx"]:
        raise RuntimeError(
            "Answer pair source is not actually correct."
        )

    return {
        "prompt": build_prompt(
            row["question"],
            row["options"],
        ),

        "chosen": ANSWER_TEMPLATE.format(
            row["answer_idx"]
        ),

        "rejected": ABSTAIN,

        "type": "answer",

        "source_train_index": int(
            row["train_index"]
        ),

        "source_id": row["id"],

        "source_prediction":
            row["prediction"],

        "source_gold":
            row["answer_idx"],

        "source_oof_confidence":
            float(row["confidence"]),

        "source_oof_fold":
            int(row["oof_fold"]),
    }


# ---------------------------------------------------------------------
# Exact stratified split
# ---------------------------------------------------------------------

def split_type_exact(
    pairs,
    seed,
):

    pairs = list(pairs)

    rng = np.random.default_rng(seed)

    order = rng.permutation(
        len(pairs)
    )

    pairs = [
        pairs[i]
        for i in order
    ]

    n_val = int(
        round(
            len(pairs)
            * VAL_FRAC
        )
    )

    val = pairs[:n_val]
    train = pairs[n_val:]

    return train, val


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

def confidence_stats(pairs):

    values = np.array(
        [
            p["source_oof_confidence"]
            for p in pairs
        ],
        dtype=float,
    )

    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(
            np.median(values)
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 76)
    print("BUILD CLEAN DPO PAIR DATASETS")
    print("=" * 76)

    # --------------------------------------------------------------
    # Load candidate pools
    # --------------------------------------------------------------

    with open(
        POOL_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        pools = json.load(f)

    if pools.get("protocol") != "clean_oof_v1":
        raise RuntimeError(
            f"Unexpected pool protocol: "
            f"{pools.get('protocol')}"
        )

    wrong_pool = pools["wrong_pool"]
    correct_pool = pools["correct_pool"]

    print(
        f"\nWrong candidate pool   : "
        f"{len(wrong_pool)}"
    )

    print(
        f"Correct candidate pool : "
        f"{len(correct_pool)}"
    )

    if len(wrong_pool) != 4159:
        raise RuntimeError(
            f"Expected 4159 wrong candidates, "
            f"got {len(wrong_pool)}"
        )

    if len(correct_pool) != 3719:
        raise RuntimeError(
            f"Expected 3719 correct candidates, "
            f"got {len(correct_pool)}"
        )

    # --------------------------------------------------------------
    # Load warm-start indices for an explicit overlap check
    # --------------------------------------------------------------

    with open(
        WARMSTART_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        warm_data = json.load(f)

    warm_indices = {
        int(x["train_index"])
        for x in warm_data["examples"]
    }

    if len(warm_indices) != 2300:
        raise RuntimeError(
            "Unexpected warm-start source count."
        )

    candidate_indices = {
        int(x["train_index"])
        for x in (
            wrong_pool
            + correct_pool
        )
    }

    overlap = (
        warm_indices
        & candidate_indices
    )

    if overlap:
        raise RuntimeError(
            f"Warm-start/DPO candidate overlap: "
            f"{sorted(overlap)[:20]}"
        )

    print(
        "Warm-start/DPO disjointness: PASS"
    )

    # --------------------------------------------------------------
    # Validate source labels
    # --------------------------------------------------------------

    bad_wrong = [
        x
        for x in wrong_pool
        if x["prediction"]
        == x["answer_idx"]
    ]

    bad_correct = [
        x
        for x in correct_pool
        if x["prediction"]
        != x["answer_idx"]
    ]

    if bad_wrong:
        raise RuntimeError(
            f"{len(bad_wrong)} wrong-pool rows "
            f"are actually correct."
        )

    if bad_correct:
        raise RuntimeError(
            f"{len(bad_correct)} correct-pool rows "
            f"are actually wrong."
        )

    print(
        "Correct/wrong source labels : PASS"
    )

    # --------------------------------------------------------------
    # Create nested source samples
    #
    # Wrong pool:
    #     confidence-weighted without replacement.
    #
    # Correct pool:
    #     uniform without replacement.
    #
    # We draw the MAX required sizes once, then each ratio uses a prefix.
    # Therefore source sets are nested across configurations.
    # --------------------------------------------------------------

    wrong_conf = np.array(
        [
            float(x["confidence"])
            for x in wrong_pool
        ],
        dtype=float,
    )

    if np.any(
        ~np.isfinite(wrong_conf)
    ):
        raise RuntimeError(
            "Non-finite wrong-pool confidence."
        )

    if np.any(
        wrong_conf < 0
    ):
        raise RuntimeError(
            "Negative confidence found."
        )

    if wrong_conf.sum() <= 0:
        raise RuntimeError(
            "Wrong-pool confidence sum is zero."
        )

    wrong_probs = (
        wrong_conf
        / wrong_conf.sum()
    )

    # Independent RNG streams.
    wrong_rng = np.random.default_rng(
        SEED + 100
    )

    correct_rng = np.random.default_rng(
        SEED + 200
    )

    max_wrong = max(
        x["n_abstain"]
        for x in RATIO_CONFIGS.values()
    )

    max_correct = max(
        x["n_answer"]
        for x in RATIO_CONFIGS.values()
    )

    wrong_order = wrong_rng.choice(
        len(wrong_pool),
        size=max_wrong,
        replace=False,
        p=wrong_probs,
    )

    correct_order = correct_rng.choice(
        len(correct_pool),
        size=max_correct,
        replace=False,
    )

    sampled_wrong = [
        wrong_pool[i]
        for i in wrong_order
    ]

    sampled_correct = [
        correct_pool[i]
        for i in correct_order
    ]

    # --------------------------------------------------------------
    # Build each ratio configuration
    # --------------------------------------------------------------

    print(
        "\n" + "-" * 76
    )

    print(
        "PAIR CONFIGURATIONS"
    )

    print(
        "-" * 76
    )

    for config_num, (
        run_name,
        config,
    ) in enumerate(
        RATIO_CONFIGS.items()
    ):

        n_abstain = config[
            "n_abstain"
        ]

        n_answer = config[
            "n_answer"
        ]

        if (
            n_abstain
            + n_answer
            != 2400
        ):
            raise RuntimeError(
                f"{run_name} does not contain "
                f"exactly 2400 pairs."
            )

        # Nested prefixes.
        wrong_sources = (
            sampled_wrong[
                :n_abstain
            ]
        )

        correct_sources = (
            sampled_correct[
                :n_answer
            ]
        )

        abstain_pairs = [
            make_abstain_pair(x)
            for x in wrong_sources
        ]

        answer_pairs = [
            make_answer_pair(x)
            for x in correct_sources
        ]

        # ----------------------------------------------------------
        # Source uniqueness
        # ----------------------------------------------------------

        source_indices = [
            p["source_train_index"]
            for p in (
                abstain_pairs
                + answer_pairs
            )
        ]

        if len(
            set(source_indices)
        ) != 2400:
            raise RuntimeError(
                f"{run_name}: duplicate "
                f"source examples."
            )

        if (
            set(source_indices)
            & warm_indices
        ):
            raise RuntimeError(
                f"{run_name}: warm-start "
                f"overlap found."
            )

        # ----------------------------------------------------------
        # Exact stratified 90/10 split
        # ----------------------------------------------------------

        abstain_train, abstain_val = (
            split_type_exact(
                abstain_pairs,
                SEED
                + 1000
                + config_num,
            )
        )

        answer_train, answer_val = (
            split_type_exact(
                answer_pairs,
                SEED
                + 2000
                + config_num,
            )
        )

        train = (
            abstain_train
            + answer_train
        )

        val = (
            abstain_val
            + answer_val
        )

        # Final shuffle inside each split.
        train_rng = np.random.default_rng(
            SEED
            + 3000
            + config_num
        )

        val_rng = np.random.default_rng(
            SEED
            + 4000
            + config_num
        )

        train = [
            train[i]
            for i in train_rng.permutation(
                len(train)
            )
        ]

        val = [
            val[i]
            for i in val_rng.permutation(
                len(val)
            )
        ]

        # ----------------------------------------------------------
        # Split overlap checks
        # ----------------------------------------------------------

        train_indices = {
            p["source_train_index"]
            for p in train
        }

        val_indices = {
            p["source_train_index"]
            for p in val
        }

        if (
            train_indices
            & val_indices
        ):
            raise RuntimeError(
                f"{run_name}: train/val overlap."
            )

        if len(
            train_indices
            | val_indices
        ) != 2400:
            raise RuntimeError(
                f"{run_name}: train+val "
                f"does not recover 2400 sources."
            )

        # ----------------------------------------------------------
        # Counts
        # ----------------------------------------------------------

        train_abstain = sum(
            p["type"] == "abstain"
            for p in train
        )

        train_answer = (
            len(train)
            - train_abstain
        )

        val_abstain = sum(
            p["type"] == "abstain"
            for p in val
        )

        val_answer = (
            len(val)
            - val_abstain
        )

        abstain_stats = (
            confidence_stats(
                abstain_pairs
            )
        )

        answer_stats = (
            confidence_stats(
                answer_pairs
            )
        )

        ratio = (
            n_abstain
            / n_answer
        )

        # ----------------------------------------------------------
        # Save
        # ----------------------------------------------------------

        output = {
            "protocol":
                "clean_dpo_pairs_v1",

            "source_protocol":
                "clean_oof_v1",

            "seed":
                SEED,

            "run_name":
                run_name,

            "total_pairs":
                2400,

            "ratio_abstain_to_answer":
                ratio,

            "n_abstain":
                n_abstain,

            "n_answer":
                n_answer,

            "sampling": {
                "wrong":
                    (
                        "confidence_weighted_"
                        "without_replacement"
                    ),

                "correct":
                    (
                        "uniform_"
                        "without_replacement"
                    ),

                "nested_across_ratios":
                    True,
            },

            "split": {
                "val_fraction":
                    VAL_FRAC,

                "stratified_by_type":
                    True,

                "train_n":
                    len(train),

                "val_n":
                    len(val),

                "train_abstain":
                    train_abstain,

                "train_answer":
                    train_answer,

                "val_abstain":
                    val_abstain,

                "val_answer":
                    val_answer,
            },

            "confidence_stats": {
                "abstain_sources":
                    abstain_stats,

                "answer_sources":
                    answer_stats,

                "all_wrong_candidate_mean":
                    float(
                        wrong_conf.mean()
                    ),
            },

            "train":
                train,

            "val":
                val,
        }

        output_file = (
            DATA_DIR
            / f"dpo_pairs_{run_name}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
            )

        # ----------------------------------------------------------
        # Print report
        # ----------------------------------------------------------

        print(
            f"\n{run_name}"
        )

        print(
            f"  Ratio              : "
            f"{ratio:.2f}:1"
        )

        print(
            f"  Abstain / answer   : "
            f"{n_abstain} / {n_answer}"
        )

        print(
            f"  Train / val        : "
            f"{len(train)} / {len(val)}"
        )

        print(
            f"  Train types        : "
            f"abstain={train_abstain}, "
            f"answer={train_answer}"
        )

        print(
            f"  Val types          : "
            f"abstain={val_abstain}, "
            f"answer={val_answer}"
        )

        print(
            f"  Abstain conf mean  : "
            f"{abstain_stats['mean']:.4f}"
        )

        print(
            f"  Answer conf mean   : "
            f"{answer_stats['mean']:.4f}"
        )

        print(
            f"  All-wrong mean     : "
            f"{wrong_conf.mean():.4f}"
        )

        print(
            f"  Unique sources     : "
            f"{len(set(source_indices))}"
        )

        print(
            f"  Warm-start overlap : 0"
        )

        print(
            f"  Saved -> "
            f"{output_file}"
        )

    # --------------------------------------------------------------
    # Cross-configuration nesting checks
    # --------------------------------------------------------------

    set_1200_wrong = {
        int(x["train_index"])
        for x in sampled_wrong[:1200]
    }

    set_1440_wrong = {
        int(x["train_index"])
        for x in sampled_wrong[:1440]
    }

    set_1600_wrong = {
        int(x["train_index"])
        for x in sampled_wrong[:1600]
    }

    if not (
        set_1200_wrong
        <= set_1440_wrong
        <= set_1600_wrong
    ):
        raise RuntimeError(
            "Wrong-source nesting failed."
        )

    set_800_correct = {
        int(x["train_index"])
        for x in sampled_correct[:800]
    }

    set_960_correct = {
        int(x["train_index"])
        for x in sampled_correct[:960]
    }

    set_1200_correct = {
        int(x["train_index"])
        for x in sampled_correct[:1200]
    }

    if not (
        set_800_correct
        <= set_960_correct
        <= set_1200_correct
    ):
        raise RuntimeError(
            "Correct-source nesting failed."
        )

    print(
        "\n" + "=" * 76
    )

    print(
        "CROSS-CONFIGURATION CHECKS"
    )

    print(
        "=" * 76
    )

    print(
        "Wrong-source nesting   : PASS"
    )

    print(
        "Correct-source nesting : PASS"
    )

    print(
        "Warm-start disjoint    : PASS"
    )

    print(
        "Fixed 2400 pair count  : PASS"
    )

    print(
        "\nCLEAN DPO PAIR BUILD: PASS"
    )


if __name__ == "__main__":
    main()
