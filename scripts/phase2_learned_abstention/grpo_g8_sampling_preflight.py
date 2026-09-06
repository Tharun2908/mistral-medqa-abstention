"""
grpo_g8_sampling_preflight.py
-----------------------------

TRAIN-ONLY sampling preflight for the clean GRPO experiment.

Purpose
-------
Before spending compute on GRPO, verify that the common E-aware
initialization actually samples enough abstentions under the exact
planned generation regime:

    G = 8
    temperature = 1.0
    top_p = 1.0
    max completion length = 16

No training occurs.
Official MedQA DEV and TEST are never loaded.

We sample a fixed set of 256 TRAIN questions and generate 8 completions
per question.

Diagnostics
-----------
Completion-level:
    correct / wrong / abstain / malformed

Group-level:
    groups with any abstention
    groups with both answer and abstention
    all-answer groups
    all-abstain groups
    all-same-class groups

Reward diagnostics for both preregistered arms:

Arm A — answer-only reward:
    correct   +1
    wrong     -1
    abstain   -1
    malformed -1

Arm B — explicit abstention reward:
    correct   +1
    abstain    0
    wrong     -1
    malformed -1

For each arm:
    zero-reward-variance groups
    non-zero-advantage groups
    mean reward std per group

This is a structural preflight only.
No configuration is selected from DEV/TEST.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Allow direct execution from scripts/phase2_learned_abstention/
REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.common.medqa_data import load_medqa


# ---------------------------------------------------------------------
# Frozen preflight configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

COMMON_ADAPTER = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo_common_warmstart"
    / "main"
    / "policy"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo"
    / "preflight"
    / "g8_sampling_preflight.json"
)

N_PROMPTS = 256

GROUP_SIZE = 8

SEED = 42

TEMPERATURE = 1.0

TOP_P = 1.0

MAX_PROMPT_LENGTH = 384

MAX_COMPLETION_LENGTH = 16

# Small prompt batching; each prompt expands to G=8 generations.
PROMPT_BATCH_SIZE = 4

ANSWER_LABELS = ["A", "B", "C", "D"]


def normalize_gold(value):
    """Normalize MedQA gold answer to A/B/C/D."""

    if isinstance(value, int):
        if 0 <= value <= 3:
            return ANSWER_LABELS[value]

    text = str(value).strip().upper()

    if text in ANSWER_LABELS:
        return text

    if text in {"0", "1", "2", "3"}:
        return ANSWER_LABELS[int(text)]

    raise ValueError(
        f"Cannot normalize gold answer: {value!r}"
    )



# ---------------------------------------------------------------------
# Shared completion parser
# ---------------------------------------------------------------------

from scripts.phase2_learned_abstention.grpo_completion_parser import (
    classify_completion,
)


# ---------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------

ARM_A_REWARD = {
    "correct": 1.0,
    "wrong": -1.0,
    "abstain": -1.0,
    "malformed": -1.0,
}

ARM_B_REWARD = {
    "correct": 1.0,
    "wrong": -1.0,
    "abstain": 0.0,
    "malformed": -1.0,
}


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

def build_prompt(question, options):

    option_lines = "\n".join(
        f"{label}: {options[label]}"
        for label in ANSWER_LABELS
    )

    return (
        f"Question: {question}\n\n"
        f"Options:\n"
        f"{option_lines}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 88)
    print("GRPO G=8 TRAIN-ONLY SAMPLING PREFLIGHT")
    print("=" * 88)

    print(f"\nPrompts               : {N_PROMPTS}")
    print(f"Generations / prompt  : {GROUP_SIZE}")
    print(
        f"Total completions     : "
        f"{N_PROMPTS * GROUP_SIZE}"
    )
    print(f"Temperature           : {TEMPERATURE}")
    print(f"Top-p                 : {TOP_P}")
    print(
        f"Max prompt length     : "
        f"{MAX_PROMPT_LENGTH}"
    )
    print(
        f"Max completion length : "
        f"{MAX_COMPLETION_LENGTH}"
    )

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------------
    # TRAIN ONLY
    # --------------------------------------------------------------

    train = load_medqa(
        "train",
        allow_test=False,
    )

    if len(train) != 10178:
        raise RuntimeError(
            f"Expected 10178 training rows, got {len(train)}"
        )

    rng = np.random.default_rng(
        SEED
    )

    selected_indices = rng.choice(
        len(train),
        size=N_PROMPTS,
        replace=False,
    )

    examples = [
        train[int(i)]
        for i in selected_indices
    ]

    print(
        "\nDataset split: TRAIN ONLY"
    )

    print(
        "DEV/TEST loaded: NO"
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(
        "Loading Mistral-7B bf16..."
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    print(
        "Loading common GRPO initialization..."
    )

    peft_model = (
        PeftModel.from_pretrained(
            base,
            str(COMMON_ADAPTER),
        )
    )

    print(
        "Merging common adapter..."
    )

    model = (
        peft_model
        .merge_and_unload()
    )

    model.eval()

    print(
        "Common initialization load: PASS"
    )

    # --------------------------------------------------------------
    # Sampling
    # --------------------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    completion_counts = Counter()

    group_records = []

    all_completions = 0

    print(
        "\nSampling..."
    )

    for batch_start in range(
        0,
        N_PROMPTS,
        PROMPT_BATCH_SIZE,
    ):

        batch_examples = examples[
            batch_start:
            batch_start + PROMPT_BATCH_SIZE
        ]

        prompts = [
            build_prompt(
                ex["question"],
                ex["options"],
            )
            for ex in batch_examples
        ]

        golds = [
            normalize_gold(
                ex["answer_idx"]
            )
            for ex in batch_examples
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        prompt_width = (
            inputs["input_ids"].shape[1]
        )

        with torch.no_grad():

            generated = model.generate(
                **inputs,

                do_sample=True,

                temperature=TEMPERATURE,

                top_p=TOP_P,

                num_return_sequences=(
                    GROUP_SIZE
                ),

                max_new_tokens=(
                    MAX_COMPLETION_LENGTH
                ),

                pad_token_id=(
                    tokenizer.eos_token_id
                ),

                eos_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        completions = [
            tokenizer.decode(
                seq[prompt_width:],
                skip_special_tokens=True,
            )
            for seq in generated
        ]

        expected = (
            len(batch_examples)
            * GROUP_SIZE
        )

        if len(completions) != expected:
            raise RuntimeError(
                "Unexpected generation count: "
                f"{len(completions)} != {expected}"
            )

        # HF groups returned sequences by source prompt.
        for local_i, ex in enumerate(
            batch_examples
        ):

            start = (
                local_i
                * GROUP_SIZE
            )

            end = (
                start
                + GROUP_SIZE
            )

            group_text = completions[
                start:end
            ]

            gold = golds[local_i]

            classes = [
                classify_completion(
                    text,
                    gold,
                )
                for text
                in group_text
            ]

            completion_counts.update(
                classes
            )

            all_completions += len(
                classes
            )

            arm_a_rewards = np.asarray(
                [
                    ARM_A_REWARD[c]
                    for c in classes
                ],
                dtype=float,
            )

            arm_b_rewards = np.asarray(
                [
                    ARM_B_REWARD[c]
                    for c in classes
                ],
                dtype=float,
            )

            valid_answer_count = sum(
                c in {
                    "correct",
                    "wrong",
                }
                for c in classes
            )

            abstain_count = (
                classes.count(
                    "abstain"
                )
            )

            malformed_count = (
                classes.count(
                    "malformed"
                )
            )

            group_records.append(
                {
                    "id":
                        ex["id"],

                    "gold":
                        gold,

                    "classes":
                        classes,

                    "completions":
                        group_text,

                    "n_answer":
                        valid_answer_count,

                    "n_abstain":
                        abstain_count,

                    "n_malformed":
                        malformed_count,

                    "arm_a_reward_mean":
                        float(
                            arm_a_rewards.mean()
                        ),

                    "arm_a_reward_std":
                        float(
                            arm_a_rewards.std()
                        ),

                    "arm_b_reward_mean":
                        float(
                            arm_b_rewards.mean()
                        ),

                    "arm_b_reward_std":
                        float(
                            arm_b_rewards.std()
                        ),
                }
            )

        done = min(
            batch_start
            + len(batch_examples),
            N_PROMPTS,
        )

        if (
            done % 32 == 0
            or done == N_PROMPTS
        ):

            print(
                f"  {done}/{N_PROMPTS} "
                f"prompts complete"
            )

    # --------------------------------------------------------------
    # Completion-level diagnostics
    # --------------------------------------------------------------

    if all_completions != (
        N_PROMPTS
        * GROUP_SIZE
    ):

        raise RuntimeError(
            "Completion count mismatch."
        )

    print("\n" + "=" * 88)
    print("COMPLETION-LEVEL DIAGNOSTICS")
    print("=" * 88)

    completion_rates = {}

    for cls in [
        "correct",
        "wrong",
        "abstain",
        "malformed",
    ]:

        count = int(
            completion_counts[cls]
        )

        rate = (
            count
            / all_completions
        )

        completion_rates[cls] = (
            float(rate)
        )

        print(
            f"{cls:12}: "
            f"{count:5d} / "
            f"{all_completions} "
            f"= {rate:.2%}"
        )

    # --------------------------------------------------------------
    # Group diagnostics
    # --------------------------------------------------------------

    def count_groups(predicate):

        return sum(
            bool(predicate(g))
            for g in group_records
        )

    groups_any_abstain = (
        count_groups(
            lambda g:
                g["n_abstain"] > 0
        )
    )

    groups_answer_and_abstain = (
        count_groups(
            lambda g:
                g["n_answer"] > 0
                and
                g["n_abstain"] > 0
        )
    )

    groups_all_answer = (
        count_groups(
            lambda g:
                g["n_answer"]
                == GROUP_SIZE
        )
    )

    groups_all_abstain = (
        count_groups(
            lambda g:
                g["n_abstain"]
                == GROUP_SIZE
        )
    )

    groups_any_malformed = (
        count_groups(
            lambda g:
                g["n_malformed"] > 0
        )
    )

    groups_all_same_class = (
        count_groups(
            lambda g:
                len(
                    set(
                        g["classes"]
                    )
                )
                == 1
        )
    )

    print("\n" + "=" * 88)
    print("GROUP-LEVEL ACTION DIVERSITY")
    print("=" * 88)

    group_stats = {
        "groups_any_abstain":
            groups_any_abstain,

        "groups_answer_and_abstain":
            groups_answer_and_abstain,

        "groups_all_answer":
            groups_all_answer,

        "groups_all_abstain":
            groups_all_abstain,

        "groups_any_malformed":
            groups_any_malformed,

        "groups_all_same_class":
            groups_all_same_class,
    }

    for name, count in (
        group_stats.items()
    ):

        print(
            f"{name:30}: "
            f"{count:4d} / "
            f"{N_PROMPTS} "
            f"= {count / N_PROMPTS:.2%}"
        )

    # --------------------------------------------------------------
    # Reward variance diagnostics
    # --------------------------------------------------------------

    arm_a_stds = np.asarray(
        [
            g["arm_a_reward_std"]
            for g in group_records
        ],
        dtype=float,
    )

    arm_b_stds = np.asarray(
        [
            g["arm_b_reward_std"]
            for g in group_records
        ],
        dtype=float,
    )

    eps = 1e-12

    arm_a_zero = int(
        np.sum(
            arm_a_stds <= eps
        )
    )

    arm_b_zero = int(
        np.sum(
            arm_b_stds <= eps
        )
    )

    print("\n" + "=" * 88)
    print("REWARD / ADVANTAGE DIAGNOSTICS")
    print("=" * 88)

    print("\nArm A — answer-only reward")

    print(
        f"zero-reward-variance groups : "
        f"{arm_a_zero}/{N_PROMPTS} "
        f"= {arm_a_zero / N_PROMPTS:.2%}"
    )

    print(
        f"non-zero-advantage groups   : "
        f"{N_PROMPTS - arm_a_zero}/{N_PROMPTS} "
        f"= {(N_PROMPTS - arm_a_zero) / N_PROMPTS:.2%}"
    )

    print(
        f"mean reward std / group     : "
        f"{arm_a_stds.mean():.4f}"
    )

    print(
        "\nArm B — explicit abstention reward"
    )

    print(
        f"zero-reward-variance groups : "
        f"{arm_b_zero}/{N_PROMPTS} "
        f"= {arm_b_zero / N_PROMPTS:.2%}"
    )

    print(
        f"non-zero-advantage groups   : "
        f"{N_PROMPTS - arm_b_zero}/{N_PROMPTS} "
        f"= {(N_PROMPTS - arm_b_zero) / N_PROMPTS:.2%}"
    )

    print(
        f"mean reward std / group     : "
        f"{arm_b_stds.mean():.4f}"
    )

    # --------------------------------------------------------------
    # Memory
    # --------------------------------------------------------------

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    print("\n" + "=" * 88)
    print("GPU MEMORY")
    print("=" * 88)

    print(
        f"Peak allocated : "
        f"{peak_gb:.2f} GB"
    )

    print(
        f"Peak reserved  : "
        f"{reserved_gb:.2f} GB"
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "protocol":
            "clean_grpo_g8_train_only_preflight_v1",

        "split":
            "train",

        "dev_loaded":
            False,

        "test_loaded":
            False,

        "n_prompts":
            N_PROMPTS,

        "num_generations":
            GROUP_SIZE,

        "total_completions":
            all_completions,

        "seed":
            SEED,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "max_prompt_length":
            MAX_PROMPT_LENGTH,

        "max_completion_length":
            MAX_COMPLETION_LENGTH,

        "common_adapter":
            str(COMMON_ADAPTER),

        "completion_counts":
            dict(completion_counts),

        "completion_rates":
            completion_rates,

        "group_stats":
            group_stats,

        "arm_a": {
            "reward":
                ARM_A_REWARD,

            "zero_reward_variance_groups":
                arm_a_zero,

            "nonzero_advantage_groups":
                N_PROMPTS - arm_a_zero,

            "mean_group_reward_std":
                float(
                    arm_a_stds.mean()
                ),
        },

        "arm_b": {
            "reward":
                ARM_B_REWARD,

            "zero_reward_variance_groups":
                arm_b_zero,

            "nonzero_advantage_groups":
                N_PROMPTS - arm_b_zero,

            "mean_group_reward_std":
                float(
                    arm_b_stds.mean()
                ),
        },

        "gpu": {
            "name":
                torch.cuda.get_device_name(0),

            "peak_allocated_gb":
                float(peak_gb),

            "peak_reserved_gb":
                float(reserved_gb),
        },

        # Keep a small audit sample rather than dumping
        # all 2,048 completion strings.
        "audit_groups":
            group_records[:12],
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
        "\nG=8 TRAIN-ONLY PREFLIGHT: PASS"
    )


if __name__ == "__main__":
    main()
