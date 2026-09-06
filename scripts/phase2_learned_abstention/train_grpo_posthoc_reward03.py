"""
train_clean_grpo.py

Clean symmetric GRPO experiment.

Both arms use:
- identical common initialization
- identical 2,160 TRAIN-only questions
- identical seed
- identical optimization budget
- identical G=8 sampling
- identical temperature
- identical LR/scheduler
- identical parser

ONLY difference:

Arm A:
    correct   +1
    wrong     -1
    abstain   -1
    malformed -1

Arm B:
    correct   +1
    wrong     -1
    abstain    0
    malformed -1

No DEV or TEST is loaded during training.

Checkpoint selection happens later on DEV using the preregistered
wrongness-AUROC rule.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.medqa_data import load_medqa
from scripts.phase2_learned_abstention.grpo_completion_parser import (
    classify_completion,
)


# ---------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

COMMON_INIT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo_common_warmstart"
    / "main"
    / "policy"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo"
)

MANIFEST_FILE = (
    OUTPUT_ROOT
    / "data"
    / "train_2160_seed42.json"
)

N_TRAIN = 2160
SEED = 42

GROUP_SIZE = 8

PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4

LEARNING_RATE = 3e-6

MAX_PROMPT_LENGTH = 384
MAX_COMPLETION_LENGTH = 16

TEMPERATURE = 1.0

EPOCHS = 1

SAVE_STEPS = 50

ANSWER_LABELS = [
    "A",
    "B",
    "C",
    "D",
]


ARM_REWARDS = {
    "A": {
        "correct": 1.0,
        "wrong": -1.0,
        "abstain": -1.0,
        "malformed": -1.0,
    },

    "B": {
        "correct": 1.0,
        "wrong": -1.0,
        "abstain": 0.3,
        "malformed": -1.0,
    },
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arm",
        choices=["A", "B"],
        required=True,
    )

    parser.add_argument(
        "--run-name",
        default="main",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# ---------------------------------------------------------------------
# Frozen TRAIN subset
# ---------------------------------------------------------------------

def build_or_verify_manifest():

    train = load_medqa(
        "train",
        allow_test=False,
    )

    if len(train) != 10178:

        raise RuntimeError(
            f"Expected 10178 TRAIN examples, "
            f"got {len(train)}"
        )

    rng = np.random.default_rng(
        SEED
    )

    permutation = rng.permutation(
        len(train)
    )

    selected_indices = [
        int(i)
        for i
        in permutation[:N_TRAIN]
    ]

    selected_ids = [
        str(
            train[i]["id"]
        )
        for i
        in selected_indices
    ]

    expected = {
        "protocol":
            "clean_grpo_train_subset_v1",

        "split":
            "train",

        "n_train":
            N_TRAIN,

        "seed":
            SEED,

        "selection":
            (
                "numpy default_rng(seed=42) "
                "permutation first 2160"
            ),

        "indices":
            selected_indices,

        "ids":
            selected_ids,

        "dev_used":
            False,

        "test_used":
            False,
    }

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MANIFEST_FILE.exists():

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(f)

        if existing != expected:

            raise RuntimeError(
                "Existing GRPO train manifest "
                "does not match frozen selection."
            )

        print(
            "Frozen TRAIN manifest: VERIFIED"
        )

    else:

        with open(
            MANIFEST_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                expected,
                f,
                indent=2,
            )

        print(
            "Frozen TRAIN manifest: CREATED"
        )

    selected = [
        train[i]
        for i
        in selected_indices
    ]

    return selected


def make_dataset(
    examples,
    smoke,
):

    if smoke:
        examples = examples[:16]

    records = []

    for ex in examples:

        records.append(
            {
                "prompt":
                    build_prompt(
                        ex["question"],
                        ex["options"],
                    ),

                "ground_truth":
                    normalize_gold(
                        ex["answer_idx"]
                    ),

                "example_id":
                    str(ex["id"]),
            }
        )

    return Dataset.from_list(
        records
    )


# ---------------------------------------------------------------------
# Reward diagnostics
# ---------------------------------------------------------------------

class RewardDiagnostics:

    def __init__(
        self,
        arm,
        group_size,
    ):

        self.arm = arm

        self.reward_map = (
            ARM_REWARDS[arm]
        )

        self.group_size = (
            group_size
        )

        self.calls = 0

        self.completion_counts = Counter()

        self.groups = 0

        self.groups_any_abstain = 0
        self.groups_answer_and_abstain = 0
        self.groups_all_answer = 0
        self.groups_all_abstain = 0
        self.groups_any_malformed = 0

        self.zero_variance_groups = 0

        self.reward_std_sum = 0.0

    def _expand_gold(
        self,
        ground_truth,
        n_completions,
    ):

        gt = list(
            ground_truth
        )

        if len(gt) == n_completions:
            return gt

        if (
            len(gt)
            * self.group_size
            == n_completions
        ):

            expanded = []

            for value in gt:

                expanded.extend(
                    [value]
                    * self.group_size
                )

            return expanded

        raise RuntimeError(
            "Unexpected ground_truth/completion "
            f"lengths: {len(gt)} vs "
            f"{n_completions}"
        )

    def __call__(
        self,
        prompts=None,
        completions=None,
        ground_truth=None,
        **kwargs,
    ):

        if completions is None:
            raise RuntimeError(
                "completions is None"
            )

        if ground_truth is None:
            raise RuntimeError(
                "ground_truth is None"
            )

        completions = list(
            completions
        )

        golds = self._expand_gold(
            ground_truth,
            len(completions),
        )

        if (
            len(completions)
            % self.group_size
            != 0
        ):

            raise RuntimeError(
                "Completion count is not "
                "divisible by G."
            )

        classes = [
            classify_completion(
                text,
                gold,
            )
            for text, gold
            in zip(
                completions,
                golds,
            )
        ]

        rewards = [
            self.reward_map[c]
            for c in classes
        ]

        self.completion_counts.update(
            classes
        )

        n_groups = (
            len(classes)
            // self.group_size
        )

        for g in range(
            n_groups
        ):

            start = (
                g
                * self.group_size
            )

            end = (
                start
                + self.group_size
            )

            group_classes = (
                classes[start:end]
            )

            group_rewards = np.asarray(
                rewards[start:end],
                dtype=float,
            )

            n_answer = sum(
                c in {
                    "correct",
                    "wrong",
                }
                for c
                in group_classes
            )

            n_abstain = (
                group_classes.count(
                    "abstain"
                )
            )

            n_malformed = (
                group_classes.count(
                    "malformed"
                )
            )

            self.groups += 1

            if n_abstain > 0:
                self.groups_any_abstain += 1

            if (
                n_answer > 0
                and
                n_abstain > 0
            ):
                self.groups_answer_and_abstain += 1

            if (
                n_answer
                == self.group_size
            ):
                self.groups_all_answer += 1

            if (
                n_abstain
                == self.group_size
            ):
                self.groups_all_abstain += 1

            if n_malformed > 0:
                self.groups_any_malformed += 1

            std = float(
                group_rewards.std()
            )

            self.reward_std_sum += std

            if std <= 1e-12:
                self.zero_variance_groups += 1

        self.calls += 1

        if (
            self.calls <= 5
            or
            self.calls % 10 == 0
        ):

            counts = dict(
                Counter(classes)
            )

            r = np.asarray(
                rewards,
                dtype=float,
            )

            print(
                f"[ARM {self.arm} reward "
                f"call {self.calls}] "
                f"mean={r.mean():.3f} "
                f"std={r.std():.3f} "
                f"classes={counts}"
            )

        return rewards

    def summary(self):

        groups = max(
            self.groups,
            1,
        )

        zero = (
            self.zero_variance_groups
        )

        return {
            "arm":
                self.arm,

            "reward_map":
                self.reward_map,

            "reward_calls":
                self.calls,

            "completion_counts":
                dict(
                    self.completion_counts
                ),

            "groups":
                self.groups,

            "groups_any_abstain":
                self.groups_any_abstain,

            "groups_answer_and_abstain":
                self.groups_answer_and_abstain,

            "groups_all_answer":
                self.groups_all_answer,

            "groups_all_abstain":
                self.groups_all_abstain,

            "groups_any_malformed":
                self.groups_any_malformed,

            "zero_reward_variance_groups":
                zero,

            "nonzero_advantage_groups":
                self.groups - zero,

            "zero_reward_variance_rate":
                float(
                    zero / groups
                ),

            "nonzero_advantage_rate":
                float(
                    (
                        self.groups
                        - zero
                    )
                    / groups
                ),

            "mean_group_reward_std":
                float(
                    self.reward_std_sum
                    / groups
                ),
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    seed_everything(
        SEED
    )

    print("=" * 88)
    print("CLEAN GRPO")
    print("=" * 88)

    print(
        f"\nArm                   : "
        f"{args.arm}"
    )

    print(
        f"Reward map            : "
        f"{ARM_REWARDS[args.arm]}"
    )

    print(
        f"Common initialization : "
        f"{COMMON_INIT}"
    )

    print(
        f"G                     : "
        f"{GROUP_SIZE}"
    )

    print(
        f"Temperature           : "
        f"{TEMPERATURE}"
    )

    print(
        f"Learning rate         : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Effective prompt batch: "
        f"{PER_DEVICE_BATCH * GRAD_ACCUM}"
    )

    print(
        f"Dataset               : "
        f"TRAIN ONLY"
    )

    print(
        "DEV/TEST loaded       : NO"
    )

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------

    examples = (
        build_or_verify_manifest()
    )

    dataset = make_dataset(
        examples,
        args.smoke,
    )

    print(
        f"Training prompts      : "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    arm_name = (
        "arm_a_answer_only"
        if args.arm == "A"
        else "posthoc_arm_b_reward03"
    )

    run_root = (
        OUTPUT_ROOT
        / arm_name
        / args.run_name
    )

    if run_root.exists():

        raise RuntimeError(
            f"Output already exists:\n"
            f"{run_root}"
        )

    checkpoint_root = (
        run_root
        / "checkpoints"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = (
        "left"
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print(
        "\nLoading Mistral-7B bf16..."
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,

            torch_dtype=(
                torch.bfloat16
            ),

            device_map="auto",
        )
    )

    base.config.use_cache = False

    print(
        "Loading common initialization "
        "as trainable LoRA..."
    )

    model = (
        PeftModel.from_pretrained(
            base,
            str(COMMON_INIT),
            is_trainable=True,
        )
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    # --------------------------------------------------------------
    # Reward
    # --------------------------------------------------------------

    reward_diagnostics = (
        RewardDiagnostics(
            arm=args.arm,
            group_size=GROUP_SIZE,
        )
    )

    # --------------------------------------------------------------
    # Config
    # --------------------------------------------------------------

    config_kwargs = dict(
        output_dir=str(
            checkpoint_root
        ),

        per_device_train_batch_size=(
            PER_DEVICE_BATCH
        ),

        gradient_accumulation_steps=(
            GRAD_ACCUM
        ),

        num_generations=(
            GROUP_SIZE
        ),

        max_prompt_length=(
            MAX_PROMPT_LENGTH
        ),

        max_completion_length=(
            MAX_COMPLETION_LENGTH
        ),

        temperature=(
            TEMPERATURE
        ),

        beta=0.0,

        learning_rate=(
            LEARNING_RATE
        ),

        lr_scheduler_type="cosine",

        warmup_ratio=0.05,

        num_train_epochs=(
            EPOCHS
        ),

        logging_steps=1,

        save_strategy="steps",

        save_steps=(
            SAVE_STEPS
        ),

        save_total_limit=None,

        bf16=True,

        fp16=False,

        gradient_checkpointing=False,

        report_to="none",

        seed=SEED,
    )

    if args.smoke:

        config_kwargs[
            "max_steps"
        ] = 2

        config_kwargs[
            "save_steps"
        ] = 1

    cfg = GRPOConfig(
        **config_kwargs
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    # TRL 0.14 expects reward functions to expose __name__.
    # Keep RewardDiagnostics as the stateful accumulator, but expose
    # it through a normal named function.
    def grpo_reward_func(
        prompts=None,
        completions=None,
        ground_truth=None,
        **kwargs,
    ):
        return reward_diagnostics(
            prompts=prompts,
            completions=completions,
            ground_truth=ground_truth,
            **kwargs,
        )

    grpo_reward_func.__name__ = (
        f"medqa_arm_{args.arm.lower()}_reward"
    )

    trainer = GRPOTrainer(
        model=model,

        reward_funcs=[
            grpo_reward_func
        ],

        args=cfg,

        train_dataset=dataset,

        processing_class=tokenizer,

        # IMPORTANT:
        # common LoRA is already loaded trainable.
        # Do not create a second PEFT adapter.
        peft_config=None,
    )

    print(
        "\nStarting GRPO training..."
    )

    torch.cuda.reset_peak_memory_stats()

    result = trainer.train()

    print(
        "\nGRPO TRAINING COMPLETE"
    )

    print(
        result.metrics
    )

    # --------------------------------------------------------------
    # Save exact final state
    # --------------------------------------------------------------

    final_dir = (
        run_root
        / "final"
    )

    trainer.save_model(
        str(final_dir)
    )

    tokenizer.save_pretrained(
        str(final_dir)
    )

    diagnostics = (
        reward_diagnostics.summary()
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    print("\n" + "=" * 88)
    print("FINAL REWARD DIAGNOSTICS")
    print("=" * 88)

    for key, value in (
        diagnostics.items()
    ):

        print(
            f"{key}: {value}"
        )

    print("\nGPU MEMORY")

    print(
        f"Peak allocated: "
        f"{peak_allocated:.2f} GB"
    )

    print(
        f"Peak reserved : "
        f"{peak_reserved:.2f} GB"
    )

    summary = {
        "protocol":
            "posthoc_grpo_abstain_reward03_v1",

        "arm":
            args.arm,

        "smoke":
            args.smoke,

        "common_initialization":
            str(COMMON_INIT),

        "train_manifest":
            str(MANIFEST_FILE),

        "n_train_prompts":
            len(dataset),

        "num_generations":
            GROUP_SIZE,

        "temperature":
            TEMPERATURE,

        "max_prompt_length":
            MAX_PROMPT_LENGTH,

        "max_completion_length":
            MAX_COMPLETION_LENGTH,

        "learning_rate":
            LEARNING_RATE,

        "beta":
            0.0,

        "per_device_batch":
            PER_DEVICE_BATCH,

        "gradient_accumulation":
            GRAD_ACCUM,

        "effective_prompt_batch":
            (
                PER_DEVICE_BATCH
                * GRAD_ACCUM
            ),

        "seed":
            SEED,

        "reward_diagnostics":
            diagnostics,

        "trainer_metrics":
            result.metrics,

        "gpu": {
            "name":
                torch.cuda.get_device_name(0),

            "peak_allocated_gb":
                float(
                    peak_allocated
                ),

            "peak_reserved_gb":
                float(
                    peak_reserved
                ),
        },

        "dev_used_for_training":
            False,

        "test_used":
            False,
    }

    with open(
        run_root
        / "train_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\nFinal adapter ->"
    )

    print(
        final_dir
    )

    if args.smoke:

        print(
            "\nGRPO SMOKE: PASS"
        )

    else:

        print(
            "\nCLEAN GRPO RUN: PASS"
        )


if __name__ == "__main__":
    main()
