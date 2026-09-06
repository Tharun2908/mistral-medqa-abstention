"""
train_correct_only_sft_control.py
---------------------------------

Matched-step control for the supervised 5-way abstention experiment.

Question:
    Do the OOF-wrong examples labeled E provide useful training signal,
    or can the same optimization budget spent only on OOF-correct
    A-D answer examples match the 5-way model?

Data:
    Start from supervised_5way_data.json.

    Keep ONLY examples whose 5-way target is A/B/C/D.
    These are exactly the examples on which OOF SFT was correct.

    Expected:
        total correct examples = 5219
        train correct examples = ~4697
        val correct examples   = ~522

Targets:
    Always the gold A-D answer.

Initialization:
    Original SFT adapter:
        Primeinvincible/mistral-medqa-lora-v3

Training recipe matched to supervised 5-way:
    LR                  = 5e-6
    batch size          = 4
    grad accumulation   = 4
    scheduler           = cosine
    warmup ratio        = 0.05
    seed                = 42
    completion-only loss
    full-sentence answer format

CRITICAL COMPUTE MATCH:
    Supervised 5-way:
        scheduler max_steps = 1144
        selected checkpoint = step 1000

    Therefore this control:
        constructs scheduler with max_steps=1144
        deliberately stops at step 1000

The smaller correct-only dataset is automatically cycled through
multiple epochs by Trainer until step 1000.

Official MedQA dev/test are NOT used for parameter learning.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

ORIGINAL_SFT_ADAPTER = (
    "Primeinvincible/mistral-medqa-lora-v3"
)

DATA_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
    / "supervised_5way_data.json"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "correct_only_sft_control"
)

ANSWER_LABELS = {"A", "B", "C", "D"}

SEED = 42

LEARNING_RATE = 5e-6

PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4

MAX_LENGTH = 1024

# Exact horizon from supervised 5-way trainer_state.json.
SCHEDULER_MAX_STEPS = 1144

# Exact selected supervised comparison point.
STOP_AT_STEP = 1000


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(question, options):

    option_lines = "\n".join(
        f"{label}: {options[label]}"
        for label in ["A", "B", "C", "D"]
    )

    return (
        f"Question: {question}\n\n"
        f"Options:\n"
        f"{option_lines}\n\n"
        f"Answer:"
    )


def build_completion(label):

    return (
        f" The answer is {label}."
    )


def load_correct_only_split():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    protocol = data.get("protocol")

    if protocol != "clean_supervised_5way_v1":
        raise RuntimeError(
            "Unexpected supervised dataset protocol: "
            f"{protocol}"
        )

    train_raw = data["train"]
    val_raw = data["val"]

    train = [
        row
        for row in train_raw
        if row["target_label"]
        in ANSWER_LABELS
    ]

    val = [
        row
        for row in val_raw
        if row["target_label"]
        in ANSWER_LABELS
    ]

    total = (
        len(train)
        + len(val)
    )

    print("=" * 80)
    print("CORRECT-ONLY CONTROL DATA")
    print("=" * 80)

    print(
        f"Original 5-way train : "
        f"{len(train_raw)}"
    )

    print(
        f"Original 5-way val   : "
        f"{len(val_raw)}"
    )

    print(
        f"Correct-only train   : "
        f"{len(train)}"
    )

    print(
        f"Correct-only val     : "
        f"{len(val)}"
    )

    print(
        f"Correct-only total   : "
        f"{total}"
    )

    if total != 5219:
        raise RuntimeError(
            "Expected exactly 5219 "
            f"OOF-correct examples, got {total}"
        )

    # Important:
    # these are not newly relabeled examples.
    # Their original supervised target must already equal gold A-D.
    for split_name, rows in [
        ("train", train),
        ("val", val),
    ]:

        for row in rows:

            target = row["target_label"]
            gold = row["gold_answer"]

            if target != gold:
                raise RuntimeError(
                    f"{split_name}: "
                    "A-D target != gold answer "
                    f"for id={row.get('id')}: "
                    f"target={target}, gold={gold}"
                )

    print(
        "Target/gold consistency: PASS"
    )

    return train, val


def tokenize_rows(
    rows,
    tokenizer,
):

    records = []

    for row in rows:

        prompt = build_prompt(
            row["question"],
            row["options"],
        )

        completion = build_completion(
            row["gold_answer"]
        )

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
        )["input_ids"]

        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
        )["input_ids"]

        # Add EOS.
        completion_ids = (
            completion_ids
            + [tokenizer.eos_token_id]
        )

        # Preserve completion completely.
        prompt_budget = (
            MAX_LENGTH
            - len(completion_ids)
        )

        if prompt_budget <= 0:
            raise RuntimeError(
                "Completion exceeds MAX_LENGTH."
            )

        if len(prompt_ids) > prompt_budget:
            prompt_ids = (
                prompt_ids[-prompt_budget:]
            )

        input_ids = (
            prompt_ids
            + completion_ids
        )

        attention_mask = (
            [1] * len(input_ids)
        )

        # Completion-only loss.
        labels = (
            [-100] * len(prompt_ids)
            + completion_ids
        )

        records.append(
            {
                "input_ids":
                    input_ids,

                "attention_mask":
                    attention_mask,

                "labels":
                    labels,
            }
        )

    return Dataset.from_list(
        records
    )


# ---------------------------------------------------------------------
# Stop exactly at selected comparison checkpoint.
# ---------------------------------------------------------------------

class StopAtStepCallback(
    TrainerCallback
):

    def __init__(
        self,
        stop_step,
    ):

        self.stop_step = (
            stop_step
        )

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):

        if (
            state.global_step
            >= self.stop_step
        ):

            control.should_training_stop = (
                True
            )

        return control


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="main",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(
        SEED
    )

    run_root = (
        OUTPUT_ROOT
        / args.run_name
    )

    checkpoint_dir = (
        run_root
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_rows, val_rows = (
        load_correct_only_split()
    )

    if args.smoke:

        train_rows = (
            train_rows[:64]
        )

        val_rows = (
            val_rows[:40]
        )

        print(
            "\nSMOKE MODE:"
            f" train={len(train_rows)},"
            f" val={len(val_rows)}"
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
        "right"
    )

    train_dataset = tokenize_rows(
        train_rows,
        tokenizer,
    )

    val_dataset = tokenize_rows(
        val_rows,
        tokenizer,
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=(
                torch.float16
            ),

            bnb_4bit_use_double_quant=True,
        )
    )

    print(
        "\nLoading base model..."
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,

            quantization_config=(
                bnb_config
            ),

            device_map="auto",
        )
    )

    base = (
        prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
        )
    )

    print(
        "Loading original SFT adapter..."
    )

    model = (
        PeftModel.from_pretrained(
            base,
            ORIGINAL_SFT_ADAPTER,
            is_trainable=True,
        )
    )

    model.config.use_cache = False

    # --------------------------------------------------------------
    # Collator
    # --------------------------------------------------------------

    collator = (
        DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    if args.smoke:

        # Smoke only verifies code path.
        max_steps = 2
        stop_step = 2
        save_steps = 2
        eval_steps = 2
        warmup_ratio = 0.0

    else:

        # CRITICAL:
        #
        # max_steps controls scheduler horizon.
        # It MUST stay 1144 to match supervised 5-way.
        #
        # callback stops optimization at step 1000.
        max_steps = (
            SCHEDULER_MAX_STEPS
        )

        stop_step = (
            STOP_AT_STEP
        )

        save_steps = 100
        eval_steps = 100
        warmup_ratio = 0.05

    training_args = (
        TrainingArguments(
            output_dir=str(
                checkpoint_dir
            ),

            max_steps=max_steps,

            per_device_train_batch_size=(
                PER_DEVICE_BATCH
            ),

            per_device_eval_batch_size=(
                PER_DEVICE_BATCH
            ),

            gradient_accumulation_steps=(
                GRAD_ACCUM
            ),

            learning_rate=(
                LEARNING_RATE
            ),

            lr_scheduler_type="cosine",

            warmup_ratio=(
                warmup_ratio
            ),

            evaluation_strategy="steps",

            eval_steps=(
                eval_steps
            ),

            save_strategy="steps",

            save_steps=(
                save_steps
            ),

            logging_steps=10,

            save_total_limit=20,

            load_best_model_at_end=False,

            bf16=False,
            fp16=False,

            gradient_checkpointing=True,

            report_to="none",

            seed=SEED,


            remove_unused_columns=False,
        )
    )

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        data_collator=collator,

        tokenizer=tokenizer,

        callbacks=[
            StopAtStepCallback(
                stop_step
            )
        ],
    )

    print("\n" + "=" * 80)
    print("CORRECT-ONLY MATCHED-STEP SFT")
    print("=" * 80)

    print(
        f"Scheduler horizon : "
        f"{max_steps}"
    )

    print(
        f"Stop step         : "
        f"{stop_step}"
    )

    print(
        f"Learning rate     : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Train examples    : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val examples      : "
        f"{len(val_dataset)}"
    )

    print(
        f"Effective batch   : "
        f"{PER_DEVICE_BATCH * GRAD_ACCUM}"
    )

    result = trainer.train()

    print("\nTraining finished.")

    print(
        f"Final global step : "
        f"{trainer.state.global_step}"
    )

    if (
        trainer.state.global_step
        != stop_step
    ):
        raise RuntimeError(
            "Expected training to stop "
            f"at step {stop_step}, got "
            f"{trainer.state.global_step}"
        )

    # --------------------------------------------------------------
    # Final evaluation at exactly step 1000.
    # --------------------------------------------------------------

    metrics = trainer.evaluate()

    print(
        "\nFinal internal validation:"
    )

    for key, value in (
        metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    # Save explicit final adapter.
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

    summary = {
        "protocol":
            "correct_only_sft_matched_step_v1",

        "source_dataset":
            str(DATA_FILE),

        "n_train":
            len(train_rows),

        "n_val":
            len(val_rows),

        "target_policy":
            (
                "keep only supervised-5way "
                "A-D examples; train gold A-D"
            ),

        "initial_adapter":
            ORIGINAL_SFT_ADAPTER,

        "learning_rate":
            LEARNING_RATE,

        "scheduler":
            "cosine",

        "scheduler_horizon_steps":
            max_steps,

        "selected_comparison_step":
            stop_step,

        "actual_final_step":
            trainer.state.global_step,

        "per_device_batch":
            PER_DEVICE_BATCH,

        "gradient_accumulation":
            GRAD_ACCUM,

        "effective_batch":
            (
                PER_DEVICE_BATCH
                * GRAD_ACCUM
            ),

        "seed":
            SEED,

        "evaluation_metrics":
            metrics,

        "official_dev_used_for_training":
            False,

        "official_test_used":
            False,
    }

    with open(
        run_root
        / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\nSaved final adapter:"
    )

    print(
        final_dir
    )

    if args.smoke:
        print(
            "\nCORRECT-ONLY SFT SMOKE: PASS"
        )
    else:
        print(
            "\nCORRECT-ONLY SFT CONTROL: PASS"
        )


if __name__ == "__main__":
    main()
