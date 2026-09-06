"""
train_supervised_5way.py
------------------------

Clean supervised 5-way abstention baseline.

Targets:
    A/B/C/D -> " The answer is X."
    E       -> " I cannot answer confidently."

Training labels come from clean OOF SFT correctness:
    OOF prediction correct -> answer
    OOF prediction wrong   -> abstain

Protocol:
    - Starts from the original MedQA SFT adapter.
    - MedQA TRAIN-derived data only.
    - Internal stratified validation only for checkpoint selection.
    - Official MedQA dev/test are untouched during training.
    - Completion-only language-model loss.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

SFT_ADAPTER = (
    "Primeinvincible/mistral-medqa-lora-v3"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    / "supervised_5way"
)

LR = 5e-6
MAX_EPOCHS = 2
MAX_LEN = 1024
SEED = 43


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Completion-only tokenization
# ---------------------------------------------------------------------

def tokenize_example(
    record,
    tokenizer,
):

    prompt_ids = tokenizer(
        record["prompt"],
        add_special_tokens=True,
    ).input_ids

    completion_ids = tokenizer(
        record["completion"],
        add_special_tokens=False,
    ).input_ids

    # Explicit EOS.
    completion_ids = (
        completion_ids
        + [tokenizer.eos_token_id]
    )

    max_prompt_len = (
        MAX_LEN
        - len(completion_ids)
    )

    if max_prompt_len <= 0:
        raise RuntimeError(
            "Completion itself exceeds MAX_LEN."
        )

    # Always preserve the full completion.
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = (
            prompt_ids[-max_prompt_len:]
        )

    input_ids = (
        prompt_ids
        + completion_ids
    )

    labels = (
        [-100] * len(prompt_ids)
        + completion_ids
    )

    return {
        "input_ids":
            input_ids,

        "attention_mask":
            [1] * len(input_ids),

        "labels":
            labels,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    set_seed(SEED)

    run_dir = (
        OUTPUT_ROOT
        / args.run_name
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    final_dir = (
        run_dir
        / "final"
    )

    if final_dir.exists():
        raise RuntimeError(
            f"Final model already exists:\n"
            f"{final_dir}"
        )

    print("=" * 72)
    print("CLEAN SUPERVISED 5-WAY TRAINING")
    print("=" * 72)

    print(f"LR         : {LR}")
    print(f"Max epochs : {MAX_EPOCHS}")
    print(f"Seed       : {SEED}")

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if (
        data.get("protocol")
        != "clean_supervised_5way_v1"
    ):
        raise RuntimeError(
            f"Unexpected protocol: "
            f"{data.get('protocol')}"
        )

    train_rows = data["train"]
    val_rows = data["val"]

    if len(train_rows) != 9160:
        raise RuntimeError(
            f"Expected 9160 train examples, "
            f"got {len(train_rows)}"
        )

    if len(val_rows) != 1018:
        raise RuntimeError(
            f"Expected 1018 val examples, "
            f"got {len(val_rows)}"
        )

    print(
        f"\nTrain examples : "
        f"{len(train_rows)}"
    )

    print(
        f"Val examples   : "
        f"{len(val_rows)}"
    )

    if args.smoke:

        train_rows = train_rows[:80]
        val_rows = val_rows[:40]

        print(
            "\nSMOKE MODE:"
        )

        print(
            f"  train={len(train_rows)}"
        )

        print(
            f"  val={len(val_rows)}"
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

    tokenizer.padding_side = "right"

    print(
        "\n5-way completion tokenization:"
    )

    completions = {
        "A":
            " The answer is A.",

        "B":
            " The answer is B.",

        "C":
            " The answer is C.",

        "D":
            " The answer is D.",

        "E":
            " I cannot answer confidently.",
    }

    for label, completion in (
        completions.items()
    ):

        ids = tokenizer(
            completion,
            add_special_tokens=False,
        ).input_ids

        print(
            f"  {label}: {ids}"
        )

        if len(ids) == 0:
            raise RuntimeError(
                f"Empty tokenization for {label}"
            )

    # --------------------------------------------------------------
    # Tokenize datasets
    # --------------------------------------------------------------

    print(
        "\nTokenizing with "
        "completion-only labels..."
    )

    train_tokenized = [
        tokenize_example(
            row,
            tokenizer,
        )
        for row in train_rows
    ]

    val_tokenized = [
        tokenize_example(
            row,
            tokenizer,
        )
        for row in val_rows
    ]

    # Sanity check.
    sample = train_tokenized[0]

    n_total = len(
        sample["labels"]
    )

    n_supervised = sum(
        label != -100
        for label in sample["labels"]
    )

    n_masked = (
        n_total
        - n_supervised
    )

    print(
        "\nCompletion-only sanity check:"
    )

    print(
        f"  total tokens      : "
        f"{n_total}"
    )

    print(
        f"  supervised tokens : "
        f"{n_supervised}"
    )

    print(
        f"  masked tokens     : "
        f"{n_masked}"
    )

    if not (
        0
        < n_supervised
        < n_total
    ):
        raise RuntimeError(
            "Completion-only masking failed."
        )

    print(
        "Completion-only masking: PASS"
    )

    train_ds = Dataset.from_list(
        train_tokenized
    )

    val_ds = Dataset.from_list(
        val_tokenized
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print(
        "\nLoading Mistral-7B in 4-bit..."
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
        )
    )

    base.config.use_cache = False

    base = (
        prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },
        )
    )

    print(
        "\nLoading original MedQA "
        "SFT adapter as trainable..."
    )

    model = PeftModel.from_pretrained(
        base,
        SFT_ADAPTER,
        is_trainable=True,
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # --------------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------------

    if args.smoke:

        epochs = 1
        eval_steps = 5
        save_steps = 5
        logging_steps = 1

    else:

        epochs = MAX_EPOCHS
        eval_steps = 100
        save_steps = 100
        logging_steps = 25

    training_args = TrainingArguments(
        output_dir=str(
            checkpoint_dir
        ),

        num_train_epochs=epochs,

        per_device_train_batch_size=4,

        per_device_eval_batch_size=4,

        gradient_accumulation_steps=4,

        learning_rate=LR,

        lr_scheduler_type="cosine",

        warmup_ratio=0.05,

        eval_strategy="steps",

        eval_steps=eval_steps,

        save_strategy="steps",

        save_steps=save_steps,

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model=
            "eval_loss",

        greater_is_better=False,

        fp16=False,

        bf16=False,

        logging_steps=logging_steps,

        report_to="none",

        max_grad_norm=1.0,

        seed=SEED,
    )

    early_stopping = (
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[
            early_stopping
        ],
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print(
        "\nStarting supervised "
        "5-way training..."
    )

    result = trainer.train()

    print(
        "\nTraining finished."
    )

    # --------------------------------------------------------------
    # Evaluate restored best checkpoint
    # --------------------------------------------------------------

    print(
        "\nEvaluating restored "
        "best checkpoint..."
    )

    eval_metrics = (
        trainer.evaluate()
    )

    for key, value in (
        eval_metrics.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        str(final_dir)
    )

    tokenizer.save_pretrained(
        str(run_dir)
    )

    metadata = {
        "protocol":
            "clean_supervised_5way_train_v1",

        "source":
            str(DATA_FILE),

        "start_adapter":
            SFT_ADAPTER,

        "learning_rate":
            LR,

        "max_epochs":
            MAX_EPOCHS,

        "actual_epochs_configured":
            epochs,

        "seed":
            SEED,

        "smoke":
            args.smoke,

        "train_n":
            len(train_rows),

        "val_n":
            len(val_rows),

        "best_checkpoint":
            trainer.state.best_model_checkpoint,

        "best_metric":
            trainer.state.best_metric,

        "train_metrics":
            result.metrics,

        "eval_metrics":
            eval_metrics,
    }

    with open(
        run_dir
        / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "SUPERVISED 5-WAY TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Best checkpoint : "
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        f"Best eval loss  : "
        f"{trainer.state.best_metric}"
    )

    print(
        f"\nSaved adapter ->\n"
        f"{final_dir}"
    )


if __name__ == "__main__":
    main()
